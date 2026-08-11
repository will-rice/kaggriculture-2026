"""Pure-Torch serialized market phase.

Order slots stay sequential because an earlier order funds a later one, but the
quantity a slot fills is solved on a tensor axis: every candidate unit-round is
quoted at once and the successful prefix is read off cumulative costs.
"""

from dataclasses import fields

import torch

from kaggriculture.constants import LAND_PRICES, SHED_CAPACITY
from kaggriculture.learn.encoding import MAX_UNITS, QUANTITIES
from kaggriculture.sim.engine import MarketActions
from kaggriculture.sim.pricing import floor_levels, prices_for, refresh_prices
from kaggriculture.sim.state import SimState
from kaggriculture.sim.tables import (
    ORDER_COST,
    ORDER_ITEMS,
    ORDER_PRODUCT,
    ORDER_ROLE,
    ORDER_SEED,
    ORDER_SHED,
    ORDER_TYPES,
    ROLE_BUY_ANIMAL,
    ROLE_BUY_PRODUCT,
    ROLE_BUY_SEED,
    ROLE_NONE,
    ROLE_SELL,
    land_quadrants,
    market_orders,
)
from kaggriculture.sim.tensors import tensor_constant

# The decoder's largest quantity bucket, and therefore the width of the axis
# that resolves a slot. Requested quantities are clamped onto it so the axis is
# sufficient by construction rather than by a host-synchronizing assertion,
# which cannot run inside a captured CUDA graph.
QUANTITY_AXIS = max(QUANTITIES)


def _clone(state: SimState) -> SimState:
    return SimState(
        **{field.name: getattr(state, field.name).clone() for field in fields(state)}
    )


def _hire(state: SimState, requested: torch.Tensor) -> None:
    device = state.step.device
    fib = [1, 1]
    while len(fib) <= 40:
        fib.append(fib[-1] + fib[-2])
    costs = tensor_constant(fib, dtype=torch.int64, device=device)[
        state.hires_today.to(torch.int64).clamp_max(40)
    ]
    slot = state.hand_count.to(torch.int64) + 1
    accepted = requested & (state.money >= costs) & (slot < MAX_UNITS)
    coordinates = ((4, 4), (5, 4), (4, 5), (5, 5))
    access = tensor_constant(coordinates, dtype=torch.int8, device=device)
    occupancy = torch.stack(
        [
            (state.alive & (state.unit_x == x) & (state.unit_y == y)).sum(dim=-1)
            for x, y in coordinates
        ],
        dim=-1,
    )
    chosen = occupancy.argmin(dim=-1)
    state.money.sub_(costs * accepted)
    state.hires_today.add_(accepted.to(state.hires_today.dtype))
    state.hand_count.add_(accepted.to(state.hand_count.dtype))
    unit_slots = tensor_constant(
        tuple(range(MAX_UNITS)), dtype=torch.int64, device=device
    )[None, None, :]
    new_unit = accepted[..., None] & (unit_slots == slot[..., None])
    state.alive.logical_or_(new_unit)
    chosen_x = access[:, 0][chosen]
    chosen_y = access[:, 1][chosen]
    state.unit_x.copy_(torch.where(new_unit, chosen_x[..., None], state.unit_x))
    state.unit_y.copy_(torch.where(new_unit, chosen_y[..., None], state.unit_y))


def _buy_land(state: SimState, requested: torch.Tensor) -> None:
    device = state.step.device
    stage = state.quadrants.to(torch.int64) - 1
    prices = tensor_constant(LAND_PRICES, dtype=torch.int64, device=device)
    available = stage < len(LAND_PRICES)
    cost = prices[stage.clamp_max(len(LAND_PRICES) - 1)]
    accepted = requested & available & (state.money >= cost)
    state.money.sub_(cost * accepted)
    state.quadrants.add_(accepted.to(state.quadrants.dtype))
    quadrants = land_quadrants(device)
    unlock = (
        accepted[..., None, None] & quadrants[stage.clamp_max(2)] & (state.kind == 1)
    )
    state.kind.masked_fill_(unlock, 0)


def _ceil_div(value: torch.Tensor, divisor: torch.Tensor) -> torch.Tensor:
    """Divide non-negative values rounding away from zero."""
    return torch.div(value + divisor - 1, divisor, rounding_mode="floor")


def _advance(
    level: torch.Tensor,
    rounds: torch.Tensor,
    adds: torch.Tensor,
    takes: torch.Tensor,
    floor_level: torch.Tensor,
) -> torch.Tensor:
    """Return the market book after a run of identical unit-rounds.

    A sale adds a unit only while the quote is above the floor, so the book
    climbs until it reaches ``floor_level`` and then holds. A purchase always
    removes a unit. At most two seats trade one product per slot, so the
    combined step is one of: hold, climb, fall, or one of each.

    Args:
        level: Book level entering the run.
        rounds: Number of unit-rounds to advance, broadcast against ``level``.
        adds: Selling seats in this run, zero, one, or two.
        takes: Buying seats in this run, zero, one, or two.
        floor_level: Lowest level whose quote is already at the price floor.

    Returns:
        The book level after ``rounds`` rounds.
    """
    headroom = _ceil_div((floor_level - level).clamp_min(0), adds.clamp_min(1))
    climbing = level + adds * torch.minimum(rounds, headroom)
    falling = level - takes * rounds
    trading = torch.maximum(level - rounds, torch.minimum(level, floor_level - 1))
    return torch.where(
        takes == 0,
        torch.where(adds == 0, level, climbing),
        torch.where(adds == 0, falling, trading),
    )


def _resolve(
    levels: torch.Tensor,
    product: torch.Tensor,
    money: torch.Tensor,
    limit: torch.Tensor,
    buying: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Read the filled quantity and its effects off one book trajectory.

    Args:
        levels: ``(B, 2, Q)`` book level quoted at each candidate unit-round.
        product: ``(B, 2)`` product each seat trades.
        money: ``(B, 2)`` coin available before the slot.
        limit: ``(B, 2)`` cap from requested quantity, stock, and shed room.
        buying: ``(B, 2)`` mask of seats paying a market quote per unit.

    Returns:
        Filled quantity, sale revenue, purchase outlay, and units of supply the
        sale adds to the book.
    """
    quoted = prices_for(torch.cat((levels, levels - 1), dim=-1), product[..., None])
    price, cost = quoted.split(levels.shape[-1], dim=-1)
    outlay = torch.nn.functional.pad(cost.cumsum(dim=-1), (1, 0))
    budget = torch.searchsorted(outlay, money[..., None], right=True) - 1
    filled = torch.where(
        buying, torch.minimum(limit, budget.squeeze(-1).clamp_min(0)), limit
    )
    taken = filled[..., None]
    revenue = torch.nn.functional.pad(price.cumsum(dim=-1), (1, 0)).gather(-1, taken)
    supply = torch.nn.functional.pad(
        (price > 1).to(torch.int64).cumsum(dim=-1), (1, 0)
    ).gather(-1, taken)
    return (
        filled,
        revenue.squeeze(-1),
        outlay.gather(-1, taken).squeeze(-1),
        supply.squeeze(-1),
    )


def apply_market_phase(original: SimState, actions: MarketActions) -> SimState:
    """Apply ten ordered market slots over a tensor quantity axis.

    Slot index is what pairs the two seats into a lockstep round, so the orders
    are compacted first: the reference reads a hole-free list per seat, and a
    caller that leaves an empty slot between two orders would otherwise couple
    the wrong pair.
    """
    expected = (original.batch_size, 2, 10)
    if tuple(actions.order_type.shape) != expected:
        raise ValueError(
            f"expected market actions {expected}, got {tuple(actions.order_type.shape)}"
        )
    actions = actions.compacted()
    state = _clone(original)
    device = state.step.device
    rules = market_orders(device)
    floors = floor_levels(device)
    axis = tensor_constant(
        tuple(range(QUANTITY_AXIS)), dtype=torch.int64, device=device
    )
    for slot in range(10):
        order_type = actions.order_type[..., slot].to(torch.int64)
        item = actions.order_item[..., slot].to(torch.int64)
        requested = (
            actions.order_qty[..., slot].to(torch.int64).clamp_max(QUANTITY_AXIS)
        )
        has_quantity = (requested > 0) | (order_type == 5) | (order_type == 6)
        active = (order_type != 0) & has_quantity & ~state.done[:, None]
        atomic = active & ((order_type == 5) | (order_type == 6))
        _hire(state, atomic & (order_type == 5))
        _buy_land(state, atomic & (order_type == 6))

        code = item + 1
        known = active & ~atomic & (code >= 0) & (code < ORDER_ITEMS)
        row = rules[
            order_type.clamp(0, ORDER_TYPES - 1), code.clamp(0, ORDER_ITEMS - 1)
        ]
        detail = torch.where(known[..., None], row, torch.zeros_like(row))
        role = detail[..., ORDER_ROLE]
        product = detail[..., ORDER_PRODUCT]
        cost = detail[..., ORDER_COST]
        selling = role == ROLE_SELL
        buying = role == ROLE_BUY_PRODUCT
        seeding = role == ROLE_BUY_SEED
        stocking = role == ROLE_BUY_ANIMAL

        room = (SHED_CAPACITY - state.shed.sum(dim=-1)).clamp_min(0).to(torch.int64)
        stock = (
            state.shed.gather(2, detail[..., ORDER_SHED, None])
            .squeeze(-1)
            .to(torch.int64)
        )
        affordable = torch.div(
            state.money, cost.clamp_min(1), rounding_mode="floor"
        ).clamp_min(0)
        limit = torch.where(selling, torch.minimum(requested, stock), requested)
        limit = torch.where(seeding, torch.minimum(requested, affordable), limit)
        limit = torch.where(
            stocking, torch.minimum(torch.minimum(requested, affordable), room), limit
        )
        limit = torch.where(buying, torch.minimum(requested, room), limit)
        limit = torch.where(role == ROLE_NONE, torch.zeros_like(limit), limit)

        trading = selling | buying
        matched = trading & (product.flip(1) == product)
        adds = (selling.to(torch.int64) + (matched & selling.flip(1)))[..., None]
        takes = (buying.to(torch.int64) + (matched & buying.flip(1)))[..., None]
        alone_adds = selling.to(torch.int64)[..., None]
        alone_takes = buying.to(torch.int64)[..., None]
        book = state.inventory.gather(1, product)[..., None]
        ceiling = floors[product][..., None]

        lockstep = _advance(book, axis, adds, takes, ceiling)
        preview, *_ = _resolve(lockstep, product, state.money, limit, buying)
        cutoff = preview.amin(dim=1, keepdim=True)[..., None]
        handover = _advance(book, cutoff, adds, takes, ceiling)
        levels = torch.where(
            axis < cutoff,
            lockstep,
            _advance(
                handover,
                (axis - cutoff).clamp_min(0),
                alone_adds,
                alone_takes,
                ceiling,
            ),
        )
        filled, revenue, outlay, supply = _resolve(
            levels, product, state.money, limit, buying
        )

        zero = torch.zeros_like(filled)
        state.money.add_(
            torch.where(selling, revenue, zero)
            - torch.where(buying, outlay, zero)
            - torch.where(seeding | stocking, filled * cost, zero)
        )
        state.shed.scatter_add_(
            2,
            detail[..., ORDER_SHED, None],
            torch.where(selling, -filled, torch.where(buying | stocking, filled, zero))[
                ..., None
            ].to(state.shed.dtype),
        )
        state.seeds.scatter_add_(
            2,
            detail[..., ORDER_SEED, None],
            torch.where(seeding, filled, zero)[..., None].to(state.seeds.dtype),
        )
        state.inventory.scatter_add_(
            1,
            product,
            torch.where(selling, supply, zero) - torch.where(buying, filled, zero),
        )
        state.prices.copy_(refresh_prices(state.inventory))
    return state
