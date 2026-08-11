"""The pre-vectorisation market phase, kept solely as a test oracle.

This is ``src/kaggriculture/sim/market.py`` as it stood at commit ``bc149a9``,
before the 65-round Python quantity scan was replaced by a tensor quantity
axis. It is a *validated* oracle: this implementation passed the full
differential campaign against the real ``kaggle_environments`` engine, so it
encodes the engine's per-unit lockstep by transcription -- each of the 65
rounds quotes both seats off the same book, commits them together, and only
then writes the book back.

The implementation that replaced it does not transcribe that loop. It computes
what the loop *would* produce, by quoting every candidate unit-round at once
and reading the filled prefix off cumulative costs, with a two-pass handover
for the rounds after the first seat stops trading. That is a proof, not a
transcription, and ``tests/sim/test_market_coupling.py`` is the thing that
checks the proof -- it fuzzes the two against each other over adversarial
states, exactly as ``tests/learn/test_mask.py`` replays the tensor legality
masks against ``kaggriculture.learn.mask``.

Do not "optimise", tidy, or vectorise anything below. Its slowness is the
entire point: it is structurally unlike the code it checks, so the two cannot
share a bug. Its only permitted edit is one that a reference-engine change
forces, and such an edit must be justified against the engine's source rather
than against the tensor implementation it is here to contradict.
"""
# ruff: noqa: D103

from dataclasses import fields

import torch

from kaggriculture.constants import ANIMALS, CROPS, LAND_PRICES, SHED_CAPACITY
from kaggriculture.learn.encoding import MAX_UNITS
from kaggriculture.sim.engine import MarketActions
from kaggriculture.sim.pricing import market_prices, refresh_prices
from kaggriculture.sim.state import (
    ANIMAL_NAMES,
    CROP_NAMES,
    PRODUCT_NAMES,
    SHED_NAMES,
    SimState,
)
from kaggriculture.sim.tensors import tensor_constant

_BUY_PRODUCTS = ("WHEAT", "FERTILIZER")


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
    unit_slots = torch.arange(MAX_UNITS, device=device)[None, None, :]
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
    y, x = torch.meshgrid(
        torch.arange(10, device=device), torch.arange(10, device=device), indexing="ij"
    )
    quadrants = torch.stack(
        ((x >= 5) & (y < 5), (x < 5) & (y >= 5), (x >= 5) & (y >= 5))
    )
    unlock = (
        accepted[..., None, None] & quadrants[stage.clamp_max(2)] & (state.kind == 1)
    )
    state.kind.masked_fill_(unlock, 0)


def _quote(
    state: SimState,
    order_type: torch.Tensor,
    item: torch.Tensor,
    active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    price = torch.zeros_like(state.money)
    valid = torch.zeros_like(active)
    for product, _name in enumerate(PRODUCT_NAMES):
        selected = active & (order_type == 1) & (item == product)
        quoted = market_prices(state.inventory[:, product], product)[:, None]
        price.copy_(torch.where(selected, quoted, price))
        valid |= selected
    for product_item, name in enumerate(_BUY_PRODUCTS):
        selected = active & (order_type == 3) & (item == product_item)
        product = PRODUCT_NAMES.index(name)
        quoted = market_prices(state.inventory[:, product] - 1, product)[:, None]
        price.copy_(torch.where(selected, quoted, price))
        valid |= selected
    for crop, name in enumerate(CROP_NAMES):
        selected = active & (order_type == 2) & (item == crop)
        price.copy_(torch.where(selected, int(CROPS[name]["seed"]), price))
        valid |= selected
    for animal, name in enumerate(ANIMAL_NAMES):
        selected = active & (order_type == 4) & (item == animal)
        price.copy_(torch.where(selected, int(ANIMALS[name]["cost"]), price))
        valid |= selected
    return price, valid


def _commit(
    state: SimState,
    order_type: torch.Tensor,
    item: torch.Tensor,
    price: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    success = torch.zeros_like(valid)
    market_delta = torch.zeros_like(state.inventory)
    room = state.shed.sum(dim=-1) < SHED_CAPACITY
    for product, name in enumerate(PRODUCT_NAMES):
        shed_item = SHED_NAMES.index(name)
        selected = valid & (order_type == 1) & (item == product)
        committed = selected & (state.shed[..., shed_item] > 0)
        state.shed[..., shed_item].sub_(committed.to(state.shed.dtype))
        state.money.add_(price * committed)
        market_delta[:, product].add_(
            (committed & (price > 1)).sum(dim=1).to(torch.int64)
        )
        success |= committed
    for product_item, name in enumerate(_BUY_PRODUCTS):
        product = PRODUCT_NAMES.index(name)
        shed_item = SHED_NAMES.index(name)
        selected = valid & (order_type == 3) & (item == product_item)
        committed = selected & (state.money >= price) & room
        state.money.sub_(price * committed)
        state.shed[..., shed_item].add_(committed.to(state.shed.dtype))
        market_delta[:, product].sub_(committed.sum(dim=1).to(torch.int64))
        success |= committed
    for crop, _name in enumerate(CROP_NAMES):
        selected = valid & (order_type == 2) & (item == crop)
        committed = selected & (state.money >= price)
        state.money.sub_(price * committed)
        state.seeds[..., crop].add_(committed.to(state.seeds.dtype))
        success |= committed
    for animal, name in enumerate(ANIMAL_NAMES):
        shed_item = SHED_NAMES.index(name)
        selected = valid & (order_type == 4) & (item == animal)
        committed = selected & (state.money >= price) & room
        state.money.sub_(price * committed)
        state.shed[..., shed_item].add_(committed.to(state.shed.dtype))
        success |= committed
    state.inventory.add_(market_delta)
    return success


def apply_market_phase(original: SimState, actions: MarketActions) -> SimState:
    """Apply ten order indices with a fixed 65-iteration quantity scan."""
    expected = (original.batch_size, 2, 10)
    if tuple(actions.order_type.shape) != expected:
        raise ValueError(
            f"expected market actions {expected}, got {tuple(actions.order_type.shape)}"
        )
    torch._assert(
        (actions.order_qty <= 64).all(), "market order quantity exceeds fixed scan"
    )
    state = _clone(original)
    for slot in range(10):
        order_type = actions.order_type[..., slot].to(torch.int64)
        item = actions.order_item[..., slot].to(torch.int64)
        remaining = actions.order_qty[..., slot].to(torch.int64).clone()
        has_quantity = (remaining > 0) | (order_type == 5) | (order_type == 6)
        active = (order_type != 0) & has_quantity & ~state.done[:, None]
        atomic = active & ((order_type == 5) | (order_type == 6))
        _hire(state, atomic & (order_type == 5))
        _buy_land(state, atomic & (order_type == 6))
        active &= ~atomic
        for _ in range(65):
            price, valid = _quote(state, order_type, item, active)
            committed = _commit(state, order_type, item, price, valid)
            remaining.sub_(committed.to(remaining.dtype))
            active = active & committed & (remaining > 0)
        state.prices.copy_(refresh_prices(state.inventory))
    return state
