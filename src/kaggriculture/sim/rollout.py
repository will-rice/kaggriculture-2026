"""On-device rollout utilities for the batched simulator."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from kaggriculture.constants import (
    ANIMALS,
    CROPS,
    LAND_PRICES,
    MARKET_PARAMS,
    SHED_CAPACITY,
)
from kaggriculture.learn.encoding import IGNORE, MAX_UNITS, UNIT_OPS
from kaggriculture.sim.decode import decode_market_buckets
from kaggriculture.sim.engine import (
    MAX_MARKET_ORDERS_PER_TURN,
    MarketActions,
    step,
    unit_quantity_ones,
)
from kaggriculture.sim.legality import legal
from kaggriculture.sim.market import QUANTITY_AXIS
from kaggriculture.sim.observe import observe
from kaggriculture.sim.state import (
    ANIMAL_NAMES,
    CROP_NAMES,
    PRODUCT_NAMES,
    SHED_NAMES,
    SimState,
    unpack,
)
from kaggriculture.sim.tensors import tensor_constant

GROWING = 0.4
ScriptedOpponent = Callable[[Mapping[str, Any]], Mapping[str, Any]]

_MARKET_TYPES = {
    "SELL": (1, PRODUCT_NAMES),
    "BUY_SEED": (2, CROP_NAMES),
    "BUY_PRODUCT": (3, ("WHEAT", "FERTILIZER")),
    "BUY_ANIMAL": (4, ANIMAL_NAMES),
}

# `potential` runs twice per collected turn, so its six price tables are frozen
# here and handed to `tensor_constant` rather than rebuilt with `torch.tensor`
# on every call. Rebuilding them copied host memory to the device inside the
# rollout loop, which is both per-turn work the season does not need and an
# unpinned host-to-device copy -- the kind CUDA graph capture refuses outright.
# The leading zero in the indexed tables is the "nothing here" code, so the
# tables are indexed by `crop`/`occupant` directly.
_SEED_COST = tuple(float(CROPS[name]["seed"]) for name in CROP_NAMES)
_CUMULATIVE_LAND = (
    0.0,
    *torch.tensor(LAND_PRICES, dtype=torch.float32).cumsum(0).tolist(),
)
_CROP_SEED = (0.0, *_SEED_COST)
_CROP_VALUE = (0.0, *(float(MARKET_PARAMS[name]["base"]) for name in CROP_NAMES))
_ANIMAL_COST = (0.0, *(float(ANIMALS[name]["cost"]) for name in ANIMAL_NAMES))
_ANIMAL_PRODUCT_VALUE = (
    0.0,
    *(
        float(MARKET_PARAMS[str(ANIMALS[name]["product"])]["base"])
        for name in ANIMAL_NAMES
    ),
)
_ITEM_VALUE = tuple(
    float(MARKET_PARAMS[name]["base"])
    if name in MARKET_PARAMS
    else float(ANIMALS[name]["cost"])
    for name in SHED_NAMES
)


@dataclass(frozen=True)
class Trajectory:
    """One fixed-length on-device segment, time-major and batch-preserving.

    Every field is a device tensor, ``illegal`` included. It is a scalar
    ``int64`` count and not a Python ``int`` because reading one back on the
    host is a synchronisation, and a single synchronisation anywhere inside
    ``collect_segment`` aborts a CUDA graph capture of it -- which is where the
    collection loop's speedup lives. Callers that genuinely need the number,
    which in practice means logging it or failing a run on it, call ``int()`` on
    it themselves, outside any captured region. What it counts is unchanged:
    sampled actions their own stored mask forbade, summed over the segment's
    turns, its alive units and its market slots, for the learning seat alone
    when a scripted opponent occupies seat 1 and for both seats otherwise.
    """

    board: torch.Tensor
    scalars: torch.Tensor
    positions: torch.Tensor
    unit_actions: torch.Tensor
    market_actions: torch.Tensor
    unit_masks: torch.Tensor
    market_masks: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    own: torch.Tensor
    potentials: torch.Tensor
    dones: torch.Tensor
    illegal: torch.Tensor


def potential(state: SimState, seat: int) -> torch.Tensor:
    """Return the six potential components directly from tensor state."""
    if seat not in (0, 1):
        raise ValueError(f"seat must be 0 or 1, got {seat}")
    device = state.step.device
    scalar = {"dtype": torch.float32, "device": device}
    seed_cost = tensor_constant(_SEED_COST, **scalar)
    seeds = (state.seeds[:, seat].to(torch.float32) * seed_cost).sum(dim=-1)
    cumulative_land = tensor_constant(_CUMULATIVE_LAND, **scalar)
    land = cumulative_land[(state.quadrants[:, seat].to(torch.int64) - 1).clamp(0, 3)]

    crop = state.crop[:, seat].to(torch.int64)
    crop_seed = tensor_constant(_CROP_SEED, **scalar)[crop]
    crop_value = tensor_constant(_CROP_VALUE, **scalar)[crop]
    plant = state.kind[:, seat] == 3
    growing = torch.where(
        plant,
        crop_seed + GROWING * state.yield_units[:, seat] * crop_value,
        0.0,
    ).sum(dim=(1, 2))

    occupant = state.occupant[:, seat].to(torch.int64)
    animal_cost = tensor_constant(_ANIMAL_COST, **scalar)[occupant]
    animal_product_value = tensor_constant(_ANIMAL_PRODUCT_VALUE, **scalar)[occupant]
    occupied = state.occupant[:, seat] > 0
    livestock = torch.where(
        occupied,
        animal_cost + GROWING * state.yield_units[:, seat] * animal_product_value,
        0.0,
    ).sum(dim=(1, 2))

    values = tensor_constant(_ITEM_VALUE, **scalar)
    carried = (state.inv_count[:, seat].to(torch.float32) * values[None, None, :]).sum(
        dim=(1, 2)
    )
    stored = (state.shed[:, seat].to(torch.float32) * values).sum(dim=-1)
    return torch.stack((seeds, land, growing, livestock, carried, stored), dim=-1)


def _sample(
    logits: torch.Tensor, mask: torch.Tensor, generator: torch.Generator | None
) -> tuple[torch.Tensor, torch.Tensor]:
    log_prob = torch.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)
    flat = log_prob.flatten(0, -2)
    chosen = torch.multinomial(flat.exp(), 1, generator=generator)
    return (
        chosen.reshape(logits.shape[:-1]),
        flat.gather(1, chosen).reshape(logits.shape[:-1]),
    )


def _transfer_quantity(quantity: object) -> int | None:
    """Return the int quantity a PICKUP/PLACE names, or ``None`` if out of domain.

    Mirrors the tolerance ``_order_quantity`` applies to market orders: only
    ``int | float | str`` coerces, exactly what the reference's own bare
    ``int(action[2])`` accepts. Unlike ``_order_quantity`` there is no cap and
    no floor here -- the value is encoded exactly as given, sign and all. The
    engine clamps at execution (a non-positive amount is already a no-op
    there), so encoding faithfully and letting it do so is the whole job.
    """
    if not isinstance(quantity, (int, float, str)):
        return None
    try:
        return int(quantity)
    except (TypeError, ValueError):
        return None


def _unit_index(action: object) -> tuple[int, int]:
    """Return one unit action's ``(op index, quantity)``.

    Quantity is 1 everywhere except a PICKUP/PLACE that names one explicitly,
    where it is that quantity, encoded faithfully and unclamped -- the
    simulator applies the engine's own clamp at execution. A quantity the
    reference's bare ``int(action[2])`` could not itself parse (a list, a
    non-numeric string) folds the whole unit action to PASS, exactly the way
    an unparseable market order becomes the "aborted" code 7: the alternative
    is claiming to execute an action the reference engine would instead raise
    on.
    """
    if not isinstance(action, (list, tuple)) or not action:
        return UNIT_OPS.index("PASS"), 1
    verb = str(action[0])
    quantity = 1
    if verb in {"PICKUP", "PLACE"} and len(action) > 2:
        parsed = _transfer_quantity(action[2])
        if parsed is None:
            return UNIT_OPS.index("PASS"), 1
        quantity = parsed
    name = (
        f"{verb}:{action[1]}"
        if verb in {"PLANT", "PICKUP", "PLACE"} and len(action) > 1
        else verb
    )
    if name not in UNIT_OPS:
        return UNIT_OPS.index("PASS"), 1
    return UNIT_OPS.index(name), quantity


@dataclass(frozen=True)
class TurnActions:
    """One turn's actions as simulator codes, independent of batch and seat.

    Attributes:
        units: One op index per unit slot, ``PASS`` where the turn named none.
        quantities: One transfer quantity per unit slot, aligned with
            ``units``. 1 wherever the slot's op is not a ``PICKUP:*`` or
            ``PLACE:*``; otherwise the quantity the turn named, faithfully and
            unclamped.
        orders: ``(order_type, order_item, order_qty)`` per market slot, in the
            queue position the reference reads them in. Type 0 is "no order",
            and ``order_item`` is -1 wherever the verb carries no item.
    """

    units: tuple[int, ...]
    quantities: tuple[int, ...]
    orders: tuple[tuple[int, int, int], ...]


def encode_turn(action: Mapping[str, Any]) -> TurnActions:
    """Return one action dict as simulator codes.

    Args:
        action: A turn in the engine's action grammar.

    Returns:
        The encoded turn. Malformed orders become the "aborted" code 7, and a
        PICKUP/PLACE naming a quantity the reference itself could not parse
        becomes PASS -- both are what the reference does with an action it
        cannot execute.

    Raises:
        ValueError: If a market order names a quantity outside the
            simulator's supported domain that the reference engine would
            still execute. See ``_encode_order``.
    """
    units = [UNIT_OPS.index("PASS")] * MAX_UNITS
    quantities = [1] * MAX_UNITS
    units[0], quantities[0] = _unit_index(action.get("farmer", ["PASS"]))
    hands = action.get("hands") or []
    if isinstance(hands, list):
        for unit, unit_action in enumerate(hands[: MAX_UNITS - 1], start=1):
            units[unit], quantities[unit] = _unit_index(unit_action)

    orders: list[tuple[int, int, int]] = [(0, -1, 0)] * MAX_MARKET_ORDERS_PER_TURN
    given = action.get("market") or []
    if isinstance(given, list):
        for slot, order in enumerate(given[:MAX_MARKET_ORDERS_PER_TURN]):
            orders[slot] = _encode_order(order)
    return TurnActions(
        units=tuple(units), quantities=tuple(quantities), orders=tuple(orders)
    )


# Verbs whose fill the reference's own ``_commit_unit`` bounds by
# ``SHED_CAPACITY`` regardless of the requested quantity: SELL cannot sell
# more than the shed holds, and BUY_PRODUCT/BUY_ANIMAL both refuse outright
# once ``sum(shed.values()) >= shed_capacity``. Every one of those bounds is
# therefore <= SHED_CAPACITY, which is < QUANTITY_AXIS today, so clamping the
# request onto the axis before that bound is applied changes nothing the
# engine would have done: min(request, engine_bound) ==
# min(min(request, QUANTITY_AXIS), engine_bound) whenever engine_bound <=
# SHED_CAPACITY < QUANTITY_AXIS, for any request. BUY_SEED is excluded on
# purpose -- seeds land in ``private["seeds"]``, never the shed, so a
# BUY_SEED fill has no such structural bound and a large request is not
# provably safe to clamp.
SHED_BOUND_VERBS = frozenset({"SELL", "BUY_PRODUCT", "BUY_ANIMAL"})

if SHED_CAPACITY >= QUANTITY_AXIS:
    raise AssertionError(
        f"SHED_CAPACITY ({SHED_CAPACITY}) >= QUANTITY_AXIS ({QUANTITY_AXIS}): "
        "_order_quantity's clamp for SELL/BUY_PRODUCT/BUY_ANIMAL is no longer "
        "provably identical to the reference engine's own shed-capacity limit, "
        "so it must go back to raising for those verbs too until this is "
        "re-proven"
    )


def _order_quantity(verb: str, item: str, quantity: object) -> int:
    """Return the in-domain quantity for one market order, or ``-1`` to abort.

    Args:
        verb: The order's verb. Determines whether an over-axis quantity
            clamps or raises -- see ``Raises``.
        item: The order's item, for the raised message only.
        quantity: The order's raw, unvalidated quantity.

    Returns:
        The quantity to encode, or ``-1`` if the reference's own
        ``int(order[2])`` would raise and abort the order -- the same
        exclusion, reached without making the call. A route is parsed from
        JSON, so anything not ``int | float | str`` here (``None``, a list, a
        dict) is exactly such a case. For ``verb in SHED_BOUND_VERBS``, a
        quantity over ``QUANTITY_AXIS`` clamps to it rather than raising --
        see the comment above ``SHED_BOUND_VERBS`` for why that is exact
        rather than approximate. Top play uses a "sell everything" idiom
        (``SELL X 999`` and the like) exactly to hit this case; refusing to
        encode it at all was refusing to replay a large slice of ordinary
        top play, not a genuine domain violation.

    Raises:
        ValueError: If the quantity is outside the simulator's supported
            domain but the reference engine's ``_parse_order`` would still
            accept it, and ``verb`` is not shed-bound -- ``BUY_SEED`` above
            ``QUANTITY_AXIS`` has no structural bound to appeal to (seeds
            never touch the shed), so a request that large is not provably
            safe to clamp and this still raises for it. Also raised for a
            non-int the reference would coerce via ``int()`` and execute (a
            numeric string, say); clamping or aborting that would diverge
            from what the reference actually replays. The RL action space
            only ever emits small int quantities, so none of this ever fires
            for it.
    """
    if not isinstance(quantity, (int, float, str)):
        return -1
    try:
        coerced = int(quantity)
    except (TypeError, ValueError):
        return -1
    if isinstance(quantity, int):
        if coerced > QUANTITY_AXIS:
            if verb in SHED_BOUND_VERBS:
                return QUANTITY_AXIS
            raise ValueError(
                f"{verb} {item!r} quantity {quantity!r} is outside the "
                "simulator's supported domain: quantity must not exceed "
                f"{QUANTITY_AXIS}"
            )
        return max(0, coerced)
    if coerced <= 0:
        return -1
    raise ValueError(
        f"{verb} {item!r} quantity {quantity!r} is outside the simulator's "
        f"supported domain: quantity must be an int, not {type(quantity).__name__}"
    )


def _encode_order(order: object) -> tuple[int, int, int]:
    """Return one market order's ``(type, item, quantity)`` codes.

    Raises:
        ValueError: If the quantity is outside the simulator's supported
            domain but the reference engine would still execute it. See
            ``_order_quantity``.
    """
    if not isinstance(order, (list, tuple)) or not order:
        return (7, -1, 0)
    verb = str(order[0])
    if verb == "HIRE":
        return (5, -1, 1)
    if verb == "BUY_LAND":
        return (6, -1, 1)
    if verb in _MARKET_TYPES and len(order) >= 3:
        kind, catalogue = _MARKET_TYPES[verb]
        item = str(order[1])
        if item not in catalogue:
            return (7, -1, 0)
        quantity = _order_quantity(verb, item, order[2])
        if quantity < 0:
            return (7, -1, 0)
        return (kind, catalogue.index(item), quantity)
    return (7, -1, 0)


def scripted_actions(
    state: SimState, opponent: ScriptedOpponent, seat: int = 1
) -> tuple[torch.Tensor, torch.Tensor, MarketActions]:
    """Bridge a dict-based scripted opponent into fixed simulator tensors."""
    if seat not in (0, 1):
        raise ValueError(f"seat must be 0 or 1, got {seat}")
    device = state.step.device
    units = torch.full(
        (state.batch_size, MAX_UNITS),
        UNIT_OPS.index("PASS"),
        dtype=torch.int16,
        device=device,
    )
    quantities = torch.ones(
        (state.batch_size, MAX_UNITS), dtype=torch.int16, device=device
    )
    markets = MarketActions.empty(state.batch_size, device=device)
    for batch in range(state.batch_size):
        encoded = encode_turn(opponent(unpack(state, batch, seat)))
        for unit, op in enumerate(encoded.units):
            units[batch, unit] = op
        for unit, quantity in enumerate(encoded.quantities):
            quantities[batch, unit] = quantity
        for slot, (kind, item, quantity) in enumerate(encoded.orders):
            markets.order_type[batch, seat, slot] = kind
            markets.order_item[batch, seat, slot] = item
            markets.order_qty[batch, seat, slot] = quantity
    return units, quantities, markets


def collect_segment(
    state: SimState,
    policy: torch.nn.Module,
    *,
    turns: int = 32,
    generator: torch.Generator | None = None,
    opponent: ScriptedOpponent | None = None,
) -> tuple[SimState, Trajectory]:
    """Collect a fixed-length segment, optionally bridging a scripted seat 1.

    With ``opponent=None`` the whole body is device work and holds no
    synchronisation, so it can be captured as a CUDA graph. That matters
    because the loop is launch-bound -- a segment costs about the same at batch
    128 as at batch 4096 -- so replaying one capture is worth several times the
    eager loop. Capture has two requirements this function cannot enforce for
    the caller: the successor state must be copied back over ``state``'s own
    buffers inside the captured region, otherwise every replay recomputes the
    same turn from unchanged inputs; and ``generator`` must either be ``None``,
    for the default CUDA generator that ``torch.cuda.graph`` registers itself,
    or a CUDA generator the caller registered with the graph. A CPU generator
    never reaches these tensors.

    A scripted ``opponent`` is the exception, and it is not one that can be
    removed: ``scripted_actions`` reads each row's state on the host and asks a
    Python function what to play. That is a synchronisation per row per turn,
    so a scripted segment cannot be captured and is not meant to be.

    A scripted opponent's bulk transfers ride the same quantity lane the
    learning seat does. Before this, a unit action had no such lane and an
    opponent that emitted a bulk ``PICKUP`` or ``PLACE`` (a quantity other
    than 1) was outside the simulator's domain, so ``encode_turn`` raised
    rather than silently transferring one item -- which is why the bridge
    tests that drove ``economic_policy.agent`` through this function once used
    only a turn or two, since it issues exactly such a bulk transfer as soon
    as its shed and worker inventories are non-empty. The domain grew to carry
    quantities in full (2026-08-21); those tests now run the opponent past the
    turn that used to raise.

    Args:
        state: The batch to advance. Left untouched; the successor is returned.
        policy: The network, called once per turn with both seats batched.
        turns: Turns to collect. Every turn is recorded.
        generator: The sampling stream, or ``None`` for the device default.
        opponent: A scripted seat 1, or ``None`` for self-play on the policy.

    Returns:
        The state after ``turns`` turns, and the segment's ``Trajectory``. Every
        field of it is a device tensor, ``illegal`` included.
    """
    records: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "board",
            "scalars",
            "positions",
            "unit_actions",
            "market_actions",
            "unit_masks",
            "market_masks",
            "log_probs",
            "values",
            "rewards",
            "own",
            "potentials",
            "dones",
        )
    }
    illegal_count = torch.zeros((), dtype=torch.int64, device=state.step.device)
    for _ in range(turns):
        observed = [observe(state, seat) for seat in range(2)]
        boards = torch.stack([value[0] for value in observed], dim=1)
        scalars = torch.stack([value[1] for value in observed], dim=1)
        positions = torch.stack([value[2] for value in observed], dim=1)
        masks = [legal(state, seat) for seat in range(2)]
        unit_masks = torch.stack([value[0] for value in masks], dim=1)
        market_masks = torch.stack([value[1] for value in masks], dim=1)
        batch = state.batch_size
        with torch.no_grad():
            unit_logits, market_logits, values = policy(
                boards.flatten(0, 1),
                scalars.flatten(0, 1),
                positions.flatten(0, 1),
            )
        unit_logits = unit_logits.reshape(batch, 2, *unit_logits.shape[1:])
        market_logits = market_logits.reshape(batch, 2, *market_logits.shape[1:])
        values = values.reshape(batch, 2, -1).squeeze(-1)
        chosen_units, unit_log = _sample(unit_logits, unit_masks, generator)
        chosen_market, market_log = _sample(market_logits, market_masks, generator)
        market_orders = decode_market_buckets(chosen_market)
        unit_quantities = unit_quantity_ones(state.batch_size, state.step.device)
        if opponent is not None:
            opponent_units, opponent_quantities, opponent_market = scripted_actions(
                state, opponent
            )
            chosen_units[:, 1].copy_(opponent_units)
            unit_quantities[:, 1].copy_(opponent_quantities)
            market_orders.order_type[:, 1].copy_(opponent_market.order_type[:, 1])
            market_orders.order_item[:, 1].copy_(opponent_market.order_item[:, 1])
            market_orders.order_qty[:, 1].copy_(opponent_market.order_qty[:, 1])
        alive = state.alive
        stored_units = torch.where(alive, chosen_units, IGNORE)
        joint_log = (unit_log * alive).sum(dim=-1) + market_log.sum(dim=-1)
        before_money = state.money.clone()
        before_margin = before_money - before_money.flip(1)
        before_potential = torch.stack(
            (potential(state, 0), potential(state, 1)), dim=1
        )
        next_state = step(
            state,
            chosen_units.to(torch.int16),
            market_orders,
            unit_quantities,
        )
        after_money = next_state.money
        after_margin = after_money - after_money.flip(1)
        records["board"].append(boards)
        records["scalars"].append(scalars)
        records["positions"].append(positions)
        records["unit_actions"].append(stored_units)
        records["market_actions"].append(chosen_market)
        records["unit_masks"].append(unit_masks)
        records["market_masks"].append(market_masks)
        records["log_probs"].append(joint_log)
        records["values"].append(values)
        records["rewards"].append((after_margin - before_margin).to(torch.float32))
        records["own"].append((after_money - before_money).to(torch.float32))
        records["potentials"].append(before_potential)
        records["dones"].append(next_state.done[:, None].expand(-1, 2))
        checked_seats = slice(0, 1) if opponent is not None else slice(None)
        illegal_count.add_(
            (
                ~unit_masks[:, checked_seats]
                .gather(-1, chosen_units[:, checked_seats, ..., None])
                .squeeze(-1)
                & alive[:, checked_seats]
            ).sum()
            + (
                ~market_masks[:, checked_seats]
                .gather(-1, chosen_market[:, checked_seats, ..., None])
                .squeeze(-1)
            ).sum()
        )
        state = next_state
    trajectory = Trajectory(
        **{name: torch.stack(values) for name, values in records.items()},
        illegal=illegal_count,
    )
    return state, trajectory
