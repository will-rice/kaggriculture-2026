"""On-device rollout utilities for the batched simulator."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from kaggriculture.action_codec import QUANTITIES, TRANSFER_OPS, UNIT_OPS
from kaggriculture.constants import (
    ANIMALS,
    CROPS,
    LAND_PRICES,
    MARKET_PARAMS,
    SHED_CAPACITY,
)
from kaggriculture.learn.encoding import (
    CARRIED_SCALE,
    IGNORE,
    MAX_UNITS,
    SEED_SCALE,
)
from kaggriculture.learn.rollout import segment_starts
from kaggriculture.learn.toad.model import (
    PolicyOutput,
    PolicyState,
    StatefulPolicy,
    uses_stateful_policy,
)
from kaggriculture.learn.toad_reward import (
    ABSOLUTE_WEIGHT,
    CAPITAL_WEIGHT,
    CITY_WEIGHT,
    FUEL_WEIGHT,
    GAME_RESULT_WEIGHT,
    MARGIN_WEIGHT,
    MONEY_WEIGHT,
    NORMALISER,
    RESEARCH_WEIGHT,
    STEP_WEIGHT,
    UNIT_WEIGHT,
)
from kaggriculture.sim.decode import decode_market_buckets
from kaggriculture.sim.engine import (
    MAX_MARKET_ORDERS_PER_TURN,
    MarketActions,
    step,
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
SamplingGenerator = torch.Generator | Sequence[torch.Generator]

_MARKET_TYPES = {
    "SELL": (1, PRODUCT_NAMES),
    "BUY_SEED": (2, CROP_NAMES),
    "BUY_PRODUCT": (3, ("WHEAT", "FERTILIZER")),
    "BUY_ANIMAL": (4, ANIMAL_NAMES),
}


@dataclass(frozen=True)
class RolloutPolicyState:
    """Actor memories carried between segments with a distinct neural opponent.

    Self-play owns one state over both flattened seats and scripted collection
    owns only the learner state, so those paths keep returning ``PolicyState``
    exactly as before.  Frozen and teacher collection run two policy objects;
    this container prevents the opponent memory from being silently restarted
    at every source segment while retaining the same three-value return.
    """

    learner: PolicyState | None
    opponent: PolicyState | None = None


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
    turns, its alive units, its quantity buckets and its market slots, for the
    learning seat alone when a scripted opponent occupies seat 1 and for both
    seats otherwise.

    ``unit_quantities`` holds bucket indices into ``QUANTITIES``, not the
    counts themselves: it is an action the learner re-scores under the head
    that produced it, the same way ``market_actions`` is. The count each one
    stands for is what reached ``step``. Every slot carries a bucket, dead
    ones included, and a slot spends it only where the op beside it is a
    ``PICKUP`` or a ``PLACE``.
    """

    board: torch.Tensor
    scalars: torch.Tensor
    positions: torch.Tensor
    unit_actions: torch.Tensor
    unit_quantities: torch.Tensor
    market_actions: torch.Tensor
    unit_masks: torch.Tensor
    unit_quantity_masks: torch.Tensor
    market_masks: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    own: torch.Tensor
    shaped: torch.Tensor
    shaped_money: torch.Tensor
    margin: torch.Tensor
    sparse: torch.Tensor
    potentials: torch.Tensor
    dones: torch.Tensor
    illegal: torch.Tensor
    hidden: torch.Tensor | None = None
    cell: torch.Tensor | None = None
    prior_belief: torch.Tensor | None = None
    state_steps: torch.Tensor | None = None
    belief_targets: torch.Tensor | None = None
    belief_valid: torch.Tensor | None = None
    final_margin: torch.Tensor | None = None
    final_bank: torch.Tensor | None = None
    final_capital: torch.Tensor | None = None
    illegal_by_stream: torch.Tensor | None = None
    sales: torch.Tensor | None = None
    units_sold: torch.Tensor | None = None
    mean_sale_price: torch.Tensor | None = None
    realisation: torch.Tensor | None = None
    bought: torch.Tensor | None = None
    sale_proceeds: torch.Tensor | None = None
    sale_market_value: torch.Tensor | None = None


def potential(state: SimState, seat: int) -> torch.Tensor:
    """Return the seat's net worth by component, directly from tensor state.

    Column for column ``learn.progress.potential``, in the order
    ``POTENTIAL_COMPONENTS`` names -- ``money`` last, and read straight off
    ``state.money`` rather than priced, because a coin is worth a coin.
    """
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
    money = state.money[:, seat].to(torch.float32)
    return torch.stack(
        (seeds, land, growing, livestock, carried, stored, money), dim=-1
    )


def _sample(
    logits: torch.Tensor,
    mask: torch.Tensor,
    generator: SamplingGenerator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_prob = torch.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)
    flat = log_prob.flatten(0, -2)
    if generator is None or isinstance(generator, torch.Generator):
        chosen = torch.multinomial(flat.exp(), 1, generator=generator)
    else:
        if len(generator) != logits.shape[0]:
            raise ValueError("one sampling generator is required per environment")
        sampled = []
        start = 0
        while start < len(generator):
            row_generator = generator[start]
            end = start + 1
            while end < len(generator) and generator[end] is row_generator:
                end += 1
            sampled.append(
                torch.multinomial(
                    log_prob[start:end].flatten(0, -2).exp(),
                    1,
                    generator=row_generator,
                )
            )
            start = end
        chosen = torch.cat(sampled)
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
        try:
            encoded = encode_turn(opponent(unpack(state, batch, seat)))
        except Exception as error:
            from kaggriculture.learn.toad.data import CollectionRowError

            raise CollectionRowError(
                row=batch,
                error_type=type(error).__name__,
                message=str(error),
            ) from error
        for unit, op in enumerate(encoded.units):
            units[batch, unit] = op
        for unit, quantity in enumerate(encoded.quantities):
            quantities[batch, unit] = quantity
        for slot, (kind, item, quantity) in enumerate(encoded.orders):
            markets.order_type[batch, seat, slot] = kind
            markets.order_item[batch, seat, slot] = item
            markets.order_qty[batch, seat, slot] = quantity
    return units, quantities, markets


def _policy_forward(
    policy: torch.nn.Module,
    board: torch.Tensor,
    scalars: torch.Tensor,
    positions: torch.Tensor,
    *,
    state: PolicyState | None,
    dones: torch.Tensor,
) -> PolicyOutput:
    """Normalize one actor step onto the accepted stateful output contract."""
    if isinstance(policy, StatefulPolicy) and uses_stateful_policy(policy.config):
        output = policy(
            board.unsqueeze(0),
            scalars.unsqueeze(0),
            positions.unsqueeze(0),
            state=state,
            dones=dones.unsqueeze(0),
        )
        return PolicyOutput(
            output.unit_logits.squeeze(0),
            output.quantity_logits.squeeze(0),
            output.market_logits.squeeze(0),
            output.values.squeeze(0),
            (
                output.belief_logits.squeeze(0)
                if output.belief_logits is not None
                else None
            ),
            output.state,
            output.input_state,
        )
    if state is not None:
        raise ValueError("a stateless policy cannot consume PolicyState")
    output = policy(board, scalars, positions)
    if isinstance(output, PolicyOutput):
        return output
    unit, quantity, market, values = output
    return PolicyOutput(unit, quantity, market, values, None, None)


def _reshape_state_rows(state: PolicyState, batch: int, seats: int) -> PolicyState:
    """Restore environment/seat axes on one flattened actor state."""
    if state.hidden.ndim == 5:
        hidden = state.hidden.reshape(
            state.hidden.shape[0], batch, seats, *state.hidden.shape[2:]
        )
        cell = state.cell.reshape(
            state.cell.shape[0], batch, seats, *state.cell.shape[2:]
        )
    else:
        hidden = state.hidden.reshape(batch, seats, *state.hidden.shape[1:])
        cell = state.cell.reshape(batch, seats, *state.cell.shape[1:])
    return PolicyState(
        hidden=hidden,
        cell=cell,
        prior_belief=state.prior_belief.reshape(batch, seats, -1),
    )


def copy_policy_state_(target: PolicyState, source: PolicyState) -> PolicyState:
    """Copy successor actor state into stable buffers for CUDA graph replay."""
    target.hidden.copy_(source.hidden)
    target.cell.copy_(source.cell)
    target.prior_belief.copy_(source.prior_belief)
    return target


_PRODUCT_SHED_INDICES = tuple(SHED_NAMES.index(name) for name in PRODUCT_NAMES)
_ANIMAL_SHED_INDICES = tuple(SHED_NAMES.index(name) for name in ANIMAL_NAMES)
_FUEL_SHED_INDICES = tuple(
    index for index, name in enumerate(SHED_NAMES) if name not in ANIMAL_NAMES
)


@dataclass(frozen=True)
class _TensorCounts:
    """The Toad reward counts for both seats of every tensor environment."""

    city: torch.Tensor
    unit: torch.Tensor
    research: torch.Tensor
    fuel: torch.Tensor
    capital: torch.Tensor
    money: torch.Tensor
    opponent: torch.Tensor


def _tensor_counts(state: SimState) -> _TensorCounts:
    """Read reward counts without decoding a simulator row on the host."""
    device = state.step.device
    animal_indices = tensor_constant(
        _ANIMAL_SHED_INDICES, dtype=torch.int64, device=device
    )
    fuel_indices = tensor_constant(_FUEL_SHED_INDICES, dtype=torch.int64, device=device)
    unlocked = (state.kind != 1).sum(dim=(-2, -1))
    plants = (state.kind == 3).sum(dim=(-2, -1))
    herd = state.shed.index_select(-1, animal_indices).sum(dim=-1)
    placed = (state.occupant > 0).sum(dim=(-2, -1))
    return _TensorCounts(
        city=(unlocked + plants).to(torch.float32),
        unit=(state.hand_count + 1).to(torch.float32),
        research=state.shop_count[:, None].expand(-1, 2).to(torch.float32),
        fuel=state.shed.index_select(-1, fuel_indices).sum(dim=-1).to(torch.float32),
        capital=(herd + placed).to(torch.float32),
        money=state.money.to(torch.float32),
        opponent=state.money.flip(1).to(torch.float32),
    )


def _toad_rewards(
    before: _TensorCounts,
    after: _TensorCounts,
    terminal: torch.Tensor,
    *,
    money_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return shaped, faithful-shaped, margin, and sparse rewards on-device."""
    result = torch.sign(after.money - after.opponent) * terminal[:, None]
    shaped_base = (
        CITY_WEIGHT * (after.city - before.city)
        + UNIT_WEIGHT * (after.unit - before.unit)
        + RESEARCH_WEIGHT * (after.research - before.research)
        + CAPITAL_WEIGHT * (after.capital - before.capital)
        + FUEL_WEIGHT * (after.fuel - before.fuel).clamp_min(0)
        + STEP_WEIGHT
    )
    terminal_reward = GAME_RESULT_WEIGHT * result
    shaped = (shaped_base + terminal_reward) / NORMALISER
    shaped_money = (
        shaped_base + money_weight * (after.money - before.money) + terminal_reward
    ) / NORMALISER
    margin = (
        MARGIN_WEIGHT
        * ((after.money - after.opponent) - (before.money - before.opponent))
        + ABSOLUTE_WEIGHT * (after.capital - before.capital)
        + terminal_reward
    ) / NORMALISER
    return shaped, shaped_money, margin, result


def belief_targets(state: SimState) -> torch.Tensor:
    """Return opposing-private labels for both acting seats, entirely on-device."""
    device = state.step.device
    product_indices = tensor_constant(
        _PRODUCT_SHED_INDICES, dtype=torch.int64, device=device
    )
    targets = []
    for seat in range(2):
        opponent = 1 - seat
        shed = state.shed[:, opponent].to(torch.float32) / SHED_CAPACITY
        seeds = state.seeds[:, opponent].to(torch.float32) / SEED_SCALE
        carried = (
            state.inv_count[:, opponent]
            .index_select(-1, product_indices)
            .sum(dim=1)
            .to(torch.float32)
            / CARRIED_SCALE
        )
        targets.append(torch.cat((shed, seeds, carried), dim=-1))
    return torch.stack(targets, dim=1)


def collect_segment(
    state: SimState,
    policy: torch.nn.Module,
    *,
    policy_state: PolicyState | RolloutPolicyState | None = None,
    turns: int = 32,
    state_unroll_length: int | None = None,
    money_weight: float = MONEY_WEIGHT,
    generator: SamplingGenerator | None = None,
    opponent: ScriptedOpponent | None = None,
    opponent_policy: torch.nn.Module | None = None,
) -> tuple[SimState, PolicyState | RolloutPolicyState | None, Trajectory]:
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
    or a CUDA generator the caller registered with the graph. Per-environment
    generators preserve game-keyed eager sampling but are not a graph-capture
    mode. A CPU generator never reaches these tensors.

    A scripted ``opponent`` is the exception, and it is not one that can be
    removed: ``scripted_actions`` reads each row's state on the host and asks a
    Python function what to play. That is a synchronisation per row per turn,
    so a scripted segment cannot be captured and is not meant to be.

    The learning seat's transfers are sampled, not fixed. Every unit draws a
    bucket from the quantity head beside its op, under a mask that mirrors
    ``learn.mask.unit_quantity_mask``, and the count it stands for is what
    ``step`` executes -- which is what lets self-play use the lane at
    simulator speed rather than only replaying a corpus through it. The
    sampling is a ``multinomial`` and a gather on tensors that never leave the
    device, so the captured region gains no synchronisation.

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
        policy_state: Recurrent actor state carried into the first observation.
            A distinct neural opponent returns ``RolloutPolicyState`` so both
            policy objects' memories cross the next segment boundary.
        turns: Turns to collect. Every turn is recorded.
        state_unroll_length: Learner unroll length whose exact segment starts
            determine the sparse entry-state rows retained in the trajectory.
        money_weight: Curriculum coefficient for the faithful shaped reward.
        generator: One sampling stream per environment, one shared stream, or
            ``None`` for the device default.
        opponent: A scripted seat 1, or ``None`` for a neural seat.
        opponent_policy: A distinct frozen/teacher seat 1 policy. Mutually
            exclusive with ``opponent``; when both are absent, self-play uses
            ``policy`` for both seats in the same forward.

    Returns:
        The state after ``turns`` turns, successor policy state, and the
        segment's ``Trajectory``. Every populated field is a device tensor,
        ``illegal`` included.
    """
    if opponent is not None and opponent_policy is not None:
        raise ValueError("scripted and neural opponents are mutually exclusive")
    if isinstance(policy_state, RolloutPolicyState):
        if opponent_policy is None:
            raise ValueError(
                "combined rollout state requires a distinct neural opponent"
            )
        learner_policy_state = policy_state.learner
        opponent_policy_state = policy_state.opponent
    else:
        learner_policy_state = policy_state
        opponent_policy_state = None
    records: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "board",
            "scalars",
            "positions",
            "unit_actions",
            "unit_quantities",
            "market_actions",
            "unit_masks",
            "unit_quantity_masks",
            "market_masks",
            "log_probs",
            "values",
            "rewards",
            "own",
            "shaped",
            "shaped_money",
            "margin",
            "sparse",
            "potentials",
            "dones",
        )
    }
    device = state.step.device
    illegal_count = torch.zeros((), dtype=torch.int64, device=device)
    retained_steps = (
        set(segment_starts(turns, state_unroll_length))
        if state_unroll_length is not None
        else {0}
    )
    retained_states: list[PolicyState] = []
    retained_indices: list[int] = []
    target_records: list[torch.Tensor] = []
    target_valid_records: list[torch.Tensor] = []
    stream_illegal = torch.zeros(
        (state.batch_size, 2), dtype=torch.int64, device=device
    )
    sales = torch.zeros((state.batch_size, 2), dtype=torch.float32, device=device)
    units_sold = torch.zeros_like(sales)
    proceeds = torch.zeros_like(sales)
    market_value = torch.zeros_like(sales)
    bought = torch.zeros_like(sales)
    actor_seats = 2 if opponent is None and opponent_policy is None else 1
    for turn in range(turns):
        observed = [observe(state, seat) for seat in range(2)]
        boards = torch.stack([value[0] for value in observed], dim=1)
        scalars = torch.stack([value[1] for value in observed], dim=1)
        positions = torch.stack([value[2] for value in observed], dim=1)
        masks = [legal(state, seat) for seat in range(2)]
        unit_masks = torch.stack([value[0] for value in masks], dim=1)
        quantity_masks = torch.stack([value[1] for value in masks], dim=1)
        market_masks = torch.stack([value[2] for value in masks], dim=1)
        batch = state.batch_size
        with torch.no_grad():
            output = _policy_forward(
                policy,
                boards[:, :actor_seats].flatten(0, 1),
                scalars[:, :actor_seats].flatten(0, 1),
                positions[:, :actor_seats].flatten(0, 1),
                state=learner_policy_state,
                dones=state.done[:, None].expand(-1, actor_seats).flatten(0, 1),
            )
        learner_policy_state = output.state
        unit_logits = output.unit_logits
        quantity_logits = output.quantity_logits
        market_logits = output.market_logits
        values = output.values
        if turn in retained_steps and output.input_state is not None:
            retained_states.append(
                _reshape_state_rows(output.input_state, batch, actor_seats)
            )
            retained_indices.append(turn)
        if output.belief_logits is not None:
            target_records.append(belief_targets(state)[:, :actor_seats])
            target_valid_records.append(
                torch.ones((batch, actor_seats), dtype=torch.bool, device=device)
            )
        unit_logits = unit_logits.reshape(batch, actor_seats, *unit_logits.shape[1:])
        quantity_logits = quantity_logits.reshape(
            batch, actor_seats, *quantity_logits.shape[1:]
        )
        market_logits = market_logits.reshape(
            batch, actor_seats, *market_logits.shape[1:]
        )
        values = values.reshape(batch, actor_seats, -1).squeeze(-1)
        main_units, main_unit_log = _sample(
            unit_logits, unit_masks[:, :actor_seats], generator
        )
        main_quantities, main_quantity_log = _sample(
            quantity_logits, quantity_masks[:, :actor_seats], generator
        )
        main_market, main_market_log = _sample(
            market_logits, market_masks[:, :actor_seats], generator
        )
        chosen_units = torch.full(
            (batch, 2, MAX_UNITS),
            UNIT_OPS.index("PASS"),
            dtype=main_units.dtype,
            device=device,
        )
        chosen_quantities = torch.zeros(
            (batch, 2, MAX_UNITS), dtype=main_quantities.dtype, device=device
        )
        chosen_market = torch.zeros(
            (batch, 2, market_masks.shape[2]),
            dtype=main_market.dtype,
            device=device,
        )
        unit_log = torch.zeros(
            (batch, 2, MAX_UNITS), dtype=main_unit_log.dtype, device=device
        )
        quantity_log = torch.zeros_like(unit_log)
        market_log = torch.zeros(
            (batch, 2, market_masks.shape[2]),
            dtype=main_market_log.dtype,
            device=device,
        )
        stored_values = torch.zeros((batch, 2), dtype=values.dtype, device=device)
        chosen_units[:, :actor_seats].copy_(main_units)
        chosen_quantities[:, :actor_seats].copy_(main_quantities)
        chosen_market[:, :actor_seats].copy_(main_market)
        unit_log[:, :actor_seats].copy_(main_unit_log)
        quantity_log[:, :actor_seats].copy_(main_quantity_log)
        market_log[:, :actor_seats].copy_(main_market_log)
        stored_values[:, :actor_seats].copy_(values)
        if opponent_policy is not None:
            with torch.no_grad():
                opponent_output = _policy_forward(
                    opponent_policy,
                    boards[:, 1],
                    scalars[:, 1],
                    positions[:, 1],
                    state=opponent_policy_state,
                    dones=state.done,
                )
            opponent_policy_state = opponent_output.state
            opponent_units, opponent_unit_log = _sample(
                opponent_output.unit_logits, unit_masks[:, 1], generator
            )
            opponent_quantities, opponent_quantity_log = _sample(
                opponent_output.quantity_logits, quantity_masks[:, 1], generator
            )
            opponent_market, opponent_market_log = _sample(
                opponent_output.market_logits, market_masks[:, 1], generator
            )
            chosen_units[:, 1].copy_(opponent_units)
            chosen_quantities[:, 1].copy_(opponent_quantities)
            chosen_market[:, 1].copy_(opponent_market)
            unit_log[:, 1].copy_(opponent_unit_log)
            quantity_log[:, 1].copy_(opponent_quantity_log)
            market_log[:, 1].copy_(opponent_market_log)
        market_orders = decode_market_buckets(chosen_market)
        # Buckets are what the head sampled and what the trajectory stores; the
        # engine is handed the quantities they stand for. The lookup is a
        # `tensor_constant` gather rather than an indexing of `QUANTITIES` on
        # the host, so nothing here reads a device tensor back.
        unit_quantities = tensor_constant(QUANTITIES, dtype=torch.int16, device=device)[
            chosen_quantities
        ]
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
        # Only a PICKUP or a PLACE spends the bucket beside it, so only those
        # slots contribute to the log-probability of the turn -- the same rule
        # `ppo.joint_log_prob` re-scores under, and for the same reason: a
        # bucket the engine never read is not part of what acted. Read off the
        # sampled op through a device-side table, never off the bucket's value,
        # which cannot tell an unspent 1 from a transfer of one item.
        transferred = (
            alive
            & tensor_constant(TRANSFER_OPS, dtype=torch.bool, device=device)[
                chosen_units
            ]
        )
        joint_log = (
            (unit_log * alive).sum(dim=-1)
            + (quantity_log * transferred).sum(dim=-1)
            + market_log.sum(dim=-1)
        )
        before_money = state.money.clone()
        before_margin = before_money - before_money.flip(1)
        before_potential = torch.stack(
            (potential(state, 0), potential(state, 1)), dim=1
        )
        before_counts = _tensor_counts(state)
        before_shed = state.shed
        before_prices = state.prices
        before_done = state.done
        next_state = step(
            state,
            chosen_units.to(torch.int16),
            market_orders,
            unit_quantities,
        )
        after_money = next_state.money
        after_margin = after_money - after_money.flip(1)
        after_counts = _tensor_counts(next_state)
        terminal = ~before_done & next_state.done
        shaped, shaped_money, margin, sparse = _toad_rewards(
            before_counts,
            after_counts,
            terminal,
            money_weight=money_weight,
        )
        records["board"].append(boards)
        records["scalars"].append(scalars)
        records["positions"].append(positions)
        records["unit_actions"].append(stored_units)
        records["unit_quantities"].append(chosen_quantities)
        records["market_actions"].append(chosen_market)
        records["unit_masks"].append(unit_masks)
        records["unit_quantity_masks"].append(quantity_masks)
        records["market_masks"].append(market_masks)
        records["log_probs"].append(joint_log)
        records["values"].append(stored_values)
        records["rewards"].append((after_margin - before_margin).to(torch.float32))
        records["own"].append((after_money - before_money).to(torch.float32))
        records["shaped"].append(shaped)
        records["shaped_money"].append(shaped_money)
        records["margin"].append(margin)
        records["sparse"].append(sparse)
        records["potentials"].append(before_potential)
        records["dones"].append(next_state.done[:, None].expand(-1, 2))
        checked_seats = slice(0, 1) if opponent is not None else slice(None)
        per_stream_illegal = (
            (~unit_masks.gather(-1, chosen_units[..., None]).squeeze(-1) & alive).sum(
                dim=-1
            )
            + (
                ~quantity_masks.gather(-1, chosen_quantities[..., None]).squeeze(-1)
            ).sum(dim=-1)
            + (~market_masks.gather(-1, chosen_market[..., None]).squeeze(-1)).sum(
                dim=-1
            )
        )
        stream_illegal.add_(per_stream_illegal)
        illegal_count.add_(per_stream_illegal[:, checked_seats].sum())

        product_indices = tensor_constant(
            _PRODUCT_SHED_INDICES, dtype=torch.int64, device=device
        )
        shed_before = before_shed.index_select(-1, product_indices)
        shed_after = next_state.shed.index_select(-1, product_indices)
        sold = (shed_before - shed_after).clamp_min(0)
        gained = (after_money - before_money) > 0
        sold = sold * gained[..., None]
        sales.add_((sold > 0).sum(dim=-1))
        units_sold.add_(sold.sum(dim=-1))
        proceeds.add_((after_money - before_money).clamp_min(0))
        market_value.add_(
            (sold.to(torch.float32) * before_prices[:, None].to(torch.float32)).sum(
                dim=-1
            )
        )
        shed_increase = (next_state.shed - before_shed).clamp_min(0).sum(dim=-1)
        bought.add_(shed_increase * ((after_money - before_money) < 0))
        state = next_state
    final_observed = [observe(state, seat) for seat in range(2)]
    records["board"].append(torch.stack([value[0] for value in final_observed], dim=1))
    records["scalars"].append(
        torch.stack([value[1] for value in final_observed], dim=1)
    )
    records["positions"].append(
        torch.stack([value[2] for value in final_observed], dim=1)
    )
    final_counts = _tensor_counts(state)
    mean_sale_price = torch.where(units_sold > 0, proceeds / units_sold, 0.0)
    mean_market_price = torch.where(units_sold > 0, market_value / units_sold, 0.0)
    realisation = torch.where(
        mean_market_price > 0, mean_sale_price / mean_market_price, 0.0
    )
    trajectory = Trajectory(
        **{name: torch.stack(values) for name, values in records.items()},
        illegal=illegal_count,
        hidden=(
            torch.stack([record.hidden for record in retained_states])
            if retained_states
            else None
        ),
        cell=(
            torch.stack([record.cell for record in retained_states])
            if retained_states
            else None
        ),
        prior_belief=(
            torch.stack([record.prior_belief for record in retained_states])
            if retained_states
            else None
        ),
        state_steps=(
            tensor_constant(tuple(retained_indices), dtype=torch.int64, device=device)
            if retained_states
            else None
        ),
        belief_targets=(torch.stack(target_records) if target_records else None),
        belief_valid=(
            torch.stack(target_valid_records) if target_valid_records else None
        ),
        final_margin=(state.money - state.money.flip(1)).to(torch.float32),
        final_bank=state.money.to(torch.float32),
        final_capital=final_counts.capital,
        illegal_by_stream=stream_illegal,
        sales=sales,
        units_sold=units_sold,
        mean_sale_price=mean_sale_price,
        realisation=realisation,
        bought=bought,
        sale_proceeds=proceeds,
        sale_market_value=market_value,
    )
    successor: PolicyState | RolloutPolicyState | None
    if opponent_policy is not None:
        successor = RolloutPolicyState(
            learner=learner_policy_state,
            opponent=opponent_policy_state,
        )
    else:
        successor = learner_policy_state
    return state, successor, trajectory
