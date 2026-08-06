"""Turning one observation into the tensors a policy network reads.

Plane order is derived from the engine's own rules tables (``CROPS``,
``ANIMALS``, ``PRODUCTS``) rather than written out by hand, so a crop or
animal added upstream shifts nothing but the width of its own block.

Both farms are public in this game, so both get real capacity. Rather than
overlay the two farms onto one shared set of tile planes -- where a crop of
ours and a crop of the opponent's at the same coordinate would silently
overwrite one another -- each farm gets its own identical block of planes,
``seat``'s own farm first and the opponent's second. The market branch keeps
a price signal and an inventory signal per product, because value in this
game is a property of a product given what the opponent is selling: an
earlier change that replaced an opponent-aware animal valuation with a
static per-animal one measured 10,000 coins worse over 20 games.

Each farm block also carries where its units stand. Without those planes the
trunk sees a board with nobody on it, so the network can learn what the day
calls for but never that *this* hand is the one standing next to the weeds.
Both farms get them, since the opponent's units are public too.
"""

from typing import Any, Mapping

import torch

from kaggriculture.constants import (
    ANIMALS,
    BOARD_SIZE,
    CROPS,
    EPISODE_STEPS,
    LAND_ORDER,
    MARKET_PARAMS,
    MAX_MARKET_ORDERS_PER_TURN,
    PRODUCTS,
    SEASON_DAYS,
    SHOPS,
    TURNS_PER_DAY,
)
from kaggriculture.observation import Tile

BOARD = BOARD_SIZE

CROP_NAMES = sorted(CROPS)
ANIMAL_NAMES = sorted(ANIMALS)
PRODUCT_NAMES = sorted(PRODUCTS)

# The main farmer plus enough hands for a full day's hiring. There is no
# engine-enforced cap on hand count: hires_today resets every day and each
# hire only costs mult * fib(hires_today), so a rich-enough farm can keep
# hiring. 13 (farmer + 12 hands, this project's own heuristic policy's
# hiring target) is not a bound on what other agents actually do: sampling
# 240 episodes across the six replay archives on disk turned up a farm with
# 14 hands (15 acting units) in a single day. MAX_UNITS is set with headroom
# above that observed value, and encode_units/encode_positions/decode_units
# raise rather than silently drop a real unit if it is ever exceeded, so a
# wider hand count can never become a misaligned label.
MAX_UNITS = 20

# Per farm, in order: one plane per crop, one per animal, four mutually
# exclusive tile states, five continuous per-tile features, then two
# occupancy planes. A crop or animal lights its own plane inside a PLANT /
# occupied-structure tile; the state planes cover everything else, including
# a bare (unoccupied) structure.
#
# Occupancy is two planes rather than one because the farmer and a hand are
# different pieces, and rather than a flag per unit because several hands may
# stand on one tile -- [[4, 3], [4, 3]] occurs in the corpus. The hands plane
# is therefore a count, scaled by MAX_UNITS so it shares the range of the
# other continuous features.
_TILE_STATES = ("WEED", "LOCKED", "EMPTY", "STRUCTURE")
_TILE_FEATURES = ("ACTIVE_TODAY", "DISTRESS", "YIELD_FRACTION", "BONUS_READY", "AGE")
_UNIT_PLANES = ("FARMER", "HANDS")

_ANIMAL_BASE = len(CROP_NAMES)
_STATE_BASE = _ANIMAL_BASE + len(ANIMAL_NAMES)
_FEATURE_BASE = _STATE_BASE + len(_TILE_STATES)
_UNIT_BASE = _FEATURE_BASE + len(_TILE_FEATURES)
_PER_FARM_PLANES = _UNIT_BASE + len(_UNIT_PLANES)

TILE_PLANES = 2 * _PER_FARM_PLANES

# Both a price and an inventory signal per product, our money and the
# opponent's, the day/hour/step phase of the season, and how much land, town
# shops and hired hands are unlocked on each side (the opponent's farm is
# public, so their counts are as knowable as ours).
SCALARS = 2 * len(PRODUCT_NAMES) + 10

_MAX_QUADRANTS = 1 + len(LAND_ORDER)


def encode_board(observation: Mapping[str, Any], seat: int) -> torch.Tensor:
    """Return the board planes for one seat.

    ``seat``'s own farm occupies the first ``TILE_PLANES // 2`` planes and the
    opponent's farm occupies the rest. Keeping the two farms in separate
    planes, rather than overlaying them on one shared set, means a crop on the
    opponent's board can never silently overwrite one of ours at the same
    coordinate.

    Args:
        observation: One turn's observation.
        seat: Which player's farm goes in the first block of planes.

    Returns:
        A ``(1, TILE_PLANES, BOARD, BOARD)`` float32 tensor.
    """
    day = observation["day"]
    farms = observation["farms"]
    planes = torch.zeros(1, TILE_PLANES, BOARD, BOARD, dtype=torch.float32)
    for block, farm in enumerate((farms[seat], farms[1 - seat])):
        base = block * _PER_FARM_PLANES
        for y, row in enumerate(farm["tiles"]):
            for x, tile in enumerate(row):
                _write_tile(planes, base, tile, y, x, day)
        _write_units(planes, base, farm)
    return planes


def _write_tile(
    planes: torch.Tensor, base: int, tile: Tile, y: int, x: int, day: int
) -> None:
    """Write one farm's tile into its block of planes, in place."""
    if tile is None:
        planes[0, base + _STATE_BASE + _TILE_STATES.index("EMPTY"), y, x] = 1.0
        return
    if tile == "LOCKED":
        planes[0, base + _STATE_BASE + _TILE_STATES.index("LOCKED"), y, x] = 1.0
        return
    kind = tile["kind"]
    if kind == "WEED":
        planes[0, base + _STATE_BASE + _TILE_STATES.index("WEED"), y, x] = 1.0
        return
    if kind == "PLANT":
        crop = tile["crop"]
        planes[0, base + CROP_NAMES.index(crop), y, x] = 1.0
        _write_features(
            planes,
            base,
            y,
            x,
            active_today=tile["watered_today"],
            distress=tile["consecutive_unwatered"],
            yield_units=tile["yield_units"],
            yield_capacity=int(CROPS[crop]["max_yield"]),
            bonus_ready=tile["fertilized_until_day"] >= day,
            age=day - tile["planted_day"],
        )
        return
    animal = tile.get("animal")
    if animal is None:
        planes[0, base + _STATE_BASE + _TILE_STATES.index("STRUCTURE"), y, x] = 1.0
        return
    planes[0, base + _ANIMAL_BASE + ANIMAL_NAMES.index(animal), y, x] = 1.0
    _write_features(
        planes,
        base,
        y,
        x,
        active_today=tile["fed_today"],
        distress=tile["consecutive_unfed"],
        yield_units=tile["yield_units"],
        yield_capacity=int(ANIMALS[animal]["max_held"]),
        bonus_ready=tile["fertilizer_available"],
        age=day - tile["placed_day"],
    )


def _write_features(
    planes: torch.Tensor,
    base: int,
    y: int,
    x: int,
    active_today: bool,
    distress: int,
    yield_units: int,
    yield_capacity: int,
    bonus_ready: bool,
    age: int,
) -> None:
    """Write the five continuous features shared by plants and animals, in place.

    A plant's ``watered_today`` / ``consecutive_unwatered`` and an animal's
    ``fed_today`` / ``consecutive_unfed`` are the same shape of feature (did
    the day's required care happen, how many days running has it been
    missed), so they share one set of planes rather than four kind-specific
    ones.
    """
    offset = base + _FEATURE_BASE
    planes[0, offset + 0, y, x] = float(active_today)
    planes[0, offset + 1, y, x] = distress / 2.0
    planes[0, offset + 2, y, x] = yield_units / yield_capacity
    planes[0, offset + 3, y, x] = float(bonus_ready)
    planes[0, offset + 4, y, x] = age / SEASON_DAYS


def _write_units(planes: torch.Tensor, base: int, farm: Mapping[str, Any]) -> None:
    """Write one farm's unit occupancy into its block of planes, in place.

    The hands plane accumulates rather than sets: several hands may stand on
    one tile, and a flag would report a crowd of five the same as a lone hand.
    """
    farmer_y, farmer_x = farm["farmer"]
    planes[0, base + _UNIT_BASE, farmer_y, farmer_x] = 1.0
    for y, x in farm["hands"]:
        planes[0, base + _UNIT_BASE + 1, y, x] += 1.0 / MAX_UNITS


def encode_scalars(observation: Mapping[str, Any], seat: int) -> torch.Tensor:
    """Return the non-spatial features for one seat.

    Both a price and an inventory signal are kept per product: the price says
    what the next unit fetches right now, and the inventory says how far
    already-committed supply -- ours or the opponent's -- has pushed it from
    baseline, which is what decides where the price goes next. Land, town
    shops and hired hands are included for both seats, since the opponent's
    farm is public.

    The global step is derived from ``day`` and ``hour`` rather than read from
    ``observation["step"]``: kaggle_environments' own core loop only ever
    writes ``step`` onto agent 0's observation (``new_state[0].observation.step
    = ...`` in its ``core.py``), so any other seat's real observation has no
    ``step`` key at all. ``day`` and ``hour`` are mirrored onto every agent by
    the game's interpreter, and reconstruct ``step`` exactly, since the
    interpreter itself derives them as ``next_step // turns_per_day`` and
    ``next_step % turns_per_day``.

    Args:
        observation: One turn's observation.
        seat: Which player to encode for.

    Returns:
        A ``(1, SCALARS)`` float32 tensor.
    """
    market = observation["market"]
    prices = market["prices"]
    inventory = market["inventory"]
    farms = observation["farms"]
    ours, theirs = farms[seat], farms[1 - seat]

    values = [
        (prices[item] - MARKET_PARAMS[item]["base"]) / MARKET_PARAMS[item]["base"]
        for item in PRODUCT_NAMES
    ]
    values += [
        (inventory[item] - MARKET_PARAMS[item]["I0"]) / MARKET_PARAMS[item]["T"]
        for item in PRODUCT_NAMES
    ]
    values += [
        ours["money"] / 10_000.0,
        theirs["money"] / 10_000.0,
        observation["day"] / SEASON_DAYS,
        observation["hour"] / TURNS_PER_DAY,
        (observation["day"] * TURNS_PER_DAY + observation["hour"]) / EPISODE_STEPS,
        len(observation["town"]["unlocked_shops"]) / len(SHOPS),
        len(ours["unlocked_quadrants"]) / _MAX_QUADRANTS,
        len(theirs["unlocked_quadrants"]) / _MAX_QUADRANTS,
        len(ours["hands"]) / 8.0,
        len(theirs["hands"]) / 8.0,
    ]
    return torch.tensor(values, dtype=torch.float32).reshape(1, SCALARS)


# One label per distinct decision. PLANT carries its crop because planting melon
# and planting wheat are different choices, not one op with a detail attached --
# the crop is the decision that collapsed our own agent's price this morning.
#
# DROP is here even though the engine's own JSON action spec never mentions it:
# the spec text lists seventeen ops, but `_apply_unit_action` in the engine's
# source implements an eighteenth, DROP, and this project's own policies emit
# it routinely. The docs are not authoritative; the engine's code is.
UNIT_OPS: tuple[str, ...] = (
    "PASS",
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "WATER",
    "HARVEST",
    "DIG",
    "FEED",
    "CARE",
    "COLLECT_FERTILIZER",
    "FERTILIZE",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "PICKUP",
    "PLACE",
    "DROP",
) + tuple(f"PLANT:{crop}" for crop in CROP_NAMES)

# torch's cross entropy ignores this index, so padded units contribute no loss.
IGNORE = -100


class TooManyUnitsError(ValueError):
    """Raised when a turn has more acting units than ``MAX_UNITS`` covers.

    Kept distinct from the plain ``ValueError`` that ``_label`` raises for an
    op outside ``UNIT_OPS`` (``tuple.index(x): x not in tuple``), so a caller
    that wants to tolerate an oversized turn -- an empirical, not proven,
    bound -- can catch exactly this and let an unknown-op bug propagate
    instead of being silently counted as the same kind of failure.
    """


def unit_count(observation: Mapping[str, Any], seat: int) -> int:
    """Return how many units act for ``seat`` from this observation.

    The single definition of how many slots are real, so positions and labels
    cannot disagree about it. The observation, not the action, is authoritative:
    the engine walks ``[farmer, *farm["hands"]]`` and its ``_farmer_position``
    returns ``None`` past the end of ``hands``, making any further op in the
    action a silent no-op.

    The counts do diverge, on 2.4% of correctly paired turns overall, and the
    rate depends entirely on which day's archive you look at: 0% across
    2026-07-30 through 08-01, then 1.7%, 8.1% and 4.4% on 08-02 through 08-04.
    Measuring one archive and generalising is how this docstring previously came
    to claim the disagreement did not exist at all. Usually the action carries
    one to five hand ops more than the farm has hands; occasionally, on 08-04,
    one fewer.

    Truncating to ``units`` is safe in both directions. The engine applies unit
    ops before ``_process_market``, where HIRE lands, so a hand hired this turn
    cannot act this turn; surplus ops address units that do not exist yet and
    the engine no-ops them identically. A short action simply leaves the trailing
    slots at ``IGNORE``, training nothing rather than training something wrong.

    That is the secondary reason for this function. The primary one is that
    ``encode_positions`` and ``encode_units`` must agree about how many slots are
    real -- if they drift apart, every unit is trained on another unit's
    surroundings and nothing raises.

    Args:
        observation: One turn's observation.
        seat: Which player's units to count.

    Returns:
        The farmer plus the hands standing on ``seat``'s farm.
    """
    return 1 + len(observation["farms"][seat]["hands"])


def encode_positions(observation: Mapping[str, Any], seat: int) -> torch.Tensor:
    """Return the tile each acting unit occupies, padded to ``MAX_UNITS``.

    Slots are ordered exactly as ``encode_units`` labels them -- slot 0 the
    farmer, slot *k* the *k*-th hand -- because the head reads slot *k*'s trunk
    column at slot *k*'s position and scores it against slot *k*'s label. If
    the two orders diverged, every unit would be trained on another unit's
    surroundings and nothing would raise.

    Padded slots take tile index 0. That is deliberate, not a fallback: the
    head gathers at these indices unconditionally, so every slot needs a valid
    one, and a padded slot's label is ``IGNORE``, so its logits never reach the
    loss and the tile it nominally read is never learned from.

    Args:
        observation: One turn's observation -- the state the decision was made
            from, so it must be the same observation the row's board encodes.
        seat: Which player's units to locate.

    Returns:
        A ``(1, MAX_UNITS)`` int64 tensor of flattened ``y * BOARD + x`` tile
        indices.

    Raises:
        TooManyUnitsError: If more units are on the board than ``MAX_UNITS``
            covers.
    """
    units = unit_count(observation, seat)
    if units > MAX_UNITS:
        raise TooManyUnitsError(f"{units} acting units exceeds MAX_UNITS={MAX_UNITS}")
    farm = observation["farms"][seat]
    positions = torch.zeros(1, MAX_UNITS, dtype=torch.int64)
    for index, (y, x) in enumerate([farm["farmer"], *farm["hands"]]):
        positions[0, index] = y * BOARD + x
    return positions


def encode_units(action: Mapping[str, Any], units: int) -> torch.Tensor:
    """Return one label per unit, padded to ``MAX_UNITS``.

    Units that did not act -- hands not yet hired -- are marked with the
    ignore index rather than a ``PASS`` label. Teaching the model that an
    absent hand chose to pass would train it to pass.

    ``units`` comes from ``unit_count`` on the observation the action was taken
    from, and ops past it are dropped rather than labelled. About 5% of corpus
    turns emit ops for hands the farm does not have; the engine no-ops them, so
    labelling them would attach a real op to a slot holding no unit -- and, now
    that the head reads each slot at its unit's tile, would train the readout on
    a padded tile nobody is standing on.

    Args:
        action: One recorded turn's action dict.
        units: How many units are actually on the board this turn.

    Returns:
        A ``(1, MAX_UNITS)`` int64 tensor of labels.

    Raises:
        TooManyUnitsError: If ``units`` exceeds ``MAX_UNITS``.
    """
    if units > MAX_UNITS:
        raise TooManyUnitsError(f"{units} acting units exceeds MAX_UNITS={MAX_UNITS}")
    ops = [action["farmer"], *action["hands"]][:units]
    labels = torch.full((1, MAX_UNITS), IGNORE, dtype=torch.int64)
    for index, op in enumerate(ops):
        labels[0, index] = _label(op)
    return labels


def _label(op: list[Any]) -> int:
    """Return the vocabulary index for one unit's recorded op.

    Args:
        op: One unit's op, e.g. ``["WATER"]`` or ``["PLANT", "MELON"]``.

    Returns:
        The op's index into ``UNIT_OPS``.
    """
    verb = str(op[0])
    name = f"PLANT:{op[1]}" if verb == "PLANT" else verb
    return UNIT_OPS.index(name)


def decode_units(logits: torch.Tensor, units: int) -> dict[str, Any]:
    """Return the action dict implied by per-unit logits.

    Args:
        logits: A ``(1, MAX_UNITS, len(UNIT_OPS))`` tensor.
        units: How many units are actually on the board this turn.

    Returns:
        An action dict with ``farmer``, ``hands`` and an empty ``market``
        (market orders are decoded elsewhere).

    Raises:
        TooManyUnitsError: If ``units`` exceeds ``MAX_UNITS``.
    """
    if units > MAX_UNITS:
        raise TooManyUnitsError(f"{units} units exceeds MAX_UNITS={MAX_UNITS}")
    chosen = logits[0, :units].argmax(dim=-1)
    ops = [_op(int(index.item())) for index in chosen]
    return {"farmer": ops[0], "hands": ops[1:], "market": []}


def _op(label: int) -> list[Any]:
    """Return the op list for one vocabulary index."""
    name = UNIT_OPS[label]
    if name.startswith("PLANT:"):
        return ["PLANT", name.split(":", 1)[1]]
    return [name]


# Every (verb, item) pair the engine's `_process_market` will actually act on.
# SELL covers the whole PRODUCTS catalogue: `_commit_unit` gates a sale only on
# whether the shed holds the item, never on which item it is. BUY_SEED and
# BUY_ANIMAL are gated by `item in CROPS` / `item in ANIMALS`, the whole
# catalogue in each case.
#
# BUY_PRODUCT is narrower. `_process_market` gates it with a literal
# `item in ("WHEAT", "FERTILIZER")` that has nothing to do with the rest of
# PRODUCTS -- CARROT, MELON and the animal products are things a farm grows,
# not things the market will sell back to it. Building this slot from
# `sorted(PRODUCTS)` instead, as the wider catalogue tempts, would add seven
# slots the engine silently no-ops on every turn the model fills them, each
# one crowding a real order out of a ten-order turn.
#
# That gate exists only as an inline literal inside `_process_market`, not as
# a named constant, so it cannot be imported. It is typed by hand here rather
# than parsed out of the engine's source at import time: this module is
# imported by the submitted agent (`kaggriculture.learn.play`), which
# runs inside the competition sandbox against a `kaggle_environments` build
# this project does not control, and `len(MARKET_SLOTS)` sets the market
# head's output shape. A source-parse that raises or silently changes shape
# on a different engine build forfeits the episode with no logs to explain
# why; a hand-typed tuple that has drifted still plays. What keeps this tuple
# honest instead is
# `test_buy_product_s_item_gate_matches_the_engine_exactly`, which parses the
# engine's source at test time, in CI, where a mismatch fails loudly and
# someone can fix it.
_BUY_PRODUCT_ITEMS = ("FERTILIZER", "WHEAT")

MARKET_SLOTS: tuple[tuple[str, str], ...] = (
    tuple(("SELL", item) for item in PRODUCT_NAMES)
    + tuple(("BUY_SEED", item) for item in CROP_NAMES)
    + tuple(("BUY_PRODUCT", item) for item in _BUY_PRODUCT_ITEMS)
    + tuple(("BUY_ANIMAL", item) for item in ANIMAL_NAMES)
)

# HIRE and BUY_LAND are atomic engine ops (`_parse_order` returns them with no
# item or quantity), so they get their own slots after the (verb, item) pairs
# rather than a row each in MARKET_SLOTS.
HIRE_SLOT = len(MARKET_SLOTS)
LAND_SLOT = len(MARKET_SLOTS) + 1

# The engine truncates a turn's market orders to `maxMarketOrdersPerTurn`
# before processing them (`_process_market`: `queues.append(q[:max_orders])`),
# reading it from the environment configuration with a default of 10. The
# corpus never configures it away from that default, and hits the cap 903
# times -- more often than it emits exactly 9 orders -- so it is a real,
# frequently-binding limit on how many orders a turn can carry, not a
# theoretical one.
MAX_ORDERS = MAX_MARKET_ORDERS_PER_TURN

# The first thirteen buckets are exact counts: 93.5% of orders in the corpus
# are 12 or fewer, and lumping that dense range into ranges would blur most of
# the distribution the model has to predict. The remaining four buckets
# summarize the long tail -- 13-20, 21-32, 33-52, 53+ -- each labelled by one
# representative quantity, so a big order is still distinguishable from no
# order without one class per order size seen only a handful of times.
QUANTITIES: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 24, 40, 64)

# The inclusive upper bound of each of the three bounded tail buckets (index
# 13, 14, 15); the fourth tail bucket (index 16, representative 64) is
# unbounded above. These are not derivable from QUANTITIES itself -- the
# representative quantities step by a roughly-1.5x progression (12, 16, 24,
# 40, 64) while the ranges they stand for widen faster (8, 12, 20, then
# open-ended), so "the next representative minus one" is not this scheme;
# the ranges are their own design choice and are written out accordingly.
_TAIL_BUCKET_MAX = (20, 32, 52)


def bucket_of(n: int) -> int:
    """Return the bucket index that quantity ``n`` falls into.

    The first thirteen buckets are exact: bucket ``k`` is quantity ``k``
    itself, for ``k`` up to 12. Past that, a quantity falls into whichever of
    the three bounded tail ranges (13-20, 21-32, 33-52) contains it, or the
    open-ended 53+ bucket if it exceeds all of them.

    Args:
        n: A non-negative quantity, possibly already summed across repeated
            orders for the same (verb, item) pair.

    Returns:
        An index into ``QUANTITIES``.
    """
    if n <= 12:
        return n
    for offset, upper in enumerate(_TAIL_BUCKET_MAX):
        if n <= upper:
            return 13 + offset
    return len(QUANTITIES) - 1


def quantity_of(bucket: int) -> int:
    """Return the quantity one bucket index represents.

    Args:
        bucket: An index into ``QUANTITIES``, typically an argmaxed logit.

    Returns:
        ``QUANTITIES[bucket]``.
    """
    return QUANTITIES[bucket]


def encode_market(action: Mapping[str, Any]) -> torch.Tensor:
    """Return one bucketed label per market slot.

    Unlike ``encode_units``, no market slot is ever ``IGNORE``. A unit slot is
    padding when the farm has no hand standing there yet; a market slot has no
    such state; every slot -- sell this product, buy that seed, hire, buy land
    -- is a genuine decision on every turn, and "trade nothing" is itself the
    meaningful class 0 rather than something to mask out: measured across all six
    archives, the corpus trades nothing on 46.3% of turns. The market loss
    therefore masks nothing, where the unit loss masks every unhired hand's
    padding.

    Repeated orders for the same (verb, item) pair are summed before
    bucketing -- 22% of order-bearing turns repeat a pair, and keeping only
    the last one would discard a real sale or purchase the teacher made.
    ``HIRE`` carries no item and is counted rather than summed, since one
    ``HIRE`` order hires exactly one hand. ``BUY_LAND`` collapses to a flag,
    since the engine performs it at most once per turn regardless of how many
    ``BUY_LAND`` orders are queued (``_do_buy_land`` is called once per
    occurrence, but a second call the same turn is a no-op once the quadrant
    is already unlocked).

    Args:
        action: One recorded turn's action dict.

    Returns:
        A ``(1, len(MARKET_SLOTS) + 2)`` int64 tensor of bucket indices.
    """
    totals = [0] * len(MARKET_SLOTS)
    hires = 0
    land = 0
    for order in action["market"]:
        verb = order[0]
        if verb == "HIRE":
            hires += 1
        elif verb == "BUY_LAND":
            land = 1
        else:
            totals[MARKET_SLOTS.index((verb, order[1]))] += int(order[2])
    labels = [bucket_of(total) for total in totals] + [bucket_of(hires), land]
    return torch.tensor(labels, dtype=torch.int64).reshape(1, len(MARKET_SLOTS) + 2)


def decode_market(logits: torch.Tensor) -> list[list[Any]]:
    """Return the market orders implied by per-slot logits.

    Orders are emitted in ``MARKET_SLOTS`` order, so every ``SELL`` precedes
    every ``BUY_*`` since the ``SELL`` block is built first. The engine
    processes one turn's orders in list order, one unit of quantity at a time
    per order before moving to the next, so a sale queued ahead of a purchase
    can fund it; queued the other way, the purchase would be evaluated against
    money the sale had not yet raised. ``HIRE`` orders follow, repeated by
    count, then ``BUY_LAND``.

    The result is truncated to ``MAX_ORDERS``: the engine silently drops
    anything past ``maxMarketOrdersPerTurn``, so emitting more here would
    never reach the game.

    Args:
        logits: A ``(1, len(MARKET_SLOTS) + 2, len(QUANTITIES))`` tensor.

    Returns:
        A list of order lists, e.g. ``["SELL", "WHEAT", 4]`` or ``["HIRE"]``.
    """
    buckets = logits[0].argmax(dim=-1)
    orders: list[list[Any]] = []
    for slot, (verb, item) in enumerate(MARKET_SLOTS):
        bucket = int(buckets[slot].item())
        if bucket != 0:
            orders.append([verb, item, quantity_of(bucket)])
    orders.extend([["HIRE"]] * quantity_of(int(buckets[HIRE_SLOT].item())))
    if int(buckets[LAND_SLOT].item()) != 0:
        orders.append(["BUY_LAND"])
    return orders[:MAX_ORDERS]
