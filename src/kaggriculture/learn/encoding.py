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
    action a silent no-op. About 5% of corpus turns emit a hand-op list that
    does not match the farm -- usually one or two too many, sometimes two too
    few -- so the two counts genuinely differ and only one of them decides
    anything.

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
