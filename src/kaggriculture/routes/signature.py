"""The state signature: an identity-free key for matching a board to a route.

A route recorded from one episode is only worth replaying against a live
board if the two are actually alike, and what makes them alike is our own
farm and the phase of the season -- not who happens to be sitting across the
table this game. A signature built from ``farms[1 - seat]`` would tie every
recorded prototype to the matchup it was harvested from, and the moment a
route store starts encoding the opponent it stops transferring to a different
one. So this module reads only ``observation["farms"][seat]`` (plus
``observation["private"]``, which already belongs to whichever seat's
observation this is) and the global fields mirrored identically to both seats
-- ``day`` and ``hour`` -- and never ``farms[1 - seat]``.

``observation["step"]`` is not used, on purpose:
``kaggle_environments``'s core loop only ever writes ``step`` onto agent 0's
observation, so a real seat-1 observation has no ``step`` key at all.
``day`` and ``hour`` are mirrored onto every agent by the game's interpreter
and reconstruct the same phase, so they are what this module reads.

This is deliberately not built on ``kaggriculture.learn.encoding``. That
module produces a dense per-tile tensor for a neural trunk and imports torch
to build it; this module has to run inside the submission sandbox, where
importing torch costs 10.7 seconds of a 60-second overage pool before route
memory has done anything. A route-memory key is also coarser than a training
tensor by design -- it describes board *composition*, not board *layout* --
so it is written here as a short, hand-built tuple of floats instead.

Every field is scaled into a comparable range at the point ``signature``
builds it, the same way ``learn/encoding.py`` scales at encode time rather
than leaving raw counts for a caller to normalize later. This is not
cosmetic: an early version of this module returned raw values and let
``distance`` weight them by phase-vs-composition group only, and on two real
corpus observations at day 16 the ``MONEY`` field alone accounted for 99.96%
of the total L1 distance -- every tile, weed and shed count combined moved
the needle 0.04%. Retrieval built on that key would pick the prototype with
the nearest bank balance and ignore what the farm actually looked like, which
is close to the worst possible matching signal: bank is an *outcome* of a
route, not a description of the board state a route should be selected for.
``test_no_single_field_dominates_a_real_distance`` in the test module pins
this down against real data.
"""

import math
from typing import Any, Mapping

from kaggriculture.constants import (
    ANIMALS,
    BOARD_SIZE,
    CROPS,
    LAND_ORDER,
    PRODUCTS,
    SEASON_DAYS,
    SHED_CAPACITY,
    TURNS_PER_DAY,
)

CROP_NAMES = tuple(sorted(CROPS))
ANIMAL_NAMES = tuple(sorted(ANIMALS))
PRODUCT_NAMES = tuple(sorted(PRODUCTS))

# A built-but-empty coop or pasture is real board state -- it cost land and
# money and it is committed to one kind of animal -- but before this module's
# first version counted only crops, animals, weeds and locked tiles, so a
# farm with three bare coops looked identical to one with bare ground. Names
# come from the rules table rather than being typed by hand, the same
# reasoning `learn/encoding.py`'s `_STRUCTURE_KINDS` gives: a new animal with
# a new structure kind gets its own field automatically instead of silently
# colliding with an existing one.
_STRUCTURE_KINDS = tuple(sorted({str(data["structure"]) for data in ANIMALS.values()}))

# One entry per value `signature` fills, in the exact order it fills them:
# tile composition (what is growing, standing, weedy or unavailable), the
# farm's operating state, the season's phase, then what is banked in the shed
# waiting to sell. `HIRES_TODAY` is deliberately absent -- see `HANDS_SCALE`.
SIGNATURE_FIELDS: tuple[str, ...] = (
    tuple(f"CROP:{crop}" for crop in CROP_NAMES)
    + tuple(f"ANIMAL:{animal}" for animal in ANIMAL_NAMES)
    + tuple(f"BARE:{kind}" for kind in _STRUCTURE_KINDS)
    + ("WEEDS", "LOCKED", "MONEY", "HANDS", "UNLOCKED_QUADRANTS")
    + ("DAY", "HOUR")
    + tuple(f"SHED:{product}" for product in PRODUCT_NAMES)
)

# Which fields carry the season's phase rather than the farm's composition;
# `distance` weights the two groups differently. Every other field is a
# composition field.
_PHASE_FIELDS = frozenset({"DAY", "HOUR"})
_PHASE_INDICES = tuple(
    index for index, name in enumerate(SIGNATURE_FIELDS) if name in _PHASE_FIELDS
)
_COMPOSITION_INDICES = tuple(
    index for index, name in enumerate(SIGNATURE_FIELDS) if name not in _PHASE_FIELDS
)

# A crop, animal, bare-structure, weed or locked count can never exceed the
# number of tiles on the board -- this is an engine-enforced bound, not a
# measured one, the same reasoning `constants.py`'s `BOARD_SIZE` already
# carries and `learn/encoding.py`'s per-tile planes rely on implicitly.
TILE_COUNT_SCALE = float(BOARD_SIZE * BOARD_SIZE)

# Unlocked quadrants are bounded the same way: NW starts unlocked and
# `LAND_ORDER` lists every quadrant left to buy, so this many is the most a
# farm can ever have -- again structural, not measured.
_MAX_QUADRANTS = float(1 + len(LAND_ORDER))

# Money is the field that broke this module's first version. Measured across
# all seven corpus archives on disk (10 episodes each, both seats, every 7th
# turn -- 14,420 seat-turns): p50 10,639, p99 126,794, max 151,787. It climbs
# through the episode rather than sitting near one value, so a linear divide
# by the max compresses early-game gaps to near-invisible while late-game
# gaps still dominate -- a 500-coin difference on day 2 and a 5,000-coin
# difference on day 20 can represent the same *proportional* divergence in
# spending, not a 10x-different one. MONEY is therefore compressed with
# `math.log1p` before scaling, so it is the relative gap that carries weight,
# not the absolute one. `MONEY_LOG_SCALE` sits just above the measured max so
# the largest bank seen in this sample normalizes to just under 1.0.
MONEY_LOG_SCALE = 160_000.0

# `HANDS`, not the measured max of `HIRES_TODAY`: this module originally
# carried both, but the engine clears `farm["hands"]` to `[]` and
# `farm["hires_today"]` to 0 together at the end of every day (kaggriculture's
# `_end_of_day`), and the only place either grows is `_do_hire`, which appends
# a hand and increments `hires_today` in the same call -- so
# `len(farm["hands"]) == farm["hires_today"]` for the whole game, verified
# against 4,320 real observations with zero mismatches. `HIRES_TODAY` carried
# no information `HANDS` did not already have, so it is dropped rather than
# kept as a second copy of the same fact. Measured over the same 14,420
# seat-turns as `MONEY_LOG_SCALE`: p50 10, p99 13, max 15 -- scaled to the
# next power of two above p99, the convention `learn/encoding.py` already
# uses for `CARE_BONUS_SCALE` and `UNIT_CARRIED_SCALE`.
HANDS_SCALE = 16.0


def signature(observation: Mapping[str, Any], seat: int) -> tuple[float, ...]:
    """Return the identity-free state key for ``seat``'s farm.

    Reads only ``observation["farms"][seat]`` and the global fields mirrored
    identically to both seats (``day``, ``hour``), plus
    ``observation["private"]``, which -- following the convention
    ``kaggriculture.learn.encoding`` already uses -- belongs to whichever
    seat's observation this is and so is never indexed by seat. Nothing about
    the opponent's farm enters this function: a signature only transfers to a
    route recorded against a different opponent if it never described this
    one in the first place.

    Every field is scaled to a comparable range here, at construction, rather
    than left raw for `distance` to normalize later -- see the module
    docstring for why an unscaled `MONEY` field made this signature useless.

    Args:
        observation: One turn's observation, as handed to the agent.
        seat: Which player's farm to describe.

    Returns:
        A fixed-width tuple of floats, ordered as ``SIGNATURE_FIELDS``, each
        roughly within ``[0, 1]``.
    """
    farm = observation["farms"][seat]
    crop_counts = dict.fromkeys(CROP_NAMES, 0)
    animal_counts = dict.fromkeys(ANIMAL_NAMES, 0)
    bare_counts = dict.fromkeys(_STRUCTURE_KINDS, 0)
    weeds = 0
    locked = 0
    for row in farm["tiles"]:
        for tile in row:
            if tile is None:
                continue
            if tile == "LOCKED":
                locked += 1
                continue
            kind = tile["kind"]
            if kind == "WEED":
                weeds += 1
            elif kind == "PLANT":
                crop_counts[tile["crop"]] += 1
            elif tile.get("animal") is not None:
                animal_counts[tile["animal"]] += 1
            else:
                # A built COOP or PASTURE with nothing placed on it yet.
                bare_counts[kind] += 1
    shed = observation["private"]["shed"]

    values = [float(crop_counts[crop]) / TILE_COUNT_SCALE for crop in CROP_NAMES]
    values += [
        float(animal_counts[animal]) / TILE_COUNT_SCALE for animal in ANIMAL_NAMES
    ]
    values += [float(bare_counts[kind]) / TILE_COUNT_SCALE for kind in _STRUCTURE_KINDS]
    values += [
        float(weeds) / TILE_COUNT_SCALE,
        float(locked) / TILE_COUNT_SCALE,
        math.log1p(farm["money"]) / math.log1p(MONEY_LOG_SCALE),
        float(len(farm["hands"])) / HANDS_SCALE,
        float(len(farm["unlocked_quadrants"])) / _MAX_QUADRANTS,
        float(observation["day"]) / SEASON_DAYS,
        float(observation["hour"]) / TURNS_PER_DAY,
    ]
    values += [float(shed[product]) / SHED_CAPACITY for product in PRODUCT_NAMES]
    return tuple(values)


# Distance weights for the two field groups, at the season's start and its
# end. Early routes are phase-locked -- every good route waters at the same
# hour on day 0, so an hour apart should cost more than a crop apart -- while
# late in the season routes have diverged by what they built rather than when
# they built it, so the ranking flips. `COMPOSITION_WEIGHT_LATE` sits well
# above `PHASE_WEIGHT_EARLY`'s mirror value because the two groups are scaled
# by different divisors (a tile count out of 100 versus an hour out of 24), so
# a single-tile composition difference is inherently quieter than a
# multi-hour phase difference before any weighting -- the weight has to
# overcome that gap, not just flip the ranking. These four numbers are set
# only to get that ranking right at the two ends this module's tests check;
# the day the crossover actually happens is unmeasured. Task 5 sweeps it
# against real route outcomes -- treat the linear interpolation below as a
# placeholder shape, not a fitted one.
PHASE_WEIGHT_EARLY = 8.0
PHASE_WEIGHT_LATE = 0.5
COMPOSITION_WEIGHT_EARLY = 0.5
COMPOSITION_WEIGHT_LATE = 40.0


def field_contributions(
    a: tuple[float, ...], b: tuple[float, ...], day: int
) -> tuple[float, ...]:
    """Return each field's weighted contribution to `distance`.

    Broken out from `distance` so a caller -- and this module's own tests --
    can see which fields a comparison actually turned on, in `SIGNATURE_FIELDS`
    order. `distance` is exactly the sum of this.

    Args:
        a: A signature returned by `signature`.
        b: A signature returned by `signature`.
        day: The season day the comparison is being made at.

    Returns:
        One non-negative float per field in `SIGNATURE_FIELDS`.
    """
    fraction = min(max(day / SEASON_DAYS, 0.0), 1.0)
    phase_weight = (
        PHASE_WEIGHT_EARLY + (PHASE_WEIGHT_LATE - PHASE_WEIGHT_EARLY) * fraction
    )
    composition_weight = (
        COMPOSITION_WEIGHT_EARLY
        + (COMPOSITION_WEIGHT_LATE - COMPOSITION_WEIGHT_EARLY) * fraction
    )
    weights = [
        phase_weight if name in _PHASE_FIELDS else composition_weight
        for name in SIGNATURE_FIELDS
    ]
    return tuple(w * abs(x - y) for w, x, y in zip(weights, a, b, strict=True))


def distance(a: tuple[float, ...], b: tuple[float, ...], day: int) -> float:
    """Return a weighted L1 distance between two signatures.

    Phase fields (``DAY``, ``HOUR``) and composition fields (everything else)
    are weighted separately, and the weighting shifts linearly over the
    season from ``*_WEIGHT_EARLY`` at day 0 to ``*_WEIGHT_LATE`` at
    ``SEASON_DAYS``. See the comment above ``PHASE_WEIGHT_EARLY`` for why the
    crossover point is a placeholder, not a measured value.

    Args:
        a: A signature returned by `signature`.
        b: A signature returned by `signature`.
        day: The season day the comparison is being made at -- the live
            board's day, not either signature's own ``DAY`` field, since `a`
            and `b` may come from different games recorded on different days.

    Returns:
        A non-negative float, 0.0 only when ``a == b``.
    """
    return sum(field_contributions(a, b, day))
