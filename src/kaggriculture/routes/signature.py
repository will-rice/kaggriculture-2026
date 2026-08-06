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
"""

from typing import Any, Mapping

from kaggriculture.constants import ANIMALS, CROPS, PRODUCTS, SEASON_DAYS

CROP_NAMES = tuple(sorted(CROPS))
ANIMAL_NAMES = tuple(sorted(ANIMALS))
PRODUCT_NAMES = tuple(sorted(PRODUCTS))

# One entry per value `signature` fills, in the exact order it fills them:
# tile composition (what is growing or built), the farm's operating state,
# the season's phase, then what is banked in the shed waiting to sell.
SIGNATURE_FIELDS: tuple[str, ...] = (
    tuple(f"CROP:{crop}" for crop in CROP_NAMES)
    + tuple(f"ANIMAL:{animal}" for animal in ANIMAL_NAMES)
    + ("WEEDS", "LOCKED", "MONEY", "HANDS", "HIRES_TODAY", "UNLOCKED_QUADRANTS")
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

    Args:
        observation: One turn's observation, as handed to the agent.
        seat: Which player's farm to describe.

    Returns:
        A fixed-width tuple of floats, ordered as ``SIGNATURE_FIELDS``.
    """
    farm = observation["farms"][seat]
    crop_counts = dict.fromkeys(CROP_NAMES, 0)
    animal_counts = dict.fromkeys(ANIMAL_NAMES, 0)
    weeds = 0
    locked = 0
    for row in farm["tiles"]:
        for tile in row:
            if tile is None:
                continue
            if tile == "LOCKED":
                locked += 1
            elif tile["kind"] == "WEED":
                weeds += 1
            elif tile["kind"] == "PLANT":
                crop_counts[tile["crop"]] += 1
            elif tile.get("animal") is not None:
                animal_counts[tile["animal"]] += 1
    shed = observation["private"]["shed"]

    values = [float(crop_counts[crop]) for crop in CROP_NAMES]
    values += [float(animal_counts[animal]) for animal in ANIMAL_NAMES]
    values += [
        float(weeds),
        float(locked),
        float(farm["money"]),
        float(len(farm["hands"])),
        float(farm["hires_today"]),
        float(len(farm["unlocked_quadrants"])),
        float(observation["day"]),
        float(observation["hour"]),
    ]
    values += [float(shed[product]) for product in PRODUCT_NAMES]
    return tuple(values)


# Distance weights for the two field groups, at the season's start and its
# end. Early routes are phase-locked -- every good route waters at the same
# hour on day 0, so an hour apart should cost more than a crop apart -- while
# late in the season routes have diverged by what they built rather than when
# they built it, so the ranking flips. These four numbers are set only to get
# that ranking right at the two ends this module's tests check; the day the
# crossover actually happens is unmeasured. Task 5 sweeps it against real
# route outcomes -- treat the linear interpolation below as a placeholder
# shape, not a fitted one.
PHASE_WEIGHT_EARLY = 8.0
PHASE_WEIGHT_LATE = 0.5
COMPOSITION_WEIGHT_EARLY = 0.5
COMPOSITION_WEIGHT_LATE = 8.0


def distance(a: tuple[float, ...], b: tuple[float, ...], day: int) -> float:
    """Return a weighted L1 distance between two signatures.

    Phase fields (``DAY``, ``HOUR``) and composition fields (everything else)
    are weighted separately, and the weighting shifts linearly over the
    season from ``*_WEIGHT_EARLY`` at day 0 to ``*_WEIGHT_LATE`` at
    ``SEASON_DAYS``. See the module-level comment above ``PHASE_WEIGHT_EARLY``
    for why the crossover point is a placeholder, not a measured value.

    Args:
        a: A signature returned by `signature`.
        b: A signature returned by `signature`.
        day: The season day the comparison is being made at -- the live
            board's day, not either signature's own ``DAY`` field, since `a`
            and `b` may come from different games recorded on different days.

    Returns:
        A non-negative float, 0.0 only when ``a == b``.
    """
    fraction = min(max(day / SEASON_DAYS, 0.0), 1.0)
    phase_weight = (
        PHASE_WEIGHT_EARLY + (PHASE_WEIGHT_LATE - PHASE_WEIGHT_EARLY) * fraction
    )
    composition_weight = (
        COMPOSITION_WEIGHT_EARLY
        + (COMPOSITION_WEIGHT_LATE - COMPOSITION_WEIGHT_EARLY) * fraction
    )

    total = sum(phase_weight * abs(a[i] - b[i]) for i in _PHASE_INDICES)
    total += sum(composition_weight * abs(a[i] - b[i]) for i in _COMPOSITION_INDICES)
    return total
