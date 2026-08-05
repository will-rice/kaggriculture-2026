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

# Per farm, in order: one plane per crop, one per animal, four mutually
# exclusive tile states, then five continuous per-tile features. A crop or
# animal lights its own plane inside a PLANT / occupied-structure tile; the
# state planes cover everything else, including a bare (unoccupied) structure.
_TILE_STATES = ("WEED", "LOCKED", "EMPTY", "STRUCTURE")
_TILE_FEATURES = ("ACTIVE_TODAY", "DISTRESS", "YIELD_FRACTION", "BONUS_READY", "AGE")

_ANIMAL_BASE = len(CROP_NAMES)
_STATE_BASE = _ANIMAL_BASE + len(ANIMAL_NAMES)
_FEATURE_BASE = _STATE_BASE + len(_TILE_STATES)
_PER_FARM_PLANES = _FEATURE_BASE + len(_TILE_FEATURES)

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


def encode_scalars(observation: Mapping[str, Any], seat: int) -> torch.Tensor:
    """Return the non-spatial features for one seat.

    Both a price and an inventory signal are kept per product: the price says
    what the next unit fetches right now, and the inventory says how far
    already-committed supply -- ours or the opponent's -- has pushed it from
    baseline, which is what decides where the price goes next. Land, town
    shops and hired hands are included for both seats, since the opponent's
    farm is public.

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
        observation["step"] / EPISODE_STEPS,
        len(observation["town"]["unlocked_shops"]) / len(SHOPS),
        len(ours["unlocked_quadrants"]) / _MAX_QUADRANTS,
        len(theirs["unlocked_quadrants"]) / _MAX_QUADRANTS,
        len(ours["hands"]) / 8.0,
        len(theirs["hands"]) / 8.0,
    ]
    return torch.tensor(values, dtype=torch.float32).reshape(1, SCALARS)
