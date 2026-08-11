"""Immutable rules-derived tables, materialized once per device.

Static Python loops build these tables. The step path only gathers from them,
which is what keeps the hot path free of runtime scans over products, crops,
animals, shed items, or action types.
"""

from functools import lru_cache

import torch

from kaggriculture.constants import ANIMALS, BOARD_SIZE, CROPS, MOVES
from kaggriculture.learn.encoding import UNIT_OPS
from kaggriculture.sim.engine import (
    ORDER_BUY_ANIMAL,
    ORDER_BUY_PRODUCT,
    ORDER_BUY_SEED,
    ORDER_MALFORMED,
    ORDER_SELL,
)
from kaggriculture.sim.state import (
    ANIMAL_NAMES,
    CROP_NAMES,
    PRODUCT_NAMES,
    SHED_NAMES,
)
from kaggriculture.sim.tensors import tensor_constant

# The two products the reference lets a farm buy back from the market.
BUY_PRODUCTS = ("WHEAT", "FERTILIZER")

# Resolver classes. Every recognized order maps onto exactly one of them, and
# an unrecognized type or item maps onto ``ROLE_NONE`` and is a dead order.
ROLE_NONE = 0
ROLE_SELL = 1
ROLE_BUY_SEED = 2
ROLE_BUY_PRODUCT = 3
ROLE_BUY_ANIMAL = 4

# Fields of one market rule row.
ORDER_ROLE = 0
ORDER_PRODUCT = 1
ORDER_SHED = 2
ORDER_SEED = 3
ORDER_COST = 4

ORDER_TYPES = ORDER_MALFORMED + 1
# Item codes run from -1 (absent) to the largest catalogue index, offset by one
# so the whole range indexes the table directly.
ORDER_ITEMS = len(PRODUCT_NAMES) + 1
_ORDER_FIELDS = 5


def _market_rows() -> list[list[list[int]]]:
    rows = [
        [[0] * _ORDER_FIELDS for _ in range(ORDER_ITEMS)] for _ in range(ORDER_TYPES)
    ]
    for product, name in enumerate(PRODUCT_NAMES):
        rows[ORDER_SELL][product + 1] = [
            ROLE_SELL,
            product,
            SHED_NAMES.index(name),
            0,
            0,
        ]
    for crop, name in enumerate(CROP_NAMES):
        rows[ORDER_BUY_SEED][crop + 1] = [
            ROLE_BUY_SEED,
            0,
            0,
            crop,
            int(CROPS[name]["seed"]),
        ]
    for choice, name in enumerate(BUY_PRODUCTS):
        rows[ORDER_BUY_PRODUCT][choice + 1] = [
            ROLE_BUY_PRODUCT,
            PRODUCT_NAMES.index(name),
            SHED_NAMES.index(name),
            0,
            0,
        ]
    for animal, name in enumerate(ANIMAL_NAMES):
        rows[ORDER_BUY_ANIMAL][animal + 1] = [
            ROLE_BUY_ANIMAL,
            0,
            SHED_NAMES.index(name),
            0,
            int(ANIMALS[name]["cost"]),
        ]
    return rows


@lru_cache(maxsize=None)
def market_orders(device: torch.device) -> torch.Tensor:
    """Return the market rule table indexed by order type and item code.

    Args:
        device: Device the table must live on.

    Returns:
        An int64 ``(type, item + 1, field)`` tensor whose fields are the
        ``ORDER_*`` columns.
    """
    return tensor_constant(_market_rows(), dtype=torch.int64, device=device)


@lru_cache(maxsize=None)
def land_quadrants(device: torch.device) -> torch.Tensor:
    """Return the tile mask each land purchase unlocks, in purchase order."""
    size = BOARD_SIZE
    half = size // 2
    rows = [
        [
            [
                int((x >= half) and (y < half)),
                int((x < half) and (y >= half)),
                int((x >= half) and (y >= half)),
            ]
            for x in range(size)
        ]
        for y in range(size)
    ]
    plane = tensor_constant(rows, dtype=torch.bool, device=device)
    return plane.permute(2, 0, 1).contiguous()


# Columns of the per-operation rule row.
UNIT_MOVE_X = 0
UNIT_MOVE_Y = 1
UNIT_MOVING = 2
UNIT_PICKUP = 3
UNIT_PLACE = 4
UNIT_PLACE_KIND = 5
UNIT_PLACE_OCCUPANT = 6
UNIT_PLANT = 7
_UNIT_FIELDS = 8

# Columns of the per-crop rule row.
CROP_ONGOING = 0
CROP_INITIAL_YIELD = 1
CROP_MAX_YIELD = 2
CROP_MAX_YIELD_DAY = 3
CROP_FIRST_YIELD_DAY = 4
CROP_WATER_WINDOW = 5
CROP_SHED = 6
_CROP_FIELDS = 7

# The tile kind an animal's structure occupies, as ``TILE_KINDS`` indexes it.
_STRUCTURE_KIND = {"COOP": 4, "PASTURE": 5}


def _unit_rows() -> list[list[int]]:
    rows = []
    for name in UNIT_OPS:
        row = [0, 0, 0, -1, -1, -1, 0, -1]
        if name in MOVES:
            row[UNIT_MOVE_X], row[UNIT_MOVE_Y] = MOVES[name]
            row[UNIT_MOVING] = 1
        item = name.partition(":")[2]
        if name.startswith("PICKUP:"):
            row[UNIT_PICKUP] = SHED_NAMES.index(item)
        if name.startswith("PLACE:"):
            row[UNIT_PLACE] = SHED_NAMES.index(item)
            if item in ANIMAL_NAMES:
                row[UNIT_PLACE_KIND] = _STRUCTURE_KIND[str(ANIMALS[item]["structure"])]
                row[UNIT_PLACE_OCCUPANT] = ANIMAL_NAMES.index(item) + 1
        if name.startswith("PLANT:"):
            row[UNIT_PLANT] = CROP_NAMES.index(item)
        rows.append(row)
    return rows


@lru_cache(maxsize=None)
def unit_ops(device: torch.device) -> torch.Tensor:
    """Return the ``(operation, field)`` unit rule table for a device."""
    return tensor_constant(_unit_rows(), dtype=torch.int64, device=device)


def _crop_rows() -> list[list[int]]:
    rows = []
    for name in CROP_NAMES:
        data = CROPS[name]
        ongoing = bool(data["ongoing"])
        rows.append(
            [
                int(ongoing),
                0 if ongoing else 1,
                int(data["max_yield"]),
                int(data["max_yield_day"]),
                int(data["first_yield_day"]),
                (int(data["max_yield_day"]) + 1) // 2,
                SHED_NAMES.index(name),
            ]
        )
    return rows


@lru_cache(maxsize=None)
def crop_rules(device: torch.device) -> torch.Tensor:
    """Return the ``(crop, field)`` growth rule table for a device."""
    return tensor_constant(_crop_rows(), dtype=torch.int64, device=device)


@lru_cache(maxsize=None)
def animal_produce(device: torch.device) -> torch.Tensor:
    """Return each animal's harvested shed item, indexed by animal."""
    return tensor_constant(
        [SHED_NAMES.index(str(ANIMALS[name]["product"])) for name in ANIMAL_NAMES],
        dtype=torch.int64,
        device=device,
    )
