"""Pure action vocabulary and decoding shared by training and submissions."""

from dataclasses import dataclass
from typing import Any

from kaggriculture.constants import ANIMALS, CROPS, MAX_MARKET_ORDERS_PER_TURN, PRODUCTS

CROP_NAMES = sorted(CROPS)
SHED_NAMES = sorted(set(PRODUCTS) | set(ANIMALS))
ANIMAL_NAMES = sorted(ANIMALS)
PRODUCT_NAMES = sorted(PRODUCTS)

UNIT_OPS: tuple[str, ...] = (
    (
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
        "DROP",
    )
    + tuple(f"PLANT:{crop}" for crop in CROP_NAMES)
    + tuple(f"PICKUP:{item}" for item in SHED_NAMES)
    + tuple(f"PLACE:{item}" for item in SHED_NAMES)
)

ITEM_VERBS = frozenset({"PLANT", "PICKUP", "PLACE"})
TRANSFER_OPS: tuple[bool, ...] = tuple(
    name.startswith(("PICKUP:", "PLACE:")) for name in UNIT_OPS
)
MAX_TRANSFER = 12

_BUY_PRODUCT_ITEMS = ("FERTILIZER", "WHEAT")
MARKET_SLOTS: tuple[tuple[str, str], ...] = (
    tuple(("SELL", item) for item in PRODUCT_NAMES)
    + tuple(("BUY_SEED", item) for item in CROP_NAMES)
    + tuple(("BUY_PRODUCT", item) for item in _BUY_PRODUCT_ITEMS)
    + tuple(("BUY_ANIMAL", item) for item in ANIMAL_NAMES)
)
HIRE_SLOT = len(MARKET_SLOTS)
LAND_SLOT = len(MARKET_SLOTS) + 1
MAX_ORDERS = MAX_MARKET_ORDERS_PER_TURN

QUANTITIES: tuple[int, ...] = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    16,
    24,
    40,
    64,
    80,
    100,
    130,
    165,
)

_TAIL_BUCKET_MAX = (20, 32, 52, 64, 80, 100, 130)

if len(_TAIL_BUCKET_MAX) != len(QUANTITIES) - 14:
    raise AssertionError(
        f"len(_TAIL_BUCKET_MAX) ({len(_TAIL_BUCKET_MAX)}) != len(QUANTITIES) "
        f"- 14 ({len(QUANTITIES) - 14})"
    )

if MAX_TRANSFER not in QUANTITIES:
    raise AssertionError(
        f"MAX_TRANSFER ({MAX_TRANSFER}) is not a value in QUANTITIES: {QUANTITIES}"
    )


@dataclass(frozen=True)
class SelectedActions:
    """The selected vocabulary rows before conversion to engine action lists."""

    unit_indices: tuple[int, ...]
    quantity_indices: tuple[int, ...]
    market_indices: tuple[int, ...]


def bucket_of(n: int) -> int:
    """Return the bucket index that represents non-negative quantity ``n``."""
    if n <= 12:
        return n
    for offset, upper in enumerate(_TAIL_BUCKET_MAX):
        if n <= upper:
            return 13 + offset
    return len(QUANTITIES) - 1


def quantity_of(bucket: int) -> int:
    """Return the representative quantity for a vocabulary bucket index."""
    return QUANTITIES[bucket]


def operation_of(label: int, quantity: int) -> list[Any]:
    """Convert a unit vocabulary label and quantity into an engine operation."""
    verb, _, item = UNIT_OPS[label].partition(":")
    if not item:
        return [verb]
    if verb == "PLANT":
        return [verb, item]
    return [verb, item, quantity]


def market_orders_of(buckets: tuple[int, ...]) -> list[list[Any]]:
    """Convert selected market quantity buckets into ordered engine orders."""
    market_rows = len(MARKET_SLOTS) + 2
    if len(buckets) != market_rows:
        raise ValueError(f"expected {market_rows} market selections")
    orders: list[list[Any]] = []
    for slot, (verb, item) in enumerate(MARKET_SLOTS):
        bucket = buckets[slot]
        if bucket != 0:
            orders.append([verb, item, quantity_of(bucket)])
    orders.extend([["HIRE"]] * quantity_of(buckets[HIRE_SLOT]))
    if buckets[LAND_SLOT] != 0:
        orders.append(["BUY_LAND"])
    return orders[:MAX_ORDERS]


def decode_selected(selected: SelectedActions) -> dict[str, Any]:
    """Convert pure selected indices into the engine action dictionary."""
    if len(selected.unit_indices) != len(selected.quantity_indices):
        raise ValueError("one quantity index is required per unit index")
    market_rows = len(MARKET_SLOTS) + 2
    if len(selected.market_indices) != market_rows:
        raise ValueError(f"expected {market_rows} market selections")
    operations = tuple(
        operation_of(op_index, quantity_of(quantity_index))
        for op_index, quantity_index in zip(
            selected.unit_indices, selected.quantity_indices, strict=True
        )
    )
    market = market_orders_of(selected.market_indices)
    return {
        "farmer": operations[0] if operations else ["PASS"],
        "hands": list(operations[1:]),
        "market": market,
    }


def safe_pass_action() -> dict[str, Any]:
    """Return the submission-safe action that cannot spend or move anything."""
    return {"farmer": ["PASS"], "hands": [], "market": []}
