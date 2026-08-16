"""A route becomes tensors once, not once per batch row."""

import torch

from kaggriculture.learn.encoding import MAX_UNITS, UNIT_OPS
from kaggriculture.search.encode import encode_route
from kaggriculture.sim.state import PRODUCT_NAMES


def test_encode_route_produces_one_row_per_turn() -> None:
    """Shapes and dtypes must match what `step` consumes."""
    route = [{"farmer": ["PASS"], "hands": [], "market": []}] * 720
    route[5] = {"farmer": ["WATER"], "hands": [], "market": [["SELL", "WHEAT", 4]]}

    encoded = encode_route(route, device="cpu")

    assert encoded.units.shape == (720, MAX_UNITS)
    assert encoded.units.dtype == torch.int16
    assert encoded.order_type.shape == (720, 10)
    assert encoded.order_type.dtype == torch.int8
    assert encoded.order_item.dtype == torch.int8
    assert encoded.order_qty.dtype == torch.int32
    assert encoded.units[5, 0] == UNIT_OPS.index("WATER")
    assert encoded.order_type[5, 0] == 1
    assert encoded.order_item[5, 0] == PRODUCT_NAMES.index("WHEAT")
    assert encoded.order_qty[5, 0] == 4
    # Sentinel values must survive int8 narrowing for unused slots.
    assert encoded.order_type[4, 0] == 0
    assert encoded.order_item[4, 0] == -1
    assert encoded.order_item[4, 0].dtype == torch.int8
    # Unused slot after the single SELL order.
    assert encoded.order_item[5, 1] == -1
    assert encoded.order_item[5, 1].dtype == torch.int8
