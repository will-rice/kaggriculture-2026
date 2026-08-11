"""Exact parity tests for tensor market pricing."""
# ruff: noqa: D103

import torch

from kaggriculture.constants import MARKET_PARAMS, market_price
from kaggriculture.sim.pricing import market_prices
from kaggriculture.sim.state import PRODUCT_NAMES


def test_market_prices_match_reference_exhaustively() -> None:
    inventories = []
    expected = []
    for product in PRODUCT_NAMES:
        params = MARKET_PARAMS[product]
        values = list(
            range(
                int(params["I0"]) - 4 * int(params["T"]),
                int(params["I0"]) + 8 * int(params["T"]) + 1,
            )
        )
        inventories.append(values)
        expected.append([market_price(product, value) for value in values])

    # Products have different sweep lengths, so exercise each independently.
    for product_index, (values, wanted) in enumerate(
        zip(inventories, expected, strict=True)
    ):
        inventory = torch.tensor(values, dtype=torch.int64)
        actual = market_prices(inventory, product_index)
        assert actual.tolist() == wanted


def test_market_prices_preserve_batch_shape() -> None:
    inventory = torch.full((2, 3), 10_000, dtype=torch.int64)

    actual = market_prices(inventory, PRODUCT_NAMES.index("WHEAT"))

    assert actual.shape == (2, 3)
    assert actual.tolist() == [[25, 25, 25], [25, 25, 25]]
