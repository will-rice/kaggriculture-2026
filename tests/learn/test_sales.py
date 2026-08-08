"""Tests for completed-sale measurement.

Every expectation is a bare literal. A sales counter that silently counted
emitted orders instead of engine completions would send the next reward design
in exactly the wrong direction, and it would look perfectly healthy while doing
it, so these assert the arithmetic rather than restating it.
"""

from typing import Any

import pytest

from kaggriculture.learn.sales import sale_metrics

PRICES = {"WHEAT": 25, "MELON": 250}


def _state(money: float, shed: dict[str, int]) -> dict[str, Any]:
    """Return one seat-0 observation with the given bank and shed."""
    return {
        "farms": [{"money": money}, {"money": 0.0}],
        "private": {"shed": shed},
        "market": {"prices": PRICES},
    }


def test_one_clean_sale_at_market_price() -> None:
    """Ten wheat at 25 is 250 coins, realisation exactly par.

    sales=1, units=10, mean_sale_price = 250/10 = 25, market = 25, ratio = 1.0.
    """
    series = [
        _state(1000.0, {"WHEAT": 10, "MELON": 0}),
        _state(1250.0, {"WHEAT": 0, "MELON": 0}),
    ]
    metrics = sale_metrics(series, seat=0)
    assert metrics["sales"] == 1.0
    assert metrics["units"] == 10.0
    assert metrics["mean_sale_price"] == pytest.approx(25.0)
    assert metrics["mean_market_price"] == pytest.approx(25.0)
    assert metrics["price_realisation"] == pytest.approx(1.0)


def test_selling_into_a_depressed_book_reads_below_par() -> None:
    """Ten wheat for 150 coins realises 15 against a 25 book: ratio 0.6.

    This is the signature that separates "sells too rarely" from "sells badly",
    and it is the number the next reward is designed around.
    """
    series = [
        _state(1000.0, {"WHEAT": 10, "MELON": 0}),
        _state(1150.0, {"WHEAT": 0, "MELON": 0}),
    ]
    metrics = sale_metrics(series, seat=0)
    assert metrics["mean_sale_price"] == pytest.approx(15.0)
    assert metrics["price_realisation"] == pytest.approx(0.6)


def test_an_emitted_order_that_never_cleared_is_not_a_sale() -> None:
    """Stock unchanged and bank unchanged: the engine dropped it.

    The trap this whole module exists to avoid. An order-counting version would
    report a sale here.
    """
    series = [
        _state(1000.0, {"WHEAT": 10, "MELON": 0}),
        _state(1000.0, {"WHEAT": 10, "MELON": 0}),
    ]
    assert sale_metrics(series, seat=0)["sales"] == 0.0


def test_buying_is_not_a_sale() -> None:
    """Stock rises and the bank falls; nothing was sold."""
    series = [
        _state(1000.0, {"WHEAT": 0, "MELON": 0}),
        _state(750.0, {"WHEAT": 10, "MELON": 0}),
    ]
    metrics = sale_metrics(series, seat=0)
    assert metrics["sales"] == 0.0
    assert metrics["units"] == 0.0


def test_two_goods_clearing_together_count_as_two_sales() -> None:
    """Ten wheat and two melon for 750 coins.

    units = 12, mean_sale_price = 750/12 = 62.5. The market reference is
    volume-weighted: (10*25 + 2*250)/12 = 62.5, so realisation is par -- an
    unweighted mean would read (25+250)/2 = 137.5 and call this a 0.45 disaster.
    """
    series = [
        _state(1000.0, {"WHEAT": 10, "MELON": 2}),
        _state(1750.0, {"WHEAT": 0, "MELON": 0}),
    ]
    metrics = sale_metrics(series, seat=0)
    assert metrics["sales"] == 2.0
    assert metrics["units"] == 12.0
    assert metrics["mean_sale_price"] == pytest.approx(62.5)
    assert metrics["mean_market_price"] == pytest.approx(62.5)
    assert metrics["price_realisation"] == pytest.approx(1.0)


def test_sales_are_counted_across_turns() -> None:
    """Two separate clears on two turns are two sales, not one."""
    series = [
        _state(1000.0, {"WHEAT": 10, "MELON": 0}),
        _state(1125.0, {"WHEAT": 5, "MELON": 0}),
        _state(1250.0, {"WHEAT": 0, "MELON": 0}),
    ]
    metrics = sale_metrics(series, seat=0)
    assert metrics["sales"] == 2.0
    assert metrics["units"] == 10.0
    assert metrics["mean_sale_price"] == pytest.approx(25.0)
