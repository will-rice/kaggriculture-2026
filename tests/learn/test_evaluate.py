"""Tests for the evaluator's incremental sale accounting.

The evaluator scores a season two observations at a time, because a
``Trajectory`` throws observations away and keeping 720 of them per environment
to score them once would not. That turns one call to ``sale_metrics`` into 719,
and an average of averages is not an average -- so the property that matters is
that the incremental totals equal the one-shot ones exactly, and that is what
these assert.
"""

from typing import Any

import pytest

from kaggriculture.learn.sales import buy_units, sale_metrics
from kaggriculture.learn.scripts.evaluate import BUY_UNITS_ALARM, Tally

PRICES = {"WHEAT": 25, "MELON": 250}


def _state(money: float, shed: dict[str, int]) -> dict[str, Any]:
    """Return one seat-0 observation with the given bank and shed."""
    return {
        "farms": [{"money": money}, {"money": 0.0}],
        "private": {"shed": shed},
        "market": {"prices": PRICES},
    }


# Uneven clears at uneven prices, so a mean of per-turn means would differ from
# the pooled mean and the equivalence below would have something to catch.
SERIES = [
    _state(1000.0, {"WHEAT": 20, "MELON": 4}),
    _state(1050.0, {"WHEAT": 16, "MELON": 4}),
    _state(1050.0, {"WHEAT": 16, "MELON": 4}),
    _state(1450.0, {"WHEAT": 16, "MELON": 2}),
    _state(1300.0, {"WHEAT": 24, "MELON": 2}),
    _state(1600.0, {"WHEAT": 4, "MELON": 2}),
]


def test_pairwise_tally_equals_scoring_the_whole_series() -> None:
    """Adding 5 consecutive pairs reads the same as one call over all 6 states."""
    tally = Tally()
    for before, after in zip(SERIES[:-1], SERIES[1:], strict=True):
        tally.add(before, after)
    record = tally.record(episodes=1)

    whole = sale_metrics(SERIES, seat=0)
    assert record["sales_per_episode"] == whole["sales"]
    assert record["mean_sale_price"] == pytest.approx(whole["mean_sale_price"])
    assert record["mean_market_price"] == pytest.approx(whole["mean_market_price"])
    assert record["realisation"] == pytest.approx(whole["price_realisation"])
    assert record["buy_units_per_episode"] == buy_units(SERIES, seat=0)


def test_the_sale_count_is_per_episode() -> None:
    """The same clears played twice read as the same rate, not twice the rate."""
    tally = Tally()
    for _ in range(2):
        for before, after in zip(SERIES[:-1], SERIES[1:], strict=True):
            tally.add(before, after)

    assert (
        tally.record(episodes=2)["sales_per_episode"]
        == sale_metrics(SERIES, seat=0)["sales"]
    )


def test_a_policy_that_never_sells_reads_zero_rather_than_dividing_by_zero() -> None:
    """No clears is a real outcome here, and it is the one hypothesis under test."""
    record = Tally().record(episodes=16)
    assert record == {
        "sales_per_episode": 0.0,
        "mean_sale_price": 0.0,
        "mean_market_price": 0.0,
        "realisation": 0.0,
        "buy_units_per_episode": 0.0,
    }


def test_the_buy_count_is_per_episode_and_below_the_alarm_here() -> None:
    """The tripwire reads a rate, so the same season twice is not twice the rate."""
    tally = Tally()
    for _ in range(2):
        for before, after in zip(SERIES[:-1], SERIES[1:], strict=True):
            tally.add(before, after)

    bought = tally.record(episodes=2)["buy_units_per_episode"]
    assert bought == buy_units(SERIES, seat=0)
    assert bought < BUY_UNITS_ALARM
