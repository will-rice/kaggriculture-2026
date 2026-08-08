"""Tests for banding the corpus by ladder rating and profiling the bands.

Three of these exist to stop a specific way the measurement could be quietly
wrong rather than to restate the code: that ``clears`` and ``sale_metrics``
disagree about what a sale is, that the town-drain transcription has drifted
from the engine, and that the rating attribution is assigning coin flips.
"""

from typing import Any

import pytest

from kaggriculture.constants import PRODUCTS
from kaggriculture.learn.bands import (
    attribute,
    band_of,
    clears,
    coverage,
    drain,
    impact,
    index,
    load,
    profiles,
    purchases,
)
from kaggriculture.learn.corpus import CORPUS
from kaggriculture.learn.sales import sale_metrics

ARCHIVE = CORPUS / "kaggriculture-episodes-2026-08-07.zip"
PRICES = {"WHEAT": 25, "MELON": 250}
real = pytest.mark.skipif(
    not ARCHIVE.exists(), reason="replay corpus not present on this machine"
)


def _state(money: float, shed: dict[str, int]) -> dict[str, Any]:
    """Return one seat-0 observation with the given bank and shed."""
    return {
        "farms": [{"money": money}, {"money": 0.0}],
        "private": {"shed": shed},
        "market": {"prices": PRICES},
    }


def test_clears_names_the_goods_that_sale_metrics_only_counts() -> None:
    """The per-good breakdown must be the aggregate's own rule, split up."""
    before = _state(1000.0, {"WHEAT": 10, "MELON": 2})
    after = _state(1750.0, {"WHEAT": 0, "MELON": 0})

    assert clears(before, after, seat=0) == {"WHEAT": 10, "MELON": 2}


def test_a_dropped_order_clears_nothing() -> None:
    """The bank did not move, so the engine filled nothing."""
    state = _state(1000.0, {"WHEAT": 10, "MELON": 0})

    assert clears(state, state, seat=0) == {}


def test_a_purchase_is_read_from_stock_rising_as_the_bank_falls() -> None:
    """The mirror of a clear, narrowed to goods the book prices."""
    before = _state(1000.0, {"WHEAT": 0, "MELON": 0})
    after = _state(750.0, {"WHEAT": 10, "MELON": 0})

    assert purchases(before, after, seat=0) == {"WHEAT": 10}


def test_a_harvest_is_not_a_purchase() -> None:
    """Stock appeared and the bank did not fall, so the field paid for it."""
    before = _state(1000.0, {"WHEAT": 0, "MELON": 0})
    after = _state(1000.0, {"WHEAT": 40, "MELON": 0})

    assert purchases(before, after, seat=0) == {}


def test_the_town_drains_the_book_on_its_own_schedule() -> None:
    """Transcribed from ``_town_consume``; a single-product shop takes double.

    Turn 0 is both a shop turn and a town-centre turn. The yarn store's only
    product is wool, so it takes two, and the centre takes one of everything it
    buys on top -- three wool in all. Wheat is not something the yarn store
    buys, so it sees the centre's one unit only.
    """
    removed = drain(0, ["YARN_STORE"], day=0)

    assert removed["WOOL"] == 3
    assert removed["WHEAT"] == 1


def test_nothing_drains_between_the_towns_intervals() -> None:
    """Turn 1 is neither a shop turn nor a centre turn."""
    assert drain(1, ["YARN_STORE"], day=0) == {}


def test_the_centre_buys_more_as_the_season_runs_out() -> None:
    """Its schedule steps up on day 10 and again on day 20."""
    assert drain(12, [], day=5)["WHEAT"] == 1
    assert drain(12, [], day=15)["WHEAT"] == 2
    assert drain(12, [], day=25)["WHEAT"] == 4


def test_impact_prices_what_one_seats_selling_took_from_the_other() -> None:
    """Two turns: seat 0 sells melon, then seat 1 sells melon into its wake.

    Melon's book punishes oversupply on a squared curve, so seat 0's fifty units
    are worth real money to seat 1. What matters is the sign and that the
    counterfactual is strictly the better price: with seat 0 out of the book,
    seat 1 sells into fifty fewer units of inventory and is paid more for every
    one of its own.
    """
    book: list[dict[str, Any]] = [
        {
            "farms": [{"money": 0.0}, {"money": 0.0}],
            "market": {"inventory": {**dict.fromkeys(PRODUCTS, 10_000), "MELON": held}},
        }
        for held in (10_000, 10_050, 10_060)
    ]
    sold = {0: [{"MELON": 50}, {}], 1: [{}, {"MELON": 10}]}

    theirs = impact(book, sold, sold, seat=0)
    ours = impact(book, sold, sold, seat=1)

    assert theirs["denied"] > 0.0
    assert 0.0 < theirs["denied_share"] < 1.0
    # Seat 1 sold after seat 0 and never before it, so it took nothing.
    assert ours["denied"] == 0.0


@real
def test_the_manifest_and_the_head_of_the_json_agree_on_the_episodes() -> None:
    """The cheap index has to cover the archive, or the bands are a subsample."""
    rows = index([ARCHIVE])

    assert len(rows) > 600
    assert all(len(row.teams) == 2 for row in rows)
    assert all(len(row.banks) == 2 for row in rows)
    assert all(row.low <= row.avg <= row.high for row in rows)


@real
def test_the_attributed_rating_orders_the_seats_better_than_a_coin_flip() -> None:
    """The check that the assignment is not noise.

    A random assignment would have the higher-rated seat out-bank the lower one
    in half of all episodes, and would not care how far apart the two ratings
    were. This asserts both halves: an overall edge, and a stronger edge on the
    episodes whose ratings are furthest apart, which is the part noise cannot
    fake.
    """
    rows = index([ARCHIVE])
    attribute(rows)

    close, far = [], []
    for row in rows:
        first, second = row.ratings
        one, two = row.banks
        if first == second or one == two:
            continue
        (far if abs(first - second) >= 100 else close).append(
            (first > second) == (one > two)
        )

    assert sum(close) / len(close) > 0.55
    assert sum(far) / len(far) > sum(close) / len(close)


@real
def test_every_band_is_a_within_day_slice_of_the_field() -> None:
    """Bands are percentiles, so their sizes follow the percentile widths."""
    rows = index([ARCHIVE])
    attribute(rows)

    assignment = band_of(rows)
    counts = dict.fromkeys(("top", "upper", "middle", "low"), 0)
    for band in assignment.values():
        counts[band] += 1

    assert counts["top"] < counts["upper"] < counts["middle"]
    assert counts["low"] > counts["top"]


@real
def test_summing_the_per_good_clears_reproduces_the_aggregate() -> None:
    """The guarantee that the two sale counters cannot drift apart.

    ``sale_metrics`` is the tested implementation and this module's per-good
    ``clears`` is the primitive underneath it. If a future edit to either one
    changed what counts as a completion, the two would disagree on a real
    episode long before anyone noticed the profile had moved.
    """
    rows = index([ARCHIVE])
    episode = load(ARCHIVE, rows[0].episode)

    for seat in (0, 1):
        observations = [step[seat]["observation"] for step in episode["steps"]]
        counted = [
            clears(before, after, seat)
            for before, after in zip(observations[:-1], observations[1:], strict=True)
        ]
        aggregate = sale_metrics(observations, seat)

        assert sum(len(turn) for turn in counted) == aggregate["sales"]
        assert sum(sum(turn.values()) for turn in counted) == aggregate["units"]


@real
def test_the_drain_transcription_matches_the_engine_on_quiet_turns() -> None:
    """On a turn where neither farm traded, the town is the only thing moving.

    This is what licenses the price-impact counterfactual. The book is additive
    in the two farms' trading only if the town's demand never depends on the
    inventory level, and the way to check that against the shipped engine
    without running it is to find the turns where the farms did nothing and
    confirm the book moved by exactly the transcribed amount.
    """
    rows = index([ARCHIVE])
    episode = load(ARCHIVE, rows[0].episode)
    steps = episode["steps"]
    views = {seat: [step[seat]["observation"] for step in steps] for seat in (0, 1)}

    checked = 0
    for turn in range(len(steps) - 1):
        state = views[0][turn]
        if any(
            views[seat][turn]["private"]["shed"]
            != views[seat][turn + 1]["private"]["shed"]
            or views[seat][turn]["farms"][seat]["money"]
            != views[seat][turn + 1]["farms"][seat]["money"]
            for seat in (0, 1)
        ):
            continue
        removed = drain(turn, state["town"]["unlocked_shops"], state["day"])
        for good in PRODUCTS:
            moved = (
                views[0][turn + 1]["market"]["inventory"][good]
                - state["market"]["inventory"][good]
            )
            assert moved == -removed.get(good, 0)
            checked += 1

    assert checked > 1000


@real
def test_a_profiled_seat_carries_a_season_and_a_price_impact() -> None:
    """One end-to-end pass, so a broken field is caught before a 480-episode run."""
    rows = index([ARCHIVE])
    attribute(rows)
    assignment = band_of(rows)
    row = next(row for row in rows if (row.day, row.episode, 0) in assignment)

    measured = profiles(load(ARCHIVE, row.episode), row, assignment)

    assert measured
    for profile in measured:
        assert len(profile.day_bank) == 30
        assert profile.day_tiles[-1] >= profile.day_tiles[0]
        assert profile.bank == row.banks[profile.seat]
        assert profile.denied >= 0.0
        assert 0.0 < profile.coverage <= 1.5


@real
def test_coverage_reports_how_much_trade_the_shed_inference_misses() -> None:
    """A completion read off the bank's sign cannot see a sale on a spending turn.

    The number matters because it bounds every count in the profile, so it is
    measured rather than assumed. It is well under 1.0, and this pins that fact
    in place so a future change that fixes it is noticed as a change.
    """
    rows = index([ARCHIVE])
    episode = load(ARCHIVE, rows[0].episode)
    steps = episode["steps"]
    views = {seat: [step[seat]["observation"] for step in steps] for seat in (0, 1)}
    net = {
        seat: [
            {
                good: clears(before, after, seat).get(good, 0)
                - purchases(before, after, seat).get(good, 0)
                for good in PRODUCTS
            }
            for before, after in zip(view[:-1], view[1:], strict=True)
        ]
        for seat, view in views.items()
    }

    assert 0.3 < coverage(views[0], net) < 1.0
