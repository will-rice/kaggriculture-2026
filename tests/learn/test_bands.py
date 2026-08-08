"""Tests for banding the corpus by ladder rating and profiling the bands.

Three of these exist to stop a specific way the measurement could be quietly
wrong rather than to restate the code: that ``clears`` and ``sale_metrics``
disagree about what a sale is, that the town-drain transcription has drifted
from the engine, and that the rating attribution is assigning coin flips.
"""

import json
import zipfile
from typing import Any

import pytest

from kaggriculture.constants import (
    FLAT_TOWN_CENTER_SELL_INTERVAL as FLAT,
)
from kaggriculture.constants import (
    LEGACY_TOWN_CENTER_SELL_INTERVAL as LEGACY,
)
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
    removed = drain(0, ["YARN_STORE"], day=0, center_interval=LEGACY)

    assert removed["WOOL"] == 3
    assert removed["WHEAT"] == 1


def test_nothing_drains_between_the_towns_intervals() -> None:
    """Turn 1 is neither a shop turn nor a centre turn."""
    assert drain(1, ["YARN_STORE"], day=0, center_interval=LEGACY) == {}


def test_a_legacy_episode_gets_the_escalating_centre_it_was_played_under() -> None:
    """The pre-1.32.6 schedule stepped up on day 10 and again on day 20.

    Deliberately *not* the installed engine's behaviour. Eight of the nine
    archives on disk, and 404 of the 675 episodes in the ninth, report
    ``townCenterSellInterval`` 12 and were played under the escalating
    schedule. Asserting the flat rate for them would make ``bands`` wrong about
    most of the data it reads.
    """
    assert drain(12, [], day=5, center_interval=LEGACY)["WHEAT"] == 1
    assert drain(12, [], day=15, center_interval=LEGACY)["WHEAT"] == 2
    assert drain(12, [], day=25, center_interval=LEGACY)["WHEAT"] == 4


def test_a_flat_rate_episode_gets_one_unit_whatever_the_day() -> None:
    """The other side of the same corpus: 271 episodes dated 2026-08-07.

    Same function, same days, and the answer has to differ -- this is what a
    single global constant got wrong, and what makes ``center_interval`` a
    required argument rather than a default.
    """
    for day in (5, 15, 25):
        assert drain(24, [], day=day, center_interval=FLAT)["WHEAT"] == 1

    # Turn 12 is a centre tick for a legacy episode and a quiet turn for a
    # flat-rate one. The two builds disagree about the same turn index, which is
    # exactly why the interval cannot be read from the installed engine.
    assert drain(12, [], day=25, center_interval=LEGACY)["WHEAT"] == 4
    assert drain(12, [], day=25, center_interval=FLAT) == {}


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


def _episode_on(interval: int) -> dict[str, Any]:
    """Return the first archived episode played at a given centre interval.

    The build is read from the head of each episode's JSON rather than by
    decoding it. ``configuration`` is the first key the exporter writes, and
    the flat-rate episodes are not at the front of the archive, so selecting
    them by decoding would mean parsing tens of 32 MB documents to find one.

    Args:
        interval: The ``townCenterSellInterval`` to look for.

    Returns:
        The decoded replay.

    Raises:
        AssertionError: If the archive holds no episode on that build.
    """
    decoder = json.JSONDecoder()
    marker = '"townCenterSellInterval": '
    with zipfile.ZipFile(ARCHIVE) as bundle:
        for name in bundle.namelist():
            if not name.endswith(".json"):
                continue
            with bundle.open(name) as member:
                head = member.read(2048).decode("utf-8", "ignore")
            value, _ = decoder.raw_decode(head, head.find(marker) + len(marker))
            if int(value) == interval:
                return load(ARCHIVE, int(name.removesuffix(".json")))
    raise AssertionError(f"no episode at townCenterSellInterval {interval}")


@real
@pytest.mark.parametrize("interval", [LEGACY, FLAT])
def test_the_drain_transcription_matches_the_build_that_played_the_episode(
    interval: int,
) -> None:
    """On a turn where neither farm traded, the town is the only thing moving.

    This is what licenses the price-impact counterfactual. The book is additive
    in the two farms' trading only if the town's demand never depends on the
    inventory level, and the way to check that without running an engine is to
    find the turns where the farms did nothing and confirm the book moved by
    exactly the transcribed amount.

    Checked against the engine *in the replay*, not the one installed, and
    checked once per build because the 2026-08-07 archive holds both: 404
    episodes at ``townCenterSellInterval`` 12 and 271 at 24. A single-episode
    version of this test passed while ``drain`` modelled one build globally,
    which is how the flat-rate quarter of the archive went unnoticed.
    """
    episode = _episode_on(interval)
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
        removed = drain(turn, state["town"]["unlocked_shops"], state["day"], interval)
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

    interval = int(episode["configuration"]["townCenterSellInterval"])

    assert 0.3 < coverage(views[0], net, interval) < 1.0
