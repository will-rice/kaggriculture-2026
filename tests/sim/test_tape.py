"""Recorded top episodes replay through the simulator to their exact banks.

This is the differential proof of the quantity lane -- both lanes together now
that the market lane's axis has been widened to cover ordinary top play (see
``sim.tape``'s module docstring). A recorded top episode exercises bulk
PICKUP/PLACE on most of its transfers and market orders past the market
lane's old 64-unit cap on most of its turns, branches the RL-action-space
differential tests structurally never reach, and the engine is deterministic
given seed and both seats' actions, so equality is exact or the lane is
wrong. Do not weaken to a tolerance; a mismatch is a bug with a turn number,
found by comparing money per turn.

Selection is by rating floor and engine version, sampled with a fixed seed
across the whole eligible population -- not sorted by rating and truncated to
the top handful. Fix round 1 found why that distinction matters: an
independent scan found 4.75% of episodes carry a ``SELL X 999`` "sell
everything" idiom, spread across the ranking rather than concentrated near
the very top, and the previous always-the-8-strongest selection drew from
exactly the slice of the ranking that happened not to contain one. A fixed
seed keeps the sample reproducible without keeping it biased toward the head
of the listing. A market order naming a quantity past even the widened axis
that a shed-bound clamp cannot safely absorb (an out-of-domain BUY_SEED
request; see ``sim.rollout.SHED_BOUND_VERBS``) is still outside the
simulator's domain, and ``encode_turn`` still raises on it. If that happens
to land inside the sample this test selects, the test below fails loudly,
naming the episode and the reason -- it does not skip the episode and
quietly report on a smaller, luckier sample.

``test_a_sell_everything_regression_episode_replays_to_the_exact_bank`` pins
the specific failure class fix round 1 found: rather than hope a random
sample happens to include a ``SELL X 999``-style episode, it searches the
archive for one and asserts it replays exactly, so this class of bug cannot
hide behind an unlucky draw again.
"""
# ruff: noqa: D103

import importlib.metadata
import json
import logging
import random
import zipfile
from pathlib import Path

import pytest
import torch

from kaggriculture.learn.corpus import CORPUS, read_manifest
from kaggriculture.search.route import Route, from_episode
from kaggriculture.sim.market import QUANTITY_AXIS
from kaggriculture.sim.rollout import SHED_BOUND_VERBS
from kaggriculture.sim.tape import replay

LOGGER = logging.getLogger(__name__)

ENGINE = importlib.metadata.version("kaggle-environments")
ARCHIVES = (
    sorted(CORPUS.glob("kaggriculture-episodes-*.zip")) if CORPUS.exists() else []
)
ARCHIVE = ARCHIVES[-1] if ARCHIVES else None

pytestmark = pytest.mark.skipif(
    ARCHIVE is None, reason="replay corpus not present on this machine"
)

# The floor Step 6 of the market-lane task set: enough episodes that this is a
# real sample of top play, not three that happened to be convenient.
EPISODE_COUNT = 8

# ``learn.corpus.select``'s own floor: a chosen episode had two strong players
# in it, not one strong player against a pushover.
MIN_RATING = 2500.0

# Fixed so the sample is the same episodes on every run -- reproducible, but
# (via the shuffle in ``_top_episodes``) not biased toward the very head of
# the ranking the way a plain sort-and-truncate is.
RANDOM_SEED = 20260821


def _top_episodes(archive: Path, count: int) -> list[dict]:
    """Return ``count`` ``ENGINE``-version episodes, sampled across all strong play.

    "Strong" is ``min_score >= MIN_RATING``, the weaker seat's rating -- the
    same column ``learn.corpus.select`` ranks on -- so a chosen episode had
    two strong players in it, not one strong player against a pushover.
    Sampling is a fixed-seed shuffle of every eligible row, not a sort by
    score truncated to the top ``count``: fix round 1's whole finding was
    that a market idiom real top play uses is spread across the ranking, not
    concentrated at its very top, so always drawing the single highest-rated
    handful can miss it forever no matter how many times the test runs.

    No cap-compliance filtering: unlike the market lane's pre-widening state,
    an ordinary strong-rated episode is now expected to stay inside the
    simulator's domain, and a selection that filtered on it anyway would risk
    quietly narrowing the sample back down without saying so. If a selected
    episode still cannot replay, the tests below say so loudly instead.

    Args:
        archive: Daily archive to select from.
        count: How many episodes to return.

    Returns:
        ``count`` engine-matched, ``MIN_RATING``-and-above episodes, in a
        reproducible but not rating-sorted order.
    """
    eligible = [row for row in read_manifest(archive) if row.min_score >= MIN_RATING]
    random.Random(RANDOM_SEED).shuffle(eligible)
    episodes: list[dict] = []
    with zipfile.ZipFile(archive) as bundle:
        for row in eligible:
            episode = json.loads(bundle.read(f"{row.episode_id}.json"))
            if episode.get("module_version") != ENGINE:
                continue
            episodes.append(episode)
            if len(episodes) >= count:
                break
    return episodes


def _names_a_sell_everything_order(route: Route) -> bool:
    """Return whether ``route`` carries a shed-bound order past ``QUANTITY_AXIS``.

    Checked on the raw recorded quantity, not through ``encode_turn``: after
    fix round 1 a shed-bound over-axis order clamps rather than raising, so a
    raise can no longer be used to detect one. This mirrors exactly what
    ``sim.rollout._order_quantity`` treats as the shed-bound, safe-to-clamp
    case -- ``SHED_BOUND_VERBS`` -- so a route this returns ``True`` for is a
    real instance of the class fix round 1 found, not an approximation of it.
    """
    for action in route:
        for order in action["market"]:
            if not order or str(order[0]) not in SHED_BOUND_VERBS or len(order) < 3:
                continue
            try:
                quantity = int(order[2])
            except (TypeError, ValueError):
                continue
            if quantity > QUANTITY_AXIS:
                return True
    return False


def _find_sentinel_episode(archive: Path) -> dict:
    """Return one real, ``ENGINE``-version episode naming a "sell everything" order.

    Scanned in ascending ``episode_id`` order so the same episode is found on
    every run of a given archive, not whichever the manifest happens to list
    first. This is the pin: rather than hope ``_top_episodes``' sample
    happens to include one, this searches for a real instance and hands it
    back so the regression test below always has one to replay.

    Args:
        archive: Daily archive to search.

    Returns:
        The first matching episode.

    Raises:
        AssertionError: If no engine-matched episode in the archive names a
            shed-bound order past ``QUANTITY_AXIS``. The corpus produced this
            idiom in 4.75% of a 400-episode sample when this pin was written;
            an archive with none at all is worth knowing about loudly, not
            skipping past.
    """
    rows = sorted(read_manifest(archive), key=lambda row: row.episode_id)
    with zipfile.ZipFile(archive) as bundle:
        for row in rows:
            episode = json.loads(bundle.read(f"{row.episode_id}.json"))
            if episode.get("module_version") != ENGINE:
                continue
            if any(
                _names_a_sell_everything_order(from_episode(episode, seat))
                for seat in (0, 1)
            ):
                return episode
    raise AssertionError(
        f"no {ENGINE}-version episode in {archive} names a shed-bound market "
        f"order above QUANTITY_AXIS ({QUANTITY_AXIS}); the regression this "
        "pin exists for cannot be checked against this archive"
    )


def _routes(episode: dict) -> tuple[Route, Route, int]:
    """Return ``(seat_zero, seat_one, seed)`` for one decoded episode."""
    return (
        from_episode(episode, 0),
        from_episode(episode, 1),
        int(episode["info"]["seed"]),
    )


def test_recorded_top_episodes_replay_to_exact_banks() -> None:
    assert ARCHIVE is not None  # narrows the type; the module skips otherwise
    episodes = _top_episodes(ARCHIVE, EPISODE_COUNT)
    assert len(episodes) == EPISODE_COUNT

    failures: list[str] = []
    for episode in episodes:
        episode_id = episode["info"]["EpisodeId"]
        seat_zero, seat_one, seed = _routes(episode)
        try:
            banks = replay([seat_zero], [seat_one], [seed])
        except ValueError as error:
            failures.append(f"episode {episode_id}: could not replay: {error}")
            continue
        expected = torch.tensor([episode["rewards"]], dtype=torch.int64)
        if not torch.equal(banks, expected):
            failures.append(
                f"episode {episode_id}: replayed {banks.tolist()} != recorded "
                f"{expected.tolist()}"
            )
        else:
            LOGGER.info("episode %d: replayed to the exact recorded bank", episode_id)

    assert not failures, (
        f"{len(failures)}/{len(episodes)} top episodes did not replay to their "
        "recorded bank:\n" + "\n".join(failures)
    )


def test_batched_replay_matches_one_at_a_time() -> None:
    assert ARCHIVE is not None  # narrows the type; the module skips otherwise
    episodes = _top_episodes(ARCHIVE, EPISODE_COUNT)
    assert len(episodes) == EPISODE_COUNT
    seat_zero, seat_one, seeds = [], [], []
    for episode in episodes:
        zero, one, seed = _routes(episode)
        seat_zero.append(zero)
        seat_one.append(one)
        seeds.append(seed)

    looped = torch.cat(
        [
            replay([zero], [one], [seed])
            for zero, one, seed in zip(seat_zero, seat_one, seeds, strict=True)
        ]
    )
    batched = replay(seat_zero, seat_one, seeds)

    assert torch.equal(looped, batched)


def test_a_sell_everything_regression_episode_replays_to_the_exact_bank() -> None:
    """Pin the exact failure class fix round 1 found: a real ``SELL X 999`` episode.

    ``_top_episodes``' random sample is not guaranteed to include one --
    that unluckiness is exactly what let this class of bug through the first
    time. This test does not rely on luck: it searches the archive for a real
    episode naming a shed-bound order above ``QUANTITY_AXIS`` and asserts it
    replays to its exact recorded bank, so the fix that made that order
    clamp instead of raise stays proven regardless of what any other test's
    sample happens to contain.
    """
    assert ARCHIVE is not None  # narrows the type; the module skips otherwise
    episode = _find_sentinel_episode(ARCHIVE)
    episode_id = episode["info"]["EpisodeId"]
    seat_zero, seat_one, seed = _routes(episode)

    banks = replay([seat_zero], [seat_one], [seed])

    expected = torch.tensor([episode["rewards"]], dtype=torch.int64)
    assert torch.equal(banks, expected), (
        f"episode {episode_id}: replayed {banks.tolist()} != recorded "
        f"{expected.tolist()}"
    )
