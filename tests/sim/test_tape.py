"""Recorded top episodes replay through the simulator to their exact banks.

This is the differential proof of the quantity lane -- the *unit* transfer
lane specifically. A recorded top episode exercises bulk transfers on most of
its PICKUP/PLACE actions, branches the RL-action-space differential tests
structurally never reach, and the engine is deterministic given seed and both
seats' actions, so equality is exact or the lane is wrong. Do not weaken to a
tolerance; a mismatch is a bug with a turn number, found by comparing money
per turn.

Selection is restricted to episodes whose actions ``encode_turn`` can
represent at all: a market order above ``sim.market.QUANTITY_AXIS`` is
outside the simulator's domain today and raises at encode time (see
``sim.tape``'s module docstring for the measured incidence). That restriction
is applied openly here, not as a silent skip -- ``_top_episodes`` returns how
many otherwise-eligible episodes it excluded and why, and both tests log that
count, so a reader sees the proof's real scope rather than mistaking a lucky
sample for full coverage of recorded top play.
"""
# ruff: noqa: D103

import importlib.metadata
import json
import logging
import zipfile
from pathlib import Path

import pytest
import torch

from kaggriculture.learn.corpus import CORPUS, read_manifest
from kaggriculture.search.route import Route, from_episode
from kaggriculture.sim.rollout import encode_turn
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


def _quantity_cap_compliant(episode: dict) -> bool:
    """Return whether both seats' whole routes stay inside the simulator's domain.

    Tries the real ``encode_turn`` on every recorded action rather than
    re-deriving the cap here, so this stays accurate as the domain widens (a
    follow-up task raises ``QUANTITY_AXIS``) without needing to change in
    lockstep with it. A ``ValueError`` is exactly what ``encode_turn`` raises
    for a market order the reference engine would execute but the simulator
    cannot yet, per ``sim.rollout._order_quantity``.
    """
    for seat in (0, 1):
        for action in from_episode(episode, seat):
            try:
                encode_turn(action)
            except ValueError:
                return False
    return True


def _top_episodes(archive: Path, count: int) -> tuple[list[dict], int]:
    """Return the ``count`` strongest cap-compliant, ``ENGINE``-version episodes.

    "Strongest" is by ``min_score``, the weaker seat's rating -- the same
    column ``learn.corpus.select`` ranks on -- so a chosen episode had two
    strong players in it, not one strong player against a pushover.

    Cap-compliance is a real filter, not an artifact of luck: most top
    episodes carry a market order above ``QUANTITY_AXIS`` (see ``sim.tape``'s
    module docstring), so selecting on rating and engine version alone would
    still, more often than not, hand back an episode ``replay`` cannot
    encode. Filtering here keeps the proof honest about what it covers; the
    exclusion count returned lets the caller say so out loud instead of
    silently narrowing the sample.

    Args:
        archive: Daily archive to select from.
        count: How many cap-compliant episodes to return.

    Returns:
        The selected episodes, best-rated first, and how many otherwise
        rating- and engine-eligible episodes were skipped for having a
        market order outside the simulator's domain.
    """
    rows = sorted(read_manifest(archive), key=lambda row: -row.min_score)
    episodes: list[dict] = []
    excluded = 0
    with zipfile.ZipFile(archive) as bundle:
        for row in rows:
            episode = json.loads(bundle.read(f"{row.episode_id}.json"))
            if episode.get("module_version") != ENGINE:
                continue
            if not _quantity_cap_compliant(episode):
                excluded += 1
                continue
            episodes.append(episode)
            if len(episodes) >= count:
                break
    return episodes, excluded


def _routes(episode: dict) -> tuple[Route, Route, int]:
    """Return ``(seat_zero, seat_one, seed)`` for one decoded episode."""
    return (
        from_episode(episode, 0),
        from_episode(episode, 1),
        int(episode["info"]["seed"]),
    )


def test_recorded_top_cap_compliant_episodes_replay_to_exact_banks() -> None:
    assert ARCHIVE is not None  # narrows the type; the module skips otherwise
    episodes, excluded = _top_episodes(ARCHIVE, 3)
    LOGGER.info(
        "selected %d cap-compliant episodes; excluded %d rating/engine-eligible "
        "episodes for a market order above QUANTITY_AXIS",
        len(episodes),
        excluded,
    )
    assert len(episodes) == 3

    for episode in episodes:
        seat_zero, seat_one, seed = _routes(episode)

        banks = replay([seat_zero], [seat_one], [seed])

        expected = torch.tensor([episode["rewards"]], dtype=torch.int64)
        assert torch.equal(banks, expected), (
            f"episode {episode['info']['EpisodeId']}: replayed {banks.tolist()} "
            f"!= recorded {expected.tolist()}"
        )


def test_batched_cap_compliant_replay_matches_one_at_a_time() -> None:
    assert ARCHIVE is not None  # narrows the type; the module skips otherwise
    episodes, excluded = _top_episodes(ARCHIVE, 3)
    LOGGER.info(
        "selected %d cap-compliant episodes; excluded %d rating/engine-eligible "
        "episodes for a market order above QUANTITY_AXIS",
        len(episodes),
        excluded,
    )
    assert len(episodes) == 3
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
