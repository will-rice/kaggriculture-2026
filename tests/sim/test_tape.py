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

Selection is by rating and engine version alone -- no filtering on whether an
episode's actions happen to stay inside the simulator's domain. A market
order naming a quantity past even the widened axis (the engine's own "sell
everything" sentinel is the one seen in practice) is still outside it, and
``encode_turn`` still raises on it rather than silently clamping. If that
happens to land inside the top-rated sample this test selects, the test below
fails loudly, naming the episode and the reason -- it does not skip the
episode and quietly report on a smaller, luckier sample.
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


def _top_episodes(archive: Path, count: int) -> list[dict]:
    """Return the ``count`` strongest ``ENGINE``-version episodes, by ``min_score``.

    "Strongest" is by ``min_score``, the weaker seat's rating -- the same
    column ``learn.corpus.select`` ranks on -- so a chosen episode had two
    strong players in it, not one strong player against a pushover.

    No cap-compliance filtering: unlike the market lane's pre-widening state,
    an ordinary top-rated episode is now expected to stay inside the
    simulator's domain, and a selection that filtered on it anyway would risk
    quietly narrowing the sample back down without saying so. If a selected
    episode still cannot replay, the tests below say so loudly instead.

    Args:
        archive: Daily archive to select from.
        count: How many episodes to return.

    Returns:
        The ``count`` best-rated, engine-matched episodes, best first.
    """
    rows = sorted(read_manifest(archive), key=lambda row: -row.min_score)
    episodes: list[dict] = []
    with zipfile.ZipFile(archive) as bundle:
        for row in rows:
            episode = json.loads(bundle.read(f"{row.episode_id}.json"))
            if episode.get("module_version") != ENGINE:
                continue
            episodes.append(episode)
            if len(episodes) >= count:
                break
    return episodes


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
