"""Recorded top episodes must replay through the simulator to their exact banks.

This is the differential proof of the whole quantity lane. A recorded top
episode exercises bulk transfers on most of its PICKUP/PLACE actions --
branches the RL-action-space differential tests structurally never reach --
and the engine is deterministic given seed and both seats' actions, so
equality is exact or the lane is wrong. Do not weaken to a tolerance; a
mismatch is a bug with a turn number, found by comparing money per turn.
"""
# ruff: noqa: D103

import importlib.metadata
import json
import zipfile
from pathlib import Path

import pytest
import torch

from kaggriculture.learn.corpus import CORPUS, read_manifest
from kaggriculture.search.route import Route, from_episode
from kaggriculture.sim.tape import replay

ENGINE = importlib.metadata.version("kaggle-environments")
ARCHIVES = (
    sorted(CORPUS.glob("kaggriculture-episodes-*.zip")) if CORPUS.exists() else []
)
ARCHIVE = ARCHIVES[-1] if ARCHIVES else None

pytestmark = pytest.mark.skipif(
    ARCHIVE is None, reason="replay corpus not present on this machine"
)


def _top_episodes(archive: Path, count: int) -> list[dict]:
    """Return the ``count`` strongest ``ENGINE``-version episodes in ``archive``.

    "Strongest" is by ``min_score``, the weaker seat's rating -- the same
    column ``learn.corpus.select`` ranks on -- so a chosen episode had two
    strong players in it, not one strong player against a pushover.
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


def test_recorded_top_episodes_replay_to_their_exact_recorded_banks() -> None:
    assert ARCHIVE is not None  # narrows the type; the module skips otherwise
    episodes = _top_episodes(ARCHIVE, 3)
    assert len(episodes) == 3

    for episode in episodes:
        seat_zero, seat_one, seed = _routes(episode)

        banks = replay([seat_zero], [seat_one], [seed])

        expected = torch.tensor([episode["rewards"]], dtype=torch.int64)
        assert torch.equal(banks, expected), (
            f"episode {episode['info']['EpisodeId']}: replayed {banks.tolist()} "
            f"!= recorded {expected.tolist()}"
        )


def test_batched_replay_of_several_episodes_matches_one_at_a_time() -> None:
    assert ARCHIVE is not None  # narrows the type; the module skips otherwise
    episodes = _top_episodes(ARCHIVE, 3)
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
