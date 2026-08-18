"""Replaying a real episode's own actions must reproduce its own result."""

import json
import zipfile
from pathlib import Path

import pytest

from kaggriculture.search.arena import play
from kaggriculture.search.route import from_episode

ARCHIVE = Path("/data/kaggriculture/episodes/kaggriculture-episodes-2026-08-15.zip")


@pytest.mark.skipif(not ARCHIVE.exists(), reason="replay corpus not on this machine")
def test_replaying_an_episode_reproduces_its_recorded_banks() -> None:
    """The engine is deterministic given seed and actions, so this is exact.

    This is the whole warrant for the arena. It covers the action grammar, seat
    assignment, market queue-position coupling and turn alignment in one
    assertion, and it is the difference between an arena that is faithful and
    one that is merely plausible. Do not weaken it to a tolerance.
    """
    with zipfile.ZipFile(ARCHIVE) as bundle:
        episode = json.loads(bundle.read("93454366.json"))
    seed = int(episode["info"]["seed"])
    expected = [(int(episode["rewards"][0]), int(episode["rewards"][1]))]

    banks = play(from_episode(episode, seat=0), from_episode(episode, seat=1), [seed])

    assert banks == expected
