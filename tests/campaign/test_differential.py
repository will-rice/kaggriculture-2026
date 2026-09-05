"""The port must agree with the reference engine on every recorded step.

An archived episode carries its seed in ``info.seed`` and the reference
engine's own observation at every step, so replaying its actions through the
port and comparing is a differential test against the oracle with no
reference-engine time at all.

Tests parametrize over plain integers, not over ``tapes.Episode`` instances:
pytest evaluates a parametrize decorator's arguments at collection time, and
loading episodes there would mean every ``pytest`` invocation in this repo —
including the pre-commit hook, on every commit — pays for parsing however
many tapes the slow test needs, before ``-m 'not slow'`` gets a chance to
deselect it. ``tapes.qualifying`` defers that work into the test body.
"""

import pytest

from kaggriculture.campaign import tapes
from kaggriculture.campaign.engine.wrapper import Engine

# every tape lives in the local episode archive.
pytestmark = pytest.mark.local_data


def strip(observation: dict) -> dict:
    """The framework adds remainingOverageTime; the engine does not know it."""
    return {k: v for k, v in observation.items() if k != "remainingOverageTime"}


@pytest.mark.parametrize("i", range(8), ids=lambda i: f"tape{i}")
def test_replaying_an_archived_episode_reproduces_every_observation(i: int) -> None:
    """Replaying an archived episode's actions must reproduce every observation."""
    episode = tapes.qualifying(i)
    engine = Engine(seed=episode.seed)
    for index in range(1, len(episode.steps)):
        recorded = episode.steps[index]
        engine.step(
            episode.steps[index][0]["action"], episode.steps[index][1]["action"]
        )
        for player in (0, 1):
            assert engine.observation(player) == strip(
                recorded[player]["observation"]
            ), f"seed {episode.seed} step {index} player {player}"
    assert (engine.bank(0), engine.bank(1)) == (
        float(episode.steps[-1][0]["reward"]),
        float(episode.steps[-1][1]["reward"]),
    )


@pytest.mark.slow
@pytest.mark.parametrize("i", range(200), ids=lambda i: f"tape{i}")
def test_two_hundred_episodes(i: int) -> None:
    """The full acceptance sweep: 200 archived episodes, zero disagreement."""
    test_replaying_an_archived_episode_reproduces_every_observation(i)
