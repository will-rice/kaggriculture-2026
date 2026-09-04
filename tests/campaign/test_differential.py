"""The port must agree with the reference engine on every recorded step.

An archived episode carries its seed in ``info.seed`` and the reference
engine's own observation at every step, so replaying its actions through the
port and comparing is a differential test against the oracle with no
reference-engine time at all.
"""

import pytest

from kaggriculture.campaign import tapes
from kaggriculture.campaign.engine.wrapper import Engine


def strip(observation: dict) -> dict:
    """The framework adds remainingOverageTime; the engine does not know it."""
    return {k: v for k, v in observation.items() if k != "remainingOverageTime"}


@pytest.mark.parametrize("episode", tapes.sample(8), ids=lambda e: str(e.seed))
def test_replaying_an_archived_episode_reproduces_every_observation(
    episode: tapes.Episode,
) -> None:
    """Replaying an archived episode's actions must reproduce every observation."""
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
@pytest.mark.parametrize("episode", tapes.sample(200), ids=lambda e: str(e.seed))
def test_two_hundred_episodes(episode: tapes.Episode) -> None:
    """The full acceptance sweep: 200 archived episodes, zero disagreement."""
    test_replaying_an_archived_episode_reproduces_every_observation(episode)
