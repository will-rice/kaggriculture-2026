"""Tests for the evaluation harness, its config, and match scoring."""

from kaggriculture.agent import Agent, EpisodeAgent
from kaggriculture.config import HarnessConfig
from kaggriculture.harness import Harness
from kaggriculture.task import MatchTask, Task


def test_episode_agent_and_match_satisfy_the_protocols() -> None:
    """The concrete types stay usable wherever the protocols are expected."""
    assert isinstance(EpisodeAgent(spec="starter"), Agent)
    assert isinstance(MatchTask(opponent="pass", seed=0), Task)


def test_match_scoring_follows_the_ladder() -> None:
    """Only the win, loss or tie matters — the coin margin does not."""
    task = MatchTask(opponent="pass", seed=0)

    assert task.evaluate([9000.0, 10.0]) == 1.0
    assert task.evaluate([10.0, 9000.0]) == 0.0
    assert task.evaluate([500.0, 500.0]) == 0.5


def test_matches_cover_every_opponent_and_seed() -> None:
    """The harness builds one task per opponent per seeded game."""
    config = HarnessConfig(games=3, seed=5, opponents=("pass", "random"))

    tasks = Harness(config=config).matches()

    assert [task.id for task in tasks] == [
        "pass:5",
        "pass:6",
        "pass:7",
        "random:5",
        "random:6",
        "random:7",
    ]


def test_harness_plays_short_episodes_in_parallel() -> None:
    """A sweep returns one scored result per task."""
    config = HarnessConfig(games=2, seed=0, opponents=("pass",), episode_steps=48, max_workers=2)
    harness = Harness(config=config)

    results = harness.run(EpisodeAgent(spec="starter"), harness.matches())

    assert len(results) == 2
    assert all(result.error is None for result in results)
    assert all(result.scores is not None for result in results)
