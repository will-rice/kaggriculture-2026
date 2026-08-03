"""Tests for the artefact that actually gets uploaded."""

import tarfile
from pathlib import Path

from kaggle_environments.agent import get_last_callable

from kaggriculture.agent import EpisodeAgent
from kaggriculture.policy import agent
from kaggriculture.scripts.package import ENTRYPOINT, build
from kaggriculture.task import MatchTask


def test_entrypoint_exposes_the_agent_last() -> None:
    """Kaggle takes the last callable in main.py, so it must be our policy."""
    loaded = get_last_callable(ENTRYPOINT.read_text(), path=str(ENTRYPOINT))

    assert loaded is agent


def test_entrypoint_plays_a_full_episode() -> None:
    """The submitted file runs as an agent path, the way the runner loads it."""
    task = MatchTask(opponent="pass", seed=0)

    scores = EpisodeAgent(spec=str(ENTRYPOINT)).run(task)

    assert task.evaluate(scores) == 1.0


def test_archive_holds_the_entrypoint_beside_the_package(tmp_path: Path) -> None:
    """Kaggle imports main.py from the archive root with the package alongside."""
    archive = build(tmp_path / "submission.tar.gz")

    with tarfile.open(archive) as tar:
        names = tar.getnames()

    assert len(names) == len(set(names))
    assert "main.py" in names
    assert "kaggriculture/policy.py" in names
    assert not any(name.startswith("kaggriculture/scripts") for name in names)
    assert not any("__pycache__" in name for name in names)
