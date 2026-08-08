"""Tests for the artefact that actually gets uploaded."""

import tarfile
from pathlib import Path

from kaggle_environments.agent import get_last_callable

from kaggriculture.agent import EpisodeAgent
from kaggriculture.boatlee_v14_policy import agent
from kaggriculture.scripts.package import ENTRYPOINT, build
from kaggriculture.task import MatchTask


def test_entrypoint_exposes_the_agent_last() -> None:
    """Kaggle takes the last callable in main.py, so it must be the served agent.

    Imported by name rather than checked structurally, so switching which agent
    `main.py` serves fails here until the switch is deliberate. It caught the
    move from kaito v21.1 to boatlee v14 on 2026-08-08.
    """
    loaded = get_last_callable(ENTRYPOINT.read_text(), path=str(ENTRYPOINT))

    assert loaded is agent


def test_entrypoint_plays_a_full_episode() -> None:
    """The submitted file runs as an agent path, the way the runner loads it."""
    task = MatchTask(opponent="pass", seed=0)

    scores = EpisodeAgent(spec=str(ENTRYPOINT)).run(task)

    assert task.evaluate(scores) == 1.0


def archive_names(tmp_path: Path) -> list[str]:
    """Return the member names of a freshly built archive."""
    with tarfile.open(build(tmp_path / "submission.tar.gz")) as tar:
        return tar.getnames()


def test_archive_holds_the_entrypoint_beside_the_package(tmp_path: Path) -> None:
    """Kaggle imports main.py from the archive root with the package alongside."""
    names = archive_names(tmp_path)

    assert len(names) == len(set(names))
    assert "main.py" in names
    assert "kaggriculture/policy.py" in names
    assert "kaggriculture/economic_policy.py" in names
    assert not any(name.startswith("kaggriculture/scripts") for name in names)
    assert not any("__pycache__" in name for name in names)


def test_the_submission_ships_nothing_that_needs_torch(tmp_path: Path) -> None:
    """The served agent replays routes; the learned policy is not in the archive.

    Shipping ``learn`` cost 39 MB of checkpoint the archive never loaded --
    measured, an archive carrying it was 41.7 MB and ``torch`` was still absent
    from ``sys.modules`` after a full episode played from it. Excluding the
    package also closes the path by which a later edit could pull torch into the
    agent, where importing it spends 10.7 seconds of a 60 second overage pool.

    Asserted on the whole package rather than on ``policy.pt`` alone: it is the
    import reaching torch that costs the pool, not the weights sitting beside
    it.
    """
    names = archive_names(tmp_path)

    assert not [name for name in names if "/learn/" in name]
    assert not [name for name in names if name.endswith(".pt")]


def test_the_submission_carries_the_prototype_store(tmp_path: Path) -> None:
    """A packaged route agent that cannot find its store raises on turn zero.

    ``routes.play`` loads ``routes.STORE`` from beside the package, and until
    the store was copied in there was nothing there: ``copytree`` walks the
    source tree and the store is gitignored, so the archive shipped the code
    that reads it and not the file. The failure lands in the sandbox on the
    first turn, where the only symptom is a zero.
    """
    names = archive_names(tmp_path)

    assert "kaggriculture/routes/prototypes.json.gz" in names
    assert "kaggriculture/routes/play.py" in names
    assert "kaggriculture/routes/store.py" in names
    assert "kaggriculture/routes/signature.py" in names


def test_the_submission_ships_no_training_code(tmp_path: Path) -> None:
    """The sandbox has no network; a wandb import forfeits the episode on turn 0.

    ``corpus.py`` and ``dataset.py`` reach for tqdm and a ``/data`` path that
    exists on the workstation and nowhere else, ``learn/scripts`` imports wandb
    outright, and ``routes/scripts/harvest.py`` imports both ``corpus`` and
    ``tqdm``.
    """
    names = archive_names(tmp_path)

    assert not any("/scripts/" in name for name in names)
    assert "kaggriculture/learn/corpus.py" not in names
    assert "kaggriculture/learn/dataset.py" not in names
    assert "kaggriculture/routes/scripts/harvest.py" not in names
