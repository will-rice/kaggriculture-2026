"""Tests for the artefact that actually gets uploaded."""

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
from kaggle_environments.agent import get_last_callable

from kaggriculture.agent import EpisodeAgent
from kaggriculture.hybrid.config import HybridConfig, to_runtime
from kaggriculture.kaito_v54_policy import agent as kaito_v54_agent
from kaggriculture.scripts.package import (
    ENTRYPOINT,
    _refuse_a_shadowed_entrypoint,
    build,
)
from kaggriculture.task import MatchTask


def test_entrypoint_exposes_the_agent_last() -> None:
    """Nothing may be defined after the agent: Kaggle plays the LAST callable.

    The invariant is about position, not identity -- appending a helper below
    the import silently ships that helper as the agent, and the episode fails
    on turn zero with no clue why. So this compares the last callable against
    whatever ``main`` binds to ``agent``, and stays true whichever agent we
    serve. An earlier version imported one policy by name and asserted equality
    with it, which only restated the import line and had to be rewritten every
    time the served agent changed.
    """
    source = ENTRYPOINT.read_text()
    namespace: dict[str, object] = {}
    exec(compile(source, str(ENTRYPOINT), "exec"), namespace)

    assert get_last_callable(source, path=str(ENTRYPOINT)) is namespace["agent"]


def test_default_entrypoint_is_pinned_to_kaito_v54() -> None:
    """Candidate packaging cannot silently change the served default policy."""
    namespace: dict[str, object] = {}
    exec(compile(ENTRYPOINT.read_text(), str(ENTRYPOINT), "exec"), namespace)
    assert namespace["agent"] is kaito_v54_agent


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


def test_hybrid_policy_imports_without_training_or_authoring_dependencies() -> None:
    """The development policy stays safe to package before it is served."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import kaggriculture.hybrid.policy; "
            "assert 'torch' not in sys.modules; "
            "assert 'pydantic' not in sys.modules; "
            "assert 'kaggriculture.search' not in sys.modules; "
            "assert 'kaggriculture.economic_policy' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr


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


def test_the_build_refuses_a_shadowed_entrypoint(tmp_path: Path) -> None:
    """A broken entrypoint must not be able to become an archive.

    The positional invariant above is also enforced at build time, because
    `main.py` is edited by automation -- a subagent decoding a public kernel
    clobbered it once by executing a notebook cell. A test reports the damage
    after the fact; a build that refuses means the broken archive never exists
    to be uploaded, which is the difference between noticing and being safe.
    """
    entrypoint = tmp_path / "main.py"
    entrypoint.write_text(
        "from kaggriculture.boatlee_v14_policy import agent\n"
        "\n"
        "\n"
        "def _appended_below():\n"
        "    return None\n"
    )

    with pytest.raises(RuntimeError, match="after its agent"):
        _refuse_a_shadowed_entrypoint(entrypoint)


def test_the_build_refuses_an_entrypoint_with_no_agent(tmp_path: Path) -> None:
    """An entrypoint binding no `agent` would ship an archive that plays nothing."""
    entrypoint = tmp_path / "main.py"
    entrypoint.write_text("def helper():\n    return None\n")

    with pytest.raises(RuntimeError, match="binds no"):
        _refuse_a_shadowed_entrypoint(entrypoint)


def test_build_refuses_a_same_named_but_different_final_callable(
    tmp_path: Path,
) -> None:
    """Callable names cannot disguise a shadow that the runner would execute."""
    entrypoint = tmp_path / "main.py"
    entrypoint.write_text(
        "def agent(observation=None, configuration=None):\n"
        "    return {'type': 'PASS'}\n"
        "real_agent = agent\n"
        "def shadow(observation=None, configuration=None):\n"
        "    return {'type': 'CONVERT'}\n"
        "shadow.__name__ = 'agent'\n"
        "agent = real_agent\n"
    )

    with pytest.raises(RuntimeError, match="after its agent"):
        _refuse_a_shadowed_entrypoint(entrypoint)


def test_extracted_hybrid_archive_plays_a_real_episode_with_runtime_headroom(
    tmp_path: Path,
) -> None:
    """The actual alternate archive stays small, pure, complete, and fast."""
    payload = to_runtime(HybridConfig.default()).to_payload()
    hybrid_main = tmp_path / "main.py"
    hybrid_main.write_text(
        "from kaggriculture.hybrid.policy import build_agent\n"
        "from kaggriculture.hybrid.runtime import RuntimeConfig\n\n"
        f"_RUNTIME = RuntimeConfig.from_payload({payload!r})\n"
        "agent = build_agent(_RUNTIME)\n\n"
        "__all__ = ['agent']\n"
    )
    archive = build(tmp_path / "hybrid.tar.gz", entrypoint=hybrid_main, required={})
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(archive) as bundle:
        names = bundle.getnames()
        bundle.extractall(extracted, filter="data")

    script = """
import json
import os
import runpy
import sys
from pathlib import Path
from time import perf_counter

root = Path(sys.argv[1])
os.nice(10)
sys.path.insert(0, str(root))
before = set(sys.modules)
namespace = runpy.run_path(str(root / "main.py"))
agent_modules = sorted(set(sys.modules) - before)
agent = namespace["agent"]
from kaggle_environments import make
from kaggriculture.constants import ENVIRONMENT, EPISODE_STEPS
from kaggriculture.economic_policy import agent as economic_agent

timings = []
def timed_agent(observation, configuration=None):
    started = perf_counter()
    try:
        return agent(observation, configuration)
    finally:
        timings.append(perf_counter() - started)

environment = make(
    ENVIRONMENT,
    configuration={"episodeSteps": EPISODE_STEPS, "seed": 850_000},
)
environment.run([timed_agent, economic_agent])
ordered = sorted(timings)
p99 = ordered[max(0, min(len(ordered) - 1, (99 * len(ordered) + 99) // 100 - 1))]
print(json.dumps({
    "statuses": [str(state.status) for state in environment.steps[-1]],
    "max_turn": max(timings),
    "p99_turn": p99,
    "agent_modules": agent_modules,
}))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(extracted)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    evidence = json.loads(result.stdout.splitlines()[-1])
    imported_roots = {name.split(".", 1)[0] for name in evidence["agent_modules"]}
    assert archive.stat().st_size < 4 * 1024 * 1024
    assert evidence["statuses"] == ["DONE", "DONE"]
    assert evidence["max_turn"] < 1.0
    assert evidence["p99_turn"] < 0.100
    assert "torch" not in imported_roots
    assert "pydantic" not in imported_roots
    assert not any("/search/" in name for name in names)
