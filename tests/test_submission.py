"""Tests for the artefact that actually gets uploaded."""

import tarfile
from pathlib import Path

import pytest

from kaggriculture.campaign import config
from kaggriculture.scripts.package import (
    ENTRYPOINT,
    _refuse_a_shadowed_entrypoint,
    build,
)

# The floor is written by the gate on promotion and lives under `run/`, which
# is not in the repository. A checkout that has never run a campaign has
# nothing to ship, so there is nothing here to prove.
needs_the_floor = pytest.mark.skipif(
    not ENTRYPOINT.exists(), reason=f"no campaign floor at {ENTRYPOINT}"
)


def test_the_entrypoint_is_the_campaign_floor() -> None:
    """What ships is the file the gate writes, not the seed committed in `src/`.

    The two used to be the same path, and a promotion wrote through `src/`
    to reach it. Shipping the floor is what lets the seed stay a seed: the
    cold start begins from `seed/main.py` while the ladder gets whatever
    the campaign has actually promoted.
    """
    assert ENTRYPOINT == config.LIVE.floor / "main.py"
    assert ENTRYPOINT != config.SEED


@needs_the_floor
def test_entrypoint_exposes_the_agent_last() -> None:
    """Nothing may be defined after the agent: Kaggle plays the LAST callable.

    The invariant is about position, not identity -- appending a helper below
    the import silently ships that helper as the agent, and the episode fails
    on turn zero with no clue why. The floor is evolved code, so it is never
    executed here: the build guard puts it through the candidate gate, whose
    dynamic half runs in a child process in a scratch directory.
    """
    _refuse_a_shadowed_entrypoint(ENTRYPOINT)


@needs_the_floor
def test_the_archive_holds_the_entrypoint_and_nothing_it_could_import(
    tmp_path: Path,
) -> None:
    """Kaggle imports main.py from the archive root, and finds only main.py.

    The engine library and the `kaggriculture` package used to ride along.
    A candidate that imported either would validate in the campaign and die
    on the ladder, which is the one failure this system cannot afford.
    """
    with tarfile.open(build(tmp_path / "submission.tar.gz")) as tar:
        names = tar.getnames()

    assert sorted(names) == ["LICENSE", "main.py"]
    assert not any(name.endswith(".so") for name in names)
    assert not any(name.startswith("kaggriculture") for name in names)


def test_the_build_refuses_a_shadowed_entrypoint(tmp_path: Path) -> None:
    """A broken entrypoint must not be able to become an archive.

    The positional invariant above is also enforced at build time, because
    the entrypoint is written by automation -- the gate overwrites the floor
    on every promotion, and a subagent decoding a public kernel clobbered one
    once by executing a notebook cell. A test reports the damage after the
    fact; a build that refuses means the broken archive never exists to be
    uploaded, which is the difference between noticing and being safe.
    """
    entrypoint = tmp_path / "main.py"
    entrypoint.write_text(
        "def agent(observation, configuration=None):\n"
        "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
        "\n"
        "\n"
        "def _appended_below():\n"
        "    return None\n"
    )

    with pytest.raises(RuntimeError, match="shadowed by"):
        _refuse_a_shadowed_entrypoint(entrypoint)


def test_the_build_refuses_an_entrypoint_with_no_agent(tmp_path: Path) -> None:
    """An entrypoint binding no `agent` would ship an archive that plays nothing."""
    entrypoint = tmp_path / "main.py"
    entrypoint.write_text("def helper():\n    return None\n")

    with pytest.raises(RuntimeError, match="no top-level function named agent"):
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

    with pytest.raises(RuntimeError, match="shadowed by"):
        _refuse_a_shadowed_entrypoint(entrypoint)
