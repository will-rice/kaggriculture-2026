"""Tests for the artefact that actually gets uploaded."""

import tarfile
from pathlib import Path

import pytest
from kaggle_environments.agent import get_last_callable

from kaggriculture.scripts.package import (
    ENTRYPOINT,
    _refuse_a_shadowed_entrypoint,
    build,
)


def test_entrypoint_exposes_the_agent_last() -> None:
    """Nothing may be defined after the agent: Kaggle plays the LAST callable.

    The invariant is about position, not identity -- appending a helper below
    the import silently ships that helper as the agent, and the episode fails
    on turn zero with no clue why. So this compares the last callable against
    whatever ``main`` binds to ``agent``, and stays true whichever agent we
    serve.
    """
    source = ENTRYPOINT.read_text()
    namespace: dict[str, object] = {}
    exec(compile(source, str(ENTRYPOINT), "exec"), namespace)

    assert get_last_callable(source, path=str(ENTRYPOINT)) is namespace["agent"]


def archive_names(tmp_path: Path) -> list[str]:
    """Return the member names of a freshly built archive."""
    with tarfile.open(build(tmp_path / "submission.tar.gz")) as tar:
        return tar.getnames()


def test_archive_holds_the_entrypoint_beside_the_package(tmp_path: Path) -> None:
    """Kaggle imports main.py from the archive root with the package alongside."""
    names = archive_names(tmp_path)

    assert len(names) == len(set(names))
    assert "main.py" in names
    assert "kaggriculture/served/main.py" in names
    assert "kaggriculture/constants.py" in names
    assert not any(name.startswith("kaggriculture/scripts") for name in names)
    assert not any(name.startswith("kaggriculture/campaign") for name in names)
    assert not any("__pycache__" in name for name in names)


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
        "def agent(observation, configuration=None):\n"
        "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
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
