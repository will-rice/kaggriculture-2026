"""Tests for the packaging guard that keeps offline tooling out of the archive."""

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from kaggriculture.routes import STORE
from kaggriculture.scripts import package as package_script
from kaggriculture.scripts.package import build
from tests.test_vendored_policies import V54_REFERENCE_BANKS

_needs_prototype_store = pytest.mark.skipif(
    not STORE.exists(), reason="prototype store not present on this machine"
)


@_needs_prototype_store
def test_the_archive_does_not_ship_the_search_package(tmp_path: Path) -> None:
    """The search package has no place in a 4 MB agent archive.

    It happens to import no torch today -- the simulator encoder that did was
    deleted -- so that is not the reason it stays excluded. It is offline
    hill-climbing tooling that plays hundreds of games to find a better route,
    work the submitted agent never does at play time, and the guard also closes
    the path by which a later edit could reintroduce a heavy import behind it.
    """
    archive = build(tmp_path / "submission.tar.gz")

    with tarfile.open(archive) as bundle:
        names = bundle.getnames()

    assert not [
        name for name in names if "/search/" in name or name.endswith("/search")
    ]


@_needs_prototype_store
def test_the_archive_ships_the_served_policy_and_plays_its_gate_episode(
    tmp_path: Path,
) -> None:
    """The archive must play the same season the gate scored, not merely build.

    ``EXCLUDED`` drops whole trees by name at every directory level, so a policy
    module lands in the archive by not matching any of them -- which is a
    property of the filename, not a decision anyone took. That is fine until the
    day it is not, and the failure is invisible: the build succeeds, the archive
    is the right size, and the agent raises on turn zero in a sandbox with no
    logs.

    So this asserts both halves at once. The served module is in the archive,
    and the archive's own ``main.py``, run from the extracted directory by the
    engine, reproduces the exact bank pair
    ``test_vendored_policies.V54_REFERENCE_BANKS`` pins for the unpackaged
    module on the same seed against the same opponent. Both seats come out of
    the extraction, so nothing in this repository is on the path.
    """
    archive = build(tmp_path / "submission.tar.gz")
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(archive) as bundle:
        names = bundle.getnames()
        bundle.extractall(extracted, filter="data")

    assert "kaggriculture/kaito_v54_policy.py" in names
    assert "kaggriculture/boatlee_v14_policy.py" in names

    script = """
import json
import runpy
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
agent = runpy.run_path(str(root / "main.py"))["agent"]
from kaggle_environments import make
from kaggriculture.boatlee_v14_policy import agent as opponent

environment = make(
    "kaggriculture", configuration={"episodeSteps": 720, "seed": 700_000}
)
environment.run([agent, opponent])
final = environment.steps[-1]
print(json.dumps({
    "statuses": [str(state.status) for state in final],
    "banks": [int(state.reward) for state in final],
}))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(extracted)],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, result.stderr
    evidence = json.loads(result.stdout.splitlines()[-1])
    assert evidence["statuses"] == ["DONE", "DONE"]
    assert tuple(evidence["banks"]) == V54_REFERENCE_BANKS


def test_build_accepts_a_self_contained_alternate_entrypoint(tmp_path: Path) -> None:
    """Candidate packaging can be tested without changing the served default."""
    entrypoint = tmp_path / "main.py"
    entrypoint.write_text(
        "from kaggriculture.hybrid.policy import agent\n\n__all__ = ['agent']\n"
    )

    archive = build(tmp_path / "hybrid.tar.gz", entrypoint=entrypoint, required={})

    with tarfile.open(archive) as bundle:
        names = bundle.getnames()
        packaged_main = bundle.extractfile("main.py")
        assert packaged_main is not None
        source = packaged_main.read().decode()
    assert source == entrypoint.read_text()
    assert not any("/search/" in name for name in names)


@pytest.mark.parametrize("escape", ["absolute", "dotdot", "symlink", "hardlink"])
def test_required_artifacts_cannot_escape_the_package_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, escape: str
) -> None:
    """Required inputs and staged destinations stay package-local by path and inode."""
    package_root = tmp_path / "source" / "kaggriculture"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    if escape == "absolute":
        artifact = outside
    elif escape == "dotdot":
        artifact = package_root / ".." / ".." / "outside.bin"
    else:
        artifact = package_root / f"{escape}.bin"
        if escape == "symlink":
            artifact.symlink_to(outside)
        else:
            os.link(outside, artifact)
    entrypoint = tmp_path / "main.py"
    entrypoint.write_text(
        "def agent(observation, configuration=None):\n    return {}\n"
    )
    monkeypatch.setattr(package_script, "PACKAGE_ROOT", package_root)

    with pytest.raises(ValueError, match="package|symlink|hardlink"):
        package_script.build(
            tmp_path / "submission.tar.gz",
            entrypoint=entrypoint,
            required={artifact: "producer"},
        )
