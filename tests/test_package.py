"""Tests for the packaging guard that keeps offline tooling out of the archive."""

import tarfile
from pathlib import Path

import pytest

from kaggriculture.routes import STORE
from kaggriculture.scripts.package import build

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
