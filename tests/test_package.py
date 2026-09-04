"""Tests for the packaging guard that keeps offline tooling out of the archive."""

import tarfile
from pathlib import Path

from kaggriculture.scripts.package import build


def test_the_archive_does_not_ship_offline_tooling(tmp_path: Path) -> None:
    """The scripts and campaign packages have no place in a 4 MB agent archive.

    ``campaign`` plays hundreds of games against a league to search for a
    better agent, work the submitted agent never does at play time, and the
    guard also closes the path by which a later edit could reintroduce a heavy
    import behind it.
    """
    archive = build(tmp_path / "submission.tar.gz")

    with tarfile.open(archive) as bundle:
        names = bundle.getnames()

    assert not any(name.startswith("kaggriculture/scripts") for name in names)
    assert not any(name.startswith("kaggriculture/campaign") for name in names)
    assert not any("__pycache__" in name for name in names)


def test_build_accepts_a_self_contained_alternate_entrypoint(tmp_path: Path) -> None:
    """Candidate packaging can be tested without changing the served default."""
    entrypoint = tmp_path / "main.py"
    entrypoint.write_text(
        "def agent(observation, configuration=None):\n"
        "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
    )

    archive = build(tmp_path / "alternate.tar.gz", entrypoint=entrypoint)

    with tarfile.open(archive) as bundle:
        packaged_main = bundle.extractfile("main.py")
        assert packaged_main is not None
        source = packaged_main.read().decode()
    assert source == entrypoint.read_text()
