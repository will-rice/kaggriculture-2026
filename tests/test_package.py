"""Tests for the packaging guard that keeps offline tooling out of the archive."""

import tarfile
from pathlib import Path

from kaggriculture.scripts.package import build


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
