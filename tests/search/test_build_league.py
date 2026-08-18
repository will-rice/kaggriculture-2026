"""The league is built from the strongest seats of the strongest episodes."""

from pathlib import Path

import pytest

from kaggriculture.search.scripts.build_league import harvest

ARCHIVE = Path("/data/kaggriculture/episodes/kaggriculture-episodes-2026-08-15.zip")


@pytest.mark.skipif(not ARCHIVE.exists(), reason="replay corpus not on this machine")
def test_harvest_writes_one_route_per_requested_opponent(tmp_path: Path) -> None:
    """Each harvested file is a 720-turn route the arena can load."""
    from kaggriculture.search.route import load

    written = harvest(ARCHIVE, count=2, output=tmp_path)

    assert len(written) == 2
    for path in written:
        assert len(load(path)) == 720
