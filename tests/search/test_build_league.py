"""The league is built from the strongest seats of the strongest episodes."""

import json
import re
import zipfile
from pathlib import Path

import pytest

from kaggriculture.search.scripts.build_league import harvest

ARCHIVE = Path("/data/kaggriculture/episodes/kaggriculture-episodes-2026-08-15.zip")


@pytest.mark.skipif(not ARCHIVE.exists(), reason="replay corpus not on this machine")
def test_harvest_writes_one_route_per_requested_opponent(tmp_path: Path) -> None:
    """Each harvested file is a 720-turn route the arena can load."""
    from kaggriculture.search.route import load

    written = harvest(ARCHIVE, count=12, output=tmp_path)

    assert len(written) == 12
    with zipfile.ZipFile(ARCHIVE) as bundle:
        for path in written:
            # Parse episode_id and seat from filename
            match = re.search(r"-(\d+)-seat(\d)\.json$", str(path))
            assert match, f"Could not parse episode_id and seat from {path}"
            episode_id = match.group(1)
            seat = int(match.group(2))

            # Verify the route loads and has correct length
            route = load(path)
            assert len(route) == 720

            # Verify the episode has the correct engine version
            episode = json.loads(bundle.read(f"{episode_id}.json"))
            assert episode.get("module_version") == "1.32.7"

            # Verify the harvested seat is the winner (strictly higher rewards)
            rewards = episode.get("rewards")
            assert isinstance(rewards, list) and len(rewards) == 2
            if seat == 0:
                assert rewards[0] > rewards[1]
            else:
                assert rewards[1] > rewards[0]
