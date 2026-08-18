"""The league is built from the strongest seats of the strongest episodes."""

import json
import re
import zipfile
from pathlib import Path

import pytest

from kaggriculture.search.scripts.build_league import ENGINE, harvest

ARCHIVE = Path("/data/kaggriculture/episodes/kaggriculture-episodes-2026-08-15.zip")


def _write_archive(path: Path, episode_id: int) -> None:
    """Write a minimal, synthetic daily archive with one harvestable episode."""
    steps = [
        [{"action": {"farmer": ["PASS"], "hands": [], "market": []}} for _ in range(2)]
        for _ in range(720)
    ]
    episode = {"module_version": ENGINE, "rewards": [10, 5], "steps": steps}
    manifest = (
        f"episode_id,avg_score,min_score,agent_count\n{episode_id},100.0,90.0,2\n"
    )
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("manifest.csv", manifest)
        bundle.writestr(f"{episode_id}.json", json.dumps(episode))


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


def test_harvest_deduplicates_an_episode_seen_in_an_earlier_archive(
    tmp_path: Path,
) -> None:
    """An episode in two daily archives must not double-weight its opponent."""
    archive_a = tmp_path / "day-a.zip"
    archive_b = tmp_path / "day-b.zip"
    _write_archive(archive_a, episode_id=1)
    _write_archive(archive_b, episode_id=1)
    output = tmp_path / "league"

    seen: set[int] = set()
    written_a = harvest(archive_a, count=3, output=output, seen=seen)
    written_b = harvest(archive_b, count=3, output=output, seen=seen)

    assert len(written_a) == 1
    assert len(written_b) == 0
