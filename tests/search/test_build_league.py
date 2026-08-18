"""The league is built from the strongest seats of the strongest episodes."""

import json
import logging
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
            assert episode.get("module_version") == ENGINE

            # Verify the harvested seat is the winner; ties award to seat 0,
            # per `harvest`'s documented tie-break.
            rewards = episode.get("rewards")
            assert isinstance(rewards, list) and len(rewards) == 2
            if seat == 0:
                assert rewards[0] >= rewards[1]
            else:
                assert rewards[1] > rewards[0]


def _write_archive_with_rewards(path: Path, episode_id: int, rewards: object) -> None:
    """Write a minimal daily archive whose single episode carries ``rewards``."""
    steps = [
        [{"action": {"farmer": ["PASS"], "hands": [], "market": []}} for _ in range(2)]
        for _ in range(720)
    ]
    episode = {"module_version": ENGINE, "rewards": rewards, "steps": steps}
    manifest = (
        f"episode_id,avg_score,min_score,agent_count\n{episode_id},100.0,90.0,2\n"
    )
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("manifest.csv", manifest)
        bundle.writestr(f"{episode_id}.json", json.dumps(episode))


def test_harvest_raises_on_a_null_reward(tmp_path: Path) -> None:
    """``[None, 823]`` passes the 2-element check but is not a real score.

    ``rewards[0] >= rewards[1]`` would otherwise raise a bare ``TypeError``
    mid-harvest instead of the documented ``ValueError`` naming the episode.
    """
    archive = tmp_path / "day.zip"
    _write_archive_with_rewards(archive, episode_id=7, rewards=[None, 823])

    with pytest.raises(ValueError, match="7"):
        harvest(archive, count=1, output=tmp_path / "league")


def _write_short_archive(path: Path, episode_id: int, turns: int) -> None:
    """Write a minimal daily archive whose single episode ended early."""
    steps = [
        [{"action": {"farmer": ["PASS"], "hands": [], "market": []}} for _ in range(2)]
        for _ in range(turns)
    ]
    episode = {"module_version": ENGINE, "rewards": [10, 5], "steps": steps}
    manifest = (
        f"episode_id,avg_score,min_score,agent_count\n{episode_id},100.0,90.0,2\n"
    )
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("manifest.csv", manifest)
        bundle.writestr(f"{episode_id}.json", json.dumps(episode))


def test_harvest_skips_a_short_episode_with_a_log_line(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An early-terminated episode must not poison the league.

    ``route.load`` demands exactly 720 turns, so saving a short episode
    unchecked would make every later ``hillclimb`` run die at startup the
    first time it loads that file.
    """
    archive = tmp_path / "day.zip"
    _write_short_archive(archive, episode_id=42, turns=10)

    with caplog.at_level(logging.INFO):
        written = harvest(archive, count=1, output=tmp_path / "league")

    assert written == []
    assert any("42" in message for message in caplog.messages)


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
