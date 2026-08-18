"""Harvest league opponents from the published episode archives.

The winning seat of a top-rated episode is a route by construction: the archive
records every action it took. Harvesting several days rather than one is
deliberate -- Kaito Fukami's own advice is that optimising against the latest
Top-30 alone loses to older meta generations still active on the ladder.
"""

import argparse
import json
import logging
import zipfile
from pathlib import Path

from kaggriculture.learn.corpus import read_manifest
from kaggriculture.search.route import from_episode, save

LOGGER = logging.getLogger(__name__)
ENGINE = "1.32.7"


def harvest(archive: Path, count: int, output: Path) -> list[Path]:
    """Write the winning seats of the archive's best episodes as routes.

    The winning seat is the one with strictly higher rewards; ties award to
    seat 0 as a consistent break.

    Args:
        archive: A daily episode ``.zip``.
        count: How many opponents to write.
        output: Directory to write them into.

    Returns:
        The paths written, best episode first.

    Raises:
        ValueError: If an episode lacks valid rewards.
    """
    output.mkdir(parents=True, exist_ok=True)
    rows = sorted(read_manifest(archive), key=lambda row: -row.avg_score)
    written: list[Path] = []
    with zipfile.ZipFile(archive) as bundle:
        for row in rows:
            if len(written) >= count:
                break
            episode = json.loads(bundle.read(f"{row.episode_id}.json"))
            if episode.get("module_version") != ENGINE:
                continue
            rewards = episode.get("rewards")
            if not isinstance(rewards, list) or len(rewards) != 2:
                raise ValueError(f"Episode {row.episode_id} missing or invalid rewards")
            seat = 0 if rewards[0] >= rewards[1] else 1
            path = output / f"{archive.stem}-{row.episode_id}-seat{seat}.json"
            save(from_episode(episode, seat), path)
            written.append(path)
            LOGGER.info(
                "%s rating %.0f bank %s", path.name, row.avg_score, rewards[seat]
            )
    return written


def main() -> None:
    """Harvest a league from one or more daily archives."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="directory for league routes")
    parser.add_argument("archives", type=Path, nargs="+", help="daily episode zips")
    parser.add_argument("--per-archive", type=int, default=3)
    arguments = parser.parse_args()
    for archive in arguments.archives:
        harvest(archive, arguments.per_archive, arguments.output)


if __name__ == "__main__":
    main()
