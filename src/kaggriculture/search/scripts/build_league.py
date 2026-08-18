"""Harvest league opponents from the published episode archives.

The winning seat of a top-rated episode is a route by construction: the archive
records every action it took. Harvesting several days rather than one is
deliberate -- Kaito Fukami's own advice is that optimising against the latest
Top-30 alone loses to older meta generations still active on the ladder.

``ENGINE`` is read from the installed ``kaggle-environments`` package rather
than hardcoded, because it is the engine the arena replays on -- deriving the
pin keeps the harvested league and the arena that scores it on the same
build by construction. Two mid-competition engine bumps have already
happened; a hardcoded pin would silently skip every episode in the next one
and build a stale or empty league.
"""

import argparse
import importlib.metadata
import json
import logging
import zipfile
from pathlib import Path

from kaggriculture.constants import EPISODE_STEPS
from kaggriculture.learn.corpus import read_manifest
from kaggriculture.search.route import from_episode, save

LOGGER = logging.getLogger(__name__)
ENGINE = importlib.metadata.version("kaggle-environments")


def harvest(
    archive: Path, count: int, output: Path, seen: set[int] | None = None
) -> list[Path]:
    """Write the winning seats of the archive's best episodes as routes.

    The winning seat is the one with strictly higher rewards; ties award to
    seat 0 as a consistent break.

    Args:
        archive: A daily episode ``.zip``.
        count: How many opponents to write.
        output: Directory to write them into.
        seen: Episode ids already harvested, skipped here and updated in
            place. The same episode can appear in more than one daily
            archive; passing the same set across archives de-duplicates it so
            it does not write two files and double-weight that opponent in
            the league. Omit it to de-duplicate only within this archive.

    Returns:
        The paths written, best episode first.

    Raises:
        ValueError: If an episode lacks valid rewards.
    """
    if seen is None:
        seen = set()
    output.mkdir(parents=True, exist_ok=True)
    rows = sorted(read_manifest(archive), key=lambda row: -row.avg_score)
    written: list[Path] = []
    with zipfile.ZipFile(archive) as bundle:
        for row in rows:
            if len(written) >= count:
                break
            if row.episode_id in seen:
                continue
            episode = json.loads(bundle.read(f"{row.episode_id}.json"))
            if episode.get("module_version") != ENGINE:
                continue
            rewards = episode.get("rewards")
            if (
                not isinstance(rewards, list)
                or len(rewards) != 2
                or not all(isinstance(value, (int, float)) for value in rewards)
            ):
                raise ValueError(f"Episode {row.episode_id} missing or invalid rewards")
            seat = 0 if rewards[0] >= rewards[1] else 1
            route = from_episode(episode, seat)
            seen.add(row.episode_id)
            if len(route) != EPISODE_STEPS:
                LOGGER.info(
                    "episode %s: skipping, %d turns (expected %d) -- an "
                    "early-terminated episode is bad league material anyway",
                    row.episode_id,
                    len(route),
                    EPISODE_STEPS,
                )
                continue
            path = output / f"{archive.stem}-{row.episode_id}-seat{seat}.json"
            save(route, path)
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
    seen: set[int] = set()
    for archive in arguments.archives:
        harvest(archive, arguments.per_archive, arguments.output, seen)


if __name__ == "__main__":
    main()
