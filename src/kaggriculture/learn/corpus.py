"""Indexing the published replay corpus.

Kaggle publishes one archive of top-rated episodes per day; five are on disk and
a nightly cron fetches more. Each archive holds ~787 episodes of roughly 27 MB,
so the corpus is about 107 GB uncompressed and cannot be unpacked.

Every selection decision needs only two things — what each player banked, and
who played — and both sit at the top level of the episode JSON beside a `steps`
array that is 99% of its bytes. This module reads the small part.
"""

import json
import logging
import zipfile
from pathlib import Path

from pydantic import BaseModel
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

CORPUS = Path("/data/kaggriculture/episodes")


class EpisodeRecord(BaseModel):
    """One episode's identity and outcome, without its steps."""

    archive: str
    name: str
    episode_id: int
    rewards: list[float]
    teams: list[str]


def index_corpus(directory: Path = CORPUS) -> list[EpisodeRecord]:
    """Return a record for every episode in every archive under ``directory``.

    Args:
        directory: Directory holding the daily ``.zip`` archives.

    Returns:
        One ``EpisodeRecord`` per episode, across all archives.
    """
    records: list[EpisodeRecord] = []
    for archive in sorted(directory.glob("*.zip")):
        found = index_archive(archive)
        LOGGER.info("%s: %d episodes", archive.name, len(found))
        records.extend(found)
    return records


def index_archive(archive: Path) -> list[EpisodeRecord]:
    """Return a record for every episode in one archive.

    Reads each member fully but parses it once, discarding the steps
    immediately. Streaming the JSON with a partial parser would be faster still
    and is not worth the dependency: this runs a handful of times, not per
    training step.

    Args:
        archive: Path to a daily ``.zip`` of episode JSON files.

    Returns:
        One ``EpisodeRecord`` per member.
    """
    records: list[EpisodeRecord] = []
    with zipfile.ZipFile(archive) as bundle:
        names = [name for name in bundle.namelist() if name.endswith(".json")]
        for name in tqdm(names, desc=archive.name, unit="ep"):
            with bundle.open(name) as member:
                episode = json.load(member)
            info = episode.get("info", {})
            records.append(
                EpisodeRecord(
                    archive=archive.name,
                    name=name,
                    episode_id=int(info.get("EpisodeId", 0)),
                    rewards=[float(value) for value in episode["rewards"]],
                    teams=[str(team) for team in info.get("TeamNames", [])],
                )
            )
    return records
