"""Selecting demonstrations from the published replay corpus.

Kaggle publishes one archive of top-rated episodes per day; six are on disk and
a nightly cron fetches more. Each archive holds ~787 episodes of roughly 27 MB,
so the corpus is about 107 GB uncompressed and cannot be unpacked.

Every archive ships a ``manifest.csv`` beside its episodes carrying each
episode's ladder ratings. That is enough to select which episodes are worth
cloning without ever decoding an episode's JSON.
"""

import csv
import io
import zipfile
from pathlib import Path

from pydantic import BaseModel

CORPUS = Path("/data/kaggriculture/episodes")


class ManifestRow(BaseModel):
    """One episode's entry in an archive's manifest."""

    episode_id: int
    avg_score: float
    min_score: float
    agent_count: int


def read_manifest(archive: Path) -> list[ManifestRow]:
    """Return the manifest rows for one archive.

    Each archive ships a ``manifest.csv`` beside its episodes carrying the
    players' ladder ratings. Reading it is instant, where establishing the same
    thing from the episodes themselves means decoding 21 GB of JSON.

    Args:
        archive: Path to a daily ``.zip``.

    Returns:
        One row per episode, in the manifest's own order.
    """
    with zipfile.ZipFile(archive) as bundle:
        text = bundle.read("manifest.csv").decode()
    return [
        ManifestRow(
            episode_id=int(entry["episode_id"]),
            avg_score=float(entry["avg_score"]),
            min_score=float(entry["min_score"]),
            agent_count=int(entry["agent_count"]),
        )
        for entry in csv.DictReader(io.StringIO(text))
    ]


class Sample(BaseModel):
    """One seat of one episode, as a demonstration to clone."""

    archive: str
    name: str
    seat: int
    rating: float


def select(
    archives: list[Path],
    min_rating: float = 2500.0,
    per_archive: int = 200,
) -> list[Sample]:
    """Return the seats worth cloning, strongest episodes first.

    Selection is on ``min_score`` — the weaker of the two players' ratings — so
    a chosen episode had two strong players in it. A large bank against a weak
    opponent is not a demonstration of strong play, and cloning one teaches
    behaviour that only works against someone who cannot respond.

    The rating floor also removes the recorded tape and its re-wrappings, which
    make up three-quarters of the field and rate around 1720, without having to
    identify them by name.

    Args:
        archives: Daily archives to select from.
        min_rating: Keep episodes whose weaker player rated at least this.
        per_archive: Cap on episodes taken from any one archive.

    Returns:
        Both seats of each chosen episode, strongest first.
    """
    chosen: list[Sample] = []
    for archive in archives:
        rows = [row for row in read_manifest(archive) if row.min_score >= min_rating]
        rows.sort(key=lambda row: -row.min_score)
        for row in rows[:per_archive]:
            for seat in range(row.agent_count):
                chosen.append(
                    Sample(
                        archive=archive.name,
                        name=f"{row.episode_id}.json",
                        seat=seat,
                        rating=row.min_score,
                    )
                )
    return chosen


def split(
    samples: list[Sample], holdout: float = 0.1
) -> tuple[list[Sample], list[Sample]]:
    """Split samples into training and holdout, never sharing an episode.

    Both seats of one episode see the same board, the same weeds and the same
    market. Splitting by seat would put near-identical states on both sides and
    turn validation accuracy into a memorisation score, so the split is by
    episode.

    Args:
        samples: Selected demonstrations.
        holdout: Fraction of episodes to hold out.

    Returns:
        Training samples and holdout samples.
    """
    episodes = sorted({sample.name for sample in samples})
    cut = int(len(episodes) * (1.0 - holdout))
    training = set(episodes[:cut])
    return (
        [sample for sample in samples if sample.name in training],
        [sample for sample in samples if sample.name not in training],
    )
