"""Streaming selected episodes into sharded training data.

The corpus cannot be unpacked, so episodes are read one at a time straight from
their archive, encoded, and appended to a shard. Turns are subsampled: 720 turns
of one season are far more correlated than they are informative, and a stride
buys diversity per byte.
"""

import json
import logging
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from kaggriculture.learn.corpus import CORPUS, Sample
from kaggriculture.learn.encoding import encode_board, encode_scalars, encode_units

LOGGER = logging.getLogger(__name__)


def build_shard(samples: list[Sample], destination: Path, stride: int = 4) -> int:
    """Encode every ``stride``-th turn of each sample into one ``.npz`` shard.

    ``MAX_UNITS`` is an empirical bound from sampling part of the corpus, not a
    proven one. A turn whose acting-unit count exceeds it is skipped rather than
    aborting the whole build; the skip is counted and logged so an unexpectedly
    non-zero count surfaces as a finding about the bound, not a silent loss.

    Args:
        samples: Demonstrations to encode.
        destination: Shard path to write.
        stride: Keep one turn in this many.

    Returns:
        How many rows were written.
    """
    boards: list[np.ndarray] = []
    scalars: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    by_archive: dict[str, list[Sample]] = {}
    for sample in samples:
        by_archive.setdefault(sample.archive, []).append(sample)

    skipped = 0
    for archive, group in by_archive.items():
        with zipfile.ZipFile(CORPUS / archive) as bundle:
            for sample in tqdm(group, desc=archive, unit="ep"):
                with bundle.open(sample.name) as member:
                    episode = json.load(member)
                skipped += _encode_episode(
                    episode, sample, stride, boards, scalars, labels
                )

    LOGGER.info("skipped %d turns for exceeding MAX_UNITS", skipped)
    np.savez_compressed(
        destination,
        boards=np.concatenate(boards) if boards else np.empty((0,)),
        scalars=np.concatenate(scalars) if scalars else np.empty((0,)),
        labels=np.concatenate(labels) if labels else np.empty((0,)),
    )
    LOGGER.info("%s: %d rows", destination.name, len(boards))
    return len(boards)


def _encode_episode(
    episode: dict[str, Any],
    sample: Sample,
    stride: int,
    boards: list[np.ndarray],
    scalars: list[np.ndarray],
    labels: list[np.ndarray],
) -> int:
    """Append one seat's encoded turns to the accumulating lists.

    Returns:
        How many turns of this episode were skipped for exceeding ``MAX_UNITS``.
    """
    skipped = 0
    for index in range(0, len(episode["steps"]), stride):
        entry = episode["steps"][index][sample.seat]
        action = entry.get("action")
        if not action:
            continue
        try:
            label = encode_units(action)
        except ValueError as error:
            skipped += 1
            LOGGER.warning(
                "skipping turn %d of %s (seat %d): %s",
                index,
                sample.name,
                sample.seat,
                error,
            )
            continue
        observation = entry["observation"]
        boards.append(encode_board(observation, sample.seat).numpy(force=True))
        scalars.append(encode_scalars(observation, sample.seat).numpy(force=True))
        labels.append(label.numpy(force=True))
    return skipped


class Shards(Dataset):
    """A dataset over one or more ``.npz`` shards, held in memory."""

    def __init__(self, paths: list[Path]) -> None:
        """Load every shard named by ``paths``."""
        boards, scalars, labels = [], [], []
        for path in paths:
            with np.load(path) as data:
                boards.append(data["boards"])
                scalars.append(data["scalars"])
                labels.append(data["labels"])
        self.boards = torch.from_numpy(np.concatenate(boards))
        self.scalars = torch.from_numpy(np.concatenate(scalars))
        self.labels = torch.from_numpy(np.concatenate(labels))

    def __len__(self) -> int:
        """Return how many rows this dataset holds."""
        return int(self.boards.shape[0])

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one row's board, scalars and labels, without batch dimensions."""
        return self.boards[index], self.scalars[index], self.labels[index]
