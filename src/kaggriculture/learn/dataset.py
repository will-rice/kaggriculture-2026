"""Streaming selected episodes into sharded training data.

The corpus cannot be unpacked, so episodes are read one at a time straight from
their archive, encoded, and appended to a shard. Turns are subsampled: 720 turns
of one season are far more correlated than they are informative, and a stride
buys diversity per byte.

``kaggle_environments``' interpreter mutates the state object it is handed
alongside the action being applied (``core.py``'s ``step``: ``action_state[index]
= {**self.state[index], "action": None}`` is built from the *previous* state,
then mutated in place by the interpreter and appended as the *next* recorded
step). So a recorded ``steps[i]`` holds the world *after* ``steps[i]["action"]``
ran, not before it. The state an agent actually saw when it chose that action
is ``steps[i - 1]["observation"]``. Every row here therefore pairs
``observation[i]`` with ``action[i + 1]`` -- the decision made *from* that
state -- and ``action[0]`` is dropped as a reset filler with nothing preceding
it to predict from.
"""

import json
import logging
import zipfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from kaggriculture.learn.corpus import CORPUS, Sample
from kaggriculture.learn.encoding import (
    TooManyUnitsError,
    encode_board,
    encode_scalars,
    encode_units,
)

LOGGER = logging.getLogger(__name__)

# One row is ~13.6 KB of board planes plus a few hundred bytes of scalars and
# labels. At this cap a shard is ~270 MB of raw float32 tensors before
# compression -- small enough that peak memory during a build is one shard,
# not the ~5.5 GiB the full corpus would concatenate to at stride=4.
ROWS_PER_SHARD = 20_000

_Row = tuple[np.ndarray, np.ndarray, np.ndarray]


def build_shard(samples: list[Sample], destination: Path, stride: int = 4) -> int:
    """Encode every ``stride``-th turn of each sample into one or more ``.npz`` shards.

    Rows are flushed to disk as soon as ``ROWS_PER_SHARD`` is reached, so a
    build spanning the whole corpus never holds more than one shard's worth of
    tensors in memory at once. The first shard is written to ``destination``
    itself; later ones suffix its stem with a shard index, so a build small
    enough to fit in one shard -- every test in this module -- sees exactly the
    file it asked for.

    ``MAX_UNITS`` is an empirical bound from sampling part of the corpus, not a
    proven one. A turn whose acting-unit count exceeds it is skipped rather than
    aborting the whole build; the skip is counted and logged so an unexpectedly
    non-zero count surfaces as a finding about the bound, not a silent loss. Any
    other encoding failure -- an op outside ``UNIT_OPS``, say -- is not this
    kind of known, bounded risk and propagates instead of being swallowed.

    Args:
        samples: Demonstrations to encode.
        destination: Path for the first (or only) shard.
        stride: Keep one turn in this many.

    Returns:
        How many rows were written in total, across every shard.
    """
    boards: list[np.ndarray] = []
    scalars: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    shard_index = 0
    written = 0

    def flush() -> None:
        nonlocal shard_index, written
        if not boards:
            return
        path = (
            destination if shard_index == 0 else _shard_path(destination, shard_index)
        )
        np.savez_compressed(
            path,
            boards=np.concatenate(boards),
            scalars=np.concatenate(scalars),
            labels=np.concatenate(labels),
        )
        LOGGER.info("%s: %d rows", path.name, len(boards))
        written += len(boards)
        shard_index += 1
        boards.clear()
        scalars.clear()
        labels.clear()

    by_archive: dict[str, list[Sample]] = {}
    for sample in samples:
        by_archive.setdefault(sample.archive, []).append(sample)

    skipped = 0
    for archive, group in by_archive.items():
        with zipfile.ZipFile(CORPUS / archive) as bundle:
            for sample in tqdm(group, desc=archive, unit="ep"):
                with bundle.open(sample.name) as member:
                    episode = json.load(member)
                for row in _encode_episode(episode, sample, stride):
                    if row is None:
                        skipped += 1
                        continue
                    board, scalar, label = row
                    boards.append(board)
                    scalars.append(scalar)
                    labels.append(label)
                    if len(boards) >= ROWS_PER_SHARD:
                        flush()

    flush()
    LOGGER.info("skipped %d turns for exceeding MAX_UNITS", skipped)
    return written


def _shard_path(destination: Path, index: int) -> Path:
    """Return the numbered path for one shard of a multi-shard build.

    Args:
        destination: The path passed to ``build_shard``.
        index: This shard's position; 0 is always ``destination`` itself.

    Returns:
        ``destination`` unchanged for shard 0, otherwise its stem suffixed
        with the shard index.
    """
    if index == 0:
        return destination
    return destination.with_name(f"{destination.stem}-{index:03d}{destination.suffix}")


def _encode_episode(
    episode: dict[str, Any], sample: Sample, stride: int
) -> Iterator[_Row | None]:
    """Yield one seat's encoded turns, one at a time.

    Row ``index`` pairs ``observation[index]`` with ``action[index + 1]``, the
    decision made *from* that state -- see the module docstring for why the
    same-index pairing is wrong. ``action[0]`` is never read as a label, and the
    final observation is never read at all, since neither has a following
    action.

    Yields:
        A row's ``(board, scalars, labels)`` arrays, or ``None`` for a turn
        skipped because it exceeded ``MAX_UNITS``.
    """
    steps = episode["steps"]
    for index in range(0, len(steps) - 1, stride):
        action = steps[index + 1][sample.seat].get("action")
        if not action:
            continue
        try:
            label = encode_units(action)
        except TooManyUnitsError as error:
            LOGGER.warning(
                "skipping turn %d of %s (seat %d): %s",
                index,
                sample.name,
                sample.seat,
                error,
            )
            yield None
            continue
        observation = steps[index][sample.seat]["observation"]
        board = encode_board(observation, sample.seat).numpy(force=True)
        scalar = encode_scalars(observation, sample.seat).numpy(force=True)
        yield board, scalar, label.numpy(force=True)


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
