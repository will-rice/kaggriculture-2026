"""Build the imitation dataset from the replay corpus."""

import argparse
import logging
from pathlib import Path

from kaggriculture.learn.corpus import CORPUS, Sample, select, split
from kaggriculture.learn.dataset import build_shard

LOGGER = logging.getLogger(__name__)

SHARDS = Path("/data/kaggriculture/imitation")

# The build owns the shard naming; `train.py` imports these rather than
# restating them, so the writer and the reader cannot disagree about what a
# shard is called.
TRAIN = "train"
HOLDOUT = "holdout"


def write_dataset(
    directory: Path, train: list[Sample], held: list[Sample], stride: int
) -> None:
    """Clear ``directory`` of a previous build and write both shard families.

    Clearing is bound to writing rather than left as a step a caller must
    remember, because forgetting it is silent: the leftover shards of a longer
    previous build are named like real ones and carry identical shapes, so they
    concatenate into training without complaint.

    Args:
        directory: Where to write the shards.
        train: Seats to encode into the training shards.
        held: Seats to encode into the holdout shards.
        stride: Keep one turn in this many.
    """
    clear_shards(directory)
    build_shard(train, directory / f"{TRAIN}.npz", stride=stride)
    build_shard(held, directory / f"{HOLDOUT}.npz", stride=stride)


def clear_shards(directory: Path) -> None:
    """Delete any shards already in ``directory`` before writing new ones.

    A build writes ``train.npz``, ``train-001.npz`` and so on until it runs out
    of rows, and never deletes. Any rebuild that produces fewer shards than the
    last one -- a higher rating floor, a longer stride, a smaller corpus --
    therefore leaves the tail of the previous build behind, and those files are
    named exactly like real shards. Training would load them, concatenate them
    against matching shapes, and report nothing: a silent mix of two datasets.

    Args:
        directory: The shard directory about to be written.
    """
    stale = sorted(directory.glob("*.npz"))
    for shard in stale:
        shard.unlink()
    if stale:
        LOGGER.info("cleared %d shard(s) from a previous build", len(stale))


def main() -> None:
    """Index, select and encode the corpus from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=SHARDS, help="shard directory")
    parser.add_argument("--stride", type=int, default=4, help="keep one turn in N")
    parser.add_argument("--min-rating", type=float, default=2500.0, help="rating floor")
    parser.add_argument("--per-archive", type=int, default=200, help="cap per archive")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    chosen = select(
        sorted(CORPUS.glob("*.zip")),
        min_rating=args.min_rating,
        per_archive=args.per_archive,
    )
    train, held = split(chosen)
    LOGGER.info(
        "selected %d seats: %d train, %d holdout", len(chosen), len(train), len(held)
    )

    write_dataset(args.out, train, held, args.stride)


if __name__ == "__main__":
    main()
