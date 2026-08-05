"""Build the imitation dataset from the replay corpus."""

import argparse
import logging
from pathlib import Path

from kaggriculture.learn.corpus import CORPUS, select, split
from kaggriculture.learn.dataset import build_shard

LOGGER = logging.getLogger(__name__)

SHARDS = Path("/data/kaggriculture/imitation")


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

    build_shard(train, args.out / "train.npz", stride=args.stride)
    build_shard(held, args.out / "holdout.npz", stride=args.stride)


if __name__ == "__main__":
    main()
