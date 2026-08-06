#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["kaggle>=1.7.4"]
# ///
"""Cache the competition's public kernels and their source.

The public meta moves faster than memory. A survey done on 3 August concluded
that only one published kernel was a real reactive agent; by 6 August the kernel
that survey named had been revised, and a competitor had published a versioned
agent lineage with measured win rates against the top of the ladder. A cache
with a fetch date is the difference between knowing that and assuming it.

Kernel sources are pulled, not just listed, because a title says what somebody
claims and the source says what they did. Anything already cached at the same
version is left alone, so this is safe to run repeatedly and cheap to run often.

Run from cron beside ``fetch_episodes.py``. The shebang makes uv resolve the
dependency above into its own cached environment, so the job does not depend on
the project's virtualenv being present or in sync.
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

LOGGER = logging.getLogger(__name__)

COMPETITION = "kaggriculture"
KERNEL_DIR = Path("/data/kaggriculture/kernels")
PAGES = 4
PAGE_SIZE = 50


def main() -> None:
    """Cache the competition's public kernels from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=KERNEL_DIR, help="cache target")
    parser.add_argument("--pages", type=int, default=PAGES, help="pages to list")
    parser.add_argument(
        "--min-votes", type=int, default=5, help="only pull source at or above this"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args.dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    listed = catalogue(api, args.pages)
    LOGGER.info("listed %d kernels", len(listed))
    write_index(args.dir, listed)

    wanted = [k for k in listed if k["votes"] >= args.min_votes]
    LOGGER.info("pulling source for %d at >= %d votes", len(wanted), args.min_votes)
    for kernel in wanted:
        pull(api, args.dir, kernel)


def catalogue(api: KaggleApi, pages: int) -> list[dict]:
    """Return every listed kernel, most-voted first.

    Args:
        api: An authenticated Kaggle API client.
        pages: How many pages of results to walk.

    Returns:
        One mapping per kernel, carrying the fields the index records.
    """
    found: list[dict] = []
    for page in range(1, pages + 1):
        batch = api.kernels_list(
            competition=COMPETITION,
            sort_by="voteCount",
            page_size=PAGE_SIZE,
            page=page,
        )
        if not batch:
            break
        found += [
            {
                "ref": item.ref,
                "title": item.title,
                "author": item.author,
                "votes": int(item.total_votes),
                "last_run": str(item.last_run_time),
            }
            for item in batch
        ]
    return found


def write_index(directory: Path, kernels: list[dict]) -> None:
    """Write the kernel index, stamped with when it was fetched.

    The stamp is the point. An undated cache invites reasoning from a snapshot
    whose age nobody can recover, which is how a three-day-old survey came to be
    quoted as the current state of the meta.

    Args:
        directory: Where the cache lives.
        kernels: The listing to record.
    """
    index = directory / "index.json"
    index.write_text(
        json.dumps(
            {
                "competition": COMPETITION,
                "fetched": datetime.now(timezone.utc).isoformat(),
                "kernels": kernels,
            },
            indent=2,
        )
    )
    LOGGER.info("wrote %s", index)


def pull(api: KaggleApi, directory: Path, kernel: dict) -> None:
    """Download one kernel's source into the cache.

    Args:
        api: An authenticated Kaggle API client.
        directory: Where the cache lives.
        kernel: One entry from ``catalogue``.
    """
    target = directory / kernel["ref"].replace("/", "__")
    if target.is_dir():
        LOGGER.info("have %s", kernel["ref"])
        return
    target.mkdir(parents=True)
    api.kernels_pull(kernel["ref"], path=str(target), metadata=True)
    LOGGER.info("pulled %s (%d votes)", kernel["ref"], kernel["votes"])


if __name__ == "__main__":
    main()
