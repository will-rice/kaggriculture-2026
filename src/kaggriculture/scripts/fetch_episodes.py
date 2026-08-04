"""Download any daily Kaggriculture replay datasets we do not have yet.

Kaggle publishes one dataset of top-rated episodes per day and an index dataset
listing them. This reads the index and fetches whatever is missing locally, so it
is safe to run repeatedly and safe to miss a day. It runs unattended from cron,
so it uses the Kaggle Python API rather than the CLI — cron's minimal PATH does
not reach the virtualenv's scripts.
"""

import argparse
import csv
import logging
import tempfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

LOGGER = logging.getLogger(__name__)

INDEX_DATASET = "kaggle/kaggriculture-episodes-index"
EPISODE_DIR = Path("/data/kaggriculture/episodes")


def main() -> None:
    """Fetch missing daily replay datasets from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=EPISODE_DIR, help="download target")
    parser.add_argument("--days", type=int, default=0, help="only the newest N days (0 = all)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    published = read_index(api)
    if args.days:
        published = published[-args.days :]
    missing = [slug for slug in published if not archive_path(args.dir, slug).exists()]
    LOGGER.info("index lists %d days, %d missing locally", len(published), len(missing))

    for slug in missing:
        download(api, slug, args.dir)
    LOGGER.info("done; %d archives present", len(list(args.dir.glob("*.zip"))))


def read_index(api: KaggleApi) -> list[str]:
    """Return the daily dataset slugs listed in the index, oldest first."""
    with tempfile.TemporaryDirectory() as staging:
        api.dataset_download_files(INDEX_DATASET, path=staging, unzip=True)
        manifest = Path(staging) / "manifest.csv"
        rows = list(csv.DictReader(manifest.read_text().splitlines()))
    return [row["daily_dataset_slug"] for row in sorted(rows, key=lambda row: row["date"])]


def archive_path(directory: Path, slug: str) -> Path:
    """Return where a daily dataset's archive lives once downloaded."""
    return directory / f"{slug}.zip"


def download(api: KaggleApi, slug: str, directory: Path) -> None:
    """Download one daily dataset, leaving it zipped."""
    LOGGER.info("downloading %s", slug)
    api.dataset_download_files(f"kaggle/{slug}", path=str(directory), unzip=False)
    archive = archive_path(directory, slug)
    if archive.exists():
        LOGGER.info("%s -> %.0f MiB", archive.name, archive.stat().st_size / 1024**2)
    else:
        LOGGER.error("%s did not produce %s", slug, archive.name)


if __name__ == "__main__":
    main()
