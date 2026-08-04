"""Package the agent and submit it to the Kaggriculture competition.

Submitting spends one of the day's submission slots and puts the agent on the
public ladder, so the script asks before uploading unless ``--yes`` is passed.
"""

import argparse
import logging
import subprocess

from kaggriculture.constants import ENVIRONMENT
from kaggriculture.scripts.package import build

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Build the archive and submit it from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("message", help="submission description shown on Kaggle")
    parser.add_argument(
        "--yes", action="store_true", help="skip the confirmation prompt"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    archive = build()
    LOGGER.info("built %s (%.1f KiB)", archive, archive.stat().st_size / 1024)

    if not args.yes:
        answer = input(
            f"Submit to '{ENVIRONMENT}' as {args.message!r}? Uses a daily slot [y/N] "
        )
        if answer.strip().lower() not in {"y", "yes"}:
            LOGGER.info("aborted")
            return

    subprocess.run(
        [
            "kaggle",
            "competitions",
            "submit",
            ENVIRONMENT,
            "-f",
            str(archive),
            "-m",
            args.message,
        ],
        check=True,
    )
    subprocess.run(["kaggle", "competitions", "submissions", ENVIRONMENT], check=True)


if __name__ == "__main__":
    main()
