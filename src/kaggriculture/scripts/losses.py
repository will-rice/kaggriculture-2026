"""Load the games we actually lost into the database a round already queries.

`campaign.losses` carries the reasoning and the work, and the loop calls its
`refresh` on a timer. This exists for when a person wants it now -- straight
after a submission, or to widen how many losses are held.

    uv run losses               the newest submission, the 30 closest losses
    uv run losses --keep 60     more of them
"""

import argparse
import logging

from kaggriculture.campaign import losses

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Fetch our closest real losses and put them in the games database."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep",
        type=int,
        default=losses.KEEP,
        help="how many of the closest losses to hold",
    )
    parser.add_argument(
        "--submission", type=int, default=0, help="which submission, newest if unset"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    loaded = losses.refresh(args.keep, args.submission)
    LOGGER.info("loaded %d new game(s)", loaded)


if __name__ == "__main__":
    main()
