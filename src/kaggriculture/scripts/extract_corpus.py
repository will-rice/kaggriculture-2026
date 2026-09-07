"""Parse every recorded game once into SQLite, so a question costs a query.

Everything the tapes hold, at the resolution it is held: one row per game,
one per (game, seat, day), one per non-zero holding, one per market order and
one per farmer or hand command. Rebuilt rather than appended to, because a
half-built dataset that looks complete is worse than none.

Aggregate only: this reads public replays and writes counts. No opponent's
source is read and no path of theirs appears in what it writes.

Run from the repository root::

    uv run extract-corpus

Then ask it things::

    sqlite3 /data/kaggriculture/corpus.sqlite
    sqlite> SELECT verb, item, sum(quantity) FROM orders GROUP BY 1, 2;

Re-run it when new archives land. It takes about as long as one pass of
`strategies` and replaces every later pass of anything.
"""

import argparse
import logging

from kaggriculture.campaign import dataset, tapes

LOGGER = logging.getLogger(__name__)
# How many of the ladder's teams to name at the end, and the fewest games a
# team must have played to be one of them. A team that played twice and won
# both is not the strongest agent on the ladder.
SHOWN = 15
LEAST = 30


def main() -> None:
    """Build the dataset, then say what it holds and who is winning."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--since",
        default=tapes.SINCE,
        help="only archives dated this day or later (default: %(default)s, the "
        "day the current engine landed)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=dataset.WORKERS,
        help="processes to divide the archives over (default: %(default)s)",
    )
    parser.add_argument(
        "--database",
        default=dataset.DATABASE,
        help="where to write it (default: %(default)s)",
    )
    arguments = parser.parse_args()

    corpus = tapes.archives(arguments.since)
    LOGGER.info("reading %d archives into %s", len(corpus), arguments.database)
    dataset.build(corpus, arguments.database, arguments.workers)
    LOGGER.info("%s", dataset.summarise(arguments.database))

    LOGGER.info("\n%-40s%8s%8s", "team", "games", "won")
    for team, games, rate in dataset.leaderboard(arguments.database, LEAST)[:SHOWN]:
        LOGGER.info("%-40s%8d%7.0f%%", team[:38], games, 100 * rate)


if __name__ == "__main__":
    main()
