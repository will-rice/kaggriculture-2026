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

from kaggriculture.campaign import config, dataset, tapes

LOGGER = logging.getLogger(__name__)
# How many of the ladder's teams to name at the end.
SHOWN = 25
LEAST = dataset.LEAST


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
        default=config.GAMES_DB,
        help="where to write it (default: %(default)s)",
    )
    arguments = parser.parse_args()

    corpus = tapes.archives(arguments.since)
    LOGGER.info("reading %d archives into %s", len(corpus), arguments.database)
    dataset.build(corpus, arguments.database, arguments.workers)
    rated = dataset.rate(arguments.database, LEAST)
    LOGGER.info("%s", dataset.summarise(arguments.database))

    LOGGER.info("\n%4s  %-38s%8s%8s%9s", "#", "team", "games", "won", "rating")
    for team, games, wins, value, place in dataset.ladder(arguments.database)[:SHOWN]:
        LOGGER.info(
            "%4d  %-38s%8d%7.0f%%%+9.3f",
            place,
            team[:36],
            games,
            100 * wins / games,
            value,
        )
    LOGGER.info("\n%d teams rated of %d that played", rated, len(corpus) and rated)


if __name__ == "__main__":
    main()
