"""Measure every claim in the store against the recorded ladder, and record it.

``winning_pace.md`` says what the ladder's winners held on each day. It does
not say what they did differently, because it never compares the two sides of
one game: a median over winners describes the whole corpus, and "hires every
day" is true of the side that lost as well.

A claim is the other thing. It names a quantity, a day, and which side leads,
and it is measured inside single games -- same map, same prices, same
opponent, one of them lost -- so what is left when the two sides differ is
what the two players did. Confirmed or refuted by the count, never by how
well it reads.

Aggregate only, like ``winning-pace``: this reads public replays and writes
counts, and no opponent's source is read or written.

Every game, not a sample of them. Selecting a claim because it scored highly
over sixty games is the winner's curse in miniature -- and it happened here:
of the eleven claims below, measured over sixty games, eight cleared the bar,
and measured over four hundred only six did. "Winners lead on planted tiles
from the second day" went 76% to 59% and stopped being a claim. Reading a
corner of the corpus is what makes that possible, so this reads all of it and
divides the archives over processes rather than reading fewer games.

Run from the repository root::

    uv run strategies

It appends to ``campaign/strategies.jsonl``, which the round prompt reads.
Run it after each day's archives land: the corpus grows, and a claim confirmed
over four hundred games can still be refuted over four thousand.
"""

import argparse
import logging

from kaggriculture.campaign import paired, strategy, tapes

LOGGER = logging.getLogger(__name__)

# The claims the store starts from: written by hand from a model's reading of
# two paired games, before any of them was measured. They are here rather than
# in a one-off script so that a store deleted by accident can be rebuilt, and
# so that what was originally asserted stays legible next to what the corpus
# said back.
#
# Three of the eleven did not survive contact with the corpus. That is the
# point of keeping them: a claim is a sentence that could be wrong, and a
# store that only ever held the winners would be a list of things that sounded
# right. `propose` matches on the form, so re-running this proposes nothing
# that is already here.
# Each stands alone: a round sees one row of a table without the rows around
# it, and three of these were originally written as continuations of the one
# above ("and decisively richer by day twenty-five"), which says nothing on
# its own.
SEED_CLAIMS: tuple[tuple[str, int, bool, str], ...] = (
    ("planted", 2, True, "winners lead on planted tiles from the second day"),
    ("planted", 20, True, "winners still lead on planted tiles at the midpoint"),
    ("planted", 29, True, "winners are still holding growing tiles at the close"),
    ("bank", 8, False, "winners are poorer on day eight: the money is in tiles"),
    ("bank", 25, True, "winners are decisively richer by day twenty-five"),
    ("hires", 15, False, "winners issue fewer hire orders than the side that loses"),
    ("hands", 16, True, "winners end up holding more hands than the side that loses"),
    ("seeds", 12, False, "winners hold less seed in store; they plant what they buy"),
    ("shed", 12, True, "winners carry more in the shed by day twelve"),
    ("sells", 20, True, "winners sell more by day twenty"),
    ("pens", 10, True, "winners build their pens out earlier"),
)


def main() -> None:
    """Propose anything missing, measure the whole store, print what it says."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--since",
        default=tapes.SINCE,
        help="only archives dated this day or later (default: %(default)s, the "
        "day the current engine landed). Earlier games were played under "
        "different prices and are filtered out by version regardless; this "
        "only decides how much of the archive is opened to find that out",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=paired.WORKERS,
        help="processes to divide the archives over (default: %(default)s)",
    )
    arguments = parser.parse_args()

    store = strategy.Strategies(strategy.STORE)
    for quantity, day, leads, prose in SEED_CLAIMS:
        store.propose(
            strategy.Form(quantity=quantity, day=day, winner_leads=leads), prose, []
        )
    corpus = tapes.archives(arguments.since)
    LOGGER.info(
        "measuring %d claims over every game in %d archives",
        len(store.claims),
        len(corpus),
    )
    paired.measure(store, corpus, arguments.workers)
    report(store)


def report(store: strategy.Strategies) -> None:
    """Print every claim with its evidence, best supported first."""
    ordered = sorted(store.claims, key=lambda claim: (claim.status, -claim.agreement))
    LOGGER.info("%-62s%9s%8s  %s", "claim", "support", "agree", "status")
    for claim in ordered:
        LOGGER.info(
            "%-62s%9d%7.0f%%  %s",
            claim.prose[:60],
            claim.support,
            100 * claim.agreement,
            claim.status,
        )


if __name__ == "__main__":
    main()
