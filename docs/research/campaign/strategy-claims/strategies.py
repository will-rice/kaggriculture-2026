"""Ask the corpus every question in the grammar, and keep what it answers.

The build order this replaced said what the strongest agents hold on each
day. It could not say what the strong agents do *differently*, because it
never compared two sides of one game -- and a median over winners describes
the whole corpus, since "hires every day" is true of the side that lost as
well.

This compares, and it compares the right two sides. Within one game, so the
map, the prices and the opponent are shared and what is left is what the two
players did. Between the stronger and the weaker *agent* by fitted rating,
not between the winner and the loser, because half of every ladder's winners
are the weaker side having a good day -- measured that way, eleven claims
over thirteen thousand games all came back between 45% and 60%.

Nothing is proposed by hand. Every quantity the dataset measures, crossed
with every day of the season, is put to the corpus and whatever settles is
what we have learnt. The eleven hand-written claims this replaces were a
model's plausible sentences, graded by the person who wrote them; the corpus
kept one.

Run from the repository root, after ``uv run extract-corpus``::

    uv run strategies

It appends to ``campaign/strategies.jsonl``, which the round prompt reads.
Re-run it when new archives land: a claim settled over one corpus can come
undone over a bigger one, and the log keeps both readings.
"""

import argparse
import logging

from kaggriculture.campaign import config, evidence, strategy

LOGGER = logging.getLogger(__name__)
# How many settled claims to print, strongest separation first.
SHOWN = 40


def main() -> None:
    """Enumerate the grammar, measure all of it, print what settled."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        default=config.GAMES_DB,
        help="the extracted corpus (default: %(default)s)",
    )
    arguments = parser.parse_args()

    store = strategy.Strategies(strategy.STORE)
    forms = evidence.questions()
    for form in forms:
        store.propose(form)
    LOGGER.info("%d questions over %d quantities", len(forms), len(strategy.QUANTITIES))

    evidence.measure(store, arguments.database)
    settled = store.settled()
    LOGGER.info(
        "\n%d of %d settled at %d games or more",
        len(settled),
        len(store.claims),
        strategy.SUPPORT,
    )
    LOGGER.info("\n%-14s%6s%10s%9s%8s", "quantity", "day", "stronger", "games", "share")
    for claim in settled[:SHOWN]:
        share = claim.agreement if claim.status == "leads" else 1 - claim.agreement
        LOGGER.info(
            "%-14s%6d%10s%9d%7.0f%%",
            claim.form.quantity,
            claim.form.day,
            "more" if claim.status == "leads" else "less",
            claim.support,
            100 * share,
        )


if __name__ == "__main__":
    main()
