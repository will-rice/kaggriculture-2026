"""Write the build order a round is shown: what the strongest agents hold, by day.

This replaces ``winning-pace``, which took medians over the winning side of
every recorded game. That is the wrong half of the corpus. About half of a
ladder's winners are the weaker agent having a good day, and measured that way
eleven quantities over thirteen thousand games all came back between 45% and
60% -- noise dressed as a finding, and three of them were shown to rounds.

Averaged over the top of a fitted rating instead, the same games say something
sharp and early. Between the top twenty-five and the bottom eighty: quadrants
open two to three days sooner, fertilizer is on the ground by day five where
the rest have none, planted tiles lead 31 to 22 by day six, and the bank is
*behind* until day twelve because all of it went into capacity. The weakest
separations in the corpus are all about the endgame.

Aggregate only, like the tables it replaces: this reads the extracted corpus
and writes counts. No opponent's source is read, no agent is named, and no
path of theirs appears in what it writes.

Run from the repository root, after ``uv run extract-corpus``::

    uv run build-order

It rewrites ``campaign/build_order.md``, which the round prompt includes whole.
"""

import argparse
import logging
import sqlite3
from pathlib import Path

from kaggriculture.campaign import dataset, strategy

LOGGER = logging.getLogger(__name__)

TABLE = Path(__file__).resolve().parents[1] / "campaign" / "build_order.md"
# Rendered per column, because a bank in coins and a quadrant count out of four
# do not want the same precision and a table of "3.0" quadrants reads as noise.
PLACES = {"bank": 0, "quadrants": 1, "planted": 1, "fertilised": 1}
HEADINGS = {
    "bank": "bank",
    "quadrants": "quadrants",
    "planted": "planted tiles",
    "fertilised": "fertilised",
    "pens": "animals",
    "hands": "hands",
    "seeds": "seed in store",
    "shed": "shed",
    "weeds": "weeds",
}

OPENING = """### The opening, as orders

What those agents send to the market on each of the first days -- the orders
themselves, not what the orders accumulate to. Counts are medians over their
games, so every line is a whole order that was really sent; a mean would give
fractions of an order, which is how the old table came to ask for 1.2
quadrants.
"""
HOLDINGS = """### What that accumulates to

The same agents' holdings at the end of each day. This is a description of
where the opening above arrives, not a second set of targets: hit the orders
and the holdings follow, while chasing the holdings directly is what could not
be done.
"""
PREAMBLE = """## How the strongest agents build

Averaged over the {games:,} games the public ladder played in the last {window}
days, across the {teams} agents that share the strongest opening on it:
**{opening}**, mean rating {rating:+.2f}.

Those agents rather than the top of the table, because a mean across the top of
the table is a mean across agents doing different things. Measured 2026-09-09:
the top twelve hold 1.16 quadrants on day three, which is 84% of them holding
one and 16% holding two. No agent holds 1.16 quadrants. Averaged, the group
that takes land on day three -- the three best agents on the ladder, 0.9
log-odds clear of fourth -- is blended with the group that waits until day six,
and the number that separates first place from fourth is deleted.

The opening is a property of the agent and not of the game: within one agent the
day it takes its second quadrant varies by a tenth of a day, while between
agents it ranges from three to six. So this is a strategy, and it is followable
in a way the mean was not.

The window is not a sample either: the field turns over completely inside a
fortnight, so a table over the whole corpus describes a blend of fields, most of
which no longer plays. None of these agents publishes a kernel, so their games
are the only view of them there is, and none is in the pool you are scored
against.

Read down a column and you have what the top of the ladder holds on that day.
Every row here is also a column of your own day tables below, so you can put
your number beside theirs directly.

This is not a target to hit and not a rule of the game. It is what the
strongest quarter of a 185-agent field happens to do, and any of it can be
beaten -- but where you differ from it sharply and lose, this is the first
place to look.

"""

# The closing summary is generated from what the corpus currently settles, not
# written down. Written down, it went stale in a day: it asserted that
# fertilizer on the ground by day five was the sharpest separation in the
# corpus at 100% of 313 games, and once the rating was fitted over a ten-day
# window instead of twenty-four days that finding was not in the top eight --
# the field had turned over and the top agents no longer fertilise before day
# twelve. A page whose numbers are live and whose prose is fixed is a page
# that lies slowly.
CLOSING_HEAD = """
What the same games say when the two sides of each are compared directly --
every quantity crossed with every day, and these are the ones that separate
the stronger agent from the weaker most sharply:
"""
SHOWN_CLAIMS = 8


def main() -> None:
    """Rewrite the build-order table from the extracted corpus."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=dataset.DATABASE,
        help="the extracted corpus (default: %(default)s)",
    )
    parser.add_argument(
        "--best",
        type=int,
        default=dataset.BEST,
        help="how many rated agents to read from (default: %(default)s)",
    )
    arguments = parser.parse_args()

    best = dataset.openings(arguments.database, arguments.best)[0]
    order = dataset.build_order(arguments.database, teams=best.teams)
    sent = dataset.opening_orders(arguments.database, teams=best.teams)
    connection = sqlite3.connect(arguments.database)
    try:
        # The games the table actually averages, not the whole corpus: the
        # preamble names this number and it has to be the windowed one.
        games = connection.execute(
            "SELECT count(*) FROM episodes WHERE played >= ?",
            (dataset.recent(connection),),
        ).fetchone()[0]
    finally:
        connection.close()
    settled = strategy.Strategies(strategy.STORE).settled()
    TABLE.write_text(render(order, sent, games, best, settled), encoding="utf-8")
    LOGGER.info("wrote %s over %d games", TABLE, games)


def render(
    order: dict[str, list[float]],
    sent: list[dataset.Order],
    games: int,
    best: dataset.Opening,
    settled: list[strategy.Claim],
) -> str:
    """The table as the round prompt includes it, days across and rows down."""
    days = [int(day) for day in order["day"]]
    lines = [
        PREAMBLE.format(
            games=games,
            teams=len(best.teams),
            opening=best.describe(),
            rating=best.rating,
            window=dataset.WINDOW,
        ).rstrip("\n"),
        "",
        OPENING.rstrip("\n"),
        "",
    ]
    for day in sorted({order.day for order in sent}):
        lines.append(f"**Day {day}**")
        lines += [f"- {order.describe()}" for order in sent if order.day == day]
        lines.append("")
    lines.append(HOLDINGS.rstrip("\n"))
    lines.append("")
    lines.append("| day | " + " | ".join(f"d{day}" for day in days) + " |")
    lines.append("| --- |" + " --- |" * len(days))
    for name in dataset.TARGETS:
        cells = " | ".join(f"{value:,.{PLACES.get(name, 1)}f}" for value in order[name])
        lines.append(f"| {HEADINGS[name]} | {cells} |")
    lines.append(CLOSING_HEAD.rstrip("\n"))
    lines.append("")
    for claim in settled[:SHOWN_CLAIMS]:
        share = claim.agreement if claim.status == "leads" else 1 - claim.agreement
        side = "more" if claim.status == "leads" else "less"
        lines.append(
            f"- **{claim.form.quantity}**, day {claim.form.day}: the stronger "
            f"side holds {side}, in {share:.0%} of {claim.support:,} games "
            f"where the two differed."
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
