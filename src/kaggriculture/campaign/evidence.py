"""Put every claim to the corpus at once, as one query per day.

This replaces a walk of the archives that took twenty-odd minutes and
answered one shape of question. The dataset has already parsed the tapes, so
what is left is a self-join: both sides of the same game on the same day,
with each side's rating beside it.

Both halves of the join matter and each fixes a different error.

The *pairing* is what makes a difference attributable. Two rows from the same
episode and the same day shared a map, a price series and an opponent, so
what separates them is what the two players did. Comparing corpus averages
instead would compare farms that never met, on different maps, at different
prices.

The *rating gap* is what makes it about strong play. Comparing the winner to
the loser of each game sounds equivalent and is not: on any ladder about half
the winners are the weaker agent having a good day, and that noise swamps
everything. Measured winner-against-loser, eleven claims over thirteen
thousand games came back between 45% and 60%. Measured stronger-against-
weaker, the same corpus separates the sides cleanly -- and reverses the sign
of at least one of them.
"""

import logging
import sqlite3

from kaggriculture.campaign import config, dataset, strategy

LOGGER = logging.getLogger(__name__)

# How far apart two ratings must be before the game counts as a comparison
# between a stronger and a weaker agent. Adjacent agents differ by less than
# the noise in their own ratings, so a pairing between them tells us which of
# two similar players had the better day -- which is the question that
# produced fifty percent everywhere.
#
# The fitted field spans about three log points, so half of one is a real gap
# and still leaves most pairings usable.
MARGIN = 0.5


def measure(
    store: strategy.Strategies, database: str = config.GAMES_DB
) -> dict[str, tuple[int, float]]:
    """Measure every claim in ``store`` against the corpus and record it.

    One query per day rather than one per claim: the join is what costs, and
    every claim about the same day reads the same joined rows. So the store
    can hold every quantity crossed with every day -- some hundreds of
    questions -- for thirty queries.

    Args:
        store: The claims to measure. Every one, the settled included: a
            claim that held over a thousand games and fails over ten thousand
            is the case this exists to catch.
        database: The dataset, already built and rated.

    Returns:
        ``{claim id: (support, agreement)}`` for every claim any game spoke
        to, where ``agreement`` is the share in which the stronger side had
        more of the quantity.
    """
    wanted: dict[int, list[strategy.Claim]] = {}
    for claim in store.claims:
        wanted.setdefault(claim.form.day, []).append(claim)
    if not wanted:
        return {}
    connection = sqlite3.connect(database)
    out: dict[str, tuple[int, float]] = {}
    try:
        # The same window the ratings were fitted over. A claim measured across
        # the whole corpus is measured over two disjoint fields: split in half,
        # the two top tens share not one name.
        first = dataset.recent(database)
        for day, claims in sorted(wanted.items()):
            quantities = sorted({claim.form.quantity for claim in claims})
            counted = _day(connection, day, quantities, first)
            for claim in claims:
                ahead, seen = counted[claim.form.quantity]
                if not seen:
                    continue
                store.record(claim.id, seen, ahead / seen)
                out[claim.id] = (seen, ahead / seen)
            LOGGER.info("day %d: %d quantities over the corpus", day, len(quantities))
    finally:
        connection.close()
    return out


def _day(
    connection: sqlite3.Connection, day: int, quantities: list[str], first: str
) -> dict[str, tuple[int, int]]:
    """For one day, how often the stronger side had more of each quantity.

    Args:
        connection: The dataset.
        day: The day both sides are read at.
        quantities: Day-row columns to compare.
        first: The earliest archive day to read, so a claim is settled on the
            field as it now plays rather than on one that has turned over.

    Returns:
        ``{quantity: (ahead, differing)}`` -- games where the stronger side
        held more, and games where the two held different amounts at all.
    """
    # Two sums per quantity, both over the same joined rows: one counts the
    # games the sides differ on and the other the games the stronger leads.
    # Written as columns rather than as one query per quantity because the
    # join is the cost and it is the same join either way.
    columns = ", ".join(
        f"sum(s.{name} <> w.{name}), sum(s.{name} > w.{name})" for name in quantities
    )
    row = connection.execute(
        f"""
        SELECT {columns}
        FROM days s
        JOIN days w ON w.episode = s.episode AND w.seat <> s.seat AND w.day = s.day
        JOIN episodes e ON e.episode = s.episode
        JOIN teams ts ON ts.team = s.team
        JOIN teams tw ON tw.team = w.team
        WHERE s.day = ? AND ts.rating - tw.rating >= ? AND e.played >= ?
        """,  # noqa: S608 - column names are the dataset's own schema
        (day, MARGIN, first),
    ).fetchone()
    return {
        name: (int(row[2 * index + 1] or 0), int(row[2 * index] or 0))
        for index, name in enumerate(quantities)
    }


def questions() -> list[strategy.Form]:
    """Every quantity crossed with every day: the whole grammar, asked at once.

    A model was asked what the winners did differently and wrote eleven
    plausible sentences, of which the corpus kept one. The grammar of a form
    is small enough to enumerate instead -- some thirty quantities over thirty
    days -- so nothing has to be guessed, and what comes back is whatever the
    corpus actually separates the sides on rather than whatever somebody
    thought of.

    Prose still has to be written for the ones that settle, and that is the
    job a model is good at: explaining a measurement, not producing one.
    """
    return [
        strategy.Form(quantity=quantity, day=day)
        for quantity in sorted(strategy.QUANTITIES)
        for day in range(dataset.DAYS)
    ]
