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

from kaggriculture.campaign import config, dataset, games, strategy

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
    out: dict[str, tuple[int, float]] = {}
    # The same window the ratings were fitted over. A claim measured across
    # the whole corpus is measured over two disjoint fields: split in half,
    # the two top tens share not one name.
    first = dataset.recent(database)
    for day, claims in sorted(wanted.items()):
        quantities = sorted({claim.form.quantity for claim in claims})
        counted = _day(day, quantities, first, database)
        for claim in claims:
            ahead, seen = counted[claim.form.quantity]
            if not seen:
                continue
            store.record(claim.id, seen, ahead / seen)
            out[claim.id] = (seen, ahead / seen)
        LOGGER.info("day %d: %d quantities over the corpus", day, len(quantities))
    return out


def _day(
    day: int, quantities: list[str], first: str, database: str = config.GAMES_DB
) -> dict[str, tuple[int, int]]:
    """For one day, how often the stronger side had more of each quantity.

    Read over the ladder alone, and said rather than relied on. Both writers
    put their day rows in one table on purpose -- the questions worth asking
    span them -- and the campaign is now 18.6M of the 19.8M rows in it, so a
    claim about how the ladder's strongest agents play is a claim about 6% of
    the table it is measured over.

    Today the join to `teams` would exclude the rest on its own: `teams` holds
    the 60 display names the ladder rated, and not one of the 294 names the
    campaign's own rows carry. That is a coincidence of naming rather than a
    guarantee -- the campaign plays harvested kernels under slugs built from
    the same Kaggle accounts those display names belong to, so either side
    changing how it spells a competitor would fold this lineage's self-play
    into a measurement of the field. The filter costs nothing (2,722 day-10
    bank comparisons either way) and states which population is meant.

    Args:
        day: The day both sides are read at.
        quantities: Day-row columns to compare.
        first: The earliest archive day to read, so a claim is settled on the
            field as it now plays rather than on one that has turned over.
        database: The games database.

    Returns:
        ``{quantity: (ahead, differing)}`` -- games where the stronger side
        held more, and games where the two held different amounts at all.
    """
    # Two sums per quantity, both over the same joined rows: one counts the
    # games the sides differ on and the other the games the stronger leads.
    # Written as columns rather than as one query per quantity because the
    # join is the cost and it is the same join either way.
    columns = ", ".join(
        f"sum(s.{name} != w.{name}), sum(s.{name} > w.{name})" for name in quantities
    )
    # The seat inequality sits in the WHERE rather than the ON: ClickHouse
    # joins on equality, and a non-equi condition in an ON clause is refused.
    row = games.query(
        f"""
        SELECT {columns}
        FROM {database}.days s
        INNER JOIN {database}.days w
            ON w.episode = s.episode AND w.day = s.day
        INNER JOIN {database}.episodes e ON e.episode = s.episode
        INNER JOIN {database}.teams ts ON ts.team = s.team
        INNER JOIN {database}.teams tw ON tw.team = w.team
        WHERE s.day = {day}
          AND s.source = 'ladder'
          AND w.seat != s.seat
          AND ts.rating - tw.rating >= {MARGIN}
          AND e.played >= '{first}'
        FORMAT TabSeparated
        """  # noqa: S608 - column names are the dataset's own schema
    ).split("\t")
    return {
        name: (int(row[2 * index + 1]), int(row[2 * index]))
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
