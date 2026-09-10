"""What the strongest opening actually does, as orders rather than as averages.

The averaged build order asked for 1.2 quadrants on day three, which no agent
holds -- but nothing in the corpus is impossible. Every order in it was
executed by a real agent in a real game. What could not be followed was our
summary, not the play.

So read the play. For the group that shares the strongest opening, the orders
they send on each of the first days, by how often they send them. Aggregate
over the group and named by no agent, which is the footing the build order and
the report already stand on -- but a sequence rather than a state, which is
what we have never extracted.
"""

import sqlite3

from kaggriculture.campaign import dataset

DAYS = 5
SHOWN = 4


def main() -> None:
    """Print the strongest opening's orders, day by day."""
    best = dataset.openings()[0]
    print(f"the strongest opening: {best.describe()}, rating {best.rating:+.2f}")
    print(f"across {len(best.teams)} agents\n")
    connection = sqlite3.connect(dataset.DATABASE)
    try:
        first = dataset.recent(connection)
        marks = ",".join("?" * len(best.teams))
        seats = connection.execute(
            f"SELECT count(DISTINCT o.episode || o.seat) FROM orders o "  # noqa: S608
            f"JOIN days d ON d.episode = o.episode AND d.seat = o.seat AND d.day = 0 "
            f"JOIN episodes e ON e.episode = o.episode "
            f"WHERE d.team IN ({marks}) AND e.played >= ?",
            (*best.teams, first),
        ).fetchone()[0]
        for day in range(DAYS):
            rows = connection.execute(
                f"SELECT o.verb, o.item, count(*), avg(o.quantity) FROM orders o "  # noqa: S608
                f"JOIN days d ON d.episode = o.episode AND d.seat = o.seat "
                f"AND d.day = 0 "
                f"JOIN episodes e ON e.episode = o.episode "
                f"WHERE d.team IN ({marks}) AND e.played >= ? AND o.day = ? "
                f"GROUP BY o.verb, o.item ORDER BY count(*) DESC LIMIT ?",
                (*best.teams, first, day, SHOWN),
            ).fetchall()
            print(f"day {day}: per seat-game, over {seats:,} of them")
            for verb, item, count, quantity in rows:
                each = count / seats
                amount = f" x{quantity:.0f}" if quantity else ""
                print(
                    f"    {verb:<14}{(item or ''):<12}{each:>7.1f} orders"
                    f"{amount}"
                )
            print()
    finally:
        connection.close()


if __name__ == "__main__":
    main()
