"""Is the build order a behaviour, or a mean over agents doing different things?

The pacer could not follow it. The corpus says the top twelve hold 1.2
quadrants on day three and a mean bank of 220 coins, and land costs 1,000. A
mean of 1.2 over a quantity that only takes whole values is not a thing any
agent does: it is a mixture, and the share is the interesting number.

If a fifth of them have rushed land by day three and the rest have not, then
the build order blends a land-rush opening with a slow one, and an agent
following the mean does neither.
"""

import pathlib
import sqlite3

from kaggriculture.campaign import dataset


def main() -> None:
    """Show the distribution behind the mean, for the days the pacer needed."""
    connection = sqlite3.connect(dataset.DATABASE)
    try:
        first = dataset.recent(connection)
        top = [
            row[0]
            for row in connection.execute(
                "SELECT team FROM teams ORDER BY rating DESC LIMIT ?", (dataset.BEST,)
            )
        ]
        marks = ",".join("?" * len(top))
        for day in (3, 6, 8, 10):
            rows = connection.execute(
                f"SELECT d.quadrants, count(*), avg(d.bank) FROM days d "  # noqa: S608
                f"JOIN episodes e ON e.episode = d.episode "
                f"WHERE d.day = ? AND d.team IN ({marks}) AND e.played >= ? "
                f"GROUP BY d.quadrants ORDER BY d.quadrants",
                (day, *top, first),
            ).fetchall()
            total = sum(count for _, count, _ in rows)
            mean = sum(q * c for q, c, _ in rows) / total
            print(f"day {day}: mean {mean:.2f} quadrants over {total:,} seat-games")
            for quadrants, count, bank in rows:
                share = count / total
                print(
                    f"    {int(quadrants)} quadrant(s): {share:>6.1%} of them, "
                    f"mean bank {bank:>9,.0f}  {'#' * int(share * 40)}"
                )
            print()
    finally:
        connection.close()


if __name__ == "__main__":
    main()
