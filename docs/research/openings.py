"""Is the opening a strategy a team has, or a thing that happens to a game?

The build order's means blend incompatible openings -- 16% of the top twelve
hold two quadrants on day three and 84% hold one -- so the obvious repair is to
separate the groups and read each on its own. That only means something if the
groups are *teams*. If a team rushes land in some games and not others, the
split is the seed talking and there is no second build order to write, only a
conditional inside one.

So: for every seat-game of the top agents, the day it reached its second and
third quadrant. Tight within a team and different between teams is a strategy.
Wide within a team is a situation.
"""

import sqlite3
import statistics

from kaggriculture.campaign import dataset

NEVER = 99


def main() -> None:
    """Per team, when it takes its second and third quadrant, and how consistently."""
    connection = sqlite3.connect(dataset.DATABASE)
    try:
        first = dataset.recent(connection)
        top = [
            (row[0], row[1])
            for row in connection.execute(
                "SELECT team, rating FROM teams ORDER BY rating DESC LIMIT ?",
                (dataset.BEST,),
            )
        ]
        print(f"{'team':<26}{'rating':>7}{'2nd quad':>12}{'spread':>8}"
              f"{'3rd quad':>12}{'games':>7}")
        rows = []
        for team, rating in top:
            reached = _reached(connection, team, first)
            if not reached:
                continue
            second = [day for day, _ in reached]
            third = [day for _, day in reached]
            rows.append((team, rating, second, third))
            print(
                f"{team[:25]:<26}{rating:>+7.2f}"
                f"{statistics.median(second):>12.0f}"
                f"{statistics.pstdev(second):>8.1f}"
                f"{statistics.median(third):>12.0f}"
                f"{len(second):>7}"
            )
        print()
        spreads = [statistics.pstdev(s) for _, _, s, _ in rows]
        betweens = [statistics.median(s) for _, _, s, _ in rows]
        print(f"within a team, the 2nd quadrant day varies by "
              f"{statistics.mean(spreads):.2f} days on average")
        print(f"between teams, the median day ranges "
              f"{min(betweens):.0f} to {max(betweens):.0f}")
        print(
            "\nstrategy if between-team spread beats within-team; "
            "situation if not"
        )
    finally:
        connection.close()


def _reached(
    connection: sqlite3.Connection, team: str, first: str
) -> list[tuple[int, int]]:
    """For each of this team's seat-games, the day it reached 2 and 3 quadrants."""
    rows = connection.execute(
        "SELECT d.episode, d.seat, min(CASE WHEN d.quadrants >= 2 THEN d.day END), "
        "min(CASE WHEN d.quadrants >= 3 THEN d.day END) "
        "FROM days d JOIN episodes e ON e.episode = d.episode "
        "WHERE d.team = ? AND e.played >= ? GROUP BY d.episode, d.seat",
        (team, first),
    ).fetchall()
    return [
        (NEVER if second is None else int(second), NEVER if third is None else int(third))
        for _, _, second, third in rows
    ]


if __name__ == "__main__":
    main()
