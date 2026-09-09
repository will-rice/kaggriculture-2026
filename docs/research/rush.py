"""What lets the top three take land on day three when the field cannot?

Their second quadrant lands on day three and everyone else's on day five or
six, and they sit 0.9 log-odds clear of fourth place. Land costs 1,000, and the
top twelve's mean bank on day three is about 200 -- so whatever pays for it is
something the three do differently in the first seventy-two hours.

Compare the three against the nine below them, day by day, on every quantity
the dataset keeps. The columns that separate them early are the mechanism.
"""

import sqlite3

from kaggriculture.campaign import dataset

RUSH = 3


def main() -> None:
    """Set the day-three rushers beside the rest of the top twelve."""
    connection = sqlite3.connect(dataset.DATABASE)
    try:
        first = dataset.recent(connection)
        top = [
            row[0]
            for row in connection.execute(
                "SELECT team FROM teams ORDER BY rating DESC LIMIT ?", (dataset.BEST,)
            )
        ]
        rush, rest = top[:RUSH], top[RUSH:]
        print(f"rushers: {', '.join(t[:18] for t in rush)}")
        print(f"the rest: {len(rest)} teams\n")
        for day in (0, 1, 2, 3, 4, 6):
            mine = _hold(connection, rush, day, first)
            theirs = _hold(connection, rest, day, first)
            gaps = sorted(
                (
                    (mine[q] - theirs[q]) / (abs(theirs[q]) + 1e-6), q, mine[q],
                    theirs[q],
                )
                for q in mine
                if abs(mine[q] - theirs[q]) > 1e-9
            )
            print(f"day {day}: what the rushers do differently")
            for share, quantity, ours, other in list(reversed(gaps))[:4]:
                print(
                    f"    {quantity:<16}{ours:>10.1f}  against{other:>10.1f}"
                    f"   {share:+7.0%}"
                )
            print()
    finally:
        connection.close()


def _hold(
    connection: sqlite3.Connection, teams: list[str], day: int, first: str
) -> dict[str, float]:
    """Mean of every tracked quantity for those teams on that day."""
    marks = ",".join("?" * len(teams))
    columns = ", ".join(f"avg({name})" for name in dataset.TARGETS)
    row = connection.execute(
        f"SELECT {columns} FROM days d "  # noqa: S608 - names are the dataset's own
        f"JOIN episodes e ON e.episode = d.episode "
        f"WHERE d.day = ? AND d.team IN ({marks}) AND e.played >= ?",
        (day, *teams, first),
    ).fetchone()
    return {
        name: float(value or 0.0)
        for name, value in zip(dataset.TARGETS, row, strict=True)
    }


if __name__ == "__main__":
    main()
