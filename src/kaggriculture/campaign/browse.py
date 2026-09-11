"""Write the games behind a verdict where an agent can ask them questions.

A round is handed the games it was scored on, and how they are handed over
decides what it can do with them. Three shapes have been tried.

Rendered into the message, six games at 30 rows of fifteen columns were 74% of
it, so bounding the message meant dropping games -- and a markdown table stops
being readable around fifteen columns while a `Day` carries sixty-odd. The
evidence was capped twice over, once by the message's size and once by the
table's width.

Written to a CSV beside `child.py`, neither cap applies and every scored game
is there at full width. But 768 games of 30 days is nine megabytes, and a round
cannot ask a nine-megabyte file anything: it has to load the whole thing and
write code before it can answer "which day did I lose". That is reading, not
browsing, and it stops working long before the interesting scale -- the public
corpus is 17,617 episodes and a million day rows.

So: SQLite. Nothing to install, nothing to teach, `.schema` documents itself,
and a query costs the same whether the table holds sixty rows or sixty million.
The size of the evidence stops being the round's problem, which is the only
property that scales.

The schema is `dataset.SCHEMA` itself, not a copy of it. That is the point of
this module rather than a convenience: the public corpus is written to that
schema and our games are written to the same one, by the same text, so a row
from a game the campaign played and a row from a game somebody published are
the same kind of thing. Attach the two and they union -- no translation, no
column that means something slightly different on one side. It holds because
`dataset.measures` is the single function that defines every measure and both
the corpus extraction and the live harness call it.

What our database has that the corpus does not is `candidate`: which seat was
ours in each episode. The corpus has no "ours", so the column would be
meaningless there; keeping it in a table of its own leaves `days` byte-for-byte
comparable and still lets the views below say "ours minus theirs".
"""

import sqlite3
from collections.abc import Sequence
from pathlib import Path

from kaggriculture.campaign import dataset, harness

# Ours, beyond the corpus's own tables. `candidate` names the seat the program
# being measured held, which is the one thing the corpus cannot have. The views
# are the two questions a round actually asks.
OURS = """
CREATE TABLE IF NOT EXISTS candidate (
    episode  TEXT PRIMARY KEY,
    seat     INTEGER,
    team     TEXT,
    matchup  INTEGER,
    season   INTEGER
);

CREATE VIEW IF NOT EXISTS gaps AS
SELECT c.matchup AS matchup, c.season AS season, o.episode AS episode,
       o.day AS day, {differences}
FROM days o
JOIN candidate c ON c.episode = o.episode AND c.seat = o.seat
JOIN days t ON t.episode = o.episode AND t.day = o.day AND t.seat <> o.seat;

CREATE VIEW IF NOT EXISTS swings AS
SELECT * FROM (
    SELECT *, bank - LAG(bank) OVER (PARTITION BY episode ORDER BY day) AS moved
    FROM gaps
) WHERE moved IS NOT NULL;
"""


def write(path: Path, games: Sequence[tuple[int, int, str, harness.Game]]) -> Path:
    """Write every day of every game to a fresh database at ``path``.

    Replaced rather than appended to: a half-written database that looks whole
    is worse than none, and the caller always holds every game it means to
    store.

    Args:
        path: Where to write. Removed first if it exists.
        games: One entry per game: its matchup, its season within that
            matchup, the name to record for the program being measured, and
            the played `Game` with its days recorded.

    Returns:
        ``path``, written.
    """
    path.unlink(missing_ok=True)
    connection = sqlite3.connect(path)
    with connection:
        connection.executescript(dataset.SCHEMA)
        connection.executescript(
            OURS.format(
                differences=",\n       ".join(
                    f"o.{column} - t.{column} AS {column}" for column in dataset.COLUMNS
                )
            )
        )
        for matchup, season, mine, game in games:
            _game(connection, matchup, season, mine, game)
    connection.close()
    return path


def _game(
    connection: sqlite3.Connection,
    matchup: int,
    season: int,
    mine: str,
    game: harness.Game,
) -> None:
    """Insert one game into the corpus's own tables, plus who was on our side.

    The seats are the engine's, not the candidate's point of view. A `Day`
    records "ours" and "theirs" relative to whoever was being measured, and
    `game.seat` says which engine seat that was, so the relative labels are
    resolved back to absolute ones here. That is what makes the row the same
    kind of row as a corpus row, where there is no "ours" at all.
    """
    # The episode id has to be unique inside this database and mean nothing
    # outside it. The opponent is not in it: a name here would be a name in a
    # file the round can read, and no opponent is named anywhere it can see.
    episode = f"m{matchup}s{season}"
    theirs = 1 - game.seat
    final = game.days[-1] if game.days else None
    connection.execute(
        "INSERT INTO episodes (episode, seed, engine, team_0, team_1, "
        "bank_0, bank_1, winner) VALUES (?, ?, 'port', ?, ?, ?, ?, ?)",
        (
            episode,
            game.seed,
            mine if game.seat == 0 else "opponent",
            mine if game.seat == 1 else "opponent",
            game.ours if game.seat == 0 else game.theirs,
            game.ours if game.seat == 1 else game.theirs,
            game.seat if game.ours > game.theirs else theirs,
        ),
    )
    connection.execute(
        "INSERT INTO candidate (episode, seat, team, matchup, season) "
        "VALUES (?, ?, ?, ?, ?)",
        (episode, game.seat, mine, matchup, season),
    )
    if final is None:
        return

    columns = ", ".join(["episode", "seat", "day", "team", *dataset.COLUMNS])
    slots = ", ".join("?" * (4 + len(dataset.COLUMNS)))
    connection.executemany(
        f"INSERT INTO days ({columns}) VALUES ({slots})",
        [
            [
                episode,
                game.seat if side == "ours" else theirs,
                day.day,
                mine if side == "ours" else "opponent",
                *(getattr(day, side).get(column, 0.0) for column in dataset.COLUMNS),
            ]
            for day in game.days
            for side in ("ours", "theirs")
        ],
    )

    # The per-crop breakdowns behind four of those totals, in the corpus's own
    # long form: a row per item rather than a column per crop, because which
    # crops exist depends on what was planted and a fixed column list is how a
    # quantity comes to be stored and never looked at.
    connection.executemany(
        "INSERT INTO holdings (episode, seat, day, kind, item, count) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            [episode, game.seat if side == "ours" else theirs, day.day, kind, item, n]
            for day in game.days
            for side, kind in (
                ("ours", "plants"),
                ("theirs", "plants"),
                ("ours", "animals"),
                ("theirs", "animals"),
                ("ours", "seeds"),
                ("ours", "shed"),
                ("theirs", "shed"),
            )
            for item, n in (getattr(day, f"{side}_{kind}", None) or {}).items()
        ],
    )
    connection.executemany(
        "INSERT INTO prices (episode, day, item, price) VALUES (?, ?, ?, ?)",
        [
            [episode, day.day, item, price]
            for day in game.days
            for item, price in day.prices.items()
        ],
    )
