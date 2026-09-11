"""The one games database: the recorded ladder and every game we play.

Two writers, one store. The nightly extraction reads the competition's replays
into it and the loop writes every game it plays into it, because the questions
worth asking span both -- is this lineage converging on what the field does, or
somewhere else -- and that is only askable while both are rows in one table.

ClickHouse, for the two properties needed at once. Eight sessions record in
parallel: SQLite admits a single writer, and a lock or a busy timeout only
decides how the second one fails. And the questions are analytical -- an
average of one measure across every day of the record -- which reads one column
here and all thirty-three in a row-oriented store. At the campaign's rate the
`days` table gains about 2.2 million rows an hour, so both properties are load
bearing rather than nice.

Compression is the third thing, and it changes the shape of the problem rather
than its speed. These are small integers repeating down a season, which is the
case columnar compression is best at: a billion rows that would be 310 GB of
row-oriented tuples land in tens of gigabytes here.

The column types come from the corpus rather than from what the names suggest.
Measured over 1,057,020 recorded days: nothing is ever negative, `plant_age` is
the only measure that is ever fractional (936,492 of them), `sold_units` runs
to 15,002,111, and every other count stays inside sixteen bits. A count stored
as an integer that turns out to be fractional truncates in silence, so this was
read off the data before it was written down.

A round reads this database too. It is handed no copy and no extract: there
is one store, and the questions it can ask of its own games are the same
questions it can ask of the ladder's.
"""

import logging
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING

from kaggriculture.campaign import config, dataset, harness

if TYPE_CHECKING:  # `evaluator` imports `harness`, which imports this.
    from kaggriculture.campaign import evaluator

LOGGER = logging.getLogger(__name__)

# A parameter everywhere rather than baked into the SQL, so a test points at
# one of its own instead of writing into the campaign's and tidying up
# afterwards -- which it did, and which left ten probe episodes in the real
# store before anything noticed.
DATABASE = config.GAMES_DB

# Counts that are cumulative over a season and can run large; `sold_units`
# reaches fifteen million in the corpus. The rest are tiles, pens and hands on
# one day and stay inside sixteen bits.
WIDE = (
    "sell_orders",
    "sold_units",
    "buy_orders",
    "bought_units",
    "seed_orders",
    "seed_units",
    "animal_orders",
    "animal_units",
    "hire_orders",
    "land_orders",
)
# The only measures that are not whole numbers. `bank` is money and the engine
# carries it as a float, so it is stored as one even though the corpus happens
# to hold no fractional bank.
REAL = ("bank", "plant_age")


# The columns each table is written with, named rather than positional. A
# positional insert keeps working right up until two column counts happen to
# match and every value lands one to the left.
COLUMNS = {
    "episodes": (
        "episode, kaggle_id, seed, engine, played, team_0, team_1, "
        "bank_0, bank_1, winner, source"
    ),
    "days": "episode, seat, day, team, " + ", ".join(dataset.COLUMNS) + ", source",
    "holdings": "episode, seat, day, kind, item, count, source",
    "prices": "episode, day, item, price, stock, source",
    "orders": "episode, seat, day, hour, verb, item, quantity, source",
    "moves": "episode, seat, day, hour, actor, verb, argument, source",
    "candidate": "episode, seat, team, matchup, season, source",
}


def column_type(measure: str) -> str:
    """The narrowest type that holds every value this measure takes."""
    if measure in REAL:
        return "Float32"
    return "UInt32" if measure in WIDE else "UInt16"


def schema(database: str = DATABASE) -> str:
    """Every table, written from `dataset.COLUMNS` so it cannot drift from it.

    `days` is ordered by source, day, episode and seat, which is the order the
    questions arrive in: almost everything filters or groups by day, across one
    source or both. ClickHouse's primary index is that ordering, so a query for
    one day reads one granule range rather than the table.

    `ReplacingMergeTree` because a champion re-scored is the same program, not a
    second one. There is no cheap `DELETE` here and there does not need to be:
    re-recording inserts, and the engine collapses the older row on merge. The
    ordering key is the identity of a row -- one episode, one seat, one day.
    """
    measures = ",\n    ".join(
        f"{measure} {column_type(measure)}" for measure in dataset.COLUMNS
    )
    return f"""
CREATE TABLE IF NOT EXISTS {database}.episodes (
    episode    String,
    kaggle_id  Int64,
    seed       Int64,
    engine     LowCardinality(String),
    played     LowCardinality(String),
    team_0     LowCardinality(String),
    team_1     LowCardinality(String),
    bank_0     Float32,
    bank_1     Float32,
    winner     Int8,
    source     LowCardinality(String),
    version    UInt64 DEFAULT toUnixTimestamp64Milli(now64(3))
) ENGINE = ReplacingMergeTree(version)
PARTITION BY source
ORDER BY (source, episode);

CREATE TABLE IF NOT EXISTS {database}.days (
    episode    String,
    seat       UInt8,
    day        UInt16,
    team       LowCardinality(String),
    source     LowCardinality(String),
    {measures},
    version    UInt64 DEFAULT toUnixTimestamp64Milli(now64(3))
) ENGINE = ReplacingMergeTree(version)
PARTITION BY source
ORDER BY (source, day, episode, seat);

CREATE TABLE IF NOT EXISTS {database}.holdings (
    episode    String,
    seat       UInt8,
    day        UInt16,
    kind       LowCardinality(String),
    item       LowCardinality(String),
    count      Int32,
    source     LowCardinality(String),
    version    UInt64 DEFAULT toUnixTimestamp64Milli(now64(3))
) ENGINE = ReplacingMergeTree(version)
PARTITION BY source
ORDER BY (source, episode, seat, day, kind, item);

CREATE TABLE IF NOT EXISTS {database}.prices (
    episode    String,
    day        UInt16,
    item       LowCardinality(String),
    price      Int32,
    stock      Int32,
    source     LowCardinality(String),
    version    UInt64 DEFAULT toUnixTimestamp64Milli(now64(3))
) ENGINE = ReplacingMergeTree(version)
PARTITION BY source
ORDER BY (source, episode, day, item);

CREATE TABLE IF NOT EXISTS {database}.orders (
    episode    String,
    seat       UInt8,
    day        UInt16,
    hour       UInt8,
    verb       LowCardinality(String),
    item       LowCardinality(String),
    quantity   Int32,
    source     LowCardinality(String)
) ENGINE = MergeTree
PARTITION BY source
ORDER BY (source, episode, seat, day, hour);

CREATE TABLE IF NOT EXISTS {database}.moves (
    episode    String,
    seat       UInt8,
    day        UInt16,
    hour       UInt8,
    -- The farmer is -1 and the hands are 0 upward, so this is signed.
    actor      Int8,
    verb       LowCardinality(String),
    argument   LowCardinality(String),
    source     LowCardinality(String)
) ENGINE = MergeTree
PARTITION BY source
ORDER BY (source, episode, seat, day, hour);

CREATE TABLE IF NOT EXISTS {database}.teams (
    team       String,
    games      UInt32,
    wins       UInt32,
    rating     Float64,
    place      UInt32
) ENGINE = ReplacingMergeTree
ORDER BY team;

CREATE TABLE IF NOT EXISTS {database}.candidate (
    episode    String,
    seat       UInt8,
    team       LowCardinality(String),
    matchup    UInt16,
    season     UInt16,
    source     LowCardinality(String),
    version    UInt64 DEFAULT toUnixTimestamp64Milli(now64(3))
) ENGINE = ReplacingMergeTree(version)
ORDER BY episode;
"""


# How each measure is written, decided by the same function that decided its
# column type. `dataset.measures` returns every quantity as a float -- a tile
# count arrives as `4.0` -- and TabSeparated will not parse "4.0" into a
# `UInt16`. Reading the cast off `column_type` rather than off a second list
# means the value written and the column it is written into cannot disagree.
def text(value: object) -> str:
    """A string column's value; a missing one is empty, never the word None."""
    return "" if value is None else str(value)


def number(value: object) -> int:
    """An integer column's value; a missing one is zero.

    Through `float` because `dataset.measures` returns every quantity as one:
    a tile count arrives as `4.0`, and `int("4.0")` raises where `int(4.0)`
    does not.
    """
    return 0 if value is None else int(float(str(value)))


def real(value: object) -> float:
    """A float column's value; a missing one is zero."""
    return 0.0 if value is None else float(str(value))


_CAST: tuple[Callable[[object], object], ...] = tuple(
    real if column_type(measure) == "Float32" else number for measure in dataset.COLUMNS
)


# What each column is written as, for the tables whose values do not already
# arrive in the right type. `dataset.measures` returns every quantity as a
# float -- a tile count arrives as `4.0` -- and TabSeparated will not parse
# "4.0" into a `UInt16`, so the insert fails rather than the field.
#
# In `Batch` rather than in either writer, because there are two of them: the
# loop recording a game it played and the extraction reading one off the
# ladder. This was a cast in the first, and the second met the same wall the
# moment it was written.
CASTS: dict[str, tuple[Callable[[object], object], ...]] = {
    "days": (text, number, number, text, *_CAST),
    # A HIRE carries no item and no quantity, and a move may carry no
    # argument. None reaches TabSeparated as the word "None", which parses
    # into a String column and fails an integer one -- so the absent ones are
    # spelled out here rather than discovered a table at a time.
    "orders": (text, number, number, number, text, text, number),
    "moves": (text, number, number, number, number, text, text),
    "prices": (text, number, text, number, number),
    "holdings": (text, number, number, text, text, number),
}


class Batch:
    """Rows for one archive, held in memory and sent a table at a time.

    Batched client-side and inserted synchronously, which is what ClickHouse
    asks for when the caller already has whole batches: `async_insert` exists
    to coalesce many small writes, and adds latency without benefit to a
    writer that arrives with an archive's worth of rows. One archive is one
    insert per table.

    Eight of these run at once, one per parsing process, which is the
    throughput the store was chosen for. They need no coordination: they write
    to the same tables and never to the same row.
    """

    def __init__(self, source: str) -> None:
        """Start an empty batch whose rows all carry ``source``."""
        self.source = source
        self.rows: dict[str, list[str]] = {}

    def add(self, table: str, values: Iterable[object]) -> None:
        """Hold one row for ``table``, cast to its columns, with the source appended."""
        casts = CASTS.get(table)
        cast = (
            [function(value) for function, value in zip(casts, values, strict=True)]
            if casts
            else list(values)
        )
        self.rows.setdefault(table, []).append(_row([*cast, self.source]))

    def send(self, database: str = DATABASE) -> dict[str, int]:
        """Insert everything held, one request per table, and forget it.

        Returns:
            How many rows each table took.
        """
        sent = {}
        for table, rows in self.rows.items():
            _insert(table, COLUMNS[table], rows, database)
            sent[table] = len(rows)
        self.rows = {}
        return sent


def query(sql: str, body: bytes | None = None) -> str:
    """Send one statement over the HTTP interface and return what comes back.

    The HTTP interface rather than a driver, because it is the one every
    ClickHouse has, it needs no dependency, and an insert is a POST of rows in
    a format the server already parses. Eight sessions posting at once is eight
    independent requests -- the parallel writing this database was chosen for.

    Args:
        sql: The statement.
        body: Rows to send with it, for an INSERT.

    Returns:
        The server's response, stripped.

    Raises:
        RuntimeError: The server refused the statement, with what it said.
    """
    url = f"{config.GAMES_URL}/?{urllib.parse.urlencode({'query': sql})}"
    request = urllib.request.Request(url, data=body or b"", method="POST")  # noqa: S310 - a localhost http url from config
    try:
        with urllib.request.urlopen(request) as response:  # noqa: S310 - as above
            return response.read().decode().strip()
    except urllib.error.HTTPError as error:
        detail = error.read().decode().strip()
        raise RuntimeError(f"{sql.splitlines()[0][:80]}: {detail}") from error


def create(database: str = DATABASE) -> None:
    """Make every table, if it is not already there."""
    query(f"CREATE DATABASE IF NOT EXISTS {database}")
    for statement in schema(database).split(";"):
        if statement.strip():
            query(statement)


def ordered(result: "evaluator.Result") -> list[str]:
    """The matchups of one evaluation, worst-beaten opponent first.

    A loss is a game lost, not a matchup lost: an opponent beaten 0.875 took
    one game in eight, and those are the games that decide whether a program
    finishes top. So the order is by rate and then by margin, which puts the
    matchups with the most to learn from first.

    The numbering is what a matchup *is*: an opponent's name never travels, so
    `matchup = 1` is the only handle there is on which games those were.
    """
    return sorted(
        (name for name in result.rates if result.states.get(name)),
        key=lambda name: (result.rates[name], result.margins[name].mean),
    )


def played(
    result: "evaluator.Result", name: str
) -> list[tuple[int, int, str, harness.Game]]:
    """One evaluation's games, keyed the way they are recorded."""
    return [
        (matchup, season, name, game)
        for matchup, opponent in enumerate(ordered(result), start=1)
        for season, game in enumerate(result.states[opponent], start=1)
    ]


def record(
    name: str,
    played: Sequence[tuple[int, int, str, harness.Game]],
    database: str = DATABASE,
) -> int:
    """Write one program's games, in parallel with whatever else is writing.

    No lock and no delete. Two sessions recording at once are two POSTs, and a
    program recorded twice inserts twice and is collapsed by the engine on the
    ordering key, so a re-scored champion is one program rather than two.

    The opponent is written as "opponent". Its roster name stays in the
    campaign's own memory: a name in a table is a name a round could read, and
    recognising one opponent is worth nothing against a field that turns over.

    Args:
        name: The program these games belong to; prefixes their episode keys.
        played: One entry per game, as `browse.games` returns them.
        database: Which database to write to. A parameter so a test can
            point at one of its own rather than tidy up after itself here.

    Returns:
        How many games were written.
    """
    if not played:
        return 0
    batch = Batch("campaign")
    for matchup, season, mine, game in played:
        episode = f"{name}m{matchup}s{season}"
        theirs = 1 - game.seat
        batch.add(
            "episodes",
            [
                episode,
                0,
                game.seed,
                "port",
                "",
                mine if game.seat == 0 else "opponent",
                mine if game.seat == 1 else "opponent",
                game.ours if game.seat == 0 else game.theirs,
                game.ours if game.seat == 1 else game.theirs,
                game.seat if game.ours > game.theirs else theirs,
            ],
        )
        batch.add("candidate", [episode, game.seat, mine, matchup, season])
        for day in game.days:
            for side in ("ours", "theirs"):
                seat = game.seat if side == "ours" else theirs
                measures = getattr(day, side)
                batch.add(
                    "days",
                    [
                        episode,
                        seat,
                        day.day,
                        mine if side == "ours" else "opponent",
                        *(measures.get(column, 0) for column in dataset.COLUMNS),
                    ],
                )
                for kind in ("plants", "animals", "seeds", "shed"):
                    counts = getattr(day, f"{side}_{kind}", None) or {}
                    for item, held in counts.items():
                        batch.add(
                            "holdings", [episode, seat, day.day, kind, item, held]
                        )
            for item, price in day.prices.items():
                batch.add("prices", [episode, day.day, item, price, 0])
    batch.send(database)
    return len(played)


def _insert(table: str, columns: str, rows: Sequence[str], database: str) -> None:
    """POST rows into one table as tab-separated values."""
    if rows:
        query(
            f"INSERT INTO {database}.{table} ({columns}) FORMAT TabSeparated",
            ("\n".join(rows) + "\n").encode(),
        )


def _row(values: Iterable[object]) -> str:
    """One tab-separated row, with the escapes that format requires."""
    return "\t".join(_field(value) for value in values)


def _field(value: object) -> str:
    """One value, escaped for TabSeparated.

    Tabs, newlines and backslashes are the format's own delimiters, so a team
    name carrying one would otherwise shift every column after it -- silently,
    because the row still parses.
    """
    if isinstance(value, float):
        return repr(value)
    text = str(value)
    for character, escape in (("\\", "\\\\"), ("\t", "\\t"), ("\n", "\\n")):
        text = text.replace(character, escape)
    return text
