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
from typing import TYPE_CHECKING, NamedTuple

from kaggriculture.campaign import config, dataset, harness

if TYPE_CHECKING:  # `evaluator` imports `harness`, which imports this.
    from kaggriculture.campaign import evaluator

LOGGER = logging.getLogger(__name__)

# A parameter everywhere rather than baked into the SQL, so a test points at
# one of its own instead of writing into the campaign's and tidying up
# afterwards -- which it did, and which left ten probe episodes in the real
# store before anything noticed.
DATABASE = config.GAMES_DB


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


class Column(NamedTuple):
    """One column: how the server stores it, and how a value is written to it.

    The pair is the point. They were two lists before -- a `CREATE TABLE` in
    one place and a tuple of casts in another -- and they disagreed the first
    time a column was added, because nothing made them agree. A column that
    cannot be declared without saying how it is written cannot drift from how
    it is written.

    Attributes:
        kind: The ClickHouse type.
        cast: What a Python value becomes on the wire. `dataset.measures`
            returns every quantity as a float, so a tile count arrives as
            `4.0` and TabSeparated will not parse that into a `UInt16`.
        written: Whether a writer supplies it. `version` is filled by the
            server and is declared but never sent.
    """

    kind: str
    cast: Callable[[object], object] = text
    written: bool = True


# Filled by the server on every insert, so that a re-recorded program's rows
# replace its earlier ones rather than joining them.
VERSION = Column("UInt64 DEFAULT toUnixTimestamp64Milli(now64(3))", written=False)
# Which source a row came from: `ladder` for a recorded game, `campaign` for
# one this lineage played. Every table carries it and every table is
# partitioned on it, so a rebuild replaces what it owns and nothing else.
SOURCE = Column("LowCardinality(String)")

# Counts that are cumulative over a season and can run large; `sold_units`
# reaches fifteen million in the corpus. The rest are tiles, pens and hands on
# one day and stay inside sixteen bits.
WIDE = frozenset(
    {
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
    }
)
# The only measures that are not whole numbers. `bank` is money and the engine
# carries it as a float, so it is stored as one even though the corpus happens
# to hold no fractional bank.
REAL = frozenset({"bank", "plant_age"})


def measure_column(measure: str) -> Column:
    """The narrowest column that holds every value this measure takes.

    Read off the corpus rather than guessed from the name. Over 1,057,020
    recorded days nothing is ever negative, `plant_age` is the only measure
    ever fractional, and `sold_units` reaches 15,002,111 where every other
    count stays inside sixteen bits.
    """
    if measure in REAL:
        return Column("Float32", real)
    return Column("UInt32" if measure in WIDE else "UInt16", number)


class Table(NamedTuple):
    """One table: its columns in order, and how the server should keep them.

    Attributes:
        columns: Name to `Column`, in the order they are declared and written.
        order: The ordering key, which is ClickHouse's primary index and, for
            a `ReplacingMergeTree`, the identity of a row.
        replacing: Whether a later row replaces an earlier one on that key. A
            champion re-scored is the same program, not a second one.
        partition: The partition expression, or "" for a table small enough
            not to want one.
    """

    columns: dict[str, Column]
    order: str
    replacing: bool = True
    partition: str = "source"


# Every table, declared once. The `CREATE TABLE` text, the column list an
# insert names, the casts a row is written through and the reconciliation of a
# database built before a column existed are all read from this -- so there is
# nothing to keep in step, and "the schema and the writer disagree" stops being
# a thing that can happen rather than a thing a test catches.
#
# It was four: a SQL template, a dict of comma-joined column names, a tuple of
# casts, and a parser that read the types back out of the SQL. They disagreed
# within the hour and the run died on `No such column source in table
# games.candidate`.
TABLES: dict[str, Table] = {
    "episodes": Table(
        {
            "episode": Column("String"),
            "kaggle_id": Column("Int64", number),
            "seed": Column("Int64", number),
            "engine": Column("LowCardinality(String)"),
            "played": Column("LowCardinality(String)"),
            "team_0": Column("LowCardinality(String)"),
            "team_1": Column("LowCardinality(String)"),
            "bank_0": Column("Float32", real),
            "bank_1": Column("Float32", real),
            "winner": Column("Int8", number),
            "source": SOURCE,
            "version": VERSION,
        },
        order="(source, episode)",
    ),
    "days": Table(
        {
            "episode": Column("String"),
            "seat": Column("UInt8", number),
            "day": Column("UInt16", number),
            "team": Column("LowCardinality(String)"),
            **{measure: measure_column(measure) for measure in dataset.COLUMNS},
            "source": SOURCE,
            "version": VERSION,
        },
        # The order the questions arrive in: almost everything filters or
        # groups by day, across one source or both, so a query for one day
        # reads a granule range rather than the table.
        order="(source, day, episode, seat)",
    ),
    "holdings": Table(
        {
            "episode": Column("String"),
            "seat": Column("UInt8", number),
            "day": Column("UInt16", number),
            "kind": Column("LowCardinality(String)"),
            "item": Column("LowCardinality(String)"),
            "count": Column("Int32", number),
            "source": SOURCE,
            "version": VERSION,
        },
        order="(source, episode, seat, day, kind, item)",
    ),
    "prices": Table(
        {
            "episode": Column("String"),
            "day": Column("UInt16", number),
            "item": Column("LowCardinality(String)"),
            "price": Column("Int32", number),
            "stock": Column("Int32", number),
            "source": SOURCE,
            "version": VERSION,
        },
        order="(source, episode, day, item)",
    ),
    "orders": Table(
        {
            "episode": Column("String"),
            "seat": Column("UInt8", number),
            "day": Column("UInt16", number),
            "hour": Column("UInt8", number),
            "verb": Column("LowCardinality(String)"),
            # A HIRE carries no item and no quantity, so both are absent
            # rather than empty, and `text`/`number` say what absent becomes.
            "item": Column("LowCardinality(String)"),
            "quantity": Column("Int32", number),
            "source": SOURCE,
        },
        order="(source, episode, seat, day, hour)",
        replacing=False,
    ),
    "moves": Table(
        {
            "episode": Column("String"),
            "seat": Column("UInt8", number),
            "day": Column("UInt16", number),
            "hour": Column("UInt8", number),
            # The farmer is -1 and the hands are 0 upward, so this is signed.
            "actor": Column("Int8", number),
            "verb": Column("LowCardinality(String)"),
            "argument": Column("LowCardinality(String)"),
            "source": SOURCE,
        },
        order="(source, episode, seat, day, hour)",
        replacing=False,
    ),
    "candidate": Table(
        {
            "episode": Column("String"),
            "seat": Column("UInt8", number),
            "team": Column("LowCardinality(String)"),
            "matchup": Column("UInt16", number),
            "season": Column("UInt16", number),
            "source": SOURCE,
            "version": VERSION,
        },
        # Partitioned like the rest even though every row of it is ours: a
        # table that carries a source and cannot be swapped a source at a time
        # is one the rebuild has to special-case, and the special case is the
        # thing partitioning replaced.
        order="episode",
    ),
    "teams": Table(
        {
            "team": Column("String"),
            "games": Column("UInt32", number),
            "wins": Column("UInt32", number),
            "rating": Column("Float64", real),
            "place": Column("UInt32", number),
        },
        order="team",
        partition="",
    ),
}


def written(table: str) -> list[str]:
    """The columns a writer supplies, in order."""
    return [name for name, column in TABLES[table].columns.items() if column.written]


def schema(database: str = DATABASE) -> list[str]:
    """One `CREATE TABLE` per table, derived from `TABLES`.

    Derived rather than written, so the declaration and everything read off it
    -- the insert's column list, the casts, the reconciliation -- come from one
    place and cannot disagree.

    Args:
        database: The database to declare the tables in.

    Returns:
        One statement per table, in `TABLES` order.
    """
    statements = []
    for name, table in TABLES.items():
        columns = ",\n    ".join(
            f"{column} {spec.kind}" for column, spec in table.columns.items()
        )
        engine = "ReplacingMergeTree(version)" if table.replacing else "MergeTree"
        if table.replacing and "version" not in table.columns:
            engine = "ReplacingMergeTree"
        partition = f"PARTITION BY {table.partition}\n" if table.partition else ""
        statements.append(
            f"CREATE TABLE IF NOT EXISTS {database}.{name} (\n"
            f"    {columns}\n"
            f") ENGINE = {engine}\n"
            f"{partition}"
            f"ORDER BY {table.order}"
        )
    return statements


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
        """Hold one row for ``table``, cast to its own columns.

        The caller supplies every column but `source`, which every table has
        and every row of one batch shares. `strict` because a row with the
        wrong number of values is a row every column after the gap lands one
        to the left of -- which parses, and is wrong.
        """
        columns = [
            spec
            for name, spec in TABLES[table].columns.items()
            if spec.written and name != "source"
        ]
        row = [spec.cast(value) for spec, value in zip(columns, values, strict=True)]
        self.rows.setdefault(table, []).append(_row([*row, self.source]))

    def send(self, database: str = DATABASE) -> dict[str, int]:
        """Insert everything held, one request per table, and forget it.

        Returns:
            How many rows each table took.
        """
        sent = {}
        for table, rows in self.rows.items():
            _insert(table, ", ".join(written(table)), rows, database)
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
    """Make every table, and bring an existing one up to the schema.

    `CREATE TABLE IF NOT EXISTS` is silent about a table that already exists,
    so a column added here would never reach a database built before it. The
    first sign of that is an insert failing on a real evaluation -- which is
    how it was found, the loop dying on `No such column source in table
    games.candidate` -- because every test builds a fresh database and a fresh
    database matches by construction.

    So the columns are reconciled afterwards: anything the schema names and
    the table lacks is added. ClickHouse adds a column to a MergeTree as
    metadata, so this is cheap however large the table.
    """
    query(f"CREATE DATABASE IF NOT EXISTS {database}")
    for statement in schema(database):
        query(statement)
    _reconcile(database)


def _reconcile(database: str) -> None:
    """Add every column `TABLES` declares that the live table does not have."""
    for name, table in TABLES.items():
        live = set(
            query(
                f"SELECT name FROM system.columns WHERE database = '{database}' "
                f"AND table = '{name}' FORMAT TabSeparated"
            ).split()
        )
        for column, spec in table.columns.items():
            if live and column not in live:
                LOGGER.warning("%s.%s lacks %s; adding it", database, name, column)
                query(f"ALTER TABLE {database}.{name} ADD COLUMN {column} {spec.kind}")


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
