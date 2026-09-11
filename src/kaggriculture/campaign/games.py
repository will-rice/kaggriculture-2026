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

What this module does not hold is the extract a round is handed. That stays a
small SQLite beside `child.py` -- see `browse` -- because a round has no
credentials, no network and no business with the ladder's games or an
opponent's name.
"""

import logging
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Sequence

from kaggriculture.campaign import config, dataset, harness

LOGGER = logging.getLogger(__name__)

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


def column_type(measure: str) -> str:
    """The narrowest type that holds every value this measure takes."""
    if measure in REAL:
        return "Float32"
    return "UInt32" if measure in WIDE else "UInt16"


def schema() -> str:
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
CREATE TABLE IF NOT EXISTS games.episodes (
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
ORDER BY (source, episode);

CREATE TABLE IF NOT EXISTS games.days (
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

CREATE TABLE IF NOT EXISTS games.holdings (
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

CREATE TABLE IF NOT EXISTS games.prices (
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

CREATE TABLE IF NOT EXISTS games.orders (
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

CREATE TABLE IF NOT EXISTS games.moves (
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

CREATE TABLE IF NOT EXISTS games.candidate (
    episode    String,
    seat       UInt8,
    team       LowCardinality(String),
    matchup    UInt16,
    season     UInt16,
    version    UInt64 DEFAULT toUnixTimestamp64Milli(now64(3))
) ENGINE = ReplacingMergeTree(version)
ORDER BY episode;
"""


def load(sqlite: str = "/var/lib/clickhouse/user_files/games.sqlite") -> dict[str, int]:
    """Replace the recorded ladder from the SQLite the extraction builds.

    The extraction still writes SQLite: it parses twenty-five archives across
    eight processes into shards and merges them, and that is a good shape for
    a parse. What it is not is a good shape for a question. So the merged file
    is loaded here, server-side -- ClickHouse reads SQLite itself, so no export,
    no CSV and no Python stands between the two, and 271 million rows land in
    about ninety seconds.

    `DROP PARTITION 'ladder'` is how the ladder is replaced. The tables are
    partitioned by source, so the rebuild removing everything it owns and
    nothing else is a property of the storage rather than a WHERE clause
    somebody has to keep correct. The campaign's own games are in the other
    partition and are not touched.

    Args:
        sqlite: The merged extraction, as the server sees it.

    Returns:
        Rows now in each table.
    """
    create()
    measures = ", ".join(dataset.COLUMNS)
    plans = {
        "episodes": (
            "episode, kaggle_id, seed, engine, played, team_0, team_1, "
            "bank_0, bank_1, winner, source",
            "episode, ifNull(kaggle_id, 0), ifNull(seed, 0), ifNull(engine, ''), "
            "ifNull(played, ''), ifNull(team_0, ''), ifNull(team_1, ''), "
            "ifNull(bank_0, 0), ifNull(bank_1, 0), ifNull(winner, -1), 'ladder'",
        ),
        "days": (
            f"episode, seat, day, team, source, {measures}",
            f"episode, seat, day, ifNull(team, ''), 'ladder', {measures}",
        ),
        "holdings": (
            "episode, seat, day, kind, item, count, source",
            "episode, seat, day, kind, item, count, 'ladder'",
        ),
        "prices": (
            "episode, day, item, price, stock, source",
            "episode, day, item, price, ifNull(stock, 0), 'ladder'",
        ),
        "orders": (
            "episode, seat, day, hour, verb, item, quantity, source",
            "episode, seat, day, hour, verb, ifNull(item, ''), "
            "ifNull(quantity, 0), 'ladder'",
        ),
        "moves": (
            "episode, seat, day, hour, actor, verb, argument, source",
            "episode, seat, day, hour, actor, verb, argument, 'ladder'",
        ),
    }
    landed = {}
    for table, (columns, select) in plans.items():
        # `episodes` is not partitioned -- it is small and the ordering key
        # starts with source, so a delete is cheap and exact.
        if table == "episodes":
            query(
                f"ALTER TABLE games.{table} DELETE WHERE source = 'ladder' "
                "SETTINGS mutations_sync = 1"
            )
        else:
            query(f"ALTER TABLE games.{table} DROP PARTITION 'ladder'")
        query(
            f"INSERT INTO games.{table} ({columns}) SELECT {select} "
            f"FROM sqlite('{sqlite}', '{table}')"
        )
        landed[table] = int(query(f"SELECT count() FROM games.{table}"))
        LOGGER.info("%s: %d rows", table, landed[table])
    return landed


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


def create() -> None:
    """Make every table, if it is not already there."""
    for statement in schema().split(";"):
        if statement.strip():
            query(statement)


# How each measure is written, decided by the same function that decided its
# column type. `dataset.measures` returns every quantity as a float -- a tile
# count arrives as `4.0` -- and TabSeparated will not parse "4.0" into a
# `UInt16`. Reading the cast off `column_type` rather than off a second list
# means the value written and the column it is written into cannot disagree.
_CAST = tuple(
    float if column_type(measure) == "Float32" else int for measure in dataset.COLUMNS
)


def record(name: str, played: Sequence[tuple[int, int, str, harness.Game]]) -> int:
    """Write one program's games, in parallel with whatever else is writing.

    No lock and no delete. Two sessions recording at once are two POSTs, and a
    program recorded twice inserts twice and is collapsed by the engine on the
    ordering key, so a re-scored champion is one program rather than two.

    Args:
        name: The program these games belong to; prefixes their episode keys.
        played: One entry per game, as `browse.games` returns them.

    Returns:
        How many games were written.
    """
    if not played:
        return 0
    rows: list[str] = []
    holdings: list[str] = []
    prices: list[str] = []
    episodes: list[str] = []
    for matchup, season, mine, game in played:
        episode = f"{name}m{matchup}s{season}"
        theirs = 1 - game.seat
        episodes.append(
            _row(
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
                    "campaign",
                ]
            )
        )
        for day in game.days:
            for side in ("ours", "theirs"):
                seat = game.seat if side == "ours" else theirs
                measures = getattr(day, side)
                rows.append(
                    _row(
                        [
                            episode,
                            seat,
                            day.day,
                            mine if side == "ours" else "opponent",
                            "campaign",
                            *(
                                cast(measures.get(column, 0))
                                for cast, column in zip(
                                    _CAST, dataset.COLUMNS, strict=True
                                )
                            ),
                        ]
                    )
                )
                for kind in ("plants", "animals", "seeds", "shed"):
                    counts = getattr(day, f"{side}_{kind}", None) or {}
                    holdings.extend(
                        _row([episode, seat, day.day, kind, item, n, "campaign"])
                        for item, n in counts.items()
                    )
            prices.extend(
                _row([episode, day.day, item, price, 0, "campaign"])
                for item, price in day.prices.items()
            )
    _insert("episodes", _EPISODE_COLUMNS, episodes)
    _insert("days", _DAY_COLUMNS, rows)
    _insert("holdings", _HOLDING_COLUMNS, holdings)
    _insert("prices", _PRICE_COLUMNS, prices)
    _insert(
        "candidate",
        "episode, seat, team, matchup, season",
        [
            _row([f"{name}m{matchup}s{season}", game.seat, mine, matchup, season])
            for matchup, season, mine, game in played
        ],
    )
    return len(played)


_EPISODE_COLUMNS = (
    "episode, kaggle_id, seed, engine, played, team_0, team_1, "
    "bank_0, bank_1, winner, source"
)
_DAY_COLUMNS = "episode, seat, day, team, source, " + ", ".join(dataset.COLUMNS)
_HOLDING_COLUMNS = "episode, seat, day, kind, item, count, source"
_PRICE_COLUMNS = "episode, day, item, price, stock, source"


def _insert(table: str, columns: str, rows: Sequence[str]) -> None:
    """POST rows into one table as tab-separated values."""
    if rows:
        query(
            f"INSERT INTO games.{table} ({columns}) FORMAT TabSeparated",
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
