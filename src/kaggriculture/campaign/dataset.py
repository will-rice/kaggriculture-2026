"""Every recorded game as a queryable table, so a question costs a query.

`paired.tally` answers one shape of question -- does the winner lead on this
quantity on this day -- and answers it by re-reading the corpus, which is
twenty-odd minutes for every question anyone thinks of. That is the wrong way
round: the parse is what costs, and it is the same parse whatever is being
asked. So the corpus is parsed once into SQLite and the questions become SQL.

It also fixes what the questions could be about. Eleven numbers per side per
day, read from one hour in twenty-four, is a small corner of what a tape
holds. Discarded until now: which agents played (every episode names both,
and half the games on the ladder are won by the weaker one, which is why
"what winners do" hovers at 50%); market orders' items and quantities, so
"sells more" counted orders rather than produce; the farmer's and the hands'
commands, which are the policy itself; and every tile's husbandry -- whether
it was watered today, how many days it has gone without, whether the animals
were fed and cared for.

Aggregate only, like `winning-pace`: this reads public replays and writes
counts. No opponent's source is read, and nothing here is a path.

Shape, with rough row counts over the twenty archives now on disk:

    episodes   14k   one per game: who played, what they banked, who won
    days      840k   one per (game, seat, day): the state at that day's close
    holdings    5M   non-zero seed, shed, crop and animal counts, per day
    orders      2M   every market order: verb, item, quantity, at which hour
    moves       8M   every farmer and hand command that was not a PASS

Tiles are aggregated into `days` rather than stored one row each: a hundred
tiles per side per day is eighty million rows to answer questions that the
counts answer.
"""

import logging
import sqlite3
import tempfile
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from tqdm import tqdm

from kaggriculture.campaign import config, tapes

LOGGER = logging.getLogger(__name__)

# Beside the archives it is built from, because it is the same data in another
# shape and it is far too big for the repository.
DATABASE = config.EPISODES.parent / "corpus.sqlite"
# Processes to divide the archives over. The same reasoning as
# `paired.WORKERS`: the campaign loop is usually running while this is.
WORKERS = 8
# A season, and the hours in a day. A day's row is that day at its last hour,
# which is what both players saw before their final decision in it.
DAYS = 30
HOURS = 24
LAST_HOUR = HOURS - 1
# What a tile says it is when it holds nothing worth counting.
BARE = ("EMPTY", "WEED", "TILLED")

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    episode    TEXT PRIMARY KEY,
    kaggle_id  INTEGER,
    seed       INTEGER,
    engine     TEXT,
    played     TEXT,
    team_0     TEXT,
    team_1     TEXT,
    bank_0     REAL,
    bank_1     REAL,
    winner     INTEGER
);
CREATE TABLE IF NOT EXISTS days (
    episode    TEXT,
    seat       INTEGER,
    day        INTEGER,
    team       TEXT,
    bank       REAL,
    planted    INTEGER,
    ripe       INTEGER,
    yield_held INTEGER,
    pens       INTEGER,
    weeds      INTEGER,
    bare       INTEGER,
    quadrants  INTEGER,
    hands      INTEGER,
    hires      INTEGER,
    seeds      INTEGER,
    shed       INTEGER,
    shops      INTEGER,
    watered    INTEGER,
    dry_worst  INTEGER,
    fertilised INTEGER,
    fed        INTEGER,
    hungry_worst INTEGER,
    cared      INTEGER,
    plant_age  REAL
);
CREATE TABLE IF NOT EXISTS holdings (
    episode    TEXT,
    seat       INTEGER,
    day        INTEGER,
    kind       TEXT,
    item       TEXT,
    count      INTEGER
);
CREATE TABLE IF NOT EXISTS orders (
    episode    TEXT,
    seat       INTEGER,
    day        INTEGER,
    hour       INTEGER,
    verb       TEXT,
    item       TEXT,
    quantity   INTEGER
);
CREATE TABLE IF NOT EXISTS moves (
    episode    TEXT,
    seat       INTEGER,
    day        INTEGER,
    hour       INTEGER,
    actor      INTEGER,
    verb       TEXT,
    argument   TEXT
);
CREATE TABLE IF NOT EXISTS prices (
    episode    TEXT,
    day        INTEGER,
    item       TEXT,
    price      INTEGER,
    stock      INTEGER
);
"""
# Built after the load, not before: an index makes every insert cost a tree
# walk, and there are twenty million of them.
INDEXES = """
CREATE INDEX IF NOT EXISTS days_episode ON days (episode, seat);
CREATE INDEX IF NOT EXISTS days_team ON days (team, day);
CREATE INDEX IF NOT EXISTS holdings_day ON holdings (episode, seat, day);
CREATE INDEX IF NOT EXISTS orders_day ON orders (episode, seat, day);
CREATE INDEX IF NOT EXISTS moves_day ON moves (episode, seat, day);
CREATE INDEX IF NOT EXISTS prices_day ON prices (episode, day);
"""
TABLES = ("episodes", "days", "holdings", "orders", "moves", "prices")


def build(corpus: list[Path], database: Path = DATABASE, workers: int = WORKERS) -> int:
    """Read every game in ``corpus`` into ``database``, replacing it.

    One process per archive, each writing its own SQLite file, then merged.
    Merged rather than written into one database because SQLite takes a
    single writer and the parse is what we are trying to parallelise; a
    worker that spent its time waiting on a write lock would divide nothing.

    Args:
        corpus: The archives to read, from `tapes.archives`.
        database: Where to write. Replaced, not appended to -- a half-built
            dataset that looks complete is worse than no dataset.
        workers: Processes to divide the archives over.

    Returns:
        How many episodes were read.
    """
    database.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(dir=database.parent) as scratch:
        shards = [Path(scratch) / f"{archive.stem}.sqlite" for archive in corpus]
        with ProcessPoolExecutor(max_workers=min(workers, len(corpus))) as pool:
            done = pool.map(_shard, corpus, shards)
            counts = list(tqdm(done, total=len(corpus), desc="archives"))
        LOGGER.info("read %d episodes; merging %d shards", sum(counts), len(shards))
        return _merge(shards, database)


def _shard(archive: Path, out: Path) -> int:
    """Read one archive into its own database. Runs in a worker process."""
    played = archive.stem[-10:]
    connection = sqlite3.connect(out)
    connection.executescript(SCHEMA)
    read = 0
    for episode in tapes.qualifying_episodes(archive):
        _insert(connection, episode, played)
        read += 1
    connection.commit()
    connection.close()
    return read


def _merge(shards: list[Path], database: Path) -> int:
    """Fold every shard into one database and index it."""
    connection = sqlite3.connect(database)
    connection.executescript(SCHEMA)
    for shard in shards:
        connection.execute("ATTACH DATABASE ? AS shard", (str(shard),))
        for table in TABLES:
            connection.execute(f"INSERT INTO {table} SELECT * FROM shard.{table}")  # noqa: S608 - table names are this module's own constants
        connection.commit()
        connection.execute("DETACH DATABASE shard")
    connection.executescript(INDEXES)
    connection.commit()
    total = connection.execute("SELECT count(*) FROM episodes").fetchone()[0]
    connection.close()
    return total


def _insert(
    connection: sqlite3.Connection, episode: tapes.Episode, played: str
) -> None:
    """Write every row one recorded game produces."""
    identity = episode.info
    key = str(identity.get("EpisodeId") or episode.seed)
    final = episode.steps[-1][0]["observation"]["farms"]
    banks = [float(farm["money"]) for farm in final]
    teams = list(identity.get("TeamNames") or ["", ""])
    winner = None if banks[0] == banks[1] else int(banks[1] > banks[0])
    connection.execute(
        "INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            key,
            identity.get("EpisodeId"),
            episode.seed,
            episode.engine_version,
            played,
            teams[0],
            teams[1],
            banks[0],
            banks[1],
            winner,
        ),
    )
    connection.executemany(
        "INSERT INTO orders VALUES (?,?,?,?,?,?,?)", _orders(episode, key)
    )
    connection.executemany(
        "INSERT INTO moves VALUES (?,?,?,?,?,?,?)", _moves(episode, key)
    )
    for day in range(DAYS):
        # Addressed by day rather than walked, which is why there is no
        # special case for the last one here: day 29 hour 23 is step 719 and
        # a season is exactly 720 steps, so the close of the season is the
        # final state by construction. `paired.sides` walks the steps instead
        # and needs an extra line to reach the same state; it lost day 29
        # twice for want of it. A short tape raises here rather than filing
        # some mid-season state under day 29.
        step = episode.steps[day * HOURS + LAST_HOUR]
        for seat in (0, 1):
            observation = step[seat]["observation"]
            connection.execute(
                "INSERT INTO days VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (key, seat, day, teams[seat], *_day(observation, seat, day)),
            )
            connection.executemany(
                "INSERT INTO holdings VALUES (?,?,?,?,?,?)",
                ((key, seat, day, *row) for row in _holdings(observation, seat)),
            )
        market = step[0]["observation"]["market"]
        connection.executemany(
            "INSERT INTO prices VALUES (?,?,?,?,?)",
            (
                (key, day, item, price, (market.get("inventory") or {}).get(item))
                for item, price in (market.get("prices") or {}).items()
            ),
        )


def _tiles(farm: dict[str, Any]) -> list[dict[str, Any]]:
    """Every tile of one farm, flattened out of its rows."""
    return [tile for row in farm["tiles"] for tile in row if isinstance(tile, dict)]


def _day(observation: dict[str, Any], seat: int, day: int) -> tuple:
    """One side's whole state at one day's close, tiles included.

    The tile columns are the half of the game the counts miss. A side holding
    forty growing tiles of which six were watered today is playing a different
    game from one holding forty of which forty were, and until now both read
    as "planted 40".
    """
    farm = observation["farms"][seat]
    private = observation.get("private") or {}
    tiles = _tiles(farm)
    crops = [tile for tile in tiles if tile.get("crop")]
    pens = [tile for tile in tiles if tile.get("animal")]
    ages = [day - tile.get("planted_day", day) for tile in crops]
    return (
        float(farm["money"]),
        len(crops),
        sum(1 for tile in tiles if tile.get("yield_units", 0) > 0),
        sum(int(tile.get("yield_units", 0)) for tile in tiles),
        len(pens),
        sum(1 for tile in tiles if tile.get("kind") == "WEED"),
        sum(1 for tile in tiles if tile.get("kind") in BARE),
        len(farm.get("unlocked_quadrants") or []),
        len(farm.get("hands") or []),
        int(farm.get("hires_today") or 0),
        sum((private.get("seeds") or {}).values()),
        sum((private.get("shed") or {}).values()),
        len((observation.get("town") or {}).get("unlocked_shops") or []),
        sum(1 for tile in crops if tile.get("watered_today")),
        max((int(tile.get("consecutive_unwatered", 0)) for tile in crops), default=0),
        sum(1 for tile in crops if int(tile.get("fertilized_until_day", -1)) > day),
        sum(1 for tile in pens if tile.get("fed_today")),
        max((int(tile.get("consecutive_unfed", 0)) for tile in pens), default=0),
        sum(1 for tile in pens if tile.get("cared_today")),
        sum(ages) / len(ages) if ages else 0.0,
    )


def _holdings(observation: dict[str, Any], seat: int) -> Iterator[tuple]:
    """Non-zero counts of everything one side holds, by kind and item.

    Long rather than one column per commodity: there are seventeen of them
    across four kinds, the engine has added some before, and a query that
    pivots is easier to write than a migration.
    """
    farm = observation["farms"][seat]
    private = observation.get("private") or {}
    for kind, counts in (("seed", private.get("seeds")), ("shed", private.get("shed"))):
        for item, count in (counts or {}).items():
            if count:
                yield kind, item, int(count)
    grown: dict[str, int] = {}
    kept: dict[str, int] = {}
    for tile in _tiles(farm):
        if tile.get("crop"):
            grown[tile["crop"]] = grown.get(tile["crop"], 0) + 1
        elif tile.get("animal"):
            kept[tile["animal"]] = kept.get(tile["animal"], 0) + 1
    for kind, counts in (("plant", grown), ("animal", kept)):
        for item, count in counts.items():
            yield kind, item, count


def _orders(episode: tapes.Episode, key: str) -> Iterator[tuple]:
    """Every market order either side submitted, with its item and quantity.

    Counting the verb alone -- which is all anything read before this -- makes
    "sells more" a claim about how many orders were sent rather than how much
    produce moved, and those are different games.
    """
    for index in range(1, len(episode.steps)):
        day, hour = divmod(index - 1, HOURS)
        for seat in (0, 1):
            action = episode.steps[index][seat].get("action") or {}
            for order in action.get("market") or ():
                parts = order if isinstance(order, list) else [order]
                yield (
                    key,
                    seat,
                    day,
                    hour,
                    str(parts[0]),
                    str(parts[1]) if len(parts) > 1 else None,
                    int(parts[2]) if len(parts) > 2 else None,
                )


def _moves(episode: tapes.Episode, key: str) -> Iterator[tuple]:
    """Every farmer and hand command that was not a PASS.

    A PASS is the absence of a decision and there are millions of them; a
    query counting moves per day gets the same answer from what is here.
    ``actor`` is -1 for the farmer and the hand's own index otherwise, so a
    query can ask about the farmer alone without knowing how many hands a
    side had hired.
    """
    for index in range(1, len(episode.steps)):
        day, hour = divmod(index - 1, HOURS)
        for seat in (0, 1):
            action = episode.steps[index][seat].get("action") or {}
            commands = [(-1, action.get("farmer") or [])]
            commands += list(enumerate(action.get("hands") or []))
            for actor, command in commands:
                tokens = [str(token) for token in command]
                if not tokens or tokens[0] == "PASS":
                    continue
                yield (
                    key,
                    seat,
                    day,
                    hour,
                    actor,
                    tokens[0],
                    " ".join(tokens[1:]) or None,
                )


def counts(database: Path = DATABASE) -> dict[str, int]:
    """Rows per table, for a caller that wants to say what it built."""
    connection = sqlite3.connect(database)
    try:
        return {
            table: connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]  # noqa: S608 - table names are this module's own constants
            for table in TABLES
        }
    finally:
        connection.close()


def summarise(database: Path = DATABASE) -> str:
    """What the dataset holds, as lines: rows per table and the field it saw."""
    lines = [f"{table:<10}{rows:>12,}" for table, rows in counts(database).items()]
    connection = sqlite3.connect(database)
    try:
        teams = connection.execute(
            "SELECT count(DISTINCT team) FROM (SELECT team_0 AS team FROM episodes "
            "UNION SELECT team_1 FROM episodes)"
        ).fetchone()[0]
        span = connection.execute(
            "SELECT min(played), max(played) FROM episodes"
        ).fetchone()
    finally:
        connection.close()
    lines.append(f"{teams} distinct teams, {span[0]} to {span[1]}")
    return "\n".join(lines)


def leaderboard(database: Path = DATABASE, least: int = 30) -> list[tuple]:
    """Every team's record over the corpus, most games won first.

    A win rate, not a rating: it says who wins and takes no view on schedule
    strength. `rating.standings` fits the Bradley-Terry model this feeds.

    Args:
        database: The dataset.
        least: Ignore teams with fewer games than this; a team that played
            twice and won both is not the best agent on the ladder.
    """
    connection = sqlite3.connect(database)
    try:
        return connection.execute(
            """
            WITH seats AS (
                SELECT team_0 AS team, winner = 0 AS won FROM episodes
                WHERE winner IS NOT NULL
                UNION ALL
                SELECT team_1, winner = 1 FROM episodes WHERE winner IS NOT NULL
            )
            SELECT team, count(*) AS games, avg(won) AS rate
            FROM seats GROUP BY team HAVING games >= ?
            ORDER BY rate DESC
            """,
            (least,),
        ).fetchall()
    finally:
        connection.close()
