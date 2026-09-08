"""Every recorded game as a queryable table, so a question costs a query.

Every question about the recorded games used to mean another walk of the
archives: twenty-odd minutes to learn one number, and the same parse each
time whatever was being asked. That is the wrong way round -- the parse is
the cost and it does not depend on the question -- so the corpus is parsed
once into SQLite and the questions become SQL.

It also fixes what the questions could be about. Eleven numbers per side per
day, read from one hour in twenty-four, is a small corner of what a tape
holds. Discarded until now: which agents played (every episode names both,
and half the games on the ladder are won by the weaker one, which is why
"what winners do" hovers at 50%); market orders' items and quantities, so
"sells more" counted orders rather than produce; the farmer's and the hands'
commands, which are the policy itself; and every tile's husbandry -- whether
it was watered today, how many days it has gone without, whether the animals
were fed and cared for.

Aggregate only: this reads public replays and writes
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

from kaggriculture.campaign import config, rating, tapes

LOGGER = logging.getLogger(__name__)

# Beside the archives it is built from, because it is the same data in another
# shape and it is far too big for the repository.
DATABASE = config.EPISODES.parent / "corpus.sqlite"
# Processes to divide the archives over. The same reasoning as
# the loop is usually running while this is, and the loop is what must not
# slow down.
WORKERS = 8
# The fewest games a team must have played to be rated. A team that played
# twice and won both is not the strongest agent on the ladder, and a rating
# fitted from two games is the prior wearing a number.
LEAST = 40
# How many days back a rating and a build order read. The ladder is not a
# fixed field: it moves under everyone, every day.
#
# Measured 2026-09-08. A byte-identical agent scored 2386.8 on 2026-09-03 and
# 1418.0 five days later. Splitting the corpus in half and fitting each half
# on its own, the two top tens share not one name -- complete turnover in
# twelve days -- and teams whose games fall mostly in the early half rate 1.1
# log-odds lower in a pooled fit purely for having played then.
#
# Bradley-Terry has no notion of time, so fitted over everything it reads
# "played against a weaker field" as "strong" and a build order averaged over
# it describes a blend of fields, most of which no longer exists. Ten days is
# the window whose fit matches today's public leaderboard, and it still leaves
# some seven thousand games -- enough that a claim can clear `SUPPORT`.
WINDOW = 10
# A season, and the hours in a day. A day's row is that day at its last hour,
# which is what both players saw before their final decision in it.
DAYS = 30
HOURS = 24
LAST_HOUR = HOURS - 1
# What a tile says it is when it holds nothing worth counting.
BARE = ("EMPTY", "WEED", "TILLED")
# The market verbs a day row carries a running total of, as
# ``column stem: verb``. Cumulative through that day and stored on the day
# rather than left to be summed out of `orders`, so that comparing two sides
# on what they have bought and sold is the same self-join as comparing them
# on anything else -- and so that it is one indexed row rather than a scan of
# twenty-three million.
#
# Both the count and the units: "sells more" measured as a count of orders
# said the stronger side sells more, and measured in units it sells less than
# half as much. They are different claims and the corpus answers them
# differently.
TALLIED = (
    ("sell_orders", "sold_units", "SELL"),
    ("buy_orders", "bought_units", "BUY_PRODUCT"),
    ("seed_orders", "seed_units", "BUY_SEED"),
    ("animal_orders", "animal_units", "BUY_ANIMAL"),
)
# Counted but never summed: neither carries a quantity in the tape.
COUNTED = (("hire_orders", "HIRE"), ("land_orders", "BUY_LAND"))

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
    seeds      INTEGER,
    shed       INTEGER,
    shops      INTEGER,
    watered    INTEGER,
    dry_worst  INTEGER,
    fertilised INTEGER,
    fed        INTEGER,
    hungry_worst INTEGER,
    cared      INTEGER,
    plant_age  REAL,
    sell_orders   INTEGER,
    sold_units    INTEGER,
    buy_orders    INTEGER,
    bought_units  INTEGER,
    seed_orders   INTEGER,
    seed_units    INTEGER,
    animal_orders INTEGER,
    animal_units  INTEGER,
    hire_orders   INTEGER,
    land_orders   INTEGER
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
CREATE TABLE IF NOT EXISTS teams (
    team       TEXT PRIMARY KEY,
    games      INTEGER,
    wins       INTEGER,
    rating     REAL,
    place      INTEGER
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
# The day row's own columns, in declaration order and without its keys. Read
# off the schema so a column added there is measured, stored and asked about
# with nothing to remember.
COLUMNS = tuple(
    line.split()[0]
    for line in SCHEMA[SCHEMA.index("days (") : SCHEMA.index("holdings (")].splitlines()
    if line.startswith("    ")
)[4:]

TABLES = ("episodes", "days", "holdings", "orders", "moves", "prices")
# Written by `rate` after the load, from what the load produced.
DERIVED = ("teams",)


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
    submitted = list(_orders(episode, key))
    connection.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?)", submitted)
    running = _running(submitted)
    connection.executemany(
        "INSERT INTO moves VALUES (?,?,?,?,?,?,?)", _moves(episode, key)
    )
    for day in range(DAYS):
        # Addressed by day rather than walked, which is why there is no
        # special case for the last one here: day 29 hour 23 is step 719 and
        # a season is exactly 720 steps, so the close of the season is the
        # final state by construction. The walk this replaced stepped through
        # the tape instead and needed an extra line to reach that state; it
        # silently had no day 29 at all, twice. A short tape raises here
        # rather than filing some mid-season state under day 29.
        step = episode.steps[day * HOURS + LAST_HOUR]
        for seat in (0, 1):
            observation = step[seat]["observation"]
            connection.execute(
                "INSERT INTO days VALUES (" + ",".join("?" * (4 + len(COLUMNS))) + ")",
                (
                    key,
                    seat,
                    day,
                    teams[seat],
                    *_row(observation, seat, day, running[seat][day]),
                ),
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


def _running(submitted: list[tuple]) -> dict[int, list[dict[str, list[int]]]]:
    """Each side's market activity as a running total through every day.

    Cumulative rather than per-day, because every claim about trading is about
    the season so far -- "has sold more by day twenty" -- and a per-day figure
    would make one quiet day look like a different strategy.

    Args:
        submitted: The rows `_orders` produced for one game.

    Returns:
        ``{seat: [{verb: [orders, units]} for each day]}``.
    """
    counts: dict[int, dict[int, dict[str, list[int]]]] = {0: {}, 1: {}}
    for _, seat, day, _, verb, _, quantity in submitted:
        tally = counts[seat].setdefault(day, {}).setdefault(verb, [0, 0])
        tally[0] += 1
        tally[1] += int(quantity or 0)
    out: dict[int, list[dict[str, list[int]]]] = {}
    for seat in (0, 1):
        total: dict[str, list[int]] = {}
        rows = []
        for day in range(DAYS):
            for verb, tally in counts[seat].get(day, {}).items():
                running = total.setdefault(verb, [0, 0])
                running[0] += tally[0]
                running[1] += tally[1]
            rows.append({verb: list(pair) for verb, pair in total.items()})
        out[seat] = rows
    return out


def _tiles(farm: dict[str, Any]) -> list[dict[str, Any]]:
    """Every tile of one farm, flattened out of its rows."""
    return [tile for row in farm["tiles"] for tile in row if isinstance(tile, dict)]


def measures(
    farm: dict[str, Any],
    private: dict[str, Any],
    town: dict[str, Any],
    day: int,
    traded: dict[str, list[int]],
) -> dict[str, float]:
    """Every quantity one side can be compared on, at one day's close.

    The single definition of each. The extraction reads it off a recorded tape
    and the harness reads it off a live engine, and both hand it the same
    shapes -- the engine's `render` is the function that produced the tape in
    the first place. Written twice they would drift, and a claim measured on
    the corpus definition and selected on the harness definition would be two
    quantities wearing one name: a comparison that agrees with itself and is
    wrong.

    Args:
        farm: One side's public farm, as an observation renders it.
        private: That side's ``seeds`` and ``shed``. Private in play, and
            shown to a program's author afterwards.
        town: The shared town, for the shops unlocked.
        day: The day being closed, which the fertilizer and age terms need.
        traded: Cumulative market activity as ``{verb: [orders, units]}``.
            Empty where nobody counted, which is how the extraction reads a
            state without replaying the orders that reached it.

    Returns:
        One value per name in `COLUMNS`.
    """
    tiles = [tile for row in farm["tiles"] for tile in row if isinstance(tile, dict)]
    crops = [tile for tile in tiles if tile.get("crop")]
    pens = [tile for tile in tiles if tile.get("animal")]
    ages = [day - int(tile.get("planted_day", day)) for tile in crops]
    out: dict[str, float] = {
        "bank": float(farm["money"]),
        "planted": float(len(crops)),
        "ripe": float(sum(1 for tile in tiles if tile.get("yield_units", 0) > 0)),
        "yield_held": float(sum(int(tile.get("yield_units", 0)) for tile in tiles)),
        "pens": float(len(pens)),
        "weeds": float(sum(1 for tile in tiles if tile.get("kind") == "WEED")),
        "bare": float(sum(1 for tile in tiles if tile.get("kind") in BARE)),
        "quadrants": float(len(farm.get("unlocked_quadrants") or [])),
        # `hires_today` is not a second quantity: over all 977,520 day rows of
        # the corpus it never once differed from the hand count, so measuring
        # both put the same finding in two claims and spent two of the five
        # slots a round is shown on one fact.
        "hands": float(len(farm.get("hands") or [])),
        "seeds": float(sum((private.get("seeds") or {}).values())),
        "shed": float(sum((private.get("shed") or {}).values())),
        "shops": float(len((town or {}).get("unlocked_shops") or [])),
        "watered": float(sum(1 for tile in crops if tile.get("watered_today"))),
        "dry_worst": float(
            max(
                (int(tile.get("consecutive_unwatered", 0)) for tile in crops),
                default=0,
            )
        ),
        "fertilised": float(
            sum(1 for tile in crops if int(tile.get("fertilized_until_day", -1)) > day)
        ),
        "fed": float(sum(1 for tile in pens if tile.get("fed_today"))),
        "hungry_worst": float(
            max((int(tile.get("consecutive_unfed", 0)) for tile in pens), default=0)
        ),
        "cared": float(sum(1 for tile in pens if tile.get("cared_today"))),
        "plant_age": sum(ages) / len(ages) if ages else 0.0,
    }
    for orders, units, verb in TALLIED:
        out[orders], out[units] = (float(n) for n in traded.get(verb, [0, 0]))
    for orders, verb in COUNTED:
        out[orders] = float(traded.get(verb, [0, 0])[0])
    return out


def _row(observation: dict[str, Any], seat: int, day: int, traded: dict) -> tuple:
    """One side's row, ordered as the `days` table declares its columns."""
    found = measures(
        observation["farms"][seat],
        observation.get("private") or {},
        observation.get("town") or {},
        day,
        traded,
    )
    return tuple(found[name] for name in COLUMNS)


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
            for table in TABLES + DERIVED
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


def rate(database: Path = DATABASE, least: int = LEAST, window: int = WINDOW) -> int:
    """Fit one Bradley-Terry strength per team and store it in ``teams``.

    A win rate says who won and takes no view on who they played; over a
    ladder where hundreds of teams meet unevenly that is most of the number.
    The agent this campaign was seeded from wins 74% of 668 games and rates
    46th of 160, and the difference between those two readings is the
    schedule.

    Only teams inside the largest connected group are rated: a team whose
    opponents all fell below ``least`` has no path to a comparison, and a
    rating fitted for it would be the prior wearing a number.

    Args:
        database: The dataset, already built.
        least: The fewest games a team must have played to be rated.
        window: Days back to read. The field turns over inside a fortnight,
            so a rating over everything rates two disjoint fields at once.

    Returns:
        How many teams were rated.
    """
    connection = sqlite3.connect(database)
    try:
        # `rate` is its own entry point, run against a database `build` may
        # have written before this table existed.
        connection.executescript(SCHEMA)
        played = connection.execute(
            "SELECT team_0, team_1, winner FROM episodes "
            "WHERE winner IS NOT NULL AND played >= ?",
            (recent(connection, window),),
        ).fetchall()
        rows = _rate(played, least)
        connection.execute("DELETE FROM teams")
        connection.executemany("INSERT INTO teams VALUES (?,?,?,?,?)", rows)
        connection.commit()
        return len(rows)
    finally:
        connection.close()


def recent(connection: sqlite3.Connection, window: int = WINDOW) -> str:
    """The first archive day inside the window, counted back from the newest.

    Off the data rather than off the clock. The fetch can be days behind --
    it was four days behind this morning -- and a window measured from today
    would then be half empty or wholly so, which is a rating over nothing
    rather than a rating over the recent field.
    """
    days = [
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT played FROM episodes ORDER BY played DESC"
        )
    ]
    return days[min(window, len(days)) - 1] if days else ""


def _rate(played: list[tuple], least: int) -> list[tuple]:
    """Every rated team as a `teams` row, strongest first."""
    counts: dict[str, list[int]] = {}
    for one, two, winner in played:
        counts.setdefault(one, [0, 0])[0] += 1
        counts.setdefault(two, [0, 0])[0] += 1
        counts[one][1] += int(winner == 0)
        counts[two][1] += int(winner == 1)
    kept = {team for team, (games, _) in counts.items() if games >= least}

    pairs: dict[tuple[str, str], list[int]] = {}
    for one, two, winner in played:
        if one == two or one not in kept or two not in kept:
            continue
        first, second = (one, two) if one < two else (two, one)
        tally = pairs.setdefault((first, second), [0, 0])
        tally[0] += int((winner == 0) == (one == first))
        tally[1] += 1

    joined = _connected(pairs)
    results = [
        (one, two, won / total, total)
        for (one, two), (won, total) in pairs.items()
        if one in joined and two in joined
    ]
    if not results:
        return []
    strengths = rating.standings(results)
    ordered = sorted(strengths.items(), key=lambda pair: -pair[1])
    return [
        (team, counts[team][0], counts[team][1], value, place)
        for place, (team, value) in enumerate(ordered, start=1)
    ]


def _connected(pairs: dict[tuple[str, str], list[int]]) -> set[str]:
    """The largest group of teams joined by games, so every rating compares."""
    neighbours: dict[str, set[str]] = {}
    for one, two in pairs:
        neighbours.setdefault(one, set()).add(two)
        neighbours.setdefault(two, set()).add(one)
    seen: set[str] = set()
    best: set[str] = set()
    for start in neighbours:
        if start in seen:
            continue
        group, queue = {start}, [start]
        while queue:
            for other in neighbours[queue.pop()]:
                if other not in group:
                    group.add(other)
                    queue.append(other)
        seen |= group
        if len(group) > len(best):
            best = group
    return best


def ladder(database: Path = DATABASE) -> list[tuple]:
    """Every rated team, strongest first, as `rate` stored them."""
    connection = sqlite3.connect(database)
    try:
        return connection.execute(
            "SELECT team, games, wins, rating, place FROM teams ORDER BY place"
        ).fetchall()
    finally:
        connection.close()


# The quantities a build order is stated in, and the only ones a round can act
# on: each is a column of the day table a round is already shown, so it can
# read its own number straight off the row beside it. `land_orders` is absent
# for that reason and quadrants stand in for it -- land is bought to unlock a
# quadrant, so the two move together and only one of them is visible in a game.
TARGETS = (
    "bank",
    "quadrants",
    "planted",
    "fertilised",
    "pens",
    "hands",
    "seeds",
    "shed",
    "weeds",
)
# How many rated agents the build order is read from, and the days it is
# stated on. Enough that no single agent's habits carry a column, and few
# enough that it is the top of the ladder rather than the middle of it -- so
# it is a share of the field rather than a count. Twenty-five was an eighth of
# the 185 teams a whole-corpus fit rated; the same eighth of the 82 a ten-day
# window rates is twelve, and taking twenty-five of those would be describing
# the top third.
BEST = 12
MARKS = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 17, 20, 23, 25, 27, 29)


def build_order(
    database: Path = DATABASE, best: int = BEST, window: int = WINDOW
) -> dict[str, list[float]]:
    """What the strongest agents hold on each day, averaged over their games.

    The `winning_pace` tables this replaces were medians over the winning side
    of every game, which is the wrong half of the corpus: about half of a
    ladder's winners are the weaker agent having a good day, and eleven
    quantities measured that way came back between 45% and 60%. Averaged over
    the top of a rating instead, the same games say something a round can act
    on -- and say it sharply, since the strongest separations in the whole
    corpus are quadrants and fertilizer in the first week.

    Args:
        database: The dataset, already built and rated.
        best: How many rated agents to read from, strongest first.
        window: Days back to average over, matching the rating's own window.
            Averaged over everything, the table blends fields that no longer
            play each other.

    Returns:
        ``{quantity: [value on each of MARKS]}``, and ``"day"`` itself.

    Raises:
        ValueError: Nothing is rated yet, so there is no top to read from.
    """
    top = [team for team, *_ in ladder(database)[:best]]
    if not top:
        raise ValueError(f"no rated teams in {database}; run `rate` first")
    marks = ",".join("?" * len(top))
    columns = ", ".join(f"avg({name})" for name in TARGETS)
    connection = sqlite3.connect(database)
    try:
        first = recent(connection, window)
        rows = [
            connection.execute(
                f"SELECT {columns} FROM days d "  # noqa: S608 - column names are this module's own constants
                f"JOIN episodes e ON e.episode = d.episode "
                f"WHERE d.day = ? AND d.team IN ({marks}) AND e.played >= ?",
                (day, *top, first),
            ).fetchone()
            for day in MARKS
        ]
    finally:
        connection.close()
    out: dict[str, list[float]] = {"day": [float(day) for day in MARKS]}
    for index, name in enumerate(TARGETS):
        out[name] = [float(row[index] or 0.0) for row in rows]
    return out


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
