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
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

from kaggriculture.campaign import config, rating, tapes

if TYPE_CHECKING:  # `games` imports this module for the measure names.
    from kaggriculture.campaign import games

LOGGER = logging.getLogger(__name__)

# Processes to divide the archives over. The same reasoning as
# the loop is usually running while this is, and the loop is what must not
# slow down.
WORKERS = 8
# The fewest games a team must have played to be rated. A team that played
# twice and won both is not the strongest agent on the ladder, and a rating
# fitted from two games is the prior wearing a number.
LEAST = 40
# The day a team never reached a quadrant at all, standing in for infinity so a
# signature is always a pair of days and sorts.
NEVER = 99
# How much of the opening to read as orders, and how many distinct orders to
# keep from each day. Five days because every settled claim about the opening
# falls inside the first week, and four orders because past that a day's list
# is the long tail of one-offs rather than the shape of the opening.
OPENING_DAYS = 5
OPENING_ORDERS = 4
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

# Built after the load, not before: an index makes every insert cost a tree
# walk, and there are twenty million of them.
# The day row's own columns, in declaration order and without its keys. Read
# off the schema so a column added there is measured, stored and asked about
# with nothing to remember.
# Every quantity `measures` returns, in the order it returns them. Written out
# rather than parsed from a CREATE TABLE: the table this used to be read from
# was SQLite's, and the schema that matters now belongs to `games`, which
# imports this. One list, and `measures` is checked against it.
COLUMNS = (
    "bank",
    "planted",
    "ripe",
    "yield_held",
    "pens",
    "weeds",
    "bare",
    "quadrants",
    "hands",
    "seeds",
    "shed",
    "shops",
    "watered",
    "dry_worst",
    "fertilised",
    "fed",
    "hungry_worst",
    "cared",
    "plant_age",
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

TABLES = ("episodes", "days", "holdings", "orders", "moves", "prices")
# Written by `rate` after the load, from what the load produced.
DERIVED = ("teams",)


def build(
    corpus: list[Path],
    database: str = config.GAMES_DB,
    workers: int = WORKERS,
) -> int:
    """Read every game in ``corpus`` into the one database, replacing the ladder.

    One process per archive, each inserting straight into ClickHouse. It used
    to write a SQLite file each, merge twenty-five of them and load the result
    -- three steps that existed because SQLite takes a single writer and the
    parse is what we were trying to parallelise. The store admits parallel
    writers, so the shards, the merge and the load all go: a worker parses an
    archive and sends it.

    Built into a staging database and swapped in a partition at a time. On
    2026-09-09 a build that deleted first hit one unreadable market order
    fifty-five minutes in and left no corpus at all -- not a stale one, none.
    `REPLACE PARTITION` is atomic per table, so a failed rebuild leaves the
    ladder exactly as it was, and the campaign's own games are in the other
    partition and are never touched either way.

    Args:
        corpus: The archives to read, from `tapes.archives`.
        database: The database to write to.
        workers: Processes to divide the archives over.

    Returns:
        How many episodes were read.
    """
    from kaggriculture.campaign import games

    games.create(database)
    staging = f"{database}_building"
    games.query(f"DROP DATABASE IF EXISTS {staging}")
    games.create(staging)
    try:
        with ProcessPoolExecutor(max_workers=min(workers, len(corpus))) as pool:
            done = pool.map(_ingest, corpus, [staging] * len(corpus))
            counts = list(tqdm(done, total=len(corpus), desc="archives"))
        # Every partitioned table, with no exceptions to remember: a table
        # with no ladder rows swaps an empty partition for an empty one.
        for name, table in games.TABLES.items():
            if table.partition:
                games.query(
                    f"ALTER TABLE {database}.{name} "
                    f"REPLACE PARTITION 'ladder' FROM {staging}.{name}"
                )
    finally:
        games.query(f"DROP DATABASE IF EXISTS {staging}")
    total = sum(counts)
    # The ladder is derived from what was just read, so it is rebuilt with
    # it: everything downstream reads `teams`, and a corpus whose ratings
    # are a day older than its games is the sort of stale nobody notices.
    rate(database)
    LOGGER.info("read %d episodes from %d archives", total, len(corpus))
    return total


def _ingest(archive: Path, database: str) -> int:
    """Parse one archive and send it. Runs in a worker process.

    The rows are held for the whole archive and sent a table at a time, which
    is the batching ClickHouse asks for: `async_insert` exists to coalesce many
    small writes and only adds latency to a writer that already arrives with a
    batch of its own.
    """
    from kaggriculture.campaign import games

    batch = games.Batch("ladder")
    read = 0
    for episode in tapes.qualifying_episodes(archive):
        _rows(batch, episode, archive.stem[-10:])
        read += 1
    batch.send(database)
    return read


def _rows(batch: "games.Batch", episode: tapes.Episode, played: str) -> None:
    """Hold every row one recorded game produces."""
    identity = episode.info
    # `is None` rather than `or`: an EpisodeId of 0 is an episode id, and
    # `or` sends it to the seed instead, where it collides with whichever
    # episode really has that seed. Two games then share a key and the row
    # counts every query makes are quietly short.
    identifier = identity.get("EpisodeId")
    key = str(episode.seed if identifier is None else identifier)
    final = episode.steps[-1][0]["observation"]["farms"]
    banks = [float(farm["money"]) for farm in final]
    teams = list(identity.get("TeamNames") or ["", ""])
    winner = -1 if banks[0] == banks[1] else int(banks[1] > banks[0])
    batch.add(
        "episodes",
        [
            key,
            identity.get("EpisodeId") or 0,
            episode.seed,
            episode.engine_version,
            played,
            teams[0],
            teams[1],
            banks[0],
            banks[1],
            winner,
        ],
    )
    submitted = list(_orders(episode, key))
    for order in submitted:
        batch.add("orders", order)
    running = _running(submitted)
    for move in _moves(episode, key):
        batch.add("moves", move)
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
            batch.add(
                "days",
                [
                    key,
                    seat,
                    day,
                    teams[seat],
                    *_row(observation, seat, day, running[seat][day]),
                ],
            )
            for row in _holdings(observation, seat):
                batch.add("holdings", [key, seat, day, *row])
        market = step[0]["observation"]["market"]
        for item, price in (market.get("prices") or {}).items():
            batch.add(
                "prices",
                [key, day, item, price, (market.get("inventory") or {}).get(item) or 0],
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
                # An empty order is the absence of one, not a broken record: a
                # side may submit several market slots and leave some unused,
                # as in `"market": [["HIRE"], []]`. The engine's own MARKET_OPS
                # calls that NONE. `_moves` drops PASS for the same reason, and
                # counting these would inflate every order tally with actions
                # nobody took.
                #
                # Left unhandled this ended a fifty-five minute extraction on
                # 2026-09-09, on one order in one episode of one archive.
                if not parts:
                    continue
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


def counts(database: str = config.GAMES_DB) -> dict[str, int]:
    """How many rows each table holds, recorded and played together."""
    from kaggriculture.campaign import games

    return {
        table: int(games.query(f"SELECT count() FROM {database}.{table}"))
        for table in games.TABLES
    }


def summarise(database: str = config.GAMES_DB) -> str:
    """One line per table, for a person at a terminal."""
    return "\n".join(
        f"{table:12} {rows:>12,}" for table, rows in counts(database).items()
    )


def rate(
    database: str = config.GAMES_DB, least: int = LEAST, window: int = WINDOW
) -> int:
    """Fit a rating per recorded team over the last ``window`` days of games.

    Bradley-Terry has no notion of time and the ladder moves under everyone, so
    the fit reads a window rather than everything: a byte-identical agent
    scored 2386.8 on 2026-09-03 and 1418.0 five days later, and the two halves
    of the corpus share not one name in their top tens.

    Returns:
        How many teams were rated.
    """
    from kaggriculture.campaign import games

    played = [
        (row.split("\t")[0], row.split("\t")[1], int(row.split("\t")[2]))
        for row in games.query(
            f"SELECT team_0, team_1, winner FROM {database}.episodes "
            f"WHERE source = 'ladder' AND played >= '{recent(database, window)}' "
            "AND winner >= 0 AND team_0 != '' AND team_1 != '' FORMAT TabSeparated"
        ).splitlines()
        if row
    ]
    rated = _rate(played, least)
    games.query(f"TRUNCATE TABLE IF EXISTS {database}.teams")
    if rated:
        games.query(
            f"INSERT INTO {database}.teams (team, games, wins, rating, place) "
            "FORMAT TabSeparated",
            ("\n".join(games._row(row) for row in rated) + "\n").encode(),
        )
    return len(rated)


def recent(database: str = config.GAMES_DB, window: int = WINDOW) -> str:
    """The earliest day the rating window includes, as the corpus dates them."""
    from kaggriculture.campaign import games

    span = games.query(
        f"SELECT min(played), max(played) FROM {database}.episodes "
        "WHERE source = 'ladder' AND played != '' FORMAT TabSeparated"
    )
    if not span or "\t" not in span:
        return ""
    earliest, latest = span.split("\t")
    # Counted back from the newest day present, not from today: the fetch runs
    # behind the ladder, so a window measured from the clock would begin after
    # the last archive ends and select nothing. Clamped to the oldest day for
    # the same reason -- more window than corpus is the whole corpus, not an
    # error.
    start = games.query(f"SELECT toString(toDate('{latest}') - {window - 1})")
    return max(start, earliest)


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


def ladder(database: str = config.GAMES_DB) -> list[tuple]:
    """Every rated team, strongest first: team, games, wins, rating, place."""
    from kaggriculture.campaign import games

    return [
        (team, int(played), int(wins), float(rating), int(place))
        for team, played, wins, rating, place in (
            row.split("	")
            for row in games.query(
                f"SELECT team, games, wins, rating, place FROM {database}.teams "
                "ORDER BY place FORMAT TabSeparated"
            ).splitlines()
            if row
        )
    ]


def leaderboard(database: str = config.GAMES_DB, least: int = 30) -> list[tuple]:
    """Every team's record over the recorded games, most games won first.

    A win rate, not a rating: it says who wins and takes no view on schedule
    strength. `rate` fits the Bradley-Terry model that does, and `ladder`
    reads it. This is read straight off the games so that it means something
    whether or not a fit has been run.

    Args:
        database: The one games database.
        least: Ignore teams with fewer games than this; a team that played
            twice and won both is not the best agent on the ladder.
    """
    from kaggriculture.campaign import games

    return [
        (team, int(played), float(rate_))
        for team, played, rate_ in (
            row.split("	")
            for row in games.query(
                f"""
                WITH seats AS (
                    SELECT team_0 AS team, winner = 0 AS won
                    FROM {database}.episodes
                    WHERE source = 'ladder' AND winner >= 0
                    UNION ALL
                    SELECT team_1 AS team, winner = 1 AS won
                    FROM {database}.episodes
                    WHERE source = 'ladder' AND winner >= 0
                )
                SELECT team, count() AS played, avg(won) AS rate
                FROM seats WHERE team != ''
                GROUP BY team HAVING played >= {least}
                ORDER BY rate DESC, played DESC
                FORMAT TabSeparated
                """
            ).splitlines()
            if row
        )
    ]
