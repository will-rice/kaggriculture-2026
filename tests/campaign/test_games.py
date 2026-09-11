"""The one games database: what it stores, and what the rebuild may not take.

These talk to a live ClickHouse and are skipped without one, because what they
check is the storage's behaviour -- that a partition drop removes the ladder
and leaves the campaign, that two writers do not need to take turns, that a
count written as a float lands in an integer column -- and none of that can be
checked against a stand-in for the thing being relied on.
"""

import urllib.error
from concurrent.futures import ThreadPoolExecutor

import pytest

from kaggriculture.campaign import dataset, games

from .test_browse import day, game


def running() -> bool:
    """Whether the server is up, so the suite can say why it skipped."""
    try:
        return games.query("SELECT 1") == "1"
    except (urllib.error.URLError, OSError, RuntimeError):
        return False


live = pytest.mark.skipif(not running(), reason="no ClickHouse on GAMES_URL")


@pytest.fixture
def scratch(request: pytest.FixtureRequest) -> str:
    """A database of this test's own, with the real schema, dropped after.

    Every live test takes this rather than writing into the campaign's own.
    They used not to, and tidied up afterwards instead -- which means tidying
    the tables somebody remembered: a write goes to five of them, the teardown
    cleared one, and ten probe episodes were sitting in the real store before
    anything noticed.
    """
    name = f"test_{abs(hash(request.node.name)):x}"[:40]
    games.query(f"DROP DATABASE IF EXISTS {name}")
    request.addfinalizer(lambda: games.query(f"DROP DATABASE IF EXISTS {name}"))
    games.create(name)
    return name


def test_the_column_types_come_from_what_the_measures_actually_hold() -> None:
    """A count written into an integer column truncates in silence.

    Measured over the 1,057,020 recorded days in the corpus: `plant_age` is the
    only measure that is ever fractional, nothing is ever negative, and
    `sold_units` reaches 15,002,111 where every other count stays inside
    sixteen bits. The types are read off that rather than guessed from the
    names, and the schema and the values written into it are decided by the
    same function so they cannot disagree.
    """
    assert games.column_type("plant_age") == "Float32"
    assert games.column_type("bank") == "Float32"
    # Fifteen million does not fit in sixteen bits.
    assert games.column_type("sold_units") == "UInt32"
    assert games.column_type("planted") == "UInt16"
    # Every measure has a type, and the casts line up with them one for one.
    assert len(games._CAST) == len(dataset.COLUMNS)
    for cast, measure in zip(games._CAST, dataset.COLUMNS, strict=True):
        assert cast is (float if games.column_type(measure) == "Float32" else int)


def test_a_measure_arrives_as_a_float_and_lands_in_an_integer_column() -> None:
    """`dataset.measures` returns every quantity as a float, tile counts too.

    So a row reaches the server as `4.0` for a column declared `UInt16`, which
    TabSeparated refuses to parse -- the whole insert fails, not the field.
    Found by eight sessions writing at once, which is the first time a real
    evaluation's measures went in.
    """
    rows = []
    measures = dict.fromkeys(dataset.COLUMNS, 0.0) | {"planted": 4.0, "bank": 7.5}
    values = [
        cast(measures[column])
        for cast, column in zip(games._CAST, dataset.COLUMNS, strict=True)
    ]

    rows.append(games._row(["e", 0, 0, "t", "campaign", *values]))

    assert "\t4\t" in rows[0], "a tile count reached the wire as a float"
    assert "7.5" in rows[0], "money lost its fraction"


@live
def test_the_rebuild_drops_the_ladder_and_keeps_the_campaign(scratch: str) -> None:
    """One store, two writers, and only one of them may delete.

    The nightly extraction replaces every recorded game. The campaign's own
    games are in the same table, and losing them to a rebuild would be silent
    and daily. Partitioning by source is what makes "the rebuild removes what
    it owns and nothing else" a property of the storage rather than a WHERE
    clause somebody has to keep right.
    """
    games.query(
        f"CREATE TABLE {scratch}.probe (episode String, source LowCardinality(String)) "
        "ENGINE = MergeTree PARTITION BY source ORDER BY (source, episode)"
    )
    games.query(
        f"INSERT INTO {scratch}.probe VALUES ('old', 'ladder'), ('ours', 'campaign')"
    )

    games.query(f"ALTER TABLE {scratch}.probe DROP PARTITION 'ladder'")
    games.query(f"INSERT INTO {scratch}.probe VALUES ('new', 'ladder')")

    kept = games.query(
        f"SELECT episode, source FROM {scratch}.probe ORDER BY episode "
        "FORMAT TabSeparated"
    )
    assert kept.split("\n") == ["new\tladder", "ours\tcampaign"]


@live
def test_eight_sessions_record_at_once_without_taking_turns(scratch: str) -> None:
    """The property the store was chosen for, read off real concurrent writes.

    SQLite admits one writer: eight sessions would queue, and a busy timeout
    only decides how the second one fails. Here they are eight independent
    requests. What this asserts is not the speed but that every row of every
    session lands -- a store that dropped writes under contention would look
    fast and be wrong.
    """
    played = [
        (
            1,
            season,
            "probe",
            game([day(n, 1.0, 2.0) for n in range(4)], seat=season % 2),
        )
        for season in range(1, 5)
    ]

    with ThreadPoolExecutor(max_workers=8) as pool:
        counts = list(
            pool.map(lambda name: games.record(name, played, scratch), "abcdefgh")
        )

    assert counts == [len(played)] * 8
    # Eight programs, four seasons each, four days, both sides of every day.
    landed = games.query(f"SELECT count() FROM {scratch}.days")
    assert int(landed) == 8 * len(played) * 4 * 2


@live
def test_a_program_recorded_twice_is_one_program(scratch: str) -> None:
    """A champion re-scored is the same program, not a second one.

    There is no cheap delete here and there does not need to be: the ordering
    key is a row's identity, so re-recording inserts and the engine collapses
    the older row. `FINAL` asks for that to have happened.
    """
    for bank in (5.0, 9.0):
        games.record("probe", [(1, 1, "probe", game([day(0, bank, 3.0)]))], scratch)

    rows = games.query(
        f"SELECT bank FROM {scratch}.days FINAL WHERE episode = 'probem1s1' "
        "AND seat = 0 FORMAT TabSeparated"
    )

    assert rows == "9", "the re-record did not replace the first"


def test_a_name_carrying_a_tab_cannot_shift_the_columns() -> None:
    """TabSeparated's delimiters are the escape's whole reason.

    A team name with a tab in it would move every column after it one to the
    left, and the row would still parse -- which is the kind of wrong that
    reaches a query rather than a log.
    """
    assert games._field("a\tb") == "a\\tb"
    assert games._field("a\nb") == "a\\nb"
    assert games._field("a\\b") == "a\\\\b"
    assert games._row(["x", 1, 2.5]) == "x\t1\t2.5"


def test_the_schema_names_every_measure_the_corpus_defines() -> None:
    """The table is written from `dataset.COLUMNS`, so it cannot fall behind it."""
    sql = games.schema()

    for measure in dataset.COLUMNS:
        assert f"{measure} {games.column_type(measure)}" in sql
    for table in (*dataset.TABLES, "candidate"):
        assert f"games.{table}" in sql


def test_the_days_table_is_ordered_for_the_questions_asked_of_it() -> None:
    """The ordering key is the primary index, so it decides what is cheap.

    Almost every question filters or groups by day, across one source or both.
    Ordered that way a query for one day reads a granule range; ordered by
    episode first it reads the table.
    """
    sql = games.schema()

    assert "PARTITION BY source" in sql
    assert "ORDER BY (source, day, episode, seat)" in sql
    # And a day's identity is the ordering key, which is what makes a
    # re-recorded program collapse rather than duplicate.
    assert "ReplacingMergeTree(version)" in sql
