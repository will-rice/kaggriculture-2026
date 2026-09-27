"""The one games database: what it stores, and what the rebuild may not take.

These talk to a live ClickHouse and are skipped without one, because what they
check is the storage's behaviour -- that a partition drop removes the ladder
and leaves the campaign, that two writers do not need to take turns, that a
count written as a float lands in an integer column -- and none of that can be
checked against a stand-in for the thing being relied on.
"""

import logging
from concurrent.futures import ThreadPoolExecutor

import pytest

from kaggriculture.campaign import dataset, games

from .fixtures import day, game, live


def test_the_column_types_come_from_what_the_measures_actually_hold() -> None:
    """A count written into an integer column truncates in silence.

    Measured over the 1,057,020 recorded days in the corpus: `plant_age` is the
    only measure that is ever fractional, nothing is ever negative, and
    `sold_units` reaches 15,002,111 where every other count stays inside
    sixteen bits. The types are read off that rather than guessed from the
    names.
    """
    assert games.measure_column("plant_age") == games.Column("Float32", games.real)
    assert games.measure_column("bank") == games.Column("Float32", games.real)
    # Fifteen million does not fit in sixteen bits.
    assert games.measure_column("sold_units").kind == "UInt32"
    assert games.measure_column("planted").kind == "UInt16"


def test_how_a_column_is_stored_and_how_it_is_written_are_one_declaration() -> None:
    """They were two lists, and they disagreed the first time one changed.

    A `CREATE TABLE` in one place and a tuple of casts in another, kept in step
    by whoever remembered. `Column` carries both, so a column that cannot be
    declared without saying how it is written cannot drift from how it is
    written -- and the `TABLES` entry is the only place either is stated.
    """
    days = games.TABLES["days"].columns

    # Every measure the campaign defines is a column of `days`, typed and cast
    # by the same declaration.
    for measure in dataset.COLUMNS:
        assert days[measure] == games.measure_column(measure)
    # An integer column casts through `number`, which is what stops a tile
    # count arriving as `4.0` and failing the insert.
    assert days["planted"].cast is games.number
    assert days["bank"].cast is games.real
    # `version` is declared and never written: the server fills it.
    assert not days["version"].written
    assert "version" not in games.written("days")


def test_a_measure_arrives_as_a_float_and_lands_in_an_integer_column() -> None:
    """`dataset.measures` returns every quantity as a float, tile counts too.

    So a row would reach the server as `4.0` for a column declared `UInt16`,
    which TabSeparated refuses to parse -- the whole insert fails, not the
    field. The batch casts each value through its own column's declaration, so
    neither writer can forget and neither can disagree with the schema.
    """
    measures = dict.fromkeys(dataset.COLUMNS, 0.0) | {"planted": 4.0, "bank": 7.5}
    batch = games.Batch("campaign")

    batch.add("days", ["e", 0, 0, "t", *(measures[c] for c in dataset.COLUMNS)])

    row = batch.rows["days"][0]
    assert "\t4\t" in row, "a tile count reached the wire as a float"
    assert "7.5" in row, "money lost its fraction"
    assert row.endswith("\tcampaign"), "the source is not the last column"


def test_a_row_with_the_wrong_number_of_values_is_refused() -> None:
    """A short row lands every column after the gap one to the left.

    Which parses, and is wrong, and is the failure a positional insert exists
    to produce. `strict` turns it into a raise at the point of the mistake.
    """
    batch = games.Batch("campaign")

    with pytest.raises(ValueError, match="argument"):
        batch.add("holdings", ["e", 0, 0, "plants"])


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


@live
def test_the_opponent_is_recorded_by_roster_name(scratch: str) -> None:
    """A matchup index is not an opponent, so the name goes in beside it.

    Both sides were written as the literal "opponent" until 2026-09-13, which
    cost the whole audit trail: the games were all there and correctly grouped,
    and nothing said which agent a group was. `candidate` holds the index and no
    name; the index is not the order the rates are recorded in; and `Pool.names`
    is insertion order with the scored program removed, so the champion sits
    mid-pool and every opponent after it shifts by one depending on who is being
    measured. Two evaluations do not agree on what matchup 7 means.

    Seat 1 here, so the assertions also pin which column is whose: getting that
    backwards would write our own banks under the opponent's name and read as a
    program that loses to itself.
    """
    games.record("probe", [(7, 1, "probe", game([day(0, 5.0, 3.0)], seat=1))], scratch)

    sides = games.query(
        f"SELECT team_0, team_1 FROM {scratch}.episodes "
        "WHERE episode = 'probem7s1' FORMAT TabSeparated"
    )
    teams = games.query(
        f"SELECT seat, team FROM {scratch}.days WHERE episode = 'probem7s1' "
        "ORDER BY seat FORMAT TabSeparated"
    )

    assert sides == "v54\tprobe", "seat 1 is ours, so team_0 is the opponent"
    assert teams == "0\tv54\n1\tprobe"


@live
def test_a_programs_games_come_back_off_the_record_as_they_were_written(
    scratch: str,
) -> None:
    """`recorded` is the inverse of `record`, seat and episode included.

    A result serialises without its games, so a champion loaded on restart
    has none and the game a round is shown has to come from here. Seat 1 and
    two matchups, so the assertions pin which bank is whose and that the
    episode key `record` wrote is the one that comes back -- the message
    hands that key to the round, which reads the day table by it.
    """
    games.record(
        "probe",
        [
            (7, 1, "probe", game([day(0, 5.0, 3.0)], seat=1)),
            (7, 2, "probe", game([day(0, 2.0, 9.0)], seat=0)),
            (3, 1, "probe", game([day(0, 4.0, 4.0)], seat=1)),
        ],
        scratch,
    )

    back = games.recorded("probe", scratch)

    assert [episode for episode, _ in back] == [
        "probem3s1",
        "probem7s1",
        "probem7s2",
    ]
    first = back[1][1]
    assert first.seat == 1 and first.opponent == "v54" and first.seed == 101
    assert (first.ours, first.theirs) == (5.0, 3.0), "seat 1's bank is ours"
    second = back[2][1]
    assert second.seat == 0 and (second.ours, second.theirs) == (2.0, 9.0)
    assert all(one.days == [] for _, one in back), "days are rows, not cargo"
    assert games.recorded("nobody", scratch) == []


def test_two_measurements_of_one_program_do_not_invent_a_game(
    scratch: str,
) -> None:
    """An episode key does not identify a game, and it used to be joined as if it did.

    A program is measured more than once -- at the start of every session that
    begins from it, and again as a candidate -- and each measurement numbers
    the matchups afresh, so `probem1s1` is a different opponent on a different
    seed each time. `recorded` joined `candidate` to `episodes` on that key and
    cross-multiplied them: measured on the live database 2026-09-27, one key
    carried three opponents, three seeds and three pairs of banks, so a round
    could be told one opponent's name beside another's result. Where two rows'
    seats disagreed the program came back as its own opponent, which is the
    `KeyError` that stopped the campaign for eight hours.

    Read from `episodes` alone, each row is one game and says who played it.
    """
    games.record("probe", [(1, 1, "probe", game([day(0, 5.0, 3.0)], seat=0))], scratch)
    # The same program measured again: the same key, another opponent, another
    # seed, another pair of banks.
    games.record(
        "probe",
        [(1, 1, "probe", game([day(0, 1.0, 8.0)], seat=1, opponent="shopforge"))],
        scratch,
    )

    back = games.recorded("probe", scratch)

    assert len(back) == 2, "one game per episode row, never their product"
    assert {one.opponent for _, one in back} == {"v54", "shopforge"}
    # Each game's banks belong to the opponent it was played against.
    against = {one.opponent: (one.ours, one.theirs) for _, one in back}
    assert against["v54"] == (5.0, 3.0)
    assert against["shopforge"] == (1.0, 8.0), "the seat-1 game comes back whole"
    assert all(one.opponent != "probe" for _, one in back)


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
    """The tables are derived from `TABLES`, which is derived from the measures."""
    sql = "\n".join(games.schema())

    for measure in dataset.COLUMNS:
        assert f"{measure} {games.measure_column(measure).kind}" in sql
    for table in games.TABLES:
        assert f"games.{table} (" in sql


def test_the_days_table_is_ordered_for_the_questions_asked_of_it() -> None:
    """The ordering key is the primary index, so it decides what is cheap.

    Almost every question filters or groups by day, across one source or both.
    Ordered that way a query for one day reads a granule range; ordered by
    episode first it reads the table.
    """
    days = games.TABLES["days"]

    assert days.partition == "source"
    assert days.order == "(source, day, episode, seat)"
    # And a day's identity is that key, which is what makes a re-recorded
    # program collapse rather than duplicate.
    assert days.replacing
    assert "ReplacingMergeTree(version)" in "\n".join(games.schema())


def test_every_table_is_partitioned_on_the_column_the_rebuild_swaps() -> None:
    """`REPLACE PARTITION` is how a rebuild replaces only what it owns.

    A table holding both sources and partitioned on something else cannot be
    swapped a source at a time, so the rebuild would have to delete by
    predicate -- which is the convention this partitioning exists to replace.
    """
    for name, table in games.TABLES.items():
        both = "source" in table.columns
        assert both == bool(table.partition), (
            f"{name} carries a source and is not partitioned on it, or the reverse"
        )


@live
def test_a_database_built_before_a_column_existed_gains_it(scratch: str) -> None:
    """`CREATE TABLE IF NOT EXISTS` is silent about a table already there.

    So a column added to the schema never reaches a database built before it,
    and the first sign is an insert failing -- in the loop, on a real
    evaluation, which is exactly how it was found: the run died on `No such
    column source in table games.candidate` having passed every test, because
    a test builds a fresh database and a fresh database matches by
    construction. Only the long-lived one was behind.
    """
    # A database as it was before `source` was added to `candidate`.
    games.query(f"DROP TABLE {scratch}.candidate")
    games.query(
        f"CREATE TABLE {scratch}.candidate (episode String, seat UInt8, "
        "team LowCardinality(String), matchup UInt16, season UInt16) "
        "ENGINE = MergeTree ORDER BY episode"
    )
    before = games.query(
        f"SELECT count() FROM system.columns WHERE database = '{scratch}' "
        "AND table = 'candidate' AND name = 'source'"
    )
    assert before == "0", "the fixture did not reproduce the old shape"

    games.create(scratch)

    # And the write that used to fail now lands.
    assert (
        games.record("probe", [(1, 1, "probe", game([day(0, 5.0, 3.0)]))], scratch) == 1
    )
    assert (
        games.query(f"SELECT source FROM {scratch}.candidate FORMAT TabSeparated")
        == "campaign"
    )


@live
def test_a_live_table_that_lost_its_partition_key_says_so(
    scratch: str, caplog: pytest.LogCaptureFixture
) -> None:
    """A partition key cannot be altered, so drift has to be reported.

    `episodes` and `candidate` were both unpartitioned in the long-lived
    database on 2026-09-13, having been built before `TABLES` gave them a key.
    `CREATE TABLE IF NOT EXISTS` is silent about them and the columns
    reconciliation cannot help, because ClickHouse has no `ALTER` for a
    partition key. So every corpus rebuild from 2026-09-09 on built its staging
    database, failed `REPLACE PARTITION` with "Wrong number of fields in the
    partition expression", tore the staging down and left the old rows -- and
    the corpus sat four days stale with nothing else to show for it.

    `test_every_table_is_partitioned_on_the_column_the_rebuild_swaps` checks the
    declaration against itself and passed throughout. Only the live table
    disagreed.
    """
    games.query(f"DROP TABLE {scratch}.episodes")
    games.query(
        f"CREATE TABLE {scratch}.episodes (episode String, source "
        "LowCardinality(String), version UInt64) ENGINE = ReplacingMergeTree(version) "
        "ORDER BY (source, episode)"
    )
    assert (
        games.query(
            f"SELECT partition_key FROM system.tables WHERE database = '{scratch}' "
            "AND name = 'episodes' FORMAT TabSeparated"
        )
        == ""
    ), "the fixture did not reproduce an unpartitioned table"

    with caplog.at_level(logging.WARNING):
        games.create(scratch)

    said = [record.getMessage() for record in caplog.records]
    assert any("episodes is partitioned by nothing" in line for line in said), (
        f"no warning named the drift: {said}"
    )


@live
def test_every_table_the_writers_use_matches_the_schema(scratch: str) -> None:
    """The reconciliation is only worth having if it covers every table.

    A column added to `days` and not reaching it is the same failure as the
    one that killed the run, with a hundred times the rows behind it.
    """
    for name, table in games.TABLES.items():
        live = set(
            games.query(
                f"SELECT name FROM system.columns WHERE database = '{scratch}' "
                f"AND table = '{name}' FORMAT TabSeparated"
            ).split()
        )
        declared = set(table.columns)
        assert declared <= live, f"{name} is missing {declared - live}"
        # And every column a writer names is one the table has.
        for column in games.written(name):
            assert column in live, f"{name} has no {column}"
