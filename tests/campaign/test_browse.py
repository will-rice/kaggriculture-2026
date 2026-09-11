"""The games a round is handed, and whether it can ask them anything."""

import sqlite3
from pathlib import Path

from kaggriculture.campaign import browse, dataset, harness


def measured(**held: float) -> dict[str, float]:
    """A full measures dict: what is named, and zero for everything else."""
    return dict.fromkeys(dataset.COLUMNS, 0.0) | held


def day(number: int, ours: float, theirs: float, **held: float) -> harness.Day:
    """One day, with something in every field and the banks where asked."""
    return harness.Day(
        day=number,
        ours_bank=ours,
        theirs_bank=theirs,
        ours=measured(bank=ours, **held),
        theirs=measured(bank=theirs),
        ours_plants={"WHEAT": 4},
        theirs_plants={"MELON": 2},
        ours_animals={},
        theirs_animals={"COW": 1},
        ours_weeds=0,
        theirs_weeds=3,
        ours_seeds={"WHEAT": 5},
        ours_shed={"WHEAT": 12},
        theirs_shed={"EGG": 3},
        ours_hands=2,
        theirs_hands=1,
        prices={"WHEAT": 25},
    )


def game(days: list[harness.Day], seat: int = 0) -> harness.Game:
    """One played game carrying those days."""
    final = days[-1]
    return harness.Game(
        opponent="v54",
        seed=101,
        seat=seat,
        ours=final.ours_bank,
        theirs=final.theirs_bank,
        worst_step_seconds=0.0,
        days=days,
    )


def test_our_games_and_the_public_corpus_are_the_same_kind_of_row(
    tmp_path: Path,
) -> None:
    """One schema, so the two attach and union without translating either.

    This is the whole reason the module exists rather than a CSV writer. The
    corpus is 17,617 episodes and a million day rows of what the field does;
    our games are what this lineage does. A question worth asking spans both --
    is the champion converging on what winners do, or somewhere else -- and it
    can only be asked while a row on one side means exactly what it means on
    the other.

    It holds because both are written by `dataset.SCHEMA` itself rather than by
    two definitions kept in step by hand, and because `dataset.measures` is the
    single function behind every measure on either side.
    """
    ours = browse.write(
        tmp_path / "seasons.db", [(1, 1, "champion_1", game([day(0, 5.0, 3.0)]))]
    )
    theirs = tmp_path / "corpus.db"
    sqlite3.connect(theirs).executescript(dataset.SCHEMA)

    db = sqlite3.connect(ours)
    db.execute(f"ATTACH '{theirs}' AS public")
    for table in ("episodes", "days", "holdings", "prices"):
        mine = [(r[1], r[2]) for r in db.execute(f"PRAGMA main.table_info({table})")]
        public = [
            (r[1], r[2]) for r in db.execute(f"PRAGMA public.table_info({table})")
        ]
        assert mine == public, f"{table} differs between ours and the corpus"
    # And the union needs no column list, which is the operational form of it.
    db.execute("SELECT * FROM days UNION ALL SELECT * FROM public.days").fetchall()


def test_the_gap_is_ours_minus_theirs_whichever_seat_we_held(tmp_path: Path) -> None:
    """A `Day` says "ours"; the corpus has no such thing, so the seats resolve.

    `Day` records both sides relative to whoever was being measured and
    `Game.seat` says which engine seat that was. Storing the relative label
    would make our rows a different kind of row from a corpus row, so it is
    resolved to an absolute seat on the way in -- and `candidate` remembers
    which one was ours, because that is the one thing the corpus cannot say.

    Seat 1 is the case that catches a resolution written the easy way round.
    """
    days = [day(0, 10.0, 4.0), day(1, 12.0, 30.0)]
    path = browse.write(tmp_path / "s.db", [(1, 1, "champion_1", game(days, seat=1))])
    db = sqlite3.connect(path)

    assert db.execute("SELECT seat FROM candidate").fetchone() == (1,)
    # Our own row is stored under seat 1, the opponent's under seat 0.
    ours = db.execute("SELECT bank FROM days WHERE seat = 1 ORDER BY day").fetchall()
    assert ours == [(10.0,), (12.0,)]
    # The view reads ours minus theirs, not seat 0 minus seat 1.
    assert db.execute("SELECT day, bank FROM gaps ORDER BY day").fetchall() == [
        (0, 6.0),
        (1, -18.0),
    ]


def test_the_worst_day_query_never_answers_with_a_first_day(tmp_path: Path) -> None:
    """`swings` is a query to get right once rather than in every round.

    `LAG` has nothing to subtract on a game's first day, and a NULL sorts
    before every number in SQLite -- so the obvious `order by moved limit 1`
    answers "day 0" for every game, which looks like an answer and is not one.
    Rounds would have hit that separately and some of them would not have
    noticed, which is exactly the work a view is for.
    """
    days = [day(0, 100.0, 100.0), day(1, 90.0, 100.0), day(2, 40.0, 100.0)]
    path = browse.write(tmp_path / "s.db", [(1, 1, "champion_1", game(days))])
    db = sqlite3.connect(path)

    assert db.execute("SELECT COUNT(*) FROM swings WHERE moved IS NULL").fetchone() == (
        0,
    )
    # Day two lost fifty against day one's ten, and it is what the query names.
    worst = db.execute(
        "SELECT day, moved FROM swings ORDER BY moved LIMIT 1"
    ).fetchone()
    assert worst == (2, -50.0)


def test_no_opponent_is_named_in_a_file_the_round_can_read(tmp_path: Path) -> None:
    """The doctrine reaches every artifact, not only the composed message.

    A round writing SQL against this database can read every string in it. The
    opponent's roster name travels no further than the campaign's own memory:
    the database calls it "opponent", the same way the message calls it nothing
    at all, because a name is the handle that makes fitting to one possible.
    """
    played = game([day(0, 5.0, 3.0)])
    assert played.opponent == "v54", "the game itself still knows"

    path = browse.write(tmp_path / "s.db", [(1, 1, "champion_1", played)])
    text = path.read_bytes().decode("utf-8", "ignore")

    assert "v54" not in text
    db = sqlite3.connect(path)
    teams = db.execute("SELECT DISTINCT team FROM days").fetchall()
    assert sorted(teams) == [("champion_1",), ("opponent",)]
