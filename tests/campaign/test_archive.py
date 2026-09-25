"""The shared database: every program, its scores, and what failed."""

import json
import time
from pathlib import Path

import pydantic
import pytest

from kaggriculture.campaign import archive
from kaggriculture.campaign.harness import Margin

AGENT = "def agent(o, c=None):\n    return {}\n"


def make(tmp_path: Path) -> archive.Database:
    """A fresh database backed by files under `tmp_path`."""
    return archive.Database(tmp_path / "db.jsonl", tmp_path / "programs")


def program(
    db: archive.Database,
    name: str,
    fitness: float,
    margin: float = 0.0,
    rating: float | None = None,
) -> archive.Program:
    """Store a source and add a program with its scores and one bank margin.

    ``rating`` defaults to ``fitness`` because most tests want the two to
    agree and care about neither; the ones about ranking set it apart.
    """
    p = archive.Program(
        id=name,
        source_path=str(db.store(AGENT, name)),
        started_from="",
        instruction="improve",
        model="gpt-5.6-luna",
        fitness=fitness,
        field=0.5,
        rating=fitness if rating is None else rating,
        rates={"v54": fitness},
        margins={"v54": Margin(mean=margin, worst=margin, best=margin)},
        created=time.time(),
    )
    db.add(p)
    return p


def test_attempts_is_the_whole_campaign_one_line_each(tmp_path: Path) -> None:
    """Every program, what it did, what it scored, and whether it survived.

    A round has been told what its siblings scored and what the promotions
    changed -- a few dozen edits out of hundreds -- and the rest, every change
    measured and refused, was on the record and reachable by nothing. This is
    the file that reaches it: one JSON object per program, greppable, with the
    opponent names left in because `games.days` already carries them by roster
    name and the question worth asking here is "did this help against the
    agent that beats us".
    """
    db = make(tmp_path)
    first = program(db, "p1", 0.4, margin=120.0)
    second = archive.Program(
        id="p2",
        source_path=str(db.store(AGENT, "p2")),
        started_from="p1",
        instruction="improve",
        model="gpt-5.6-luna",
        fitness=0.45,
        rates={"v54": 0.45},
        margins={"v54": Margin(mean=300.0, worst=300.0, best=300.0)},
        changed="every SELL WHEAT tripled",
        created=time.time(),
    )
    db.add(second)

    out = tmp_path / "attempts.jsonl"
    written = db.attempts(out, promoted={"p2"})

    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert written == 2 and len(lines) == 2
    assert lines[0] == {
        "id": "p1",
        "from": "",
        "instruction": "improve",
        "changed": "",
        "wins": 0.4,
        "margin": 120,
        "promoted": False,
        "rates": {"v54": 0.4},
    }
    assert lines[1]["from"] == first.id
    assert lines[1]["changed"] == "every SELL WHEAT tripled"
    assert lines[1]["promoted"] is True
    assert lines[1]["margin"] == 300


def test_a_program_recorded_before_changed_existed_still_loads(
    tmp_path: Path,
) -> None:
    """The field is defaulted, so the archive's four hundred prior rows load.

    `margins` set the precedent: a field added to the record must not make the
    campaign unable to replay its own log.
    """
    db = make(tmp_path)
    old = program(db, "old", 0.3).model_dump()
    del old["changed"]
    (tmp_path / "db.jsonl").write_text(
        json.dumps({"event": "program", "program": old}) + "\n", encoding="utf-8"
    )

    replayed = make(tmp_path)

    assert replayed.get("old").changed == ""


def test_a_program_with_no_model_is_a_bug_not_a_legacy_case() -> None:
    """Every program is written by a call that used a model; nothing defaults it.

    The campaign is starting fresh, and the old-format log this would have
    tolerated is moved aside at cutover rather than replayed, so there is no
    record to stay compatible with.
    """
    with pytest.raises(pydantic.ValidationError, match="model"):
        archive.Program.model_validate(
            {
                "id": "a",
                "source_path": "x",
                "started_from": "",
                "instruction": "improve",
                "fitness": 0.5,
                "created": 0.0,
            }
        )


def test_top_ranks_by_games_won(tmp_path: Path) -> None:
    """`top` is the programs that won the most, which is the objective.

    It decides what a session starts from. It ranked on the Bradley-Terry
    rating until 2026-09-12, when every candidate began playing the whole pool
    on the same seeds in both seats -- which makes the rate comparable directly
    and leaves the rating estimating something already measured.

    The rating also stopped separating the top. Measured that day: the best
    program the campaign had produced tied on rating with one winning 12.5
    points fewer games, so the rank-weighted draw took the worse one half the
    time; the second and third best by games won sat at rating ranks 14 and 10,
    outside the parent pool.
    """
    db = make(tmp_path)
    # Deliberately disagreeing: `c` has the rating, `b` won the games.
    program(db, "a", 0.1, rating=-2.0)
    program(db, "b", 0.7, rating=0.1)
    program(db, "c", 0.4, rating=1.5)

    assert [p.id for p in db.top(2)] == ["b", "c"]


def test_an_unrated_program_is_ranked_on_what_it_won(
    tmp_path: Path,
) -> None:
    """A missing rating is no longer a reason to sort last.

    The seed has no rating -- it is scored before any pool is loaded -- and
    while `top` ranked on the rating that put it below every rated program
    however many games it won. It ranks on games won now, so the seed competes
    on the same terms as everything else and an absent rating decides nothing.
    """
    db = make(tmp_path)
    program(db, "rated_badly", 0.0, rating=-9.0)
    unrated = archive.Program(
        id="seed",
        source_path=str(db.store(AGENT, "seed")),
        started_from="",
        instruction="seed",
        model="",
        fitness=0.9,
        field=0.9,
        created=time.time(),
    )
    db.add(unrated)

    assert [p.id for p in db.top(2)] == ["seed", "rated_badly"]


def test_the_log_survives_a_restart(tmp_path: Path) -> None:
    """Everything replays: the programs and the failures beside them."""
    db = make(tmp_path)
    program(db, "a", 0.5)
    db.record_failure(
        archive.Failure(
            started_from="a",
            instruction="improve",
            reason="syntax: bad",
            created=time.time(),
        )
    )

    again = archive.Database(tmp_path / "db.jsonl", tmp_path / "programs")

    assert [p.id for p in again.programs] == ["a"]
    assert [f.reason for f in again.failures("a")] == ["syntax: bad"]


def test_failures_are_looked_up_by_what_they_started_from(tmp_path: Path) -> None:
    """The prompt shows a lineage only its own failures."""
    db = make(tmp_path)
    for started, reason in (("a", "one"), ("b", "two"), ("a", "three")):
        db.record_failure(
            archive.Failure(
                started_from=started,
                instruction="improve",
                reason=reason,
                created=time.time(),
            )
        )
    assert [f.reason for f in db.failures("a")] == ["one", "three"]


def test_an_event_that_cannot_be_applied_is_never_logged(tmp_path: Path) -> None:
    """A rejected write leaves the log exactly as it was, so a restart still works."""
    db = make(tmp_path)
    program(db, "a", 0.5)
    log = tmp_path / "db.jsonl"
    before = log.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        program(db, "a", 0.9)

    assert log.read_text(encoding="utf-8") == before
    again = archive.Database(log, tmp_path / "programs")
    assert [p.id for p in again.programs] == ["a"]


def test_add_rejects_an_id_the_database_already_holds(tmp_path: Path) -> None:
    """Two programs with one id would silently lose one; that is a bug, not a write."""
    db = make(tmp_path)
    program(db, "a", 0.5)
    with pytest.raises(ValueError, match="duplicate program id: 'a'"):
        program(db, "a", 0.9)
    assert db.get("a").fitness == 0.5


def test_get_raises_for_an_unknown_program(tmp_path: Path) -> None:
    """An id the database does not hold is a programming error, not a None."""
    with pytest.raises(KeyError, match="nope"):
        make(tmp_path).get("nope")


def test_top_breaks_a_tie_on_the_bank_margin(tmp_path: Path) -> None:
    """Which of several programs that won nothing came closest to winning.

    Until some program wins a game they all have the same rates and so the
    same rating, and sorting on rating alone leaves the ties in insertion
    order: the gate would spend the exam block on the first three programs a
    campaign ever wrote and the session draw would pick uniformly from the
    first ten, however they played. The margins are the only signal there is
    before the first win, because a rating is blind to margin as the
    competition is.
    """
    db = make(tmp_path)
    program(db, "first", 0.0, margin=-900.0, rating=-6.0)
    program(db, "closest", 0.0, margin=-5.0, rating=-6.0)
    program(db, "middling", 0.0, margin=-300.0, rating=-6.0)
    assert [p.id for p in db.top(2)] == ["closest", "middling"]


def test_the_margin_never_outranks_the_rating(tmp_path: Path) -> None:
    """A program that rates higher is above one that only lost narrowly."""
    db = make(tmp_path)
    program(db, "narrow", 0.0, margin=-1.0, rating=-3.0)
    program(db, "winner", 0.5, margin=-800.0, rating=-1.0)
    assert [p.id for p in db.top(2)] == ["winner", "narrow"]
