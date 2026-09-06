"""The shared database: every program, its scores, and what failed."""

import time
from pathlib import Path

import pydantic
import pytest

from kaggriculture.campaign import archive
from kaggriculture.campaign.evaluator import DeepResult
from kaggriculture.campaign.harness import Margin

AGENT = "def agent(o, c=None):\n    return {}\n"


def make(tmp_path: Path) -> archive.Database:
    """A fresh database backed by files under `tmp_path`."""
    return archive.Database(tmp_path / "db.jsonl", tmp_path / "programs")


def program(
    db: archive.Database, name: str, fitness: float, margin: float = 0.0
) -> archive.Program:
    """Store a source and add a program with `fitness` and one bank margin."""
    p = archive.Program(
        id=name,
        source_path=str(db.store(AGENT, name)),
        started_from="",
        instruction="improve",
        model="gpt-5.6-luna",
        fitness=fitness,
        field=0.5,
        rates={"v54": fitness},
        margins={"v54": Margin(mean=margin, worst=margin, best=margin)},
        created=time.time(),
    )
    db.add(p)
    return p


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


def test_top_ranks_by_fitness(tmp_path: Path) -> None:
    """`top` is the best programs by fitness, best first."""
    db = make(tmp_path)
    program(db, "a", 0.1)
    program(db, "b", 0.7)
    program(db, "c", 0.4)
    assert [p.id for p in db.top(2)] == ["b", "c"]


def test_a_deep_result_is_stored_on_the_program(tmp_path: Path) -> None:
    """A confirmed program carries its deep result."""
    db = make(tmp_path)
    program(db, "a", 0.5)
    result = DeepResult(
        program_id="a",
        score=0.6,
        low=0.5,
        high=0.7,
        rates={"v54": 0.6},
        intervals={"v54": (0.5, 0.7)},
        field=0.6,
        held_out={},
        games=128,
    )
    db.record_deep("a", result)
    assert db.get("a").deep == result


def test_the_log_survives_a_restart(tmp_path: Path) -> None:
    """Everything replays: programs, deep results and failures."""
    db = make(tmp_path)
    program(db, "a", 0.5)
    db.record_deep(
        "a",
        DeepResult(
            program_id="a",
            score=0.6,
            low=0.5,
            high=0.7,
            rates={},
            intervals={},
            field=0.6,
            held_out={},
            games=128,
        ),
    )
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
    assert again.get("a").deep is not None
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

    with pytest.raises(KeyError, match="ghost"):
        db.record_deep(
            "ghost",
            DeepResult(
                program_id="ghost",
                score=0.6,
                low=0.5,
                high=0.7,
                rates={},
                intervals={},
                field=0.6,
                held_out={},
                games=128,
            ),
        )

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

    Until some program wins a game every fitness is 0.0, and sorting on
    fitness alone leaves the ties in insertion order: the gate would spend
    the exam block on the first three programs a campaign ever wrote and the
    session draw would pick uniformly from the first ten, however they
    played. The margins are the only signal there is before the first win.
    """
    db = make(tmp_path)
    program(db, "first", 0.0, margin=-900.0)
    program(db, "closest", 0.0, margin=-5.0)
    program(db, "middling", 0.0, margin=-300.0)
    assert [p.id for p in db.top(2)] == ["closest", "middling"]


def test_the_margin_never_outranks_the_fitness(tmp_path: Path) -> None:
    """A program that won is above one that only lost narrowly, always."""
    db = make(tmp_path)
    program(db, "narrow", 0.0, margin=-1.0)
    program(db, "winner", 0.5, margin=-800.0)
    assert [p.id for p in db.top(2)] == ["winner", "narrow"]
