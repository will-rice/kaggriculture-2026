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
) -> archive.Program:
    """Store a source and add a program with its score and one bank margin."""
    p = archive.Program(
        id=name,
        source_path=str(db.store(AGENT, name)),
        started_from="",
        instruction="improve",
        model="gpt-5.6-luna",
        fitness=fitness,
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
    db.promoted_as("p2", "champion_1")

    out = tmp_path / "attempts.jsonl"
    written = db.attempts(out)

    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert written == 2 and len(lines) == 2
    assert lines[0] == {
        "id": "p1",
        "from": "",
        "instruction": "improve",
        "changed": "",
        "wins": 0.4,
        "margin": 120,
        "swept_margin": 120,
        "promoted": False,
        "rates": {"v54": 0.4},
    }
    assert lines[1]["from"] == first.id
    assert lines[1]["changed"] == "every SELL WHEAT tripled"
    assert lines[1]["promoted"] is True
    assert lines[1]["margin"] == 300


def test_a_promotion_is_an_event_the_log_replays(tmp_path: Path) -> None:
    """Which attempts survived a gate is on the record, and survives a restart.

    `gate.promote` copies a program to `champion_N.py` and keeps no id, so the
    one fact about an attempt worth more than its score was recoverable only by
    diffing champion files against every stored source. An event, like the
    rest: folded before it is written, replayed on start.
    """
    db = make(tmp_path)
    program(db, "p1", 0.4)
    program(db, "p2", 0.5)
    db.promoted_as("p2", "champion_3")

    assert db.promoted == {"p2": "champion_3"}
    assert make(tmp_path).promoted == {"p2": "champion_3"}
    with pytest.raises(ValueError, match="unknown program"):
        db.promoted_as("nobody", "champion_4")
    # And a refused event never reached the log, so a replay does not trip on it.
    assert make(tmp_path).promoted == {"p2": "champion_3"}


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

    It decides what a session starts from. The margin led this ranking for a
    few hours on 2026-09-25 and it was ranking seed blocks: margins are
    measured on maps redrawn every `SEED_ROTATION` candidates, and across 494
    recorded programs 19,240 coins of spread sat between blocks against 8,253
    within one, while the gap it was resolving at the top was 5,621.
    """
    db = make(tmp_path)
    # Deliberately disagreeing: `b` won the most games, `c` banked the most.
    program(db, "a", 0.1, margin=-500.0)
    program(db, "b", 0.7, margin=800.0)
    program(db, "c", 0.4, margin=2_000.0)

    assert [p.id for p in db.top(2)] == ["b", "c"]


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
    program(db, "first", 0.0, margin=-900.0)
    program(db, "closest", 0.0, margin=-5.0)
    program(db, "middling", 0.0, margin=-300.0)
    assert [p.id for p in db.top(2)] == ["closest", "middling"]


def test_the_tie_is_broken_on_the_swept_opponents_alone(tmp_path: Path) -> None:
    """Where the rate is pinned, the bank there ranks -- and only there.

    Both programs win the same share of games. One banks more against the
    opponent it sweeps, the other against the one it only half beats, and a
    mean over the pool calls the second the better: that is a margin grown by
    giving up games, which is what `swept_margin` exists to refuse.
    """
    db = make(tmp_path)
    for name, swept, contested in (
        ("economy", 4_000.0, 0.0),
        ("trader", 1_000.0, 9_000.0),
    ):
        db.add(
            archive.Program(
                id=name,
                source_path=str(db.store(AGENT, name)),
                started_from="",
                instruction="improve",
                model="m",
                fitness=0.75,
                rates={"swept": 1.0, "contested": 0.5},
                margins={
                    "swept": Margin(mean=swept, worst=swept, best=swept),
                    "contested": Margin(
                        mean=contested, worst=contested, best=contested
                    ),
                },
                created=time.time(),
            )
        )

    assert [p.id for p in db.top(2)] == ["economy", "trader"]
    # The whole-pool mean is what the record shows and it disagrees, which is
    # the disagreement this ranking exists to have.
    assert archive.mean_margin(db.get("trader")) > archive.mean_margin(
        db.get("economy")
    )
    assert archive.swept_margin(db.get("economy")) > archive.swept_margin(
        db.get("trader")
    )


def test_a_stored_program_records_what_its_edit_did(tmp_path: Path) -> None:
    """The record says what changed, not only what it scored.

    Every promotion's change was rendered for the prompt by reading the
    champion files back and diffing them; the four hundred and fifty edits that
    were measured and refused were described nowhere. The moment a program is
    stored is the only one where both plans are already files, so that is where
    the sentence is written.
    """
    from tests.campaign.test_plan import PLAN, packed

    parent = tmp_path / "parent.py"
    parent.write_text(packed(PLAN), encoding="utf-8")
    # Through JSON, which is the trip a plan makes anyway and leaves the
    # literal's mixed value types behind.
    flipped = json.loads(json.dumps(PLAN))
    flipped["settings"]["front_run"] = True
    child = tmp_path / "child.py"
    child.write_text(packed(flipped), encoding="utf-8")

    said = archive.changed(child, parent)

    assert said and said != "the plan is unchanged"
    assert "front_run" in said


def test_a_program_whose_parent_carries_no_plan_records_nothing(
    tmp_path: Path,
) -> None:
    """Empty rather than a crash: a scored program must never be lost to a sentence.

    The seed has no parent, programs before champion_17 carry no packed plan,
    and a round can rewrite the controller into something `split` refuses. None
    of those is a failed evaluation.
    """
    from tests.campaign.test_plan import PLAN, packed

    bare = tmp_path / "bare.py"
    bare.write_text("def agent(o, c=None):\n    return {}\n", encoding="utf-8")
    child = tmp_path / "child.py"
    child.write_text(packed(PLAN), encoding="utf-8")

    assert archive.changed(child, bare) == ""
    assert archive.changed(child, None) == ""
