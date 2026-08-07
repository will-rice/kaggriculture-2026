"""Tests for the guard that decides whether to submit without a human.

This is the only code in the project that spends a public, limited resource
unattended, and only the latest two submissions are scored -- so a wrong
decision does not merely waste a slot, it displaces a good agent from the
scored pair. Each refusal is pinned individually, because a guard that refuses
for the wrong reason is one that will eventually accept for the wrong reason.
"""

import json
from pathlib import Path

import pytest

from kaggriculture.scripts import autonomous


def submissions(*dates: str) -> list[dict[str, object]]:
    """Return Kaggle submission rows for the given dates, newest first."""
    return [
        {"date": date, "description": "irrelevant", "score": None} for date in dates
    ]


def ledger(tmp_path: Path, agent: str, date: str) -> Path:
    """Write a ledger recording what we last put on the ladder."""
    path = tmp_path / "submitted.json"
    path.write_text(json.dumps({"agent": agent, "date": date, "reason": "test"}))
    return path


def test_what_we_defend_comes_from_our_own_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kaggle's description is a human label; the ledger is the machine state.

    An earlier version parsed the module out of the description, which picks the
    wrong one when a sentence names two and finds nothing when a human submits
    by hand.
    """
    monkeypatch.setattr(
        autonomous,
        "LEDGER",
        ledger(tmp_path, "kaggriculture.kaito_policy", "2026-08-07 01:19"),
    )

    assert autonomous.last_submitted(submissions("2026-08-07 01:19")) == (
        "kaggriculture.kaito_policy"
    )


def test_a_submission_made_outside_this_system_makes_it_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hand-made submission supersedes our ledger without updating it.

    Kaggle then carries an agent we did not record, so our ledger names
    something no longer on the leading edge. Refusing is the only honest
    answer: gating against an opponent we are not defending says nothing about
    whether to replace what is there.
    """
    monkeypatch.setattr(
        autonomous,
        "LEDGER",
        ledger(tmp_path, "kaggriculture.kaito_policy", "2026-08-07 01:19"),
    )

    assert autonomous.last_submitted(submissions("2026-08-07 09:02")) is None


def test_no_ledger_means_unknown_rather_than_assumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Before this system's first submission, what is defended is genuinely unknown."""
    monkeypatch.setattr(autonomous, "LEDGER", tmp_path / "absent.json")

    assert autonomous.last_submitted(submissions("2026-08-07 01:19")) is None


def test_a_ledger_with_no_kaggle_submissions_is_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ledger describing a submission Kaggle has no record of is not evidence."""
    monkeypatch.setattr(
        autonomous,
        "LEDGER",
        ledger(tmp_path, "kaggriculture.kaito_policy", "2026-08-07 01:19"),
    )

    assert autonomous.last_submitted([]) is None


def test_a_stale_gate_verdict_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gate result is evidence about the revision it measured and nothing else.

    Accepting one from a different commit is what turns a gate into a rubber
    stamp: the code that was measured is not the code that would ship.
    """
    verdict = tmp_path / "gate.json"
    verdict.write_text(json.dumps({"revision": "deadbee", "opponent": "x", "low": 0.9}))
    monkeypatch.setattr(autonomous, "GATE_VERDICT", verdict)

    assert autonomous.gate_verdict("a122855") is None
    assert autonomous.gate_verdict("deadbee") is not None
