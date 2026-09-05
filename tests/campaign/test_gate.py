"""The promotion rule on hand-built numbers, and the floor write."""

import logging
import subprocess
from pathlib import Path

import pytest

from kaggriculture.campaign import archive, config, evaluator, gate, pool


def result(
    pid: str, score: float, low: float, rates: dict[str, float], width: float = 0.05
) -> evaluator.DeepResult:
    """A hand-built DeepResult with symmetric per-opponent intervals."""
    return evaluator.DeepResult(
        program_id=pid,
        score=score,
        low=low,
        high=min(1.0, score + (score - low)),
        rates=rates,
        intervals={
            n: (max(0.0, r - width), min(1.0, r + width)) for n, r in rates.items()
        },
        field=score,
        held_out={},
        games=128,
    )


def test_no_champion_promotes_anything_with_a_score() -> None:
    """With no champion yet, any candidate is promoted."""
    ok, why = gate.promotion(result("c", 0.3, 0.25, {"a": 0.3}), None)
    assert ok


def test_lower_bound_must_beat_the_champion_point_estimate() -> None:
    """The candidate's lower Wilson bound must exceed the champion's score."""
    champ = result("k", 0.60, 0.55, {"a": 0.6})
    assert not gate.promotion(result("c", 0.62, 0.58, {"a": 0.62}), champ)[0]
    assert gate.promotion(result("c", 0.70, 0.65, {"a": 0.70}), champ)[0]


def test_field_may_not_drop_more_than_two_points() -> None:
    """A candidate that otherwise beats the bound still fails on field drop."""
    champ = result("k", 0.60, 0.55, {"a": 0.6})
    candidate = result("c", 0.70, 0.65, {"a": 0.70})
    candidate = candidate.model_copy(update={"field": 0.57})
    assert not gate.promotion(candidate, champ)[0]


def test_no_opponent_may_regress_beyond_noise() -> None:
    """A single opponent regressing past the wider interval blocks promotion."""
    champ = result("k", 0.60, 0.55, {"a": 0.9, "b": 0.3})
    candidate = result("c", 0.70, 0.65, {"a": 0.6, "b": 0.8})
    ok, why = gate.promotion(candidate, champ)
    assert not ok and "a" in why


def test_field_drop_of_exactly_the_tolerance_still_passes() -> None:
    """A field drop of exactly FIELD_TOLERANCE is not a drop beyond it."""
    champ = result("k", 0.60, 0.55, {"a": 0.6})
    candidate = result("c", 0.70, 0.65, {"a": 0.70})
    candidate = candidate.model_copy(
        update={"field": champ.field - gate.FIELD_TOLERANCE}
    )
    assert gate.promotion(candidate, champ)[0]


def test_opponent_regression_of_exactly_the_width_still_passes() -> None:
    """A rate exactly champion - width is not a regression beyond it."""
    champ = result("k", 0.60, 0.55, {"a": 0.9})
    width = champ.intervals["a"][1] - champ.intervals["a"][0]
    candidate = result("c", 0.70, 0.65, {"a": champ.rates["a"] - width})
    assert gate.promotion(candidate, champ)[0]


def _program_and_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> archive.Program:
    """Patch config/gate paths into tmp_path and build a program to promote."""
    monkeypatch.setattr(config, "FLOOR", tmp_path / "floor" / "agent")
    monkeypatch.setattr(config, "POOL", tmp_path / "pool.json")
    monkeypatch.setattr(config, "EPOCHS", tmp_path / "epochs.jsonl")
    monkeypatch.setattr(gate, "SERVED", tmp_path / "served" / "main.py")
    source = tmp_path / "prog.py"
    source.write_text(
        "def agent(o, c=None):\n"
        "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
    )
    return archive.Program(
        id="p9",
        island=0,
        source_path=str(source),
        parents=[],
        kind="full",
        fitness_sum=0.5,
        n_evals=1,
        status="ok",
        reason="",
        created=0.0,
    )


def test_promote_writes_a_read_only_floor_and_updates_the_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """promote() writes the floor read-only, mirrors served/, and saves the pool."""
    program = _program_and_pool(tmp_path, monkeypatch)
    source = Path(program.source_path)
    p = pool.Pool(
        opponents={"a": "/x/a.py", "b": "/x/b.py"}, weights={"a": 0.5, "b": 0.5}
    )
    name = gate.promote(
        program, result("p9", 0.7, 0.65, {"a": 0.9, "b": 0.5}), p, commit=False
    )
    assert name == "champion_1"
    floor = config.FLOOR / "main.py"
    assert (
        floor.read_text() == source.read_text()
        and (floor.stat().st_mode & 0o777) == 0o444
    )
    assert gate.SERVED.read_text() == source.read_text()
    saved = pool.Pool.load(config.POOL)
    assert "champion_1" in saved.names() and saved.weights["b"] > saved.weights["a"]
    assert "champion_1" in (tmp_path / "epochs.jsonl").read_text()


def test_a_second_promotion_on_the_saved_pool_yields_champion_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The champion counter reads the pool that was just saved, not a stale one."""
    program = _program_and_pool(tmp_path, monkeypatch)
    p = pool.Pool(
        opponents={"a": "/x/a.py", "b": "/x/b.py"}, weights={"a": 0.5, "b": 0.5}
    )
    gate.promote(
        program, result("p9", 0.7, 0.65, {"a": 0.9, "b": 0.5}), p, commit=False
    )
    reloaded = pool.Pool.load(config.POOL)
    name = gate.promote(
        program,
        result("p9", 0.8, 0.75, {"a": 0.9, "b": 0.6}),
        reloaded,
        commit=False,
    )
    assert name == "champion_2"


def test_a_failed_commit_is_logged_and_never_undoes_the_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A CalledProcessError from git is swallowed; the on-disk promotion stands."""
    program = _program_and_pool(tmp_path, monkeypatch)
    p = pool.Pool(
        opponents={"a": "/x/a.py", "b": "/x/b.py"}, weights={"a": 0.5, "b": 0.5}
    )

    def failing_run(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, ["git"], stderr="nothing to commit")

    monkeypatch.setattr(gate.subprocess, "run", failing_run)
    with caplog.at_level(logging.WARNING):
        name = gate.promote(
            program, result("p9", 0.7, 0.65, {"a": 0.9, "b": 0.5}), p, commit=True
        )
    assert name == "champion_1"
    floor = config.FLOOR / "main.py"
    assert (floor.stat().st_mode & 0o777) == 0o444
    saved = pool.Pool.load(config.POOL)
    assert "champion_1" in saved.names()
    assert any(
        "champion_1" in record.message and "nothing to commit" in record.message
        for record in caplog.records
    )
