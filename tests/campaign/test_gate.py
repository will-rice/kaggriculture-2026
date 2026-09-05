"""The promotion rule on hand-built numbers, and the floor write."""

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


def test_promote_writes_a_read_only_floor_and_updates_the_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """promote() writes the floor read-only, mirrors served/, and saves the pool."""
    monkeypatch.setattr(config, "FLOOR", tmp_path / "floor" / "agent")
    monkeypatch.setattr(config, "POOL", tmp_path / "pool.json")
    monkeypatch.setattr(config, "EPOCHS", tmp_path / "epochs.jsonl")
    monkeypatch.setattr(gate, "SERVED", tmp_path / "served" / "main.py")
    source = tmp_path / "prog.py"
    source.write_text(
        "def agent(o, c=None):\n"
        "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
    )
    program = archive.Program(
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
