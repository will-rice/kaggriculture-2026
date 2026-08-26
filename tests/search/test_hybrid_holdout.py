"""Durable single-use holdout orchestration and its Task 8 boundary."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from kaggriculture.search import evolution
from kaggriculture.search.scripts import hybrid_holdout

FAKE_SEEDS = (901, 903)


def _run(
    tmp_path: Path,
    *,
    finalists: tuple[dict[str, object], ...] = ({"index": 0, "genome": [1.0]},),
    deterministic: Callable[[Mapping[str, object]], bool] = lambda row: True,
    evaluate: Callable[[Mapping[str, object]], Mapping[str, object]] = lambda row: {
        **row,
        "passed": True,
    },
) -> dict[str, object]:
    snapshot = tmp_path / "snapshots"
    snapshot.mkdir(exist_ok=True)
    protected = tmp_path / "finalists.json"
    protected.write_text("{}")
    return hybrid_holdout._execute_single_use(
        output=tmp_path / "promotion.json",
        run_identity={"seeds": list(FAKE_SEEDS), "frontier": "economic"},
        finalists=finalists,
        protected_paths=(protected,),
        snapshot_root=snapshot,
        deterministic=deterministic,
        evaluate=evaluate,
    )


def test_holdout_uses_the_authoritative_task8_finalist_loader() -> None:
    """Task 9 does not maintain an independent finalist parser or validator."""
    assert hybrid_holdout.FinalistArtifact is evolution.FinalistArtifact


def test_crash_claim_forbids_replay_and_spends_no_second_holdout(
    tmp_path: Path,
) -> None:
    """A crash after the exclusive claim permanently burns that exact attempt."""
    calls = 0

    def crash(row: Mapping[str, object]) -> Mapping[str, object]:
        nonlocal calls
        calls += 1
        raise RuntimeError("crash after first holdout game")

    with pytest.raises(RuntimeError, match="crash after"):
        _run(tmp_path, evaluate=crash)
    with pytest.raises(RuntimeError, match="replay is forbidden"):
        _run(tmp_path, evaluate=crash)

    assert calls == 1


def test_completed_result_is_reused_idempotently_without_any_games(
    tmp_path: Path,
) -> None:
    """A completed exact result survives reruns without determinism or holdout play."""
    first = _run(tmp_path)

    def forbidden(row: Mapping[str, object]) -> bool:
        raise AssertionError(f"unexpected game for {row}")

    def forbidden_evaluation(row: Mapping[str, object]) -> Mapping[str, object]:
        raise AssertionError(f"unexpected holdout for {row}")

    second = _run(tmp_path, deterministic=forbidden, evaluate=forbidden_evaluation)

    assert second == first
    assert second["status"] == "complete"


def test_existing_or_concurrent_claim_is_exclusive(tmp_path: Path) -> None:
    """The filesystem claim cannot be acquired by a second contender."""
    snapshot = tmp_path / "snapshots"
    snapshot.mkdir()
    claims = tmp_path / ".promotion.json.claims"
    identity = {"run_identity": {"seeds": list(FAKE_SEEDS)}, "finalist": {"index": 0}}
    path = hybrid_holdout._claim_path(claims, {"seeds": list(FAKE_SEEDS)}, {"index": 0})

    assert hybrid_holdout._exclusive_claim(path, identity) is None
    with pytest.raises(RuntimeError, match="replay is forbidden"):
        hybrid_holdout._exclusive_claim(path, identity)


def test_malformed_or_conflicting_output_is_rejected_before_games(
    tmp_path: Path,
) -> None:
    """Existing output cannot be silently replaced or interpreted leniently."""
    output = tmp_path / "promotion.json"
    output.write_text('{"schema_version":2,"schema_version":2}')
    calls = 0

    def counted(row: Mapping[str, object]) -> bool:
        nonlocal calls
        calls += 1
        return True

    with pytest.raises(ValueError, match="duplicate JSON key"):
        _run(tmp_path, deterministic=counted)
    assert calls == 0


def test_each_finalist_verdict_persists_before_a_later_crash(tmp_path: Path) -> None:
    """A second-finalist failure cannot erase the first finalist's verdict."""
    finalists: tuple[dict[str, object], ...] = ({"index": 0}, {"index": 1})

    def evaluate(row: Mapping[str, object]) -> Mapping[str, object]:
        if row["index"] == 1:
            raise RuntimeError("second crashed")
        return {**row, "passed": True}

    with pytest.raises(RuntimeError, match="second crashed"):
        _run(tmp_path, finalists=finalists, evaluate=evaluate)

    output = json.loads((tmp_path / "promotion.json").read_text())
    assert output["status"] == "partial"
    assert output["results"] == [{"index": 0, "passed": True}]
    assert len(list((tmp_path / ".promotion.json.claims").glob("*.json"))) == 2


def test_failed_determinism_claims_and_spends_zero_holdout(tmp_path: Path) -> None:
    """Action-trace replay is a pre-holdout gate for every finalist."""
    evaluated = 0

    def evaluate(row: Mapping[str, object]) -> Mapping[str, object]:
        nonlocal evaluated
        evaluated += 1
        return row

    with pytest.raises(RuntimeError, match="nondeterministic"):
        _run(tmp_path, deterministic=lambda row: False, evaluate=evaluate)

    assert evaluated == 0
    assert not (tmp_path / ".promotion.json.claims").exists()


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_output_aliases_of_every_input_are_rejected_before_writes(
    tmp_path: Path, alias_kind: str
) -> None:
    """Canonical and inode aliases cannot overwrite holdout input evidence."""
    snapshot = tmp_path / "snapshots"
    snapshot.mkdir()
    protected = tmp_path / "frontier.json"
    protected.write_text("evidence")
    output = tmp_path / "promotion.json"
    if alias_kind == "symlink":
        output.symlink_to(protected)
    else:
        os.link(protected, output)

    with pytest.raises(ValueError, match="input artifact|symlink"):
        hybrid_holdout._execute_single_use(
            output=output,
            run_identity={"seeds": list(FAKE_SEEDS)},
            finalists=({"index": 0},),
            protected_paths=(protected,),
            snapshot_root=snapshot,
            deterministic=lambda row: True,
            evaluate=lambda row: row,
        )
    assert protected.read_text() == "evidence"


def test_holdout_cli_has_no_seed_override_and_requires_output() -> None:
    """The production command cannot substitute inspected seeds or an implicit path."""
    options = {action.dest for action in hybrid_holdout.parser()._actions}
    assert "seeds" not in options
    with pytest.raises(SystemExit):
        hybrid_holdout.parser().parse_args(
            ["--finalists", "finalists.json", "--frontier-report", "frontier.json"]
        )
