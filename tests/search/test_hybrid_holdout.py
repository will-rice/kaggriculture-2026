"""Durable single-use holdout orchestration and its Task 8 boundary."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import pytest

from kaggriculture.hybrid.config import HybridConfig
from kaggriculture.search import evolution
from kaggriculture.search.fitness import strength_weights
from kaggriculture.search.frontier import (
    FrontierArtifact,
    FrontierMatchup,
    FrontierReport,
    FrontierRow,
    VerifiedFrontier,
)
from kaggriculture.search.scripts import hybrid_holdout
from kaggriculture.search.scripts.frontier_round_robin import FRONTIER_SEEDS
from tests.search.test_evolution import _completed_verified_search

FAKE_SEEDS = (901, 903)


def _identity(
    terminal: str = "a", determinism: tuple[int, ...] = (801, 803)
) -> dict[str, object]:
    return {
        "promotion_protocol_version": hybrid_holdout._PROMOTION_PROTOCOL_VERSION,
        "terminal_state_integrity_digest": terminal * 64,
        "determinism_seeds": list(determinism),
        "promotion_seeds": list(FAKE_SEEDS),
        "frontier": "economic",
    }


def _run(
    tmp_path: Path,
    *,
    output_name: str = "promotion.json",
    protected_name: str = "finalists.json",
    run_identity: Mapping[str, object] | None = None,
    finalists: tuple[dict[str, object], ...] = ({"index": 0, "genome": [1.0]},),
    deterministic: Callable[[Mapping[str, object]], bool] = lambda row: True,
    evaluate: Callable[[Mapping[str, object]], Mapping[str, object]] = lambda row: {
        **row,
        "passed": True,
    },
) -> dict[str, object]:
    snapshot = tmp_path / "snapshots"
    snapshot.mkdir(exist_ok=True)
    protected = tmp_path / protected_name
    protected.write_text("{}")
    return hybrid_holdout._execute_single_use(
        output=tmp_path / output_name,
        run_identity=run_identity or _identity(),
        finalists=finalists,
        protected_paths=(protected,),
        snapshot_root=snapshot,
        deterministic=deterministic,
        evaluate=evaluate,
    )


def test_same_certified_finalist_two_output_paths_reuses_one_claim(
    tmp_path: Path,
) -> None:
    """Changing only aggregate output location cannot spend another holdout."""
    evaluations = 0
    determinism_games = 0

    def deterministic(row: Mapping[str, object]) -> bool:
        nonlocal determinism_games
        determinism_games += 1
        return True

    def evaluated(row: Mapping[str, object]) -> Mapping[str, object]:
        nonlocal evaluations
        evaluations += 1
        return {**row, "passed": True}

    _run(
        tmp_path,
        output_name="promotion-a.json",
        deterministic=deterministic,
        evaluate=evaluated,
    )
    _run(
        tmp_path,
        output_name="promotion-b.json",
        deterministic=deterministic,
        evaluate=evaluated,
    )

    assert evaluations == 1
    assert determinism_games == 1


def test_copied_finalist_path_and_new_output_reuse_one_claim(tmp_path: Path) -> None:
    """The finalist filename is not part of certified promotion identity."""
    evaluations = 0

    def evaluated(row: Mapping[str, object]) -> Mapping[str, object]:
        nonlocal evaluations
        evaluations += 1
        return {**row, "passed": True}

    _run(
        tmp_path,
        output_name="promotion-a.json",
        protected_name="original-finalists.json",
        evaluate=evaluated,
    )
    _run(
        tmp_path,
        output_name="promotion-b.json",
        protected_name="copied-finalists.json",
        evaluate=evaluated,
    )

    assert evaluations == 1


def test_distinct_certified_terminal_states_never_share_claims(tmp_path: Path) -> None:
    """The terminal integrity digest separates independent certified searches."""
    evaluations = 0

    def evaluated(row: Mapping[str, object]) -> Mapping[str, object]:
        nonlocal evaluations
        evaluations += 1
        return row

    _run(
        tmp_path,
        output_name="promotion-a.json",
        run_identity=_identity("a"),
        evaluate=evaluated,
    )
    _run(
        tmp_path,
        output_name="promotion-b.json",
        run_identity=_identity("b"),
        evaluate=evaluated,
    )

    claim_dirs = tuple(
        path
        for path in (tmp_path / ".hybrid-promotion-claims").iterdir()
        if path.is_dir()
    )
    assert evaluations == 2
    assert len(claim_dirs) == 2


def test_determinism_seed_drift_conflicts_with_existing_terminal_claim(
    tmp_path: Path,
) -> None:
    """Prerequisite seed drift cannot create fresh evidence for one terminal state."""
    _run(
        tmp_path,
        output_name="promotion-a.json",
        run_identity=_identity(determinism=(801, 803)),
    )

    with pytest.raises(ValueError, match="terminal promotion identity conflicts"):
        _run(
            tmp_path,
            output_name="promotion-b.json",
            run_identity=_identity(determinism=(805, 807)),
        )


def test_concurrent_prerequisite_identities_serialize_before_any_games(
    tmp_path: Path,
) -> None:
    """One terminal sentinel wins; a different identity conflicts pre-game."""
    snapshot = tmp_path / "snapshots"
    snapshot.mkdir()
    protected = tmp_path / "finalists.json"
    protected.write_text("{}")
    start = threading.Barrier(2)
    pregame = threading.Barrier(2)
    evaluations = 0
    lock = threading.Lock()

    def deterministic(row: Mapping[str, object]) -> bool:
        del row
        try:
            pregame.wait(timeout=0.2)
        except threading.BrokenBarrierError:
            pass
        return True

    def evaluated(row: Mapping[str, object]) -> Mapping[str, object]:
        nonlocal evaluations
        with lock:
            evaluations += 1
        return row

    def invoke(index: int) -> object:
        start.wait()
        identity = _identity(determinism=(801 + 2 * index, 803 + 2 * index))
        try:
            return hybrid_holdout._execute_single_use(
                output=tmp_path / f"promotion-{index}.json",
                run_identity=identity,
                finalists=({"index": 0},),
                protected_paths=(protected,),
                snapshot_root=snapshot,
                deterministic=deterministic,
                evaluate=evaluated,
            )
        except (RuntimeError, ValueError) as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = tuple(pool.map(invoke, (0, 1)))

    errors = tuple(item for item in outcomes if isinstance(item, Exception))
    sentinels = list((tmp_path / ".hybrid-promotion-claims").rglob("identity.json"))
    assert len(errors) == 1
    assert "terminal promotion identity conflicts" in str(errors[0])
    assert len(sentinels) == 1
    assert evaluations <= 1


def test_same_identity_concurrency_allows_only_one_holdout_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Shared preflight may replay, but the finalist claim stays single-use."""
    snapshot = tmp_path / "snapshots"
    snapshot.mkdir()
    protected = tmp_path / "finalists.json"
    protected.write_text("{}")
    pregame = threading.Barrier(2)
    evaluations = 0
    determinism_runs = 0
    lock = threading.Lock()
    original_fdopen = os.fdopen

    def delayed_fdopen(*args: object, **kwargs: object) -> object:
        time.sleep(0.05)
        return cast(Any, original_fdopen)(*args, **kwargs)

    monkeypatch.setattr(hybrid_holdout.os, "fdopen", delayed_fdopen)

    def deterministic(row: Mapping[str, object]) -> bool:
        nonlocal determinism_runs
        del row
        with lock:
            determinism_runs += 1
        pregame.wait(timeout=1)
        return True

    def evaluated(row: Mapping[str, object]) -> Mapping[str, object]:
        nonlocal evaluations
        with lock:
            evaluations += 1
        time.sleep(0.1)
        return row

    def invoke(index: int) -> object:
        try:
            return hybrid_holdout._execute_single_use(
                output=tmp_path / f"promotion-{index}.json",
                run_identity=_identity(),
                finalists=({"index": 0},),
                protected_paths=(protected,),
                snapshot_root=snapshot,
                deterministic=deterministic,
                evaluate=evaluated,
            )
        except RuntimeError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = tuple(pool.map(invoke, (0, 1)))

    errors = tuple(item for item in outcomes if isinstance(item, RuntimeError))
    assert determinism_runs == 2
    assert evaluations == 1
    assert len(errors) == 1
    assert "replay is forbidden" in str(errors[0])


def test_existing_terminal_sentinel_rejects_duplicate_keys_before_games(
    tmp_path: Path,
) -> None:
    """Malformed terminal identity can neither be collapsed nor replaced."""
    snapshot = tmp_path / "snapshots"
    snapshot.mkdir()
    identity = _identity()
    sentinel = hybrid_holdout._canonical_claim_dir(snapshot, identity) / "identity.json"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text('{"schema_version":1,"schema_version":1}')
    determinism_runs = 0

    def deterministic(row: Mapping[str, object]) -> bool:
        nonlocal determinism_runs
        determinism_runs += 1
        return True

    with pytest.raises(ValueError, match="duplicate JSON key"):
        hybrid_holdout._execute_single_use(
            output=tmp_path / "promotion.json",
            run_identity=identity,
            finalists=({"index": 0},),
            protected_paths=(tmp_path / "finalists.json",),
            snapshot_root=snapshot,
            deterministic=deterministic,
            evaluate=lambda row: row,
        )

    assert determinism_runs == 0


@pytest.mark.parametrize("target", ["output", "input"])
def test_lexical_claim_tree_path_cannot_escape_through_intermediate_symlink(
    tmp_path: Path, target: str
) -> None:
    """Lexical containment is rejected even when resolution lands outside."""
    snapshot = tmp_path / "snapshots"
    snapshot.mkdir()
    identity = _identity()
    claims_tree = hybrid_holdout._canonical_claim_dir(snapshot, identity).parent
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_target = outside / "evidence.json"
    outside_target.write_text("unchanged")
    claims_tree.mkdir()
    escape = claims_tree / "escape"
    escape.symlink_to(outside, target_is_directory=True)
    lexical = escape / "evidence.json"
    output = lexical if target == "output" else tmp_path / "promotion.json"
    protected = lexical if target == "input" else tmp_path / "finalists.json"
    if target == "output":
        protected.write_text("{}")

    with pytest.raises(ValueError, match="lexically aliases canonical claim tree"):
        hybrid_holdout._execute_single_use(
            output=output,
            run_identity=identity,
            finalists=({"index": 0},),
            protected_paths=(protected,),
            snapshot_root=snapshot,
            deterministic=lambda row: True,
            evaluate=lambda row: row,
        )

    assert outside_target.read_text() == "unchanged"


@pytest.mark.parametrize("target", ["output", "input"])
def test_outputs_and_copied_inputs_cannot_alias_canonical_claim_tree(
    tmp_path: Path, target: str
) -> None:
    """Aggregate and input paths are both disjoint from durable claim evidence."""
    snapshot = tmp_path / "snapshots"
    snapshot.mkdir()
    identity = _identity()
    claim_dir = hybrid_holdout._canonical_claim_dir(snapshot, identity)
    inside = claim_dir.parent / "alias.json"
    output = inside if target == "output" else tmp_path / "promotion.json"
    protected = inside if target == "input" else tmp_path / "finalists.json"
    protected.parent.mkdir(parents=True, exist_ok=True)
    protected.write_text("{}")

    with pytest.raises(ValueError, match="canonical claim tree"):
        hybrid_holdout._execute_single_use(
            output=output,
            run_identity=identity,
            finalists=({"index": 0},),
            protected_paths=(protected,),
            snapshot_root=snapshot,
            deterministic=lambda row: True,
            evaluate=lambda row: row,
        )


def test_holdout_uses_the_authoritative_task8_finalist_loader() -> None:
    """Task 9 does not maintain an independent finalist parser or validator."""
    assert hybrid_holdout.FinalistArtifact is evolution.FinalistArtifact


def _legacy_frontier_context(
    state: evolution.SearchState,
) -> tuple[VerifiedFrontier, FrontierReport]:
    identity = state.identity
    source_paths = {
        name: source_path
        for name, source_path, _snapshot_path, _digest in identity.league_snapshots
    }
    artifacts = tuple(
        FrontierArtifact(
            name=name,
            provenance="real legacy normalization fixture",
            relative_path=Path(source_paths[name]).name,
            sha256=digest,
        )
        for name, _kind, digest in identity.league_identity
    )
    frontier = VerifiedFrontier(
        engine=identity.engine,
        opponents=source_paths,
        artifacts=artifacts,
        manifest_sha256=identity.manifest_sha256,
    )
    names = tuple(frontier.opponents)
    assert names == ("frontier", "boatlee_v14_current")
    games = 2 * len(FRONTIER_SEEDS)
    rows = (
        FrontierRow(
            name=names[0],
            games=games,
            field_win_points=1.0,
            worst_matchup_win_points=1.0,
            paired_margin=0.25,
            failures=(),
            runtime_seconds=0.0,
            matchups={names[1]: 1.0},
            matchup_results=(
                FrontierMatchup(names[1], games, 1.0, games, 0, 0, 0.25, (), 0.0),
            ),
        ),
        FrontierRow(
            name=names[1],
            games=games,
            field_win_points=0.0,
            worst_matchup_win_points=0.0,
            paired_margin=-0.25,
            failures=(),
            runtime_seconds=0.0,
            matchups={names[0]: 0.0},
            matchup_results=(
                FrontierMatchup(names[0], games, 0.0, 0, 0, games, -0.25, (), 0.0),
            ),
        ),
    )
    report = FrontierReport(
        engine=identity.engine,
        seeds=FRONTIER_SEEDS,
        frontier_name=names[0],
        failures=(),
        runtime_seconds=0.0,
        rows=rows,
        manifest_sha256=identity.manifest_sha256,
        source_sha256={artifact.name: artifact.sha256 for artifact in artifacts},
    )
    return frontier, report


def test_holdout_normalizes_a_real_legacy_certification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The actual legacy loader supplies configs, snapshots, and measured weights."""
    state, certification, _ = _completed_verified_search(monkeypatch, tmp_path)
    certification = cast(Any, certification)
    finalists = tmp_path / "certified-finalists.json"
    evolution.write_finalists(state, finalists, certification=certification)
    frontier, report = _legacy_frontier_context(state)

    promotion = hybrid_holdout.load_promotion_input(
        finalists,
        frontier,
        report,
        report_sha256="a" * 64,
    )

    assert promotion.configs == tuple(row.config for row in state.elites)
    assert all(isinstance(config, HybridConfig) for config in promotion.configs)
    assert promotion.league == certification.snapshot.opponents
    assert promotion.weights == strength_weights(report)
    assert promotion.weights.as_dict() == dict(state.identity.strength_weights)
    assert tuple(row["config"] for row in promotion.finalist_rows) == tuple(
        config.model_dump(mode="json") for config in promotion.configs
    )


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
    claims = tmp_path / ".hybrid-promotion-claims" / ("a" * 64)
    identity = {
        "run_identity": {"promotion_seeds": list(FAKE_SEEDS)},
        "finalist": {"index": 0},
    }
    path = hybrid_holdout._claim_path(
        claims, {"promotion_seeds": list(FAKE_SEEDS)}, {"index": 0}
    )

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
    claim_files = list((tmp_path / ".hybrid-promotion-claims").rglob("*.json"))
    assert sum(path.name != "identity.json" for path in claim_files) == 2


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
    evidence = list((tmp_path / ".hybrid-promotion-claims").rglob("*.json"))
    assert [path.name for path in evidence] == ["identity.json"]


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
            run_identity=_identity(),
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
