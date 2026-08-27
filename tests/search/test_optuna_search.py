"""Sequential coordinator contracts for progressive Optuna search."""

# Test names and protocol-double methods are the behavioral documentation.
# ruff: noqa: D101,D102,D103,D105

from __future__ import annotations

import json
import signal
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import optuna
import pytest

from kaggriculture.hybrid.config import HybridConfig
from kaggriculture.search.arena import GameKey, GameResult, GameTask
from kaggriculture.search.fitness import StrengthWeights
from kaggriculture.search.optuna_protocol import (
    RUNG_1_SEEDS,
    RUNG_2_SEEDS,
    RUNG_3_SEEDS,
    RungEvidence,
    RungSpec,
    build_rung_evidence,
    evidence_path,
    write_rung_evidence_atomic,
)
from kaggriculture.search.optuna_search import (
    OptunaCoordinator,
    PilotVerdict,
    SearchInputs,
    SearchInterrupted,
    SearchRunConfig,
    TrialEvaluationError,
    economic_stop_gate,
    enqueue_warm_starts,
    pilot_gate,
    run_search,
)
from kaggriculture.search.optuna_space import config_sha256, parameters_for_config
from kaggriculture.search.optuna_state import (
    PanelIdentity,
    PrunerIdentity,
    SamplerIdentity,
    SeedBankIdentity,
    StudyIdentity,
    StudyPaths,
    reconcile_running_trials,
    validate_study_evidence,
)


class RecordingTrial:
    """A real semantic-space trial double with observable prune ordering."""

    def __init__(self, *, prune_after_step: int | None = None) -> None:
        self.number = 0
        self.params = parameters_for_config(HybridConfig.default())
        self.prune_after_step = prune_after_step
        self.events: list[str] = []
        self.user_attrs: dict[str, object] = {}
        self.system_attrs: dict[str, object] = {}
        self._step: int | None = None

    def suggest_int(self, name: str, low: int, high: int, *, step: int = 1) -> int:
        value = self.params[name]
        assert type(value) is int and low <= value <= high and (value - low) % step == 0
        return value

    def suggest_float(self, name: str, low: float, high: float) -> float:
        value = self.params[name]
        assert type(value) is float and low <= value <= high
        return value

    def suggest_categorical(self, name: str, choices: Sequence[str]) -> str:
        value = self.params[name]
        assert type(value) is str and value in choices
        return value

    def set_user_attr(self, key: str, value: object) -> None:
        self.user_attrs[key] = value

    def set_system_attr(self, key: str, value: object) -> None:
        self.system_attrs[key] = value

    def report(self, value: float, step: int) -> None:
        assert value >= 0.0
        self._step = step
        self.events.append(f"report:{step}")

    def should_prune(self) -> bool:
        assert self._step is not None
        self.events.append(f"prune:{self._step}")
        return self._step == self.prune_after_step


class RecordingArena:
    def __init__(self, failure_at: GameKey | None = None) -> None:
        self.failure_at = failure_at
        self.requested_keys: list[GameKey] = []
        self.open = False

    def __enter__(self) -> "RecordingArena":
        self.open = True
        return self

    def __exit__(self, *_: object) -> None:
        self.open = False

    def run(self, tasks: Sequence[GameTask]) -> tuple[GameResult, ...]:
        assert self.open
        self.requested_keys.extend(task.key for task in tasks)
        return tuple(
            GameResult(
                key=task.key,
                ours=None if task.key == self.failure_at else 100,
                theirs=None if task.key == self.failure_at else 50,
                runtime_seconds=0.25,
                failure="engine exploded" if task.key == self.failure_at else None,
            )
            for task in tasks
        )


NAMES = (
    "economic_policy",
    "weak_public",
    "second_weak_public",
    "boatlee_v14_current",
    "frontier",
    "median",
    "public_6",
    "public_7",
    "public_8",
    "public_9",
    "public_10",
)
RUNGS = (
    RungSpec(1, 1, NAMES[:3], RUNG_1_SEEDS),
    RungSpec(2, 4, NAMES[:6], RUNG_2_SEEDS),
    RungSpec(3, 16, NAMES, RUNG_3_SEEDS),
)


def forbidden_finalist_writer(*_args: object) -> None:
    pytest.fail("nonterminal search attempted to write finalists")


def coordinator_fixture(
    root: Path, *, failure_at: GameKey | None = None
) -> OptunaCoordinator:
    return OptunaCoordinator(
        paths=StudyPaths.from_root(root),
        league={name: f"{name}.py" for name in NAMES},
        weights=StrengthWeights(dict.fromkeys(NAMES, 1)),
        rungs=RUNGS,
        arena=RecordingArena(failure_at),  # type: ignore[arg-type]
    )


def load_evidence(root: Path, trial: int, rung: int) -> RungEvidence:
    return RungEvidence.model_validate_json(
        evidence_path(root, trial, rung).read_bytes()
    )


def test_objective_writes_each_rung_before_reporting_and_pruning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trial = RecordingTrial(prune_after_step=4)
    coordinator = coordinator_fixture(tmp_path)
    original = __import__(
        "kaggriculture.search.optuna_search", fromlist=["write_rung_evidence_atomic"]
    ).write_rung_evidence_atomic

    def recording_write(root: Path, evidence: RungEvidence) -> tuple[Path, str]:
        trial.events.append(f"write:rung{evidence.rung}")
        return original(root, evidence)

    monkeypatch.setattr(
        "kaggriculture.search.optuna_search.write_rung_evidence_atomic",
        recording_write,
    )
    with coordinator.arena:
        with pytest.raises(optuna.TrialPruned, match="rung 2"):
            coordinator.objective(trial)  # type: ignore[arg-type]
    assert trial.events == [
        "write:rung1",
        "report:1",
        "prune:1",
        "write:rung2",
        "report:4",
        "prune:4",
    ]
    assert len(coordinator.arena.requested_keys) == 24 + 168  # type: ignore[attr-defined]


def test_game_failure_fails_trial_instead_of_returning_bad_score(
    tmp_path: Path,
) -> None:
    coordinator = coordinator_fixture(
        tmp_path, failure_at=GameKey("economic_policy", 860_000, 0)
    )
    with coordinator.arena:
        with pytest.raises(TrialEvaluationError, match="economic_policy/860000/seat0"):
            coordinator.objective(RecordingTrial())  # type: ignore[arg-type]
    assert load_evidence(tmp_path, trial=0, rung=1).failures


def test_invalid_semantic_parameters_fail_before_arena_work(tmp_path: Path) -> None:
    class InvalidTrial(RecordingTrial):
        def suggest_int(self, name: str, low: int, high: int, *, step: int = 1) -> int:
            if name == "liquidation_start_day":
                return 99
            return super().suggest_int(name, low, high, step=step)

    coordinator = coordinator_fixture(tmp_path)
    trial = InvalidTrial()
    with coordinator.arena:
        with pytest.raises(TrialEvaluationError, match="liquidation_start_day"):
            coordinator.objective(trial)  # type: ignore[arg-type]
    assert trial.system_attrs["fail_reason"]
    assert coordinator.arena.requested_keys == []  # type: ignore[attr-defined]


def test_completed_rung_three_returns_exact_stored_objective(tmp_path: Path) -> None:
    coordinator = coordinator_fixture(tmp_path)
    with coordinator.arena:
        value = coordinator.objective(RecordingTrial())  # type: ignore[arg-type]
    assert value == load_evidence(tmp_path, trial=0, rung=3).objective
    assert len(coordinator.arena.requested_keys) == 24 + 168 + 512  # type: ignore[attr-defined]


def warm_configs() -> tuple[HybridConfig, ...]:
    default = HybridConfig.default()
    alternatives = tuple(
        default.model_copy(update={"liquidation_start_day": day})
        for day in range(20, 30)
        if day != default.liquidation_start_day
    )
    return (default, *alternatives[:7])


def identity_fixture(warm_starts: Sequence[HybridConfig]) -> StudyIdentity:
    return StudyIdentity(
        engine="test-engine",
        manifest_sha256="a" * 64,
        frontier_report_sha256="b" * 64,
        source_sha256=(),
        league_snapshots=(),
        space_sha256="c" * 64,
        sampler=SamplerIdentity(),
        pruner=PrunerIdentity(),
        seed_banks=SeedBankIdentity(
            rung_1=RUNG_1_SEEDS,
            rung_2=RUNG_2_SEEDS,
            rung_3=RUNG_3_SEEDS,
            protected_promotion_sha256="d" * 64,
        ),
        panels=PanelIdentity(
            rung_1=cast(tuple[str, str, str], NAMES[:3]),
            rung_2=cast(tuple[str, str, str, str, str, str], NAMES[:6]),
            rung_3=NAMES,
        ),
        strength_weights=tuple((name, 1) for name in NAMES),
        objective_schema="win-primary-margin-epsilon-v1",
        warm_start_sha256=tuple(config_sha256(config) for config in warm_starts),
        legacy_state_sha256="e" * 64,
    )


def write_identity(paths: StudyPaths, identity: StudyIdentity) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.identity.write_text(
        json.dumps(
            identity.model_dump(mode="json"),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def execute_warm_trials(
    study: optuna.Study,
    configs: Sequence[HybridConfig],
    *,
    pruned: bool = False,
) -> None:
    enqueue_warm_starts(study, configs)

    def objective(trial: optuna.Trial) -> float:
        from kaggriculture.search.optuna_space import suggest_config

        suggest_config(trial)
        if pruned:
            trial.report(0.0, 1)
            raise optuna.TrialPruned()
        return 0.0

    study.optimize(objective, n_trials=8)


def test_fresh_study_enqueues_eight_warm_starts_once() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study()
    configs = warm_configs()
    enqueue_warm_starts(study, configs)
    enqueue_warm_starts(study, configs)
    assert len(study.trials) == 8
    assert [trial.system_attrs["fixed_params"] for trial in study.trials] == [
        parameters_for_config(config) for config in configs
    ]


def test_stop_after_counts_complete_pruned_and_failed_trials_on_resume(
    tmp_path: Path,
) -> None:
    study = optuna.create_study()
    configs = warm_configs()
    execute_warm_trials(study, configs)
    for state, count in (
        (optuna.trial.TrialState.COMPLETE, 2),
        (optuna.trial.TrialState.PRUNED, 18),
        (optuna.trial.TrialState.FAIL, 4),
    ):
        for _ in range(count):
            value = 0.0 if state is not optuna.trial.TrialState.FAIL else None
            study.add_trial(optuna.trial.create_trial(state=state, value=value))
    paths = StudyPaths.from_root(tmp_path)
    inputs = SearchInputs(
        study=study,
        paths=paths,
        identity=identity_fixture(configs),
        league={},
        weights=StrengthWeights(dict.fromkeys(NAMES, 1)),
        rungs=RUNGS,
        warm_starts=configs,
    )

    def forbidden_arena(_workers: int) -> RecordingArena:
        raise AssertionError("terminal budget must not create an arena")

    summary = run_search(
        inputs,
        SearchRunConfig(stop_after=32),
        callbacks=(),
        arena_factory=forbidden_arena,  # type: ignore[arg-type]
        finalist_writer=forbidden_finalist_writer,
    )
    assert summary.started_trials == 0
    assert summary.terminal_trials == 32
    assert (summary.complete_trials, summary.pruned_trials, summary.failed_trials) == (
        10,
        18,
        4,
    )


def test_only_validated_512_terminal_completion_writes_finalists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    study = optuna.create_study()
    configs = warm_configs()
    execute_warm_trials(study, configs)
    for _ in range(504):
        study.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.FAIL))
    identity = identity_fixture(configs)
    inputs = SearchInputs(
        study=study,
        paths=StudyPaths.from_root(tmp_path),
        identity=identity,
        league={},
        weights=StrengthWeights(dict.fromkeys(NAMES, 1)),
        rungs=RUNGS,
        warm_starts=configs,
    )
    calls: list[str] = []
    monkeypatch.setattr(
        "kaggriculture.search.optuna_search.economic_stop_gate", lambda *_: False
    )
    monkeypatch.setattr(
        "kaggriculture.search.optuna_search.validate_study_evidence",
        lambda *_: calls.append("validate"),
    )

    summary = run_search(
        inputs,
        SearchRunConfig(stop_after=512),
        callbacks=(),
        arena_factory=lambda _workers: pytest.fail("complete study opened arena"),  # type: ignore[arg-type]
        finalist_writer=lambda *_: calls.append("write"),
    )

    assert summary.stopped_reason == "complete"
    assert calls == ["validate", "write"]


@pytest.mark.parametrize("stop_after", (32, 128, 511))
def test_operational_runs_below_512_never_write_existing_terminal_study(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stop_after: int
) -> None:
    """An operational boundary below 512 cannot certify a prefilled study."""
    study = optuna.create_study()
    configs = warm_configs()
    execute_warm_trials(study, configs)
    for _ in range(504):
        study.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.FAIL))
    inputs = SearchInputs(
        study=study,
        paths=StudyPaths.from_root(tmp_path),
        identity=identity_fixture(configs),
        league={},
        weights=StrengthWeights(dict.fromkeys(NAMES, 1)),
        rungs=RUNGS,
        warm_starts=configs,
    )
    monkeypatch.setattr(
        "kaggriculture.search.optuna_search.economic_stop_gate", lambda *_: False
    )

    summary = run_search(
        inputs,
        SearchRunConfig(stop_after=stop_after),
        callbacks=(),
        arena_factory=lambda _workers: pytest.fail("complete study opened arena"),  # type: ignore[arg-type]
        finalist_writer=lambda *_: pytest.fail("sub-512 operation wrote finalists"),
    )

    assert summary.stopped_reason == "stop_after"
    assert summary.started_trials == 0


def write_rung_one_evidence(
    paths: StudyPaths, trial_number: int, *, wins: bool
) -> None:
    completed = {
        GameKey(name, seed, seat): GameResult(
            GameKey(name, seed, seat),
            100 if wins else 0,
            0 if wins else 100,
            0.01,
        )
        for name in NAMES[:3]
        for seed in RUNG_1_SEEDS
        for seat in (0, 1)
    }
    evidence = build_rung_evidence(
        trial_number,
        warm_configs()[trial_number % 8],
        RUNGS[0],
        completed,
        StrengthWeights(dict.fromkeys(NAMES, 1)),
    )
    write_rung_evidence_atomic(paths.root, evidence)


def study_with_128_terminal_trials(configs: Sequence[HybridConfig]) -> optuna.Study:
    study = optuna.create_study()
    execute_warm_trials(study, configs)
    for _ in range(120):
        study.add_trial(
            optuna.trial.create_trial(
                state=optuna.trial.TrialState.COMPLETE,
                value=0.0,
            )
        )
    return study


def test_no_economic_points_at_128_writes_diagnostic_not_finalists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configs = warm_configs()
    identity = identity_fixture(configs)
    paths = StudyPaths.from_root(tmp_path)
    write_identity(paths, identity)
    study = study_with_128_terminal_trials(configs)
    study.set_user_attr("study_identity_sha256", identity.digest)
    for number in range(128):
        write_rung_one_evidence(paths, number, wins=False)
    monkeypatch.setattr(
        "kaggriculture.search.optuna_search.validate_study_evidence",
        lambda *_: None,
    )
    inputs = SearchInputs(
        study=study,
        paths=paths,
        identity=identity,
        league={},
        weights=StrengthWeights(dict.fromkeys(NAMES, 1)),
        rungs=RUNGS,
        warm_starts=configs,
    )

    summary = run_search(
        inputs,
        SearchRunConfig(stop_after=512),
        callbacks=(),
        arena_factory=lambda _workers: pytest.fail("arena must not open"),  # type: ignore[arg-type]
        finalist_writer=forbidden_finalist_writer,
    )
    assert summary.stopped_reason == "no_economic_points_at_128"
    diagnostic = json.loads(paths.diagnostic.read_text())
    assert diagnostic["reason"] == "no_economic_points_at_128"
    assert diagnostic["best_margin_trial"] == 0
    assert not paths.finalists.exists()


class InterruptingArena(RecordingArena):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    def __exit__(self, *_: object) -> None:
        super().__exit__()
        self.closed = True

    def run(self, tasks: Sequence[GameTask]) -> tuple[GameResult, ...]:
        raise SearchInterrupted("test interruption")


class EntryInterruptingArena(InterruptingArena):
    def __enter__(self) -> "EntryInterruptingArena":
        super().__enter__()
        signal.raise_signal(signal.SIGINT)
        return self


def test_interrupt_closes_arena_preserves_evidence_and_remains_reconcilable(
    tmp_path: Path,
) -> None:
    configs = warm_configs()
    identity = identity_fixture(configs)
    paths = StudyPaths.from_root(tmp_path)
    write_identity(paths, identity)
    study = optuna.create_study()
    study.set_user_attr("study_identity_sha256", identity.digest)
    arena = InterruptingArena()
    inputs = SearchInputs(
        study=study,
        paths=paths,
        identity=identity,
        league={name: f"{name}.py" for name in NAMES},
        weights=StrengthWeights(dict.fromkeys(NAMES, 1)),
        rungs=RUNGS,
        warm_starts=configs,
    )
    summary = run_search(
        inputs,
        SearchRunConfig(stop_after=32),
        callbacks=(),
        arena_factory=lambda _workers: arena,  # type: ignore[arg-type]
        finalist_writer=forbidden_finalist_writer,
    )
    assert summary.stopped_reason == "interrupted"
    assert arena.closed
    assert not paths.finalists.exists()
    assert study.trials[0].state in (
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.RUNNING,
    )
    reconcile_running_trials(study)
    validate_study_evidence(study, paths, identity)


def test_interrupt_after_evidence_write_finishes_optuna_commit_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configs = warm_configs()
    identity = identity_fixture(configs)
    paths = StudyPaths.from_root(tmp_path)
    write_identity(paths, identity)
    study = optuna.create_study()
    study.set_user_attr("study_identity_sha256", identity.digest)
    arena = RecordingArena()
    inputs = SearchInputs(
        study=study,
        paths=paths,
        identity=identity,
        league={name: f"{name}.py" for name in NAMES},
        weights=StrengthWeights(dict.fromkeys(NAMES, 1)),
        rungs=RUNGS,
        warm_starts=configs,
    )
    original_write = write_rung_evidence_atomic
    injected = False

    def interrupt_after_write(root: Path, evidence: RungEvidence) -> tuple[Path, str]:
        nonlocal injected
        result = original_write(root, evidence)
        if not injected:
            injected = True
            signal.raise_signal(signal.SIGINT)
        return result

    monkeypatch.setattr(
        "kaggriculture.search.optuna_search.write_rung_evidence_atomic",
        interrupt_after_write,
    )
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }

    summary = run_search(
        inputs,
        SearchRunConfig(stop_after=32),
        callbacks=(),
        arena_factory=lambda _workers: arena,  # type: ignore[arg-type]
        finalist_writer=forbidden_finalist_writer,
    )

    assert summary.stopped_reason == "interrupted"
    assert not arena.open
    assert not paths.finalists.exists()
    assert {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    } == previous_handlers
    trial = study.trials[0]
    evidence = load_evidence(tmp_path, trial=0, rung=1)
    assert trial.intermediate_values == {1: evidence.objective}
    assert trial.user_attrs["evidence_sha256"]
    assert trial.state in (
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.RUNNING,
    )
    reconcile_running_trials(study)
    validate_study_evidence(study, paths, identity)


def test_interrupt_during_arena_entry_closes_it_and_restores_handlers(
    tmp_path: Path,
) -> None:
    configs = warm_configs()
    identity = identity_fixture(configs)
    paths = StudyPaths.from_root(tmp_path)
    write_identity(paths, identity)
    study = optuna.create_study()
    study.set_user_attr("study_identity_sha256", identity.digest)
    arena = EntryInterruptingArena()
    inputs = SearchInputs(
        study=study,
        paths=paths,
        identity=identity,
        league={name: f"{name}.py" for name in NAMES},
        weights=StrengthWeights(dict.fromkeys(NAMES, 1)),
        rungs=RUNGS,
        warm_starts=configs,
    )
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }

    summary = run_search(
        inputs,
        SearchRunConfig(stop_after=32),
        callbacks=(),
        arena_factory=lambda _workers: arena,  # type: ignore[arg-type]
        finalist_writer=forbidden_finalist_writer,
    )

    assert summary.stopped_reason == "interrupted"
    assert arena.closed
    assert not paths.finalists.exists()
    assert {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    } == previous_handlers
    reconcile_running_trials(study)
    validate_study_evidence(study, paths, identity)


def test_pilot_gate_requires_clean_nondefault_same_rung_improvement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configs = warm_configs()
    identity = identity_fixture(configs)
    paths = StudyPaths.from_root(tmp_path)
    write_identity(paths, identity)
    study = optuna.create_study()
    execute_warm_trials(study, configs, pruned=True)
    for _ in range(24):
        study.add_trial(
            optuna.trial.create_trial(
                state=optuna.trial.TrialState.PRUNED,
                value=0.0,
            )
        )
    study.set_user_attr("study_identity_sha256", identity.digest)
    for number in range(32):
        write_rung_one_evidence(paths, number, wins=number == 1)
    monkeypatch.setattr(
        "kaggriculture.search.optuna_search.validate_study_evidence",
        lambda *_: None,
    )

    verdict = pilot_gate(study, paths)
    assert verdict == PilotVerdict(
        passed=True,
        reasons=(),
        terminal_trials=32,
        best_trial=1,
        default_trial=0,
        best_same_rung_delta=1.000001,
        failures=0,
    )


def test_economic_gate_does_nothing_before_128_trials(tmp_path: Path) -> None:
    study = optuna.create_study()
    for _ in range(127):
        study.add_trial(
            optuna.trial.create_trial(
                state=optuna.trial.TrialState.FAIL,
            )
        )
    assert not economic_stop_gate(study, StudyPaths.from_root(tmp_path))
    assert not (tmp_path / "diagnostic.json").exists()


def test_pilot_gate_rejects_an_extra_running_trial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configs = warm_configs()
    identity = identity_fixture(configs)
    paths = StudyPaths.from_root(tmp_path)
    write_identity(paths, identity)
    study = optuna.create_study()
    execute_warm_trials(study, configs, pruned=True)
    for _ in range(24):
        study.add_trial(
            optuna.trial.create_trial(
                state=optuna.trial.TrialState.PRUNED,
                value=0.0,
            )
        )
    study.ask()
    study.set_user_attr("study_identity_sha256", identity.digest)
    for number in range(32):
        write_rung_one_evidence(paths, number, wins=number == 1)
    monkeypatch.setattr(
        "kaggriculture.search.optuna_search.validate_study_evidence",
        lambda *_: None,
    )

    verdict = pilot_gate(study, paths)
    assert not verdict.passed
    assert "running_trials=1, expected 0" in verdict.reasons
