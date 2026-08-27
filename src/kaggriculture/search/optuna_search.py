"""Sequential Optuna coordinator for progressive hybrid-policy trials."""

from __future__ import annotations

import json
import os
import signal
import tempfile
import warnings
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Iterator, Literal, cast

import optuna
from optuna.trial import FixedTrial, FrozenTrial, TrialState
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from kaggriculture.hybrid.config import HybridConfig, to_runtime
from kaggriculture.search.arena import GameKey, GameResult, HybridOpponent, Opponent
from kaggriculture.search.arena_pool import PersistentArena
from kaggriculture.search.fitness import StrengthWeights
from kaggriculture.search.optuna_protocol import (
    RungEvidence,
    RungSpec,
    build_rung_evidence,
    evidence_path,
    missing_game_tasks,
    write_rung_evidence_atomic,
)
from kaggriculture.search.optuna_space import (
    config_sha256,
    parameters_for_config,
    suggest_config,
)
from kaggriculture.search.optuna_state import (
    StudyEvidenceError,
    StudyIdentity,
    StudyPaths,
    terminal_counts,
    validate_study_evidence,
)

TrialCallback = Callable[[optuna.Study, FrozenTrial], None]
ArenaFactory = Callable[[int], PersistentArena]
SignalHandler = Callable[[int, FrameType | None], object] | int | None


class TrialEvaluationError(RuntimeError):
    """A complete trial that cannot be eligible because execution failed."""


class SearchInterrupted(RuntimeError):  # noqa: N818 - plan-mandated internal name
    """SIGINT or SIGTERM requested a bounded coordinator shutdown."""


@dataclass
class _InterruptionController:
    """Defer signal delivery until durable state and owned resources are safe."""

    study: optuna.Study
    _requested_signal: signal.Signals | None = None
    _delivery_enabled: bool = False
    _commit_depth: int = 0

    def request(self, signum: int) -> None:
        """Record one signal and deliver it only outside a protected boundary."""
        if self._requested_signal is None:
            self._requested_signal = signal.Signals(signum)
        try:
            self.study.stop()
        except RuntimeError:
            pass
        self._deliver_if_safe()

    def enable_delivery(self) -> None:
        """Make a pending request visible once arena cleanup is guaranteed."""
        self._delivery_enabled = True
        self._deliver_if_safe()

    def disable_delivery(self) -> None:
        """Keep signals bounded while arena ownership is changing."""
        self._delivery_enabled = False

    @contextmanager
    def commit_boundary(self) -> Iterator[None]:
        """Finish evidence attrs and reporting before delivering a signal."""
        completed = False
        self._commit_depth += 1
        try:
            yield
            completed = True
        finally:
            self._commit_depth -= 1
            if completed:
                self._deliver_if_safe()

    def _deliver_if_safe(self) -> None:
        if (
            self._requested_signal is None
            or not self._delivery_enabled
            or self._commit_depth
        ):
            return
        name = self._requested_signal.name
        raise SearchInterrupted(f"received {name}")


@dataclass(frozen=True)
class SearchInputs:
    """Every preflighted input needed by the sequential coordinator."""

    study: optuna.Study
    paths: StudyPaths
    identity: StudyIdentity
    league: Mapping[str, Opponent]
    weights: StrengthWeights
    rungs: tuple[RungSpec, RungSpec, RungSpec]
    warm_starts: tuple[HybridConfig, ...]


class SearchRunConfig(BaseModel):
    """Operational bounds that may change without changing study identity."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    workers: int = Field(default=32, ge=1, le=32)
    stop_after: int = Field(default=512, ge=1, le=512)


class SearchSummary(BaseModel):
    """Canonical terminal-state counts returned by one coordinator invocation."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    started_trials: int = Field(ge=0)
    terminal_trials: int = Field(ge=0, le=512)
    complete_trials: int = Field(ge=0)
    pruned_trials: int = Field(ge=0)
    failed_trials: int = Field(ge=0)
    stopped_reason: Literal[
        "stop_after", "no_economic_points_at_128", "complete", "interrupted"
    ]
    best_trial: int | None


class PilotVerdict(BaseModel):
    """Every independent criterion required to continue beyond the pilot."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, allow_inf_nan=False
    )

    passed: bool
    reasons: tuple[str, ...]
    terminal_trials: int = Field(ge=0)
    best_trial: int | None
    default_trial: int = 0
    best_same_rung_delta: float | None
    failures: int = Field(ge=0)


@dataclass(frozen=True)
class OptunaCoordinator:
    """Own progressive evidence and pruning for one open persistent arena."""

    paths: StudyPaths
    league: Mapping[str, Opponent]
    weights: StrengthWeights
    rungs: tuple[RungSpec, RungSpec, RungSpec]
    arena: PersistentArena
    interruptions: _InterruptionController | None = None

    def objective(self, trial: optuna.Trial) -> float:
        """Evaluate one semantic policy through complete, durable rung boundaries."""
        try:
            config = suggest_config(trial)
        except ValidationError as error:
            reason = f"invalid semantic parameters: {error}"
            _set_trial_system_attr(trial, "fail_reason", reason)
            raise TrialEvaluationError(reason) from error
        candidate = HybridOpponent(to_runtime(config))
        completed: dict[GameKey, GameResult] = {}
        evidence: RungEvidence | None = None
        for spec in self.rungs:
            tasks = missing_game_tasks(
                candidate,
                self.league,
                spec.opponents,
                spec.seeds,
                completed,
            )
            rows = self.arena.run(tasks)
            completed.update((row.key, row) for row in rows)
            evidence = build_rung_evidence(
                trial.number,
                config,
                spec,
                completed,
                self.weights,
            )
            boundary = (
                self.interruptions.commit_boundary()
                if self.interruptions is not None
                else nullcontext()
            )
            with boundary:
                path, digest = write_rung_evidence_atomic(self.paths.root, evidence)
                _set_trial_summary(trial, evidence, path, digest, self.paths.root)
                if not evidence.failures:
                    if evidence.objective is None:
                        raise RuntimeError("clean rung evidence has no objective")
                    trial.report(evidence.objective, step=spec.resource_step)
            if evidence.failures:
                raise TrialEvaluationError("; ".join(evidence.failures))
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned after rung {spec.rung}")
        if evidence is None or evidence.objective is None:
            raise RuntimeError("completed rung-three evidence has no objective")
        return evidence.objective


def _set_trial_summary(
    trial: optuna.Trial,
    evidence: RungEvidence,
    path: Path,
    digest: str,
    root: Path,
) -> None:
    """Persist canonical scalar/count/path facts while raw games stay on disk."""
    attributes: dict[str, object] = {
        "config_sha256": evidence.config_sha256,
        "rung": evidence.rung,
        "resource_step": evidence.resource_step,
        "primary": evidence.primary,
        "dense_margin": evidence.dense_margin,
        "objective": evidence.objective,
        "games": len(evidence.games),
        "failures": len(evidence.failures),
        "runtime_seconds": sum(game.runtime_seconds for game in evidence.games),
        "evidence_path": str(path.relative_to(root)),
        "evidence_sha256": digest,
    }
    for name, matchup in evidence.matchups.items():
        prefix = f"opponent/{name}"
        attributes.update(
            {
                f"{prefix}/wins": matchup.wins,
                f"{prefix}/draws": matchup.draws,
                f"{prefix}/losses": matchup.losses,
                f"{prefix}/games": matchup.games,
                f"{prefix}/win_points": matchup.win_points,
                f"{prefix}/paired_normalized_margin": (
                    matchup.paired_normalized_margin
                ),
                f"{prefix}/runtime_seconds": matchup.runtime_seconds,
            }
        )
    for key, value in attributes.items():
        trial.set_user_attr(key, value)


def enqueue_warm_starts(
    study: optuna.Study, warm_starts: Sequence[HybridConfig]
) -> None:
    """Enqueue the eight identity-bound starts once or validate their exact rows."""
    starts = tuple(warm_starts)
    if len(starts) != 8 or any(
        not isinstance(config, HybridConfig) for config in starts
    ):
        raise ValueError("warm starts must contain exactly eight HybridConfig values")
    expected_digests = tuple(config_sha256(config) for config in starts)
    if len(expected_digests) != len(set(expected_digests)):
        raise ValueError("warm starts must contain eight distinct configurations")
    trials = study.get_trials(deepcopy=False)
    if not trials:
        for config, digest in zip(starts, expected_digests, strict=True):
            study.enqueue_trial(
                parameters_for_config(config),
                user_attrs={"warm_start_sha256": digest},
            )
        return
    if len(trials) < 8 or tuple(trial.number for trial in trials[:8]) != tuple(
        range(8)
    ):
        raise ValueError("study does not contain the complete trials 0-7 warm start")
    for number, (trial, config, digest) in enumerate(
        zip(trials[:8], starts, expected_digests, strict=True)
    ):
        expected = parameters_for_config(config)
        fixed = trial.system_attrs.get("fixed_params")
        actual = dict(trial.params) if trial.params else fixed
        if actual != expected:
            raise ValueError(f"trial {number} parameters differ from warm start")
        try:
            rebuilt = suggest_config(FixedTrial(cast(dict[str, object], actual)))
        except (KeyError, ValueError, ValidationError) as error:
            raise ValueError(
                f"trial {number} warm start parameters are invalid"
            ) from error
        if config_sha256(rebuilt) != digest:
            raise ValueError(f"trial {number} config digest differs from warm start")


def run_search(
    inputs: SearchInputs,
    config: SearchRunConfig,
    callbacks: Sequence[TrialCallback],
    arena_factory: ArenaFactory = PersistentArena,
) -> SearchSummary:
    """Run one bounded sequential optimization invocation with one arena."""
    enqueue_warm_starts(inputs.study, inputs.warm_starts)
    initial_counts = terminal_counts(inputs.study)
    initial_terminal = sum(initial_counts.values())
    if economic_stop_gate(inputs.study, inputs.paths):
        return _search_summary(inputs.study, 0, "no_economic_points_at_128")
    remaining = max(0, config.stop_after - initial_terminal)
    if remaining == 0:
        reason: Literal["stop_after", "complete"] = (
            "complete"
            if initial_terminal == inputs.identity.maximum_trials
            else "stop_after"
        )
        return _search_summary(inputs.study, 0, reason)

    arena = arena_factory(config.workers)
    interruptions = _InterruptionController(inputs.study)
    coordinator = OptunaCoordinator(
        paths=inputs.paths,
        league=inputs.league,
        weights=inputs.weights,
        rungs=inputs.rungs,
        arena=arena,
        interruptions=interruptions,
    )
    started_trials = 0
    economic_stopped = False

    def objective(trial: optuna.Trial) -> float:
        nonlocal started_trials
        started_trials += 1
        try:
            return coordinator.objective(trial)
        except SearchInterrupted:
            _record_interruption(trial)
            raise

    def stop_gate(study: optuna.Study, _trial: FrozenTrial) -> None:
        nonlocal economic_stopped
        counts = terminal_counts(study)
        if sum(counts.values()) >= 128 and economic_stop_gate(study, inputs.paths):
            economic_stopped = True
            study.stop()

    interrupted = False
    previous_handlers: dict[signal.Signals, SignalHandler] = {}
    try:
        previous_handlers = _install_signal_handlers(interruptions)
        with arena:
            try:
                interruptions.enable_delivery()
                inputs.study.optimize(
                    objective,
                    n_trials=remaining,
                    n_jobs=1,
                    callbacks=(*callbacks, stop_gate),
                    catch=(TrialEvaluationError,),
                )
            finally:
                interruptions.disable_delivery()
        interruptions.enable_delivery()
    except SearchInterrupted:
        interrupted = True
    finally:
        interruptions.disable_delivery()
        _restore_signal_handlers(previous_handlers)
    if interrupted:
        return _search_summary(inputs.study, started_trials, "interrupted")
    if economic_stopped:
        return _search_summary(
            inputs.study, started_trials, "no_economic_points_at_128"
        )
    terminal = sum(terminal_counts(inputs.study).values())
    reason = "complete" if terminal == inputs.identity.maximum_trials else "stop_after"
    return _search_summary(inputs.study, started_trials, reason)


def pilot_gate(study: optuna.Study, paths: StudyPaths) -> PilotVerdict:
    """Return every independent reason the permanent 32-trial pilot cannot continue."""
    counts = terminal_counts(study)
    terminal = sum(counts.values())
    reasons: list[str] = []
    if terminal != 32:
        reasons.append(f"terminal_trials={terminal}, expected 32")
    running = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
    if running:
        reasons.append(f"running_trials={len(running)}, expected 0")
    if counts["failed"]:
        reasons.append(f"failed_trials={counts['failed']}, expected 0")
    reasons.extend(_pilot_validation_reasons(study, paths))
    if paths.finalists.exists():
        reasons.append("finalists artifact exists before full completion")

    evidence, evidence_reasons = _pilot_evidence(study, paths)
    reasons.extend(evidence_reasons)
    best_delta, comparison_reasons = _pilot_default_comparison(study, evidence)
    reasons.extend(comparison_reasons)
    best_trial = _best_evidence_trial(evidence)
    return PilotVerdict(
        passed=not reasons,
        reasons=tuple(reasons),
        terminal_trials=terminal,
        best_trial=best_trial,
        default_trial=0,
        best_same_rung_delta=best_delta,
        failures=counts["failed"],
    )


def economic_stop_gate(study: optuna.Study, paths: StudyPaths) -> bool:
    """Atomically stop a 128-trial study that never scores on economic policy."""
    trials = tuple(
        trial
        for trial in study.get_trials(deepcopy=False)
        if trial.state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)
    )
    if len(trials) < 128:
        return False
    identity = _stored_identity(paths)
    validate_study_evidence(study, paths, identity)
    first_boundary = {trial.number for trial in trials[:128]}
    evidence = _latest_terminal_evidence(study, paths)
    boundary_evidence = {
        number: row for number, row in evidence.items() if number in first_boundary
    }
    any_economic_points = any(
        not row.failures
        and "economic_policy" in row.matchups
        and row.matchups["economic_policy"].win_points > 0.0
        for row in boundary_evidence.values()
    )
    if any_economic_points:
        if paths.diagnostic.exists():
            raise StudyEvidenceError(
                "economic diagnostic exists despite positive economic win points"
            )
        return False
    if paths.finalists.exists():
        raise StudyEvidenceError("finalists exist at failed 128-trial economic gate")
    best_number = _best_margin_trial(boundary_evidence)
    best = boundary_evidence.get(best_number) if best_number is not None else None
    diagnostic = {
        "schema": "optuna-economic-diagnostic-v1",
        "reason": "no_economic_points_at_128",
        "terminal_trials": 128,
        "best_margin_trial": best_number,
        "best_margin_rung": best.rung if best is not None else None,
        "best_dense_margin": best.dense_margin if best is not None else None,
        "best_config_sha256": best.config_sha256 if best is not None else None,
        "study_identity_sha256": identity.digest,
    }
    _write_canonical_atomic(paths.diagnostic, diagnostic)
    return True


def _record_interruption(trial: optuna.Trial) -> None:
    """Make an Optuna-eager FAIL acceptable to the next-start validator."""
    _set_trial_system_attr(trial, "interruption_reason", "process_restarted")


def _set_trial_system_attr(trial: optuna.Trial, key: str, value: str) -> None:
    """Use Optuna's deprecated-but-public 4.9 trial durability boundary."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        trial.set_system_attr(key, value)


def _install_signal_handlers(
    interruptions: _InterruptionController,
) -> dict[signal.Signals, SignalHandler]:
    """Install bounded handlers only at the coordinator run boundary."""
    previous: dict[signal.Signals, SignalHandler] = {}

    def handle(signum: int, _frame: FrameType | None) -> None:
        interruptions.request(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, handle)
    return previous


def _restore_signal_handlers(
    previous: Mapping[signal.Signals, SignalHandler],
) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _search_summary(
    study: optuna.Study,
    started_trials: int,
    reason: Literal[
        "stop_after", "no_economic_points_at_128", "complete", "interrupted"
    ],
) -> SearchSummary:
    counts = terminal_counts(study)
    return SearchSummary(
        started_trials=started_trials,
        terminal_trials=sum(counts.values()),
        complete_trials=counts["complete"],
        pruned_trials=counts["pruned"],
        failed_trials=counts["failed"],
        stopped_reason=reason,
        best_trial=_best_complete_trial(study),
    )


def _best_complete_trial(study: optuna.Study) -> int | None:
    try:
        return study.best_trial.number
    except ValueError:
        return None


def _pilot_validation_reasons(
    study: optuna.Study, paths: StudyPaths
) -> tuple[str, ...]:
    try:
        validate_study_evidence(study, paths, _stored_identity(paths))
    except (OSError, ValueError, StudyEvidenceError) as error:
        return (f"invalid_study_evidence: {error}",)
    return ()


def _pilot_evidence(
    study: optuna.Study, paths: StudyPaths
) -> tuple[dict[int, RungEvidence], tuple[str, ...]]:
    try:
        return _latest_terminal_evidence(study, paths), ()
    except (OSError, ValueError, ValidationError) as error:
        return {}, (f"invalid_pilot_evidence: {error}",)


def _pilot_default_comparison(
    study: optuna.Study, evidence: Mapping[int, RungEvidence]
) -> tuple[float | None, tuple[str, ...]]:
    reasons: list[str] = []
    default = evidence.get(0)
    trials = study.get_trials(deepcopy=False)
    default_params_ok = bool(
        trials
        and trials[0].number == 0
        and dict(trials[0].params) == parameters_for_config(HybridConfig.default())
    )
    if default is None or default.failures or default.objective is None:
        reasons.append("default trial 0 has no clean terminal rung evidence")
    if not default_params_ok:
        reasons.append("trial 0 parameters differ from HybridConfig.default()")
    challengers: list[tuple[float, int]] = []
    if default is not None and not default.failures and default.objective is not None:
        challengers = [
            (row.objective - default.objective, number)
            for number, row in evidence.items()
            if number != 0
            and row.rung == default.rung
            and not row.failures
            and row.objective is not None
            and row.objective > default.objective
        ]
    if not challengers:
        reasons.append("no non-default clean trial beats trial 0 at the same rung")
        return None, tuple(reasons)
    best_delta, _ = max(challengers, key=lambda item: (item[0], -item[1]))
    return best_delta, tuple(reasons)


def _stored_identity(paths: StudyPaths) -> StudyIdentity:
    source = paths.identity.read_bytes()
    identity = StudyIdentity.model_validate_json(source)
    if source != _canonical_json(identity.model_dump(mode="json")):
        raise StudyEvidenceError("identity.json is not canonical JSON")
    return identity


def _latest_terminal_evidence(
    study: optuna.Study, paths: StudyPaths
) -> dict[int, RungEvidence]:
    terminal_numbers = {
        trial.number
        for trial in study.get_trials(deepcopy=False)
        if trial.state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)
    }
    latest: dict[int, RungEvidence] = {}
    if not paths.evidence.exists():
        return latest
    for path in paths.evidence.iterdir():
        evidence = RungEvidence.model_validate_json(path.read_bytes())
        if path != evidence_path(paths.root, evidence.trial_number, evidence.rung):
            raise StudyEvidenceError(f"evidence path is not canonical: {path.name}")
        if evidence.trial_number not in terminal_numbers:
            continue
        current = latest.get(evidence.trial_number)
        if current is None or evidence.rung > current.rung:
            latest[evidence.trial_number] = evidence
    return latest


def _best_evidence_trial(evidence: Mapping[int, RungEvidence]) -> int | None:
    eligible = tuple(
        (number, row)
        for number, row in evidence.items()
        if not row.failures and row.objective is not None
    )
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda item: (item[1].rung, cast(float, item[1].objective), -item[0]),
    )[0]


def _best_margin_trial(evidence: Mapping[int, RungEvidence]) -> int | None:
    clean = tuple(
        (number, row)
        for number, row in evidence.items()
        if not row.failures and row.dense_margin is not None
    )
    if not clean:
        return None
    return max(
        clean,
        key=lambda item: (
            cast(float, item[1].dense_margin),
            item[1].rung,
            cast(float, item[1].objective),
            -item[0],
        ),
    )[0]


def _write_canonical_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    source = _canonical_json(payload) + b"\n"
    if path.exists():
        if path.read_bytes() != source:
            raise StudyEvidenceError(f"existing {path.name} differs from stop gate")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(source)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)  # noqa: PTH105
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
