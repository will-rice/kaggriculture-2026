"""Failure-isolated, single-run W&B telemetry for Optuna hybrid trials."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, Self, cast

import optuna
from optuna.trial import FrozenTrial
from pydantic import BaseModel, ConfigDict

from kaggriculture.search.optuna_state import StudyIdentity


class WandbSettings(BaseModel):
    """Operational W&B settings that do not affect the study identity."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    enabled: bool = True
    entity: str = "will-rice"
    project: str = "kaggriculture-2026"


class WandbRun(Protocol):
    """The small W&B run surface used by the telemetry observer."""

    def log(
        self,
        metrics: Mapping[str, float],
        *,
        step: int,
        commit: bool | None = None,
    ) -> None:
        """Log metrics to the active run."""
        ...

    def finish(self, exit_code: int = 0, quiet: bool = True) -> None:
        """Finish the active run."""
        ...


class WandbApi(Protocol):
    """The W&B module surface shared with the official Optuna callback."""

    run: WandbRun | None

    def init(self, **kwargs: object) -> WandbRun:
        """Initialize one active W&B run."""
        ...


@dataclass(frozen=True)
class WandbDependencies:
    """Late-bound W&B dependencies, allowing no-W&B operation and fake tests."""

    api: WandbApi
    callback_factory: Callable[..., object]


def load_wandb_dependencies() -> WandbDependencies:
    """Load optional telemetry packages only when W&B was requested."""
    from optuna_integration import WeightsAndBiasesCallback

    import wandb

    return WandbDependencies(cast(WandbApi, wandb), WeightsAndBiasesCallback)


def wandb_config(identity: StudyIdentity, revision: str) -> dict[str, object]:
    """Return immutable study facts displayed in the persistent W&B run."""
    return {
        "study_identity_sha256": identity.digest,
        "study_identity": identity.model_dump(mode="json"),
        "revision": revision,
    }


@dataclass
class WandbSession:
    """A best-effort W&B observer that cannot affect Optuna's authority."""

    run: WandbRun | None
    official: Callable[[optuna.Study, FrozenTrial], None] | None
    maximum_trials: int | None = None
    disabled_reason: str | None = None

    @classmethod
    def disabled(cls, reason: str) -> Self:
        """Create an inert session after an optional telemetry failure."""
        return cls(run=None, official=None, maximum_trials=None, disabled_reason=reason)

    @classmethod
    def open(
        cls,
        settings: WandbSettings,
        identity: StudyIdentity,
        revision: str,
        *,
        dependencies: WandbDependencies | None = None,
    ) -> Self:
        """Open exactly one official single-run callback or degrade locally."""
        if not settings.enabled:
            return cls.disabled("disabled_by_config")
        try:
            resolved = dependencies or load_wandb_dependencies()
            init_kwargs: dict[str, object] = {
                "entity": settings.entity,
                "project": settings.project,
                "id": f"hybrid-optuna-{identity.digest[:20]}",
                "name": "hybrid-optuna-v2",
                "resume": "allow",
                "config": wandb_config(identity, revision),
            }
            # Optuna Integration 4.9.0 creates its single run in this
            # constructor. Supplying its init kwargs keeps that authoritative
            # callback and avoids a second explicit ``wandb.init`` here.
            official = resolved.callback_factory(
                wandb_kwargs=init_kwargs,
                as_multirun=False,
            )
            run = resolved.api.run
            if run is None:
                raise RuntimeError("official W&B callback did not initialize a run")
            return cls(
                run=run,
                official=cast(Callable[[optuna.Study, FrozenTrial], None], official),
                maximum_trials=identity.maximum_trials,
            )
        except Exception as error:
            return cls.disabled(_failure_reason(error))

    def callback(self, study: optuna.Study, trial: FrozenTrial) -> None:
        """Merge rich scalar telemetry into the official trial-number row."""
        if self.disabled_reason is not None:
            return
        assert (
            self.run is not None
            and self.official is not None
            and self.maximum_trials is not None
        )
        try:
            metrics = trial_metrics(trial)
            metrics.update(_current_best_metrics(study))
            metrics.update(_progress_metrics(study, self.maximum_trials))
            self.run.log(metrics, step=trial.number, commit=False)
            self.official(study, trial)
            # The official 4.9 callback uses an explicit step without commit.
            # W&B therefore keeps that row pending; finalize the merged row
            # explicitly so a process crash cannot lose this terminal trial.
            self.run.log({}, step=trial.number, commit=True)
        except Exception as error:
            self.disabled_reason = _failure_reason(error)

    @property
    def callbacks(self) -> tuple[Callable[[optuna.Study, FrozenTrial], None], ...]:
        """Expose the callback only while the observer remains usable."""
        return () if self.disabled_reason is not None else (self.callback,)

    def close(self) -> None:
        """Finish the optional run without allowing telemetry to fail search."""
        if self.run is None:
            return
        try:
            self.run.finish(quiet=True)
        except Exception as error:
            self.disabled_reason = _failure_reason(error)
        finally:
            self.run = None


def trial_metrics(trial: FrozenTrial) -> dict[str, float]:
    """Return finite scalar terminal-trial telemetry without mutating Optuna."""
    metrics = {
        "trial/number": float(trial.number),
        "trial/state": float(trial.state.value),
    }
    if trial.value is not None:
        metrics["trial/value"] = _finite_scalar(trial.value, "trial/value")
    for name, value in trial.user_attrs.items():
        if name in {"throughput_games_per_second", "eta_seconds"}:
            continue
        if isinstance(value, bool | int | float):
            metrics[name] = _finite_scalar(value, name)
    return metrics


def _finite_scalar(value: bool | int | float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _current_best_metrics(study: optuna.Study) -> dict[str, float]:
    """Return finite scalar facts from the current best without changing study state."""
    try:
        best = study.best_trial
    except ValueError:
        return {}
    metrics = {"best/trial": float(best.number)}
    if best.value is not None:
        metrics["best/value"] = _finite_scalar(best.value, "best/value")
    for name, value in best.user_attrs.items():
        if isinstance(value, bool | int | float):
            metrics[f"best/{name.removeprefix('opponent/')}"] = _finite_scalar(
                value,
                f"best/{name}",
            )
    return metrics


def _progress_metrics(study: optuna.Study, maximum_trials: int) -> dict[str, float]:
    """Derive finite throughput and ETA from authoritative terminal trial facts."""
    terminal = study.get_trials(
        deepcopy=False,
        states=(
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.FAIL,
        ),
    )
    terminal_count = len(terminal)
    if terminal_count == 0 or terminal_count > maximum_trials:
        raise ValueError("terminal trial count is outside the configured budget")
    games = sum(_positive_attr(item, "games") for item in terminal)
    runtime = sum(_positive_attr(item, "trial_wall_seconds") for item in terminal)
    throughput = games / runtime
    average_runtime = runtime / terminal_count
    eta = average_runtime * (maximum_trials - terminal_count)
    return {
        "terminal_trials": float(terminal_count),
        "remaining_trials": float(maximum_trials - terminal_count),
        "throughput_games_per_second": _finite_scalar(
            throughput,
            "throughput_games_per_second",
        ),
        "eta_seconds": _finite_scalar(eta, "eta_seconds"),
    }


def _positive_attr(trial: FrozenTrial, name: str) -> float:
    """Read one required positive finite Task 5 scalar from a terminal trial."""
    value = trial.user_attrs.get(name)
    if not isinstance(value, bool | int | float):
        raise ValueError(f"{name} is required for terminal trial telemetry")
    numeric = _finite_scalar(value, name)
    if numeric <= 0.0:
        raise ValueError(f"{name} must be positive for terminal trial telemetry")
    return numeric


def _failure_reason(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"
