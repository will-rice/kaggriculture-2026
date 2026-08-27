"""One-run, failure-isolated W&B telemetry contracts."""

# Test names are the behavioral documentation.
# ruff: noqa: D101, D102, D103

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Literal, cast

import optuna
import pytest
from optuna.trial import FrozenTrial

from kaggriculture.search.optuna_state import StudyIdentity
from kaggriculture.search.optuna_wandb import (
    WandbDependencies,
    WandbRun,
    WandbSession,
    WandbSettings,
    trial_metrics,
)

from .test_optuna_search import identity_fixture, warm_configs


@dataclass(frozen=True)
class LoggedRow:
    metrics: dict[str, float]
    step: int
    commit: bool


@dataclass(frozen=True)
class LogCall:
    metrics: dict[str, float]
    step: int
    commit: bool | None


class FakeRun:
    def __init__(
        self,
        api: FakeWandb,
        *,
        fail_log_at: Literal["rich", "commit"] | None = None,
        fail_finish: bool = False,
    ) -> None:
        self.api = api
        self.fail_log_at = fail_log_at
        self.fail_finish = fail_finish

    def log(
        self,
        metrics: Mapping[str, float],
        *,
        step: int,
        commit: bool | None = None,
    ) -> None:
        call = LogCall(dict(metrics), step, commit)
        self.api.log_attempts.append(call)
        if self.fail_log_at == "rich" and commit is False:
            raise RuntimeError("rich log unavailable")
        if self.fail_log_at == "commit" and commit is True:
            raise RuntimeError("final commit unavailable")
        self.api.log_calls.append(call)
        self.api.pending.setdefault(step, {}).update(metrics)
        if commit is True:
            self.api.history.append(LoggedRow(self.api.pending.pop(step), step, True))

    def finish(self, exit_code: int = 0, quiet: bool = True) -> None:
        del exit_code, quiet
        self.api.finish_calls += 1
        if self.fail_finish:
            raise RuntimeError("finish unavailable")
        for step in sorted(self.api.pending):
            self.api.history.append(LoggedRow(self.api.pending[step], step, True))
        self.api.pending.clear()


class FakeWandb:
    def __init__(
        self,
        *,
        fail_init: bool = False,
        fail_log_at: Literal["rich", "commit"] | None = None,
        fail_finish: bool = False,
    ) -> None:
        self.fail_init = fail_init
        self.fail_log_at = fail_log_at
        self.fail_finish = fail_finish
        self.init_calls: list[dict[str, object]] = []
        self.history: list[LoggedRow] = []
        self.log_calls: list[LogCall] = []
        self.log_attempts: list[LogCall] = []
        self.pending: dict[int, dict[str, float]] = {}
        self.finish_calls = 0
        self.run: WandbRun | None = None

    def init(self, **kwargs: object) -> WandbRun:
        self.init_calls.append(kwargs)
        if self.fail_init:
            raise RuntimeError("init unavailable")
        self.run = FakeRun(
            self,
            fail_log_at=self.fail_log_at,
            fail_finish=self.fail_finish,
        )
        return self.run


def fake_official_factory(
    api: FakeWandb, *, fail: bool = False
) -> Callable[..., object]:
    """Model the official callback's uncommitted explicit-step W&B log."""

    def factory(
        *, wandb_kwargs: dict[str, object], as_multirun: bool
    ) -> Callable[[optuna.Study, FrozenTrial], None]:
        assert as_multirun is False
        api.init(**wandb_kwargs)

        def official(_study: optuna.Study, trial: FrozenTrial) -> None:
            if fail:
                raise RuntimeError("official callback unavailable")
            assert api.run is not None
            api.run.log({"value": float(trial.value or 0.0)}, step=trial.number)

        return official

    return factory


@pytest.fixture
def identity() -> StudyIdentity:
    return identity_fixture(warm_configs())


@pytest.fixture
def study() -> optuna.Study:
    return optuna.create_study(direction="maximize")


def terminal_trial(number: int) -> object:
    return SimpleNamespace(
        number=number,
        state=optuna.trial.TrialState.COMPLETE,
        value=0.625,
        user_attrs={
            "rung": 3,
            "resource_step": 16,
            "games": 704,
            "failures": 0,
            "runtime_seconds": 32.0,
            "arena_wall_seconds": 6.0,
            "trial_wall_seconds": 8.0,
            "primary": 0.5,
            "dense_margin": 0.25,
            "objective": 0.50000025,
            "opponent/economic_policy/win_points": 0.125,
            "opponent/economic_policy/games": 64,
            "best_trial": 1,
            "best/economic/win_points": 0.125,
        },
    )


def set_task_five_attrs(
    trial: optuna.Trial,
    *,
    games: int = 24,
    runtime_seconds: float = 12.0,
    arena_wall_seconds: float = 2.0,
    trial_wall_seconds: float = 3.0,
) -> None:
    trial.set_user_attr("rung", 1)
    trial.set_user_attr("resource_step", 1)
    trial.set_user_attr("games", games)
    trial.set_user_attr("failures", 0)
    trial.set_user_attr("runtime_seconds", runtime_seconds)
    trial.set_user_attr("arena_wall_seconds", arena_wall_seconds)
    trial.set_user_attr("trial_wall_seconds", trial_wall_seconds)
    trial.set_user_attr("primary", 0.5)
    trial.set_user_attr("dense_margin", 0.25)
    trial.set_user_attr("objective", 0.50000025)
    trial.set_user_attr("opponent/economic_policy/win_points", 0.125)


def frozen_trial_payload(
    trials: list[FrozenTrial],
) -> list[tuple[object, ...]]:
    return [
        (
            trial.number,
            trial.state,
            trial.value,
            dict(trial.params),
            dict(trial.user_attrs),
            dict(trial.intermediate_values),
        )
        for trial in trials
    ]


def test_session_initializes_once_and_combines_rich_metrics_with_official_callback(
    identity: StudyIdentity, study: optuna.Study
) -> None:
    def objective(trial: optuna.Trial) -> float:
        set_task_five_attrs(trial)
        return 0.625

    study.optimize(objective, n_trials=2)
    api = FakeWandb()
    session = WandbSession.open(
        WandbSettings(),
        identity,
        "abc123",
        dependencies=WandbDependencies(api, fake_official_factory(api)),
    )

    for trial in study.trials:
        session.callbacks[0](study, trial)
    assert [row.step for row in api.history] == [0, 1]
    assert not api.pending
    session.close()

    assert len(api.init_calls) == 1
    assert api.init_calls[0]["id"] == f"hybrid-optuna-{identity.digest[:20]}"
    assert [row.step for row in api.history] == [0, 1]
    assert all(row.commit for row in api.history)
    assert [call.commit for call in api.log_calls] == [
        False,
        None,
        True,
        False,
        None,
        True,
    ]
    assert api.history[1].metrics["best/economic_policy/win_points"] == 0.125
    assert api.finish_calls == 1


@pytest.mark.parametrize(
    (
        "failure",
        "reason",
        "attempt_commits",
        "successful_commits",
        "pending_before_close",
        "history_after_close",
    ),
    (
        ("import", "ImportError: wandb missing", (), (), False, ()),
        ("init", "RuntimeError: init unavailable", (), (), False, ()),
        ("rich", "RuntimeError: rich log unavailable", (False,), (), False, ()),
        (
            "official",
            "RuntimeError: official callback unavailable",
            (False,),
            (False,),
            True,
            (0,),
        ),
        (
            "commit",
            "RuntimeError: final commit unavailable",
            (False, None, True),
            (False, None),
            True,
            (0,),
        ),
        (
            "finish",
            "RuntimeError: finish unavailable",
            (False, None, True),
            (False, None, True),
            False,
            (0,),
        ),
    ),
)
def test_wandb_failure_never_changes_study_or_raises(
    monkeypatch: pytest.MonkeyPatch,
    identity: StudyIdentity,
    study: optuna.Study,
    failure: str,
    reason: str,
    attempt_commits: tuple[bool | None, ...],
    successful_commits: tuple[bool | None, ...],
    pending_before_close: bool,
    history_after_close: tuple[int, ...],
) -> None:
    def objective(trial: optuna.Trial) -> float:
        set_task_five_attrs(trial)
        return 1.0

    study.optimize(objective, n_trials=1)
    before = frozen_trial_payload(study.trials)
    api = FakeWandb(
        fail_init=failure == "init",
        fail_log_at=cast(Literal["rich", "commit"] | None, failure)
        if failure in {"rich", "commit"}
        else None,
        fail_finish=failure == "finish",
    )
    if failure == "import":
        monkeypatch.setattr(
            "kaggriculture.search.optuna_wandb.load_wandb_dependencies",
            lambda: (_ for _ in ()).throw(ImportError("wandb missing")),
        )
        session = WandbSession.open(WandbSettings(), identity, "abc123")
    else:
        session = WandbSession.open(
            WandbSettings(),
            identity,
            "abc123",
            dependencies=WandbDependencies(
                api,
                fake_official_factory(api, fail=failure == "official"),
            ),
        )
    session.callback(study, study.trials[0])
    if failure == "finish":
        assert session.disabled_reason is None
    else:
        assert session.disabled_reason == reason
    assert [call.commit for call in api.log_attempts] == list(attempt_commits)
    assert [call.commit for call in api.log_calls] == list(successful_commits)
    assert bool(api.pending) is pending_before_close
    session.close()

    assert frozen_trial_payload(study.trials) == before
    assert session.disabled_reason == reason
    assert [row.step for row in api.history] == list(history_after_close)
    assert api.finish_calls == (0 if failure in {"import", "init"} else 1)


def test_trial_metrics_rejects_nonfinite_values_and_preserves_scalar_telemetry() -> (
    None
):
    trial = cast(FrozenTrial, terminal_trial(8))
    metrics = trial_metrics(trial)

    assert metrics["trial/number"] == 8.0
    assert metrics["trial/state"] == float(optuna.trial.TrialState.COMPLETE.value)
    assert metrics["opponent/economic_policy/win_points"] == 0.125
    assert {
        "rung",
        "resource_step",
        "games",
        "failures",
        "runtime_seconds",
        "arena_wall_seconds",
        "trial_wall_seconds",
        "primary",
        "dense_margin",
        "objective",
    } <= metrics.keys()
    trial.user_attrs["runtime_seconds"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        trial_metrics(trial)


def test_callback_includes_current_best_trial_scalars(identity: StudyIdentity) -> None:
    study = optuna.create_study(direction="maximize")

    def objective(trial: optuna.Trial) -> float:
        set_task_five_attrs(trial)
        trial.set_user_attr("opponent/economic_policy/win_points", 0.125)
        return float(trial.number)

    study.optimize(objective, n_trials=2)
    api = FakeWandb()
    session = WandbSession.open(
        WandbSettings(),
        identity,
        "abc123",
        dependencies=WandbDependencies(api, fake_official_factory(api)),
    )
    session.callback(study, study.trials[0])

    assert api.history[-1].metrics["best/trial"] == 1.0
    assert api.history[-1].metrics["best/economic_policy/win_points"] == 0.125


def test_callback_derives_throughput_and_eta_from_task_five_trial_facts(
    identity: StudyIdentity,
) -> None:
    study = optuna.create_study(direction="maximize")

    def objective(trial: optuna.Trial) -> float:
        set_task_five_attrs(trial)
        return 0.5

    study.optimize(objective, n_trials=1)
    api = FakeWandb()
    session = WandbSession.open(
        WandbSettings(),
        identity,
        "abc123",
        dependencies=WandbDependencies(api, fake_official_factory(api)),
    )
    session.callback(study, study.trials[0])

    assert api.history[0].metrics["throughput_games_per_second"] == 8.0
    assert api.history[0].metrics["eta_seconds"] == 3.0 * 511.0


@pytest.mark.parametrize("runtime", (0.0, float("nan")))
def test_invalid_trial_wall_time_disables_telemetry_without_changing_study(
    identity: StudyIdentity,
    runtime: float,
) -> None:
    study = optuna.create_study(direction="maximize")

    def objective(trial: optuna.Trial) -> float:
        set_task_five_attrs(trial, trial_wall_seconds=runtime)
        return 0.5

    study.optimize(objective, n_trials=1)
    before = frozen_trial_payload(study.trials)
    api = FakeWandb()
    session = WandbSession.open(
        WandbSettings(),
        identity,
        "abc123",
        dependencies=WandbDependencies(api, fake_official_factory(api)),
    )
    session.callback(study, study.trials[0])

    assert api.history == []
    assert session.disabled_reason is not None
    assert frozen_trial_payload(study.trials) == before


def test_reopened_session_only_receives_new_terminal_trial(
    identity: StudyIdentity,
) -> None:
    study = optuna.create_study(direction="maximize")
    api = FakeWandb()
    dependencies = WandbDependencies(api, fake_official_factory(api))
    first = WandbSession.open(
        WandbSettings(), identity, "abc123", dependencies=dependencies
    )

    def objective(trial: optuna.Trial) -> float:
        set_task_five_attrs(trial)
        return float(trial.number)

    study.optimize(objective, n_trials=8, callbacks=first.callbacks)
    first.close()
    second = WandbSession.open(
        WandbSettings(), identity, "abc123", dependencies=dependencies
    )
    study.optimize(objective, n_trials=1, callbacks=second.callbacks)
    second.close()

    assert [row.step for row in api.history] == list(range(9))
    assert [call["id"] for call in api.init_calls] == [
        f"hybrid-optuna-{identity.digest[:20]}",
        f"hybrid-optuna-{identity.digest[:20]}",
    ]
