"""One-run, failure-isolated W&B telemetry contracts."""

# Test names are the behavioral documentation.
# ruff: noqa: D101, D102, D103

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

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


class FakeRun:
    def __init__(
        self,
        api: FakeWandb,
        *,
        fail_log: bool = False,
        fail_finish: bool = False,
    ) -> None:
        self.api = api
        self.fail_log = fail_log
        self.fail_finish = fail_finish

    def log(self, metrics: Mapping[str, float], *, step: int, commit: bool) -> None:
        if self.fail_log:
            raise RuntimeError("log unavailable")
        self.api.history.append(LoggedRow(dict(metrics), step, commit))

    def finish(self, exit_code: int = 0, quiet: bool = True) -> None:
        del exit_code, quiet
        self.api.finish_calls += 1
        if self.fail_finish:
            raise RuntimeError("finish unavailable")


class FakeWandb:
    def __init__(
        self,
        *,
        fail_init: bool = False,
        fail_log: bool = False,
        fail_finish: bool = False,
    ) -> None:
        self.fail_init = fail_init
        self.fail_log = fail_log
        self.fail_finish = fail_finish
        self.init_calls: list[dict[str, object]] = []
        self.history: list[LoggedRow] = []
        self.finish_calls = 0
        self.run: WandbRun | None = None

    def init(self, **kwargs: object) -> WandbRun:
        self.init_calls.append(kwargs)
        if self.fail_init:
            raise RuntimeError("init unavailable")
        self.run = FakeRun(self, fail_log=self.fail_log, fail_finish=self.fail_finish)
        return self.run


def fake_official_factory(
    api: FakeWandb, *, fail: bool = False
) -> Callable[..., object]:
    """Model the official callback committing the existing current run row."""

    def factory(
        *, wandb_kwargs: dict[str, object], as_multirun: bool
    ) -> Callable[[optuna.Study, FrozenTrial], None]:
        assert as_multirun is False
        api.init(**wandb_kwargs)

        def official(_study: optuna.Study, trial: FrozenTrial) -> None:
            if fail:
                raise RuntimeError("official callback unavailable")
            api.history[-1] = LoggedRow(api.history[-1].metrics, trial.number, True)

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
            "primary": 0.5,
            "dense_margin": 0.25,
            "objective": 0.50000025,
            "opponent/economic_policy/win_points": 0.125,
            "opponent/economic_policy/games": 64,
            "best_trial": 1,
            "best/economic/win_points": 0.125,
            "eta_seconds": 12.0,
            "throughput_games_per_second": 22.0,
        },
    )


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
    api = FakeWandb()
    session = WandbSession.open(
        WandbSettings(),
        identity,
        "abc123",
        dependencies=WandbDependencies(api, fake_official_factory(api)),
    )

    for trial in (terminal_trial(0), terminal_trial(1)):
        session.callbacks[0](study, cast(FrozenTrial, trial))
    session.close()

    assert len(api.init_calls) == 1
    assert api.init_calls[0]["id"] == f"hybrid-optuna-{identity.digest[:20]}"
    assert [row.step for row in api.history] == [0, 1]
    assert all(row.commit for row in api.history)
    assert api.history[1].metrics["best/economic/win_points"] == 0.125
    assert api.finish_calls == 1


@pytest.mark.parametrize("failure", ("import", "init", "log", "official", "finish"))
def test_wandb_failure_never_changes_study_or_raises(
    monkeypatch: pytest.MonkeyPatch,
    identity: StudyIdentity,
    study: optuna.Study,
    failure: str,
) -> None:
    study.optimize(lambda _trial: 1.0, n_trials=1)
    before = frozen_trial_payload(study.trials)
    if failure == "import":
        monkeypatch.setattr(
            "kaggriculture.search.optuna_wandb.load_wandb_dependencies",
            lambda: (_ for _ in ()).throw(ImportError("wandb missing")),
        )
        session = WandbSession.open(WandbSettings(), identity, "abc123")
    else:
        api = FakeWandb(
            fail_init=failure == "init",
            fail_log=failure == "log",
            fail_finish=failure == "finish",
        )
        session = WandbSession.open(
            WandbSettings(),
            identity,
            "abc123",
            dependencies=WandbDependencies(
                api,
                fake_official_factory(api, fail=failure == "official"),
            ),
        )
    session.callback(study, cast(FrozenTrial, terminal_trial(0)))
    session.close()

    assert frozen_trial_payload(study.trials) == before
    assert session.disabled_reason is not None


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
        "primary",
        "dense_margin",
        "objective",
        "throughput_games_per_second",
        "eta_seconds",
    } <= metrics.keys()
    trial.user_attrs["eta_seconds"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        trial_metrics(trial)


def test_callback_includes_current_best_trial_scalars(identity: StudyIdentity) -> None:
    study = optuna.create_study(direction="maximize")

    def objective(trial: optuna.Trial) -> float:
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


def test_reopened_session_only_receives_new_terminal_trial(
    identity: StudyIdentity,
) -> None:
    study = optuna.create_study(direction="maximize")
    api = FakeWandb()
    dependencies = WandbDependencies(api, fake_official_factory(api))
    first = WandbSession.open(
        WandbSettings(), identity, "abc123", dependencies=dependencies
    )
    study.optimize(
        lambda trial: float(trial.number), n_trials=8, callbacks=first.callbacks
    )
    first.close()
    second = WandbSession.open(
        WandbSettings(), identity, "abc123", dependencies=dependencies
    )
    study.optimize(
        lambda trial: float(trial.number), n_trials=1, callbacks=second.callbacks
    )
    second.close()

    assert [row.step for row in api.history] == list(range(9))
    assert [call["id"] for call in api.init_calls] == [
        f"hybrid-optuna-{identity.digest[:20]}",
        f"hybrid-optuna-{identity.digest[:20]}",
    ]
