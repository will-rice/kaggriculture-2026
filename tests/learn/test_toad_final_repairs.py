"""Final whole-foundation review regressions."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import lightning
import pytest
import torch
from pydantic import ValidationError
from torch.utils.data import DataLoader

from kaggriculture.learn.scripts import curriculum, toad
from kaggriculture.learn.toad.callbacks import ActorSyncCallback
from kaggriculture.learn.toad.config import ToadConfig
from kaggriculture.learn.toad.data import (
    BatchKind,
    CollectionAssignment,
    LearnerBatch,
    ReferenceRoundSource,
    RoundBatchExpander,
    RoundMeta,
    ToadDataModule,
)
from kaggriculture.learn.toad.lightning import ResumeConfigError, ToadLightningModule
from tests.learn.test_toad_control_fixture import (
    _assert_state_equal,
    control_fixture_batch,
    control_fixture_config,
    control_fixture_threads,
    load_control_fixture,
)
from tests.learn.test_toad_data import _trajectory
from tests.learn.test_toad_lightning import _BatchDataset


def _mixed_config() -> ToadConfig:
    base = control_fixture_config()
    return base.model_copy(
        update={
            "population": base.population.model_copy(
                update={
                    "selfplay": 0.5,
                    "scripted": 0.5,
                    "environments_per_rank": 2,
                }
            ),
            "optimizer": base.optimizer.model_copy(
                update={"batch_segments": 1, "unroll_length": 16, "value_passes": 1}
            ),
        }
    )


def test_scripted_assignment_resolves_the_real_economic_agent() -> None:
    """The control default must be executable, not the logical dashboard ID."""
    source = ReferenceRoundSource(_mixed_config(), assignments=())
    assignment = CollectionAssignment(0, 0, "economic", BatchKind.SCRIPTED)

    assert source._worker_input(assignment).versus == toad.OPPONENT


@pytest.mark.slow
def test_real_scripted_collection_smoke() -> None:
    """One default scripted game reaches the shipped economic policy."""
    config = _mixed_config()
    source = ReferenceRoundSource(config, assignments=())
    policy = toad.Policy(blocks=1, channels=16, value_bound=toad.VALUE_BOUND)
    source.publish_actor(policy.state_dict(), version=0)

    trajectories = source.collect_assignment(
        CollectionAssignment(0, 0, "economic", BatchKind.SCRIPTED)
    )

    assert len(trajectories) == 1
    assert trajectories[0].dones[-1]


def test_data_resume_preserves_the_published_actor_off_sync_and_warmup() -> None:
    """Resume restores the generating actor, not current learner weights."""
    base = _mixed_config()
    config = base.model_copy(
        update={
            "optimizer": base.optimizer.model_copy(update={"value_warmup_batches": 2})
        }
    )
    source = ReferenceRoundSource(config, assignments=())
    data = ToadDataModule(config, source)
    data.publish_actor({"weight": torch.tensor([1.0])}, version=3)
    state = data.state_dict()
    resumed_source = ReferenceRoundSource(config, assignments=())
    resumed = ToadDataModule(config, resumed_source)
    resumed.load_state_dict(state)
    learner = torch.nn.Linear(1, 1, bias=False)
    learner.weight.data.fill_(9.0)
    module = SimpleNamespace(
        policy=learner,
        actor_version=3,
        actor_source_global_step=0,
        collection_round=4,
        global_step=8,
        _round_started_warming=True,
    )
    trainer = SimpleNamespace(datamodule=resumed, global_step=8)
    callback = ActorSyncCallback(every_rounds=4)

    callback.on_fit_start(
        cast(lightning.Trainer, trainer), cast(lightning.LightningModule, module)
    )
    assert resumed_source.actor_version == 3
    assert resumed_source.actor_state["weight"].tolist() == [1.0]

    boundary = SimpleNamespace(end_of_round=True)
    callback.on_train_batch_end(
        cast(lightning.Trainer, trainer),
        cast(lightning.LightningModule, module),
        None,
        boundary,
        0,
    )
    assert resumed_source.actor_state["weight"].tolist() == [1.0]
    module._round_started_warming = False
    module.collection_round = 3
    callback.on_train_batch_end(
        cast(lightning.Trainer, trainer),
        cast(lightning.LightningModule, module),
        None,
        boundary,
        1,
    )
    assert resumed_source.actor_state["weight"].tolist() == [1.0]
    module.collection_round = 4
    callback.on_train_batch_end(
        cast(lightning.Trainer, trainer),
        cast(lightning.LightningModule, module),
        None,
        boundary,
        2,
    )
    assert resumed_source.actor_version == 4
    assert resumed_source.actor_state["weight"].tolist() == [[9.0]]


def test_mixed_round_keeps_legacy_fresh_then_global_replay_order() -> None:
    """Mirror segments precede econ, and every fresh batch precedes critic replay."""
    config = _mixed_config()
    scripted = _trajectory(32)
    scripted.shaped.fill_(2.0)
    selfplay = _trajectory(32)
    selfplay.shaped.fill_(1.0)
    assignments = (
        CollectionAssignment(0, 0, "economic", BatchKind.SCRIPTED),
        CollectionAssignment(1, 1, "self", BatchKind.SELFPLAY),
    )
    source = ReferenceRoundSource(
        config,
        assignments=assignments,
        collect_assignment=lambda item: (
            (selfplay,) if item.kind is BatchKind.SELFPLAY else (scripted,)
        ),
    )

    batches = list(source)

    fresh = [batch for batch in batches if not batch.baseline_only]
    replay = [batch for batch in batches if batch.baseline_only]
    assert [batch.baseline_only for batch in batches] == [False] * len(fresh) + [
        True
    ] * len(replay)
    assert [batch.kind for batch in fresh] == [
        BatchKind.SELFPLAY,
        BatchKind.SELFPLAY,
        BatchKind.SCRIPTED,
        BatchKind.SCRIPTED,
    ]
    assert [float(batch.segments[0]["shaped"][0]) for batch in fresh] == [
        1.0,
        1.0,
        2.0,
        2.0,
    ]


def test_mixed_full_round_matches_legacy_parameters_adam_and_lr(
    tmp_path: Path,
) -> None:
    """The ordered mixed stream is numerically identical at every trained state."""
    fixture = load_control_fixture()
    trajectory = fixture["trajectory"]
    assert isinstance(trajectory, toad.Trajectory)
    scripted = replace(
        trajectory,
        shaped_money=trajectory.shaped_money * 0.5,
    )
    config = _mixed_config()
    assignments = (
        CollectionAssignment(0, 0, "economic", BatchKind.SCRIPTED),
        CollectionAssignment(1, 1, "self", BatchKind.SELFPLAY),
    )
    source = ReferenceRoundSource(
        config,
        assignments=assignments,
        collect_assignment=lambda item: (
            (trajectory,) if item.kind is BatchKind.SELFPLAY else (scripted,)
        ),
    )
    actual_batches = list(source)
    expected_batches = list(
        RoundBatchExpander(
            batch_segments=config.optimizer.batch_segments,
            unroll_length=config.optimizer.unroll_length,
            value_passes=config.optimizer.value_passes,
            seed=config.runtime.seed,
        ).expand(
            [trajectory, scripted],
            RoundMeta(
                round_id=0,
                actor_version=0,
                game_ids=(1, 0),
                seeds=(1, 0),
                opponent_ids=("self", "economic"),
                kind=BatchKind.MIXED,
            ),
            trajectory_kinds=[BatchKind.SELFPLAY, BatchKind.SCRIPTED],
        )
    )

    def train(
        batches: Sequence[LearnerBatch], root: Path
    ) -> tuple[ToadLightningModule, lightning.Trainer]:
        module = ToadLightningModule(config)
        module.policy.load_state_dict(
            cast(dict[str, torch.Tensor], fixture["initial_model"])
        )
        trainer = lightning.Trainer(
            accelerator="cpu",
            devices=1,
            max_steps=len(batches),
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            default_root_dir=root,
            gradient_clip_val=config.optimizer.clip_grad_norm,
            gradient_clip_algorithm="norm",
        )
        loader = DataLoader(_BatchDataset(*batches), batch_size=None, num_workers=0)
        trainer.fit(module, train_dataloaders=loader)
        return module, trainer

    with control_fixture_threads():
        actual, actual_trainer = train(actual_batches, tmp_path / "actual")
        expected, expected_trainer = train(expected_batches, tmp_path / "expected")

    _assert_state_equal(actual.policy.state_dict(), expected.policy.state_dict())
    _assert_state_equal(
        actual_trainer.optimizers[0].state_dict(),
        expected_trainer.optimizers[0].state_dict(),
    )
    assert (
        actual_trainer.lr_scheduler_configs[0].scheduler.get_last_lr()
        == expected_trainer.lr_scheduler_configs[0].scheduler.get_last_lr()
    )


@pytest.mark.parametrize(
    "update, message",
    [
        (
            {"population": {"selfplay": 0.5, "scripted": 0.0, "frozen_opponent": 0.5}},
            "frozen_opponent",
        ),
        (
            {"population": {"selfplay": 0.5, "scripted": 0.0, "teacher_distill": 0.5}},
            "teacher_distill",
        ),
        ({"optimizer": {"vtrace_pg_cost": 0.5}}, "vtrace_pg_cost"),
        ({"optimizer": {"upgo_pg_cost": 0.5}}, "upgo_pg_cost"),
        ({"optimizer": {"baseline_cost": 0.5}}, "baseline_cost"),
        ({"optimizer": {"teacher_baseline_cost": 0.5}}, "teacher_baseline_cost"),
    ],
)
def test_foundation_rejects_accepted_but_unimplemented_controls(
    update: dict[str, object], message: str
) -> None:
    """Every accepted non-control setting must have a native consumer."""
    with pytest.raises(ValidationError, match=message):
        ToadConfig.model_validate(update)


def test_legacy_resume_requires_migration_before_runtime(tmp_path: Path) -> None:
    """A legacy runner envelope must never be handed to Lightning restore."""
    legacy = tmp_path / "old.pt"
    legacy.touch()

    with pytest.raises(ValidationError, match="migration"):
        toad._legacy_config(["--resume", str(legacy)])


def test_legacy_name_is_safe_and_isolates_output() -> None:
    """Named compatibility runs cannot collide in the shared default directory."""
    left = toad._legacy_config(["--name", "arm-a"])
    right = toad._legacy_config(["--name", "arm-b"])

    assert left.runtime.output_dir == Path("run/toad/arm-a")
    assert right.runtime.output_dir == Path("run/toad/arm-b")
    with pytest.raises(ValueError, match="safe run name"):
        toad._legacy_config(["--name", "../escape"])


def test_historical_config_does_not_recheck_initialization_files(
    tmp_path: Path,
) -> None:
    """Resume schema validation must not depend on moved historical inputs."""
    warm = tmp_path / "warm.pt"
    warm.touch()
    payload = ToadConfig(curriculum={"warm_start_checkpoint": warm}).model_dump(
        mode="json"
    )
    warm.unlink()
    from kaggriculture.learn.toad.config import validate_stored_config

    stored = validate_stored_config(payload)

    assert stored.curriculum.warm_start_checkpoint == warm
    with pytest.raises(ValidationError, match="not readable"):
        ToadConfig.model_validate(payload)


def test_resume_checks_teacher_topology_and_valid_heads(tmp_path: Path) -> None:
    """Teacher semantics are checkpoint structure, not a rediscovered side effect."""
    teacher = toad.Policy(blocks=1, channels=16, value_bound=toad.VALUE_BOUND)
    path = tmp_path / "teacher.pt"
    torch.save(teacher.state_dict(), path)
    base = control_fixture_config()
    config = base.model_copy(
        update={
            "population": base.population.model_copy(
                update={"teacher_checkpoint": path, "teacher_blocks": 1}
            ),
            "optimizer": base.optimizer.model_copy(update={"teacher_kl_cost": 1.0}),
        }
    )
    stored = ToadLightningModule(config)
    checkpoint: dict[str, object] = {}
    stored.on_save_checkpoint(checkpoint)
    teacher_meta = cast(
        dict[str, object], cast(dict[str, object], checkpoint["toad"])["teacher"]
    )
    teacher_meta["quantity"] = False
    restored = ToadLightningModule(config)

    with pytest.raises(ResumeConfigError, match="teacher metadata mismatch"):
        restored.on_load_checkpoint(checkpoint)


def test_final_lr_multiplier_controls_the_native_floor() -> None:
    """The accepted schedule floor must change the actual LambdaLR function."""
    base = control_fixture_config()
    config = base.model_copy(
        update={
            "optimizer": base.optimizer.model_copy(
                update={"final_lr_multiplier": 0.25}
            ),
            "runtime": base.runtime.model_copy(update={"total_environment_steps": 1}),
        }
    )
    from kaggriculture.learn.toad.lightning import round_decay

    assert round_decay(config)(10_000) == 0.25


def test_invalid_legacy_cli_has_no_logging_or_warning_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full validation precedes logging and compatibility warnings."""
    events: list[str] = []
    monkeypatch.setattr(logging, "basicConfig", lambda **_: events.append("logging"))
    monkeypatch.setattr(warnings, "warn", lambda *_, **__: events.append("warning"))

    with pytest.raises(ValueError, match="safe run name"):
        toad.main(["--name", "../escape"])

    assert events == []


def test_invalid_curriculum_has_no_logging_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Teacher/config resolution completes before curriculum logging setup."""
    events: list[str] = []
    monkeypatch.setattr(curriculum, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(toad, "RUNS", tmp_path)
    monkeypatch.setattr(logging, "basicConfig", lambda **_: events.append("logging"))

    with pytest.raises(FileNotFoundError):
        curriculum.main(["phase2"])

    assert events == []


def test_native_logs_one_stable_round_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loss, clocks, population, objective, critic, and actor lag log once per round."""
    module = ToadLightningModule(control_fixture_config())
    records: list[dict[str, object]] = []
    monkeypatch.setattr(
        module, "log_dict", lambda record, **_: records.append(dict(record))
    )
    batch = control_fixture_batch(load_control_fixture())
    first = replace(
        batch,
        end_of_round=False,
        round_metrics={"objective/win_rate_vs_econ": 0.5, "diag/n_econ_envs": 1},
    )
    last = replace(batch, first_of_round=False, collected_steps=0)

    module.training_step(first, 0)
    module.training_step(last, 1)

    assert len(records) == 1
    assert {
        "objective/win_rate_vs_econ",
        "diag/n_econ_envs",
        "critic/baseline_self_consistency",
        "diag/vtrace_pg",
        "diag/environment_steps",
        "diag/optimizer_steps",
        "diag/actor_version",
        "diag/actor_lag_optimizer_steps",
    } <= records[0].keys()


def test_checkpoint_discovery_ignores_malformed_native_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unrelated step-like files cannot crash numeric latest selection."""
    monkeypatch.setattr(curriculum, "OUTPUT_ROOT", tmp_path)
    output = tmp_path / "phase1"
    output.mkdir()
    (output / "step-final.ckpt").touch()
    valid = output / "step-12.ckpt"
    valid.touch()

    assert curriculum._checkpoint("phase1") == valid
