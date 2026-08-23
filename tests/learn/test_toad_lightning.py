"""Numerical parity tests for the native Toad Lightning learner."""

from dataclasses import replace
from pathlib import Path
from typing import cast

import lightning
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from kaggriculture.learn.scripts import toad
from kaggriculture.learn.toad.data import LearnerBatch
from kaggriculture.learn.toad.lightning import ToadLightningModule, compute_loss
from tests.learn.test_toad_control_fixture import (
    _assert_state_equal,
    control_fixture_batch,
    control_fixture_config,
    control_fixture_threads,
    load_control_fixture,
)


class _BatchDataset(Dataset[LearnerBatch]):
    """Finite typed dataset for exercising Lightning's real data-transfer path."""

    def __init__(self, *batches: LearnerBatch) -> None:
        self.batches = batches

    def __getitem__(self, index: int) -> LearnerBatch:
        return self.batches[index]

    def __len__(self) -> int:
        return len(self.batches)


def _one_batch_loader(fixture: dict[str, object]) -> DataLoader[LearnerBatch]:
    return DataLoader(
        _BatchDataset(control_fixture_batch(fixture)),
        batch_size=None,
        num_workers=0,
    )


def test_lightning_training_step_matches_control_fixture(tmp_path: Path) -> None:
    """Automatic optimization must reproduce the frozen Adam step exactly."""
    fixture = load_control_fixture()
    module = ToadLightningModule(control_fixture_config())
    module.policy.load_state_dict(
        cast(dict[str, torch.Tensor], fixture["initial_model"])
    )
    trainer = lightning.Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        max_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
        gradient_clip_val=toad.CLIP_GRADS,
        gradient_clip_algorithm="norm",
    )

    with control_fixture_threads():
        trainer.fit(module, train_dataloaders=_one_batch_loader(fixture))

        updated_model = cast(dict[str, torch.Tensor], fixture["updated_model"])
        for name, value in module.policy.state_dict().items():
            assert torch.equal(value, updated_model[name])
        actual_optimizer = trainer.optimizers[0].state_dict()
        expected_optimizer = cast(dict[str, object], fixture["updated_optimizer"])
        _assert_state_equal(actual_optimizer["state"], expected_optimizer["state"])
        scheduler = trainer.lr_scheduler_configs[0].scheduler
        _assert_state_equal(scheduler.state_dict(), fixture["scheduler_state"])
        assert scheduler.get_last_lr() == fixture["scheduler_last_lr"]


def test_compute_loss_honors_baseline_only_and_optional_teacher() -> None:
    """Warmup must isolate the value loss while the declared teacher path works."""
    fixture = load_control_fixture()
    config = control_fixture_config()
    learner = toad.Policy(blocks=1, channels=16, value_bound=toad.VALUE_BOUND)
    learner.load_state_dict(cast(dict[str, torch.Tensor], fixture["initial_model"]))
    teacher_policy = toad.Policy(
        blocks=1, channels=16, value_bound=toad.VALUE_BOUND
    ).eval()
    teacher_config = config.model_copy(
        update={
            "optimizer": config.optimizer.model_copy(update={"teacher_kl_cost": 1.0})
        }
    )

    report = compute_loss(
        learner,
        control_fixture_batch(fixture),
        teacher_config,
        teacher=toad.Teacher(policy=teacher_policy, quantity=True),
        baseline_only=True,
    )

    assert torch.equal(report.total, report.terms["baseline"])
    assert report.terms["teacher"].item() > 0.0


def test_module_loads_warm_start_before_training(tmp_path: Path) -> None:
    """The typed warm-start field initializes policy weights without resume state."""
    source = toad.Policy(blocks=1, channels=16, value_bound=toad.VALUE_BOUND)
    checkpoint = tmp_path / "warm.pt"
    torch.save(source.state_dict(), checkpoint)
    config = control_fixture_config().model_copy(
        update={
            "model": control_fixture_config().model.model_copy(
                update={"warm_start_checkpoint": checkpoint}
            )
        }
    )

    module = ToadLightningModule(config)

    for name, value in source.state_dict().items():
        assert torch.equal(module.policy.state_dict()[name], value), name


def test_module_loads_frozen_typed_teacher_and_uses_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Teacher topology and missing-head compatibility reach the native loss."""
    fixture = load_control_fixture()
    teacher_policy = toad.Policy(blocks=1, channels=16, value_bound=toad.VALUE_BOUND)
    teacher_state = {
        name: value
        for name, value in teacher_policy.state_dict().items()
        if not name.startswith("quantity_head.")
    }
    checkpoint = tmp_path / "teacher.pt"
    torch.save(teacher_state, checkpoint)
    base = control_fixture_config()
    config = base.model_copy(
        update={
            "population": base.population.model_copy(
                update={"teacher_checkpoint": checkpoint, "teacher_blocks": 1}
            ),
            "optimizer": base.optimizer.model_copy(update={"teacher_kl_cost": 1.0}),
        }
    )
    module = ToadLightningModule(config)
    module.train()
    monkeypatch.setattr(module, "log_dict", lambda *args, **kwargs: None)
    module.policy.load_state_dict(
        cast(dict[str, torch.Tensor], fixture["initial_model"])
    )
    batch = control_fixture_batch(fixture)

    with_teacher = module.training_step(batch, 0)
    without_teacher = compute_loss(module.policy, batch, config).total

    assert module.teacher is not None
    assert len(module.teacher.policy.blocks) == 1
    assert not module.teacher.quantity
    assert not module.teacher.policy.training
    assert not any(
        parameter.requires_grad for parameter in module.teacher.policy.parameters()
    )
    assert not torch.equal(with_teacher, without_teacher)


def test_scheduler_steps_only_after_the_batch_that_closes_a_round(
    tmp_path: Path,
) -> None:
    """A value replay adds an Adam step but never a second LR or round step."""
    fixture = load_control_fixture()
    config = control_fixture_config()
    module = ToadLightningModule(config)
    module.policy.load_state_dict(
        cast(dict[str, torch.Tensor], fixture["initial_model"])
    )
    batch = control_fixture_batch(fixture)
    policy_batch = replace(batch, end_of_round=False)
    replay_batch = replace(
        batch,
        baseline_only=True,
        first_of_round=False,
        collected_steps=0,
    )
    loader = DataLoader(
        _BatchDataset(policy_batch, replay_batch), batch_size=None, num_workers=0
    )
    trainer = lightning.Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        max_steps=2,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
        gradient_clip_val=config.optimizer.clip_grad_norm,
        gradient_clip_algorithm="norm",
    )

    trainer.fit(module, train_dataloaders=loader)

    scheduler = trainer.lr_scheduler_configs[0].scheduler
    assert trainer.global_step == 2
    assert module.environment_steps == fixture["collected_steps"]
    assert module.collection_round == fixture["collection_rounds"]
    assert scheduler.last_epoch == fixture["scheduler_steps"]
    assert scheduler.get_last_lr() == fixture["scheduler_last_lr"]


def test_cpu_fast_dev_run_uses_automatic_optimization(tmp_path: Path) -> None:
    """The native module must complete Lightning's one-batch CPU smoke path."""
    fixture = load_control_fixture()
    config = control_fixture_config()
    module = ToadLightningModule(config)
    trainer = lightning.Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
        gradient_clip_val=config.optimizer.clip_grad_norm,
        gradient_clip_algorithm="norm",
    )

    trainer.fit(module, train_dataloaders=_one_batch_loader(fixture))

    assert module.automatic_optimization
    assert trainer.global_step == 1
    assert module.environment_steps == fixture["collected_steps"]
    assert module.collection_round == 1
