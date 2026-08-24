"""Numerical parity tests for the native Toad Lightning learner."""

from dataclasses import replace
from pathlib import Path
from typing import cast

import lightning
import pytest
import torch
from kaggle_environments import make
from torch.utils.data import DataLoader, Dataset

import kaggriculture.learn.rollout as rollout_module
from kaggriculture.constants import BOARD_SIZE, ENVIRONMENT
from kaggriculture.learn import toad_loss
from kaggriculture.learn.scripts import toad
from kaggriculture.learn.toad.config import ToadConfig
from kaggriculture.learn.toad.data import BatchKind, LearnerBatch, segments
from kaggriculture.learn.toad.lightning import (
    ToadLightningModule,
    compute_loss,
    round_decay,
)
from kaggriculture.learn.toad.model import PolicyOutput, PolicyState, StatefulPolicy
from tests.learn.test_toad_control_fixture import (
    _assert_state_equal,
    control_fixture_batch,
    control_fixture_config,
    control_fixture_threads,
    load_control_fixture,
)
from tests.learn.test_toad_data import _recurrent_trajectory


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


def _recurrent_config() -> ToadConfig:
    return toad.ToadConfig.model_validate(
        {
            "model": {
                "blocks": 1,
                "channels": 16,
                "recurrent": True,
                "recurrent_channels": 3,
                "recurrent_layers": 2,
            },
            "optimizer": {"value_warmup_batches": 0},
        }
    )


def test_recurrent_module_uses_stateful_policy_while_control_keeps_bare_policy() -> (
    None
):
    """Changing control to a wrapper would migrate foundation checkpoint keys."""
    recurrent = ToadLightningModule(_recurrent_config())
    control = ToadLightningModule(control_fixture_config())

    assert isinstance(recurrent.policy, StatefulPolicy)
    assert isinstance(control.policy, toad.Policy)
    assert not isinstance(control.policy, StatefulPolicy)


@pytest.mark.parametrize(
    "model",
    [
        {"transformer": True, "transformer_blocks": 1},
        {"local_patch": True, "local_patch_blocks": 1},
        {"interaction_value": True},
    ],
)
def test_optional_module_uses_exact_stateful_topology(
    model: dict[str, object], tmp_path: Path
) -> None:
    """Optional production learners must not fall back to bare Policy."""
    payload = {"blocks": 1, "channels": 16, **model}
    config = ToadConfig.model_validate({"model": payload})
    source = StatefulPolicy(config.model)
    path = tmp_path / "stateful.pt"
    torch.save(source.state_dict(), path)
    restored_config = config.model_copy(
        update={
            "curriculum": config.curriculum.model_copy(
                update={"warm_start_checkpoint": path}
            )
        }
    )

    restored = ToadLightningModule(restored_config)

    assert isinstance(restored.policy, StatefulPolicy)
    for name, tensor in source.state_dict().items():
        assert torch.equal(restored.policy.state_dict()[name], tensor)


def test_compute_loss_replays_segment_state_and_shifts_action_dones() -> None:
    """A terminal action resets the following observation, not its own logits."""
    config = _recurrent_config()
    policy = StatefulPolicy(config.model)
    trajectory = _recurrent_trajectory()
    assert trajectory.hidden is not None
    recurrent_segments = segments(trajectory, 16)
    recurrent_segments[0]["dones"][-1] = True
    captured: dict[str, object] = {}

    class SpyPolicy(StatefulPolicy):
        def forward(
            self,
            board: torch.Tensor,
            scalars: torch.Tensor,
            positions: torch.Tensor,
            state: PolicyState | None = None,
            dones: torch.Tensor | None = None,
        ) -> PolicyOutput:
            captured["state"] = state
            captured["dones"] = dones
            return super().forward(board, scalars, positions, state, dones)

    spy = SpyPolicy(config.model)
    spy.load_state_dict(policy.state_dict())
    batch = LearnerBatch(
        segments=(recurrent_segments[0], recurrent_segments[1]),
        kind=BatchKind.SELFPLAY,
        baseline_only=False,
        first_of_round=True,
        end_of_round=True,
        collected_steps=32,
        round_id=0,
        actor_version=0,
        game_ids=(0,),
        opponent_ids=("self",),
    )

    report = compute_loss(spy, batch, config)

    initial = captured["state"]
    replay_dones = captured["dones"]
    assert isinstance(initial, PolicyState)
    assert isinstance(replay_dones, torch.Tensor)
    assert torch.equal(initial.hidden[:, 0], trajectory.hidden[0])
    assert torch.equal(initial.hidden[:, 1], trajectory.hidden[16])
    assert torch.equal(replay_dones[0], torch.tensor([False, False]))
    assert torch.equal(replay_dones[15], torch.tensor([False, False]))
    assert torch.equal(replay_dones[16], torch.tensor([True, True]))
    assert torch.isfinite(report.total)


def test_recurrent_compute_loss_rejects_control_segments_without_state() -> None:
    """Active recurrence cannot silently replay a segment from zero memory."""
    config = _recurrent_config()
    policy = StatefulPolicy(config.model)
    batch = replace(
        control_fixture_batch(load_control_fixture()),
        segments=(segments(_recurrent_trajectory(), 16)[0],),
    )
    control_segment = {
        name: tensor
        for name, tensor in batch.segments[0].items()
        if not name.startswith("initial_")
    }

    with pytest.raises(ValueError, match="require initial_hidden"):
        compute_loss(policy, replace(batch, segments=(control_segment,)), config)


def test_padded_local_slots_cannot_change_loss_or_parameter_gradients() -> None:
    """Placeholder positions and logits must be invisible to policy and entropy."""
    fixture = load_control_fixture()
    control_config = control_fixture_config()
    config = control_config.model_copy(
        update={
            "model": control_config.model.model_copy(
                update={"local_patch": True, "local_patch_blocks": 1}
            )
        }
    )
    baseline = StatefulPolicy(config.model)

    class PerturbedPaddedPolicy(StatefulPolicy):
        padded_units: torch.Tensor
        padded_quantities: torch.Tensor

        def forward(
            self,
            board: torch.Tensor,
            scalars: torch.Tensor,
            positions: torch.Tensor,
            state: PolicyState | None = None,
            dones: torch.Tensor | None = None,
        ) -> PolicyOutput:
            output = super().forward(board, scalars, positions, state, dones)
            unit_delta = torch.zeros_like(output.unit_logits)
            unit_delta[..., -1, :] = torch.linspace(
                -500.0,
                500.0,
                output.unit_logits.shape[-1],
                device=output.unit_logits.device,
            )
            quantity_delta = torch.zeros_like(output.quantity_logits)
            quantity_delta[..., -1, :] = torch.linspace(
                700.0,
                -700.0,
                output.quantity_logits.shape[-1],
                device=output.quantity_logits.device,
            )
            self.padded_units = output.unit_logits + unit_delta
            self.padded_quantities = output.quantity_logits + quantity_delta
            self.padded_units.retain_grad()
            self.padded_quantities.retain_grad()
            return replace(
                output,
                unit_logits=self.padded_units,
                quantity_logits=self.padded_quantities,
            )

    perturbed = PerturbedPaddedPolicy(config.model)
    perturbed.load_state_dict(baseline.state_dict())
    batch = control_fixture_batch(fixture)
    padded_segments = []
    for segment in batch.segments:
        unit_masks = segment["unit_masks"].clone()
        unit_masks[:, -1] = False
        unit_masks[:, -1, 0] = True
        quantity_masks = segment["unit_quantity_masks"].clone()
        quantity_masks[:, -1] = False
        quantity_masks[:, -1, 0] = True
        unit_actions = segment["unit_actions"].clone()
        unit_actions[:, -1] = 0
        unit_quantities = segment["unit_quantities"].clone()
        unit_quantities[:, -1] = 0
        padded_segments.append(
            {
                **segment,
                "unit_masks": unit_masks,
                "unit_quantity_masks": quantity_masks,
                "unit_actions": unit_actions,
                "unit_quantities": unit_quantities,
            }
        )
    batch = replace(batch, segments=tuple(padded_segments))
    assert all(
        torch.equal(
            segment["unit_masks"][:, -1].sum(dim=-1),
            torch.ones(segment["unit_masks"].shape[0], dtype=torch.int64),
        )
        for segment in batch.segments
    )
    changed_segments = []
    for segment in batch.segments:
        positions = segment["positions"].clone()
        positions[:, -1] = BOARD_SIZE * BOARD_SIZE - 1
        changed_segments.append({**segment, "positions": positions})

    baseline_report = compute_loss(baseline, batch, config)
    baseline_report.total.backward()
    changed_report = compute_loss(
        perturbed,
        replace(batch, segments=tuple(changed_segments)),
        config,
    )
    changed_report.total.backward()

    torch.testing.assert_close(changed_report.total, baseline_report.total)
    assert perturbed.padded_units.grad is not None
    assert perturbed.padded_quantities.grad is not None
    assert not perturbed.padded_units.grad[..., -1, :].any()
    assert not perturbed.padded_quantities.grad[..., -1, :].any()
    for (baseline_name, baseline_parameter), (
        changed_name,
        changed_parameter,
    ) in zip(baseline.named_parameters(), perturbed.named_parameters(), strict=True):
        assert baseline_name == changed_name
        if baseline_parameter.grad is None or changed_parameter.grad is None:
            assert baseline_parameter.grad is None
            assert changed_parameter.grad is None
        else:
            torch.testing.assert_close(
                changed_parameter.grad, baseline_parameter.grad, atol=1e-6, rtol=1e-6
            )


def test_transfer_batch_moves_recurrent_initial_state_with_segment_tensors() -> None:
    """Lightning device transfer cannot strand recurrent state on actor CPU."""
    config = _recurrent_config()
    module = ToadLightningModule(config)
    batch = LearnerBatch(
        segments=(segments(_recurrent_trajectory(), 16)[0],),
        kind=BatchKind.SELFPLAY,
        baseline_only=False,
        first_of_round=True,
        end_of_round=True,
        collected_steps=16,
        round_id=0,
        actor_version=0,
        game_ids=(0,),
        opponent_ids=("self",),
    )

    transferred = module.transfer_batch_to_device(batch, torch.device("meta"), 0)

    assert all(
        tensor.device.type == "meta" for tensor in transferred.segments[0].values()
    )


def test_module_loads_only_exact_recurrent_warm_start_weights(tmp_path: Path) -> None:
    """Enabled warm starts load every stateful key and reject bare checkpoints."""
    base = _recurrent_config()
    source = StatefulPolicy(base.model)
    exact_path = tmp_path / "recurrent.pt"
    torch.save(source.state_dict(), exact_path)
    exact_config = base.model_copy(
        update={
            "curriculum": base.curriculum.model_copy(
                update={"warm_start_checkpoint": exact_path}
            )
        }
    )

    restored = ToadLightningModule(exact_config)

    for name, tensor in source.state_dict().items():
        assert torch.equal(restored.policy.state_dict()[name], tensor)

    bare_path = tmp_path / "bare.pt"
    torch.save(toad.Policy(blocks=1, channels=16).state_dict(), bare_path)
    bare_config = base.model_copy(
        update={
            "curriculum": base.curriculum.model_copy(
                update={"warm_start_checkpoint": bare_path}
            )
        }
    )
    with pytest.raises(RuntimeError, match="Missing key.*control"):
        ToadLightningModule(bare_config)


def test_actor_and_learner_outputs_agree_across_a_reset_segment_boundary() -> None:
    """Segment 16 must reproduce actor values/log-probs after action 15 ended."""
    config = _recurrent_config()
    policy = StatefulPolicy(config.model).eval()
    environment = make(ENVIRONMENT, configuration={"episodeSteps": 3, "seed": 31})
    environment.reset(2)
    request = [(environment.state[0].observation, 0)]
    turns = []
    states = []
    state = None
    action_dones = torch.zeros(32, dtype=torch.bool)
    action_dones[15] = True
    action_dones[-1] = True
    for step in range(32):
        decided, next_states = rollout_module._decide(
            policy,
            request,
            torch.Generator().manual_seed(100 + step),
            states=[state],
            dones=[bool(action_dones[step - 1]) if step else False],
        )
        turn = decided[0]
        assert turn.policy_state is not None
        turns.append(turn)
        states.append(turn.policy_state)
        state = next_states[0]
    assert state is not None
    trajectory = replace(
        _recurrent_trajectory(),
        board=torch.cat([turn.board for turn in turns]),
        scalars=torch.cat([turn.scalars for turn in turns]),
        positions=torch.cat([turn.positions for turn in turns]),
        unit_actions=torch.cat([turn.units for turn in turns]),
        unit_quantities=torch.cat([turn.quantities for turn in turns]),
        market_actions=torch.cat([turn.market for turn in turns]),
        unit_masks=torch.cat([turn.unit_mask for turn in turns]),
        unit_quantity_masks=torch.cat([turn.quantity_mask for turn in turns]),
        market_masks=torch.cat([turn.market_mask for turn in turns]),
        log_probs=torch.cat([turn.log_prob for turn in turns]),
        values=torch.cat([turn.value for turn in turns]),
        dones=action_dones,
        hidden=torch.stack([record.hidden for record in states] + [state.hidden]),
        cell=torch.stack([record.cell for record in states] + [state.cell]),
        prior_belief=torch.stack(
            [record.prior_belief for record in states] + [state.prior_belief]
        ),
    )
    replay_segments = segments(trajectory, 16)
    batch = LearnerBatch(
        segments=tuple(replay_segments),
        kind=BatchKind.SELFPLAY,
        baseline_only=False,
        first_of_round=True,
        end_of_round=True,
        collected_steps=32,
        round_id=0,
        actor_version=0,
        game_ids=(0,),
        opponent_ids=("self",),
    )
    captured: dict[str, torch.Tensor] = {}

    def capture_loss(**kwargs: object) -> toad_loss.Losses:
        values = cast(torch.Tensor, kwargs["values"])
        learner = cast(torch.Tensor, kwargs["learner_log_probs"])
        captured.update(values=values.detach(), learner=learner.detach())
        zero = values.sum() * 0
        return toad_loss.Losses(zero, zero, zero, zero, zero, zero)

    compute_loss(policy, batch, config, _losses=capture_loss)

    expected_values = torch.stack((trajectory.values[:16], trajectory.values[16:]), 1)
    expected_log_probs = torch.stack(
        (trajectory.log_probs[:16], trajectory.log_probs[16:]), 1
    )
    assert torch.allclose(captured["values"], expected_values, atol=1e-6, rtol=1e-6)
    assert torch.allclose(captured["learner"], expected_log_probs, atol=1e-6, rtol=1e-6)


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
            "curriculum": control_fixture_config().curriculum.model_copy(
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


def test_module_loads_policy_from_lightning_checkpoint_envelopes(
    tmp_path: Path,
) -> None:
    """Native phase checkpoints initialize both warm starts and teachers."""
    source = toad.Policy(blocks=1, channels=16, value_bound=toad.VALUE_BOUND)
    checkpoint = tmp_path / "phase1.ckpt"
    torch.save(
        {
            "state_dict": {
                **{
                    f"policy.{name}": value
                    for name, value in source.state_dict().items()
                },
                "teacher_policy.ignored": torch.ones(1),
            }
        },
        checkpoint,
    )
    base = control_fixture_config()
    warm = base.model_copy(
        update={
            "curriculum": base.curriculum.model_copy(
                update={"warm_start_checkpoint": checkpoint}
            )
        }
    )
    teacher = base.model_copy(
        update={
            "population": base.population.model_copy(
                update={"teacher_checkpoint": checkpoint, "teacher_blocks": 1}
            ),
            "optimizer": base.optimizer.model_copy(update={"teacher_kl_cost": 1.0}),
        }
    )

    warm_module = ToadLightningModule(warm)
    teacher_module = ToadLightningModule(teacher)

    for name, value in source.state_dict().items():
        assert torch.equal(warm_module.policy.state_dict()[name], value), name
        assert teacher_module.teacher_policy is not None
        assert torch.equal(teacher_module.teacher_policy.state_dict()[name], value), (
            name
        )


def test_module_loads_policy_from_legacy_runner_envelope(tmp_path: Path) -> None:
    """The one-release migration path accepts the old learner envelope."""
    source = toad.Policy(blocks=1, channels=16, value_bound=toad.VALUE_BOUND)
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"learner": source.state_dict(), "steps": 1}, checkpoint)
    base = control_fixture_config()
    config = base.model_copy(
        update={
            "curriculum": base.curriculum.model_copy(
                update={"warm_start_checkpoint": checkpoint}
            )
        }
    )

    module = ToadLightningModule(config)

    for name, value in source.state_dict().items():
        assert torch.equal(module.policy.state_dict()[name], value), name


def test_round_decay_uses_typed_mixture_and_environment_quota(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scheduler round sizing must describe the games collection actually runs."""
    seen: list[tuple[float, int, int, float]] = []

    def fake_decay(
        fraction: float,
        total_steps: int,
        environments: int,
        min_lr_multiplier: float,
    ) -> object:
        seen.append((fraction, total_steps, environments, min_lr_multiplier))
        return lambda _: 1.0

    monkeypatch.setattr(toad, "_decay", fake_decay)
    base = control_fixture_config()
    config = base.model_copy(
        update={
            "population": base.population.model_copy(
                update={
                    "selfplay": 0.75,
                    "scripted": 0.25,
                    "environments_per_rank": 4,
                }
            )
        }
    )

    round_decay(config)

    assert seen == [
        (
            0.25,
            config.runtime.total_environment_steps,
            4,
            config.optimizer.final_lr_multiplier,
        )
    ]


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
