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
import kaggriculture.learn.toad.lightning as lightning_module
from kaggriculture.constants import BOARD_SIZE, ENVIRONMENT
from kaggriculture.learn import toad_loss
from kaggriculture.learn.encoding import TRANSFER_OPS
from kaggriculture.learn.scripts import toad
from kaggriculture.learn.toad.config import ModelConfig, TeacherSpec, ToadConfig
from kaggriculture.learn.toad.data import BatchKind, LearnerBatch, segments
from kaggriculture.learn.toad.lightning import (
    ToadLightningModule,
    compute_loss,
    round_decay,
)
from kaggriculture.learn.toad.model import PolicyOutput, PolicyState, StatefulPolicy
from kaggriculture.learn.toad.population import LoadedTeacher
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


def _loaded_teacher(
    policy: toad.Policy,
    config: ToadConfig,
    *,
    operation: bool = True,
    quantity: bool = False,
    market: bool = True,
    value: bool = False,
) -> LoadedTeacher:
    """Pair a real frozen policy with an explicit in-memory test contract."""
    policy.requires_grad_(False).eval()
    heads = tuple(
        name
        for name, declared in (
            ("operation", operation),
            ("quantity", quantity),
            ("market", market),
            ("value", value),
        )
        if declared
    )
    return LoadedTeacher(
        policy=policy,
        spec=TeacherSpec(
            checkpoint=Path("teacher.pt"),
            sha256="0" * 64,
            blocks=config.model.blocks,
            operation=operation,
            quantity=quantity,
            market=market,
            value=value,
        ),
        model=config.model,
        actual_heads=heads,
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
        unit_actions[:, -1] = TRANSFER_OPS.index(True)
        unit_valid = segment["unit_valid"].clone()
        unit_valid[:, -1] = False
        unit_quantities = segment["unit_quantities"].clone()
        unit_quantities[:, -1] = 0
        padded_segments.append(
            {
                **segment,
                "unit_masks": unit_masks,
                "unit_quantity_masks": quantity_masks,
                "unit_actions": unit_actions,
                "unit_valid": unit_valid,
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


def test_module_distinguishes_exact_and_control_migration_warm_starts(
    tmp_path: Path,
) -> None:
    """Enabled initialization may migrate control weights without weakening resume."""
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
    assert restored.warm_start_migration is not None
    assert restored.warm_start_migration.mode == "exact_stateful"

    bare_path = tmp_path / "bare.pt"
    bare_state = toad.Policy(blocks=1, channels=16).state_dict()
    torch.save(bare_state, bare_path)
    bare_config = base.model_copy(
        update={
            "curriculum": base.curriculum.model_copy(
                update={"warm_start_checkpoint": bare_path}
            )
        }
    )
    migrated = ToadLightningModule(bare_config)
    assert isinstance(migrated.policy, StatefulPolicy)
    assert migrated.warm_start_migration is not None
    assert migrated.warm_start_migration.mode == "control_migration"
    for name, tensor in bare_state.items():
        assert torch.equal(migrated.policy.control.state_dict()[name], tensor)


@pytest.mark.parametrize("layout", ["bare", "legacy_learner", "lightning_policy"])
def test_enabled_warm_start_migrates_every_control_checkpoint_layout(
    tmp_path: Path, layout: str
) -> None:
    """Bare, legacy learner, and foundation Lightning weights migrate explicitly."""
    config = _recurrent_config()
    torch.manual_seed(211)
    policy = StatefulPolicy(config.model)
    optional_before = {
        name: value.detach().clone()
        for name, value in policy.state_dict().items()
        if not name.startswith("control.")
    }
    torch.manual_seed(223)
    control = toad.Policy(blocks=1, channels=16, value_bound=toad.VALUE_BOUND)
    control_state = dict(control.state_dict())
    checkpoint: object
    if layout == "bare":
        checkpoint = control_state
    elif layout == "legacy_learner":
        checkpoint = {"learner": control_state}
    else:
        checkpoint = {
            "state_dict": {
                f"policy.{name}": value for name, value in control_state.items()
            }
        }
    path = tmp_path / f"{layout}.pt"
    torch.save(checkpoint, path)

    result = lightning_module.initialize_policy_from_checkpoint(policy, path)

    assert result.mode == "control_migration"
    assert result.source == layout
    assert set(result.fresh_keys) == set(optional_before)
    for name, value in control_state.items():
        assert torch.equal(policy.control.state_dict()[name], value), name
    for name, value in optional_before.items():
        assert torch.equal(policy.state_dict()[name], value), name


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
            record_states=[step in (0, 16)],
        )
        turn = decided[0]
        turns.append(turn)
        if turn.policy_state is not None:
            states.append(turn.policy_state)
        state = next_states[0]
    assert state is not None
    assert len(states) == 2
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
        hidden=torch.stack([record.hidden for record in states]),
        cell=torch.stack([record.cell for record in states]),
        prior_belief=torch.stack([record.prior_belief for record in states]),
        state_steps=torch.tensor([0, 16]),
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


def test_frozen_opponent_training_step_logs_all_population_kinds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A frozen-only fresh batch cannot index a two-kind metrics table."""
    module = ToadLightningModule(control_fixture_config())
    records: list[dict[str, object]] = []
    monkeypatch.setattr(module, "log_dict", lambda record: records.append(dict(record)))
    batch = control_fixture_batch(load_control_fixture())
    frozen = replace(
        batch,
        kind=BatchKind.FROZEN_OPPONENT,
        segment_kinds=(BatchKind.FROZEN_OPPONENT,) * len(batch.segments),
    )

    module.training_step(frozen, 0)

    assert len(records) == 1
    assert {
        "diag/population_selfplay_segments": records[0][
            "diag/population_selfplay_segments"
        ],
        "diag/population_scripted_segments": records[0][
            "diag/population_scripted_segments"
        ],
        "diag/population_frozen_opponent_segments": records[0][
            "diag/population_frozen_opponent_segments"
        ],
        "diag/population_teacher_distill_segments": records[0][
            "diag/population_teacher_distill_segments"
        ],
    } == {
        "diag/population_selfplay_segments": 0,
        "diag/population_scripted_segments": 0,
        "diag/population_frozen_opponent_segments": 4,
        "diag/population_teacher_distill_segments": 0,
    }


def test_teacher_distill_training_step_logs_all_population_kinds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A teacher-only fresh batch has the same stable four-kind metric schema."""
    teacher = toad.Policy(blocks=1, channels=16, value_bound=toad.VALUE_BOUND)
    checkpoint = tmp_path / "teacher.pt"
    torch.save(teacher.state_dict(), checkpoint)
    config = ToadConfig.model_validate(
        {
            "model": {"blocks": 1, "channels": 16},
            "population": {
                "selfplay": 0.0,
                "scripted": 0.0,
                "teacher_distill": 1.0,
                "teacher": {
                    "checkpoint": checkpoint,
                    "blocks": 1,
                    "operation": True,
                    "quantity": True,
                    "market": True,
                },
            },
            "optimizer": {"value_warmup_batches": 0, "teacher_kl_cost": 0.25},
        }
    )
    module = ToadLightningModule(config)
    records: list[dict[str, object]] = []
    monkeypatch.setattr(module, "log_dict", lambda record: records.append(dict(record)))
    batch = control_fixture_batch(load_control_fixture())
    distill = replace(
        batch,
        kind=BatchKind.TEACHER_DISTILL,
        segment_kinds=(BatchKind.TEACHER_DISTILL,) * len(batch.segments),
    )

    module.training_step(distill, 0)

    assert len(records) == 1
    assert records[0]["diag/population_selfplay_segments"] == 0
    assert records[0]["diag/population_scripted_segments"] == 0
    assert records[0]["diag/population_frozen_opponent_segments"] == 0
    assert records[0]["diag/population_teacher_distill_segments"] == 4


def test_stateful_student_computes_loss_against_a_control_teacher(
    tmp_path: Path,
) -> None:
    """Action regularization flattens public observations for the bare teacher."""
    checkpoint = tmp_path / "teacher.pt"
    torch.save(
        toad.Policy(
            blocks=1,
            channels=16,
            value_bound=toad.VALUE_BOUND,
        ).state_dict(),
        checkpoint,
    )
    config = ToadConfig.model_validate(
        {
            "model": {
                "blocks": 1,
                "channels": 16,
                "recurrent": True,
                "recurrent_channels": 3,
                "recurrent_layers": 2,
            },
            "population": {
                "teacher": {
                    "checkpoint": checkpoint,
                    "blocks": 1,
                    "quantity": True,
                }
            },
            "optimizer": {"value_warmup_batches": 0, "teacher_kl_cost": 0.25},
        }
    )
    module = ToadLightningModule(config)
    recurrent_segments = segments(_recurrent_trajectory(), 16)
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

    report = compute_loss(
        module.policy,
        batch,
        module.config,
        module.teacher,
    )

    assert module.teacher is not None
    assert module.teacher.model == ModelConfig.control(blocks=1, channels=16)
    assert torch.isfinite(report.total)
    assert report.terms["teacher/operation_kl"].item() > 0.0
    assert report.terms["teacher/quantity_kl"].item() > 0.0
    assert report.terms["teacher/market_kl"].item() > 0.0


def test_teacher_loss_reports_each_raw_and_weighted_declared_head() -> None:
    """A single aggregate cannot conceal a wrong head or coefficient."""
    fixture = load_control_fixture()
    base = control_fixture_config()
    config = base.model_copy(
        update={
            "optimizer": base.optimizer.model_copy(update={"teacher_kl_cost": 0.25})
        }
    )
    learner = toad.Policy(blocks=1, channels=16, value_bound=toad.VALUE_BOUND)
    learner.load_state_dict(cast(dict[str, torch.Tensor], fixture["initial_model"]))
    torch.manual_seed(17)
    teacher_policy = toad.Policy(blocks=1, channels=16, value_bound=toad.VALUE_BOUND)
    teacher = _loaded_teacher(
        teacher_policy,
        config,
        operation=True,
        quantity=False,
        market=True,
        value=False,
    )

    report = compute_loss(learner, control_fixture_batch(fixture), config, teacher)

    assert report.terms["teacher/operation_kl"].item() > 0.0
    assert report.terms["teacher/market_kl"].item() > 0.0
    assert report.terms["teacher/quantity_kl"].item() == 0.0
    assert report.terms["teacher/value"].item() == 0.0
    torch.testing.assert_close(
        report.terms["teacher/operation_kl_weighted"],
        report.terms["teacher/operation_kl"] * 0.25,
    )
    torch.testing.assert_close(
        report.terms["teacher/market_kl_weighted"],
        report.terms["teacher/market_kl"] * 0.25,
    )
    torch.testing.assert_close(
        report.terms["teacher"],
        report.terms["teacher/operation_kl_weighted"]
        + report.terms["teacher/market_kl_weighted"],
    )


def test_round_log_reports_each_raw_and_weighted_teacher_term() -> None:
    """Dashboard records must preserve the declared-head loss breakdown."""
    fixture = load_control_fixture()
    base = control_fixture_config()
    config = base.model_copy(
        update={
            "optimizer": base.optimizer.model_copy(
                update={"teacher_kl_cost": 0.25, "teacher_baseline_cost": 0.5}
            )
        }
    )
    learner = toad.Policy(blocks=1, channels=16, value_bound=toad.VALUE_BOUND)
    learner.load_state_dict(cast(dict[str, torch.Tensor], fixture["initial_model"]))
    teacher = _loaded_teacher(
        toad.Policy(blocks=1, channels=16, value_bound=toad.VALUE_BOUND),
        config,
        quantity=True,
        value=True,
    )
    report = compute_loss(learner, control_fixture_batch(fixture), config, teacher)
    module = ToadLightningModule(base)
    module._round_fresh_terms = [report.terms]
    module._round_baselines = [report.terms["baseline"]]

    record = module._round_log_record()

    for name in (
        "teacher/operation_kl",
        "teacher/operation_kl_weighted",
        "teacher/quantity_kl",
        "teacher/quantity_kl_weighted",
        "teacher/market_kl",
        "teacher/market_kl_weighted",
        "teacher/value",
        "teacher/value_weighted",
    ):
        torch.testing.assert_close(cast(torch.Tensor, record[name]), report.terms[name])


def test_round_log_reports_raw_head_entropy_statistics() -> None:
    """Round diagnostics expose sums, decision counts, and safe head means."""
    fixture = load_control_fixture()
    config = control_fixture_config()
    learner = toad.Policy(blocks=1, channels=16, value_bound=toad.VALUE_BOUND)
    learner.load_state_dict(cast(dict[str, torch.Tensor], fixture["initial_model"]))
    report = compute_loss(learner, control_fixture_batch(fixture), config)
    module = ToadLightningModule(config)
    module._round_fresh_terms = [report.terms]
    module._round_fresh_entropy = [report.entropy]
    module._round_baselines = [report.terms["baseline"]]

    record = module._round_log_record()

    for name, stat in report.entropy.items():
        torch.testing.assert_close(
            stat.mean, stat.sum / stat.valid.to(dtype=stat.sum.dtype).clamp_min(1.0)
        )
        torch.testing.assert_close(
            cast(torch.Tensor, record[f"entropy/{name}_sum"]), stat.sum
        )
        torch.testing.assert_close(
            cast(torch.Tensor, record[f"entropy/{name}_valid"]), stat.valid.float()
        )
        torch.testing.assert_close(
            cast(torch.Tensor, record[f"entropy/{name}_mean"]),
            stat.mean,
        )


def test_declared_quantity_and_value_alignment_have_independent_weights() -> None:
    """Policy KL and teacher baseline alignment use their own typed costs."""
    fixture = load_control_fixture()
    base = control_fixture_config()
    config = base.model_copy(
        update={
            "optimizer": base.optimizer.model_copy(
                update={"teacher_kl_cost": 0.25, "teacher_baseline_cost": 0.5}
            )
        }
    )
    learner = toad.Policy(blocks=1, channels=16, value_bound=toad.VALUE_BOUND)
    learner.load_state_dict(cast(dict[str, torch.Tensor], fixture["initial_model"]))
    teacher = _loaded_teacher(
        toad.Policy(blocks=1, channels=16, value_bound=toad.VALUE_BOUND),
        config,
        operation=False,
        quantity=True,
        market=False,
        value=True,
    )

    report = compute_loss(learner, control_fixture_batch(fixture), config, teacher)

    assert report.terms["teacher/quantity_kl"].item() > 0.0
    assert report.terms["teacher/value"].item() > 0.0
    assert report.terms["teacher/operation_kl"].item() == 0.0
    assert report.terms["teacher/market_kl"].item() == 0.0
    torch.testing.assert_close(
        report.terms["teacher/quantity_kl_weighted"],
        report.terms["teacher/quantity_kl"] * 0.25,
    )
    torch.testing.assert_close(
        report.terms["teacher/value_weighted"],
        report.terms["teacher/value"] * 0.5,
    )
    torch.testing.assert_close(
        report.total,
        report.terms["vtrace_pg"]
        + report.terms["upgo_pg"]
        + report.terms["baseline"]
        + report.terms["entropy"]
        + report.terms["teacher"]
        + report.terms["teacher/value_weighted"],
    )


def test_teacher_distill_retains_rl_losses_and_matches_normal_teacher_routing() -> None:
    """The batch kind adds teacher semantics without becoming teacher-only."""
    fixture = load_control_fixture()
    base = control_fixture_config()
    config = base.model_copy(
        update={
            "optimizer": base.optimizer.model_copy(update={"teacher_kl_cost": 0.25})
        }
    )
    learner = toad.Policy(blocks=1, channels=16, value_bound=toad.VALUE_BOUND)
    learner.load_state_dict(cast(dict[str, torch.Tensor], fixture["initial_model"]))
    teacher = _loaded_teacher(
        toad.Policy(blocks=1, channels=16, value_bound=toad.VALUE_BOUND),
        config,
        quantity=True,
    )
    normal_batch = control_fixture_batch(fixture)
    teacher_batch = replace(normal_batch, kind=BatchKind.TEACHER_DISTILL)

    normal = compute_loss(learner, normal_batch, config, teacher)
    distill = compute_loss(learner, teacher_batch, config, teacher)

    for name in (
        "vtrace_pg",
        "upgo_pg",
        "baseline",
        "entropy",
        "teacher/operation_kl",
        "teacher/quantity_kl",
        "teacher/market_kl",
        "teacher",
        "total",
    ):
        assert torch.equal(distill.terms[name], normal.terms[name]), name
    assert distill.terms["baseline"].item() != 0.0
    assert distill.terms["entropy"].item() != 0.0


def test_mixed_teacher_distill_loss_uses_segment_provenance() -> None:
    """Teacher-only routing in a mixed batch cannot reach unrelated segments."""
    fixture = load_control_fixture()
    base = control_fixture_config()
    config = base.model_copy(
        update={"optimizer": base.optimizer.model_copy(update={"teacher_kl_cost": 1.0})}
    )
    learner = toad.Policy(blocks=1, channels=16, value_bound=toad.VALUE_BOUND)
    learner.load_state_dict(cast(dict[str, torch.Tensor], fixture["initial_model"]))
    teacher = _loaded_teacher(
        toad.Policy(blocks=1, channels=16, value_bound=toad.VALUE_BOUND),
        config,
        quantity=True,
    )
    batch = control_fixture_batch(fixture)
    kinds = (BatchKind.TEACHER_DISTILL,) + (BatchKind.SELFPLAY,) * (
        len(batch.segments) - 1
    )
    mixed = replace(batch, kind=BatchKind.MIXED, segment_kinds=kinds)
    teacher_only = replace(
        batch,
        segments=(batch.segments[0],),
        kind=BatchKind.TEACHER_DISTILL,
        segment_kinds=(BatchKind.TEACHER_DISTILL,),
    )

    mixed_report = compute_loss(learner, mixed, config, teacher)
    teacher_report = compute_loss(learner, teacher_only, config, teacher)

    for name in (
        "teacher/operation_kl",
        "teacher/quantity_kl",
        "teacher/market_kl",
    ):
        torch.testing.assert_close(mixed_report.terms[name], teacher_report.terms[name])


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
                update={
                    "teacher": TeacherSpec(
                        checkpoint=checkpoint,
                        blocks=1,
                        quantity=False,
                    )
                }
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
    assert not module.teacher.spec.quantity
    assert module.teacher.spec.sha256 is not None
    assert module.config.population.teacher == module.teacher.spec
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
                update={
                    "teacher": TeacherSpec(
                        checkpoint=checkpoint,
                        blocks=1,
                        quantity=True,
                    )
                }
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
