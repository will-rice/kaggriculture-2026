"""Lightning's automatic-optimization boundary for the native Toad learner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self, cast

import lightning
import torch
import torch.nn.functional as functional
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim.lr_scheduler import LRScheduler

from kaggriculture.learn import toad_loss
from kaggriculture.learn.encoding import (
    CROP_NAMES,
    PRODUCT_NAMES,
    SHED_NAMES,
    transfer_slots,
)
from kaggriculture.learn.model import Policy, load_policy_weights
from kaggriculture.learn.ppo import entropy_of, joint_log_prob
from kaggriculture.learn.toad.config import (
    STRUCTURAL_FIELDS,
    ToadConfig,
    structural_fingerprint,
    validate_stored_config,
)
from kaggriculture.learn.toad.data import BatchKind, LearnerBatch
from kaggriculture.learn.toad.model import (
    PolicyOutput,
    PolicyState,
    StatefulPolicy,
    uses_stateful_policy,
)

if TYPE_CHECKING:
    from kaggriculture.learn.scripts.toad import Teacher


@dataclass(frozen=True)
class LossReport:
    """Differentiable total and detached-by-caller diagnostic loss terms."""

    total: torch.Tensor
    terms: Mapping[str, torch.Tensor]


class ResumeConfigError(ValueError):
    """The effective config cannot safely consume the stored trainer state."""


PolicyLike = Policy | StatefulPolicy

CheckpointLayout = Literal["bare", "legacy_learner", "lightning_policy"]


@dataclass(frozen=True)
class WarmStartMigration:
    """Auditable result of the initialization-only checkpoint path."""

    mode: Literal["exact_stateful", "control_migration", "legacy_control"]
    source: CheckpointLayout
    fresh_keys: tuple[str, ...] = ()


def _belief_loss_terms(
    output: PolicyOutput,
    segments: tuple[dict[str, torch.Tensor], ...],
) -> dict[str, torch.Tensor]:
    """Return safely masked belief loss, family errors, and valid-row count."""
    required = {"belief_targets", "belief_valid"}
    if any(required.difference(segment) for segment in segments):
        raise ValueError(
            "belief learner batches require belief_targets and belief_valid"
        )
    if output.belief_logits is None:
        raise ValueError("belief-enabled policy did not emit belief logits")
    belief_targets = torch.stack(
        [segment["belief_targets"] for segment in segments], dim=1
    )
    valid_rows = torch.stack([segment["belief_valid"] for segment in segments], dim=1)
    if valid_rows.dtype is not torch.bool:
        raise ValueError("belief_valid must be boolean")
    acted_beliefs = output.belief_logits[:-1]
    if acted_beliefs.shape != belief_targets.shape:
        raise ValueError(
            "belief target shape does not match acted belief logits: "
            f"{tuple(belief_targets.shape)} != {tuple(acted_beliefs.shape)}"
        )
    target_mask = valid_rows.unsqueeze(-1)
    safe_targets = belief_targets.masked_fill(~target_mask, 0.0)
    error = functional.smooth_l1_loss(acted_beliefs, safe_targets, reduction="none")
    error = error.masked_fill(~target_mask, 0.0)
    valid_count = valid_rows.sum().to(dtype=error.dtype)

    def family_error(start: int, stop: int) -> torch.Tensor:
        family = error[..., start:stop]
        denominator = (valid_count * (stop - start)).clamp_min(1.0)
        return family.sum() / denominator

    shed_stop = len(SHED_NAMES)
    seed_stop = shed_stop + len(CROP_NAMES)
    carried_stop = seed_stop + len(PRODUCT_NAMES)
    return {
        "belief": family_error(0, carried_stop),
        "belief_shed": family_error(0, shed_stop),
        "belief_seeds": family_error(shed_stop, seed_stop),
        "belief_carried": family_error(seed_stop, carried_stop),
        "belief_valid": valid_count,
    }


def _checkpoint_policy_weights(
    path: Path,
) -> tuple[Mapping[str, torch.Tensor], CheckpointLayout]:
    """Extract policy tensors and retain their historical container layout."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("policy checkpoint must contain a mapping")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        if not isinstance(state_dict, Mapping):
            raise ValueError("Lightning checkpoint state_dict must be a mapping")
        weights = {
            name.removeprefix("policy."): value
            for name, value in state_dict.items()
            if isinstance(name, str) and name.startswith("policy.")
        }
        if not weights:
            raise ValueError("Lightning checkpoint has no policy.* weights")
        source: CheckpointLayout = "lightning_policy"
    elif "learner" in checkpoint:
        weights = checkpoint["learner"]
        if not isinstance(weights, Mapping):
            raise ValueError("legacy checkpoint learner must be a mapping")
        source = "legacy_learner"
    else:
        weights = checkpoint
        source = "bare"
    return cast(Mapping[str, torch.Tensor], weights), source


def load_checkpoint_policy(policy: PolicyLike, path: Path) -> list[str]:
    """Strictly load worker/resume policy weights from historical containers."""
    typed_weights, _source = _checkpoint_policy_weights(path)
    if isinstance(policy, StatefulPolicy):
        if uses_stateful_policy(policy.config):
            policy.load_state_dict(typed_weights, strict=True)
        else:
            policy.load_control_state_dict(typed_weights)
        return []
    return load_policy_weights(policy, typed_weights)


def initialize_policy_from_checkpoint(
    policy: PolicyLike, path: Path
) -> WarmStartMigration:
    """Initialize a policy, explicitly migrating control-only foundation weights.

    This is intentionally separate from ``load_checkpoint_policy``: actor,
    worker, and resume loads remain exact, while a user-declared warm start may
    seed the shared control trunk and leave optional modules at their deliberate
    constructor initialization.
    """
    weights, source = _checkpoint_policy_weights(path)
    if isinstance(policy, StatefulPolicy) and uses_stateful_policy(policy.config):
        if set(weights) == set(policy.state_dict()):
            policy.load_state_dict(weights, strict=True)
            return WarmStartMigration("exact_stateful", source)
        keys = tuple(weights)
        prefixed = [name.startswith("control.") for name in keys]
        if any(prefixed) and not all(prefixed):
            raise ValueError("warm-start checkpoint mixes bare and control key layouts")
        control_weights = (
            {name.removeprefix("control."): value for name, value in weights.items()}
            if prefixed
            else dict(weights)
        )
        policy.control.load_state_dict(control_weights, strict=True)
        fresh = tuple(
            sorted(
                name for name in policy.state_dict() if not name.startswith("control.")
            )
        )
        return WarmStartMigration("control_migration", source, fresh)
    if isinstance(policy, StatefulPolicy):
        policy.load_control_state_dict(weights)
    else:
        load_policy_weights(policy, weights)
    return WarmStartMigration("legacy_control", source)


def read_path(config: ToadConfig, path: str) -> object:
    """Read one declared dotted configuration path."""
    value: object = config
    for part in path.split("."):
        value = getattr(value, part)
    return value


def assert_resume_compatible(effective: ToadConfig, stored: ToadConfig) -> None:
    """Reject only declared structural differences on resume."""
    differences = {
        path: (read_path(stored, path), read_path(effective, path))
        for path in STRUCTURAL_FIELDS
        if read_path(stored, path) != read_path(effective, path)
    }
    if differences:
        rendered = ", ".join(
            f"{path}: stored={before!r}, effective={after!r}"
            for path, (before, after) in differences.items()
        )
        raise ResumeConfigError(f"structural config mismatch: {rendered}")


def compute_loss(
    policy: PolicyLike,
    batch: LearnerBatch,
    config: ToadConfig,
    teacher: Teacher | None = None,
    *,
    baseline_only: bool | None = None,
    _losses: Callable[..., toad_loss.Losses] = toad_loss.losses,
) -> LossReport:
    """Return Toad's loss tensors without mutating optimizer or gradients."""
    # Imported lazily so the legacy runner can delegate here without creating a
    # module-import cycle. These are the runner's existing, numerically pinned
    # layout and teacher-divergence helpers.
    from kaggriculture.learn.scripts.toad import _acted, _kl

    segments = batch.segments

    def stacked(name: str) -> torch.Tensor:
        return torch.stack([segment[name] for segment in segments], dim=1)

    board = stacked("board")
    scalars = stacked("scalars")
    positions = stacked("positions")
    unit_actions = stacked("unit_actions")
    unit_quantity_actions = stacked("unit_quantities")
    market_actions = stacked("market_actions")
    unit_masks = stacked("unit_masks")
    unit_quantity_masks = stacked("unit_quantity_masks")
    market_masks = stacked("market_masks")
    behaviour = stacked("log_probs")
    rewards = stacked(config.curriculum.reward_field)
    dones = stacked("dones")

    turns, width = behaviour.shape
    if isinstance(policy, StatefulPolicy):
        if policy.config.recurrent or policy.config.belief:
            required = {"initial_hidden", "initial_cell", "initial_belief"}
            missing = required.difference(segments[0])
            if missing or any(required.difference(segment) for segment in segments):
                raise ValueError(
                    "stateful learner batches require initial_hidden, "
                    "initial_cell, and initial_belief"
                )
            hidden_rows = [segment["initial_hidden"] for segment in segments]
            cell_rows = [segment["initial_cell"] for segment in segments]
            layer_dimension = 1 if hidden_rows[0].ndim == 4 else 0
            initial = PolicyState(
                hidden=torch.stack(hidden_rows, dim=layer_dimension),
                cell=torch.stack(cell_rows, dim=layer_dimension),
                prior_belief=torch.stack(
                    [segment["initial_belief"] for segment in segments]
                ),
            )
            replay_dones = torch.cat((torch.zeros_like(dones[:1]), dones), dim=0)
        else:
            initial = None
            replay_dones = None
        output = policy(
            board,
            scalars,
            positions,
            state=initial,
            dones=replay_dones,
        )
        unit_logits = output.unit_logits
        quantity_logits = output.quantity_logits
        market_logits = output.market_logits
        values = output.values
    else:
        legacy_output = policy(
            board.flatten(0, 1), scalars.flatten(0, 1), positions.flatten(0, 1)
        )
        unit_logits, quantity_logits, market_logits, values = legacy_output
    values = values.view(turns + 1, width)
    bootstrap_value = values[-1].detach()
    values = values[:-1]
    if isinstance(policy, StatefulPolicy):
        unit_logits = unit_logits[:-1].flatten(0, 1)
        quantity_logits = quantity_logits[:-1].flatten(0, 1)
        market_logits = market_logits[:-1].flatten(0, 1)
    else:
        unit_logits = _acted(unit_logits, turns, width)
        quantity_logits = _acted(quantity_logits, turns, width)
        market_logits = _acted(market_logits, turns, width)

    belief_enabled = isinstance(policy, StatefulPolicy) and policy.config.belief
    belief_terms: dict[str, torch.Tensor] = {}
    if belief_enabled:
        belief_terms = _belief_loss_terms(output, segments)
    flat_unit_actions = unit_actions.flatten(0, 1)
    flat_unit_masks = unit_masks.flatten(0, 1)
    flat_quantity_masks = unit_quantity_masks.flatten(0, 1)
    flat_market_masks = market_masks.flatten(0, 1)
    units = torch.log_softmax(
        unit_logits.masked_fill(~flat_unit_masks, -torch.inf), dim=-1
    )
    quantities = torch.log_softmax(
        quantity_logits.masked_fill(~flat_quantity_masks, -torch.inf), dim=-1
    )
    market = torch.log_softmax(
        market_logits.masked_fill(~flat_market_masks, -torch.inf), dim=-1
    )
    transferred = transfer_slots(flat_unit_actions)
    learner_log_probs = joint_log_prob(
        units,
        quantities,
        market,
        flat_unit_actions,
        unit_quantity_actions.flatten(0, 1),
        market_actions.flatten(0, 1),
    ).view(turns, width)
    negative_entropy = -(
        entropy_of(units, flat_unit_masks).sum(dim=-1)
        + entropy_of(quantities, flat_quantity_masks)
        .masked_fill(~transferred, 0.0)
        .sum(dim=-1)
        + entropy_of(market, flat_market_masks).sum(dim=-1)
    ).view(turns, width)

    teacher_kl = None
    if teacher is not None:
        with torch.no_grad():
            teacher_units, teacher_quantity, teacher_market, _ = teacher.policy(
                board.flatten(0, 1),
                scalars.flatten(0, 1),
                positions.flatten(0, 1),
            )
        teacher_kl = _kl(
            units, _acted(teacher_units, turns, width), flat_unit_masks
        ).view(turns, width) + _kl(
            market, _acted(teacher_market, turns, width), flat_market_masks
        ).view(turns, width)
        if teacher.quantity:
            teacher_kl = teacher_kl + _kl(
                quantities,
                _acted(teacher_quantity, turns, width),
                flat_quantity_masks,
            ).view(turns, width)

    loss = _losses(
        behaviour_log_probs=behaviour,
        learner_log_probs=learner_log_probs,
        negative_entropy=negative_entropy,
        values=values,
        bootstrap_value=bootstrap_value,
        rewards=rewards,
        dones=dones,
        discounting=config.optimizer.gamma,
        baseline_only=batch.baseline_only if baseline_only is None else baseline_only,
        teacher_kl=teacher_kl,
        teacher_kl_cost=config.optimizer.teacher_kl_cost,
        entropy_cost=config.optimizer.entropy_cost,
        lmb=config.optimizer.lmb,
    )
    total = loss.total
    if belief_enabled:
        total = total + config.model.belief_loss_weight * belief_terms["belief"]
    return LossReport(
        total=total,
        terms={
            "vtrace_pg": loss.vtrace_pg,
            "upgo_pg": loss.upgo_pg,
            "baseline": loss.baseline,
            "entropy": loss.entropy,
            "teacher": loss.teacher,
            **belief_terms,
            "total": total,
        },
    )


def round_decay(config: ToadConfig) -> Callable[[int], float]:
    """Return the control schedule keyed to one step per collection round."""
    from kaggriculture.learn.scripts.toad import _decay

    return _decay(
        config.population.scripted,
        total_steps=config.runtime.total_environment_steps,
        environments=config.population.environments_per_rank,
        min_lr_multiplier=config.optimizer.final_lr_multiplier,
    )


class ToadLightningModule(lightning.LightningModule):
    """Native Toad policy trained through Lightning automatic optimization."""

    def __init__(self, config: ToadConfig) -> None:
        super().__init__()
        self.config = config
        self.policy: PolicyLike = (
            StatefulPolicy(config.model)
            if uses_stateful_policy(config.model)
            else Policy(
                blocks=config.model.blocks,
                channels=config.model.channels,
                value_bound=config.model.value_bound,
                kernel_size=config.model.kernel_size,
                activation=config.model.activation,
            )
        )
        self.warm_start_migration: WarmStartMigration | None = None
        if config.curriculum.warm_start_checkpoint is not None:
            self.warm_start_migration = initialize_policy_from_checkpoint(
                self.policy, config.curriculum.warm_start_checkpoint
            )
        self.teacher_policy: Policy | None = None
        self.teacher: Teacher | None = None
        if config.population.teacher_checkpoint is not None:
            from kaggriculture.learn.scripts.toad import Teacher

            self.teacher_policy = Policy(
                blocks=config.population.teacher_blocks or config.model.blocks,
                channels=config.model.channels,
                value_bound=config.model.value_bound,
            )
            missing = load_checkpoint_policy(
                self.teacher_policy, config.population.teacher_checkpoint
            )
            self.teacher_policy.eval()
            self.teacher_policy.requires_grad_(False)
            self.teacher = Teacher(
                policy=self.teacher_policy,
                quantity=not any(name.startswith("quantity_head.") for name in missing),
            )
        self.environment_steps = 0
        self.collection_round = 0
        self.actor_version = 0
        self.actor_source_global_step = 0
        self.warmup_remaining = config.optimizer.value_warmup_batches
        self._round_ended = False
        self._round_started_warming = False
        self._round_fresh_terms: list[Mapping[str, torch.Tensor]] = []
        self._round_baselines: list[torch.Tensor] = []
        self._round_metrics: dict[str, float | int] = {}
        self._round_population = {"selfplay": 0, "scripted": 0}
        self.save_hyperparameters(config.model_dump(mode="json"))

    def train(self, mode: bool = True) -> Self:
        """Change learner mode while keeping the frozen teacher in evaluation."""
        super().train(mode)
        if self.teacher_policy is not None:
            self.teacher_policy.eval()
        return self

    def transfer_batch_to_device(
        self,
        batch: LearnerBatch,
        device: torch.device,
        dataloader_idx: int,
    ) -> LearnerBatch:
        """Move tensors while preserving the immutable learner-batch contract."""
        return replace(
            batch,
            segments=tuple(
                {name: tensor.to(device) for name, tensor in segment.items()}
                for segment in batch.segments
            ),
        )

    def training_step(self, batch: LearnerBatch, batch_idx: int) -> torch.Tensor:
        """Compute one optimizer-sized batch and advance logical round clocks."""
        if batch.first_of_round:
            self._round_started_warming = self.warmup_remaining > 0
            self._round_fresh_terms = []
            self._round_baselines = []
            self._round_metrics = dict(batch.round_metrics)
            self._round_population = {"selfplay": 0, "scripted": 0}
        baseline_only = batch.baseline_only or self.warmup_remaining > 0
        report = compute_loss(
            self.policy,
            batch,
            self.config,
            teacher=self.teacher,
            baseline_only=baseline_only,
        )
        self._round_ended = batch.end_of_round
        self._round_baselines.append(report.terms["baseline"].detach())
        if not batch.baseline_only:
            self._round_fresh_terms.append(
                {name: value.detach() for name, value in report.terms.items()}
            )
            for kind in batch.segment_kinds or (batch.kind,) * len(batch.segments):
                if kind is not BatchKind.MIXED:
                    self._round_population[kind.value] += 1
        if not batch.baseline_only and self.warmup_remaining:
            self.warmup_remaining -= 1
        self.environment_steps += batch.collected_steps
        if batch.end_of_round:
            self.collection_round += 1
        if batch.end_of_round:
            self.log_dict(self._round_log_record())
        return report.total

    def _round_log_record(self) -> dict[str, torch.Tensor | int | float]:
        """Return one stable dashboard record for the completed logical round."""
        fresh = self._round_fresh_terms

        def mean_term(name: str) -> torch.Tensor:
            values = [terms[name] for terms in fresh if name in terms]
            return torch.stack(values).mean() if values else torch.tensor(float("nan"))

        optimizer_steps = int(self.global_step) + 1
        try:
            lr = float(self.trainer.optimizers[0].param_groups[0]["lr"])
        except RuntimeError:
            lr = float(self.config.optimizer.lr)
        baseline_passes = torch.stack(self._round_baselines).mean()
        record: dict[str, torch.Tensor | int | float] = {
            **self._round_metrics,
            "diag/update": self.collection_round,
            "diag/steps": self.environment_steps,
            "diag/environment_steps": self.environment_steps,
            "diag/optimizer_steps": optimizer_steps,
            "diag/collection_round": self.collection_round,
            "diag/population_selfplay_segments": self._round_population["selfplay"],
            "diag/population_scripted_segments": self._round_population["scripted"],
            "diag/actor_version": self.actor_version,
            "diag/actor_lag_optimizer_steps": (
                optimizer_steps - self.actor_source_global_step
            ),
            "diag/lr": lr,
            "diag/warming": self._round_started_warming,
            "diag/warmup_left": self.warmup_remaining,
            "diag/vtrace_pg": mean_term("vtrace_pg"),
            "diag/upgo_pg": mean_term("upgo_pg"),
            "critic/baseline_self_consistency": mean_term("baseline"),
            "critic/baseline_passes_self_consistency": baseline_passes,
            "diag/entropy": mean_term("entropy"),
            "diag/teacher_kl": mean_term("teacher"),
            "loss/belief": mean_term("belief"),
            "belief/shed_error": mean_term("belief_shed"),
            "belief/seeds_error": mean_term("belief_seeds"),
            "belief/carried_error": mean_term("belief_carried"),
            "belief/valid_count": mean_term("belief_valid"),
            "diag/total_loss": mean_term("total"),
        }
        return record

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Construct the pinned Adam optimizer and round-stepped LR schedule."""
        optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.config.optimizer.lr,
            eps=self.config.optimizer.adam_eps,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            round_decay(self.config),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def lr_scheduler_step(self, scheduler: LRScheduler, metric: object | None) -> None:
        """Advance LR exactly once when an optimizer step closes a round."""
        if self._round_ended:
            scheduler.step()

    def on_save_checkpoint(self, checkpoint: dict[str, object]) -> None:
        """Extend Lightning's authoritative state with Toad's logical clocks."""
        checkpoint["toad"] = {
            "config": self.config.model_dump(mode="json"),
            "fingerprint": structural_fingerprint(self.config),
            "environment_steps": self.environment_steps,
            "collection_round": self.collection_round,
            "actor_version": self.actor_version,
            "actor_source_global_step": self.actor_source_global_step,
            "warmup_remaining": self.warmup_remaining,
            "teacher": self._teacher_metadata(),
        }

    def _teacher_metadata(self) -> dict[str, object]:
        """Return structural teacher semantics that must survive resume."""
        return {
            "present": self.teacher is not None,
            "blocks": (
                self.config.population.teacher_blocks or self.config.model.blocks
                if self.teacher is not None
                else None
            ),
            "quantity": self.teacher.quantity if self.teacher is not None else None,
        }

    def on_load_checkpoint(self, checkpoint: dict[str, object]) -> None:
        """Validate structure and restore Toad's logical clocks."""
        state = cast(dict[str, object], checkpoint["toad"])
        stored_config = validate_stored_config(state["config"])
        assert_resume_compatible(self.config, stored_config)
        stored_teacher = cast(dict[str, object], state["teacher"])
        current_teacher = self._teacher_metadata()
        if stored_teacher != current_teacher:
            raise ResumeConfigError(
                "teacher metadata mismatch: "
                f"stored={stored_teacher!r}, effective={current_teacher!r}"
            )
        self.environment_steps = cast(int, state["environment_steps"])
        self.collection_round = cast(int, state["collection_round"])
        self.actor_version = cast(int, state["actor_version"])
        self.actor_source_global_step = cast(int, state["actor_source_global_step"])
        self.warmup_remaining = cast(int, state["warmup_remaining"])
