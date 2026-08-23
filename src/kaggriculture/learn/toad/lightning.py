"""Lightning's automatic-optimization boundary for the native Toad learner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast

import lightning
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim.lr_scheduler import LRScheduler

from kaggriculture.learn import toad_loss
from kaggriculture.learn.encoding import transfer_slots
from kaggriculture.learn.model import Policy, load_policy_weights
from kaggriculture.learn.ppo import entropy_of, joint_log_prob
from kaggriculture.learn.toad.config import (
    STRUCTURAL_FIELDS,
    ToadConfig,
    structural_fingerprint,
)
from kaggriculture.learn.toad.data import LearnerBatch

if TYPE_CHECKING:
    from kaggriculture.learn.scripts.toad import Teacher


@dataclass(frozen=True)
class LossReport:
    """Differentiable total and detached-by-caller diagnostic loss terms."""

    total: torch.Tensor
    terms: Mapping[str, torch.Tensor]


class ResumeConfigError(ValueError):
    """The effective config cannot safely consume the stored trainer state."""


def load_checkpoint_policy(policy: Policy, path: Path) -> list[str]:
    """Load policy weights from bare, legacy-runner, or Lightning checkpoints."""
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
    elif "learner" in checkpoint:
        weights = checkpoint["learner"]
        if not isinstance(weights, Mapping):
            raise ValueError("legacy checkpoint learner must be a mapping")
    else:
        weights = checkpoint
    return load_policy_weights(policy, cast(Mapping[str, torch.Tensor], weights))


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
    policy: Policy,
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
    unit_logits, quantity_logits, market_logits, values = policy(
        board.flatten(0, 1), scalars.flatten(0, 1), positions.flatten(0, 1)
    )
    values = values.view(turns + 1, width)
    bootstrap_value = values[-1].detach()
    values = values[:-1]
    unit_logits = _acted(unit_logits, turns, width)
    quantity_logits = _acted(quantity_logits, turns, width)
    market_logits = _acted(market_logits, turns, width)
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
    return LossReport(
        total=loss.total,
        terms={
            "vtrace_pg": loss.vtrace_pg,
            "upgo_pg": loss.upgo_pg,
            "baseline": loss.baseline,
            "entropy": loss.entropy,
            "teacher": loss.teacher,
            "total": loss.total,
        },
    )


def round_decay(config: ToadConfig) -> Callable[[int], float]:
    """Return the control schedule keyed to one step per collection round."""
    from kaggriculture.learn.scripts.toad import _decay

    return _decay(
        config.population.scripted,
        total_steps=config.runtime.total_environment_steps,
        environments=config.population.environments_per_rank,
    )


class ToadLightningModule(lightning.LightningModule):
    """Native Toad policy trained through Lightning automatic optimization."""

    def __init__(self, config: ToadConfig) -> None:
        super().__init__()
        self.config = config
        self.policy = Policy(
            blocks=config.model.blocks,
            channels=config.model.channels,
            value_bound=config.model.value_bound,
        )
        if config.curriculum.warm_start_checkpoint is not None:
            load_checkpoint_policy(self.policy, config.curriculum.warm_start_checkpoint)
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
        baseline_only = batch.baseline_only or self.warmup_remaining > 0
        report = compute_loss(
            self.policy,
            batch,
            self.config,
            teacher=self.teacher,
            baseline_only=baseline_only,
        )
        self._round_ended = batch.end_of_round
        if not batch.baseline_only and self.warmup_remaining:
            self.warmup_remaining -= 1
        self.environment_steps += batch.collected_steps
        if batch.end_of_round:
            self.collection_round += 1
        self.log_dict(
            {f"loss/{name}": value.detach() for name, value in report.terms.items()}
        )
        return report.total

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
        }

    def on_load_checkpoint(self, checkpoint: dict[str, object]) -> None:
        """Validate structure and restore Toad's logical clocks."""
        state = cast(dict[str, object], checkpoint["toad"])
        stored_config = ToadConfig.model_validate(state["config"])
        assert_resume_compatible(self.config, stored_config)
        self.environment_steps = cast(int, state["environment_steps"])
        self.collection_round = cast(int, state["collection_round"])
        self.actor_version = cast(int, state["actor_version"])
        self.actor_source_global_step = cast(int, state["actor_source_global_step"])
        self.warmup_remaining = cast(int, state["warmup_remaining"])
