"""Lightning's automatic-optimization boundary for the native Toad learner."""

from __future__ import annotations

import math
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
    EntropyControllerConfig,
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
from kaggriculture.learn.toad.population import LoadedTeacher, load_teacher

if TYPE_CHECKING:
    from kaggriculture.learn.scripts.toad import Teacher


@dataclass(frozen=True)
class EntropyControllerState:
    """Checkpointable target and multiplier after one completed round."""

    target: float
    multiplier: float
    last_steps: int


def update_entropy_controller(
    state: EntropyControllerState,
    observed: float | None,
    steps: int,
    config: EntropyControllerConfig,
) -> EntropyControllerState:
    """Advance one controller using FP32 multiplicative target feedback."""
    if state.last_steps < 0 or steps < 0 or steps < state.last_steps:
        raise ValueError("controller steps must be nonnegative and nondecreasing")
    finite_values = (state.target, state.multiplier)
    if observed is not None:
        finite_values = (*finite_values, observed)
    if not all(math.isfinite(value) for value in finite_values):
        raise ValueError("controller state and observation must be finite")

    delta = steps - state.last_steps
    values = torch.tensor(
        [
            state.target,
            state.multiplier,
            state.target if observed is None else observed,
            config.target_change_per_step,
            config.target_floor,
            config.multiplier_change_per_step,
            config.minimum,
            config.maximum,
        ],
        dtype=torch.float32,
    )
    target = torch.maximum(values[4], values[0] + values[3] * delta)
    if observed is None:
        return EntropyControllerState(
            target=float(target),
            multiplier=state.multiplier,
            last_steps=steps,
        )
    change = values[5] * delta
    multiplier = values[1]
    if values[2] > target:
        multiplier = multiplier * (1.0 - change)
    elif values[2] < target:
        multiplier = multiplier * (1.0 + change)
    elif config.minimum <= state.multiplier <= config.maximum:
        return EntropyControllerState(
            target=float(target),
            multiplier=state.multiplier,
            last_steps=steps,
        )
    multiplier = multiplier.clamp(min=values[6], max=values[7])
    return EntropyControllerState(
        target=float(target),
        multiplier=float(multiplier),
        last_steps=steps,
    )


@dataclass(frozen=True)
class EntropyStat:
    """Positive entropy mass and the number of decisions that produced it."""

    sum: torch.Tensor
    valid: torch.Tensor

    @property
    def mean(self) -> torch.Tensor:
        """Return the safe FP32 diagnostic mean without storing one eagerly."""
        return self.sum / self.valid.to(dtype=self.sum.dtype).clamp_min(1.0)


@dataclass(frozen=True)
class HeadEntropy:
    """Entropy statistics for Toad's three action heads."""

    operation: EntropyStat
    quantity: EntropyStat
    market: EntropyStat

    def items(self) -> tuple[tuple[str, EntropyStat], ...]:
        """Return named head statistics in stable action order."""
        return (
            ("operation", self.operation),
            ("quantity", self.quantity),
            ("market", self.market),
        )


def _entropy_by_row(  # noqa: C901
    *,
    unit_log_probs: torch.Tensor,
    quantity_log_probs: torch.Tensor,
    market_log_probs: torch.Tensor,
    unit_masks: torch.Tensor,
    quantity_masks: torch.Tensor,
    market_masks: torch.Tensor,
    unit_actions: torch.Tensor,
    unit_valid: torch.Tensor,
) -> HeadEntropy:
    """Return FP32 entropy sum/count pairs retaining every leading row."""
    if unit_valid.dtype is not torch.bool:
        raise ValueError("unit_valid must be boolean")
    if unit_valid.shape != unit_actions.shape:
        raise ValueError("unit_valid must match unit_actions exactly")
    if unit_valid.device != unit_actions.device:
        raise ValueError("unit_valid and unit_actions must share a device")
    if any(
        tensor.device != unit_valid.device
        for tensor in (
            unit_log_probs,
            quantity_log_probs,
            market_log_probs,
            unit_masks,
            quantity_masks,
            market_masks,
        )
    ):
        raise ValueError("entropy logits, masks, and unit_valid must share a device")
    if any(
        mask.dtype is not torch.bool
        for mask in (unit_masks, quantity_masks, market_masks)
    ):
        raise ValueError("entropy masks must be boolean")
    if (
        unit_log_probs.shape[:-1] != unit_valid.shape
        or unit_masks.shape != unit_log_probs.shape
    ):
        raise ValueError("unit operation logits and masks must match unit_valid")
    if (
        quantity_log_probs.shape[:-1] != unit_valid.shape
        or quantity_masks.shape != quantity_log_probs.shape
    ):
        raise ValueError("unit quantity logits and masks must match unit_valid")
    if market_masks.shape != market_log_probs.shape:
        raise ValueError("market logits and masks must match")
    if market_log_probs.shape[:-2] != unit_valid.shape[:-1]:
        raise ValueError("market leading dimensions must match unit_valid")
    operation_valid = unit_valid
    quantity_valid = transfer_slots(unit_actions) & operation_valid
    market_valid = torch.ones_like(market_masks[..., 0], dtype=torch.bool)

    def stat(
        log_probs: torch.Tensor, mask: torch.Tensor, valid: torch.Tensor
    ) -> EntropyStat:
        # This is an explicit FP32 island: entropy's masked products otherwise
        # inherit autocast precision even though the network forward may not.
        safe_log_probs = log_probs.float().masked_fill(~mask, 0.0)
        per_slot = entropy_of(safe_log_probs, mask).masked_fill(~valid, 0.0)
        return EntropyStat(sum=per_slot.sum(dim=-1), valid=valid.sum(dim=-1))

    return HeadEntropy(
        operation=stat(unit_log_probs, unit_masks, operation_valid),
        quantity=stat(quantity_log_probs, quantity_masks, quantity_valid),
        market=stat(market_log_probs, market_masks, market_valid),
    )


def _reduce_head_entropy(by_row: HeadEntropy) -> HeadEntropy:
    """Reduce row-preserving statistics without taking a premature mean."""
    return HeadEntropy(
        operation=EntropyStat(
            sum=by_row.operation.sum.sum(), valid=by_row.operation.valid.sum()
        ),
        quantity=EntropyStat(
            sum=by_row.quantity.sum.sum(), valid=by_row.quantity.valid.sum()
        ),
        market=EntropyStat(
            sum=by_row.market.sum.sum(), valid=by_row.market.valid.sum()
        ),
    )


def compute_head_entropy(
    unit_log_probs: torch.Tensor,
    quantity_log_probs: torch.Tensor,
    market_log_probs: torch.Tensor,
    unit_masks: torch.Tensor,
    quantity_masks: torch.Tensor,
    market_masks: torch.Tensor,
    unit_actions: torch.Tensor,
    unit_valid: torch.Tensor,
) -> HeadEntropy:
    """Measure differentiable, positive entropy by action head.

    ``unit_valid`` is frozen from the original observation-bounded encoded
    labels, rather than inferred from mutable learner action values or masks.
    Quantity rows additionally require the current operation to spend a
    quantity. Market slots have no such padding and are counted from their
    actual tensor shape.
    """
    by_row = _entropy_by_row(
        unit_log_probs=unit_log_probs,
        quantity_log_probs=quantity_log_probs,
        market_log_probs=market_log_probs,
        unit_masks=unit_masks,
        quantity_masks=quantity_masks,
        market_masks=market_masks,
        unit_actions=unit_actions,
        unit_valid=unit_valid,
    )
    return _reduce_head_entropy(by_row)


@dataclass(frozen=True)
class LossReport:
    """Differentiable total, head entropy, and detached-by-caller diagnostics."""

    total: torch.Tensor
    terms: Mapping[str, torch.Tensor]
    entropy: HeadEntropy


class ResumeConfigError(ValueError):
    """The effective config cannot safely consume the stored trainer state."""


PolicyLike = Policy | StatefulPolicy

CheckpointLayout = Literal["bare", "legacy_learner", "lightning_policy"]
_ONLINE_BATCH_KINDS = (
    BatchKind.SELFPLAY,
    BatchKind.SCRIPTED,
    BatchKind.FROZEN_OPPONENT,
    BatchKind.TEACHER_DISTILL,
)


def _empty_population_counts() -> dict[str, int]:
    """Return the stable online-kind metric schema for one logical round."""
    return {kind.value: 0 for kind in _ONLINE_BATCH_KINDS}


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


def compute_loss(  # noqa: C901
    policy: PolicyLike,
    batch: LearnerBatch,
    config: ToadConfig,
    teacher: LoadedTeacher | Teacher | None = None,
    *,
    baseline_only: bool | None = None,
    entropy_state: Mapping[str, EntropyControllerState] | None = None,
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
    unit_valid = stacked("unit_valid")
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
    flat_unit_valid = unit_valid.flatten(0, 1)
    flat_unit_masks = unit_masks.flatten(0, 1)
    flat_quantity_masks = unit_quantity_masks.flatten(0, 1)
    flat_market_masks = market_masks.flatten(0, 1)
    units = torch.log_softmax(
        unit_logits.float().masked_fill(~flat_unit_masks, -torch.inf), dim=-1
    )
    quantities = torch.log_softmax(
        quantity_logits.float().masked_fill(~flat_quantity_masks, -torch.inf), dim=-1
    )
    market = torch.log_softmax(
        market_logits.float().masked_fill(~flat_market_masks, -torch.inf), dim=-1
    )
    learner_log_probs = joint_log_prob(
        units,
        quantities,
        market,
        flat_unit_actions,
        unit_quantity_actions.flatten(0, 1),
        market_actions.flatten(0, 1),
        unit_valid=flat_unit_valid,
    ).view(turns, width)
    entropy_by_row = _entropy_by_row(
        unit_log_probs=units,
        quantity_log_probs=quantities,
        market_log_probs=market,
        unit_masks=flat_unit_masks,
        quantity_masks=flat_quantity_masks,
        market_masks=flat_market_masks,
        unit_actions=flat_unit_actions,
        unit_valid=flat_unit_valid,
    )
    head_entropy = _reduce_head_entropy(entropy_by_row)
    # Keep the fixed-coefficient control path byte-for-byte shaped as the
    # historical per-row summed entropy consumed by the authoritative loss.
    negative_entropy = -(
        entropy_by_row.operation.sum
        + entropy_by_row.quantity.sum
        + entropy_by_row.market.sum
    ).view(turns, width)

    zero = values.sum() * 0.0
    teacher_raw = {
        "operation_kl": zero,
        "quantity_kl": zero,
        "market_kl": zero,
        "value": zero,
    }
    teacher_kl = None
    if teacher is not None:
        declared = (
            {
                "operation": teacher.spec.operation,
                "quantity": teacher.spec.quantity,
                "market": teacher.spec.market,
                "value": teacher.spec.value,
            }
            if isinstance(teacher, LoadedTeacher)
            else {
                "operation": True,
                "quantity": teacher.quantity,
                "market": True,
                "value": False,
            }
        )
        with torch.no_grad():
            teacher_units, teacher_quantity, teacher_market, teacher_values = (
                teacher.policy(
                    board.flatten(0, 1),
                    scalars.flatten(0, 1),
                    positions.flatten(0, 1),
                )
            )
        if batch.segment_kinds:
            if len(batch.segment_kinds) != width:
                raise ValueError("segment_kinds must align exactly with batch segments")
            distill_columns = torch.tensor(
                [kind is BatchKind.TEACHER_DISTILL for kind in batch.segment_kinds],
                dtype=torch.bool,
                device=values.device,
            )
            teacher_columns = (
                distill_columns
                if distill_columns.any()
                else torch.ones_like(distill_columns)
            )
        elif batch.kind is BatchKind.MIXED:
            raise ValueError("mixed teacher loss requires per-segment kind provenance")
        else:
            teacher_columns = torch.ones(width, dtype=torch.bool, device=values.device)

        teacher_kl = torch.zeros(
            turns, width, dtype=torch.float32, device=values.device
        )

        def head_kl(
            learner_log_probs: torch.Tensor,
            teacher_head_logits: torch.Tensor,
            mask: torch.Tensor,
        ) -> torch.Tensor:
            per_step = _kl(
                learner_log_probs.float(),
                _acted(teacher_head_logits, turns, width).float(),
                mask,
            ).view(turns, width)
            return per_step.masked_fill(~teacher_columns.unsqueeze(0), 0.0)

        if declared["operation"]:
            operation_kl = head_kl(units, teacher_units, flat_unit_masks)
            teacher_raw["operation_kl"] = toad_loss.reduce(operation_kl)
            teacher_kl = teacher_kl + operation_kl
        if declared["quantity"]:
            quantity_kl = head_kl(quantities, teacher_quantity, flat_quantity_masks)
            teacher_raw["quantity_kl"] = toad_loss.reduce(quantity_kl)
            teacher_kl = teacher_kl + quantity_kl
        if declared["market"]:
            market_kl = head_kl(market, teacher_market, flat_market_masks)
            teacher_raw["market_kl"] = toad_loss.reduce(market_kl)
            teacher_kl = teacher_kl + market_kl
        if declared["value"]:
            acted_teacher_values = _acted(
                teacher_values.unsqueeze(-1), turns, width
            ).view(turns, width)
            value_error = functional.smooth_l1_loss(
                values.float(), acted_teacher_values.float(), reduction="none"
            ).masked_fill(~teacher_columns.unsqueeze(0), 0.0)
            teacher_raw["value"] = toad_loss.reduce(value_error)

    effective_baseline_only = (
        batch.baseline_only if baseline_only is None else baseline_only
    )
    loss = _losses(
        behaviour_log_probs=behaviour,
        learner_log_probs=learner_log_probs,
        negative_entropy=negative_entropy,
        values=values,
        bootstrap_value=bootstrap_value,
        rewards=rewards,
        dones=dones,
        discounting=config.optimizer.gamma,
        baseline_only=effective_baseline_only,
        teacher_kl=teacher_kl,
        teacher_kl_cost=config.optimizer.teacher_kl_cost,
        entropy_cost=(
            0.0 if config.optimizer.adaptive_entropy else config.optimizer.entropy_cost
        ),
        lmb=config.optimizer.lmb,
    )
    total = loss.total
    adaptive_entropy_terms: dict[str, torch.Tensor] = {}
    entropy_term = loss.entropy
    if config.optimizer.adaptive_entropy:
        expected_heads = {name for name, _stat in head_entropy.items()}
        if entropy_state is None or set(entropy_state) != expected_heads:
            raise ValueError(
                "adaptive entropy requires operation, quantity, and market state"
            )
        weighted_terms = []
        for name, stat in head_entropy.items():
            state = entropy_state[name]
            if not math.isfinite(state.multiplier):
                raise ValueError("entropy multipliers must be finite")
            raw = -(stat.sum.float() / stat.valid.float().clamp_min(1.0))
            weighted = raw * state.multiplier
            adaptive_entropy_terms[f"entropy/{name}_raw"] = raw
            adaptive_entropy_terms[f"entropy/{name}_weighted"] = weighted
            weighted_terms.append(weighted)
        entropy_term = torch.stack(weighted_terms).sum()
        if not effective_baseline_only:
            total = total + entropy_term
    teacher_weighted = {
        "operation_kl": config.optimizer.teacher_kl_cost * teacher_raw["operation_kl"],
        "quantity_kl": config.optimizer.teacher_kl_cost * teacher_raw["quantity_kl"],
        "market_kl": config.optimizer.teacher_kl_cost * teacher_raw["market_kl"],
        "value": config.optimizer.teacher_baseline_cost * teacher_raw["value"],
    }
    total = total + teacher_weighted["value"]
    if belief_enabled:
        total = total + config.model.belief_loss_weight * belief_terms["belief"]
    terms = {
        "vtrace_pg": loss.vtrace_pg,
        "upgo_pg": loss.upgo_pg,
        "baseline": loss.baseline,
        "entropy": entropy_term,
        "teacher": loss.teacher,
        **belief_terms,
        **adaptive_entropy_terms,
        "total": total,
    }
    if teacher is not None:
        terms.update(
            {
                "teacher/operation_kl": teacher_raw["operation_kl"],
                "teacher/operation_kl_weighted": teacher_weighted["operation_kl"],
                "teacher/quantity_kl": teacher_raw["quantity_kl"],
                "teacher/quantity_kl_weighted": teacher_weighted["quantity_kl"],
                "teacher/market_kl": teacher_raw["market_kl"],
                "teacher/market_kl_weighted": teacher_weighted["market_kl"],
                "teacher/value": teacher_raw["value"],
                "teacher/value_weighted": teacher_weighted["value"],
            }
        )
    return LossReport(total=total, terms=terms, entropy=head_entropy)


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
        self.teacher: LoadedTeacher | None = None
        if config.population.teacher is not None:
            self.teacher = load_teacher(config.population.teacher, config.model)
            self.teacher_policy = self.teacher.policy
            self.config = config.model_copy(
                update={
                    "population": config.population.model_copy(
                        update={"teacher": self.teacher.spec}
                    )
                }
            )
        self.environment_steps = 0
        self.collection_round = 0
        self.actor_version = 0
        self.actor_source_global_step = 0
        self.warmup_remaining = config.optimizer.value_warmup_batches
        self.entropy_state = {
            name: EntropyControllerState(
                target=controller.initial_target,
                multiplier=controller.initial_multiplier,
                last_steps=0,
            )
            for name, controller in self.config.optimizer.entropy.items()
        }
        self._round_ended = False
        self._round_started_warming = False
        self._round_fresh_terms: list[Mapping[str, torch.Tensor]] = []
        self._round_fresh_entropy: list[HeadEntropy] = []
        self._round_baselines: list[torch.Tensor] = []
        self._round_metrics: dict[str, float | int] = {}
        self._round_population = _empty_population_counts()
        self.save_hyperparameters(self.config.model_dump(mode="json"))

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
            self._round_fresh_entropy = []
            self._round_baselines = []
            self._round_metrics = dict(batch.round_metrics)
            self._round_population = _empty_population_counts()
        baseline_only = batch.baseline_only or self.warmup_remaining > 0
        report = compute_loss(
            self.policy,
            batch,
            self.config,
            teacher=self.teacher,
            baseline_only=baseline_only,
            entropy_state=self.entropy_state,
        )
        self._round_ended = batch.end_of_round
        self._round_baselines.append(report.terms["baseline"].detach())
        if not batch.baseline_only:
            self._round_fresh_terms.append(
                {name: value.detach() for name, value in report.terms.items()}
            )
            self._round_fresh_entropy.append(
                HeadEntropy(
                    operation=EntropyStat(
                        sum=report.entropy.operation.sum.detach().float(),
                        valid=report.entropy.operation.valid.detach().to(torch.int64),
                    ),
                    quantity=EntropyStat(
                        sum=report.entropy.quantity.sum.detach().float(),
                        valid=report.entropy.quantity.valid.detach().to(torch.int64),
                    ),
                    market=EntropyStat(
                        sum=report.entropy.market.sum.detach().float(),
                        valid=report.entropy.market.valid.detach().to(torch.int64),
                    ),
                )
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
            self._update_entropy_controllers()
            self.log_dict(self._round_log_record())
            self._round_fresh_entropy = []
        return report.total

    def _aggregate_round_entropy(self, name: str) -> EntropyStat:
        """Reduce detached fresh-batch statistics for one completed round."""
        values = [getattr(item, name) for item in self._round_fresh_entropy]
        if not values:
            return EntropyStat(
                sum=torch.tensor(0.0, device=self.device),
                valid=torch.tensor(0, dtype=torch.int64, device=self.device),
            )
        return EntropyStat(
            sum=torch.stack([stat.sum.float() for stat in values]).sum(),
            valid=torch.stack([stat.valid.to(torch.int64) for stat in values]).sum(),
        )

    def _update_entropy_controllers(self) -> None:
        """Apply one fresh-round observation after the global step clock advances."""
        if not self.config.optimizer.adaptive_entropy:
            return
        for name, controller in self.config.optimizer.entropy.items():
            stat = self._aggregate_round_entropy(name)
            observed = float(stat.mean) if bool(stat.valid > 0) else None
            self.entropy_state[name] = update_entropy_controller(
                self.entropy_state[name],
                observed,
                self.environment_steps,
                controller,
            )

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
            "diag/population_frozen_opponent_segments": self._round_population[
                "frozen_opponent"
            ],
            "diag/population_teacher_distill_segments": self._round_population[
                "teacher_distill"
            ],
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
            "teacher/operation_kl": mean_term("teacher/operation_kl"),
            "teacher/operation_kl_weighted": mean_term("teacher/operation_kl_weighted"),
            "teacher/quantity_kl": mean_term("teacher/quantity_kl"),
            "teacher/quantity_kl_weighted": mean_term("teacher/quantity_kl_weighted"),
            "teacher/market_kl": mean_term("teacher/market_kl"),
            "teacher/market_kl_weighted": mean_term("teacher/market_kl_weighted"),
            "teacher/value": mean_term("teacher/value"),
            "teacher/value_weighted": mean_term("teacher/value_weighted"),
            "loss/belief": mean_term("belief"),
            "belief/shed_error": mean_term("belief_shed"),
            "belief/seeds_error": mean_term("belief_seeds"),
            "belief/carried_error": mean_term("belief_carried"),
            "belief/valid_count": mean_term("belief_valid"),
            "diag/total_loss": mean_term("total"),
        }
        for name in ("operation", "quantity", "market"):
            stat = self._aggregate_round_entropy(name)
            observed = (
                stat.mean
                if bool(stat.valid > 0)
                else torch.full_like(stat.sum, float("nan"))
            )
            state = self.entropy_state[name]
            record[f"entropy/{name}_sum"] = stat.sum
            record[f"entropy/{name}_valid"] = stat.valid.float()
            record[f"entropy/{name}_mean"] = observed
            record[f"entropy/{name}_observed"] = observed
            record[f"entropy/{name}_target"] = state.target
            record[f"entropy/{name}_multiplier"] = state.multiplier
            record[f"entropy/{name}_raw_loss"] = mean_term(f"entropy/{name}_raw")
            record[f"entropy/{name}_weighted_loss"] = mean_term(
                f"entropy/{name}_weighted"
            )
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
            "entropy_state": {
                name: {
                    "target": state.target,
                    "multiplier": state.multiplier,
                    "last_steps": state.last_steps,
                }
                for name, state in self.entropy_state.items()
            },
            "teacher": self._teacher_metadata(),
        }

    def _teacher_metadata(self) -> dict[str, object]:
        """Return structural teacher semantics that must survive resume."""
        return {"present": False} if self.teacher is None else self.teacher.metadata()

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
        stored_entropy = cast(dict[str, object], state["entropy_state"])
        expected_heads = {name for name, _ in self.config.optimizer.entropy.items()}
        if set(stored_entropy) != expected_heads:
            raise ResumeConfigError(
                "entropy controller state must contain operation, quantity, and market"
            )
        restored_entropy: dict[str, EntropyControllerState] = {}
        for name, controller in self.config.optimizer.entropy.items():
            payload = cast(dict[str, object], stored_entropy[name])
            restored = EntropyControllerState(
                target=float(cast(float, payload["target"])),
                multiplier=float(cast(float, payload["multiplier"])),
                last_steps=int(cast(int, payload["last_steps"])),
            )
            if (
                restored.target < 0
                or not math.isfinite(restored.target)
                or not math.isfinite(restored.multiplier)
                or not controller.minimum <= restored.multiplier <= controller.maximum
                or restored.last_steps < 0
                or restored.last_steps > self.environment_steps
            ):
                raise ResumeConfigError(f"invalid entropy controller state for {name}")
            restored_entropy[name] = restored
        self.entropy_state = restored_entropy
