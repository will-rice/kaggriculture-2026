"""Lightning's automatic-optimization boundary for the native Toad learner."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, cast

import lightning
import torch
import torch.nn.functional as functional
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim import Optimizer
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
from kaggriculture.learn.toad.compile import (
    canonicalize_policy_state_keys,
    prepare_policy_state_keys_for_load,
    unwrap_compiled,
)
from kaggriculture.learn.toad.config import (
    EntropyControllerConfig,
    ToadConfig,
    structural_fingerprint,
    validate_stored_config,
)
from kaggriculture.learn.toad.data import BatchKind, LearnerBatch, _all_gather_objects
from kaggriculture.learn.toad.model import (
    PolicyOutput,
    PolicyState,
    StatefulPolicy,
    uses_stateful_policy,
)
from kaggriculture.learn.toad.population import (
    LoadedTeacher,
    SnapshotManifest,
    load_teacher,
)

if TYPE_CHECKING:
    from kaggriculture.learn.scripts.toad import Teacher


@dataclass(frozen=True)
class EntropyControllerState:
    """Checkpointable target and multiplier after one completed round."""

    target: float
    multiplier: float
    last_steps: int


def _validate_controller_steps(last_steps: object, steps: object) -> None:
    """Reject malformed controller clocks before ordering or arithmetic."""
    if (
        not isinstance(last_steps, int)
        or isinstance(last_steps, bool)
        or not isinstance(steps, int)
        or isinstance(steps, bool)
        or last_steps < 0
        or steps < 0
    ):
        raise ValueError("controller steps must be nonnegative non-boolean integers")
    if steps < last_steps:
        raise ValueError("controller steps must be nondecreasing")


def _controller_fp32_values(
    state: EntropyControllerState,
    observed: float | None,
    delta: int,
    config: EntropyControllerConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialize controller inputs and target, rejecting FP32 overflow."""
    try:
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
        delta_fp32 = torch.tensor(delta, dtype=torch.float32)
    except (OverflowError, RuntimeError) as error:
        raise ValueError("computed controller values must be finite in FP32") from error
    if not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(delta_fp32)):
        raise ValueError("computed controller values must be finite in FP32")
    target = torch.maximum(values[4], values[0] + values[3] * delta_fp32)
    if not bool(torch.isfinite(target)):
        raise ValueError("computed controller values must be finite in FP32")
    return values, delta_fp32, target


def update_entropy_controller(
    state: EntropyControllerState,
    observed: float | None,
    steps: int,
    config: EntropyControllerConfig,
) -> EntropyControllerState:
    """Advance one controller using FP32 multiplicative target feedback."""
    _validate_controller_steps(state.last_steps, steps)
    finite_values = (state.target, state.multiplier)
    if observed is not None:
        finite_values = (*finite_values, observed)
    if not all(math.isfinite(value) for value in finite_values):
        raise ValueError("controller state and observation must be finite")

    delta = steps - state.last_steps
    values, delta_fp32, target = _controller_fp32_values(state, observed, delta, config)
    if observed is None:
        return EntropyControllerState(
            target=float(target),
            multiplier=state.multiplier,
            last_steps=steps,
        )
    change = values[5] * delta_fp32
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
    if not bool(torch.isfinite(multiplier)):
        raise ValueError("computed controller values must be finite in FP32")
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
class _FiniteTensorCheck:
    """Device-resident per-tensor flags with lazy failure-name diagnostics."""

    names: tuple[str, ...]
    flags: tuple[torch.Tensor, ...]

    @classmethod
    def from_tensors(cls, tensors: Mapping[str, torch.Tensor]) -> Self:
        """Build scalar device flags without reading any of them on the host."""
        return cls(
            names=tuple(tensors),
            flags=tuple(torch.isfinite(tensor).all() for tensor in tensors.values()),
        )

    def merged(self, other: Self) -> Self:
        """Combine phase checks while retaining stable diagnostic order."""
        return type(self)(self.names + other.names, self.flags + other.flags)

    def local_bad(self, *, device: torch.device | None = None) -> torch.Tensor:
        """Return one device scalar representing every tensor in this phase."""
        if not self.flags:
            return torch.tensor(False, device=device)
        return ~torch.stack(self.flags).all()

    def nonfinite_names(self) -> list[str]:
        """Read individual flags only after the aggregate reports a failure."""
        if not self.flags or not bool(self.local_bad()):
            return []
        return [
            name
            for name, finite in zip(self.names, self.flags, strict=True)
            if not bool(finite)
        ]

    def __iter__(self) -> Iterable[str]:
        """Retain the existing diagnostic seam for tests and integrations."""
        return iter(self.nonfinite_names())


@dataclass(frozen=True)
class LossReport:
    """Differentiable total, head entropy, and detached-by-caller diagnostics."""

    total: torch.Tensor
    terms: Mapping[str, torch.Tensor]
    entropy: HeadEntropy
    debug_dtypes: Mapping[str, torch.dtype]
    provenance: NonFiniteProvenance
    finite_check: _FiniteTensorCheck

    @property
    def nonfinite_names(self) -> tuple[str, ...]:
        """Materialize names only when a caller explicitly asks for diagnostics."""
        return tuple(self.finite_check.nonfinite_names())


@dataclass(frozen=True)
class FP32PolicyTerms:
    """The numerically sensitive policy outputs after the autocast boundary."""

    unit_logits: torch.Tensor
    quantity_logits: torch.Tensor
    market_logits: torch.Tensor
    values: torch.Tensor
    belief_logits: torch.Tensor | None


def fp32_policy_terms(output: PolicyOutput, batch: LearnerBatch) -> FP32PolicyTerms:
    """Promote all loss-facing policy outputs while leaving feature compute alone."""
    for segment in batch.segments:
        for name in ("unit_masks", "unit_quantity_masks", "market_masks", "unit_valid"):
            if segment[name].dtype is not torch.bool:
                raise ValueError(f"{name} must remain boolean across the FP32 boundary")
    return FP32PolicyTerms(
        unit_logits=output.unit_logits.float(),
        quantity_logits=output.quantity_logits.float(),
        market_logits=output.market_logits.float(),
        values=output.values.float(),
        belief_logits=(
            None if output.belief_logits is None else output.belief_logits.float()
        ),
    )


class NonFiniteProvenance:
    """Detached batch identity kept after forward for finite-gradient checks."""

    def __init__(
        self,
        *,
        batch_kind: str,
        game_ids: tuple[int, ...],
        opponent_digests: tuple[str | None, ...],
        actor_version: int,
        precision: str,
        state_norms: Mapping[str, float] | None = None,
        state_norm_tensors: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        self.batch_kind = batch_kind
        self.game_ids = game_ids
        self.opponent_digests = opponent_digests
        self.actor_version = actor_version
        self.precision = precision
        self._state_norms = dict(state_norms or {})
        self._state_norm_tensors = dict(state_norm_tensors or {})

    @property
    def state_norms(self) -> dict[str, float]:
        """Materialize recurrent diagnostics only after a finite failure."""
        if self._state_norm_tensors:
            self._state_norms.update(
                {name: float(value) for name, value in self._state_norm_tensors.items()}
            )
            self._state_norm_tensors = {}
        return dict(self._state_norms)

    @classmethod
    def from_batch(
        cls, batch: LearnerBatch, state: PolicyState | None, precision: str
    ) -> Self:
        """Capture detached device diagnostics without healthy-path host reads."""
        state_norm_tensors = {
            name: value.detach().float().norm()
            for name, value in (
                ("hidden", None if state is None else state.hidden),
                ("cell", None if state is None else state.cell),
            )
            if value is not None
        }
        return cls(
            batch_kind=batch.kind.value,
            game_ids=batch.game_ids,
            opponent_digests=batch.opponent_digests,
            actor_version=batch.actor_version,
            precision=precision,
            state_norm_tensors=state_norm_tensors,
        )

    def as_dict(self) -> dict[str, object]:
        """Return gathered primitives with no tensors or autograd ownership."""
        return {
            "batch_kind": self.batch_kind,
            "game_ids": self.game_ids,
            "opponent_digests": self.opponent_digests,
            "actor_version": self.actor_version,
            "precision": self.precision,
            "state_norms": self.state_norms,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        """Validate gathered primitives into canonical bad-rank provenance."""
        batch_kind = payload.get("batch_kind")
        game_ids = payload.get("game_ids")
        opponent_digests = payload.get("opponent_digests")
        actor_version = payload.get("actor_version")
        precision = payload.get("precision")
        state_norms = payload.get("state_norms")
        if (
            not isinstance(batch_kind, str)
            or not isinstance(game_ids, (tuple, list))
            or not all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in game_ids
            )
            or not isinstance(opponent_digests, (tuple, list))
            or not all(
                item is None or isinstance(item, str) for item in opponent_digests
            )
            or not isinstance(actor_version, int)
            or isinstance(actor_version, bool)
            or actor_version < 0
            or not isinstance(precision, str)
            or not isinstance(state_norms, Mapping)
        ):
            raise ValueError("distributed non-finite provenance is malformed")
        norms = {
            str(name): float(value)
            for name, value in state_norms.items()
            if isinstance(name, str) and isinstance(value, (int, float))
        }
        if len(norms) != len(state_norms):
            raise ValueError("distributed non-finite state norms are malformed")
        return cls(
            batch_kind=batch_kind,
            game_ids=tuple(cast(Sequence[int], game_ids)),
            opponent_digests=tuple(cast(Sequence[str | None], opponent_digests)),
            actor_version=actor_version,
            precision=precision,
            state_norms=norms,
        )


class NonFiniteTrainingError(RuntimeError):
    """A finite-check failure with collection provenance for recovery and triage."""

    def __init__(
        self,
        tensor_names: tuple[str, ...],
        *,
        provenance: NonFiniteProvenance,
    ) -> None:
        self.tensor_names = tensor_names
        self.batch_kind = provenance.batch_kind
        self.game_ids = provenance.game_ids
        self.opponent_digests = provenance.opponent_digests
        self.actor_version = provenance.actor_version
        self.precision = provenance.precision
        self.state_norms = provenance.state_norms
        self.distributed_details: tuple[Mapping[str, object] | None, ...] = ()
        super().__init__(
            "non-finite training tensors "
            f"{tensor_names}; kind={self.batch_kind}; game_ids={self.game_ids}; "
            f"opponent_digests={self.opponent_digests}; "
            f"actor_version={self.actor_version}; precision={self.precision}; "
            f"state_norms={self.state_norms}"
        )

    @classmethod
    def from_batch(
        cls,
        tensor_names: list[str],
        batch: LearnerBatch,
        state: PolicyState | None,
        precision: str,
    ) -> Self:
        """Create a structured local finite failure without distributed collectives."""
        return cls.from_provenance(
            tensor_names,
            NonFiniteProvenance.from_batch(batch, state, precision),
        )

    @classmethod
    def from_provenance(
        cls, tensor_names: list[str], provenance: NonFiniteProvenance
    ) -> Self:
        """Create an error from detached data; Task 3 can synchronize this seam."""
        return cls(tuple(tensor_names), provenance=provenance)

    @classmethod
    def render_distributed(
        cls,
        details: Sequence[Mapping[str, object] | None],
        fallback: NonFiniteProvenance,
    ) -> Self:
        """Render identical gathered bad-rank provenance on every process."""
        del fallback
        failures = [detail for detail in details if detail is not None]
        if not failures:
            raise ValueError("distributed non-finite error requires a bad rank")
        first = min(failures, key=lambda detail: int(cast(int, detail["rank"])))
        names = [str(name) for name in cast(Sequence[object], first["tensor_names"])]
        provenance_payload = first.get("provenance")
        if not isinstance(provenance_payload, Mapping):
            raise ValueError("distributed non-finite provenance is malformed")
        provenance = NonFiniteProvenance.from_dict(
            cast(Mapping[str, object], provenance_payload)
        )
        error = cls.from_provenance(names, provenance)
        error.distributed_details = tuple(details)
        error.args = (f"{error.args[0]}; distributed_details={tuple(details)!r}",)
        return error


def _nonfinite_tensor_names(tensors: Mapping[str, torch.Tensor]) -> list[str]:
    """Synchronize once on a healthy phase and diagnose names only on failure."""
    return _FiniteTensorCheck.from_tensors(tensors).nonfinite_names()


class ResumeConfigError(ValueError):
    """The effective config cannot safely consume the stored trainer state."""


_INVALID_CHECKPOINT_STATE = "invalid Toad checkpoint state"
_ENTROPY_HEADS = frozenset(("operation", "quantity", "market"))
_ENTROPY_STATE_FIELDS = frozenset(("target", "multiplier", "last_steps"))


def _checkpoint_mapping(value: object) -> Mapping[str, object]:
    """Require a mapping without accepting sequence-shaped checkpoint data."""
    if not isinstance(value, Mapping):
        raise ResumeConfigError(_INVALID_CHECKPOINT_STATE)
    return cast(Mapping[str, object], value)


def _checkpoint_nonnegative_int(value: object) -> int:
    """Return a checkpoint clock only when its runtime type is strictly integral."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ResumeConfigError(_INVALID_CHECKPOINT_STATE)
    return value


def _checkpoint_finite_float(value: object) -> float:
    """Return a finite checkpoint float without coercing another runtime type."""
    if not isinstance(value, float):
        raise ResumeConfigError(_INVALID_CHECKPOINT_STATE)
    if not math.isfinite(value):
        raise ResumeConfigError(_INVALID_CHECKPOINT_STATE)
    return value


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
class ReducibleMetric:
    """A detached metric numerator/count plus its cross-rank interpretation."""

    total: torch.Tensor
    count: torch.Tensor
    reduction: Literal["mean", "sum", "max"] = "mean"

    @property
    def value(self) -> torch.Tensor:
        """Form a mean only after callers have reduced numerator and count."""
        if self.reduction == "mean":
            return self.total / self.count.float().clamp_min(1.0)
        return self.total


class RoundMetricAccumulator:
    """Accumulate detached diagnostics and emit one completed-round record."""

    def __init__(self) -> None:
        self.reset()

    def reset(self, initial: Mapping[str, float | int] | None = None) -> None:
        """Begin an empty round while preserving collection-owned metrics."""
        self.initial = dict(initial or {})
        self.collected_steps = 0
        self.learner_seconds = 0.0
        self.learner_batches = 0
        self.games: dict[str, set[int]] = {
            kind.value: set() for kind in _ONLINE_BATCH_KINDS
        }
        self.opponents: dict[str, set[tuple[int, str, str | None]]] = defaultdict(set)
        self.losses: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.mask_sums: dict[str, torch.Tensor] = {}
        self.mask_counts: dict[str, torch.Tensor] = {}
        self.state_norms: dict[str, list[torch.Tensor]] = {
            "hidden": [],
            "cell": [],
        }
        self.terminal_resets: list[torch.Tensor] = []

    def update(
        self,
        batch: LearnerBatch,
        report: LossReport,
    ) -> None:
        """Add one trained batch without performing distributed logging."""
        self.collected_steps += batch.collected_steps
        if not batch.baseline_only:
            kinds = set(batch.segment_kinds or (batch.kind,))
            kind_name = (
                next(iter(kinds)).value
                if len(kinds) == 1 and BatchKind.MIXED not in kinds
                else BatchKind.MIXED.value
            )
            self.losses[f"loss/by_kind/{kind_name}"].append(
                report.total.detach().float()
            )
            self._record_selections(batch)
            opponents = set(batch.segment_opponent_ids or batch.opponent_ids)
            opponent_id = next(iter(opponents)) if len(opponents) == 1 else "mixed"
            safe_id = opponent_id.replace("/", "_")
            self.losses[f"loss/by_opponent/{safe_id}"].append(
                report.total.detach().float()
            )
            self._record_masks(batch)
            self._record_state(batch)

    def finish_learner_batch(self, learner_seconds: float) -> None:
        """Close timing only after Lightning completes the optimizer step."""
        self.learner_seconds += learner_seconds
        self.learner_batches += 1

    def _record_selections(self, batch: LearnerBatch) -> None:
        """Count each collected game once despite its repeated learner segments."""
        segment_row_count = len(batch.segment_game_ids)
        if segment_row_count and all(
            len(values) == segment_row_count
            for values in (
                batch.segment_kinds,
                batch.segment_opponent_ids,
                batch.segment_opponent_digests,
            )
        ):
            rows = zip(
                batch.segment_game_ids,
                batch.segment_kinds,
                batch.segment_opponent_ids,
                batch.segment_opponent_digests,
                strict=True,
            )
        else:
            rows = (
                (game_id, batch.kind, opponent_id, digest)
                for game_id, opponent_id, digest in zip(
                    batch.game_ids,
                    batch.opponent_ids,
                    batch.opponent_digests or (None,) * len(batch.game_ids),
                    strict=True,
                )
            )
        for game_id, kind, opponent_id, digest in rows:
            if kind is BatchKind.MIXED:
                continue
            self.games[kind.value].add(game_id)
            self.opponents[opponent_id].add((game_id, kind.value, digest))

    def _record_masks(self, batch: LearnerBatch) -> None:
        """Accumulate legal-action density only across valid decision slots."""
        for segment in batch.segments:
            unit_valid = segment["unit_valid"].bool()
            quantity_valid = transfer_slots(segment["unit_actions"]) & unit_valid
            masks = {
                "operation": (segment["unit_masks"], unit_valid),
                "quantity": (segment["unit_quantity_masks"], quantity_valid),
                "market": (
                    segment["market_masks"],
                    torch.ones_like(segment["market_masks"][..., 0], dtype=torch.bool),
                ),
            }
            for name, (mask, valid) in masks.items():
                selected = mask[valid]
                mask_sum = selected.detach().float().sum()
                mask_count = torch.tensor(
                    selected.numel(), dtype=torch.int64, device=mask.device
                )
                self.mask_sums[name] = (
                    self.mask_sums.get(name, mask_sum.new_zeros(())) + mask_sum
                )
                self.mask_counts[name] = (
                    self.mask_counts.get(name, mask_count.new_zeros(())) + mask_count
                )

    def _record_state(self, batch: LearnerBatch) -> None:
        """Capture recurrent boundary magnitudes and actual terminal resets."""
        for segment in batch.segments:
            for name, field in (("hidden", "initial_hidden"), ("cell", "initial_cell")):
                state = segment.get(field)
                if state is not None:
                    self.state_norms[name].append(state.detach().float().norm())
            self.terminal_resets.append(segment["dones"].detach().to(torch.int64).sum())

    def compute(self) -> dict[str, torch.Tensor | float | int]:
        """Return the stable metric schema for the current completed round."""
        record: dict[str, torch.Tensor | float | int] = dict(self.initial)
        collection_seconds = float(
            self.initial.get("throughput/collection_seconds", float("nan"))
        )
        record["throughput/collection_seconds"] = collection_seconds
        record["throughput/collection_wait_seconds"] = collection_seconds
        record["throughput/collection_steps_per_second"] = (
            self.collected_steps / collection_seconds
            if math.isfinite(collection_seconds) and collection_seconds > 0
            else float("nan")
        )
        record["throughput/learner_seconds"] = self.learner_seconds
        record["throughput/learner_steps_per_second"] = (
            self.collected_steps / self.learner_seconds
            if self.learner_seconds > 0
            else float("nan")
        )
        record["throughput/learner_batches"] = self.learner_batches
        for kind in _ONLINE_BATCH_KINDS:
            count = len(self.games[kind.value])
            record[f"collection/games/{kind.value}"] = count
            record[f"population/selections/{kind.value}"] = count
        for opponent_id, selections in self.opponents.items():
            safe_id = opponent_id.replace("/", "_")
            record[f"population/opponent/{safe_id}"] = len(selections)
        for name, values in self.losses.items():
            record[name] = torch.stack(values).mean()
        for name in ("operation", "quantity", "market"):
            mask_sum = self.mask_sums.get(name)
            mask_count = self.mask_counts.get(name)
            if mask_sum is None or mask_count is None or not bool(mask_count > 0):
                record[f"mask/{name}_density"] = float("nan")
            else:
                record[f"mask/{name}_density"] = mask_sum / mask_count.float()
        for name in ("hidden", "cell"):
            values = self.state_norms[name]
            record[f"state/{name}_norm"] = torch.stack(values).mean() if values else 0.0
        record["state/terminal_resets"] = (
            torch.stack(self.terminal_resets).sum().float()
            if self.terminal_resets
            else 0.0
        )
        return record

    def reducible_values(  # noqa: C901
        self, device: torch.device
    ) -> dict[str, ReducibleMetric]:
        """Return every global metric as a numerator/count or explicit total."""
        values: dict[str, ReducibleMetric] = {}

        def add(
            name: str,
            total: torch.Tensor | float | int,
            count: int | torch.Tensor,
            reduction: Literal["mean", "sum", "max"] = "mean",
        ) -> None:
            values[name] = ReducibleMetric(
                total=torch.as_tensor(total, dtype=torch.float32, device=device),
                count=torch.as_tensor(count, dtype=torch.int64, device=device),
                reduction=reduction,
            )

        for name, raw in self.initial.items():
            if name == "throughput/collection_seconds":
                continue
            value = float(raw)
            finite = math.isfinite(value)
            reduction: Literal["mean", "sum", "max"] = "mean"
            total: float = value
            count = int(finite)
            opponent_return = "collection/return_by_opponent/"
            if name.startswith(opponent_return):
                suffix = name.removeprefix(opponent_return)
                total = float(
                    self.initial.get(
                        f"collection/return_sum_by_opponent/{suffix}", value
                    )
                )
                count = int(
                    self.initial.get(f"collection/return_count_by_opponent/{suffix}", 1)
                )
            if (
                name.startswith("collection/games/")
                or name.startswith("collection/games_by_opponent/")
                or name.startswith("collection/return_sum_by_opponent/")
                or name.startswith("collection/return_count_by_opponent/")
                or name.startswith("diag/n_")
                or name == "diag/illegal"
            ):
                reduction = "sum"
            elif name.endswith("_max"):
                reduction = "max"
            add(
                name,
                total if finite else (float("-inf") if reduction == "max" else 0.0),
                count if finite else 0,
                reduction,
            )
        for kind in _ONLINE_BATCH_KINDS:
            count = len(self.games[kind.value])
            add(f"collection/games/{kind.value}", count, 1, "sum")
            add(f"population/selections/{kind.value}", count, 1, "sum")
        for opponent_id, selections in self.opponents.items():
            safe_id = opponent_id.replace("/", "_")
            add(f"population/opponent/{safe_id}", len(selections), 1, "sum")
        for name, tensors in self.losses.items():
            if tensors:
                add(name, torch.stack(tensors).sum(), len(tensors))
        for name in ("operation", "quantity", "market"):
            mask_total = self.mask_sums.get(name)
            mask_count = self.mask_counts.get(name)
            if mask_total is not None and mask_count is not None:
                add(f"mask/{name}_density", mask_total, mask_count)
        for name in ("hidden", "cell"):
            tensors = self.state_norms[name]
            if tensors:
                add(f"state/{name}_norm", torch.stack(tensors).sum(), len(tensors))
            else:
                add(f"state/{name}_norm", 0.0, 1)
        resets = (
            torch.stack(self.terminal_resets).sum()
            if self.terminal_resets
            else torch.tensor(0.0, device=device)
        )
        add("state/terminal_resets", resets, 1, "sum")
        add("throughput/learner_batches", self.learner_batches, 1, "sum")
        return values

    def local_timings(self) -> dict[str, float]:
        """Return process-local timings that must retain their rank identity."""
        return {
            "collection_seconds": float(
                self.initial.get("throughput/collection_seconds", float("nan"))
            ),
            "learner_seconds": self.learner_seconds,
        }


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


def policy_from_checkpoint(
    path: Path, *, legacy_value_bound: float | None = 1.0
) -> PolicyLike:
    """Reconstruct and strictly load the policy described by ``path``.

    Native Lightning checkpoints carry the complete :class:`ToadConfig`; that
    stored topology is authoritative, including optional stateful components
    whose architecture cannot be inferred safely from tensor shapes. Historical
    bare and ``learner`` envelopes remain readable as legacy control policies.

    Args:
        path: Native Lightning or retained historical policy checkpoint.
        legacy_value_bound: Non-parameter value transform for historical
            checkpoints, which did not serialize a model configuration.

    Returns:
        A strictly loaded policy in evaluation mode.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("policy checkpoint must contain a mapping")

    toad_state = checkpoint.get("toad")
    if toad_state is not None:
        state = _checkpoint_mapping(toad_state)
        if "config" not in state:
            raise ResumeConfigError("native Toad checkpoint has no stored config")
        model = validate_stored_config(state["config"]).model
        policy: PolicyLike = (
            StatefulPolicy(model)
            if uses_stateful_policy(model)
            else Policy(
                blocks=model.blocks,
                channels=model.channels,
                value_bound=model.value_bound,
                kernel_size=model.kernel_size,
                activation=model.activation,
            )
        )
    else:
        weights, _layout = _checkpoint_policy_weights(path)
        stem = weights.get("stem.weight")
        if not isinstance(stem, torch.Tensor) or stem.ndim != 4:
            raise ValueError("legacy policy checkpoint has no valid stem.weight")
        block_indices = {
            int(name.split(".", 2)[1])
            for name in weights
            if isinstance(name, str)
            and name.startswith("blocks.")
            and len(name.split(".", 2)) == 3
            and name.split(".", 2)[1].isdigit()
        }
        if not block_indices:
            raise ValueError("legacy policy checkpoint has no residual blocks")
        policy = Policy(
            blocks=max(block_indices) + 1,
            channels=int(stem.shape[0]),
            value_bound=legacy_value_bound,
            kernel_size=int(stem.shape[-1]),
        )

    load_checkpoint_policy(policy, path)
    return policy.eval()


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


_RESUME_OVERRIDE_FIELDS = frozenset(
    {
        "runtime.accelerator",
        "runtime.devices",
        "runtime.num_nodes",
        "runtime.strategy",
        "runtime.log_every_n_steps",
        "runtime.profiler",
        "runtime.output_dir",
        "runtime.resume",
        # Initialization provenance is consumed before the authoritative
        # checkpoint exists and cannot affect any resumed assignment or update.
        "curriculum.warm_start_checkpoint",
    }
)


def _config_leaves(value: object, prefix: str = "") -> dict[str, object]:
    """Flatten a resolved JSON config so new fields are compared by default."""
    if not isinstance(value, Mapping):
        return {prefix: value}
    leaves: dict[str, object] = {}
    for name, child in value.items():
        path = f"{prefix}.{name}" if prefix else str(name)
        leaves.update(_config_leaves(child, path))
    return leaves


def assert_resume_compatible(effective: ToadConfig, stored: ToadConfig) -> None:
    """Compare the complete resolved config except proven operational overrides."""
    stored_leaves = _config_leaves(stored.model_dump(mode="json"))
    effective_leaves = _config_leaves(effective.model_dump(mode="json"))
    paths = (set(stored_leaves) | set(effective_leaves)) - _RESUME_OVERRIDE_FIELDS
    differences = {
        path: (stored_leaves.get(path), effective_leaves.get(path))
        for path in sorted(paths)
        if stored_leaves.get(path) != effective_leaves.get(path)
    }
    if differences:
        rendered = ", ".join(
            f"{path}: stored={before!r}, effective={after!r}"
            for path, (before, after) in differences.items()
        )
        raise ResumeConfigError(f"resume config mismatch: {rendered}")


def compute_loss(  # noqa: C901
    policy: PolicyLike | torch.nn.Module,
    batch: LearnerBatch,
    config: ToadConfig,
    teacher: LoadedTeacher | Teacher | None = None,
    *,
    baseline_only: bool | None = None,
    entropy_state: Mapping[str, EntropyControllerState] | None = None,
    _raise_on_nonfinite: bool = True,
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
    behaviour = stacked("log_probs").float()
    rewards = stacked(config.curriculum.reward_field).float()
    dones = stacked("dones")

    turns, width = behaviour.shape
    eager_policy = unwrap_compiled(policy)
    stateful_policy = eager_policy if isinstance(eager_policy, StatefulPolicy) else None
    initial_check = _FiniteTensorCheck((), ())
    if stateful_policy is not None:
        if stateful_policy.config.recurrent or stateful_policy.config.belief:
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
        initial_inputs = (
            {
                "input/initial_hidden": initial.hidden.float(),
                "input/initial_cell": initial.cell.float(),
                "input/initial_belief": initial.prior_belief.float(),
            }
            if initial is not None
            else {}
        )
        initial_check = _FiniteTensorCheck.from_tensors(initial_inputs)
        initial_nonfinite = (
            initial_check.nonfinite_names() if _raise_on_nonfinite else []
        )
        if initial_nonfinite and _raise_on_nonfinite:
            raise NonFiniteTrainingError.from_batch(
                initial_nonfinite, batch, initial, config.runtime.precision
            )
        output = policy(
            board,
            scalars,
            positions,
            state=initial,
            dones=replay_dones,
        )
    else:
        legacy_output = policy(
            board.flatten(0, 1), scalars.flatten(0, 1), positions.flatten(0, 1)
        )
        output = PolicyOutput(*legacy_output, belief_logits=None, state=None)
    policy_terms = fp32_policy_terms(output, batch)
    unit_logits = policy_terms.unit_logits
    quantity_logits = policy_terms.quantity_logits
    market_logits = policy_terms.market_logits
    values = policy_terms.values
    loss_output = replace(
        output,
        unit_logits=unit_logits,
        quantity_logits=quantity_logits,
        market_logits=market_logits,
        values=values,
        belief_logits=policy_terms.belief_logits,
    )
    values = values.view(turns + 1, width)
    bootstrap_value = values[-1].detach()
    values = values[:-1]
    if stateful_policy is not None:
        unit_logits = unit_logits[:-1].flatten(0, 1)
        quantity_logits = quantity_logits[:-1].flatten(0, 1)
        market_logits = market_logits[:-1].flatten(0, 1)
    else:
        unit_logits = _acted(unit_logits, turns, width)
        quantity_logits = _acted(quantity_logits, turns, width)
        market_logits = _acted(market_logits, turns, width)

    belief_enabled = stateful_policy is not None and stateful_policy.config.belief
    belief_terms: dict[str, torch.Tensor] = {}
    if belief_enabled:
        belief_terms = _belief_loss_terms(loss_output, segments)
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
    teacher_outputs: dict[str, torch.Tensor] = {}
    teacher_kl = None
    if teacher is not None:
        declared = (
            {
                head: head in teacher.actual_heads
                for head in ("operation", "quantity", "market", "value")
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
        teacher_units = teacher_units.float()
        teacher_quantity = teacher_quantity.float()
        teacher_market = teacher_market.float()
        teacher_values = teacher_values.float()
        teacher_outputs = {
            "teacher/unit_logits": teacher_units,
            "teacher/quantity_logits": teacher_quantity,
            "teacher/market_logits": teacher_market,
            "teacher/values": teacher_values,
        }

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
            return per_step

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
            )
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
    checked = {
        "input/board": board,
        "input/scalars": scalars,
        "input/positions": positions,
        "input/unit_actions": unit_actions,
        "input/unit_quantity_actions": unit_quantity_actions,
        "input/market_actions": market_actions,
        "input/unit_masks": unit_masks,
        "input/unit_quantity_masks": unit_quantity_masks,
        "input/market_masks": market_masks,
        "input/behaviour_log_probs": behaviour,
        "input/rewards": rewards,
        "input/dones": dones,
        "unit_logits": policy_terms.unit_logits,
        "quantity_logits": policy_terms.quantity_logits,
        "market_logits": policy_terms.market_logits,
        "values": policy_terms.values,
        **teacher_outputs,
        **loss.intermediates,
        **(
            {"belief_logits": policy_terms.belief_logits}
            if policy_terms.belief_logits is not None
            else {}
        ),
        **terms,
    }
    state = output.state
    if state is not None:
        checked.update(
            {
                "output/state_hidden": state.hidden.float(),
                "output/state_cell": state.cell.float(),
                "output/state_prior_belief": state.prior_belief.float(),
            }
        )
    finite_check = initial_check.merged(_FiniteTensorCheck.from_tensors(checked))
    nonfinite = finite_check.nonfinite_names() if _raise_on_nonfinite else []
    if nonfinite and _raise_on_nonfinite:
        raise NonFiniteTrainingError.from_batch(
            nonfinite, batch, state, config.runtime.precision
        )
    provenance = NonFiniteProvenance.from_batch(batch, state, config.runtime.precision)
    return LossReport(
        total=total,
        terms=terms,
        entropy=head_entropy,
        debug_dtypes={
            "learner_log_probs": learner_log_probs.dtype,
            "importance_ratios": loss.intermediates.get(
                "importance_ratios", learner_log_probs
            ).dtype,
            "value_targets": loss.intermediates.get(
                "td_lambda/value_targets", values
            ).dtype,
            "entropy": entropy_term.dtype,
            "teacher": loss.teacher.dtype,
        },
        provenance=provenance,
        finite_check=finite_check,
    )


def round_decay(config: ToadConfig) -> Callable[[int], float]:
    """Return LR multiplier keyed only to reduced global environment steps."""
    total = config.runtime.total_environment_steps
    floor = config.optimizer.final_lr_multiplier

    def decay(environment_steps: int) -> float:
        return max(1.0 - environment_steps / total, floor)

    return decay


def _canonicalize_policy_checkpoint(
    module: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    local_metadata: dict[str, object],
) -> None:
    """Save compiled policy weights under the eager portable key layout."""
    del module, local_metadata
    canonicalize_policy_state_keys(state_dict, prefix=prefix)


def _prepare_policy_checkpoint_load(
    module: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    local_metadata: dict[str, object],
    strict: bool,
    missing_keys: list[str],
    unexpected_keys: list[str],
    error_messages: list[str],
) -> None:
    """Load canonical checkpoints into the current eager or compiled policy."""
    del local_metadata, strict, missing_keys, unexpected_keys, error_messages
    prepare_policy_state_keys_for_load(module, state_dict, prefix=prefix)


class ToadLightningModule(lightning.LightningModule):
    """Native Toad policy trained through Lightning automatic optimization."""

    def __init__(self, config: ToadConfig) -> None:
        super().__init__()
        self.config = config
        initial_policy: PolicyLike = (
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
        self.policy: torch.nn.Module = initial_policy
        self.warm_start_migration: WarmStartMigration | None = None
        if config.curriculum.warm_start_checkpoint is not None:
            self.warm_start_migration = initialize_policy_from_checkpoint(
                initial_policy, config.curriculum.warm_start_checkpoint
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
        self.round_local_steps = 0
        self.collection_round = 0
        self.actor_version = 0
        self.actor_source_global_step = 0
        self.population_manifest = SnapshotManifest()
        self._authoritative_checkpoint_identity: tuple[int, str] | None = None
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
        self._round_population = _empty_population_counts()
        self.round_metrics = RoundMetricAccumulator()
        self._round_reduced_metrics: dict[str, ReducibleMetric] = {}
        self._round_reduced_terms: dict[str, torch.Tensor] = {}
        self._round_reduced_entropy: dict[str, EntropyStat] = {}
        self._round_reduced_baseline = torch.tensor(float("nan"))
        self._round_rank_timings: dict[str, float] = {}
        self._round_global_steps = 0
        self._learner_started: float | None = None
        self._fit_started_monotonic: float | None = None
        self._pending_round_flush = False
        self._last_finite_provenance: NonFiniteProvenance | None = None
        self.register_state_dict_post_hook(_canonicalize_policy_checkpoint)
        self.register_load_state_dict_pre_hook(_prepare_policy_checkpoint_load)
        self.save_hyperparameters(self.config.model_dump(mode="json"))

    def train(self, mode: bool = True) -> Self:
        """Change learner mode while keeping the frozen teacher in evaluation."""
        super().train(mode)
        if self.teacher_policy is not None:
            self.teacher_policy.eval()
        return self

    def on_fit_start(self) -> None:
        """Start the process-local elapsed clock for this Lightning fit."""
        self._fit_started_monotonic = time.monotonic()

    def _fit_elapsed_hours(self) -> float:
        """Return process-local fit hours, which intentionally reset on resume."""
        if self._fit_started_monotonic is None:
            return 0.0
        return max(time.monotonic() - self._fit_started_monotonic, 0.0) / 3_600.0

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
            self._round_population = _empty_population_counts()
            self.round_metrics.reset(batch.round_metrics)
            self._round_reduced_metrics = {}
            self._round_reduced_terms = {}
            self._round_reduced_entropy = {}
            self._round_rank_timings = {}
            self._round_global_steps = 0
        baseline_only = batch.baseline_only or self.warmup_remaining > 0
        if self._learner_started is None:
            self._learner_started = time.perf_counter()
        report = compute_loss(
            self.policy,
            batch,
            self.config,
            teacher=self.teacher,
            baseline_only=baseline_only,
            entropy_state=self.entropy_state,
            _raise_on_nonfinite=False,
        )
        self._last_finite_provenance = report.provenance
        self._synchronize_nonfinite(report.finite_check, report.provenance)
        self.round_metrics.update(batch, report)
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
        self.round_local_steps += batch.collected_steps
        if batch.end_of_round:
            self._pending_round_flush = True
        if self._trainer is None:
            self._finish_learner_batch()
        return report.total

    def compute_report(self, batch: LearnerBatch) -> LossReport:
        """Compute a report for tests and diagnostics without mutating module clocks."""
        return compute_loss(
            self.policy,
            batch,
            self.config,
            teacher=self.teacher,
            entropy_state=self.entropy_state,
        )

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Optimizer | LightningOptimizer,
        optimizer_closure: Callable[[], Any] | None = None,
    ) -> None:
        """Close learner timing after backward, clipping, sync, and optimizer."""
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        self._last_finite_provenance = None
        self._finish_learner_batch()

    def on_after_backward(self) -> None:
        """Fail before an optimizer step when otherwise-finite gradients overflow.

        The stored provenance contains only detached primitives.  A later DDP
        boundary can all-reduce this local flag and render the same structured
        error on every rank before any rank enters its optimizer step.
        """
        gradients = {
            f"grad/{name}": parameter.grad
            for name, parameter in unwrap_compiled(self.policy).named_parameters()
            if parameter.grad is not None
        }
        finite_check = _FiniteTensorCheck.from_tensors(gradients)
        if self._last_finite_provenance is not None:
            self._synchronize_nonfinite(finite_check, self._last_finite_provenance)

    def _reduce_tensor(
        self, value: torch.Tensor, reduce_op: Literal["sum", "max"] = "sum"
    ) -> torch.Tensor:
        """Reduce through Lightning only when a distributed trainer owns us."""
        if self._trainer is None or int(getattr(self.trainer, "world_size", 1)) == 1:
            return value
        return cast(
            torch.Tensor,
            self.trainer.strategy.reduce(value, reduce_op=reduce_op),
        )

    def _synchronize_nonfinite(
        self,
        check: _FiniteTensorCheck | list[str],
        provenance: NonFiniteProvenance,
    ) -> None:
        """Make every rank gather and raise the same finite-check failure."""
        local_bad = (
            check.local_bad(device=self.device)
            if isinstance(check, _FiniteTensorCheck)
            else torch.tensor(bool(check), device=self.device)
        ).to(
            dtype=torch.int32,
        )
        any_bad = self._reduce_tensor(local_bad, reduce_op="max")
        if not bool(any_bad.item()):
            return
        tensor_names = (
            check.nonfinite_names() if isinstance(check, _FiniteTensorCheck) else check
        )
        rank = (
            0 if self._trainer is None else int(getattr(self.trainer, "global_rank", 0))
        )
        world_size = (
            1 if self._trainer is None else int(getattr(self.trainer, "world_size", 1))
        )
        local_detail: dict[str, object] | None = None
        if tensor_names:
            local_detail = {
                "rank": rank,
                "tensor_names": tuple(tensor_names),
                "provenance": provenance.as_dict(),
            }
        gathered = _all_gather_objects(local_detail, world_size, allow_none=True)
        details = tuple(
            cast(Mapping[str, object] | None, detail) for detail in gathered
        )
        raise NonFiniteTrainingError.render_distributed(details, provenance)

    def _finish_learner_batch(self) -> None:
        """Record one full optimization interval and flush a due round once."""
        if self._learner_started is None:
            raise RuntimeError("learner timing finished without a started batch")
        self.round_metrics.finish_learner_batch(
            time.perf_counter() - self._learner_started
        )
        self._learner_started = None
        if self._pending_round_flush:
            self._finalize_round()
            self.flush_round_metrics()

    def _finalize_round(self) -> None:
        """Advance global sample/controller clocks from the all-rank boundary."""
        local = torch.tensor(
            self.round_local_steps,
            dtype=torch.int64,
            device=self.device,
        )
        global_delta = self._reduce_tensor(local, reduce_op="sum")
        self.environment_steps += int(global_delta.item())
        self._round_global_steps = int(global_delta.item())
        self.round_local_steps = 0
        self.collection_round += 1
        self._reduce_round_statistics()
        self._update_entropy_controllers()

    def _reduce_round_statistics(self) -> None:
        """Reduce all global metric numerators/counts before forming means."""
        local_metrics = self.round_metrics.reducible_values(self.device)
        world_size = (
            1 if self._trainer is None else int(getattr(self.trainer, "world_size", 1))
        )
        gathered_specs = _all_gather_objects(
            {name: metric.reduction for name, metric in local_metrics.items()},
            world_size,
        )
        specs: dict[str, Literal["mean", "sum", "max"]] = {}
        for payload in gathered_specs:
            if not isinstance(payload, Mapping):
                raise RuntimeError("distributed metric schema is malformed")
            for name, reduction in payload.items():
                if not isinstance(name, str) or reduction not in {"mean", "sum", "max"}:
                    raise RuntimeError("distributed metric schema is malformed")
                typed_reduction = cast(Literal["mean", "sum", "max"], reduction)
                if name in specs and specs[name] != typed_reduction:
                    raise RuntimeError(f"distributed metric mode differs for {name}")
                specs[name] = typed_reduction
        reduced_metrics: dict[str, ReducibleMetric] = {}
        for name in sorted(specs):
            reduction = specs[name]
            local = local_metrics.get(name)
            neutral = float("-inf") if reduction == "max" else 0.0
            total = (
                torch.tensor(neutral, device=self.device)
                if local is None
                else local.total.to(self.device)
            )
            count = (
                torch.tensor(0, dtype=torch.int64, device=self.device)
                if local is None
                else local.count.to(self.device)
            )
            reduced_metrics[name] = ReducibleMetric(
                total=self._reduce_tensor(
                    total,
                    reduce_op="max" if reduction == "max" else "sum",
                ),
                count=self._reduce_tensor(count, reduce_op="sum"),
                reduction=reduction,
            )
        self._round_reduced_metrics = reduced_metrics
        self._round_reduced_terms = self._reduce_named_means(self._round_fresh_terms)
        baseline = self._reduce_named_means(
            ({"baseline_pass": value} for value in self._round_baselines)
        )
        self._round_reduced_baseline = baseline.get(
            "baseline_pass", torch.tensor(float("nan"), device=self.device)
        )
        self._round_reduced_entropy = {
            name: EntropyStat(
                sum=self._reduce_tensor(
                    self._aggregate_round_entropy(name).sum.float(), reduce_op="sum"
                ),
                valid=self._reduce_tensor(
                    self._aggregate_round_entropy(name).valid.to(torch.int64),
                    reduce_op="sum",
                ),
            )
            for name in ("operation", "quantity", "market")
        }
        self._round_population = {
            name: int(
                self._reduce_tensor(
                    torch.tensor(value, dtype=torch.int64, device=self.device),
                    reduce_op="sum",
                ).item()
            )
            for name, value in self._round_population.items()
        }
        local_timing: dict[str, object] = {
            "rank": (
                0
                if self._trainer is None
                else int(getattr(self.trainer, "global_rank", 0))
            ),
            **self.round_metrics.local_timings(),
        }
        gathered_timings = _all_gather_objects(local_timing, world_size)
        rank_timings: dict[str, float] = {}
        for payload in gathered_timings:
            if not isinstance(payload, Mapping):
                raise RuntimeError("distributed timing payload is malformed")
            timing = cast(Mapping[str, object], payload)
            rank = int(cast(int, timing["rank"]))
            for name in ("collection_seconds", "learner_seconds"):
                rank_timings[f"rank/{rank}/throughput/{name}"] = float(
                    cast(float, timing[name])
                )
        self._round_rank_timings = rank_timings

    def _reduce_named_means(
        self, rows: Iterable[Mapping[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """Reduce named detached totals and counts, tolerating absent optional keys."""
        materialized = tuple(rows)
        local_names = sorted({name for row in materialized for name in row})
        world_size = (
            1 if self._trainer is None else int(getattr(self.trainer, "world_size", 1))
        )
        gathered_names = _all_gather_objects(tuple(local_names), world_size)
        names = sorted(
            {
                str(name)
                for payload in gathered_names
                for name in cast(Sequence[object], payload)
            }
        )
        reduced: dict[str, torch.Tensor] = {}
        for name in names:
            values = [row[name].detach().float() for row in materialized if name in row]
            total = (
                torch.stack(values).sum()
                if values
                else torch.tensor(0.0, device=self.device)
            )
            count = torch.tensor(len(values), dtype=torch.int64, device=self.device)
            global_total = self._reduce_tensor(total, reduce_op="sum")
            global_count = self._reduce_tensor(count, reduce_op="sum")
            reduced[name] = (
                global_total / global_count.float()
                if bool(global_count > 0)
                else torch.tensor(float("nan"), device=self.device)
            )
        return reduced

    def flush_round_metrics(self) -> None:
        """Log and clear the sole completed-round accumulator without DDP sync."""
        self.log_dict(
            self._round_log_record(),
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )
        self._pending_round_flush = False
        self._round_fresh_entropy = []
        self.round_metrics.reset()

    def _aggregate_round_entropy(self, name: str) -> EntropyStat:
        """Reduce detached fresh-batch statistics for one completed round."""
        if name in self._round_reduced_entropy:
            return self._round_reduced_entropy[name]
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

        def mean_term(name: str) -> torch.Tensor:
            reduced = self._round_reduced_terms.get(name)
            if reduced is not None:
                return reduced
            local = [
                row[name].detach().float()
                for row in self._round_fresh_terms
                if name in row
            ]
            return (
                torch.stack(local).mean()
                if local
                else torch.tensor(float("nan"), device=self.device)
            )

        optimizer_steps = int(self.global_step) + 1
        try:
            lr = float(self.trainer.optimizers[0].param_groups[0]["lr"])
        except RuntimeError:
            lr = float(self.config.optimizer.lr)
        reduced_metrics = {
            name: (
                metric.value
                if bool(metric.count > 0)
                else torch.tensor(float("nan"), device=self.device)
            )
            for name, metric in self._round_reduced_metrics.items()
        }
        collection_timings = [
            value
            for name, value in self._round_rank_timings.items()
            if name.endswith("/collection_seconds") and math.isfinite(value)
        ]
        learner_timings = [
            value
            for name, value in self._round_rank_timings.items()
            if name.endswith("/learner_seconds") and math.isfinite(value)
        ]
        collection_seconds = (
            max(collection_timings) if collection_timings else float("nan")
        )
        learner_seconds = max(learner_timings) if learner_timings else float("nan")
        record: dict[str, torch.Tensor | int | float] = {
            **reduced_metrics,
            **self._round_rank_timings,
            "throughput/collection_seconds": collection_seconds,
            "throughput/collection_wait_seconds": collection_seconds,
            "throughput/collection_steps_per_second": (
                self._round_global_steps / collection_seconds
                if math.isfinite(collection_seconds) and collection_seconds > 0
                else float("nan")
            ),
            "throughput/learner_seconds": learner_seconds,
            "throughput/learner_steps_per_second": (
                self._round_global_steps / learner_seconds
                if math.isfinite(learner_seconds) and learner_seconds > 0
                else float("nan")
            ),
            "diag/update": self.collection_round,
            "diag/steps": self.environment_steps,
            "diag/hours": self._fit_elapsed_hours(),
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
            "critic/baseline_passes_self_consistency": (
                self._round_reduced_baseline
                if self._round_reduced_terms
                else (
                    torch.stack(self._round_baselines).mean()
                    if self._round_baselines
                    else torch.tensor(float("nan"), device=self.device)
                )
            ),
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
            "belief/loss": mean_term("belief"),
            "progress/environment_steps": float(self.environment_steps),
            "progress/optimizer_steps": float(optimizer_steps),
            "actor/version": float(self.actor_version),
            "actor/lag_optimizer_steps": float(
                optimizer_steps - self.actor_source_global_step
            ),
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
        """Set LR from the reduced global environment boundary once per round."""
        del metric
        if self._round_ended:
            scheduler.step(self.environment_steps)

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
            "population_manifest": self.population_manifest.model_dump(mode="json"),
            "teacher": self._teacher_metadata(),
        }

    def _teacher_metadata(self) -> dict[str, object]:
        """Return structural teacher semantics that must survive resume."""
        return {"present": False} if self.teacher is None else self.teacher.metadata()

    def on_load_checkpoint(self, checkpoint: dict[str, object]) -> None:
        """Validate structure and restore Toad's logical clocks."""
        try:
            state = _checkpoint_mapping(checkpoint["toad"])
            stored_config = validate_stored_config(state["config"])
            assert_resume_compatible(self.config, stored_config)
            stored_teacher = _checkpoint_mapping(state["teacher"])
            current_teacher = self._teacher_metadata()
            if stored_teacher != current_teacher:
                raise ResumeConfigError(
                    "teacher metadata mismatch: "
                    f"stored={stored_teacher!r}, effective={current_teacher!r}"
                )

            environment_steps = _checkpoint_nonnegative_int(state["environment_steps"])
            collection_round = _checkpoint_nonnegative_int(state["collection_round"])
            actor_version = _checkpoint_nonnegative_int(state["actor_version"])
            actor_source_global_step = _checkpoint_nonnegative_int(
                state["actor_source_global_step"]
            )
            warmup_remaining = _checkpoint_nonnegative_int(state["warmup_remaining"])
            population_manifest = SnapshotManifest.model_validate(
                state.get("population_manifest", {})
            )
            stored_entropy = _checkpoint_mapping(state["entropy_state"])
            if set(stored_entropy) != _ENTROPY_HEADS:
                raise ResumeConfigError(_INVALID_CHECKPOINT_STATE)

            restored_entropy: dict[str, EntropyControllerState] = {}
            for name, controller in self.config.optimizer.entropy.items():
                payload = _checkpoint_mapping(stored_entropy[name])
                if set(payload) != _ENTROPY_STATE_FIELDS:
                    raise ResumeConfigError(_INVALID_CHECKPOINT_STATE)
                target = _checkpoint_finite_float(payload["target"])
                multiplier = _checkpoint_finite_float(payload["multiplier"])
                last_steps = _checkpoint_nonnegative_int(payload["last_steps"])
                is_disabled_initial_target = (
                    not self.config.optimizer.adaptive_entropy
                    and last_steps == 0
                    and target == controller.initial_target
                )
                if (
                    (
                        target < controller.target_floor
                        and not is_disabled_initial_target
                    )
                    or not controller.minimum <= multiplier <= controller.maximum
                    or last_steps > environment_steps
                    or (
                        self.config.optimizer.adaptive_entropy
                        and last_steps != environment_steps
                    )
                ):
                    raise ResumeConfigError(_INVALID_CHECKPOINT_STATE)
                restored_entropy[name] = EntropyControllerState(
                    target=target,
                    multiplier=multiplier,
                    last_steps=last_steps,
                )
        except ResumeConfigError:
            raise
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ResumeConfigError(_INVALID_CHECKPOINT_STATE) from error

        self.environment_steps = environment_steps
        self.collection_round = collection_round
        self.actor_version = actor_version
        self.actor_source_global_step = actor_source_global_step
        self.warmup_remaining = warmup_remaining
        self.population_manifest = population_manifest
        self.entropy_state = restored_entropy
