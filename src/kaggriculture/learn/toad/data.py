"""Synchronous, typed collection batches for the reference Toad backend."""

from __future__ import annotations

import math
import random
import time
import traceback
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import lightning
import torch
from torch.utils.data import DataLoader, IterableDataset

from kaggriculture.constants import EPISODE_STEPS
from kaggriculture.learn.encoding import IGNORE
from kaggriculture.learn.rollout import Trajectory, segment_starts
from kaggriculture.learn.toad.config import (
    ModelConfig,
    ToadConfig,
    structural_fingerprint,
)
from kaggriculture.learn.toad.population import (
    EmptySnapshotPoolError,
    SnapshotEntry,
    SnapshotIntegrityError,
    SnapshotManifest,
    SnapshotPool,
    SnapshotPoolIdentity,
    SnapshotStore,
    TeacherCompatibilityError,
    load_teacher,
    load_teacher_from_state,
    resolve_teacher_model,
    sha256_file,
)
from kaggriculture.learn.toad_loss import UNROLL_LENGTH

if TYPE_CHECKING:
    from kaggriculture.sim.rollout import Trajectory as NativeTrajectory

# Segment fields are defined beside the public segmenter so all producers share
# the exact one-bootstrap-row contract.
ACTED_FIELDS = (
    "unit_actions",
    "unit_quantities",
    "market_actions",
    "unit_masks",
    "unit_quantity_masks",
    "market_masks",
    "log_probs",
    "shaped",
    "shaped_money",
    "margin",
    "own",
    "sparse",
    "dones",
)
BELIEF_FIELDS = ("belief_targets", "belief_valid")
OBSERVED_FIELDS = ("board", "scalars", "positions")


@dataclass(frozen=True)
class ReferenceWorkerInput:
    """Resolved architecture and weights for one reference collector worker."""

    actor_state: dict[str, torch.Tensor]
    seeds: list[int]
    model: ModelConfig
    versus: str | None
    money_weight: float
    unroll_length: int = UNROLL_LENGTH
    opponent_state: dict[str, torch.Tensor] | None = None
    opponent_model: ModelConfig | None = None


class BatchKind(StrEnum):
    """Which policy population supplied a learner batch."""

    SELFPLAY = "selfplay"
    SCRIPTED = "scripted"
    FROZEN_OPPONENT = "frozen_opponent"
    TEACHER_DISTILL = "teacher_distill"
    MIXED = "mixed"


@dataclass(frozen=True)
class RoundMeta:
    """Immutable provenance for one collected logical round."""

    round_id: int
    actor_version: int
    game_ids: tuple[int, ...]
    seeds: tuple[int, ...]
    opponent_ids: tuple[str, ...]
    kind: BatchKind
    opponent_digests: tuple[str | None, ...] = ()

    @classmethod
    def control(cls, round_id: int = 0) -> RoundMeta:
        """Return the fixed self-play provenance used by control tests."""
        return cls(
            round_id=round_id,
            actor_version=0,
            game_ids=(0,),
            seeds=(0,),
            opponent_ids=("self",),
            kind=BatchKind.SELFPLAY,
        )


@dataclass(frozen=True)
class LearnerBatch:
    """One optimizer-sized immutable handoff from collection to learning."""

    segments: tuple[dict[str, torch.Tensor], ...]
    kind: BatchKind
    baseline_only: bool
    first_of_round: bool
    end_of_round: bool
    collected_steps: int
    round_id: int
    actor_version: int
    game_ids: tuple[int, ...]
    opponent_ids: tuple[str, ...]
    segment_kinds: tuple[BatchKind, ...] = ()
    seeds: tuple[int, ...] = ()
    opponent_digests: tuple[str | None, ...] = ()
    segment_opponent_ids: tuple[str, ...] = ()
    segment_opponent_digests: tuple[str | None, ...] = ()
    segment_game_ids: tuple[int, ...] = ()
    segment_seeds: tuple[int, ...] = ()
    round_metrics: Mapping[str, float | int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze observation-derived unit padding for legacy segment inputs.

        New segments carry ``unit_valid`` directly. This narrow construction
        fallback admits historical fixtures once, before learner-side tests or
        transformations can edit action labels; entropy and action scoring
        never infer padding from those mutable labels.
        """
        normalized = []
        for segment in self.segments:
            if "unit_valid" in segment:
                normalized.append(segment)
                continue
            normalized.append(
                {
                    **segment,
                    "unit_valid": segment["unit_actions"] != IGNORE,
                }
            )
        object.__setattr__(self, "segments", tuple(normalized))


@dataclass(frozen=True, init=False)
class OpponentAssignment:
    """One game assigned to the synchronous reference collector."""

    game_id: int
    seed: int
    kind: BatchKind
    opponent_id: str
    checkpoint: Path | None = None
    checkpoint_sha256: str | None = None

    def __init__(
        self,
        game_id: int,
        seed: int,
        kind: BatchKind | str,
        opponent_id: str | BatchKind,
        checkpoint: Path | None = None,
        checkpoint_sha256: str | None = None,
    ) -> None:
        """Accept the public order plus the legacy opponent-before-kind order."""
        try:
            resolved_kind = BatchKind(kind)
        except ValueError:
            try:
                resolved_kind = BatchKind(opponent_id)
            except (TypeError, ValueError):
                raise ValueError(f"{kind!r} is not a valid BatchKind") from None
            kind, opponent_id = resolved_kind, kind
        if not isinstance(opponent_id, str):
            raise TypeError("opponent_id must be a string")
        object.__setattr__(self, "game_id", game_id)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "kind", resolved_kind)
        object.__setattr__(self, "opponent_id", opponent_id)
        object.__setattr__(self, "checkpoint", checkpoint)
        object.__setattr__(self, "checkpoint_sha256", checkpoint_sha256)


# Keep the foundation-stage spelling as an exact compatibility alias.
CollectionAssignment = OpponentAssignment


@dataclass(frozen=True)
class OpponentBinding:
    """Immutable neural-opponent identity, independent of game provenance."""

    kind: BatchKind
    checkpoint: Path
    sha256: str


_ONLINE_KINDS = (
    BatchKind.SELFPLAY,
    BatchKind.SCRIPTED,
    BatchKind.FROZEN_OPPONENT,
    BatchKind.TEACHER_DISTILL,
)


def rank_game_ids(
    start: int,
    per_rank: int,
    rank: int,
    world_size: int,
) -> tuple[int, ...]:
    """Return one rank's contiguous slice of a global round allocation."""
    if start < 0:
        raise ValueError("start must be nonnegative")
    if per_rank < 1:
        raise ValueError("per_rank must be positive")
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be in [0, world_size)")
    rank_start = start + rank * per_rank
    return tuple(range(rank_start, rank_start + per_rank))


@dataclass(frozen=True)
class RoundBatchCounts:
    """Deterministic policy/value optimizer quotas for one rank and round."""

    policy: int
    value: int

    @property
    def total(self) -> int:
        """Return the complete optimizer-batch count."""
        return self.policy + self.value


@dataclass(frozen=True)
class RoundBatchSignature:
    """One rank's actual completed collection shape before learner entry."""

    rank: int
    round_id: int
    expected: RoundBatchCounts
    policy: int
    value: int
    first_markers: int
    end_markers: int


class DistributedRoundMismatch(RuntimeError):  # noqa: N818
    """Ranks cannot safely enter DDP with different logical round shapes."""


class DistributedCollectionError(RuntimeError):
    """At least one rank failed before a collected round became safe to yield."""

    def __init__(self, details: Sequence[Mapping[str, object]]) -> None:
        self.distributed_details = tuple(details)
        rendered = "; ".join(
            f"rank={detail.get('rank')} {detail.get('type')}: "
            f"{detail.get('message')}\n{detail.get('traceback')}"
            for detail in self.distributed_details
        )
        super().__init__("distributed collection failed: " + rendered)


def expected_round_batch_counts(config: ToadConfig) -> RoundBatchCounts:
    """Derive one rank's exact optimizer quotas before collection starts."""
    counts = _round_counts(config)
    trajectories = sum(counts.values()) + counts[BatchKind.SELFPLAY]
    segments_per_trajectory = len(
        segment_starts(EPISODE_STEPS - 1, config.optimizer.unroll_length)
    )
    policy = (trajectories * segments_per_trajectory) // config.optimizer.batch_segments
    return RoundBatchCounts(
        policy=policy,
        value=policy * config.optimizer.value_passes,
    )


def round_batch_signature(
    batches: Sequence[LearnerBatch],
    *,
    expected: RoundBatchCounts,
    rank: int,
    round_id: int,
) -> RoundBatchSignature:
    """Describe actual batches and boundary markers without consuming tensors."""
    return RoundBatchSignature(
        rank=rank,
        round_id=round_id,
        expected=expected,
        policy=sum(not batch.baseline_only for batch in batches),
        value=sum(batch.baseline_only for batch in batches),
        first_markers=sum(batch.first_of_round for batch in batches),
        end_markers=sum(batch.end_of_round for batch in batches),
    )


def validate_distributed_round_signatures(
    signatures: Sequence[RoundBatchSignature],
) -> None:
    """Reject every quota, batch-count, round, and marker disagreement together."""
    if not signatures:
        raise DistributedRoundMismatch("distributed round has no rank signatures")
    expected_ranks = list(range(len(signatures)))
    if sorted(signature.rank for signature in signatures) != expected_ranks:
        raise DistributedRoundMismatch(
            f"distributed round ranks are incomplete: signatures={signatures!r}"
        )
    expected = signatures[0].expected
    round_id = signatures[0].round_id
    invalid = [
        signature
        for signature in signatures
        if signature.expected != expected
        or signature.round_id != round_id
        or signature.policy != expected.policy
        or signature.value != expected.value
        or signature.first_markers != 1
        or signature.end_markers != 1
    ]
    if invalid:
        rendered = "; ".join(
            "rank="
            f"{item.rank} round_id={item.round_id} expected={item.expected!r} "
            f"policy={item.policy} value={item.value} "
            f"first_markers={item.first_markers} end_markers={item.end_markers}"
            for item in signatures
        )
        raise DistributedRoundMismatch("distributed round mismatch: " + rendered)


def _all_gather_objects(
    local: object,
    world_size: int,
    *,
    allow_none: bool = False,
) -> tuple[object, ...]:
    """Gather one small boundary payload through the initialized process group."""
    if world_size == 1:
        return (local,)
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise DistributedRoundMismatch(
            "distributed round preflight requires an initialized process group"
        )
    gathered: list[object | None] = [None] * world_size
    torch.distributed.all_gather_object(gathered, local)
    if not allow_none and any(item is None for item in gathered):
        raise DistributedRoundMismatch("distributed round gather returned no payload")
    return tuple(gathered)


def _round_counts(config: ToadConfig) -> dict[BatchKind, int]:
    """Allocate exact environment quotas by deterministic largest remainder."""
    environments = config.population.environments_per_rank
    probabilities = {
        BatchKind.SELFPLAY: config.population.selfplay,
        BatchKind.SCRIPTED: config.population.scripted,
        BatchKind.FROZEN_OPPONENT: config.population.frozen_opponent,
        BatchKind.TEACHER_DISTILL: config.population.teacher_distill,
    }
    raw = {
        kind: probability * environments for kind, probability in probabilities.items()
    }
    counts = {kind: math.floor(value) for kind, value in raw.items()}
    remaining = environments - sum(counts.values())
    order = sorted(
        raw,
        key=lambda kind: (-(raw[kind] - counts[kind]), kind.value),
    )
    for kind in order[:remaining]:
        counts[kind] += 1
    if sum(counts.values()) != environments:
        raise AssertionError("population quotas must allocate every environment")
    return counts


def _teacher_identity(
    config: ToadConfig, teacher: SnapshotEntry | None
) -> tuple[Path, str, str]:
    """Resolve and hash one teacher checkpoint once for a whole allocation."""
    spec = config.population.teacher
    if spec is None:
        raise ValueError("teacher-distill allocation requires a checkpoint")
    teacher_path = spec.checkpoint
    teacher_digest = spec.sha256 or sha256_file(teacher_path)
    if teacher is not None and (
        teacher.path != teacher_path or teacher.sha256 != teacher_digest
    ):
        raise ValueError("runtime teacher identity does not match population.teacher")
    return (
        teacher_path,
        teacher_digest,
        f"teacher:{teacher_path.name}:{teacher_digest[:12]}",
    )


def _round_opponent(
    config: ToadConfig,
    kind: BatchKind,
    game_id: int,
    pool: SnapshotPool | None,
    teacher_identity: tuple[Path, str, str] | None,
) -> tuple[str, Path | None, str | None]:
    """Resolve one already-validated kind slot to an auditable opponent."""
    if kind is BatchKind.SELFPLAY:
        return "self", None, None
    if kind is BatchKind.SCRIPTED:
        return config.population.scripted_opponent, None, None
    if kind is BatchKind.FROZEN_OPPONENT:
        if pool is None:
            raise AssertionError("frozen quota requires a rebound collection pool")
        selected = pool.sample(game_id)
        opponent_id = (
            f"snapshot:{selected.run_id}:{selected.environment_steps}:"
            f"{selected.sha256[:12]}"
        )
        return opponent_id, selected.path, selected.sha256
    if teacher_identity is None:
        raise AssertionError("teacher quota requires a precomputed identity")
    checkpoint, digest, opponent_id = teacher_identity
    return opponent_id, checkpoint, digest


def allocate_round(
    config: ToadConfig,
    start_game_id: int,
    round_id: int,
    pool: SnapshotPool | None = None,
    teacher: SnapshotEntry | None = None,
) -> tuple[OpponentAssignment, ...]:
    """Allocate one deterministic, quota-exact collection round.

    Kind slots are shuffled before monotonically increasing game IDs are bound,
    preserving contiguous global identity while varying opponent order by round.
    """
    counts = _round_counts(config)

    kinds = [kind for kind in _ONLINE_KINDS for _ in range(counts[kind])]
    population_seed = config.runtime.seed + config.population.population_seed
    random.Random(population_seed ^ round_id).shuffle(kinds)

    collection_pool = pool
    if counts[BatchKind.FROZEN_OPPONENT]:
        if pool is None:
            raise EmptySnapshotPoolError(
                "frozen opponent requested before pool population"
            )
        collection_pool = pool.rebind(
            structure=structural_fingerprint(config),
            seed=config.population.population_seed,
        )

    teacher_identity: tuple[Path, str, str] | None = None
    if counts[BatchKind.TEACHER_DISTILL]:
        teacher_identity = _teacher_identity(config, teacher)

    assignments: list[OpponentAssignment] = []
    for offset, kind in enumerate(kinds):
        game_id = start_game_id + offset
        opponent_id, checkpoint, checkpoint_sha256 = _round_opponent(
            config, kind, game_id, collection_pool, teacher_identity
        )
        assignments.append(
            OpponentAssignment(
                game_id=game_id,
                seed=config.runtime.seed + game_id,
                opponent_id=opponent_id,
                kind=kind,
                checkpoint=checkpoint,
                checkpoint_sha256=checkpoint_sha256,
            )
        )
    return tuple(assignments)


class CollectionError(RuntimeError):
    """Collection failed while retaining exact affected-game provenance."""

    def __init__(
        self,
        message: str,
        *,
        game_ids: Sequence[int] = (),
        seeds: Sequence[int] = (),
    ) -> None:
        self.game_ids = tuple(game_ids)
        self.seeds = tuple(seeds)
        super().__init__(message)


class CollectionRowError(RuntimeError):
    """A grouped collector failure attributable to one vector row."""

    def __init__(self, row: int, error_type: str, message: str) -> None:
        self.row = row
        self.error_type = error_type
        self.original_message = message
        super().__init__(row, error_type, message)

    def __str__(self) -> str:
        """Render the stable worker-side row and original exception."""
        return f"row={self.row} {self.error_type}: {self.original_message}"


class PopulationResumeMigrationError(RuntimeError):
    """A legacy checkpoint cannot identify its required durable population."""


class CollectorCheckpointError(ValueError):
    """Collector checkpoint state is malformed or cannot name the next stream."""


_INVALID_COLLECTOR_STATE = "invalid collector checkpoint state"


def _checkpoint_nonnegative_int(value: object) -> int:
    """Accept only exact non-boolean nonnegative collector counters."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CollectorCheckpointError(_INVALID_COLLECTOR_STATE)
    return value


def _checkpoint_policy_state(path: Path) -> dict[str, torch.Tensor]:
    """Extract exact policy tensors from a bare, legacy, or Lightning checkpoint."""
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, Mapping):
        raise ValueError("policy checkpoint must contain a mapping")
    candidate: object
    if "state_dict" in loaded:
        state_dict = loaded["state_dict"]
        if not isinstance(state_dict, Mapping):
            raise ValueError("Lightning checkpoint state_dict must be a mapping")
        candidate = {
            name.removeprefix("policy."): value
            for name, value in state_dict.items()
            if isinstance(name, str) and name.startswith("policy.")
        }
        if not candidate:
            raise ValueError("Lightning checkpoint has no policy.* weights")
    elif "learner" in loaded:
        candidate = loaded["learner"]
    else:
        candidate = loaded
    if not isinstance(candidate, Mapping) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in candidate.items()
    ):
        raise ValueError("policy checkpoint must contain only named tensors")
    return cast(dict[str, torch.Tensor], dict(candidate))


def _strictly_validate_policy_state(
    model: ModelConfig, state: Mapping[str, torch.Tensor]
) -> None:
    """Prove checkpoint tensors exactly instantiate the resolved policy topology."""
    from kaggriculture.learn.model import Policy
    from kaggriculture.learn.toad.model import StatefulPolicy, uses_stateful_policy

    if uses_stateful_policy(model):
        policy: Policy | StatefulPolicy = StatefulPolicy(model)
    else:
        policy = Policy(
            blocks=model.blocks,
            channels=model.channels,
            value_bound=model.value_bound,
            kernel_size=model.kernel_size,
            activation=model.activation,
        )
    policy.load_state_dict(state, strict=True)


def _recurrent_state_rows(
    trajectory: Trajectory,
    recurrent_fields: tuple[torch.Tensor | None, ...],
    turns: int,
    starts: tuple[int, ...],
) -> dict[int, int] | None:
    """Validate dense or sparse recurrent storage and map acted steps to rows."""
    if trajectory.state_steps is None:
        if any(
            field is not None and field.shape[0] != turns + 1
            for field in recurrent_fields
        ):
            raise ValueError(
                "dense recurrent trajectories require one state per row "
                "plus trailing state"
            )
        return None
    if not all(field is not None for field in recurrent_fields):
        raise ValueError("state_steps require active recurrent trajectory state")
    state_steps = trajectory.state_steps
    if state_steps.ndim != 1 or state_steps.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise ValueError("state_steps must be a rank-1 integer tensor")
    listed = [int(step) for step in state_steps.tolist()]
    if listed != sorted(set(listed)) or any(
        step < 0 or step >= turns for step in listed
    ):
        raise ValueError("state_steps must be sorted unique acted-row indices")
    if any(
        field is not None and field.shape[0] != len(listed)
        for field in recurrent_fields
    ):
        raise ValueError("sparse recurrent fields must align exactly with state_steps")
    rows = {step: row for row, step in enumerate(listed)}
    missing = [start for start in starts if start not in rows]
    if missing:
        raise ValueError(
            f"missing recurrent state at required segment start {missing[0]}"
        )
    return rows


def segments(
    trajectory: Trajectory, unroll_length: int
) -> list[dict[str, torch.Tensor]]:
    """Chop an episode into end-anchored, bootstrap-carrying unrolls.

    The final transition is retained by dropping only the opening remainder.
    Observed fields include a trailing state for value bootstrapping; on the
    terminal unroll that state is clamped to the final observed row, whose done
    mask removes it from every return target.
    """
    turns = int(trajectory.dones.shape[0])
    belief_fields = (trajectory.belief_targets, trajectory.belief_valid)
    if any(field is not None for field in belief_fields) and not all(
        field is not None for field in belief_fields
    ):
        raise ValueError("belief trajectories require targets and validity")
    if any(field is not None and field.shape[0] != turns for field in belief_fields):
        raise ValueError("belief trajectories require one target and validity per turn")
    recurrent_fields = (
        trajectory.hidden,
        trajectory.cell,
        trajectory.prior_belief,
    )
    if any(field is not None for field in recurrent_fields) and not all(
        field is not None for field in recurrent_fields
    ):
        raise ValueError(
            "recurrent trajectories require hidden, cell, and prior_belief"
        )
    starts = segment_starts(turns, unroll_length)
    state_rows = _recurrent_state_rows(trajectory, recurrent_fields, turns, starts)
    return [
        {
            **{
                name: getattr(trajectory, name)[start : start + unroll_length]
                for name in ACTED_FIELDS
            },
            "unit_valid": trajectory.unit_actions[start : start + unroll_length]
            != IGNORE,
            **{
                name: getattr(trajectory, name)[start : start + unroll_length]
                for name in BELIEF_FIELDS
                if getattr(trajectory, name) is not None
            },
            **{
                name: getattr(trajectory, name)[
                    torch.arange(start, start + unroll_length + 1).clamp(max=turns - 1)
                ]
                for name in OBSERVED_FIELDS
            },
            **(
                {
                    "initial_hidden": trajectory.hidden[
                        start if state_rows is None else state_rows[start]
                    ]
                    .detach()
                    .clone(),
                    "initial_cell": trajectory.cell[
                        start if state_rows is None else state_rows[start]
                    ]
                    .detach()
                    .clone(),
                    "initial_belief": trajectory.prior_belief[
                        start if state_rows is None else state_rows[start]
                    ]
                    .detach()
                    .clone(),
                }
                if trajectory.hidden is not None
                and trajectory.cell is not None
                and trajectory.prior_belief is not None
                else {}
            ),
        }
        for start in starts
    ]


class RoundBatchExpander:
    """Expand a fresh collection round into optimizer-sized learner batches."""

    def __init__(
        self,
        *,
        batch_segments: int,
        value_passes: int,
        seed: int,
        unroll_length: int = UNROLL_LENGTH,
    ) -> None:
        self.batch_segments = batch_segments
        self.unroll_length = unroll_length
        self.value_passes = value_passes
        self.seed = seed

    def expand(
        self,
        trajectories: Sequence[Trajectory],
        meta: RoundMeta,
        *,
        trajectory_kinds: Sequence[BatchKind] | None = None,
        trajectory_assignments: Sequence[CollectionAssignment] | None = None,
        kind_metas: Mapping[BatchKind, RoundMeta] | None = None,
        round_metrics: Mapping[str, float | int] | None = None,
    ) -> Iterator[LearnerBatch]:
        """Yield fresh policy batches followed by deterministic value replays."""
        kinds = trajectory_kinds or [meta.kind] * len(trajectories)
        assignments: Sequence[CollectionAssignment | None] = (
            trajectory_assignments
            if trajectory_assignments is not None
            else [None] * len(trajectories)
        )
        all_segments = [
            (segment, kind, assignment)
            for trajectory, kind, assignment in zip(
                trajectories, kinds, assignments, strict=True
            )
            for segment in segments(trajectory, self.unroll_length)
        ]
        collected_steps = sum(
            int(trajectory.dones.shape[0]) for trajectory in trajectories
        )
        yield from self._expand_entries(
            all_segments,
            meta,
            collected_steps=collected_steps,
            kind_metas=kind_metas,
            round_metrics=round_metrics,
        )

    def expand_native(
        self,
        entries: Sequence[
            tuple[dict[str, torch.Tensor], BatchKind, CollectionAssignment]
        ],
        meta: RoundMeta,
        *,
        collected_steps: int,
        kind_metas: Mapping[BatchKind, RoundMeta] | None = None,
        round_metrics: Mapping[str, float | int] | None = None,
    ) -> Iterator[LearnerBatch]:
        """Expand already-segmented on-device trajectories without CPU conversion."""
        yield from self._expand_entries(
            entries,
            meta,
            collected_steps=collected_steps,
            kind_metas=kind_metas,
            round_metrics=round_metrics,
        )

    def _expand_entries(
        self,
        all_segments: Sequence[
            tuple[
                dict[str, torch.Tensor],
                BatchKind,
                CollectionAssignment | None,
            ]
        ],
        meta: RoundMeta,
        *,
        collected_steps: int,
        kind_metas: Mapping[BatchKind, RoundMeta] | None,
        round_metrics: Mapping[str, float | int] | None,
    ) -> Iterator[LearnerBatch]:
        """Apply the common optimizer grouping and provenance contract."""
        groups = [
            tuple(all_segments[start : start + self.batch_segments])
            for start in range(
                0, len(all_segments) - self.batch_segments + 1, self.batch_segments
            )
        ]
        if not groups:
            return

        pending: list[
            tuple[
                tuple[
                    tuple[
                        dict[str, torch.Tensor],
                        BatchKind,
                        CollectionAssignment | None,
                    ],
                    ...,
                ],
                bool,
            ]
        ] = [(group, False) for group in groups]
        generator = torch.Generator().manual_seed(self.seed + meta.round_id)
        for _ in range(self.value_passes):
            order = torch.randperm(len(all_segments), generator=generator).tolist()
            pending.extend(
                (
                    tuple(
                        all_segments[index]
                        for index in order[start : start + self.batch_segments]
                    ),
                    True,
                )
                for start in range(
                    0,
                    len(all_segments) - self.batch_segments + 1,
                    self.batch_segments,
                )
            )

        for index, (entries, baseline_only) in enumerate(pending):
            batch_segments = tuple(entry[0] for entry in entries)
            segment_kinds = tuple(entry[1] for entry in entries)
            segment_assignments = tuple(entry[2] for entry in entries)
            batch_kind = (
                segment_kinds[0]
                if all(kind is segment_kinds[0] for kind in segment_kinds)
                else BatchKind.MIXED
            )
            batch_meta = (kind_metas or {}).get(batch_kind, meta)
            yield LearnerBatch(
                segments=batch_segments,
                kind=batch_kind,
                baseline_only=baseline_only,
                first_of_round=index == 0,
                end_of_round=index == len(pending) - 1,
                collected_steps=collected_steps if index == 0 else 0,
                round_id=batch_meta.round_id,
                actor_version=batch_meta.actor_version,
                game_ids=batch_meta.game_ids,
                opponent_ids=batch_meta.opponent_ids,
                segment_kinds=segment_kinds,
                seeds=batch_meta.seeds,
                opponent_digests=batch_meta.opponent_digests,
                segment_opponent_ids=tuple(
                    assignment.opponent_id
                    for assignment in segment_assignments
                    if assignment is not None
                ),
                segment_opponent_digests=tuple(
                    assignment.checkpoint_sha256
                    for assignment in segment_assignments
                    if assignment is not None
                ),
                segment_game_ids=tuple(
                    assignment.game_id
                    for assignment in segment_assignments
                    if assignment is not None
                ),
                segment_seeds=tuple(
                    assignment.seed
                    for assignment in segment_assignments
                    if assignment is not None
                ),
                round_metrics=round_metrics or {},
            )


def _native_recurrent_row(
    field: torch.Tensor,
    row: int,
    environment: int,
    seat: int,
) -> torch.Tensor:
    """Select one unbatched sparse policy-state row from native storage."""
    if field.ndim == 7:
        return field[row, :, environment, seat]
    return field[row, environment, seat]


def _native_state_rows(
    state_steps: torch.Tensor | None,
    recurrent: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None],
    starts: tuple[int, ...],
) -> dict[int, int]:
    """Validate sparse native actor state and map acted steps to state rows."""
    if state_steps is None:
        if any(field is not None for field in recurrent):
            raise ValueError("native recurrent fields require sparse state_steps")
        return {}
    if not all(field is not None for field in recurrent):
        raise ValueError("native state_steps require every recurrent field")
    listed = [int(step) for step in state_steps.tolist()]
    if listed != list(starts):
        raise ValueError("native recurrent state_steps must equal segment starts")
    return {step: row for row, step in enumerate(listed)}


def native_segments(
    trajectory: object,
    assignments: Sequence[CollectionAssignment],
    *,
    unroll_length: int,
    recorded_seats: int,
    clamp_terminal: bool = True,
) -> list[tuple[dict[str, torch.Tensor], BatchKind, CollectionAssignment]]:
    """Slice a tensor trajectory directly into the accepted learner schema.

    The simulator owns ``(time, environment, seat, ...)`` device tensors.  This
    adapter removes those two collection axes without constructing the CPU
    ``learn.rollout.Trajectory`` type: self-play retains both seats and every
    other population kind retains learner seat zero only.
    """
    from kaggriculture.sim.rollout import Trajectory as NativeTrajectory

    if not isinstance(trajectory, NativeTrajectory):
        raise TypeError("native segment expansion requires sim.rollout.Trajectory")
    if len(assignments) != trajectory.board.shape[1]:
        raise ValueError("one native assignment is required per environment row")
    if recorded_seats not in (1, 2):
        raise ValueError("recorded_seats must be one or two")
    turns = int(trajectory.dones.shape[0])
    starts = segment_starts(turns, unroll_length)
    recurrent = (
        trajectory.hidden,
        trajectory.cell,
        trajectory.prior_belief,
    )
    state_rows = _native_state_rows(trajectory.state_steps, recurrent, starts)

    result: list[tuple[dict[str, torch.Tensor], BatchKind, CollectionAssignment]] = []
    for environment, assignment in enumerate(assignments):
        seats = range(2) if recorded_seats == 2 else range(1)
        for seat in seats:
            for start in starts:
                observation_rows = torch.arange(
                    start,
                    start + unroll_length + 1,
                    device=trajectory.board.device,
                )
                if clamp_terminal:
                    observation_rows.clamp_(max=turns - 1)
                segment = {
                    **{
                        name: getattr(trajectory, name)[
                            start : start + unroll_length, environment, seat
                        ]
                        for name in ACTED_FIELDS
                    },
                    "unit_valid": trajectory.unit_actions[
                        start : start + unroll_length, environment, seat
                    ]
                    != IGNORE,
                    **{
                        name: getattr(trajectory, name)[
                            start : start + unroll_length, environment, seat
                        ]
                        for name in BELIEF_FIELDS
                        if getattr(trajectory, name) is not None
                    },
                    **{
                        name: getattr(trajectory, name)[
                            observation_rows, environment, seat
                        ]
                        for name in OBSERVED_FIELDS
                    },
                }
                if state_rows:
                    row = state_rows[start]
                    assert trajectory.hidden is not None
                    assert trajectory.cell is not None
                    assert trajectory.prior_belief is not None
                    segment.update(
                        initial_hidden=_native_recurrent_row(
                            trajectory.hidden, row, environment, seat
                        )
                        .detach()
                        .clone(),
                        initial_cell=_native_recurrent_row(
                            trajectory.cell, row, environment, seat
                        )
                        .detach()
                        .clone(),
                        initial_belief=trajectory.prior_belief[row, environment, seat]
                        .detach()
                        .clone(),
                    )
                result.append((segment, assignment.kind, assignment))
    return result


class ReferenceRoundSource(Iterable[LearnerBatch]):
    """Reference-rollout source with explicit actor publication and provenance."""

    def __init__(
        self,
        config: ToadConfig,
        *,
        assignments: Iterable[CollectionAssignment] | None = None,
        collect_assignment: Callable[[CollectionAssignment], Sequence[Trajectory]]
        | None = None,
        pool: SnapshotPool | None = None,
        teacher: SnapshotEntry | None = None,
    ) -> None:
        self.config = config
        self._assignments = tuple(assignments) if assignments is not None else None
        self._collector = collect_assignment
        self.pool = (
            pool.rebind(
                structure=structural_fingerprint(config),
                seed=config.population.population_seed,
            )
            if pool is not None
            else None
        )
        self.teacher = teacher
        self.actor_state: dict[str, torch.Tensor] = {}
        self.actor_version = 0
        self.next_game_id = 0
        self.rng = random.Random(config.runtime.seed)
        self._next_round_id = 0
        self._restored_actor = False
        self.global_rank = 0
        self.world_size = 1
        self._round_validator: Callable[[Sequence[LearnerBatch], int], None] | None = (
            None
        )

    def configure_distributed(self, *, rank: int, world_size: int) -> None:
        """Bind this rank-local collector to one resolved Lightning topology."""
        # Reuse the pure allocator's complete topology validation without
        # consuming or advancing the stream.
        rank_game_ids(0, 1, rank, world_size)
        self.global_rank = rank
        self.world_size = world_size

    def set_round_validator(
        self,
        validator: Callable[[Sequence[LearnerBatch], int], None] | None,
    ) -> None:
        """Install the all-rank batch-shape gate run before the first yield."""
        self._round_validator = validator

    def _ensure_initial_pool(self) -> None:
        """Require fit-start to install bootstrap population without local writes."""
        population = self.config.population
        structure = structural_fingerprint(self.config)
        if self.pool is not None and self.pool.manifest.entries:
            self.pool = self.pool.rebind(
                structure=structure,
                seed=population.population_seed,
            )
            return
        if not population.initial_snapshots and not population.snapshot_at_start:
            return
        raise RuntimeError(
            "population bootstrap was not installed by the coordinated fit-start "
            "boundary"
        )

    def publish_actor(
        self, state_dict: Mapping[str, torch.Tensor], version: int
    ) -> None:
        """Install the immutable CPU snapshot used by the next collection."""
        self.actor_state = dict(state_dict)
        self.actor_version = version
        self._restored_actor = False

    def collect_assignment(
        self, assignment: CollectionAssignment
    ) -> Sequence[Trajectory]:
        """Collect one assigned game through the existing reference collector."""
        if not self.actor_state:
            raise RuntimeError("reference collection needs a published actor state")
        from kaggriculture.learn.scripts.toad import _play_reference

        opponents = self._materialize_opponents((assignment,))
        return _play_reference(self._worker_input(assignment, opponents))

    def _worker_input(
        self,
        assignment: CollectionAssignment,
        opponents: Mapping[OpponentBinding, tuple[dict[str, torch.Tensor], ModelConfig]]
        | None = None,
    ) -> ReferenceWorkerInput:
        """Return one resolved typed worker request with no architecture drift."""
        return self._worker_input_chunk((assignment,), opponents)

    def _worker_input_chunk(
        self,
        assignments: Sequence[CollectionAssignment],
        opponents: Mapping[OpponentBinding, tuple[dict[str, torch.Tensor], ModelConfig]]
        | None = None,
    ) -> ReferenceWorkerInput:
        """Build one worker request for games sharing an immutable binding."""
        if not assignments:
            raise ValueError("reference worker chunks cannot be empty")
        assignment = assignments[0]
        versus = None
        if assignment.kind is BatchKind.SCRIPTED:
            from kaggriculture.learn.scripts.toad import OPPONENT

            versus = (
                OPPONENT
                if assignment.opponent_id == "economic"
                else assignment.opponent_id
            )
        opponent_state: dict[str, torch.Tensor] | None = None
        opponent_model: ModelConfig | None = None
        if assignment.checkpoint_sha256 is not None:
            binding = self._assignment_binding(assignment)
            if binding is None or opponents is None or binding not in opponents:
                raise RuntimeError("neural opponent was not materialized for its round")
            if any(self._assignment_binding(item) != binding for item in assignments):
                raise RuntimeError("worker chunk mixes immutable opponent bindings")
            opponent_state, opponent_model = opponents[binding]
        return ReferenceWorkerInput(
            actor_state=self.actor_state,
            seeds=[item.seed for item in assignments],
            model=self.config.model,
            versus=versus,
            money_weight=self.config.curriculum.money_weight,
            unroll_length=self.config.optimizer.unroll_length,
            opponent_state=opponent_state,
            opponent_model=opponent_model,
        )

    def _materialize_opponents(
        self, assignments: Sequence[CollectionAssignment]
    ) -> dict[OpponentBinding, tuple[dict[str, torch.Tensor], ModelConfig]]:
        """Validate every binding while caching only content-addressed tensors."""
        validated: dict[OpponentBinding, tuple[ModelConfig, SnapshotEntry | None]] = {}
        for binding, group in self._group_opponent_assignments(assignments).items():
            validated[binding] = self._validate_opponent_group(binding, group)

        frozen_entries = [
            entry for _model, entry in validated.values() if entry is not None
        ]
        tensor_cache = (
            self.pool.load_many(frozen_entries)
            if frozen_entries and self.pool is not None
            else {}
        )
        materialized: dict[
            OpponentBinding, tuple[dict[str, torch.Tensor], ModelConfig]
        ] = {}
        for binding, (model, _entry) in validated.items():
            if binding.kind is BatchKind.FROZEN_OPPONENT:
                state = tensor_cache[binding.sha256]
            else:
                spec = self.config.population.teacher
                if spec is None or spec.checkpoint != binding.checkpoint:
                    raise SnapshotIntegrityError(
                        "teacher assignment does not match population.teacher"
                    )
                try:
                    confirmed_spec = spec.model_copy(update={"sha256": binding.sha256})
                    if binding.sha256 in tensor_cache:
                        if sha256_file(binding.checkpoint) != binding.sha256:
                            raise TeacherCompatibilityError(
                                "teacher digest does not match assignment binding"
                            )
                        loaded_teacher = load_teacher_from_state(
                            confirmed_spec,
                            self.config.model,
                            tensor_cache[binding.sha256],
                            sha256=binding.sha256,
                        )
                    else:
                        loaded_teacher = load_teacher(
                            confirmed_spec,
                            self.config.model,
                        )
                except TeacherCompatibilityError as error:
                    raise SnapshotIntegrityError(
                        "teacher checkpoint is incompatible: "
                        f"{binding.checkpoint}: {error}"
                    ) from error
                state = dict(loaded_teacher.policy.state_dict())
                tensor_cache.setdefault(binding.sha256, state)
                model = loaded_teacher.model
            materialized[binding] = (state, model)
        return materialized

    @classmethod
    def _group_opponent_assignments(
        cls, assignments: Sequence[CollectionAssignment]
    ) -> dict[OpponentBinding, list[CollectionAssignment]]:
        """Group only exact neural bindings; digest equality is not identity."""
        groups: dict[OpponentBinding, list[CollectionAssignment]] = {}
        for assignment in assignments:
            binding = cls._assignment_binding(assignment)
            if binding is None:
                continue
            groups.setdefault(binding, []).append(assignment)
        return groups

    @staticmethod
    def _assignment_binding(
        assignment: CollectionAssignment,
    ) -> OpponentBinding | None:
        """Return an exact neural binding or reject incomplete path/digest pairs."""
        digest = assignment.checkpoint_sha256
        checkpoint = assignment.checkpoint
        if digest is None:
            if checkpoint is not None:
                raise SnapshotIntegrityError(
                    "opponent checkpoint path has no digest binding"
                )
            return None
        if checkpoint is None:
            raise SnapshotIntegrityError(
                f"opponent digest {digest} has no checkpoint path"
            )
        return OpponentBinding(assignment.kind, checkpoint, digest)

    def _validate_opponent_group(
        self, binding: OpponentBinding, group: Sequence[CollectionAssignment]
    ) -> tuple[ModelConfig, SnapshotEntry | None]:
        """Validate one path/kind/digest binding without merging provenance."""
        if any(self._assignment_binding(item) != binding for item in group):
            raise SnapshotIntegrityError(
                f"inconsistent assignment binding for digest {binding.sha256}"
            )
        if binding.kind is BatchKind.FROZEN_OPPONENT:
            if self.pool is None:
                raise EmptySnapshotPoolError(
                    "frozen opponent requested before pool population"
                )
            identity_matches = [
                entry
                for entry in self.pool.manifest.entries
                if entry.sha256 == binding.sha256 and entry.path == binding.checkpoint
            ]
            matches = identity_matches
            if len(matches) > 1:
                matches = [
                    entry
                    for entry in identity_matches
                    if all(
                        assignment.opponent_id
                        == (
                            f"snapshot:{entry.run_id}:{entry.environment_steps}:"
                            f"{entry.sha256[:12]}"
                        )
                        for assignment in group
                    )
                ]
            if len(matches) != 1:
                raise SnapshotIntegrityError(
                    "selected frozen opponent is not in the bound pool: "
                    f"{binding.sha256}"
                )
            model = self.config.model
            entry: SnapshotEntry | None = matches[0]
        elif binding.kind is BatchKind.TEACHER_DISTILL:
            teacher = self.config.population.teacher
            if teacher is None:
                raise ValueError(
                    "teacher-distill assignment requires population.teacher"
                )
            model = resolve_teacher_model(teacher, self.config.model)
            entry = None
        else:
            raise ValueError(
                f"non-neural assignment unexpectedly carries a digest: {binding.kind}"
            )
        return model, entry

    @classmethod
    def _worker_chunks(
        cls, assignments: Sequence[CollectionAssignment]
    ) -> tuple[tuple[CollectionAssignment, ...], ...]:
        """Chunk neural games by exact binding while retaining non-neural calls."""
        chunks: list[list[CollectionAssignment]] = []
        neural_chunks: dict[OpponentBinding, int] = {}
        for assignment in assignments:
            binding = cls._assignment_binding(assignment)
            if binding is None:
                chunks.append([assignment])
                continue
            index = neural_chunks.get(binding)
            if index is None:
                neural_chunks[binding] = len(chunks)
                chunks.append([assignment])
            else:
                chunks[index].append(assignment)
        return tuple(tuple(chunk) for chunk in chunks)

    def _collect_round(
        self, assignments: Sequence[CollectionAssignment]
    ) -> list[tuple[CollectionAssignment, Sequence[Trajectory]]]:
        """Collect all assignments through one typed, round-scoped pool."""
        opponents = self._materialize_opponents(assignments)
        if self._collector is not None:
            collected = []
            for assignment in assignments:
                try:
                    collected.append((assignment, self._collector(assignment)))
                except Exception as error:
                    raise self._collection_error(assignment) from error
            return collected

        if not self.actor_state:
            raise RuntimeError("reference collection needs a published actor state")
        from kaggriculture.learn.scripts.toad import _play_reference

        chunks = self._worker_chunks(assignments)

        with ProcessPoolExecutor(
            max_workers=self.config.population.collection_processes
        ) as pool:
            results = iter(
                pool.map(
                    _play_reference,
                    (self._worker_input_chunk(chunk, opponents) for chunk in chunks),
                )
            )
            collected = []
            for chunk in chunks:
                try:
                    trajectories = tuple(next(results))
                except Exception as error:
                    raise self._collection_error(chunk, error) from error
                if len(chunk) == 1:
                    collected.append((chunk[0], trajectories))
                    continue
                if len(trajectories) != len(chunk):
                    raise CollectionError(
                        "neural worker chunk returned an unexpected trajectory count: "
                        f"games={len(chunk)} trajectories={len(trajectories)}"
                    )
                collected.extend(
                    (assignment, (trajectory,))
                    for assignment, trajectory in zip(chunk, trajectories, strict=True)
                )
        return collected

    @staticmethod
    def _collection_error(
        assignments: CollectionAssignment | Sequence[CollectionAssignment],
        error: Exception | None = None,
    ) -> CollectionError:
        """Build a stable row-local or vector-wide grouped collection error."""
        singular = isinstance(assignments, CollectionAssignment)
        group = (assignments,) if singular else tuple(assignments)
        if not group:
            raise ValueError("collection failure requires an affected assignment")
        affected = group
        detail = ""
        if isinstance(error, CollectionRowError):
            if error.row < 0 or error.row >= len(group):
                raise CollectionError(
                    "collection worker returned an invalid failing row: "
                    f"row={error.row} rows={len(group)}",
                    game_ids=tuple(item.game_id for item in group),
                    seeds=tuple(item.seed for item in group),
                ) from error
            affected = (group[error.row],)
            detail = f" cause={error.error_type}: {error.original_message}"
        elif error is not None:
            detail = f" cause={type(error).__name__}: {error}"
        rendered = (
            "collection failed for "
            f"game_id={affected[0].game_id} seed={affected[0].seed} "
            f"opponent={affected[0].opponent_id}{detail}"
            if singular
            else "collection failed for "
            f"game_ids={tuple(item.game_id for item in affected)} "
            f"seeds={tuple(item.seed for item in affected)} "
            f"opponents={tuple(item.opponent_id for item in affected)}{detail}"
        )
        return CollectionError(
            rendered,
            game_ids=tuple(item.game_id for item in affected),
            seeds=tuple(item.seed for item in affected),
        )

    def _iter_local_round(self) -> Iterator[LearnerBatch]:
        """Materialize one local round without entering distributed collectives."""
        self._ensure_initial_pool()
        assignments = self._assignments
        generated = assignments is None
        first_game_id = self.next_game_id
        round_id = self._next_round_id
        if assignments is None:
            local_ids = rank_game_ids(
                first_game_id,
                self.config.population.environments_per_rank,
                self.global_rank,
                self.world_size,
            )
            assignments = allocate_round(
                self.config,
                local_ids[0],
                round_id,
                self.pool,
                self.teacher,
            )
        if not assignments:
            return
        collection_started = time.perf_counter()
        collected = self._collect_round(assignments)
        collection_seconds = time.perf_counter() - collection_started
        ordered = sorted(
            collected,
            key=lambda entry: _ONLINE_KINDS.index(entry[0].kind),
        )
        expander = RoundBatchExpander(
            batch_segments=self.config.optimizer.batch_segments,
            unroll_length=self.config.optimizer.unroll_length,
            value_passes=self.config.optimizer.value_passes,
            seed=self.config.runtime.seed,
        )
        ordered_assignments = tuple(entry[0] for entry in ordered)
        trajectories = [
            trajectory for _, assigned in ordered for trajectory in assigned
        ]
        trajectory_kinds = [
            assignment.kind
            for assignment, assigned in ordered
            for _trajectory in assigned
        ]
        trajectory_assignments = [
            assignment for assignment, assigned in ordered for _trajectory in assigned
        ]
        present = set(trajectory_kinds)
        round_kind = next(iter(present)) if len(present) == 1 else BatchKind.MIXED
        meta = RoundMeta(
            round_id=round_id,
            actor_version=self.actor_version,
            game_ids=tuple(item.game_id for item in ordered_assignments),
            seeds=tuple(item.seed for item in ordered_assignments),
            opponent_ids=tuple(item.opponent_id for item in ordered_assignments),
            kind=round_kind,
            opponent_digests=tuple(
                item.checkpoint_sha256 for item in ordered_assignments
            ),
        )
        kind_metas = {
            kind: RoundMeta(
                round_id=round_id,
                actor_version=self.actor_version,
                game_ids=tuple(
                    item.game_id for item in ordered_assignments if item.kind is kind
                ),
                seeds=tuple(
                    item.seed for item in ordered_assignments if item.kind is kind
                ),
                opponent_ids=tuple(
                    item.opponent_id
                    for item in ordered_assignments
                    if item.kind is kind
                ),
                kind=kind,
                opponent_digests=tuple(
                    item.checkpoint_sha256
                    for item in ordered_assignments
                    if item.kind is kind
                ),
            )
            for kind in present
        }
        mirror = [
            trajectory
            for assignment, assigned in ordered
            if assignment.kind is BatchKind.SELFPLAY
            for trajectory in assigned
        ]
        scripted = [
            trajectory
            for assignment, assigned in ordered
            if assignment.kind is BatchKind.SCRIPTED
            for trajectory in assigned
        ]
        from kaggriculture.learn.scripts.toad import _collection_metrics

        collection_metrics = (
            _collection_metrics(mirror, scripted, self.config.curriculum.reward_field)
            if mirror or scripted
            else {}
        )
        collection_metrics["throughput/collection_seconds"] = collection_seconds
        for kind in _ONLINE_KINDS:
            kind_rows = [
                trajectory
                for assignment, assigned in ordered
                if assignment.kind is kind
                for trajectory in assigned
            ]
            collection_metrics[f"collection/games/{kind.value}"] = len(
                {
                    assignment.game_id
                    for assignment, _assigned in ordered
                    if assignment.kind is kind
                }
            )
            collection_metrics[f"collection/return/{kind.value}"] = (
                float(
                    torch.stack(
                        [
                            getattr(trajectory, self.config.curriculum.reward_field)
                            .float()
                            .sum()
                            for trajectory in kind_rows
                        ]
                    ).mean()
                )
                if kind_rows
                else float("nan")
            )
        opponent_groups: dict[str, list[Trajectory]] = {}
        opponent_game_ids: dict[str, set[int]] = {}
        for assignment, assigned in ordered:
            opponent_groups.setdefault(assignment.opponent_id, []).extend(assigned)
            opponent_game_ids.setdefault(assignment.opponent_id, set()).add(
                assignment.game_id
            )
        for opponent_id, opponent_rows in opponent_groups.items():
            safe_id = opponent_id.replace("/", "_")
            returns = torch.stack(
                [
                    getattr(trajectory, self.config.curriculum.reward_field)
                    .float()
                    .sum()
                    for trajectory in opponent_rows
                ]
            )
            collection_metrics[f"collection/games_by_opponent/{safe_id}"] = len(
                opponent_game_ids[opponent_id]
            )
            collection_metrics[f"collection/return_by_opponent/{safe_id}"] = float(
                returns.mean()
            )
            collection_metrics[f"collection/return_sum_by_opponent/{safe_id}"] = float(
                returns.sum()
            )
            collection_metrics[f"collection/return_count_by_opponent/{safe_id}"] = len(
                opponent_rows
            )

        batches = tuple(
            expander.expand(
                trajectories,
                meta,
                trajectory_kinds=trajectory_kinds,
                trajectory_assignments=trajectory_assignments,
                kind_metas=kind_metas,
                round_metrics=collection_metrics,
            )
        )
        if not batches:
            raise CollectionError(
                f"collection round {round_id} produced no complete learner batches"
            )
        if generated:
            self.next_game_id = (
                first_game_id
                + self.config.population.environments_per_rank * self.world_size
            )
        self._next_round_id = round_id + 1
        yield from batches

    def __iter__(self) -> Iterator[LearnerBatch]:
        """Acknowledge full local materialization before any batch-shape gather."""
        if self.world_size == 1:
            batches = tuple(self._iter_local_round())
        else:
            local_error: Exception | None = None
            local_traceback = ""
            try:
                batches = tuple(self._iter_local_round())
            except Exception as error:
                batches = ()
                local_error = error
                local_traceback = traceback.format_exc()
            status: dict[str, object] = {
                "ok": local_error is None,
                "rank": self.global_rank,
                "type": None if local_error is None else type(local_error).__name__,
                "message": None if local_error is None else str(local_error),
                "traceback": local_traceback,
            }
            gathered = _all_gather_objects(status, self.world_size)
            failures = tuple(
                cast(Mapping[str, object], item)
                for item in gathered
                if isinstance(item, Mapping) and not bool(item.get("ok"))
            )
            if failures:
                distributed = DistributedCollectionError(failures)
                if local_error is not None:
                    raise distributed from local_error
                raise distributed
        if self._round_validator is not None:
            self._round_validator(batches, self._next_round_id - 1)
        yield from batches


@dataclass(frozen=True)
class _NativeCollection:
    """One compatible vector group and its sequential on-device chunks."""

    assignments: tuple[CollectionAssignment, ...]
    trajectories: tuple[NativeTrajectory, ...]
    recorded_seats: int


class NativeRoundSource(ReferenceRoundSource):
    """Synchronous tensor-simulator source emitting common batches directly."""

    def __init__(
        self,
        config: ToadConfig,
        *,
        assignments: Iterable[CollectionAssignment] | None = None,
        pool: SnapshotPool | None = None,
        teacher: SnapshotEntry | None = None,
    ) -> None:
        super().__init__(
            config,
            assignments=assignments,
            pool=pool,
            teacher=teacher,
        )
        self.device = torch.device(config.runtime.rollout_device)

    def _native_policy(
        self,
        model: ModelConfig,
        state: Mapping[str, torch.Tensor],
    ) -> torch.nn.Module:
        """Instantiate one exact eager actor identity on the rollout device."""
        from kaggriculture.learn.model import Policy
        from kaggriculture.learn.toad.model import StatefulPolicy, uses_stateful_policy

        policy: torch.nn.Module
        if uses_stateful_policy(model):
            policy = StatefulPolicy(model)
        else:
            policy = Policy(
                blocks=model.blocks,
                channels=model.channels,
                value_bound=model.value_bound,
                kernel_size=model.kernel_size,
                activation=model.activation,
            )
        policy.load_state_dict(state, strict=True)
        return policy.to(self.device).eval()

    def _scripted_opponent(self) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
        """Resolve the configured native scripted identity without substitution."""
        if self.config.population.scripted_opponent != "economic":
            raise CollectionError(
                "native scripted collection supports only the verified economic "
                f"opponent, got {self.config.population.scripted_opponent!r}"
            )
        from kaggriculture import economic_policy

        return economic_policy.agent

    def _collect_native_round(
        self, assignments: Sequence[CollectionAssignment]
    ) -> list[_NativeCollection]:
        """Collect assignments as on-device trajectories, preserving bindings."""
        if not self.actor_state:
            raise RuntimeError("native collection needs a published actor state")
        from kaggriculture.learn.toad.model import PolicyState
        from kaggriculture.sim.config import Config as SimConfig
        from kaggriculture.sim.engine import reset
        from kaggriculture.sim.rollout import RolloutPolicyState, collect_segment

        actor = self._native_policy(self.config.model, self.actor_state)
        opponents = self._materialize_opponents(assignments)
        chunks: list[list[CollectionAssignment]] = []
        chunk_indices: dict[BatchKind | OpponentBinding, int] = {}
        for assignment in assignments:
            binding = self._assignment_binding(assignment)
            key: BatchKind | OpponentBinding = binding or assignment.kind
            index = chunk_indices.get(key)
            if index is None:
                chunk_indices[key] = len(chunks)
                chunks.append([assignment])
            else:
                chunks[index].append(assignment)

        collected: list[_NativeCollection] = []
        for mutable_chunk in chunks:
            chunk = tuple(mutable_chunk)
            assignment = chunk[0]
            try:
                scripted = (
                    self._scripted_opponent()
                    if assignment.kind is BatchKind.SCRIPTED
                    else None
                )
                opponent_policy: torch.nn.Module | None = None
                if assignment.kind in {
                    BatchKind.FROZEN_OPPONENT,
                    BatchKind.TEACHER_DISTILL,
                }:
                    binding = self._assignment_binding(assignment)
                    if binding is None or binding not in opponents:
                        raise CollectionError(
                            "native neural opponent was not materialized"
                        )
                    opponent_state, opponent_model = opponents[binding]
                    opponent_policy = self._native_policy(
                        opponent_model,
                        opponent_state,
                    )
                state = reset(
                    SimConfig(),
                    torch.tensor([item.seed for item in chunk], device=self.device),
                )
                generators = tuple(
                    torch.Generator(device=self.device).manual_seed(item.seed)
                    for item in chunk
                )
                unroll = self.config.optimizer.unroll_length
                turns = EPISODE_STEPS - 1
                remainder = turns % unroll
                lengths = ((remainder,) if remainder else ()) + (unroll,) * (
                    turns // unroll
                )
                policy_state: PolicyState | RolloutPolicyState | None = None
                trajectories = []
                for length in lengths:
                    state, policy_state, trajectory = collect_segment(
                        state,
                        actor,
                        policy_state=policy_state,
                        turns=length,
                        state_unroll_length=unroll,
                        money_weight=self.config.curriculum.money_weight,
                        generator=generators,
                        opponent=scripted,
                        opponent_policy=opponent_policy,
                    )
                    trajectories.append(trajectory)
                recorded_seats = 2 if assignment.kind is BatchKind.SELFPLAY else 1
                collected.append(
                    _NativeCollection(
                        assignments=chunk,
                        trajectories=tuple(trajectories),
                        recorded_seats=recorded_seats,
                    )
                )
            except Exception as error:
                raise self._collection_error(chunk, error) from error
        return collected

    def _native_metrics(  # noqa: C901 - one direct mirror of the public metric map
        self,
        collected: Sequence[_NativeCollection],
        seconds: float,
    ) -> dict[str, float | int]:
        """Reproduce the accepted round metrics directly from device trajectories."""
        from kaggriculture.learn.critic import critic_scores
        from kaggriculture.sim.rollout import Trajectory as NativeTrajectory

        metrics: dict[str, float | int] = {
            "throughput/collection_seconds": seconds,
        }
        streams: list[dict[str, Any]] = []
        flattened: list[CollectionAssignment] = []
        for collection in collected:
            trajectories = collection.trajectories
            if not trajectories or not all(
                isinstance(trajectory, NativeTrajectory) for trajectory in trajectories
            ):
                raise TypeError("native metrics require sim.rollout.Trajectory chunks")
            final = trajectories[-1]

            def required(trajectory: NativeTrajectory, name: str) -> torch.Tensor:
                value = getattr(trajectory, name)
                if not isinstance(value, torch.Tensor):
                    raise ValueError(
                        f"native trajectory is missing metric field {name}"
                    )
                return value

            for environment, assignment in enumerate(collection.assignments):
                flattened.append(assignment)
                for seat in range(collection.recorded_seats):
                    series = {
                        name: torch.cat(
                            [
                                getattr(trajectory, name)[:, environment, seat]
                                for trajectory in trajectories
                            ]
                        ).float()
                        for name in (
                            "values",
                            "shaped",
                            "shaped_money",
                            "margin",
                            "own",
                            "sparse",
                            "dones",
                        )
                    }
                    units = torch.stack(
                        [
                            required(trajectory, "units_sold")[environment, seat]
                            for trajectory in trajectories
                        ]
                    ).sum()
                    proceeds = torch.stack(
                        [
                            required(trajectory, "sale_proceeds")[environment, seat]
                            for trajectory in trajectories
                        ]
                    ).sum()
                    market_value = torch.stack(
                        [
                            required(trajectory, "sale_market_value")[environment, seat]
                            for trajectory in trajectories
                        ]
                    ).sum()
                    mean_sale = torch.where(units > 0, proceeds / units, 0.0)
                    mean_market = torch.where(units > 0, market_value / units, 0.0)
                    streams.append(
                        {
                            "assignment": assignment,
                            **series,
                            "selected": series[self.config.curriculum.reward_field],
                            "final_margin": float(
                                required(final, "final_margin")[environment, seat]
                            ),
                            "final_bank": float(
                                required(final, "final_bank")[environment, seat]
                            ),
                            "final_capital": float(
                                required(final, "final_capital")[environment, seat]
                            ),
                            "illegal": int(
                                sum(
                                    required(trajectory, "illegal_by_stream")[
                                        environment, seat
                                    ]
                                    for trajectory in trajectories
                                )
                            ),
                            "sales": float(
                                sum(
                                    required(trajectory, "sales")[environment, seat]
                                    for trajectory in trajectories
                                )
                            ),
                            "units_sold": float(units),
                            "mean_sale_price": float(mean_sale),
                            "realisation": float(
                                torch.where(
                                    mean_market > 0,
                                    mean_sale / mean_market,
                                    0.0,
                                )
                            ),
                            "bought": float(
                                sum(
                                    required(trajectory, "bought")[environment, seat]
                                    for trajectory in trajectories
                                )
                            ),
                        }
                    )

        mirror = [
            stream
            for stream in streams
            if cast(CollectionAssignment, stream["assignment"]).kind
            is BatchKind.SELFPLAY
        ]
        econ = [
            stream
            for stream in streams
            if cast(CollectionAssignment, stream["assignment"]).kind
            is BatchKind.SCRIPTED
        ]
        legacy = mirror + econ

        def mean(values: Sequence[float]) -> float:
            return sum(values) / len(values) if values else float("nan")

        def population(
            population_streams: Sequence[dict[str, Any]], suffix: str
        ) -> dict[str, float]:
            return {
                f"diag/{name}_{suffix}": mean(
                    [float(stream[name]) for stream in population_streams]
                )
                for name in (
                    "mean_sale_price",
                    "realisation",
                    "sales",
                    "units_sold",
                    "bought",
                    "final_capital",
                )
            }

        def critic(
            population_streams: Sequence[dict[str, Any]], suffix: str
        ) -> dict[str, float]:
            scores = critic_scores(
                [cast(torch.Tensor, stream["values"]) for stream in population_streams],
                [
                    cast(torch.Tensor, stream["selected"])
                    for stream in population_streams
                ],
                [
                    cast(torch.Tensor, stream["dones"]).bool()
                    for stream in population_streams
                ],
            )
            return {
                f"critic/ev_{suffix}": scores["ev_vs_return"],
                f"critic/ev_within_turn_{suffix}": scores["ev_vs_return_within_turn"],
            }

        if legacy:
            banks = [float(stream["final_bank"]) for stream in legacy]
            metrics.update(
                {
                    "diag/bank_mean": mean(banks),
                    "diag/bank_max": max(banks),
                    "diag/bank_mirror": mean(
                        [float(stream["final_bank"]) for stream in mirror]
                    ),
                    "diag/bank_vs_econ": mean(
                        [float(stream["final_bank"]) for stream in econ]
                    ),
                    "diag/n_econ_envs": len(econ),
                    "objective/win_rate_vs_econ": mean(
                        [float(float(stream["final_margin"]) > 0.0) for stream in econ]
                    ),
                    "diag/mirror_decisive_rate": 2.0
                    * mean(
                        [
                            float(float(stream["final_margin"]) > 0.0)
                            for stream in mirror
                        ]
                    ),
                    "objective/margin_vs_econ": mean(
                        [float(stream["final_margin"]) for stream in econ]
                    ),
                    "diag/margin_mean_mirror": mean(
                        [float(stream["final_margin"]) for stream in mirror]
                    ),
                    **population(econ, "vs_econ"),
                    **population(mirror, "mirror"),
                    **critic(econ, "econ_games"),
                    **critic(mirror, "mirror_games"),
                    "proxy/shaped_mean": mean(
                        [
                            float(cast(torch.Tensor, stream["shaped"]).sum())
                            for stream in legacy
                        ]
                    ),
                    "proxy/shaped_reward_mean": mean(
                        [
                            float(cast(torch.Tensor, stream["selected"]).sum())
                            for stream in legacy
                        ]
                    ),
                    "diag/illegal": sum(int(stream["illegal"]) for stream in legacy),
                    "diag/gross_purchases": mean(
                        [
                            float(
                                (
                                    -cast(torch.Tensor, stream["own"]).clamp(max=0.0)
                                ).sum()
                            )
                            for stream in legacy
                        ]
                    ),
                    "proxy/money_term": mean(
                        [
                            float(
                                (
                                    cast(torch.Tensor, stream["shaped_money"])
                                    - cast(torch.Tensor, stream["shaped"])
                                ).sum()
                            )
                            for stream in legacy
                        ]
                    ),
                }
            )

        for kind in _ONLINE_KINDS:
            values = [
                float(cast(torch.Tensor, stream["selected"]).sum())
                for stream in streams
                if cast(CollectionAssignment, stream["assignment"]).kind is kind
            ]
            metrics[f"collection/games/{kind.value}"] = len(
                {
                    assignment.game_id
                    for assignment in flattened
                    if assignment.kind is kind
                }
            )
            metrics[f"collection/return/{kind.value}"] = mean(values)

        for opponent_id in {assignment.opponent_id for assignment in flattened}:
            values = [
                float(cast(torch.Tensor, stream["selected"]).sum())
                for stream in streams
                if cast(CollectionAssignment, stream["assignment"]).opponent_id
                == opponent_id
            ]
            safe_id = opponent_id.replace("/", "_")
            total = sum(values)
            metrics[f"collection/games_by_opponent/{safe_id}"] = len(
                {
                    assignment.game_id
                    for assignment in flattened
                    if assignment.opponent_id == opponent_id
                }
            )
            metrics[f"collection/return_by_opponent/{safe_id}"] = mean(values)
            metrics[f"collection/return_sum_by_opponent/{safe_id}"] = total
            metrics[f"collection/return_count_by_opponent/{safe_id}"] = len(values)
        return metrics

    def _iter_local_round(self) -> Iterator[LearnerBatch]:
        """Collect one rank-owned round and expand native device tensors directly."""
        self._ensure_initial_pool()
        assignments = self._assignments
        generated = assignments is None
        first_game_id = self.next_game_id
        round_id = self._next_round_id
        if assignments is None:
            local_ids = rank_game_ids(
                first_game_id,
                self.config.population.environments_per_rank,
                self.global_rank,
                self.world_size,
            )
            assignments = allocate_round(
                self.config,
                local_ids[0],
                round_id,
                self.pool,
                self.teacher,
            )
        if not assignments:
            return
        started = time.perf_counter()
        collected = self._collect_native_round(assignments)
        collection_seconds = time.perf_counter() - started
        ordered = sorted(
            collected,
            key=lambda entry: _ONLINE_KINDS.index(entry.assignments[0].kind),
        )
        ordered_assignments = tuple(
            assignment for entry in ordered for assignment in entry.assignments
        )
        present = {assignment.kind for assignment in ordered_assignments}
        round_kind = next(iter(present)) if len(present) == 1 else BatchKind.MIXED
        meta = RoundMeta(
            round_id=round_id,
            actor_version=self.actor_version,
            game_ids=tuple(item.game_id for item in ordered_assignments),
            seeds=tuple(item.seed for item in ordered_assignments),
            opponent_ids=tuple(item.opponent_id for item in ordered_assignments),
            kind=round_kind,
            opponent_digests=tuple(
                item.checkpoint_sha256 for item in ordered_assignments
            ),
        )
        kind_metas = {
            kind: RoundMeta(
                round_id=round_id,
                actor_version=self.actor_version,
                game_ids=tuple(
                    item.game_id for item in ordered_assignments if item.kind is kind
                ),
                seeds=tuple(
                    item.seed for item in ordered_assignments if item.kind is kind
                ),
                opponent_ids=tuple(
                    item.opponent_id
                    for item in ordered_assignments
                    if item.kind is kind
                ),
                kind=kind,
                opponent_digests=tuple(
                    item.checkpoint_sha256
                    for item in ordered_assignments
                    if item.kind is kind
                ),
            )
            for kind in present
        }
        entries = []
        for collection in ordered:
            chunk_entries = [
                native_segments(
                    trajectory,
                    collection.assignments,
                    unroll_length=self.config.optimizer.unroll_length,
                    recorded_seats=collection.recorded_seats,
                    clamp_terminal=index == len(collection.trajectories) - 1,
                )
                for index, trajectory in enumerate(collection.trajectories)
            ]
            stream_count = len(collection.assignments) * collection.recorded_seats
            for stream in range(stream_count):
                entries.extend(chunk[stream] for chunk in chunk_entries if chunk)
        collected_steps = (EPISODE_STEPS - 1) * sum(
            len(collection.assignments) * collection.recorded_seats
            for collection in ordered
        )
        expander = RoundBatchExpander(
            batch_segments=self.config.optimizer.batch_segments,
            unroll_length=self.config.optimizer.unroll_length,
            value_passes=self.config.optimizer.value_passes,
            seed=self.config.runtime.seed,
        )
        batches = tuple(
            expander.expand_native(
                entries,
                meta,
                collected_steps=collected_steps,
                kind_metas=kind_metas,
                round_metrics=self._native_metrics(ordered, collection_seconds),
            )
        )
        if not batches:
            raise CollectionError(
                f"collection round {round_id} produced no complete learner batches"
            )
        if generated:
            self.next_game_id = (
                first_game_id
                + self.config.population.environments_per_rank * self.world_size
            )
        self._next_round_id = round_id + 1
        yield from batches


class RoundIterableDataset(IterableDataset[LearnerBatch]):
    """Thin iterable dataset that preserves the source's round ownership."""

    def __init__(self, source: ReferenceRoundSource, config: ToadConfig) -> None:
        super().__init__()
        self.source = source
        self.config = config

    def __iter__(self) -> Iterator[LearnerBatch]:
        """Yield batches directly from the owned source."""
        return iter(self.source)


class ToadDataModule(lightning.LightningDataModule):
    """Synchronous Lightning bridge for reference collection batches."""

    def __init__(self, config: ToadConfig, source: ReferenceRoundSource) -> None:
        super().__init__()
        self.config = config
        self.source = source

    def setup(self, stage: str | None) -> None:
        """Bind rank ownership and reject unequal quotas before collection."""
        del stage
        trainer = self.trainer
        if trainer is None:
            return
        rank = int(trainer.global_rank)
        world_size = int(trainer.world_size)
        self.source.configure_distributed(rank=rank, world_size=world_size)
        if world_size == 1:
            return
        local = expected_round_batch_counts(self.config)
        gathered = _all_gather_objects(local, world_size)
        if any(item != local for item in gathered):
            raise DistributedRoundMismatch(
                "expected round quotas differ across ranks: "
                f"rank={rank}, gathered={gathered!r}"
            )
        self.source.set_round_validator(self._validate_round_batches)

    def _validate_round_batches(
        self, batches: Sequence[LearnerBatch], round_id: int
    ) -> None:
        """Collectively gate actual counts and markers before DDP sees a batch."""
        local = round_batch_signature(
            batches,
            expected=expected_round_batch_counts(self.config),
            rank=self.source.global_rank,
            round_id=round_id,
        )
        gathered = _all_gather_objects(local, self.source.world_size)
        try:
            signatures = tuple(
                cast(RoundBatchSignature, signature) for signature in gathered
            )
        except TypeError as error:
            raise DistributedRoundMismatch(
                "distributed round gather returned malformed signatures"
            ) from error
        if not all(isinstance(item, RoundBatchSignature) for item in signatures):
            raise DistributedRoundMismatch(
                "distributed round gather returned malformed signatures"
            )
        validate_distributed_round_signatures(signatures)

    def publish_actor(
        self, state_dict: Mapping[str, torch.Tensor], version: int
    ) -> None:
        """Detach learner weights before making them available to collection."""
        self.source.publish_actor(
            {
                name: tensor.detach().to("cpu", copy=True)
                for name, tensor in state_dict.items()
            },
            version,
        )

    def materialize_population_bootstrap(
        self,
        store: SnapshotStore,
        actor_state: Mapping[str, torch.Tensor],
        *,
        environment_steps: int,
        round_id: int,
        run_id: str,
    ) -> None:
        """Materialize every configured bootstrap source in rank-zero order."""
        existing_run_ids = {entry.run_id for entry in store.manifest.entries}
        for path in self.config.population.initial_snapshots:
            initial_run_id = f"initial:{path}"
            if initial_run_id in existing_run_ids:
                continue
            state = _checkpoint_policy_state(path)
            _strictly_validate_policy_state(self.config.model, state)
            store.add(
                state,
                environment_steps=environment_steps,
                round_id=round_id,
                run_id=initial_run_id,
            )
            existing_run_ids.add(initial_run_id)
        if self.config.population.snapshot_at_start:
            actor_run_id = f"{run_id}:actor-at-start"
            legacy_actor_exists = any(
                entry.run_id == run_id
                and entry.environment_steps == environment_steps
                and entry.round_id == round_id
                for entry in store.manifest.entries
            )
            if actor_run_id in existing_run_ids or legacy_actor_exists:
                return
            _strictly_validate_policy_state(self.config.model, actor_state)
            store.add(
                actor_state,
                environment_steps=environment_steps,
                round_id=round_id,
                run_id=actor_run_id,
            )

    def publish_manifest(self, manifest: SnapshotManifest) -> None:
        """Bind collection to the callback's exact durable population view."""
        directory = self.population_directory()
        try:
            store = SnapshotStore.open_generation(
                directory,
                capacity=self.config.population.pool_capacity,
                structure=structural_fingerprint(self.config),
                manifest=manifest,
            )
        except SnapshotIntegrityError as error:
            raise SnapshotIntegrityError(
                "published population manifest does not match a durable generation: "
                f"{error}"
            ) from error
        pool = SnapshotPool.from_store(
            store,
            seed=self.config.population.population_seed,
        )
        # Validate each immutable boundary artifact before exposing the new pool.
        # Different boundaries may legitimately contain identical weights, so
        # validating the whole manifest through ``SnapshotPool.load_many`` would
        # incorrectly reject equal digests with distinct entry metadata.
        for entry in manifest.entries:
            store.load(entry)
        self.source.pool = pool

    def open_population_store(self, manifest: SnapshotManifest) -> SnapshotStore:
        """Open the exact restored generation or the current fresh-run head."""
        directory = self.population_directory()
        pool = self.source.pool
        if pool is not None and pool.manifest == manifest:
            return SnapshotStore.open_generation(
                directory,
                capacity=self.config.population.pool_capacity,
                structure=structural_fingerprint(self.config),
                manifest=manifest,
            )
        return SnapshotStore(
            directory,
            capacity=self.config.population.pool_capacity,
            structure=structural_fingerprint(self.config),
        )

    def population_directory(self) -> Path:
        """Return restored pool identity, or the configured fresh-run directory."""
        pool = self.source.pool
        if pool is not None and pool.manifest.entries:
            return pool.identity().directory
        return (self.config.runtime.output_dir / "population").resolve()

    def train_dataloader(self) -> DataLoader[LearnerBatch]:
        """Return the intentionally single-process iterable loader."""
        return DataLoader(
            RoundIterableDataset(self.source, self.config),
            batch_size=None,
            num_workers=0,
        )

    def state_dict(self) -> dict[str, object]:
        """Serialize the stream position needed for the next collection."""
        pool_identity = (
            self.source.pool.identity().model_dump(mode="json")
            if self.source.pool is not None and self.source.pool.manifest.entries
            else None
        )
        return {
            "next_game_id": self.source.next_game_id,
            "next_round_id": self.source._next_round_id,
            "collector_rng": self.source.rng.getstate(),
            "published_actor_version": self.source.actor_version,
            "published_actor_state": {
                name: tensor.detach().to("cpu", copy=True)
                for name, tensor in self.source.actor_state.items()
            },
            "population_pool": pool_identity,
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:  # noqa: C901
        """Restore the collector stream without retaining stale actor weights."""
        population = self.config.population
        durable_pool_required = bool(
            population.frozen_opponent
            or population.initial_snapshots
            or population.snapshot_at_start
            or population.snapshot_every_environment_steps is not None
            or (self.source.pool is not None and self.source.pool.manifest.entries)
        )
        if "population_pool" not in state_dict and durable_pool_required:
            raise PopulationResumeMigrationError(
                "checkpoint is missing population_pool identity for active frozen "
                "population; checkpoint migration is required"
            )
        try:
            next_game_id = _checkpoint_nonnegative_int(state_dict["next_game_id"])
            next_round_id = _checkpoint_nonnegative_int(
                state_dict.get(
                    "next_round_id",
                    next_game_id // self.config.population.environments_per_rank,
                )
            )
            actor_version = _checkpoint_nonnegative_int(
                state_dict["published_actor_version"]
            )
            if self.source._assignments is None:
                per_rank = self.config.population.environments_per_rank
                if next_game_id % per_rank or next_game_id < next_round_id * per_rank:
                    raise CollectorCheckpointError(_INVALID_COLLECTOR_STATE)

            restored_rng = random.Random()
            restored_rng.setstate(cast(tuple[Any, ...], state_dict["collector_rng"]))
            actor_state = state_dict["published_actor_state"]
            if not isinstance(actor_state, Mapping):
                raise CollectorCheckpointError(_INVALID_COLLECTOR_STATE)
            restored_actor_state: dict[str, torch.Tensor] = {}
            for name, tensor in actor_state.items():
                if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
                    raise CollectorCheckpointError(_INVALID_COLLECTOR_STATE)
                restored_actor_state[name] = tensor.detach().to("cpu", copy=True)
            restored_pool = self.source.pool
            if "population_pool" in state_dict:
                pool_payload = state_dict["population_pool"]
                restored_pool = None
                if pool_payload is not None:
                    identity = SnapshotPoolIdentity.model_validate(pool_payload)
                    restored_pool = SnapshotPool.reopen(
                        identity,
                        structure=structural_fingerprint(self.config),
                        seed=self.config.population.population_seed,
                        capacity=self.config.population.pool_capacity,
                    )
        except CollectorCheckpointError:
            raise
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise CollectorCheckpointError(_INVALID_COLLECTOR_STATE) from error
        self.source.next_game_id = next_game_id
        self.source._next_round_id = next_round_id
        self.source.rng = restored_rng
        self.source.actor_version = actor_version
        self.source.actor_state = restored_actor_state
        self.source.pool = restored_pool
        self.source._restored_actor = True

    def consume_restored_actor(self) -> bool:
        """Return and clear whether checkpoint load supplied the actor snapshot."""
        restored = self.source._restored_actor
        self.source._restored_actor = False
        return restored
