"""Synchronous, typed collection batches for the reference Toad backend."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import lightning
import torch
from torch.utils.data import DataLoader, IterableDataset

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
    """Collection failed while retaining the exact assigned game provenance."""


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

        collected_steps = sum(
            int(trajectory.dones.shape[0]) for trajectory in trajectories
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

    def _ensure_initial_pool(self) -> None:
        """Materialize declared first-round snapshots before opponent selection."""
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

        initial: list[tuple[dict[str, torch.Tensor], str]] = []
        for path in population.initial_snapshots:
            state = _checkpoint_policy_state(path)
            _strictly_validate_policy_state(self.config.model, state)
            initial.append((state, f"initial:{path}"))
        if population.snapshot_at_start:
            if not self.actor_state:
                raise RuntimeError(
                    "snapshot_at_start requires a published actor before collection"
                )
            _strictly_validate_policy_state(self.config.model, self.actor_state)
            initial.append((dict(self.actor_state), "actor-at-start"))

        store = SnapshotStore(
            self.config.runtime.output_dir / "population",
            capacity=population.pool_capacity,
            structure=structure,
        )
        for index, (state, run_id) in enumerate(initial):
            store.add(
                state,
                environment_steps=index,
                round_id=0,
                run_id=run_id,
            )
        self.pool = SnapshotPool.from_store(
            store,
            seed=population.population_seed,
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
            matches = [
                entry
                for entry in self.pool.manifest.entries
                if entry.sha256 == binding.sha256 and entry.path == binding.checkpoint
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
                    raise self._collection_error(chunk[0]) from error
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
    def _collection_error(assignment: CollectionAssignment) -> CollectionError:
        """Build the stable error that identifies one failed worker input."""
        return CollectionError(
            "collection failed for "
            f"game_id={assignment.game_id} seed={assignment.seed} "
            f"opponent={assignment.opponent_id}"
        )

    def __iter__(self) -> Iterator[LearnerBatch]:
        """Collect one assignment set, then expose it as one logical round."""
        self._ensure_initial_pool()
        assignments = self._assignments
        generated = assignments is None
        first_game_id = self.next_game_id
        round_id = self._next_round_id
        if assignments is None:
            assignments = allocate_round(
                self.config,
                first_game_id,
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
            collection_metrics[f"collection/games_by_opponent/{safe_id}"] = len(
                opponent_game_ids[opponent_id]
            )
            collection_metrics[f"collection/return_by_opponent/{safe_id}"] = float(
                torch.stack(
                    [
                        getattr(trajectory, self.config.curriculum.reward_field)
                        .float()
                        .sum()
                        for trajectory in opponent_rows
                    ]
                ).mean()
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
                first_game_id + self.config.population.environments_per_rank
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

    def publish_manifest(self, manifest: SnapshotManifest) -> None:
        """Bind collection to the callback's exact durable population view."""
        directory = self.population_directory()
        store = SnapshotStore(
            directory,
            capacity=self.config.population.pool_capacity,
            structure=structural_fingerprint(self.config),
        )
        if store.manifest != manifest:
            raise SnapshotIntegrityError(
                "published population manifest does not match durable store"
            )
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

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
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
            if self.source._assignments is None and next_game_id != (
                next_round_id * self.config.population.environments_per_rank
            ):
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
