"""Synchronous, typed collection batches for the reference Toad backend."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, cast

import lightning
import torch
from torch.utils.data import DataLoader, IterableDataset

from kaggriculture.learn.rollout import Trajectory, segment_starts
from kaggriculture.learn.toad.config import ModelConfig, ToadConfig
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


class BatchKind(StrEnum):
    """Which policy population supplied a learner batch."""

    SELFPLAY = "selfplay"
    SCRIPTED = "scripted"
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
    round_metrics: Mapping[str, float | int] = field(default_factory=dict)


@dataclass(frozen=True)
class CollectionAssignment:
    """One game assigned to the synchronous reference collector."""

    game_id: int
    seed: int
    opponent_id: str
    kind: BatchKind


class CollectionError(RuntimeError):
    """Collection failed while retaining the exact assigned game provenance."""


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
        kind_metas: Mapping[BatchKind, RoundMeta] | None = None,
        round_metrics: Mapping[str, float | int] | None = None,
    ) -> Iterator[LearnerBatch]:
        """Yield fresh policy batches followed by deterministic value replays."""
        kinds = trajectory_kinds or [meta.kind] * len(trajectories)
        all_segments = [
            (segment, kind)
            for trajectory, kind in zip(trajectories, kinds, strict=True)
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
            tuple[tuple[tuple[dict[str, torch.Tensor], BatchKind], ...], bool]
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
    ) -> None:
        self.config = config
        self._assignments = tuple(assignments) if assignments is not None else None
        self._collector = collect_assignment
        self.actor_state: dict[str, torch.Tensor] = {}
        self.actor_version = 0
        self.next_game_id = 0
        self.rng = random.Random(config.runtime.seed)
        self._next_round_id = 0
        self._restored_actor = False

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

        return _play_reference(self._worker_input(assignment))

    def _worker_input(self, assignment: CollectionAssignment) -> ReferenceWorkerInput:
        """Return one resolved typed worker request with no architecture drift."""
        versus = None
        if assignment.kind is BatchKind.SCRIPTED:
            from kaggriculture.learn.scripts.toad import OPPONENT

            versus = (
                OPPONENT
                if assignment.opponent_id == "economic"
                else assignment.opponent_id
            )
        return ReferenceWorkerInput(
            actor_state=self.actor_state,
            seeds=[assignment.seed],
            model=self.config.model,
            versus=versus,
            money_weight=self.config.curriculum.money_weight,
            unroll_length=self.config.optimizer.unroll_length,
        )

    def _collect_round(
        self, assignments: Sequence[CollectionAssignment]
    ) -> list[tuple[CollectionAssignment, Sequence[Trajectory]]]:
        """Collect all assignments through one typed, round-scoped pool."""
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

        with ProcessPoolExecutor(
            max_workers=self.config.population.collection_processes
        ) as pool:
            results = iter(
                pool.map(_play_reference, map(self._worker_input, assignments))
            )
            collected = []
            for assignment in assignments:
                try:
                    collected.append((assignment, next(results)))
                except Exception as error:
                    raise self._collection_error(assignment) from error
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
        assignments = self._assignments
        if assignments is None:
            first_game_id = self.next_game_id
            environments = self.config.population.environments_per_rank
            self.next_game_id += environments
            scripted = int(environments * self.config.population.scripted)
            assignments = tuple(
                CollectionAssignment(
                    game_id=game_id,
                    seed=game_id + self.config.runtime.seed,
                    opponent_id=(
                        self.config.population.scripted_opponent
                        if game_id - first_game_id < scripted
                        else "self"
                    ),
                    kind=(
                        BatchKind.SCRIPTED
                        if game_id - first_game_id < scripted
                        else BatchKind.SELFPLAY
                    ),
                )
                for game_id in range(first_game_id, self.next_game_id)
            )
        if not assignments:
            return
        collected = self._collect_round(assignments)
        ordered = sorted(
            collected,
            key=lambda entry: 0 if entry[0].kind is BatchKind.SELFPLAY else 1,
        )
        round_id = self._next_round_id
        self._next_round_id += 1
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
        present = set(trajectory_kinds)
        round_kind = next(iter(present)) if len(present) == 1 else BatchKind.MIXED
        meta = RoundMeta(
            round_id=round_id,
            actor_version=self.actor_version,
            game_ids=tuple(item.game_id for item in ordered_assignments),
            seeds=tuple(item.seed for item in ordered_assignments),
            opponent_ids=tuple(item.opponent_id for item in ordered_assignments),
            kind=round_kind,
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

        yield from expander.expand(
            trajectories,
            meta,
            trajectory_kinds=trajectory_kinds,
            kind_metas=kind_metas,
            round_metrics=_collection_metrics(
                mirror, scripted, self.config.curriculum.reward_field
            ),
        )


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

    def train_dataloader(self) -> DataLoader[LearnerBatch]:
        """Return the intentionally single-process iterable loader."""
        return DataLoader(
            RoundIterableDataset(self.source, self.config),
            batch_size=None,
            num_workers=0,
        )

    def state_dict(self) -> dict[str, object]:
        """Serialize the stream position needed for the next collection."""
        return {
            "next_game_id": self.source.next_game_id,
            "collector_rng": self.source.rng.getstate(),
            "published_actor_version": self.source.actor_version,
            "published_actor_state": {
                name: tensor.detach().to("cpu", copy=True)
                for name, tensor in self.source.actor_state.items()
            },
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore the collector stream without retaining stale actor weights."""
        self.source.next_game_id = cast(int, state_dict["next_game_id"])
        self.source._next_round_id = (
            self.source.next_game_id // self.config.population.environments_per_rank
        )
        self.source.rng.setstate(cast(tuple[Any, ...], state_dict["collector_rng"]))
        self.source.actor_version = cast(int, state_dict["published_actor_version"])
        actor_state = cast(
            Mapping[str, torch.Tensor], state_dict["published_actor_state"]
        )
        self.source.actor_state = {
            name: tensor.detach().to("cpu", copy=True)
            for name, tensor in actor_state.items()
        }
        self.source._restored_actor = True

    def consume_restored_actor(self) -> bool:
        """Return and clear whether checkpoint load supplied the actor snapshot."""
        restored = self.source._restored_actor
        self.source._restored_actor = False
        return restored
