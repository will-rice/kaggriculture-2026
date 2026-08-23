"""Synchronous, typed collection batches for the reference Toad backend."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, cast

import lightning
import torch
from torch.utils.data import DataLoader, IterableDataset

from kaggriculture.learn.rollout import Trajectory
from kaggriculture.learn.toad.config import ToadConfig
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
OBSERVED_FIELDS = ("board", "scalars", "positions")
WorkerInput = tuple[dict[str, torch.Tensor], list[int], int, int, str | None, float]


class BatchKind(StrEnum):
    """Which policy population supplied a learner batch."""

    SELFPLAY = "selfplay"
    SCRIPTED = "scripted"


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


@dataclass(frozen=True)
class CollectionAssignment:
    """One game assigned to the synchronous reference collector."""

    game_id: int
    seed: int
    opponent_id: str
    kind: BatchKind


class CollectionError(RuntimeError):
    """Collection failed while retaining the exact assigned game provenance."""


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
    return [
        {
            **{
                name: getattr(trajectory, name)[start : start + unroll_length]
                for name in ACTED_FIELDS
            },
            **{
                name: getattr(trajectory, name)[
                    torch.arange(start, start + unroll_length + 1).clamp(max=turns - 1)
                ]
                for name in OBSERVED_FIELDS
            },
        }
        for start in range(
            turns % unroll_length, turns - unroll_length + 1, unroll_length
        )
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
        self, trajectories: Sequence[Trajectory], meta: RoundMeta
    ) -> Iterator[LearnerBatch]:
        """Yield fresh policy batches followed by deterministic value replays."""
        all_segments = [
            segment
            for trajectory in trajectories
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

        pending: list[tuple[tuple[dict[str, torch.Tensor], ...], bool]] = [
            (group, False) for group in groups
        ]
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
        for index, (batch_segments, baseline_only) in enumerate(pending):
            yield LearnerBatch(
                segments=batch_segments,
                kind=meta.kind,
                baseline_only=baseline_only,
                first_of_round=index == 0,
                end_of_round=index == len(pending) - 1,
                collected_steps=collected_steps if index == 0 else 0,
                round_id=meta.round_id,
                actor_version=meta.actor_version,
                game_ids=meta.game_ids,
                opponent_ids=meta.opponent_ids,
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

    def publish_actor(
        self, state_dict: Mapping[str, torch.Tensor], version: int
    ) -> None:
        """Install the immutable CPU snapshot used by the next collection."""
        self.actor_state = dict(state_dict)
        self.actor_version = version

    def collect_assignment(
        self, assignment: CollectionAssignment
    ) -> Sequence[Trajectory]:
        """Collect one assigned game through the existing reference collector."""
        if not self.actor_state:
            raise RuntimeError("reference collection needs a published actor state")
        from kaggriculture.learn.scripts.toad import _play

        return _play(self._worker_input(assignment))

    def _worker_input(self, assignment: CollectionAssignment) -> WorkerInput:
        """Return the legacy worker tuple for one provenance-bearing game."""
        versus = (
            assignment.opponent_id if assignment.kind is BatchKind.SCRIPTED else None
        )
        return (
            self.actor_state,
            [assignment.seed],
            self.config.model.blocks,
            self.config.model.channels,
            versus,
            self.config.curriculum.money_weight,
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
        from kaggriculture.learn.scripts.toad import _play

        with ProcessPoolExecutor(
            max_workers=self.config.population.collection_processes
        ) as pool:
            results = iter(pool.map(_play, map(self._worker_input, assignments)))
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
        grouped: dict[
            BatchKind, list[tuple[CollectionAssignment, Sequence[Trajectory]]]
        ] = {
            BatchKind.SCRIPTED: [],
            BatchKind.SELFPLAY: [],
        }
        for assignment, trajectories in collected:
            grouped[assignment.kind].append((assignment, trajectories))

        round_id = self._next_round_id
        self._next_round_id += 1
        expander = RoundBatchExpander(
            batch_segments=self.config.optimizer.batch_segments,
            unroll_length=self.config.optimizer.unroll_length,
            value_passes=self.config.optimizer.value_passes,
            seed=self.config.runtime.seed,
        )
        batches: list[LearnerBatch] = []
        total_steps = 0
        for kind in (BatchKind.SCRIPTED, BatchKind.SELFPLAY):
            entries = grouped[kind]
            if not entries:
                continue
            kind_assignments = tuple(entry[0] for entry in entries)
            trajectories = [
                trajectory for _, assigned in entries for trajectory in assigned
            ]
            total_steps += sum(
                int(trajectory.dones.shape[0]) for trajectory in trajectories
            )
            meta = RoundMeta(
                round_id=round_id,
                actor_version=self.actor_version,
                game_ids=tuple(item.game_id for item in kind_assignments),
                seeds=tuple(item.seed for item in kind_assignments),
                opponent_ids=tuple(item.opponent_id for item in kind_assignments),
                kind=kind,
            )
            batches.extend(expander.expand(trajectories, meta))
        for index, batch in enumerate(batches):
            yield replace(
                batch,
                first_of_round=index == 0,
                end_of_round=index == len(batches) - 1,
                collected_steps=total_steps if index == 0 else 0,
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
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore the collector stream without retaining stale actor weights."""
        self.source.next_game_id = cast(int, state_dict["next_game_id"])
        self.source._next_round_id = (
            self.source.next_game_id // self.config.population.environments_per_rank
        )
        self.source.rng.setstate(cast(tuple[Any, ...], state_dict["collector_rng"]))
        self.source.actor_version = cast(int, state_dict["published_actor_version"])
