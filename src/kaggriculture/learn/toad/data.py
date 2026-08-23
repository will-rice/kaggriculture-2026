"""Synchronous, typed collection batches for the reference Toad backend."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

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
        self._collector = collect_assignment or self.collect_assignment
        self.actor_state: dict[str, torch.Tensor] = {}
        self.actor_version = 0
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
        from concurrent.futures import ProcessPoolExecutor

        from kaggriculture.learn.scripts.toad import _collect

        econ_fraction = 0.0 if assignment.kind is BatchKind.SELFPLAY else 1.0
        with ProcessPoolExecutor(max_workers=1) as pool:
            mirror, scripted = _collect(
                pool,
                self.actor_state,
                (assignment.seed,),
                self.config.model.blocks,
                self.config.model.channels,
                econ_fraction,
            )
        return mirror + scripted

    def __iter__(self) -> Iterator[LearnerBatch]:
        """Collect one assignment set, then expose it as one logical round."""
        assignments = self._assignments
        if assignments is None:
            assignments = tuple(
                CollectionAssignment(
                    game_id=game_id,
                    seed=game_id + self.config.runtime.seed,
                    opponent_id="self",
                    kind=BatchKind.SELFPLAY,
                )
                for game_id in range(self.config.population.environments_per_rank)
            )
        if not assignments:
            return
        if any(
            assignment.kind is not assignments[0].kind for assignment in assignments
        ):
            raise ValueError("one collection round must contain one batch kind")

        trajectories: list[Trajectory] = []
        for assignment in assignments:
            try:
                trajectories.extend(self._collector(assignment))
            except Exception as error:
                raise CollectionError(
                    "collection failed for "
                    f"game_id={assignment.game_id} seed={assignment.seed} "
                    f"opponent={assignment.opponent_id}"
                ) from error
        meta = RoundMeta(
            round_id=self._next_round_id,
            actor_version=self.actor_version,
            game_ids=tuple(assignment.game_id for assignment in assignments),
            seeds=tuple(assignment.seed for assignment in assignments),
            opponent_ids=tuple(assignment.opponent_id for assignment in assignments),
            kind=assignments[0].kind,
        )
        self._next_round_id += 1
        yield from RoundBatchExpander(
            batch_segments=self.config.optimizer.batch_segments,
            unroll_length=self.config.optimizer.unroll_length,
            value_passes=self.config.optimizer.value_passes,
            seed=self.config.runtime.seed,
        ).expand(trajectories, meta)


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
