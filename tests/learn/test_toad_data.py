"""Typed optimizer-boundary batches for the native Toad collector."""

from collections.abc import Sequence

import pytest
import torch

from kaggriculture.learn.encoding import (
    MARKET_SLOTS,
    MAX_UNITS,
    QUANTITIES,
    SCALARS,
    TILE_PLANES,
    UNIT_OPS,
)
from kaggriculture.learn.rollout import Trajectory
from kaggriculture.learn.toad.config import ToadConfig
from kaggriculture.learn.toad.data import (
    BatchKind,
    CollectionAssignment,
    CollectionError,
    ReferenceRoundSource,
    RoundBatchExpander,
    RoundMeta,
    ToadDataModule,
)


def _trajectory(turns: int) -> Trajectory:
    """Return a complete, cheap recorded episode with a terminal final row."""
    slots = len(MARKET_SLOTS) + 2
    dones = torch.zeros(turns, dtype=torch.bool)
    dones[-1] = True
    rewards = torch.ones(turns)
    return Trajectory(
        board=torch.zeros(turns, TILE_PLANES, 10, 10),
        scalars=torch.zeros(turns, SCALARS),
        positions=torch.zeros(turns, MAX_UNITS, dtype=torch.int64),
        unit_actions=torch.zeros(turns, MAX_UNITS, dtype=torch.int64),
        unit_quantities=torch.zeros(turns, MAX_UNITS, dtype=torch.int64),
        market_actions=torch.zeros(turns, slots, dtype=torch.int64),
        unit_masks=torch.ones(turns, MAX_UNITS, len(UNIT_OPS), dtype=torch.bool),
        unit_quantity_masks=torch.ones(
            turns, MAX_UNITS, len(QUANTITIES), dtype=torch.bool
        ),
        market_masks=torch.ones(turns, slots, len(QUANTITIES), dtype=torch.bool),
        log_probs=torch.zeros(turns),
        values=torch.zeros(turns),
        rewards=rewards,
        own=rewards,
        shaped=rewards,
        shaped_money=rewards,
        margin=rewards,
        sparse=rewards,
        potentials=torch.zeros(turns, 1),
        dones=dones,
        final_margin=1.0,
        final_bank=1.0,
        final_capital=1.0,
        illegal=0,
        sales=0.0,
        units_sold=0.0,
        mean_sale_price=0.0,
        realisation=0.0,
        bought=0.0,
    )


def test_round_expansion_preserves_policy_and_value_pass_counts() -> None:
    """Dropping a fresh group or counting a replay as collection is a bug."""
    trajectories = [_trajectory(turns=32) for _ in range(2)]
    expander = RoundBatchExpander(batch_segments=4, value_passes=2, seed=7)

    batches = list(expander.expand(trajectories, RoundMeta.control(round_id=3)))

    policy = [batch for batch in batches if not batch.baseline_only]
    replay = [batch for batch in batches if batch.baseline_only]
    assert len(policy) == 1
    assert len(replay) == 2
    assert all(len(batch.segments) == 4 for batch in batches)
    assert batches[0].first_of_round
    assert batches[-1].end_of_round
    assert sum(batch.collected_steps for batch in batches) == 64
    assert all(batch.collected_steps == 0 for batch in batches[1:])


def test_collection_error_names_the_failed_game() -> None:
    """A collection failure must retain the assignment that actually failed."""
    assignment = CollectionAssignment(
        game_id=41,
        seed=99,
        opponent_id="economic",
        kind=BatchKind.SCRIPTED,
    )

    def fail(_: CollectionAssignment) -> Sequence[Trajectory]:
        raise RuntimeError("boom")

    source = ReferenceRoundSource(
        ToadConfig.control(), assignments=(assignment,), collect_assignment=fail
    )

    with pytest.raises(CollectionError, match="game_id=41 seed=99 opponent=economic"):
        next(iter(source))


def test_data_module_detaches_actor_before_publication() -> None:
    """Collector publication must never retain a gradient-bearing learner tensor."""
    source = ReferenceRoundSource(
        ToadConfig.control(), assignments=(), collect_assignment=lambda _: ()
    )
    module = ToadDataModule(ToadConfig.control(), source)
    weight = torch.ones(2, requires_grad=True)

    module.publish_actor({"weight": weight}, version=4)

    assert source.actor_version == 4
    assert source.actor_state["weight"].device.type == "cpu"
    assert not source.actor_state["weight"].requires_grad
    loader = module.train_dataloader()
    assert loader.batch_size is None
    assert loader.num_workers == 0
