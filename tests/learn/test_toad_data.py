"""Typed optimizer-boundary batches for the native Toad collector."""

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import kaggriculture.learn.rollout as rollout_module
from kaggriculture.learn.encoding import (
    MARKET_SLOTS,
    MAX_UNITS,
    QUANTITIES,
    SCALARS,
    TILE_PLANES,
    UNIT_OPS,
)
from kaggriculture.learn.rollout import Trajectory
from kaggriculture.learn.scripts import toad
from kaggriculture.learn.toad.config import ToadConfig
from kaggriculture.learn.toad.data import (
    BatchKind,
    CollectionAssignment,
    CollectionError,
    OpponentAssignment,
    ReferenceRoundSource,
    ReferenceWorkerInput,
    RoundBatchExpander,
    RoundMeta,
    ToadDataModule,
    allocate_round,
    segments,
)
from kaggriculture.learn.toad.model import StatefulPolicy
from kaggriculture.learn.toad.population import (
    EmptySnapshotPoolError,
    SnapshotEntry,
    SnapshotIntegrityError,
    SnapshotPool,
    SnapshotStore,
    sha256_file,
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


def _allocation_config(
    *,
    selfplay: float,
    scripted: float,
    frozen: float,
    teacher: float,
    environments: int,
    teacher_checkpoint: Path | None = None,
) -> ToadConfig:
    """Build allocation settings without bypassing relationship validation."""
    return ToadConfig.model_validate(
        {
            "population": {
                "selfplay": selfplay,
                "scripted": scripted,
                "frozen_opponent": frozen,
                "teacher_distill": teacher,
                "teacher_checkpoint": teacher_checkpoint,
                "environments_per_rank": environments,
                "snapshot_at_start": frozen > 0,
            }
        },
        context={"historical": True},
    )


def test_round_allocator_realizes_largest_remainder_quotas_and_contiguous_ids(
    tmp_path: Path,
) -> None:
    """Fractional/tied quotas cannot lose an environment or perturb global IDs."""
    torch.save({"weight": torch.ones(1)}, tmp_path / "teacher.pt")
    config = _allocation_config(
        selfplay=0.25,
        scripted=0.25,
        frozen=0.25,
        teacher=0.25,
        environments=7,
        teacher_checkpoint=tmp_path / "teacher.pt",
    )
    store = SnapshotStore(tmp_path / "pool", capacity=1, structure="model-v1")
    store.add(
        {"weight": torch.ones(1)},
        environment_steps=1,
        round_id=0,
        run_id="run",
    )
    pool = SnapshotPool.from_store(store, seed=19)

    assignments = allocate_round(
        config,
        start_game_id=100,
        round_id=2,
        pool=pool,
    )

    assert Counter(item.kind for item in assignments) == {
        BatchKind.FROZEN_OPPONENT: 2,
        BatchKind.SCRIPTED: 2,
        BatchKind.SELFPLAY: 2,
        BatchKind.TEACHER_DISTILL: 1,
    }
    assert [item.game_id for item in assignments] == list(range(100, 107))
    assert [item.seed for item in assignments] == list(range(100, 107))
    assert assignments == allocate_round(
        config,
        start_game_id=100,
        round_id=2,
        pool=pool,
    )


def test_round_allocator_rejects_frozen_quota_without_a_pool() -> None:
    """A selected frozen slot cannot silently become self-play."""
    config = _allocation_config(
        selfplay=0.0,
        scripted=0.0,
        frozen=1.0,
        teacher=0.0,
        environments=1,
    )

    with pytest.raises(EmptySnapshotPoolError):
        allocate_round(config, 0, 0, SnapshotPool.empty(), None)


def test_round_allocator_asserts_zero_and_uneven_quotas_exactly() -> None:
    """Zero-probability kinds stay absent when one remainder goes to scripted."""
    config = _allocation_config(
        selfplay=0.1,
        scripted=0.9,
        frozen=0.0,
        teacher=0.0,
        environments=3,
    )

    counts = Counter(item.kind for item in allocate_round(config, 20, 1))

    actual = {kind: counts[kind] for kind in BatchKind if kind is not BatchKind.MIXED}
    assert actual == {
        BatchKind.SELFPLAY: 0,
        BatchKind.SCRIPTED: 3,
        BatchKind.FROZEN_OPPONENT: 0,
        BatchKind.TEACHER_DISTILL: 0,
    }


def test_opponent_assignment_uses_the_population_stage_positional_order() -> None:
    """The public assignment API orders kind before opponent identity."""
    assignment = OpponentAssignment(
        9,
        10,
        BatchKind.SCRIPTED,
        "economic",
        None,
        None,
    )

    assert assignment.kind is BatchKind.SCRIPTED
    assert assignment.opponent_id == "economic"


def test_teacher_assignment_carries_the_actual_checkpoint_digest(
    tmp_path: Path,
) -> None:
    """Teacher provenance must identify checkpoint bytes, not only a path string."""
    checkpoint = tmp_path / "teacher.pt"
    torch.save({"weight": torch.tensor([3.0])}, checkpoint)
    config = _allocation_config(
        selfplay=0.0,
        scripted=0.0,
        frozen=0.0,
        teacher=1.0,
        environments=1,
        teacher_checkpoint=checkpoint,
    )

    assignment = allocate_round(config, 11, 4)[0]

    assert assignment.kind is BatchKind.TEACHER_DISTILL
    assert assignment.checkpoint == checkpoint
    assert assignment.checkpoint_sha256 == sha256_file(checkpoint)
    assert assignment.checkpoint_sha256 is not None
    assert assignment.checkpoint_sha256[:12] in assignment.opponent_id


def test_round_source_loads_one_selected_frozen_digest_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated games validate and materialize one snapshot once per round."""
    config = ToadConfig.model_validate(
        {
            "model": {"blocks": 1, "channels": 4},
            "population": {
                "selfplay": 0.0,
                "scripted": 0.0,
                "frozen_opponent": 1.0,
                "environments_per_rank": 2,
            },
        }
    )
    opponent = toad.Policy(blocks=1, channels=4, value_bound=toad.VALUE_BOUND)
    store = SnapshotStore(tmp_path / "pool", capacity=1, structure="model-v1")
    entry = store.add(
        opponent.state_dict(),
        environment_steps=9,
        round_id=1,
        run_id="frozen-run",
    )
    pool = SnapshotPool.from_store(store, seed=3)
    loads: list[str] = []
    real_load = pool.load

    def counted_load(selected: SnapshotEntry) -> dict[str, torch.Tensor]:
        loads.append(entry.sha256)
        return real_load(selected)

    monkeypatch.setattr(pool, "load", counted_load)
    seen: list[CollectionAssignment] = []

    def collect(assignment: CollectionAssignment) -> tuple[Trajectory]:
        seen.append(assignment)
        return (_trajectory(64),)

    source = ReferenceRoundSource(config, pool=pool, collect_assignment=collect)
    batches = list(source)

    assert loads == [entry.sha256]
    assert {assignment.checkpoint_sha256 for assignment in seen} == {entry.sha256}
    assert all(batch.kind is BatchKind.FROZEN_OPPONENT for batch in batches)
    assert all(set(batch.opponent_digests) == {entry.sha256} for batch in batches)
    assert {batch.segment_game_ids[0] for batch in batches} == {0, 1}
    assert all(
        len(batch.segment_game_ids)
        == len(batch.segment_opponent_ids)
        == len(batch.segment_opponent_digests)
        == len(batch.segment_kinds)
        == 4
        for batch in batches
    )
    assert all(
        set(batch.segment_opponent_digests) == {entry.sha256} for batch in batches
    )


def test_selected_corrupt_frozen_member_aborts_before_collection(
    tmp_path: Path,
) -> None:
    """Collector callbacks never receive a substituted game after digest failure."""
    config = ToadConfig.model_validate(
        {
            "population": {
                "selfplay": 0.0,
                "scripted": 0.0,
                "frozen_opponent": 1.0,
                "environments_per_rank": 1,
            }
        }
    )
    actor = toad.Policy(
        blocks=config.model.blocks,
        channels=config.model.channels,
        value_bound=toad.VALUE_BOUND,
    )
    store = SnapshotStore(tmp_path / "pool", capacity=1, structure="model-v1")
    entry = store.add(actor.state_dict(), environment_steps=1, round_id=0, run_id="run")
    pool = SnapshotPool.from_store(store, seed=0)
    entry.path.write_bytes(b"corrupt")
    collected: list[int] = []

    def collect_unexpected(assignment: CollectionAssignment) -> tuple[Trajectory, ...]:
        collected.append(assignment.game_id)
        return ()

    source = ReferenceRoundSource(
        config,
        pool=pool,
        collect_assignment=collect_unexpected,
    )

    with pytest.raises(SnapshotIntegrityError, match=entry.sha256):
        list(source)

    assert collected == []


def test_reference_worker_strictly_builds_a_frozen_neural_opponent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frozen opponents use the actor architecture seam and immutable eval weights."""
    config = ToadConfig.model_validate(
        {"model": {"blocks": 1, "channels": 4, "recurrent": True}}
    )
    actor = StatefulPolicy(config.model)
    opponent = StatefulPolicy(config.model)
    seen: list[StatefulPolicy] = []

    def fake_rollout(
        learner: object,
        frozen: object,
        seeds: Sequence[int],
        *,
        state_unroll_length: int | None = None,
    ) -> list[Trajectory]:
        assert isinstance(learner, StatefulPolicy)
        assert isinstance(frozen, StatefulPolicy)
        assert frozen is not learner
        assert list(seeds) == [31]
        assert not frozen.training
        assert not any(parameter.requires_grad for parameter in frozen.parameters())
        seen.append(frozen)
        return []

    monkeypatch.setattr(toad, "rollout_many", fake_rollout)
    work = ReferenceWorkerInput(
        actor_state=dict(actor.state_dict()),
        seeds=[31],
        model=config.model,
        versus=None,
        money_weight=0.01,
        opponent_state=dict(opponent.state_dict()),
        opponent_model=config.model,
    )

    assert toad._play_reference(work) == []
    assert tuple(seen[0].state_dict()) == tuple(opponent.state_dict())


def test_snapshot_at_start_populates_before_first_frozen_selection(
    tmp_path: Path,
) -> None:
    """The declared actor snapshot must exist before game zero samples the pool."""
    config = ToadConfig.model_validate(
        {
            "model": {"blocks": 1, "channels": 4},
            "population": {
                "selfplay": 0.0,
                "scripted": 0.0,
                "frozen_opponent": 1.0,
                "environments_per_rank": 1,
                "snapshot_at_start": True,
            },
            "runtime": {"output_dir": tmp_path / "run"},
        }
    )
    actor = toad.Policy(blocks=1, channels=4, value_bound=toad.VALUE_BOUND)
    seen: list[CollectionAssignment] = []
    source = ReferenceRoundSource(
        config,
        collect_assignment=lambda assignment: (
            seen.append(assignment) or (_trajectory(64),)
        ),
    )
    source.publish_actor(actor.state_dict(), version=5)

    batches = list(source)

    assert source.pool is not None
    assert len(source.pool.manifest.entries) == 1
    assert seen[0].checkpoint_sha256 == source.pool.manifest.entries[0].sha256
    assert batches[0].actor_version == 5


def test_initial_snapshot_is_strictly_checked_before_pool_publication(
    tmp_path: Path,
) -> None:
    """An incompatible initial checkpoint cannot acquire a trusted pool fingerprint."""
    checkpoint = tmp_path / "incompatible.pt"
    torch.save({"wrong": torch.ones(1)}, checkpoint)
    config = ToadConfig.model_validate(
        {
            "model": {"blocks": 1, "channels": 4},
            "population": {
                "selfplay": 0.0,
                "scripted": 0.0,
                "frozen_opponent": 1.0,
                "environments_per_rank": 1,
                "initial_snapshots": [checkpoint],
            },
            "runtime": {"output_dir": tmp_path / "run"},
        }
    )
    collected: list[int] = []

    def collect_unexpected(assignment: CollectionAssignment) -> tuple[Trajectory, ...]:
        collected.append(assignment.game_id)
        return ()

    source = ReferenceRoundSource(
        config,
        collect_assignment=collect_unexpected,
    )

    with pytest.raises(RuntimeError, match="Missing key|Unexpected key"):
        list(source)

    assert collected == []
    assert not (tmp_path / "run" / "population" / "manifest.json").exists()


def test_teacher_distill_source_materializes_real_checkpoint_provenance(
    tmp_path: Path,
) -> None:
    """Teacher slots collect learner trajectories under their exact checkpoint ID."""
    checkpoint = tmp_path / "teacher.pt"
    teacher = toad.Policy(blocks=1, channels=4, value_bound=toad.VALUE_BOUND)
    torch.save(teacher.state_dict(), checkpoint)
    config = ToadConfig.model_validate(
        {
            "model": {"blocks": 1, "channels": 4},
            "population": {
                "selfplay": 0.0,
                "scripted": 0.0,
                "teacher_distill": 1.0,
                "teacher_checkpoint": checkpoint,
                "teacher_blocks": 1,
                "environments_per_rank": 1,
            },
        }
    )
    seen: list[CollectionAssignment] = []
    source = ReferenceRoundSource(
        config,
        collect_assignment=lambda assignment: (
            seen.append(assignment) or (_trajectory(64),)
        ),
    )

    batches = list(source)

    digest = sha256_file(checkpoint)
    assert seen[0].checkpoint_sha256 == digest
    assert all(batch.kind is BatchKind.TEACHER_DISTILL for batch in batches)
    assert all(batch.opponent_digests == (digest,) for batch in batches)
    assert all(set(batch.segment_opponent_digests) == {digest} for batch in batches)


def _recurrent_trajectory(turns: int = 32, layers: int = 2) -> Trajectory:
    """Return a trajectory with distinct autograd-connected state per row."""
    base = _trajectory(turns)
    source = torch.arange((turns + 1) * layers * 3 * 10 * 10, dtype=torch.float32).view(
        turns + 1, layers, 3, 10, 10
    )
    source.requires_grad_()
    belief = torch.arange((turns + 1) * 9, dtype=torch.float32).view(turns + 1, 9)
    belief.requires_grad_()
    return replace(
        base,
        hidden=source * 2,
        cell=source.square(),
        prior_belief=belief.sigmoid(),
    )


def test_segments_store_only_detached_state_before_their_first_observation() -> None:
    """Start 16 must replay state 16, never actor state 15 or all later states."""
    trajectory = _recurrent_trajectory()
    assert trajectory.hidden is not None
    assert trajectory.cell is not None
    assert trajectory.prior_belief is not None

    batches = segments(trajectory, unroll_length=16)

    assert torch.equal(batches[0]["initial_hidden"], trajectory.hidden[0])
    assert torch.equal(batches[1]["initial_hidden"], trajectory.hidden[16])
    assert torch.equal(batches[1]["initial_cell"], trajectory.cell[16])
    assert torch.equal(batches[1]["initial_belief"], trajectory.prior_belief[16])
    assert "hidden" not in batches[1]
    assert "cell" not in batches[1]
    assert "prior_belief" not in batches[1]
    assert not batches[0]["initial_hidden"].requires_grad
    assert not batches[0]["initial_cell"].requires_grad
    assert not batches[0]["initial_belief"].requires_grad


@pytest.mark.parametrize(
    ("turns", "unroll_length", "expected"),
    [
        (32, 16, (0, 16)),
        (31, 16, (15,)),
        (719, 16, tuple(range(15, 704, 16))),
        (10, 4, (2, 6)),
    ],
)
def test_segment_starts_are_exactly_end_anchored(
    turns: int, unroll_length: int, expected: tuple[int, ...]
) -> None:
    """Collector and learner must share literal starts for every episode shape."""
    assert rollout_module.segment_starts(turns, unroll_length) == expected


def test_sparse_recurrent_segments_clone_exact_boundary_states() -> None:
    """Sparse multi-layer/belief state must detach and not retain trajectory storage."""
    dense = _recurrent_trajectory(turns=32, layers=2)
    assert dense.hidden is not None
    assert dense.cell is not None
    assert dense.prior_belief is not None
    starts = torch.tensor([0, 16], dtype=torch.int64)
    sparse = replace(
        dense,
        hidden=dense.hidden[starts],
        cell=dense.cell[starts],
        prior_belief=dense.prior_belief[starts],
        state_steps=starts,
    )

    batches = segments(sparse, 16)

    assert len(batches) == 2
    assert batches[1]["initial_hidden"].shape == (2, 3, 10, 10)
    assert batches[1]["initial_belief"].shape == (9,)
    sparse_hidden = sparse.hidden
    assert sparse_hidden is not None
    expected = sparse_hidden[1].detach().clone()
    sparse_hidden[1].detach().zero_()
    assert torch.equal(batches[1]["initial_hidden"], expected)
    assert batches[1]["initial_hidden"].untyped_storage().data_ptr() != (
        sparse_hidden.untyped_storage().data_ptr()
    )


def test_sparse_recurrent_segments_reject_a_missing_active_start() -> None:
    """An active recurrent trajectory may not silently initialize a missing unroll."""
    dense = _recurrent_trajectory(turns=32)
    assert dense.hidden is not None
    assert dense.cell is not None
    assert dense.prior_belief is not None
    sparse = replace(
        dense,
        hidden=dense.hidden[[0]],
        cell=dense.cell[[0]],
        prior_belief=dense.prior_belief[[0]],
        state_steps=torch.tensor([0]),
    )

    with pytest.raises(ValueError, match="missing recurrent state.*16"):
        segments(sparse, 16)


def test_default_recurrent_state_memory_is_boundary_sized_without_allocation() -> None:
    """The fixed 719-decision horizon stores 44, not 720, default-width states."""
    starts = rollout_module.segment_starts(719, 16)
    dense_elements = 720 * 2 * 128 * 10 * 10
    sparse_elements = len(starts) * 2 * 128 * 10 * 10

    assert len(starts) == 44
    assert starts[0] == 15
    assert starts[-1] == 703
    assert sparse_elements * 16 < dense_elements


def test_reference_worker_carries_the_optimizer_unroll_length() -> None:
    """Worker IPC must tell production rollout which boundary states to retain."""
    config = ToadConfig.model_validate(
        {
            "model": {"blocks": 1, "channels": 16, "recurrent": True},
            "optimizer": {"unroll_length": 7},
        }
    )
    source = ReferenceRoundSource(config)
    work = source._worker_input(CollectionAssignment(0, 1, "self", BatchKind.SELFPLAY))

    assert work.unroll_length == 7


def test_recurrent_reference_worker_receives_the_resolved_model_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker reconstruction must retain recurrent layers and exact stateful keys."""
    config = ToadConfig.model_validate(
        {
            "model": {
                "blocks": 1,
                "channels": 16,
                "kernel_size": 5,
                "activation": "leaky_relu",
                "recurrent": True,
                "recurrent_channels": 3,
                "recurrent_layers": 2,
            }
        }
    )
    policy = StatefulPolicy(config.model)
    assignment = CollectionAssignment(4, 12, "self", BatchKind.SELFPLAY)
    seen: list[tuple[object, dict[str, torch.Tensor]]] = []

    def fake_play(work: ReferenceWorkerInput) -> list[Trajectory]:
        seen.append((work.model, work.actor_state))
        return []

    monkeypatch.setattr(toad, "_play_reference", fake_play)
    source = ReferenceRoundSource(config, assignments=(assignment,))
    source.publish_actor(policy.state_dict(), version=3)

    source.collect_assignment(assignment)

    assert seen[0][0] == config.model
    assert tuple(seen[0][1]) == tuple(policy.state_dict())


def test_recurrent_reference_worker_rejects_bare_control_weights() -> None:
    """Enabled actor components may never survive as random worker parameters."""
    config = ToadConfig.model_validate(
        {
            "model": {
                "blocks": 1,
                "channels": 16,
                "recurrent": True,
                "recurrent_channels": 3,
            }
        }
    )
    bare = toad.Policy(blocks=1, channels=16, value_bound=toad.VALUE_BOUND)
    work = ReferenceWorkerInput(
        actor_state=dict(bare.state_dict()),
        seeds=[0],
        model=config.model,
        versus=None,
        money_weight=0.01,
    )

    with pytest.raises(RuntimeError, match="Missing key.*control"):
        toad._play_reference(work)


def test_recurrent_reference_worker_builds_and_loads_the_stateful_actor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real worker entrypoint must consume every resolved model dimension."""
    config = ToadConfig.model_validate(
        {
            "model": {
                "blocks": 1,
                "channels": 16,
                "kernel_size": 5,
                "activation": "leaky_relu",
                "recurrent": True,
                "recurrent_channels": 3,
                "recurrent_layers": 2,
            }
        }
    )
    expected = StatefulPolicy(config.model)
    seen: list[StatefulPolicy] = []

    def fake_rollout(
        actor: object,
        opponent: object,
        seeds: Sequence[int],
        *,
        state_unroll_length: int | None = None,
    ) -> list[Trajectory]:
        assert isinstance(actor, StatefulPolicy)
        assert opponent is actor
        assert list(seeds) == [19]
        assert state_unroll_length == 16
        seen.append(actor)
        return []

    monkeypatch.setattr(toad, "rollout_many", fake_rollout)
    work = ReferenceWorkerInput(
        actor_state=dict(expected.state_dict()),
        seeds=[19],
        model=config.model,
        versus=None,
        money_weight=0.01,
    )

    assert toad._play_reference(work) == []
    assert seen[0].config == config.model
    assert seen[0].control.stem.kernel_size == (5, 5)
    assert isinstance(seen[0].control.market[1], torch.nn.LeakyReLU)
    for name, tensor in expected.state_dict().items():
        assert torch.equal(seen[0].state_dict()[name], tensor)


@pytest.mark.parametrize(
    "model",
    [
        {"transformer": True, "transformer_blocks": 1},
        {"local_patch": True, "local_patch_blocks": 1},
        {"interaction_value": True},
    ],
)
def test_optional_reference_worker_builds_exact_stateful_actor(
    model: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Workers must construct and strictly load optional stateful weights."""
    config = ToadConfig.model_validate(
        {"model": {"blocks": 1, "channels": 16, **model}}
    )
    expected = StatefulPolicy(config.model)
    seen: list[StatefulPolicy] = []

    def fake_rollout(
        actor: object,
        opponent: object,
        seeds: Sequence[int],
        *,
        state_unroll_length: int | None = None,
    ) -> list[Trajectory]:
        assert isinstance(actor, StatefulPolicy)
        assert opponent is actor
        assert state_unroll_length == 16
        seen.append(actor)
        return []

    monkeypatch.setattr(toad, "rollout_many", fake_rollout)
    work = ReferenceWorkerInput(
        actor_state=dict(expected.state_dict()),
        seeds=[23],
        model=config.model,
        versus=None,
        money_weight=0.01,
    )

    assert toad._play_reference(work) == []
    for name, tensor in expected.state_dict().items():
        assert torch.equal(seen[0].state_dict()[name], tensor)


@pytest.mark.parametrize(
    "model",
    [
        {"transformer": True, "transformer_blocks": 1},
        {"local_patch": True, "local_patch_blocks": 1},
        {"interaction_value": True},
    ],
)
def test_optional_reference_worker_runs_real_stateless_rollout(
    model: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Typed worker reconstruction must cross the real decide seam unmocked."""
    monkeypatch.setattr(rollout_module, "EPISODE_STEPS", 3)
    config = ToadConfig.model_validate({"model": {"blocks": 1, "channels": 4, **model}})
    actor = StatefulPolicy(config.model)
    work = ReferenceWorkerInput(
        actor_state=dict(actor.state_dict()),
        seeds=[29],
        model=config.model,
        versus=None,
        money_weight=0.01,
    )

    trajectories = toad._play_reference(work)

    assert len(trajectories) == 2
    assert all(len(trajectory.dones) == 2 for trajectory in trajectories)
    assert all(trajectory.hidden is None for trajectory in trajectories)


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


def test_source_aggregates_assignments_into_one_logical_round() -> None:
    """Per-game boundaries would advance the scheduler and actor too often."""
    assignments = (
        CollectionAssignment(7, 17, "self", BatchKind.SELFPLAY),
        CollectionAssignment(8, 18, "self", BatchKind.SELFPLAY),
    )
    source = ReferenceRoundSource(
        ToadConfig.control(),
        assignments=assignments,
        collect_assignment=lambda _: (_trajectory(64),),
    )

    batches = list(source)

    assert {batch.round_id for batch in batches} == {0}
    assert sum(batch.end_of_round for batch in batches) == 1
    assert batches[-1].end_of_round
    assert batches[0].game_ids == (7, 8)
    assert batches[0].opponent_ids == ("self", "self")
    assert sum(batch.collected_steps for batch in batches) == 128


def test_default_round_honors_typed_population_quotas_and_one_clock() -> None:
    """Mixed opponent kinds remain one collection and scheduler boundary."""
    base = ToadConfig.control()
    config = base.model_copy(
        update={
            "population": base.population.model_copy(
                update={
                    "selfplay": 0.75,
                    "scripted": 0.25,
                    "environments_per_rank": 4,
                }
            )
        }
    )
    seen: list[CollectionAssignment] = []

    def collect(assignment: CollectionAssignment) -> Sequence[Trajectory]:
        seen.append(assignment)
        return (_trajectory(64),)

    source = ReferenceRoundSource(config, collect_assignment=collect)

    batches = list(source)

    assert [assignment.kind for assignment in seen].count(BatchKind.SCRIPTED) == 1
    assert [assignment.kind for assignment in seen].count(BatchKind.SELFPLAY) == 3
    assert [assignment.opponent_id for assignment in seen] == [
        "self",
        "self",
        "self",
        "economic",
    ]
    assert {batch.round_id for batch in batches} == {0}
    assert sum(batch.first_of_round for batch in batches) == 1
    assert sum(batch.end_of_round for batch in batches) == 1
    assert batches[0].first_of_round
    assert batches[-1].end_of_round
    assert batches[0].collected_steps == 256
    assert all(batch.collected_steps == 0 for batch in batches[1:])
    scripted = [batch for batch in batches if batch.kind is BatchKind.SCRIPTED]
    selfplay = [batch for batch in batches if batch.kind is BatchKind.SELFPLAY]
    assert scripted[0].game_ids == (3,)
    assert scripted[0].opponent_ids == ("economic",)
    assert selfplay[0].game_ids == (0, 1, 2)
    assert selfplay[0].opponent_ids == ("self", "self", "self")

    next_round = list(source)
    assert {batch.round_id for batch in next_round} == {1}
    assert next_round[0].game_ids == (5, 6, 7)


def test_default_collection_uses_one_typed_round_pool_and_fans_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Collection-process width applies once to every assignment in a round."""
    base = ToadConfig.control()
    config = base.model_copy(
        update={
            "population": base.population.model_copy(
                update={
                    "selfplay": 0.5,
                    "scripted": 0.5,
                    "environments_per_rank": 4,
                    "collection_processes": 3,
                }
            ),
            "curriculum": base.curriculum.model_copy(update={"money_weight": 0.01}),
        }
    )
    pool_widths: list[int] = []
    work: list[ReferenceWorkerInput] = []

    class FakePool:
        def __init__(self, max_workers: int) -> None:
            pool_widths.append(max_workers)

        def __enter__(self) -> "FakePool":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def map(
            self,
            function: Callable[[ReferenceWorkerInput], object],
            assignments: Iterable[ReferenceWorkerInput],
        ) -> Iterable[object]:
            assigned = list(assignments)
            work.extend(assigned)
            return map(function, assigned)

    monkeypatch.setattr("kaggriculture.learn.toad.data.ProcessPoolExecutor", FakePool)
    monkeypatch.setattr(toad, "_play_reference", lambda _: [_trajectory(64)])
    source = ReferenceRoundSource(config)
    source.publish_actor({"weight": torch.ones(1)}, version=7)

    list(source)

    assert pool_widths == [3]
    assert len(work) == 4
    assert all(item.actor_state == source.actor_state for item in work)
    assert [item.versus for item in work] == [
        toad.OPPONENT,
        None,
        None,
        toad.OPPONENT,
    ]
    assert all(item.money_weight == 0.01 for item in work)


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
    with torch.no_grad():
        weight.zero_()
    assert source.actor_state["weight"].tolist() == [1.0, 1.0]
    assert source.actor_state["weight"].data_ptr() != weight.data_ptr()
    loader = module.train_dataloader()
    assert loader.batch_size is None
    assert loader.num_workers == 0


def test_reference_worker_uses_the_typed_money_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker must not fall back to an inherited or process-local reward default."""
    seen: list[float] = []
    policy = toad.Policy(blocks=1, channels=16, value_bound=toad.VALUE_BOUND)

    def fake_rollout(*args: object, **kwargs: object) -> list[Trajectory]:
        seen.append(toad.money_weight())
        return []

    monkeypatch.setattr(toad, "rollout_many", fake_rollout)

    toad._play((policy.state_dict(), [0], 1, 16, None, 0.01))

    assert seen == [0.01]
