"""Typed optimizer-boundary batches for the native Toad collector."""

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch

import kaggriculture.learn.rollout as rollout_module
from kaggriculture.learn.encoding import (
    IGNORE,
    MARKET_SLOTS,
    MAX_UNITS,
    QUANTITIES,
    SCALARS,
    TILE_PLANES,
    UNIT_OPS,
)
from kaggriculture.learn.rollout import Trajectory
from kaggriculture.learn.scripts import toad
from kaggriculture.learn.toad.config import (
    ModelConfig,
    ToadConfig,
    structural_fingerprint,
)
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
from kaggriculture.learn.toad.lightning import ToadLightningModule
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


def _materialize_population_bootstrap(
    data_module: ToadDataModule,
    config: ToadConfig,
    actor_state: dict[str, torch.Tensor],
) -> None:
    """Model the rank-zero fit-start boundary in direct source tests."""
    store = SnapshotStore(
        data_module.population_directory(),
        capacity=config.population.pool_capacity,
        structure=structural_fingerprint(config),
    )
    data_module.materialize_population_bootstrap(
        store,
        actor_state,
        environment_steps=0,
        round_id=0,
        run_id=config.curriculum.phase,
    )
    data_module.publish_manifest(store.manifest)


def test_segments_freeze_unit_valid_from_original_encoded_actions() -> None:
    """Later learner action edits cannot redefine which observed slots existed."""
    trajectory = _trajectory(4)
    actions = trajectory.unit_actions.clone()
    actions[:, -1] = IGNORE

    segment = segments(replace(trajectory, unit_actions=actions), 4)[0]

    assert segment["unit_valid"].dtype is torch.bool
    assert not segment["unit_valid"][:, -1].any()
    assert segment["unit_valid"][:, :-1].all()


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
                "teacher": (
                    {
                        "checkpoint": teacher_checkpoint,
                        "operation": True,
                        "quantity": True,
                        "market": True,
                    }
                    if teacher_checkpoint is not None
                    else None
                ),
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
    store = SnapshotStore(
        tmp_path / "pool",
        capacity=1,
        structure=structural_fingerprint(config),
    )
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


def test_supplied_pool_semantic_structure_mismatch_fails_before_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same-shaped weights cannot cross a different resolved model fingerprint."""
    config = ToadConfig.model_validate(
        {
            "model": {"blocks": 1, "channels": 4, "activation": "leaky_relu"},
            "population": {
                "selfplay": 0.0,
                "scripted": 0.0,
                "frozen_opponent": 1.0,
                "environments_per_rank": 1,
                "snapshot_at_start": True,
            },
        }
    )
    foreign = config.model_copy(
        update={"model": config.model.model_copy(update={"activation": "relu"})}
    )
    policy = toad.Policy(
        blocks=1,
        channels=4,
        value_bound=toad.VALUE_BOUND,
        activation="leaky_relu",
    )
    store = SnapshotStore(
        tmp_path / "pool",
        capacity=1,
        structure=structural_fingerprint(foreign),
    )
    store.add(policy.state_dict(), environment_steps=1, round_id=0, run_id="run")
    pool = SnapshotPool.from_store(store, seed=config.population.population_seed)
    manifest_path = tmp_path / "pool" / "manifest.json"
    manifest_before = manifest_path.read_bytes()
    sampled: list[int] = []
    collected: list[int] = []
    real_sample = pool.sample

    def sample(game_id: int) -> SnapshotEntry:
        sampled.append(game_id)
        return real_sample(game_id)

    monkeypatch.setattr(pool, "sample", sample)
    with pytest.raises(SnapshotIntegrityError, match="structure.*config"):
        ReferenceRoundSource(
            config,
            pool=pool,
            collect_assignment=lambda assignment: (
                collected.append(assignment.game_id) or (_trajectory(64),)
            ),
        )

    assert sampled == []
    assert collected == []
    assert manifest_path.read_bytes() == manifest_before


def test_allocator_rebinds_external_pool_seed_to_population_seed(
    tmp_path: Path,
) -> None:
    """Pool-construction seed cannot alter configured frozen selections."""
    config = ToadConfig.model_validate(
        {
            "model": {"blocks": 1, "channels": 4},
            "population": {
                "selfplay": 0.0,
                "scripted": 0.0,
                "frozen_opponent": 1.0,
                "environments_per_rank": 8,
                "population_seed": 17,
                "snapshot_at_start": True,
            },
        }
    )
    policy = toad.Policy(blocks=1, channels=4, value_bound=toad.VALUE_BOUND)
    store = SnapshotStore(
        tmp_path / "pool",
        capacity=3,
        structure=structural_fingerprint(config),
    )
    for index in range(3):
        state = {name: value.clone() for name, value in policy.state_dict().items()}
        first = next(iter(state.values()))
        first.view(-1)[0] = index
        store.add(
            state,
            environment_steps=index,
            round_id=0,
            run_id=f"run-{index}",
        )

    left = allocate_round(
        config,
        100,
        3,
        SnapshotPool.from_store(store, seed=1),
    )
    right = allocate_round(
        config,
        100,
        3,
        SnapshotPool.from_store(store, seed=999),
    )

    assert [item.checkpoint_sha256 for item in left] == [
        item.checkpoint_sha256 for item in right
    ]


def test_failed_collection_does_not_commit_ids_and_retry_is_identical() -> None:
    """A worker failure cannot consume the global stream or logical round clock."""
    base = ToadConfig.control()
    config = base.model_copy(
        update={
            "population": base.population.model_copy(
                update={"environments_per_rank": 2}
            )
        }
    )
    seen: list[CollectionAssignment] = []
    failed = False

    def collect(assignment: CollectionAssignment) -> tuple[Trajectory]:
        nonlocal failed
        seen.append(assignment)
        if not failed and len(seen) == 2:
            failed = True
            raise RuntimeError("simulated worker failure")
        return (_trajectory(64),)

    source = ReferenceRoundSource(config, collect_assignment=collect)

    with pytest.raises(CollectionError, match="game_id"):
        list(source)
    first_attempt = tuple(seen)
    assert source.next_game_id == 0
    assert source._next_round_id == 0

    seen.clear()
    batches = list(source)

    assert tuple(seen) == first_attempt
    assert source.next_game_id == 2
    assert source._next_round_id == 1
    assert {batch.round_id for batch in batches} == {0}


def test_empty_pool_failure_does_not_consume_round_identity(tmp_path: Path) -> None:
    """An empty frozen pool leaves both collection clocks at their retry point."""
    config = ToadConfig.model_validate(
        {
            "population": {
                "selfplay": 0.0,
                "scripted": 0.0,
                "frozen_opponent": 1.0,
                "environments_per_rank": 1,
                "snapshot_at_start": True,
            }
        }
    )
    store = SnapshotStore(
        tmp_path / "pool",
        capacity=1,
        structure=structural_fingerprint(config),
    )
    manifest_path = tmp_path / "pool" / "manifest.json"
    assert not manifest_path.exists()
    source = ReferenceRoundSource(
        config,
        pool=SnapshotPool.from_store(store, seed=config.population.population_seed),
    )

    with pytest.raises(EmptySnapshotPoolError):
        allocate_round(
            config,
            source.next_game_id,
            source._next_round_id,
            source.pool,
        )

    assert source.next_game_id == 0
    assert source._next_round_id == 0
    assert not manifest_path.exists()


def test_teacher_checkpoint_identity_is_hashed_once_per_round_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multiple teacher slots share one precomputed byte identity."""
    checkpoint = tmp_path / "teacher.pt"
    torch.save({"weight": torch.ones(1)}, checkpoint)
    config = _allocation_config(
        selfplay=0.0,
        scripted=0.0,
        frozen=0.0,
        teacher=1.0,
        environments=4,
        teacher_checkpoint=checkpoint,
    )
    calls: list[Path] = []
    real_hash = sha256_file

    def counted_hash(path: Path) -> str:
        calls.append(path)
        return real_hash(path)

    monkeypatch.setattr("kaggriculture.learn.toad.data.sha256_file", counted_hash)

    assignments = allocate_round(config, 0, 0)

    assert len(assignments) == 4
    assert calls == [checkpoint]


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


def test_collection_assignment_accepts_raw_string_legacy_positional_order() -> None:
    """The one-release opponent-before-kind call accepts a serialized kind string."""
    assignment = CollectionAssignment(9, 10, "economic", "scripted")

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
                "snapshot_at_start": True,
            },
        }
    )
    opponent = toad.Policy(blocks=1, channels=4, value_bound=toad.VALUE_BOUND)
    store = SnapshotStore(
        tmp_path / "pool",
        capacity=1,
        structure=structural_fingerprint(config),
    )
    entry = store.add(
        opponent.state_dict(),
        environment_steps=9,
        round_id=1,
        run_id="frozen-run",
    )
    pool = SnapshotPool.from_store(store, seed=3)
    loads: list[tuple[str, ...]] = []
    real_load_many = SnapshotPool.load_many

    def counted_load_many(
        selected_pool: SnapshotPool, selected: Sequence[SnapshotEntry]
    ) -> dict[str, dict[str, torch.Tensor]]:
        loads.append(tuple(item.sha256 for item in selected))
        return real_load_many(selected_pool, selected)

    monkeypatch.setattr(SnapshotPool, "load_many", counted_load_many)
    seen: list[CollectionAssignment] = []

    def collect(assignment: CollectionAssignment) -> tuple[Trajectory]:
        seen.append(assignment)
        return (_trajectory(64),)

    source = ReferenceRoundSource(config, pool=pool, collect_assignment=collect)
    batches = list(source)

    assert loads == [(entry.sha256,)]
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
                "snapshot_at_start": True,
            }
        }
    )
    actor = toad.Policy(
        blocks=config.model.blocks,
        channels=config.model.channels,
        value_bound=toad.VALUE_BOUND,
    )
    store = SnapshotStore(
        tmp_path / "pool",
        capacity=1,
        structure=structural_fingerprint(config),
    )
    entry = store.add(actor.state_dict(), environment_steps=1, round_id=0, run_id="run")
    pool = SnapshotPool.from_store(store, seed=0)
    entry.path.write_bytes(b"corrupt")
    manifest_before = (tmp_path / "pool" / "manifest.json").read_bytes()
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
    assert source.next_game_id == 0
    assert source._next_round_id == 0
    assert (tmp_path / "pool" / "manifest.json").read_bytes() == manifest_before


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
    data_module = ToadDataModule(config, source)
    data_module.publish_actor(actor.state_dict(), version=5)
    _materialize_population_bootstrap(data_module, config, dict(actor.state_dict()))

    batches = list(source)

    assert source.pool is not None
    assert len(source.pool.manifest.entries) == 1
    assert seen[0].checkpoint_sha256 == source.pool.manifest.entries[0].sha256
    assert batches[0].actor_version == 5


@pytest.mark.parametrize("bootstrap", ["snapshot_at_start", "initial_snapshot"])
def test_snapshot_bootstrap_resume_reopens_exact_pool_without_republication(
    tmp_path: Path, bootstrap: str
) -> None:
    """Resume restores one bootstrapped manifest and the next frozen assignment."""
    actor = toad.Policy(blocks=1, channels=4, value_bound=toad.VALUE_BOUND)
    initial = tmp_path / "initial.pt"
    torch.save(actor.state_dict(), initial)
    bootstrap_config = (
        {"snapshot_at_start": True}
        if bootstrap == "snapshot_at_start"
        else {"initial_snapshots": [initial]}
    )
    config = ToadConfig.model_validate(
        {
            "model": {"blocks": 1, "channels": 4},
            "population": {
                "selfplay": 0.0,
                "scripted": 0.0,
                "frozen_opponent": 1.0,
                "environments_per_rank": 1,
                "pool_capacity": 3,
                "population_seed": 29,
                **bootstrap_config,
            },
            "runtime": {"output_dir": tmp_path / "run"},
        }
    )
    uninterrupted_seen: list[CollectionAssignment] = []
    source = ReferenceRoundSource(
        config,
        collect_assignment=lambda assignment: (
            uninterrupted_seen.append(assignment) or (_trajectory(64),)
        ),
    )
    data = ToadDataModule(config, source)
    data.publish_actor(actor.state_dict(), version=4)
    _materialize_population_bootstrap(data, config, dict(actor.state_dict()))
    list(source)
    state = data.state_dict()
    manifest_path = tmp_path / "run" / "population" / "manifest.json"
    manifest_before = manifest_path.read_bytes()
    list(source)
    expected = uninterrupted_seen[-1]

    resumed_seen: list[CollectionAssignment] = []
    resumed_source = ReferenceRoundSource(
        config,
        collect_assignment=lambda assignment: (
            resumed_seen.append(assignment) or (_trajectory(64),)
        ),
    )
    resumed = ToadDataModule(config, resumed_source)
    resumed.load_state_dict(state)

    list(resumed_source)

    assert resumed_seen == [expected]
    assert manifest_path.read_bytes() == manifest_before
    assert resumed_source.pool is not None
    assert len(resumed_source.pool.manifest.entries) == 1


@pytest.mark.parametrize("bootstrap", ["snapshot_at_start", "initial_snapshot"])
def test_legacy_frozen_resume_requires_population_identity_before_mutation(
    tmp_path: Path, bootstrap: str
) -> None:
    """A legacy payload cannot guess which durable bootstrap manifest it owned."""
    actor = toad.Policy(blocks=1, channels=4, value_bound=toad.VALUE_BOUND)
    initial = tmp_path / "initial.pt"
    torch.save(actor.state_dict(), initial)
    bootstrap_config = (
        {"snapshot_at_start": True}
        if bootstrap == "snapshot_at_start"
        else {"initial_snapshots": [initial]}
    )
    config = ToadConfig.model_validate(
        {
            "model": {"blocks": 1, "channels": 4},
            "population": {
                "selfplay": 0.0,
                "scripted": 0.0,
                "frozen_opponent": 1.0,
                "environments_per_rank": 1,
                "pool_capacity": 2,
                **bootstrap_config,
            },
            "runtime": {"output_dir": tmp_path / "run"},
        }
    )
    original_source = ReferenceRoundSource(config, assignments=())
    original = ToadDataModule(config, original_source)
    original.publish_actor(actor.state_dict(), version=4)
    _materialize_population_bootstrap(original, config, dict(actor.state_dict()))
    legacy_state = original.state_dict()
    legacy_state.pop("population_pool")
    manifest_path = tmp_path / "run" / "population" / "manifest.json"
    manifest_before = manifest_path.read_bytes()
    directory_before = {
        path: (path.stat().st_mtime_ns, path.read_bytes())
        for path in (tmp_path / "run" / "population").iterdir()
    }

    resumed_source = ReferenceRoundSource(config, assignments=())
    resumed_source.next_game_id = 73
    resumed_source._next_round_id = 9
    resumed_source.actor_version = 11
    resumed = ToadDataModule(config, resumed_source)

    with pytest.raises(RuntimeError, match="population_pool.*migration.*required"):
        resumed.load_state_dict(legacy_state)

    assert resumed_source.next_game_id == 73
    assert resumed_source._next_round_id == 9
    assert resumed_source.actor_version == 11
    assert resumed_source.pool is None
    assert manifest_path.read_bytes() == manifest_before
    assert {
        path: (path.stat().st_mtime_ns, path.read_bytes())
        for path in (tmp_path / "run" / "population").iterdir()
    } == directory_before


def test_legacy_control_resume_without_population_identity_remains_compatible() -> None:
    """Pool-free control checkpoints retain their historical state seam."""
    config = ToadConfig.control()
    source = ReferenceRoundSource(config, assignments=())
    data = ToadDataModule(config, source)
    source.next_game_id = 41
    source._next_round_id = 3
    source.actor_version = 6
    state = data.state_dict()
    state.pop("population_pool")

    resumed_source = ReferenceRoundSource(config, assignments=())
    ToadDataModule(config, resumed_source).load_state_dict(state)

    assert resumed_source.next_game_id == 41
    assert resumed_source._next_round_id == 3
    assert resumed_source.actor_version == 6
    assert resumed_source.pool is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("next_game_id", True),
        ("next_game_id", 1.5),
        ("next_game_id", -1),
        ("next_round_id", False),
        ("next_round_id", 0.5),
        ("next_round_id", -1),
        ("published_actor_version", True),
        ("published_actor_version", 1.5),
        ("published_actor_version", -1),
    ],
)
def test_collector_restore_rejects_malformed_counters_before_mutation(
    field: str,
    value: object,
) -> None:
    """Collector clocks never coerce booleans, fractions, or negative values."""
    config = ToadConfig.control()
    stored_source = ReferenceRoundSource(config)
    state = ToadDataModule(config, stored_source).state_dict()
    state[field] = value
    resumed_source = ReferenceRoundSource(config)
    resumed_source.next_game_id = 24
    resumed_source._next_round_id = 1
    resumed_source.actor_version = 3
    before = (
        resumed_source.next_game_id,
        resumed_source._next_round_id,
        resumed_source.actor_version,
        resumed_source.rng.getstate(),
    )

    with pytest.raises(ValueError, match="invalid collector checkpoint state"):
        ToadDataModule(config, resumed_source).load_state_dict(state)

    assert (
        resumed_source.next_game_id,
        resumed_source._next_round_id,
        resumed_source.actor_version,
        resumed_source.rng.getstate(),
    ) == before


def test_active_collector_restore_requires_round_and_game_stream_alignment() -> None:
    """A generated stream cannot resume from a skipped or repeated game range."""
    base = ToadConfig.control()
    config = base.model_copy(
        update={
            "population": base.population.model_copy(
                update={"environments_per_rank": 2}
            )
        }
    )
    state = ToadDataModule(config, ReferenceRoundSource(config)).state_dict()
    state["next_game_id"] = 3
    state["next_round_id"] = 1
    resumed_source = ReferenceRoundSource(config)

    with pytest.raises(ValueError, match="invalid collector checkpoint state"):
        ToadDataModule(config, resumed_source).load_state_dict(state)

    assert resumed_source.next_game_id == 0
    assert resumed_source._next_round_id == 0


def test_external_pool_seed_is_normalized_before_precollection_checkpoint(
    tmp_path: Path,
) -> None:
    """Checkpoint identity never persists a supplied pool's construction seed."""
    config = ToadConfig.model_validate(
        {
            "model": {"blocks": 1, "channels": 4},
            "population": {
                "selfplay": 0.0,
                "scripted": 0.0,
                "frozen_opponent": 1.0,
                "environments_per_rank": 4,
                "population_seed": 29,
                "pool_capacity": 2,
                "snapshot_at_start": True,
            },
        }
    )
    actor = toad.Policy(blocks=1, channels=4, value_bound=toad.VALUE_BOUND)
    store = SnapshotStore(
        tmp_path / "pool",
        capacity=2,
        structure=structural_fingerprint(config),
    )
    for index in range(2):
        state = {name: value.clone() for name, value in actor.state_dict().items()}
        next(iter(state.values())).view(-1)[0] = index
        store.add(
            state,
            environment_steps=index,
            round_id=0,
            run_id=f"run-{index}",
        )
    external = SnapshotPool.from_store(store, seed=999)
    source = ReferenceRoundSource(config, assignments=(), pool=external)
    data = ToadDataModule(config, source)

    checkpoint = data.state_dict()
    identity = cast(dict[str, object], checkpoint["population_pool"])
    assert identity["seed"] == config.population.population_seed

    resumed_source = ReferenceRoundSource(config, assignments=())
    ToadDataModule(config, resumed_source).load_state_dict(checkpoint)
    assert source.pool is not None
    assert resumed_source.pool is not None
    assert source.pool.identity() == resumed_source.pool.identity()
    assert allocate_round(config, 80, 7, source.pool) == allocate_round(
        config, 80, 7, resumed_source.pool
    )


def test_same_digest_frozen_and_teacher_bindings_keep_independent_semantics(
    tmp_path: Path,
) -> None:
    """Digest equality may share tensors but never merges binding provenance."""
    preliminary = ToadConfig.model_validate({"model": {"blocks": 1, "channels": 4}})
    policy = toad.Policy(blocks=1, channels=4, value_bound=toad.VALUE_BOUND)
    store = SnapshotStore(
        tmp_path / "pool",
        capacity=1,
        structure=structural_fingerprint(preliminary),
    )
    entry = store.add(
        policy.state_dict(), environment_steps=1, round_id=0, run_id="run"
    )
    teacher_path = tmp_path / "teacher.pt"
    teacher_path.write_bytes(entry.path.read_bytes())
    config = ToadConfig.model_validate(
        {
            "model": {"blocks": 1, "channels": 4},
            "population": {
                "teacher": {
                    "checkpoint": teacher_path,
                    "sha256": entry.sha256,
                    "blocks": 1,
                    "operation": True,
                    "quantity": True,
                    "market": True,
                }
            },
        }
    )
    pool = SnapshotPool.from_store(store, seed=config.population.population_seed)
    assignments = (
        OpponentAssignment(
            0,
            0,
            BatchKind.FROZEN_OPPONENT,
            "snapshot",
            entry.path,
            entry.sha256,
        ),
        OpponentAssignment(
            1,
            1,
            BatchKind.TEACHER_DISTILL,
            "teacher",
            teacher_path,
            entry.sha256,
        ),
    )
    collected: list[int] = []
    source = ReferenceRoundSource(
        config,
        assignments=assignments,
        pool=pool,
        collect_assignment=lambda assignment: (
            collected.append(assignment.game_id) or (_trajectory(64),)
        ),
    )

    batches = list(source)

    assert collected == [0, 1]
    assert {
        (kind, opponent_id, digest)
        for batch in batches
        for kind, opponent_id, digest in zip(
            batch.segment_kinds,
            batch.segment_opponent_ids,
            batch.segment_opponent_digests,
            strict=True,
        )
    } == {
        (BatchKind.FROZEN_OPPONENT, "snapshot", entry.sha256),
        (BatchKind.TEACHER_DISTILL, "teacher", entry.sha256),
    }


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
    data_module = ToadDataModule(config, source)
    actor = toad.Policy(blocks=1, channels=4, value_bound=toad.VALUE_BOUND)

    with pytest.raises(RuntimeError, match="Missing key|Unexpected key"):
        _materialize_population_bootstrap(data_module, config, dict(actor.state_dict()))

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
                "teacher": {
                    "checkpoint": checkpoint,
                    "blocks": 1,
                    "operation": True,
                    "quantity": True,
                    "market": True,
                },
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


def test_teacher_distill_rejects_a_missing_declared_head_before_collection(
    tmp_path: Path,
) -> None:
    """A rollout-incomplete teacher cannot reach the external collector seam."""
    checkpoint = tmp_path / "teacher.pt"
    teacher = toad.Policy(blocks=1, channels=4, value_bound=toad.VALUE_BOUND)
    torch.save(
        {
            name: value
            for name, value in teacher.state_dict().items()
            if not name.startswith("quantity_head.")
        },
        checkpoint,
    )
    config = ToadConfig.model_validate(
        {
            "model": {"blocks": 1, "channels": 4},
            "population": {
                "selfplay": 0.0,
                "scripted": 0.0,
                "teacher_distill": 1.0,
                "teacher": {
                    "checkpoint": checkpoint,
                    "blocks": 1,
                    "operation": True,
                    "quantity": True,
                    "market": True,
                },
                "environments_per_rank": 1,
            },
        }
    )
    collected: list[int] = []
    source = ReferenceRoundSource(
        config,
        collect_assignment=lambda assignment: (
            collected.append(assignment.game_id) or (_trajectory(64),)
        ),
    )

    with pytest.raises(
        SnapshotIntegrityError, match="teacher checkpoint is incompatible"
    ):
        list(source)

    assert collected == []


def test_stateful_collection_builds_a_strict_control_teacher_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Teacher rollout uses a bare control opponent beside a stateful learner."""
    checkpoint = tmp_path / "teacher.pt"
    teacher = toad.Policy(blocks=1, channels=4, value_bound=toad.VALUE_BOUND)
    torch.save(teacher.state_dict(), checkpoint)
    config = ToadConfig.model_validate(
        {
            "model": {
                "blocks": 1,
                "channels": 4,
                "recurrent": True,
                "recurrent_channels": 3,
            },
            "population": {
                "selfplay": 0.0,
                "scripted": 0.0,
                "teacher_distill": 1.0,
                "teacher": {
                    "checkpoint": checkpoint,
                    "blocks": 1,
                    "operation": True,
                    "quantity": True,
                    "market": True,
                },
                "environments_per_rank": 1,
            },
        }
    )
    module = ToadLightningModule(config)
    source = ReferenceRoundSource(module.config)
    source.publish_actor(module.policy.state_dict(), version=1)
    assignment = allocate_round(module.config, 0, 0)[0]
    seen: list[tuple[object, object]] = []

    def fake_rollout(
        learner: object,
        opponent: object,
        seeds: Sequence[int],
        *,
        state_unroll_length: int | None = None,
    ) -> list[Trajectory]:
        seen.append((learner, opponent))
        return []

    monkeypatch.setattr(toad, "rollout_many", fake_rollout)

    source.collect_assignment(assignment)

    assert len(seen) == 1
    assert isinstance(seen[0][0], StatefulPolicy)
    assert isinstance(seen[0][1], toad.Policy)
    assert not isinstance(seen[0][1], StatefulPolicy)
    assert seen[0][1].stem.kernel_size == (3, 3)


def test_confirmed_teacher_digest_fails_closed_after_path_mutation(
    tmp_path: Path,
) -> None:
    """Collection must retain module-confirmed bytes across the handoff."""
    checkpoint = tmp_path / "teacher.pt"
    first = toad.Policy(blocks=1, channels=4, value_bound=toad.VALUE_BOUND)
    torch.save(first.state_dict(), checkpoint)
    config = ToadConfig.model_validate(
        {
            "model": {"blocks": 1, "channels": 4},
            "population": {
                "selfplay": 0.0,
                "scripted": 0.0,
                "teacher_distill": 1.0,
                "teacher": {
                    "checkpoint": checkpoint,
                    "blocks": 1,
                    "operation": True,
                    "quantity": True,
                    "market": True,
                },
                "environments_per_rank": 1,
            },
        }
    )
    module = ToadLightningModule(config)
    assert module.config.population.teacher is not None
    confirmed = module.config.population.teacher.sha256
    assert confirmed is not None
    torch.save(
        toad.Policy(blocks=1, channels=4, value_bound=toad.VALUE_BOUND).state_dict(),
        checkpoint,
    )
    collected: list[int] = []
    source = ReferenceRoundSource(
        module.config,
        collect_assignment=lambda assignment: (
            collected.append(assignment.game_id) or (_trajectory(64),)
        ),
    )

    with pytest.raises(SnapshotIntegrityError, match=confirmed):
        list(source)

    assert collected == []


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


def test_legacy_vs_econ_metrics_exclude_neural_opponents() -> None:
    """Frozen and teacher outcomes cannot contaminate the scripted control curve."""
    scripted = replace(_trajectory(32), final_margin=-1.0)
    frozen = replace(_trajectory(32), final_margin=1.0)
    teacher = replace(_trajectory(32), final_margin=1.0)
    assignments = (
        CollectionAssignment(0, 0, "economic", BatchKind.SCRIPTED),
        CollectionAssignment(1, 1, "frozen", BatchKind.FROZEN_OPPONENT),
        CollectionAssignment(2, 2, "teacher", BatchKind.TEACHER_DISTILL),
    )
    by_kind = {
        BatchKind.SCRIPTED: scripted,
        BatchKind.FROZEN_OPPONENT: frozen,
        BatchKind.TEACHER_DISTILL: teacher,
    }
    source = ReferenceRoundSource(
        ToadConfig.control(),
        assignments=assignments,
        collect_assignment=lambda assignment: (by_kind[assignment.kind],),
    )

    first = next(iter(source))

    assert first.round_metrics["diag/n_econ_envs"] == 1
    assert first.round_metrics["objective/win_rate_vs_econ"] == 0.0


def test_collection_game_metrics_count_ids_not_selfplay_seats() -> None:
    """A mirrored game produces two trajectories but only one selected game ID."""
    assignment = CollectionAssignment(7, 7, "self", BatchKind.SELFPLAY)
    source = ReferenceRoundSource(
        ToadConfig.control(),
        assignments=(assignment,),
        collect_assignment=lambda _assignment: (_trajectory(32), _trajectory(32)),
    )

    first = next(iter(source))

    assert first.round_metrics["collection/games/selfplay"] == 1


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


def test_production_neural_chunk_constructs_and_loads_once_for_multiple_games(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One immutable frozen binding runs every assigned seed in one worker chunk."""
    config = ToadConfig.model_validate(
        {
            "model": {"blocks": 1, "channels": 4},
            "population": {
                "selfplay": 0.0,
                "scripted": 0.0,
                "frozen_opponent": 1.0,
                "environments_per_rank": 2,
                "collection_processes": 2,
                "snapshot_at_start": True,
            },
        }
    )
    actor = toad.Policy(blocks=1, channels=4, value_bound=toad.VALUE_BOUND)
    store = SnapshotStore(
        tmp_path / "pool",
        capacity=1,
        structure=structural_fingerprint(config),
    )
    entry = store.add(
        actor.state_dict(), environment_steps=0, round_id=0, run_id="frozen"
    )
    pool = SnapshotPool.from_store(store, seed=config.population.population_seed)
    works: list[ReferenceWorkerInput] = []

    class FakePool:
        def __init__(self, max_workers: int) -> None:
            assert max_workers == 2

        def __enter__(self) -> "FakePool":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def map(
            self,
            function: Callable[[ReferenceWorkerInput], object],
            inputs: Iterable[ReferenceWorkerInput],
        ) -> Iterable[object]:
            chunks = list(inputs)
            works.extend(chunks)
            return map(function, chunks)

    constructions: list[bool] = []
    real_reference_policy = toad._reference_policy

    def counted_reference_policy(
        model: ModelConfig,
        state: dict[str, torch.Tensor],
        *,
        frozen: bool,
    ) -> object:
        constructions.append(frozen)
        return real_reference_policy(model, state, frozen=frozen)

    def fake_rollout(
        learner: object,
        opponent: object,
        seeds: Sequence[int],
        *,
        state_unroll_length: int | None = None,
    ) -> list[Trajectory]:
        assert learner is not opponent
        assert state_unroll_length == config.optimizer.unroll_length
        return [_trajectory(64) for _seed in seeds]

    loads: list[tuple[Path, ...]] = []
    real_load_many = SnapshotPool.load_many

    def counted_load_many(
        selected_pool: SnapshotPool, selected: Sequence[SnapshotEntry]
    ) -> dict[str, dict[str, torch.Tensor]]:
        loads.append(tuple(item.path for item in selected))
        return real_load_many(selected_pool, selected)

    monkeypatch.setattr("kaggriculture.learn.toad.data.ProcessPoolExecutor", FakePool)
    monkeypatch.setattr(toad, "_reference_policy", counted_reference_policy)
    monkeypatch.setattr(toad, "rollout_many", fake_rollout)
    monkeypatch.setattr(SnapshotPool, "load_many", counted_load_many)
    source = ReferenceRoundSource(config, pool=pool)
    source.publish_actor(actor.state_dict(), version=7)

    batches = list(source)

    assert len(works) == 1
    assert works[0].seeds == [0, 1]
    assert constructions == [False, True]
    assert loads == [(entry.path,)]
    assert {game_id for batch in batches for game_id in batch.segment_game_ids} == {
        0,
        1,
    }


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
