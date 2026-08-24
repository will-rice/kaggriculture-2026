"""Distributed ownership contracts for synchronous Toad collection."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import lightning
import pytest
import torch
from pydantic import ValidationError

import kaggriculture.learn.toad.callbacks as toad_callbacks
import kaggriculture.learn.toad.lightning as toad_lightning
import kaggriculture.sim.day as sim_day
from kaggriculture.learn.encoding import (
    MARKET_SLOTS,
    MAX_UNITS,
    QUANTITIES,
    SCALARS,
    TILE_PLANES,
    UNIT_OPS,
)
from kaggriculture.learn.rollout import Trajectory
from kaggriculture.learn.scripts.toad import RuntimePreflightError, runtime_preflight
from kaggriculture.learn.toad import data
from kaggriculture.learn.toad.callbacks import (
    ActorSyncCallback,
    BoundaryCheckpoint,
    DistributedBoundaryError,
    EnvironmentStepStop,
    PopulationSnapshotCallback,
)
from kaggriculture.learn.toad.config import RuntimeConfig, ToadConfig
from kaggriculture.learn.toad.lightning import (
    NonFiniteProvenance,
    NonFiniteTrainingError,
    ReducibleMetric,
    ToadLightningModule,
)

# Characterized control is 0 wins in 2 games; this is its 95% Wilson interval,
# rounded outward so the acceptance boundary is a stable predeclared literal.
_ECONOMIC_CONTROL_WIN_RATE_INTERVAL = (0.0, 0.66)


def test_rank_game_ids_are_disjoint_and_contiguous() -> None:
    """Changing a rank offset must never overlap another rank's game IDs."""
    ids = [
        data.rank_game_ids(start=100, per_rank=3, rank=rank, world_size=2)
        for rank in range(2)
    ]

    assert ids == [(100, 101, 102), (103, 104, 105)]
    assert set(ids[0]).isdisjoint(ids[1])


def test_changed_world_size_continues_after_checkpoint_boundary() -> None:
    """A resumed boundary is absolute and independent of its old world size."""
    resumed = [
        data.rank_game_ids(start=106, per_rank=2, rank=rank, world_size=3)
        for rank in range(3)
    ]

    assert resumed == [(106, 107), (108, 109), (110, 111)]
    assert sorted(game_id for rank_ids in resumed for game_id in rank_ids) == list(
        range(106, 112)
    )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"start": -1, "per_rank": 1, "rank": 0, "world_size": 1}, "start"),
        ({"start": 0, "per_rank": 0, "rank": 0, "world_size": 1}, "per_rank"),
        ({"start": 0, "per_rank": 1, "rank": 1, "world_size": 1}, "rank"),
        ({"start": 0, "per_rank": 1, "rank": 0, "world_size": 0}, "world_size"),
    ],
)
def test_rank_game_ids_reject_invalid_topology(
    arguments: dict[str, int], message: str
) -> None:
    """Malformed rank metadata must fail before it can perturb the stream."""
    with pytest.raises(ValueError, match=message):
        data.rank_game_ids(**arguments)


def test_runtime_config_types_ddp_topology() -> None:
    """Node count and DDP selection are validated serializable run identity."""
    runtime = RuntimeConfig(num_nodes=2, strategy="ddp", devices=3)

    assert runtime.num_nodes == 2
    assert runtime.strategy == "ddp"
    assert runtime.devices == 3
    with pytest.raises(ValidationError):
        RuntimeConfig(num_nodes=0)
    with pytest.raises(ValidationError):
        RuntimeConfig(strategy="not-ddp")  # ty: ignore[invalid-argument-type]


def test_preflight_requires_ddp_for_resolved_world_size_above_one() -> None:
    """Lightning must not silently choose a multi-process strategy."""
    config = ToadConfig.model_validate(
        {"runtime": {"accelerator": "cpu", "devices": 2, "strategy": "auto"}}
    )

    with pytest.raises(RuntimePreflightError, match="strategy='ddp'"):
        runtime_preflight(config)


def test_preflight_accepts_explicit_cpu_ddp() -> None:
    """A fully explicit two-process CPU request reaches Trainer unchanged."""
    config = ToadConfig.model_validate(
        {"runtime": {"accelerator": "cpu", "devices": 2, "strategy": "ddp"}}
    )

    runtime_preflight(config)


def _trajectory(turns: int = 4) -> Trajectory:
    """Return one cheap valid scripted trajectory for stream-ownership tests."""
    market_slots = len(MARKET_SLOTS) + 2
    dones = torch.zeros(turns, dtype=torch.bool)
    dones[-1] = True
    rewards = torch.ones(turns)
    return Trajectory(
        board=torch.zeros(turns, TILE_PLANES, 10, 10),
        scalars=torch.zeros(turns, SCALARS),
        positions=torch.zeros(turns, MAX_UNITS, dtype=torch.int64),
        unit_actions=torch.zeros(turns, MAX_UNITS, dtype=torch.int64),
        unit_quantities=torch.zeros(turns, MAX_UNITS, dtype=torch.int64),
        market_actions=torch.zeros(turns, market_slots, dtype=torch.int64),
        unit_masks=torch.ones(turns, MAX_UNITS, len(UNIT_OPS), dtype=torch.bool),
        unit_quantity_masks=torch.ones(
            turns, MAX_UNITS, len(QUANTITIES), dtype=torch.bool
        ),
        market_masks=torch.ones(turns, market_slots, len(QUANTITIES), dtype=torch.bool),
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


def _stream_config() -> ToadConfig:
    """Return a two-game scripted round with one optimizer batch per game."""
    return ToadConfig.model_validate(
        {
            "population": {
                "selfplay": 0.0,
                "scripted": 1.0,
                "environments_per_rank": 2,
                "collection_processes": 1,
            },
            "optimizer": {
                "unroll_length": 4,
                "batch_segments": 1,
                "value_passes": 0,
            },
        }
    )


def test_each_rank_owns_unique_ids_but_the_same_global_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rank-local sources advance one shared global stream, not local streams."""
    monkeypatch.setattr(
        data,
        "_all_gather_objects",
        lambda local, world_size, **_kwargs: (local,) * world_size,
    )
    config = _stream_config()
    seen: list[list[int]] = []
    next_ids: list[int] = []
    for rank in range(2):
        rank_seen: list[int] = []

        def collector(
            target: list[int],
        ) -> Callable[[data.CollectionAssignment], tuple[Trajectory]]:
            def collect(assignment: data.CollectionAssignment) -> tuple[Trajectory]:
                target.append(assignment.game_id)
                return (_trajectory(),)

            return collect

        source = data.ReferenceRoundSource(
            config, collect_assignment=collector(rank_seen)
        )
        source.configure_distributed(rank=rank, world_size=2)
        batches = list(source)
        assert sum(batch.first_of_round for batch in batches) == 1
        assert sum(batch.end_of_round for batch in batches) == 1
        seen.append(rank_seen)
        next_ids.append(source.next_game_id)

    assert seen == [[0, 1], [2, 3]]
    assert next_ids == [4, 4]


def test_changed_world_size_resume_advances_from_the_absolute_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restoring with more ranks neither recomputes nor gaps the next game ID."""
    monkeypatch.setattr(
        data,
        "_all_gather_objects",
        lambda local, world_size, **_kwargs: (local,) * world_size,
    )
    config = _stream_config()
    original = data.ReferenceRoundSource(
        config,
        collect_assignment=lambda _assignment: (_trajectory(),),
    )
    original.configure_distributed(rank=0, world_size=2)
    list(original)
    state = data.ToadDataModule(config, original).state_dict()

    seen: list[int] = []

    def collect(assignment: data.CollectionAssignment) -> tuple[Trajectory]:
        seen.append(assignment.game_id)
        return (_trajectory(),)

    resumed = data.ReferenceRoundSource(config, collect_assignment=collect)
    resumed.configure_distributed(rank=0, world_size=3)
    data.ToadDataModule(config, resumed).load_state_dict(state)
    list(resumed)

    assert seen == [4, 5]
    assert resumed.next_game_id == 10
    next_ids = [
        data.rank_game_ids(start=4, per_rank=2, rank=rank, world_size=3)
        for rank in range(3)
    ]
    assert sorted(game_id for ids in next_ids for game_id in ids) == list(range(4, 10))


def test_expected_round_counts_are_kind_aware_and_equal_per_rank() -> None:
    """Mirror games contribute two seats while scripted games contribute one."""
    config = ToadConfig.model_validate(
        {
            "population": {
                "selfplay": 0.5,
                "scripted": 0.5,
                "environments_per_rank": 2,
            },
            "optimizer": {
                "unroll_length": 719,
                "batch_segments": 1,
                "value_passes": 2,
            },
        }
    )

    counts = data.expected_round_batch_counts(config)

    assert counts == data.RoundBatchCounts(policy=3, value=6)


def test_distributed_round_signature_rejects_one_short_rank() -> None:
    """A short rank is diagnosed collectively before the first backward."""
    expected = data.RoundBatchCounts(policy=2, value=2)
    signatures = (
        data.RoundBatchSignature(
            rank=0,
            round_id=7,
            expected=expected,
            policy=2,
            value=2,
            first_markers=1,
            end_markers=1,
        ),
        data.RoundBatchSignature(
            rank=1,
            round_id=7,
            expected=expected,
            policy=1,
            value=2,
            first_markers=1,
            end_markers=0,
        ),
    )

    with pytest.raises(data.DistributedRoundMismatch, match="rank=1.*end_markers=0"):
        data.validate_distributed_round_signatures(signatures)


def test_data_module_preflights_equal_quotas_before_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config drift is rejected in setup, before a rank starts its collector."""
    config = _stream_config()
    source = data.ReferenceRoundSource(
        config,
        collect_assignment=lambda _assignment: pytest.fail("collection started"),
    )
    module = data.ToadDataModule(config, source)
    module.trainer = cast(
        lightning.Trainer,
        SimpleNamespace(global_rank=0, world_size=2),
    )
    local = data.expected_round_batch_counts(config)
    remote = data.RoundBatchCounts(policy=local.policy + 1, value=local.value)
    monkeypatch.setattr(
        data, "_all_gather_objects", lambda _local, _world: (local, remote)
    )

    with pytest.raises(data.DistributedRoundMismatch, match="expected round quotas"):
        module.setup("fit")

    assert source.global_rank == 0
    assert source.world_size == 2


def test_empty_local_round_reaches_failure_gather_before_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero-batch rank must notify peers before any signature collective."""
    config = _stream_config().model_copy(
        update={
            "optimizer": _stream_config().optimizer.model_copy(
                update={"batch_segments": 100}
            )
        }
    )
    source = data.ReferenceRoundSource(
        config,
        collect_assignment=lambda _assignment: (_trajectory(),),
    )
    source.configure_distributed(rank=0, world_size=2)
    monkeypatch.setattr(
        data,
        "_all_gather_objects",
        lambda local, world_size, **_kwargs: (local,) * world_size,
    )

    def reject_empty(_batches: Sequence[data.LearnerBatch], _round_id: int) -> None:
        raise AssertionError("batch signature ran after failed collection")

    source.set_round_validator(reject_empty)

    with pytest.raises(
        data.DistributedCollectionError, match="must be exactly divisible"
    ):
        list(source)


def test_round_steps_use_the_all_rank_sum_not_the_local_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sample budget advances by actual experience from every rank."""
    from tests.learn.test_toad_control_fixture import (
        control_fixture_batch,
        control_fixture_config,
        load_control_fixture,
    )

    module = ToadLightningModule(control_fixture_config())
    batch = replace(
        control_fixture_batch(load_control_fixture()),
        collected_steps=32,
        first_of_round=True,
        end_of_round=True,
    )
    monkeypatch.setattr(
        module,
        "_reduce_tensor",
        lambda value, reduce_op="sum": value * 2 if reduce_op == "sum" else value,
        raising=False,
    )
    monkeypatch.setattr(module, "log_dict", lambda *_args, **_kwargs: None)

    module.training_step(batch, 0)

    assert module.environment_steps == 64
    assert module.round_local_steps == 0
    assert module.collection_round == 1


def test_nonfinite_forward_reaches_the_synchronization_seam_before_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One bad rank cannot raise locally while peers enter DDP backward."""
    from tests.learn.test_toad_control_fixture import (
        control_fixture_batch,
        control_fixture_config,
        load_control_fixture,
    )

    module = ToadLightningModule(control_fixture_config())
    batch = control_fixture_batch(load_control_fixture())
    segments = list(batch.segments)
    segments[0] = dict(segments[0])
    segments[0]["board"] = segments[0]["board"].clone()
    segments[0]["board"][0, 0, 0, 0] = torch.nan
    bad = replace(batch, segments=tuple(segments))
    synchronized: list[tuple[str, ...]] = []

    def synchronize(names: list[str], provenance: object) -> None:
        del provenance
        synchronized.append(tuple(names))
        raise NonFiniteTrainingError.from_batch(
            names,
            bad,
            None,
            module.config.runtime.precision,
        )

    monkeypatch.setattr(module, "_synchronize_nonfinite", synchronize, raising=False)

    with pytest.raises(NonFiniteTrainingError):
        module.training_step(bad, 0)

    assert synchronized
    assert "input/board" in synchronized[0]


def test_gradient_finite_flag_is_synchronized_even_when_this_rank_is_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clean rank still learns that a peer produced non-finite gradients."""
    from tests.learn.test_toad_control_fixture import (
        control_fixture_batch,
        control_fixture_config,
        load_control_fixture,
    )

    module = ToadLightningModule(control_fixture_config())
    report = module.compute_report(control_fixture_batch(load_control_fixture()))
    module._last_finite_provenance = report.provenance
    synchronized: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        module,
        "_synchronize_nonfinite",
        lambda names, _provenance: synchronized.append(tuple(names)),
        raising=False,
    )

    module.on_after_backward()

    assert synchronized == [()]


def test_one_bad_rank_uses_canonical_bad_provenance_on_every_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clean rank must raise the same structured failure as the bad rank."""
    from tests.learn.test_toad_control_fixture import (
        control_fixture_batch,
        control_fixture_config,
        load_control_fixture,
    )

    module = ToadLightningModule(control_fixture_config())
    local = module.compute_report(
        control_fixture_batch(load_control_fixture())
    ).provenance
    bad = NonFiniteProvenance.from_dict(
        local.as_dict() | {"game_ids": (99,), "actor_version": 7}
    )
    module.trainer = cast(
        lightning.Trainer,
        SimpleNamespace(world_size=2, global_rank=0),
    )
    monkeypatch.setattr(
        module, "_reduce_tensor", lambda *_args, **_kwargs: torch.tensor(1)
    )
    details = (
        None,
        {
            "rank": 1,
            "tensor_names": ("input/board",),
            "provenance": bad.as_dict(),
        },
    )
    monkeypatch.setattr(
        toad_lightning, "_all_gather_objects", lambda *_args, **_kwargs: details
    )

    with pytest.raises(NonFiniteTrainingError) as caught:
        module._synchronize_nonfinite([], local)

    assert caught.value.game_ids == (99,)
    assert caught.value.actor_version == 7
    assert caught.value.distributed_details == details


def test_all_bad_ranks_choose_rank_zero_provenance_canonically() -> None:
    """Every caller renders an identical error when every rank is bad."""
    first = NonFiniteProvenance(
        batch_kind="scripted",
        game_ids=(10,),
        opponent_digests=(),
        actor_version=2,
        precision="32-true",
        state_norms={},
    )
    second = NonFiniteProvenance.from_dict(
        first.as_dict() | {"game_ids": (11,), "actor_version": 3}
    )
    details = (
        {"rank": 0, "tensor_names": ("loss/total",), "provenance": first.as_dict()},
        {"rank": 1, "tensor_names": ("grad/stem",), "provenance": second.as_dict()},
    )

    left = NonFiniteTrainingError.render_distributed(details, second)
    right = NonFiniteTrainingError.render_distributed(details, first)

    assert str(left) == str(right)
    assert left.game_ids == right.game_ids == (10,)
    assert left.tensor_names == right.tensor_names == ("loss/total",)


def test_rank_local_collection_failure_is_gathered_before_batch_collectives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed collector must make successful peers leave with the same error."""
    config = _stream_config()
    source = data.ReferenceRoundSource(
        config,
        collect_assignment=lambda assignment: (_trajectory(),),
    )
    source.configure_distributed(rank=0, world_size=2)
    source.publish_actor({}, 0)
    peer = {
        "ok": False,
        "rank": 1,
        "type": "CollectionError",
        "message": "collection failed for game_id=2 seed=2 opponent=economic",
        "traceback": "peer traceback sentinel",
    }

    def gather(local: object, _world: int, **_kwargs: object) -> tuple[object, object]:
        return local, peer

    monkeypatch.setattr(data, "_all_gather_objects", gather)

    with pytest.raises(data.DistributedCollectionError) as caught:
        list(source)

    assert "game_id=2 seed=2" in str(caught.value)
    assert "peer traceback sentinel" in str(caught.value)


def test_round_metrics_reduce_numerators_before_means_and_keep_rank_timings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unequal rank counts must produce a weighted mean, not mean-of-means."""
    config = _stream_config()
    module = ToadLightningModule(config)
    module.trainer = cast(
        lightning.Trainer,
        SimpleNamespace(world_size=2, global_rank=0),
    )
    module.round_metrics.reset({"throughput/collection_seconds": 1.5})
    module.round_metrics.learner_seconds = 0.5
    monkeypatch.setattr(
        module.round_metrics,
        "reducible_values",
        lambda _device: {
            "loss/example": ReducibleMetric(
                total=torch.tensor(2.0), count=torch.tensor(1)
            )
        },
    )
    reductions = 0

    def reduce(value: torch.Tensor, reduce_op: str = "sum") -> torch.Tensor:
        nonlocal reductions
        del reduce_op
        reductions += 1
        if reductions == 1:
            return value.new_tensor(8.0)
        if reductions == 2:
            return value.new_tensor(3)
        return value

    def gather(local: object, _world_size: int) -> tuple[object, object]:
        if isinstance(local, dict) and "loss/example" in local:
            return local, local
        if isinstance(local, dict) and "rank" in local:
            return local, {
                "rank": 1,
                "collection_seconds": 2.5,
                "learner_seconds": 0.75,
            }
        return local, local

    monkeypatch.setattr(module, "_reduce_tensor", reduce)
    monkeypatch.setattr(toad_lightning, "_all_gather_objects", gather)

    module._reduce_round_statistics()

    torch.testing.assert_close(
        module._round_reduced_metrics["loss/example"].value,
        torch.tensor(8.0 / 3.0),
    )
    assert module._round_rank_timings == {
        "rank/0/throughput/collection_seconds": 1.5,
        "rank/0/throughput/learner_seconds": 0.5,
        "rank/1/throughput/collection_seconds": 2.5,
        "rank/1/throughput/learner_seconds": 0.75,
    }


def test_opponent_return_carries_true_sum_and_count_for_unequal_ranks() -> None:
    """Per-opponent global means must weight ranks by their trajectory counts."""
    name = "economic"
    first = toad_lightning.RoundMetricAccumulator()
    first.reset(
        {
            f"collection/return_by_opponent/{name}": 10.0,
            f"collection/return_sum_by_opponent/{name}": 10.0,
            f"collection/return_count_by_opponent/{name}": 1,
        }
    )
    second = toad_lightning.RoundMetricAccumulator()
    second.reset(
        {
            f"collection/return_by_opponent/{name}": 20.0,
            f"collection/return_sum_by_opponent/{name}": 60.0,
            f"collection/return_count_by_opponent/{name}": 3,
        }
    )

    left = first.reducible_values(torch.device("cpu"))[
        f"collection/return_by_opponent/{name}"
    ]
    right = second.reducible_values(torch.device("cpu"))[
        f"collection/return_by_opponent/{name}"
    ]

    assert (left.total + right.total).item() == 70.0
    assert (left.count + right.count).item() == 4
    assert ((left.total + right.total) / (left.count + right.count)).item() == 17.5


class _RecordingStrategy:
    """Replay rank-zero object broadcasts for one simulated sibling rank."""

    def __init__(
        self,
        replay: list[object] | None = None,
        checkpoint_writes: list[str] | None = None,
    ) -> None:
        self.events: list[str] = []
        self.payloads: list[object] = []
        self.replay = list(replay or [])
        self.checkpoint_writes = checkpoint_writes

    def broadcast(self, value: object, src: int = 0) -> object:
        del src
        self.events.append("broadcast")
        if self.replay:
            return self.replay.pop(0)
        self.payloads.append(value)
        return value

    def barrier(self) -> None:
        self.events.append("barrier")

    def save_checkpoint(self, checkpoint: object, path: Path) -> None:
        if self.checkpoint_writes is None:
            raise AssertionError("nonzero rank attempted a checkpoint write")
        self.checkpoint_writes.append(str(path))
        torch.save(checkpoint, path)


def test_fit_start_materializes_initial_file_and_actor_once_on_rank_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All bootstrap sources become one rank-zero generation before collection."""
    initial_policy = ToadLightningModule(_stream_config()).policy.state_dict()
    initial_path = tmp_path / "initial.pt"
    torch.save(initial_policy, initial_path)
    config = _stream_config().model_copy(
        update={
            "population": _stream_config().population.model_copy(
                update={
                    "initial_snapshots": (initial_path,),
                    "snapshot_at_start": True,
                    "pool_capacity": 4,
                }
            ),
            "runtime": _stream_config().runtime.model_copy(
                update={"output_dir": tmp_path}
            ),
        }
    )
    rank0_module = ToadLightningModule(config)
    rank0_data = data.ToadDataModule(config, data.ReferenceRoundSource(config))
    writes: list[str] = []
    rank0_strategy = _RecordingStrategy(checkpoint_writes=writes)
    rank0_trainer = cast(
        lightning.Trainer,
        SimpleNamespace(
            datamodule=rank0_data,
            is_global_zero=True,
            global_rank=0,
            world_size=2,
            strategy=rank0_strategy,
            _checkpoint_connector=SimpleNamespace(dump_checkpoint=lambda: {}),
        ),
    )
    monkeypatch.setattr(
        toad_callbacks,
        "_all_gather_objects",
        lambda local, _world, **_kwargs: (
            local,
            {"ok": True, "rank": 1, "type": None, "message": None},
        ),
    )
    PopulationSnapshotCallback().on_fit_start(rank0_trainer, rank0_module)

    rank1_module = ToadLightningModule(config)
    rank1_data = data.ToadDataModule(config, data.ReferenceRoundSource(config))
    rank1_strategy = _RecordingStrategy(rank0_strategy.payloads)
    rank1_trainer = cast(
        lightning.Trainer,
        SimpleNamespace(
            datamodule=rank1_data,
            is_global_zero=False,
            global_rank=1,
            world_size=2,
            strategy=rank1_strategy,
        ),
    )
    PopulationSnapshotCallback().on_fit_start(rank1_trainer, rank1_module)

    snapshots = list((tmp_path / "population").glob("snapshot-*.pt"))
    assert len(snapshots) == 2
    assert len(rank0_module.population_manifest.entries) == 2
    assert rank1_module.population_manifest == rank0_module.population_manifest
    assert rank0_data.source.pool is not None
    assert rank1_data.source.pool is not None


def test_resume_preserves_capacity_evicted_bootstrap_manifest(tmp_path: Path) -> None:
    """A restored nonempty generation must not infer an evicted initial is missing."""
    base = _stream_config()
    initial_path = tmp_path / "initial.pt"
    torch.save(ToadLightningModule(base).policy.state_dict(), initial_path)
    config = base.model_copy(
        update={
            "population": base.population.model_copy(
                update={
                    "initial_snapshots": (initial_path,),
                    "pool_capacity": 1,
                    "snapshot_every_environment_steps": 8,
                }
            ),
            "runtime": base.runtime.model_copy(update={"output_dir": tmp_path}),
        }
    )

    def save_checkpoint(path: str) -> None:
        Path(path).write_bytes(b"authoritative")

    module = ToadLightningModule(config)
    data_module = data.ToadDataModule(config, data.ReferenceRoundSource(config))
    trainer = cast(
        lightning.Trainer,
        SimpleNamespace(datamodule=data_module, save_checkpoint=save_checkpoint),
    )
    callback = PopulationSnapshotCallback()
    callback.on_fit_start(trainer, module)
    assert module.population_manifest.entries[0].run_id.startswith("initial:")

    module.environment_steps = 8
    module.collection_round = 1
    callback.on_train_batch_end(
        trainer,
        module,
        None,
        SimpleNamespace(end_of_round=True),
        0,
    )
    authoritative = module.population_manifest
    assert len(authoritative.entries) == 1
    assert authoritative.entries[0].run_id == config.curriculum.phase
    manifest_bytes = (tmp_path / "population" / "manifest.json").read_bytes()
    checkpoint: dict[str, object] = {}
    module.on_save_checkpoint(checkpoint)
    data_state = data_module.state_dict()

    resumed_module = ToadLightningModule(config)
    resumed_module.on_load_checkpoint(checkpoint)
    resumed_data = data.ToadDataModule(
        config,
        data.ReferenceRoundSource(config),
    )
    resumed_data.load_state_dict(data_state)
    resumed_trainer = cast(
        lightning.Trainer,
        SimpleNamespace(datamodule=resumed_data, save_checkpoint=save_checkpoint),
    )

    PopulationSnapshotCallback().on_fit_start(resumed_trainer, resumed_module)

    assert resumed_module.population_manifest == authoritative
    assert resumed_data.source.pool is not None
    assert resumed_data.source.pool.manifest == authoritative
    assert (tmp_path / "population" / "manifest.json").read_bytes() == manifest_bytes


def test_boundary_install_failure_is_acknowledged_by_every_rank(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A peer that cannot install the payload stops all ranks before collection."""
    config = _stream_config().model_copy(
        update={
            "runtime": _stream_config().runtime.model_copy(
                update={"output_dir": tmp_path}
            )
        }
    )
    module = ToadLightningModule(config)
    data_module = data.ToadDataModule(config, data.ReferenceRoundSource(config))
    strategy = _RecordingStrategy()
    trainer = cast(
        lightning.Trainer,
        SimpleNamespace(
            datamodule=data_module,
            is_global_zero=True,
            global_rank=0,
            world_size=2,
            strategy=strategy,
        ),
    )
    peer_failure = {
        "ok": False,
        "rank": 1,
        "type": "SnapshotIntegrityError",
        "message": "peer install failed",
    }
    monkeypatch.setattr(
        toad_callbacks,
        "_all_gather_objects",
        lambda local, _world, **_kwargs: (local, peer_failure),
        raising=False,
    )

    with pytest.raises(DistributedBoundaryError, match="peer install failed"):
        toad_callbacks._run_rank_zero_boundary(
            trainer,
            module,
            lambda: None,
            include_manifest=False,
        )


def test_boundary_checkpoint_is_rank_zero_write_then_barrier_and_broadcast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nonzero ranks enter the callback but cannot publish a second checkpoint."""
    monkeypatch.setattr(
        toad_callbacks,
        "_all_gather_objects",
        lambda local, world_size, **_kwargs: (local,) * world_size,
    )
    path = tmp_path / "ddp-boundary"
    config = ToadConfig.model_validate(
        {
            "runtime": {
                "output_dir": path,
                "checkpoint_every_environment_steps": 1,
            }
        }
    )
    rank0_module = ToadLightningModule(config)
    rank0_module.environment_steps = 1
    rank0_module.collection_round = 1
    rank0_data = data.ToadDataModule(config, data.ReferenceRoundSource(config))
    writes: list[str] = []
    rank0_strategy = _RecordingStrategy(checkpoint_writes=writes)

    def forbidden_trainer_save(_pathname: str) -> None:
        raise AssertionError(
            "DDP callback used Trainer.save_checkpoint and its barrier"
        )

    rank0_trainer = SimpleNamespace(
        datamodule=rank0_data,
        save_checkpoint=forbidden_trainer_save,
        is_global_zero=True,
        global_rank=0,
        world_size=2,
        strategy=rank0_strategy,
        _checkpoint_connector=SimpleNamespace(dump_checkpoint=lambda: {"rank": 0}),
    )
    rank0_callback = BoundaryCheckpoint(path, every_environment_steps=1)
    rank0_callback._next_environment_steps = 1
    batch = SimpleNamespace(end_of_round=True)
    rank0_callback.on_train_batch_end(
        rank0_trainer,  # ty: ignore[invalid-argument-type]
        rank0_module,
        None,
        batch,
        0,
    )

    rank1_module = ToadLightningModule(config)
    rank1_module.environment_steps = 1
    rank1_module.collection_round = 1
    rank1_data = data.ToadDataModule(config, data.ReferenceRoundSource(config))
    rank1_strategy = _RecordingStrategy(rank0_strategy.payloads)

    def forbidden_write(_pathname: str) -> None:
        raise AssertionError("nonzero rank attempted a checkpoint write")

    rank1_trainer = SimpleNamespace(
        datamodule=rank1_data,
        save_checkpoint=forbidden_write,
        is_global_zero=False,
        global_rank=1,
        world_size=2,
        strategy=rank1_strategy,
    )
    rank1_callback = BoundaryCheckpoint(path, every_environment_steps=1)
    rank1_callback._next_environment_steps = 1
    rank1_callback.on_train_batch_end(
        rank1_trainer,  # ty: ignore[invalid-argument-type]
        rank1_module,
        None,
        batch,
        0,
    )

    assert len(writes) == 1
    assert rank0_strategy.events == ["broadcast", "barrier", "broadcast"]
    assert rank1_strategy.events == ["broadcast", "barrier", "broadcast"]
    assert rank1_module.environment_steps == rank0_module.environment_steps == 1


def test_terminal_checkpoint_is_exact_and_rank_zero_owned_below_cadence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fit-end publishes one exact boundary and gives every rank its identity."""
    monkeypatch.setattr(
        toad_callbacks,
        "_all_gather_objects",
        lambda local, world_size, **_kwargs: (local,) * world_size,
    )
    path = tmp_path / "ddp-terminal"
    config = ToadConfig.model_validate(
        {
            "runtime": {
                "output_dir": path,
                "checkpoint_every_environment_steps": 100,
            }
        }
    )
    stale = path / "step-999.ckpt"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")

    rank0_module = ToadLightningModule(config)
    rank0_module.environment_steps = 40
    rank0_module.collection_round = 1
    rank0_data = data.ToadDataModule(config, data.ReferenceRoundSource(config))
    writes: list[str] = []
    rank0_strategy = _RecordingStrategy(checkpoint_writes=writes)
    rank0_trainer = SimpleNamespace(
        datamodule=rank0_data,
        is_global_zero=True,
        global_rank=0,
        world_size=2,
        strategy=rank0_strategy,
        _checkpoint_connector=SimpleNamespace(dump_checkpoint=lambda: {"rank": 0}),
    )
    rank0_callback = BoundaryCheckpoint(path, every_environment_steps=100)
    rank0_callback.on_fit_end(
        rank0_trainer,  # ty: ignore[invalid-argument-type]
        rank0_module,
    )

    rank1_module = ToadLightningModule(config)
    rank1_module.environment_steps = 40
    rank1_module.collection_round = 1
    rank1_data = data.ToadDataModule(config, data.ReferenceRoundSource(config))
    rank1_strategy = _RecordingStrategy(rank0_strategy.payloads)
    rank1_trainer = SimpleNamespace(
        datamodule=rank1_data,
        is_global_zero=False,
        global_rank=1,
        world_size=2,
        strategy=rank1_strategy,
    )
    rank1_callback = BoundaryCheckpoint(path, every_environment_steps=100)
    rank1_callback.on_fit_end(
        rank1_trainer,  # ty: ignore[invalid-argument-type]
        rank1_module,
    )

    exact = path / "step-40.ckpt"
    assert writes == [str(exact.with_suffix(".tmp"))]
    assert exact.is_file()
    assert stale.read_bytes() == b"stale"
    assert rank0_callback.final_checkpoint == exact
    assert rank1_callback.final_checkpoint == exact
    assert rank0_strategy.events == ["broadcast", "barrier", "broadcast"]
    assert rank1_strategy.events == ["broadcast", "barrier", "broadcast"]


class _StopAfterOneRound(lightning.Callback):
    """End the first integration phase at its first durable boundary."""

    def on_train_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        del outputs, batch_idx
        module = cast(ToadLightningModule, pl_module)
        learner_batch = cast(data.LearnerBatch, batch)
        if learner_batch.end_of_round and module.collection_round == 1:
            trainer.should_stop = True


def _state_digest(state_dict: dict[str, torch.Tensor]) -> str:
    """Hash canonical tensor content without depending on serialization bytes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _parameter_digest(module: ToadLightningModule) -> str:
    """Hash a module's policy parameters in their canonical tensor form."""
    return _state_digest(dict(module.policy.state_dict()))


class _DdpProbe(lightning.Callback):
    """Persist rank-local evidence after a real subprocess DDP run."""

    def __init__(
        self,
        output_dir: Path,
        phase: str,
        seen_game_ids: list[int],
        seen_opponent_ids: list[str],
    ) -> None:
        self.output_dir = output_dir
        self.phase = phase
        self.seen_game_ids = seen_game_ids
        self.seen_opponent_ids = seen_opponent_ids
        self.batch_count = 0
        self.first_markers = 0
        self.end_markers = 0
        self.initial_parameter_digest: str | None = None

    def on_fit_start(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
    ) -> None:
        """Capture the restored pre-update policy identically on every rank."""
        del trainer
        self.initial_parameter_digest = _parameter_digest(
            cast(ToadLightningModule, pl_module)
        )

    def on_train_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        del trainer, pl_module, outputs, batch_idx
        learner_batch = cast(data.LearnerBatch, batch)
        for game_id in learner_batch.game_ids:
            if game_id not in self.seen_game_ids:
                self.seen_game_ids.append(game_id)
        for opponent_id in learner_batch.opponent_ids:
            if opponent_id not in self.seen_opponent_ids:
                self.seen_opponent_ids.append(opponent_id)
        self.batch_count += 1
        self.first_markers += int(learner_batch.first_of_round)
        self.end_markers += int(learner_batch.end_of_round)

    def on_fit_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
    ) -> None:
        module = cast(ToadLightningModule, pl_module)
        source = _data_module_for_probe(trainer).source
        optimizer = trainer.optimizers[0]
        payload = {
            "rank": trainer.global_rank,
            "game_ids": self.seen_game_ids,
            "opponent_ids": self.seen_opponent_ids,
            "batch_count": self.batch_count,
            "first_markers": self.first_markers,
            "end_markers": self.end_markers,
            "initial_parameter_digest": self.initial_parameter_digest,
            "parameter_digest": _parameter_digest(module),
            "environment_steps": module.environment_steps,
            "collection_round": module.collection_round,
            "next_game_id": source.next_game_id,
            "next_round_id": source._next_round_id,
            "manifest_entries": len(module.population_manifest.entries),
            "manifest": [
                {
                    "state_digest": _state_digest(
                        cast(
                            dict[str, torch.Tensor],
                            torch.load(
                                entry.path,
                                map_location="cpu",
                                weights_only=True,
                            ),
                        )
                    ),
                    "environment_steps": entry.environment_steps,
                    "round_id": entry.round_id,
                }
                for entry in module.population_manifest.entries
            ],
            "actor_version": module.actor_version,
            "actor_source_global_step": module.actor_source_global_step,
            "entropy_state": {
                name: {
                    "target": state.target,
                    "multiplier": state.multiplier,
                    "last_steps": state.last_steps,
                }
                for name, state in module.entropy_state.items()
            },
            "lr": [group["lr"] for group in optimizer.param_groups],
        }
        final = self.output_dir / f"{self.phase}-rank-{trainer.global_rank}.json"
        temporary = final.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True))
        temporary.replace(final)


def _data_module_for_probe(trainer: lightning.Trainer) -> data.ToadDataModule:
    """Narrow Lightning's optional datamodule type for the integration probe."""
    module = getattr(trainer, "datamodule", None)
    if not isinstance(module, data.ToadDataModule):
        raise RuntimeError("DDP fixture requires a ToadDataModule")
    return module


def _ddp_fixture_config(
    output_dir: Path,
    resume: Path | None,
    *,
    native: bool = False,
) -> ToadConfig:
    """Return the tiny, fully offline two-rank integration configuration."""
    return ToadConfig.model_validate(
        {
            "model": {"blocks": 1, "channels": 4},
            "population": {
                "selfplay": 0.0,
                "scripted": 1.0,
                "environments_per_rank": 1,
                "collection_processes": 1,
                "actor_sync_every_rounds": 1,
                "snapshot_every_environment_steps": 8,
                "initial_snapshots": [output_dir / "bootstrap.pt"],
            },
            "optimizer": {
                "unroll_length": 2,
                "batch_segments": 1,
                "value_warmup_batches": 0,
                "value_passes": 0,
            },
            "runtime": {
                "accelerator": "cpu",
                "devices": 2,
                "strategy": "ddp",
                "precision": "32-true",
                "total_environment_steps": 16,
                "checkpoint_every_environment_steps": 8,
                "output_dir": output_dir,
                "resume": resume,
                "rollout_backend": "native" if native else "reference",
                "rollout_device": "cpu",
            },
        }
    )


def _ddp_fixture_trajectory(game_id: int) -> Trajectory:
    """Make rank inputs distinct while retaining the valid learner schema."""
    trajectory = _trajectory()
    scale = float(game_id + 1)
    rewards = trajectory.rewards * scale
    return replace(
        trajectory,
        board=trajectory.board + scale / 100.0,
        scalars=trajectory.scalars + scale / 100.0,
        rewards=rewards,
        own=rewards,
        shaped=rewards,
        shaped_money=rewards,
        margin=rewards,
        sparse=rewards,
        final_margin=scale,
        final_bank=scale,
        final_capital=scale,
    )


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_shortened_collection_horizon_matches_expected_round_geometry(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expected DDP quotas use the source's real horizon on both backends."""
    config = ToadConfig.model_validate(
        {
            "model": {"blocks": 1, "channels": 4},
            "population": {
                "selfplay": 0.0,
                "scripted": 1.0,
                "environments_per_rank": 1,
                "collection_processes": 1,
            },
            "optimizer": {
                "unroll_length": 2,
                "batch_segments": 1,
                "value_warmup_batches": 0,
                "value_passes": 0,
            },
            "runtime": {
                "rollout_backend": backend,
                "rollout_device": "cpu",
            },
        }
    )
    assignment = data.CollectionAssignment(
        game_id=0,
        seed=117,
        opponent_id="economic",
        kind=data.BatchKind.SCRIPTED,
    )
    if backend == "native":
        monkeypatch.setattr(sim_day, "EPISODE_STEPS", 5)
        source = data.NativeRoundSource(
            config,
            assignments=(assignment,),
            episode_steps=5,
        )
        source.publish_actor(ToadLightningModule(config).policy.state_dict(), version=0)
    else:
        source = data.ReferenceRoundSource(
            config,
            assignments=(assignment,),
            collect_assignment=lambda _assignment: (_trajectory(),),
            episode_steps=5,
        )

    def gather(local: object, world_size: int, **_kwargs: object) -> tuple[object, ...]:
        if isinstance(local, data.RoundBatchSignature):
            return (local, replace(local, rank=1))
        return (local,) * world_size

    monkeypatch.setattr(data, "_all_gather_objects", gather)
    data_module = data.ToadDataModule(config, source)
    data_module.trainer = cast(
        lightning.Trainer,
        SimpleNamespace(global_rank=0, world_size=2),
    )
    data_module.setup("fit")
    batches = tuple(source)
    expected = data.expected_round_batch_counts(
        config,
        episode_steps=source.episode_steps,
    )

    assert data.expected_round_batch_counts(config).policy == 359
    assert expected.policy == sum(not batch.baseline_only for batch in batches) == 2
    assert expected.value == sum(batch.baseline_only for batch in batches) == 0


def _run_ddp_fixture(output_dir: Path, phase: str, resume: Path | None) -> None:
    """Run inside Lightning's parent and child subprocesses."""
    torch.set_num_threads(1)
    lightning.seed_everything(117, workers=True)
    config = _ddp_fixture_config(output_dir, resume)
    seen_game_ids: list[int] = []
    seen_opponent_ids: list[str] = []

    def collect(assignment: data.CollectionAssignment) -> tuple[Trajectory]:
        seen_game_ids.append(assignment.game_id)
        seen_opponent_ids.append(assignment.opponent_id)
        return (_ddp_fixture_trajectory(assignment.game_id),)

    source = data.ReferenceRoundSource(
        config,
        collect_assignment=collect,
        episode_steps=5,
    )
    data_module = data.ToadDataModule(config, source)
    callbacks: list[lightning.Callback] = [
        ActorSyncCallback(1),
        EnvironmentStepStop(config.runtime.total_environment_steps),
        PopulationSnapshotCallback(),
        BoundaryCheckpoint(config.runtime.output_dir),
        _DdpProbe(output_dir, phase, seen_game_ids, seen_opponent_ids),
    ]
    if phase.endswith("initial"):
        callbacks.append(_StopAfterOneRound())
    trainer = lightning.Trainer(
        accelerator="cpu",
        devices=2,
        strategy="ddp",
        precision="32-true",
        max_steps=-1,
        max_epochs=-1,
        use_distributed_sampler=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        callbacks=callbacks,
    )
    trainer.fit(
        ToadLightningModule(config),
        datamodule=data_module,
        ckpt_path=resume,
    )


def _run_native_ddp_fixture(output_dir: Path, phase: str, resume: Path | None) -> None:
    """Run the real tensor simulator and NativeRoundSource under CPU Gloo."""
    torch.set_num_threads(1)
    lightning.seed_everything(117, workers=True)
    sim_day.EPISODE_STEPS = 5  # ty: ignore[invalid-assignment]
    config = _ddp_fixture_config(output_dir, resume, native=True)
    source = data.NativeRoundSource(config, episode_steps=5)
    data_module = data.ToadDataModule(config, source)
    seen_game_ids: list[int] = []
    seen_opponent_ids: list[str] = []
    callbacks: list[lightning.Callback] = [
        ActorSyncCallback(1),
        EnvironmentStepStop(config.runtime.total_environment_steps),
        PopulationSnapshotCallback(),
        BoundaryCheckpoint(config.runtime.output_dir),
        _DdpProbe(output_dir, phase, seen_game_ids, seen_opponent_ids),
    ]
    if phase.endswith("initial"):
        callbacks.append(_StopAfterOneRound())
    trainer = lightning.Trainer(
        accelerator="cpu",
        devices=2,
        strategy="ddp",
        precision="32-true",
        max_steps=-1,
        max_epochs=-1,
        use_distributed_sampler=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        callbacks=callbacks,
    )
    trainer.fit(
        ToadLightningModule(config),
        datamodule=data_module,
        ckpt_path=resume,
    )


def _launch_ddp_fixture(
    output_dir: Path, phase: str, resume: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Launch a clean script so Lightning can re-exec it for rank one."""
    output_dir.mkdir(parents=True, exist_ok=True)
    bootstrap = output_dir / "bootstrap.pt"
    if not bootstrap.exists():
        torch.manual_seed(314)
        bootstrap_config = ToadConfig.model_validate(
            {"model": {"blocks": 1, "channels": 4}}
        )
        torch.save(ToadLightningModule(bootstrap_config).policy.state_dict(), bootstrap)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--ddp-fixture",
        str(output_dir),
        phase,
    ]
    if resume is not None:
        command.append(str(resume))
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
            "WANDB_MODE": "disabled",
            "OMP_NUM_THREADS": "1",
        }
    )
    return subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def _launch_native_ddp_fixture(
    output_dir: Path, phase: str, resume: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Launch a clean two-rank native-source acceptance process."""
    output_dir.mkdir(parents=True, exist_ok=True)
    bootstrap = output_dir / "bootstrap.pt"
    if not bootstrap.exists():
        torch.manual_seed(314)
        bootstrap_config = ToadConfig.model_validate(
            {"model": {"blocks": 1, "channels": 4}}
        )
        torch.save(ToadLightningModule(bootstrap_config).policy.state_dict(), bootstrap)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--native-ddp-fixture",
        str(output_dir),
        phase,
    ]
    if resume is not None:
        command.append(str(resume))
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
            "WANDB_MODE": "disabled",
            "OMP_NUM_THREADS": "1",
        }
    )
    return subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def _probe_payloads(output_dir: Path, phase: str) -> list[dict[str, object]]:
    """Load the two rank-local probe records in rank order."""
    return [
        json.loads((output_dir / f"{phase}-rank-{rank}.json").read_text())
        for rank in range(2)
    ]


def test_real_two_process_cpu_ddp_synchronizes_synthetic_split_and_resume(
    tmp_path: Path,
) -> None:
    """Exercise synchronization and resume with distinct synthetic rank inputs."""
    initial = _launch_ddp_fixture(tmp_path, "initial")
    assert initial.returncode == 0, initial.stdout + initial.stderr
    first = _probe_payloads(tmp_path, "initial")

    assert [payload["game_ids"] for payload in first] == [[0], [1]]
    assert [payload["opponent_ids"] for payload in first] == [
        ["economic"],
        ["economic"],
    ]
    assert all("scripted_win_rates" not in payload for payload in first)
    assert [payload["batch_count"] for payload in first] == [2, 2]
    assert [payload["first_markers"] for payload in first] == [1, 1]
    assert [payload["end_markers"] for payload in first] == [1, 1]
    assert len({payload["parameter_digest"] for payload in first}) == 1
    assert len({payload["initial_parameter_digest"] for payload in first}) == 1
    assert first[0]["parameter_digest"] != first[0]["initial_parameter_digest"]
    assert [payload["environment_steps"] for payload in first] == [8, 8]
    assert [payload["next_game_id"] for payload in first] == [2, 2]
    assert [payload["manifest_entries"] for payload in first] == [2, 2]
    checkpoints = sorted(tmp_path.glob("step-*.ckpt"))
    snapshots = sorted((tmp_path / "population").glob("snapshot-*.pt"))
    assert [path.name for path in checkpoints] == ["step-0.ckpt", "step-8.ckpt"]
    assert len(snapshots) == 2

    resumed = _launch_ddp_fixture(tmp_path, "resume", checkpoints[-1])
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    second = _probe_payloads(tmp_path, "resume")

    assert [payload["game_ids"] for payload in second] == [[2], [3]]
    assert [payload["opponent_ids"] for payload in second] == [
        ["economic"],
        ["economic"],
    ]
    assert all("scripted_win_rates" not in payload for payload in second)
    assert [payload["batch_count"] for payload in second] == [2, 2]
    assert len({payload["parameter_digest"] for payload in second}) == 1
    assert [payload["environment_steps"] for payload in second] == [16, 16]
    assert [payload["collection_round"] for payload in second] == [2, 2]
    assert [payload["next_game_id"] for payload in second] == [4, 4]
    assert [payload["manifest_entries"] for payload in second] == [3, 3]

    uninterrupted_dir = tmp_path / "uninterrupted"
    uninterrupted = _launch_ddp_fixture(uninterrupted_dir, "uninterrupted")
    assert uninterrupted.returncode == 0, uninterrupted.stdout + uninterrupted.stderr
    full = _probe_payloads(uninterrupted_dir, "uninterrupted")

    assert [payload["game_ids"] for payload in full] == [[0, 2], [1, 3]]
    assert [payload["opponent_ids"] for payload in full] == [
        ["economic", "economic"],
        ["economic", "economic"],
    ]
    assert all("scripted_win_rates" not in payload for payload in full)
    assert [payload["batch_count"] for payload in full] == [4, 4]
    assert [payload["first_markers"] for payload in full] == [2, 2]
    assert [payload["end_markers"] for payload in full] == [2, 2]
    for resumed_rank, uninterrupted_rank in zip(second, full, strict=True):
        assert (
            resumed_rank["parameter_digest"] == uninterrupted_rank["parameter_digest"]
        )
        assert (
            resumed_rank["environment_steps"] == uninterrupted_rank["environment_steps"]
        )
        assert (
            resumed_rank["collection_round"] == uninterrupted_rank["collection_round"]
        )
        assert resumed_rank["next_game_id"] == uninterrupted_rank["next_game_id"]
        assert resumed_rank["next_round_id"] == uninterrupted_rank["next_round_id"]
        assert resumed_rank["manifest"] == uninterrupted_rank["manifest"]
        assert resumed_rank["actor_version"] == uninterrupted_rank["actor_version"]
        assert (
            resumed_rank["actor_source_global_step"]
            == uninterrupted_rank["actor_source_global_step"]
        )
        assert resumed_rank["entropy_state"] == uninterrupted_rank["entropy_state"]
        assert resumed_rank["lr"] == uninterrupted_rank["lr"]


def test_real_two_process_cpu_ddp_native_source_split_and_resume(
    tmp_path: Path,
) -> None:
    """NativeRoundSource trains through the tensor simulator on both Gloo ranks."""
    initial = _launch_native_ddp_fixture(tmp_path, "native-initial")
    assert initial.returncode == 0, initial.stdout + initial.stderr
    first = _probe_payloads(tmp_path, "native-initial")

    assert [payload["game_ids"] for payload in first] == [[0], [1]]
    assert set(cast(list[int], first[0]["game_ids"])).isdisjoint(
        cast(list[int], first[1]["game_ids"])
    )
    assert [payload["opponent_ids"] for payload in first] == [
        ["economic"],
        ["economic"],
    ]
    assert [payload["batch_count"] for payload in first] == [2, 2]
    assert [payload["first_markers"] for payload in first] == [1, 1]
    assert [payload["end_markers"] for payload in first] == [1, 1]
    assert len({payload["parameter_digest"] for payload in first}) == 1
    assert first[0]["parameter_digest"] != first[0]["initial_parameter_digest"]
    assert [payload["environment_steps"] for payload in first] == [8, 8]
    assert [payload["next_game_id"] for payload in first] == [2, 2]
    checkpoints = sorted(tmp_path.glob("step-*.ckpt"))
    snapshots = sorted((tmp_path / "population").glob("snapshot-*.pt"))
    assert [path.name for path in checkpoints] == ["step-0.ckpt", "step-8.ckpt"]
    assert len(snapshots) == 2

    resumed = _launch_native_ddp_fixture(tmp_path, "native-resume", checkpoints[-1])
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    second = _probe_payloads(tmp_path, "native-resume")
    assert [payload["game_ids"] for payload in second] == [[2], [3]]
    assert [payload["batch_count"] for payload in second] == [2, 2]
    assert len({payload["parameter_digest"] for payload in second}) == 1
    assert [payload["environment_steps"] for payload in second] == [16, 16]
    assert [payload["next_game_id"] for payload in second] == [4, 4]

    uninterrupted_dir = tmp_path / "native-uninterrupted"
    uninterrupted = _launch_native_ddp_fixture(
        uninterrupted_dir, "native-uninterrupted"
    )
    assert uninterrupted.returncode == 0, uninterrupted.stdout + uninterrupted.stderr
    full = _probe_payloads(uninterrupted_dir, "native-uninterrupted")
    assert [payload["game_ids"] for payload in full] == [[0, 2], [1, 3]]
    assert [payload["batch_count"] for payload in full] == [4, 4]
    for resumed_rank, uninterrupted_rank in zip(second, full, strict=True):
        for name in (
            "parameter_digest",
            "environment_steps",
            "collection_round",
            "next_game_id",
            "next_round_id",
            "manifest",
            "actor_version",
            "actor_source_global_step",
            "entropy_state",
            "lr",
        ):
            assert resumed_rank[name] == uninterrupted_rank[name]


@pytest.mark.slow
def test_fixed_seed_economic_reference_collection_resumes_within_interval() -> None:
    """Real economic rollouts retain their characterized result after resume."""
    config = ToadConfig.model_validate(
        {
            "model": {"blocks": 1, "channels": 4},
            "population": {
                "selfplay": 0.0,
                "scripted": 1.0,
                "environments_per_rank": 1,
                "collection_processes": 1,
            },
            "optimizer": {
                "unroll_length": 719,
                "batch_segments": 1,
                "value_warmup_batches": 0,
                "value_passes": 0,
            },
            "runtime": {"seed": 117},
        }
    )
    torch.manual_seed(314)
    actor = ToadLightningModule(config).policy.state_dict()
    uninterrupted_source = data.ReferenceRoundSource(config)
    uninterrupted_data = data.ToadDataModule(config, uninterrupted_source)
    uninterrupted_data.publish_actor(actor, version=0)

    first = tuple(uninterrupted_source)
    split_boundary = uninterrupted_data.state_dict()
    uninterrupted_second = tuple(uninterrupted_source)
    resumed_source = data.ReferenceRoundSource(config)
    resumed_data = data.ToadDataModule(config, resumed_source)
    resumed_data.load_state_dict(split_boundary)
    resumed_second = tuple(resumed_source)

    assert len(first) == len(uninterrupted_second) == len(resumed_second) == 1
    assert first[0].game_ids == (0,)
    assert first[0].seeds == (117,)
    assert first[0].opponent_ids == ("economic",)
    assert uninterrupted_second[0].game_ids == resumed_second[0].game_ids == (1,)
    assert uninterrupted_second[0].seeds == resumed_second[0].seeds == (118,)
    assert (
        uninterrupted_second[0].opponent_ids
        == resumed_second[0].opponent_ids
        == ("economic",)
    )
    uninterrupted_metrics = {
        name: value
        for name, value in uninterrupted_second[0].round_metrics.items()
        if not name.startswith("throughput/")
    }
    resumed_metrics = {
        name: value
        for name, value in resumed_second[0].round_metrics.items()
        if not name.startswith("throughput/")
    }
    assert uninterrupted_metrics == pytest.approx(resumed_metrics, nan_ok=True)
    observed_win_rate = (
        sum(
            float(batch[0].round_metrics["objective/win_rate_vs_econ"])
            for batch in (first, uninterrupted_second)
        )
        / 2.0
    )
    lower, upper = _ECONOMIC_CONTROL_WIN_RATE_INTERVAL
    assert lower <= observed_win_rate <= upper
    assert resumed_source.next_game_id == uninterrupted_source.next_game_id == 2
    assert resumed_source._next_round_id == uninterrupted_source._next_round_id == 2


@pytest.mark.economic_acceptance
def test_production_trainer_resume_retains_heldout_economic_result(
    tmp_path: Path,
) -> None:
    """Train, checkpoint, resume, and evaluate a fixed real economic budget."""
    games_per_round = 2
    round_steps = games_per_round * 719
    total_steps = round_steps * 2

    def config(output_dir: Path) -> ToadConfig:
        return ToadConfig.model_validate(
            {
                "model": {"blocks": 1, "channels": 4},
                "population": {
                    "selfplay": 0.0,
                    "scripted": 1.0,
                    "environments_per_rank": games_per_round,
                    "collection_processes": games_per_round,
                    "actor_sync_every_rounds": 1,
                },
                "optimizer": {
                    "unroll_length": 719,
                    "batch_segments": games_per_round,
                    "value_warmup_batches": 0,
                    "value_passes": 0,
                },
                "runtime": {
                    "seed": 117,
                    "accelerator": "cpu",
                    "devices": 1,
                    "precision": "32-true",
                    "total_environment_steps": total_steps,
                    "checkpoint_every_environment_steps": round_steps,
                    "output_dir": output_dir,
                },
            }
        )

    torch.manual_seed(314)
    seed_module = ToadLightningModule(config(tmp_path / "seed"))
    initial = {
        name: tensor.detach().cpu().clone()
        for name, tensor in seed_module.policy.state_dict().items()
    }

    def fit(
        resolved: ToadConfig,
        *,
        checkpoint: Path | None = None,
        stop_after_one: bool = False,
    ) -> tuple[ToadLightningModule, data.ReferenceRoundSource]:
        lightning.seed_everything(117, workers=True)
        source = data.ReferenceRoundSource(resolved)
        data_module = data.ToadDataModule(resolved, source)
        module = ToadLightningModule(resolved)
        if checkpoint is None:
            module.policy.load_state_dict(initial, strict=True)
        callbacks: list[lightning.Callback] = [
            ActorSyncCallback(1),
            EnvironmentStepStop(total_steps),
            BoundaryCheckpoint(resolved.runtime.output_dir),
        ]
        if stop_after_one:
            callbacks.append(_StopAfterOneRound())
        trainer = lightning.Trainer(
            accelerator="cpu",
            devices=1,
            precision="32-true",
            max_steps=-1,
            max_epochs=-1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            deterministic=True,
            gradient_clip_val=resolved.optimizer.clip_grad_norm,
            gradient_clip_algorithm="norm",
            callbacks=callbacks,
            use_distributed_sampler=False,
        )
        trainer.fit(module, datamodule=data_module, ckpt_path=checkpoint)
        return module, source

    full_config = config(tmp_path / "full")
    full_module, full_source = fit(full_config)
    split_config = config(tmp_path / "split")
    split_module, _split_source = fit(split_config, stop_after_one=True)
    boundary = split_config.runtime.output_dir / f"step-{round_steps}.ckpt"
    assert boundary.is_file()
    resumed_module, resumed_source = fit(split_config, checkpoint=boundary)

    assert split_module.environment_steps == round_steps
    assert (
        full_module.environment_steps == resumed_module.environment_steps == total_steps
    )
    assert full_source.next_game_id == resumed_source.next_game_id == 4
    for name, tensor in full_module.policy.state_dict().items():
        torch.testing.assert_close(
            tensor,
            resumed_module.policy.state_dict()[name],
            rtol=0.0,
            atol=0.0,
        )

    heldout_seeds = (911, 919, 929, 937, 947, 953, 967, 977)
    heldout = tuple(
        data.CollectionAssignment(
            game_id=10_000 + index,
            seed=seed,
            kind=data.BatchKind.SCRIPTED,
            opponent_id="economic",
        )
        for index, seed in enumerate(heldout_seeds)
    )

    def evaluate(module: ToadLightningModule) -> tuple[float, ...]:
        evaluation = data.ReferenceRoundSource(
            config(tmp_path / "evaluation"), assignments=heldout
        )
        evaluation.publish_actor(module.policy.state_dict(), module.actor_version)
        collected = evaluation._collect_round(heldout)
        return tuple(
            float(trajectory.final_margin)
            for _assignment, trajectories in collected
            for trajectory in trajectories
        )

    full_margins = evaluate(full_module)
    resumed_margins = evaluate(resumed_module)
    assert resumed_margins == pytest.approx(full_margins, rel=0.0, abs=0.0)
    expected_margins = (
        -169_549.0,
        -72_036.0,
        -113_768.0,
        -147_151.0,
        -131_777.0,
        -107_019.0,
        -128_220.0,
        -156_478.0,
    )
    assert full_margins == pytest.approx(expected_margins, rel=0.0, abs=0.0)
    assert sum(margin > 0.0 for margin in full_margins) / len(full_margins) == 0.0


if __name__ == "__main__" and len(sys.argv) >= 4:
    if sys.argv[1] == "--ddp-fixture":
        _run_ddp_fixture(
            Path(sys.argv[2]),
            sys.argv[3],
            Path(sys.argv[4]) if len(sys.argv) == 5 else None,
        )
    elif sys.argv[1] == "--native-ddp-fixture":
        _run_native_ddp_fixture(
            Path(sys.argv[2]),
            sys.argv[3],
            Path(sys.argv[4]) if len(sys.argv) == 5 else None,
        )
