"""End-to-end contracts for checkpointed Toad population training."""

from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import lightning
import pytest
import torch

import kaggriculture.learn.rollout as rollout_module
from kaggriculture.learn.model import Policy
from kaggriculture.learn.scripts import toad
from kaggriculture.learn.toad.callbacks import PopulationSnapshotCallback
from kaggriculture.learn.toad.config import ToadConfig, structural_fingerprint
from kaggriculture.learn.toad.data import (
    BatchKind,
    ReferenceRoundSource,
    ToadDataModule,
    allocate_round,
)
from kaggriculture.learn.toad.lightning import ToadLightningModule
from kaggriculture.learn.toad.population import (
    SnapshotIntegrityError,
    SnapshotManifest,
    SnapshotPool,
    SnapshotStore,
)


def _population_config(tmp_path: Path) -> ToadConfig:
    base = ToadConfig.control()
    return base.model_copy(
        update={"runtime": base.runtime.model_copy(update={"output_dir": tmp_path})}
    )


def _callback_trainer(data: ToadDataModule) -> lightning.Trainer:
    """Expose the checkpoint seam required by durable manifest publication."""

    def save_checkpoint(path: str) -> None:
        Path(path).write_bytes(b"authoritative checkpoint")

    return cast(
        lightning.Trainer,
        SimpleNamespace(datamodule=data, save_checkpoint=save_checkpoint),
    )


def _metric_float(value: object) -> float:
    """Narrow one logged scalar without weakening the record type."""
    if isinstance(value, torch.Tensor):
        return float(value.item())
    assert isinstance(value, (int, float))
    return float(value)


def test_all_population_kinds_cross_the_real_worker_rollout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each population kind reconstructs and runs through the production worker."""
    monkeypatch.setattr(rollout_module, "EPISODE_STEPS", 3)
    teacher_path = tmp_path / "teacher.pt"
    teacher_policy = Policy(blocks=1, channels=4, value_bound=1.0)
    torch.save(teacher_policy.state_dict(), teacher_path)
    config = ToadConfig.model_validate(
        {
            "model": {"blocks": 1, "channels": 4, "value_bound": 1.0},
            "population": {
                "selfplay": 0.25,
                "scripted": 0.25,
                "frozen_opponent": 0.25,
                "teacher_distill": 0.25,
                "teacher": {
                    "checkpoint": teacher_path,
                    "blocks": 1,
                    "operation": True,
                    "quantity": True,
                    "market": True,
                },
                "environments_per_rank": 4,
                "collection_processes": 1,
                "snapshot_at_start": True,
            },
            "optimizer": {"unroll_length": 2, "batch_segments": 5},
            "runtime": {"output_dir": tmp_path},
        }
    )
    actor = Policy(blocks=1, channels=4, value_bound=1.0)
    store = SnapshotStore(
        tmp_path / "population",
        capacity=config.population.pool_capacity,
        structure=structural_fingerprint(config),
    )
    frozen = store.add(
        actor.state_dict(), environment_steps=0, round_id=0, run_id="worker-smoke"
    )
    pool = SnapshotPool.from_store(store, seed=config.population.population_seed)
    source = ReferenceRoundSource(config, assignments=(), pool=pool)
    source.publish_actor(actor.state_dict(), version=0)
    assignments = allocate_round(config, 0, 0, pool)
    opponents = source._materialize_opponents(assignments)

    results = {
        assignment.kind: toad._play_reference(
            source._worker_input(assignment, opponents)
        )
        for assignment in assignments
    }

    assert set(results) == {
        BatchKind.SELFPLAY,
        BatchKind.SCRIPTED,
        BatchKind.FROZEN_OPPONENT,
        BatchKind.TEACHER_DISTILL,
    }
    assert len(results[BatchKind.SELFPLAY]) == 2
    assert all(
        len(trajectories) == 1
        for kind, trajectories in results.items()
        if kind is not BatchKind.SELFPLAY
    )
    assert all(
        len(trajectory.dones) == 2
        for trajectories in results.values()
        for trajectory in trajectories
    )
    neural = {
        assignment.kind: assignment.checkpoint_sha256
        for assignment in assignments
        if assignment.checkpoint_sha256 is not None
    }
    assert neural[BatchKind.FROZEN_OPPONENT] == frozen.sha256
    assert len(neural[BatchKind.TEACHER_DISTILL] or "") == 64


def test_publish_manifest_binds_only_the_verified_durable_store(
    tmp_path: Path,
) -> None:
    """Collection must see the exact durable view published by the callback."""
    config = _population_config(tmp_path)
    source = ReferenceRoundSource(config, assignments=())
    data = ToadDataModule(config, source)
    store = SnapshotStore(
        tmp_path / "population",
        capacity=config.population.pool_capacity,
        structure=structural_fingerprint(config),
    )
    store.add(
        {"weight": torch.tensor([1.0])},
        environment_steps=100,
        round_id=1,
        run_id="test",
    )

    data.publish_manifest(store.manifest)

    assert source.pool is not None
    assert source.pool.manifest == store.manifest
    assert source.pool.sample(7) == store.manifest.entries[0]
    original_pool = source.pool
    with pytest.raises(SnapshotIntegrityError, match="published population manifest"):
        data.publish_manifest(SnapshotManifest())
    assert source.pool is original_pool


def test_corrupt_manifest_publication_is_failure_atomic(tmp_path: Path) -> None:
    """Digest failure cannot mutate either the collector or checkpoint manifest."""
    config = _population_config(tmp_path)
    module = ToadLightningModule(config)
    store = SnapshotStore(
        tmp_path / "population",
        capacity=config.population.pool_capacity,
        structure=structural_fingerprint(config),
    )
    store.add(
        module.policy.state_dict(),
        environment_steps=0,
        round_id=0,
        run_id="valid",
    )
    entry = store.add(
        module.policy.state_dict(),
        environment_steps=1,
        round_id=0,
        run_id="corrupt",
    )
    entry.path.write_bytes(b"corrupt")
    source = ReferenceRoundSource(config, assignments=())
    data = ToadDataModule(config, source)
    callback = PopulationSnapshotCallback()

    with pytest.raises(SnapshotIntegrityError, match="digest"):
        callback.on_fit_start(_callback_trainer(data), module)

    assert source.pool is None
    assert module.population_manifest == SnapshotManifest()


def test_population_resume_keeps_checkpointed_pool_after_output_override(
    tmp_path: Path,
) -> None:
    """An operational output change cannot silently migrate population identity."""
    original_root = tmp_path / "original"
    resumed_root = tmp_path / "resumed"
    base = _population_config(original_root)
    config = base.model_copy(
        update={
            "population": base.population.model_copy(
                update={
                    "snapshot_at_start": True,
                    "snapshot_every_environment_steps": 100,
                }
            )
        }
    )
    module = ToadLightningModule(config)
    source = ReferenceRoundSource(config, assignments=())
    data = ToadDataModule(config, source)
    callback = PopulationSnapshotCallback()
    callback.on_fit_start(_callback_trainer(data), module)
    checkpoint: dict[str, object] = {}
    module.on_save_checkpoint(checkpoint)
    data_state = data.state_dict()

    resumed_config = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(update={"output_dir": resumed_root})
        }
    )
    resumed_module = ToadLightningModule(resumed_config)
    resumed_module.on_load_checkpoint(checkpoint)
    resumed_source = ReferenceRoundSource(resumed_config, assignments=())
    resumed_data = ToadDataModule(resumed_config, resumed_source)
    resumed_data.load_state_dict(data_state)
    resumed_callback = PopulationSnapshotCallback()
    resumed_trainer = _callback_trainer(resumed_data)

    resumed_callback.on_fit_start(resumed_trainer, resumed_module)
    resumed_module.environment_steps = 100
    resumed_module.collection_round = 1
    resumed_callback.on_train_batch_end(
        resumed_trainer,
        resumed_module,
        None,
        SimpleNamespace(end_of_round=True),
        0,
    )

    assert resumed_callback.store is not None
    assert resumed_callback.store.directory == (original_root / "population").resolve()
    assert len(resumed_module.population_manifest.entries) == 2
    assert not (resumed_root / "population").exists()


def test_failed_checkpoint_publication_preserves_prior_authoritative_generation(
    tmp_path: Path,
) -> None:
    """A manifest advance cannot invalidate the last completed checkpoint."""
    base = _population_config(tmp_path)
    config = base.model_copy(
        update={
            "population": base.population.model_copy(
                update={
                    "pool_capacity": 1,
                    "snapshot_at_start": True,
                    "snapshot_every_environment_steps": 1,
                }
            )
        }
    )
    module = ToadLightningModule(config)
    source = ReferenceRoundSource(config, assignments=())
    data = ToadDataModule(config, source)
    callback = PopulationSnapshotCallback()
    callback.on_fit_start(_callback_trainer(data), module)
    prior_manifest = module.population_manifest
    prior_member = prior_manifest.entries[0]
    checkpoint: dict[str, object] = {}
    module.on_save_checkpoint(checkpoint)
    data_state = data.state_dict()
    # Simulate the pre-generation layout of an authoritative checkpoint that
    # predates this protocol.  The next manifest mutation must retain it first.
    for generation in (tmp_path / "population").glob("manifest-*.json"):
        generation.unlink()

    def fail_checkpoint(_path: str) -> None:
        raise OSError("simulated checkpoint interruption")

    failing_trainer = cast(
        lightning.Trainer,
        SimpleNamespace(datamodule=data, save_checkpoint=fail_checkpoint),
    )
    module.environment_steps = 1
    module.collection_round = 1
    with pytest.raises(OSError, match="simulated checkpoint interruption"):
        callback.on_train_batch_end(
            failing_trainer,
            module,
            None,
            SimpleNamespace(end_of_round=True),
            0,
        )

    assert module.population_manifest != prior_manifest
    assert prior_member.path.is_file()

    resumed_module = ToadLightningModule(config)
    resumed_module.on_load_checkpoint(checkpoint)
    resumed_source = ReferenceRoundSource(config, assignments=())
    resumed_data = ToadDataModule(config, resumed_source)
    resumed_data.load_state_dict(data_state)
    resumed_callback = PopulationSnapshotCallback()
    resumed_callback.on_fit_start(_callback_trainer(resumed_data), resumed_module)

    assert resumed_module.population_manifest == prior_manifest
    assert resumed_source.pool is not None
    assert resumed_source.pool.manifest == prior_manifest
    assert resumed_source.pool.load(prior_member)


def test_round_metrics_flush_once_with_stable_population_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial batches cannot emit or reset the completed-round dashboard row."""
    from tests.learn.test_toad_control_fixture import (
        control_fixture_batch,
        control_fixture_config,
        load_control_fixture,
    )

    module = ToadLightningModule(control_fixture_config())
    records: list[tuple[dict[str, object], dict[str, object]]] = []
    monkeypatch.setattr(
        module,
        "log_dict",
        lambda record, **kwargs: records.append((dict(record), kwargs)),
    )
    batch = control_fixture_batch(load_control_fixture())
    first = replace(
        batch,
        end_of_round=False,
        collected_steps=32,
        round_metrics={"throughput/collection_seconds": 2.0},
    )
    last = replace(batch, first_of_round=False, collected_steps=0)

    module.training_step(first, 0)
    assert records == []
    module.training_step(last, 1)

    assert len(records) == 1
    record, kwargs = records[0]
    assert kwargs == {"on_step": True, "on_epoch": False, "sync_dist": False}
    assert record["throughput/collection_seconds"] == 2.0
    assert record["throughput/collection_steps_per_second"] == 16.0
    assert record["progress/environment_steps"] == 32.0
    assert "throughput/learner_steps_per_second" in record
    assert "belief/loss" in record
    assert "mask/operation_density" in record
    assert "population/selections/selfplay" in record


def test_population_game_counts_are_unique_and_source_metrics_cannot_overwrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two self-play seats from one game remain one population selection."""
    from tests.learn.test_toad_control_fixture import (
        control_fixture_batch,
        control_fixture_config,
        load_control_fixture,
    )

    module = ToadLightningModule(control_fixture_config())
    records: list[dict[str, object]] = []
    monkeypatch.setattr(
        module, "log_dict", lambda record, **_: records.append(dict(record))
    )
    batch = control_fixture_batch(load_control_fixture())
    duplicated_seats = replace(
        batch,
        game_ids=(7,),
        opponent_ids=("self",),
        opponent_digests=(None,),
        segment_game_ids=(7,) * len(batch.segments),
        segment_kinds=(BatchKind.SELFPLAY,) * len(batch.segments),
        segment_opponent_ids=("self",) * len(batch.segments),
        segment_opponent_digests=(None,) * len(batch.segments),
        round_metrics={"collection/games/selfplay": 2},
    )

    module.training_step(duplicated_seats, 0)

    assert records[0]["collection/games/selfplay"] == 1
    assert records[0]["population/selections/selfplay"] == 1


def test_mixed_batch_loss_is_not_duplicated_under_each_population(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One aggregate mixed loss has one honest mixed denominator and label."""
    from tests.learn.test_toad_control_fixture import (
        control_fixture_batch,
        control_fixture_config,
        load_control_fixture,
    )

    module = ToadLightningModule(control_fixture_config())
    records: list[dict[str, object]] = []
    monkeypatch.setattr(
        module, "log_dict", lambda record, **_: records.append(dict(record))
    )
    batch = control_fixture_batch(load_control_fixture())
    count = len(batch.segments)
    kinds = tuple(
        BatchKind.SELFPLAY if index % 2 == 0 else BatchKind.SCRIPTED
        for index in range(count)
    )
    mixed = replace(
        batch,
        kind=BatchKind.MIXED,
        segment_kinds=kinds,
        segment_game_ids=tuple(range(count)),
        segment_opponent_ids=tuple(
            "self" if kind is BatchKind.SELFPLAY else "economic" for kind in kinds
        ),
        segment_opponent_digests=(None,) * count,
    )

    module.training_step(mixed, 0)

    record = records[0]
    assert "loss/by_kind/mixed" in record
    assert "loss/by_opponent/mixed" in record
    assert "loss/by_opponent/self" not in record
    assert "loss/by_opponent/economic" not in record


def test_learner_timing_finishes_after_optimizer_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The timing endpoint must follow closure backward/clipping/optimizer work."""
    from tests.learn.test_toad_control_fixture import (
        control_fixture_batch,
        control_fixture_config,
        load_control_fixture,
    )

    module = ToadLightningModule(control_fixture_config())
    batch = control_fixture_batch(load_control_fixture())
    records: list[dict[str, object]] = []
    events: list[str] = []

    class _Optimizer:
        param_groups = [{"lr": module.config.optimizer.lr}]

        def step(self, closure: Callable[[], object]) -> None:
            closure()
            assert records == []
            events.append("optimizer")

    optimizer = _Optimizer()
    trainer = cast(
        lightning.Trainer,
        SimpleNamespace(global_step=0, optimizers=[optimizer]),
    )
    module.trainer = trainer
    monkeypatch.setattr(
        module, "log_dict", lambda record, **_: records.append(dict(record))
    )
    monkeypatch.setattr(
        "kaggriculture.learn.toad.lightning.time.perf_counter",
        lambda: 15.0 if events else 10.0,
    )

    module.optimizer_step(
        0,
        0,
        cast(Any, optimizer),
        lambda: module.training_step(batch, 0),
    )

    assert events == ["optimizer"]
    assert len(records) == 1
    assert records[0]["throughput/learner_seconds"] == 5.0


def test_four_kind_round_trains_and_reports_actual_mix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two real assignments per online kind retain identity through learning."""
    from tests.learn.test_toad_data import _trajectory

    teacher_path = tmp_path / "teacher.pt"
    teacher_policy = Policy(blocks=1, channels=4, value_bound=1.0)
    torch.save(teacher_policy.state_dict(), teacher_path)
    config = ToadConfig.model_validate(
        {
            "model": {"blocks": 1, "channels": 4, "value_bound": 1.0},
            "population": {
                "selfplay": 0.25,
                "scripted": 0.25,
                "frozen_opponent": 0.25,
                "teacher_distill": 0.25,
                "teacher": {
                    "checkpoint": teacher_path,
                    "blocks": 1,
                    "operation": True,
                    "quantity": True,
                    "market": True,
                },
                "environments_per_rank": 8,
                "collection_processes": 1,
                "population_seed": 17,
                "snapshot_every_environment_steps": 256,
                "snapshot_at_start": True,
            },
            "optimizer": {
                "adaptive_entropy": True,
                "batch_segments": 1,
                "unroll_length": 16,
                "teacher_kl_cost": 0.01,
                "value_warmup_batches": 0,
            },
            "runtime": {"output_dir": tmp_path, "total_environment_steps": 512},
        }
    )
    module = ToadLightningModule(config)
    store = SnapshotStore(
        tmp_path / "population",
        capacity=config.population.pool_capacity,
        structure=structural_fingerprint(config),
    )
    frozen = store.add(
        module.policy.state_dict(),
        environment_steps=0,
        round_id=0,
        run_id="integration",
    )
    pool = SnapshotPool.from_store(store, seed=config.population.population_seed)
    source = ReferenceRoundSource(
        config,
        collect_assignment=lambda _assignment: (_trajectory(32),),
        pool=pool,
    )
    source.publish_actor(module.policy.state_dict(), version=0)
    data = ToadDataModule(config, source)
    trainer = _callback_trainer(data)
    snapshot_callback = PopulationSnapshotCallback()
    snapshot_callback.on_fit_start(trainer, module)

    batches = list(source)
    records: list[dict[str, object]] = []
    monkeypatch.setattr(
        module, "log_dict", lambda record, **_: records.append(dict(record))
    )
    losses = [module.training_step(batch, index) for index, batch in enumerate(batches)]

    game_kinds = {
        game_id: kind
        for batch in batches
        if not batch.baseline_only
        for game_id, kind in zip(
            batch.segment_game_ids, batch.segment_kinds, strict=True
        )
    }
    assert Counter(game_kinds.values()) == {
        kind: 2 for kind in BatchKind if kind is not BatchKind.MIXED
    }
    game_opponents = {
        game_id: (opponent_id, digest)
        for batch in batches
        if not batch.baseline_only
        for game_id, opponent_id, digest in zip(
            batch.segment_game_ids,
            batch.segment_opponent_ids,
            batch.segment_opponent_digests,
            strict=True,
        )
    }
    assert all(opponent_id for opponent_id, _digest in game_opponents.values())
    neural_digests = [
        digest for _opponent_id, digest in game_opponents.values() if digest is not None
    ]
    assert len(neural_digests) == 4
    assert frozen.sha256 in neural_digests
    assert all(len(digest) == 64 for digest in neural_digests)
    assert all(bool(torch.isfinite(loss)) for loss in losses)
    assert len(records) == 1
    record = records[0]
    assert record["population/selections/selfplay"] == 2
    assert record["population/selections/scripted"] == 2
    assert record["population/selections/frozen_opponent"] == 2
    assert record["population/selections/teacher_distill"] == 2
    assert _metric_float(record["entropy/quantity_valid"]) <= _metric_float(
        record["entropy/operation_valid"]
    )
    assert _metric_float(record["throughput/collection_steps_per_second"]) > 0
    assert _metric_float(record["throughput/collection_wait_seconds"]) > 0
    assert all(
        f"loss/by_kind/{kind.value}" in record
        for kind in BatchKind
        if kind is not BatchKind.MIXED
    )
    assert "state/hidden_norm" in record
    assert "state/cell_norm" in record
    assert "state/terminal_resets" in record
    assert any(name.startswith("collection/games_by_opponent/") for name in record)
    assert "belief/loss" in record
    assert record["progress/environment_steps"] == 256.0

    # Manual ``training_step`` calls do not run an optimizer. Represent the
    # parameter update that precedes a real boundary snapshot so the next
    # population round contains two distinct policies across three provenance
    # entries (the external seed and actor-at-start intentionally share weights).
    with torch.no_grad():
        next(module.policy.parameters()).add_(1e-4)
    snapshot_callback.on_train_batch_end(
        trainer,
        module,
        None,
        batches[-1],
        len(batches) - 1,
    )
    checkpoint: dict[str, object] = {}
    module.on_save_checkpoint(checkpoint)
    data_state = data.state_dict()
    assert len(module.population_manifest.entries) == 3
    assert len({entry.sha256 for entry in module.population_manifest.entries}) == 2

    resumed_module = ToadLightningModule(config)
    # Lightning restores module weights before invoking ``on_load_checkpoint``.
    resumed_module.load_state_dict(module.state_dict())
    resumed_module.on_load_checkpoint(checkpoint)
    resumed_source = ReferenceRoundSource(
        config,
        collect_assignment=lambda _assignment: (_trajectory(32),),
    )
    resumed_data = ToadDataModule(config, resumed_source)
    resumed_data.load_state_dict(data_state)
    resumed_callback = PopulationSnapshotCallback()
    resumed_callback.on_fit_start(
        _callback_trainer(resumed_data),
        resumed_module,
    )

    assert resumed_module.entropy_state == module.entropy_state
    assert resumed_module.population_manifest == module.population_manifest
    assert resumed_source.pool is not None
    assert source.pool is not None
    assert allocate_round(
        config,
        source.next_game_id,
        source._next_round_id,
        source.pool,
    ) == allocate_round(
        config,
        resumed_source.next_game_id,
        resumed_source._next_round_id,
        resumed_source.pool,
    )

    uninterrupted_next = list(source)
    resumed_next = list(resumed_source)
    assert [
        (
            batch.segment_game_ids,
            batch.segment_kinds,
            batch.segment_opponent_ids,
            batch.segment_opponent_digests,
        )
        for batch in resumed_next
    ] == [
        (
            batch.segment_game_ids,
            batch.segment_kinds,
            batch.segment_opponent_ids,
            batch.segment_opponent_digests,
        )
        for batch in uninterrupted_next
    ]
    monkeypatch.setattr(resumed_module, "log_dict", lambda _record, **_: None)
    for index, (uninterrupted_batch, resumed_batch) in enumerate(
        zip(uninterrupted_next, resumed_next, strict=True)
    ):
        uninterrupted_loss = module.training_step(uninterrupted_batch, index)
        resumed_loss = resumed_module.training_step(resumed_batch, index)
        torch.testing.assert_close(resumed_loss, uninterrupted_loss)

    assert resumed_module.entropy_state == module.entropy_state
    assert all(
        state.last_steps == 512 for state in resumed_module.entropy_state.values()
    )
