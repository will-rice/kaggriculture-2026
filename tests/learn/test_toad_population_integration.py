"""End-to-end contracts for checkpointed Toad population training."""

from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import lightning
import pytest
import torch

from kaggriculture.learn.model import Policy
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


def _metric_float(value: object) -> float:
    """Narrow one logged scalar without weakening the record type."""
    if isinstance(value, torch.Tensor):
        return float(value.item())
    assert isinstance(value, (int, float))
    return float(value)


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
    trainer = cast(lightning.Trainer, SimpleNamespace(datamodule=data))
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
    assert len(module.population_manifest.entries) == 2

    resumed_module = ToadLightningModule(config)
    resumed_module.on_load_checkpoint(checkpoint)
    resumed_source = ReferenceRoundSource(
        config,
        collect_assignment=lambda _assignment: (_trajectory(32),),
    )
    resumed_data = ToadDataModule(config, resumed_source)
    resumed_data.load_state_dict(data_state)
    resumed_callback = PopulationSnapshotCallback()
    resumed_callback.on_fit_start(
        cast(lightning.Trainer, SimpleNamespace(datamodule=resumed_data)),
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
