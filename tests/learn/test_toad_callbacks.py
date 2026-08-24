"""Round-boundary callback tests for the native Toad trainer."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import lightning
import torch

from kaggriculture.learn.toad.callbacks import (
    ActorSyncCallback,
    BoundaryCheckpoint,
    EnvironmentStepStop,
    PopulationSnapshotCallback,
)
from kaggriculture.learn.toad.config import ToadConfig, structural_fingerprint
from kaggriculture.learn.toad.lightning import ToadLightningModule
from kaggriculture.learn.toad.population import SnapshotManifest, SnapshotStore


class _DataModule:
    """Record actor publications without replacing callback behavior."""

    def __init__(self, config: ToadConfig | None = None) -> None:
        self.config = config
        self.published: list[tuple[dict[str, torch.Tensor], int]] = []
        self.manifests: list[SnapshotManifest] = []

    def publish_actor(self, state_dict: dict[str, torch.Tensor], version: int) -> None:
        self.published.append((dict(state_dict), version))

    def publish_manifest(self, manifest: SnapshotManifest) -> None:
        self.manifests.append(manifest)

    def materialize_population_bootstrap(
        self,
        store: SnapshotStore,
        actor_state: dict[str, torch.Tensor],
        *,
        environment_steps: int,
        round_id: int,
        run_id: str,
    ) -> None:
        """Mirror rank-zero bootstrap writes for callback-only tests."""
        if self.config is None:
            return
        for path in self.config.population.initial_snapshots:
            store.add(
                torch.load(path, map_location="cpu", weights_only=True),
                environment_steps=environment_steps,
                round_id=round_id,
                run_id=f"initial:{path}",
            )
        if self.config.population.snapshot_at_start:
            store.add(
                actor_state,
                environment_steps=environment_steps,
                round_id=round_id,
                run_id=f"{run_id}:actor-at-start",
            )


class _Trainer:
    """Small Trainer boundary exposing only callback-owned effects."""

    def __init__(self, data: _DataModule) -> None:
        self.datamodule = data
        self.global_step = 99
        self.should_stop = False
        self.saved: list[Path] = []

    def save_checkpoint(self, path: str) -> None:
        destination = Path(path)
        destination.write_text("complete")
        self.saved.append(destination)


def _callback_fixture(
    *,
    environment_steps: int = 0,
    round_id: int = 0,
    actor_version: int = 0,
    round_started_warming: bool = False,
) -> tuple[_Trainer, SimpleNamespace, _DataModule]:
    data = _DataModule()
    trainer = _Trainer(data)
    module = SimpleNamespace(
        policy=torch.nn.Linear(1, 1),
        config=ToadConfig.control(),
        environment_steps=environment_steps,
        collection_round=round_id,
        actor_version=actor_version,
        actor_source_global_step=0,
        population_manifest=SnapshotManifest(),
        _authoritative_checkpoint_identity=None,
        global_step=trainer.global_step,
        _round_started_warming=round_started_warming,
    )
    return trainer, module, data


def _batch(*, end_of_round: bool) -> SimpleNamespace:
    return SimpleNamespace(end_of_round=end_of_round)


def _lightning_trainer(trainer: _Trainer) -> lightning.Trainer:
    return cast(lightning.Trainer, trainer)


def _lightning_module(module: SimpleNamespace) -> lightning.LightningModule:
    return cast(lightning.LightningModule, module)


def test_fit_start_publishes_the_restored_learner_once() -> None:
    """A resumed collector must not begin from the source's stale actor."""
    callback = ActorSyncCallback(every_rounds=2)
    trainer, module, data = _callback_fixture(actor_version=7)

    callback.on_fit_start(_lightning_trainer(trainer), _lightning_module(module))

    assert [version for _, version in data.published] == [7]
    published = data.published[0][0]
    assert all(
        torch.equal(published[name], value)
        for name, value in module.policy.state_dict().items()
    )


def test_actor_is_not_published_during_warmup() -> None:
    """A round that began in value warmup keeps its original actor pinned."""
    callback = ActorSyncCallback(every_rounds=2)
    trainer, module, data = _callback_fixture(round_id=2, round_started_warming=True)

    callback.on_train_batch_end(
        _lightning_trainer(trainer),
        _lightning_module(module),
        None,
        _batch(end_of_round=True),
        0,
    )

    assert data.published == []
    assert module.actor_version == 0


def test_actor_publication_uses_only_eligible_completed_rounds() -> None:
    """A partial batch or off-cadence round must not advance actor lag state."""
    callback = ActorSyncCallback(every_rounds=2)
    trainer, module, data = _callback_fixture(round_id=1, actor_version=4)

    callback.on_train_batch_end(
        _lightning_trainer(trainer),
        _lightning_module(module),
        None,
        _batch(end_of_round=False),
        0,
    )
    callback.on_train_batch_end(
        _lightning_trainer(trainer),
        _lightning_module(module),
        None,
        _batch(end_of_round=True),
        1,
    )
    module.collection_round = 2
    callback.on_train_batch_end(
        _lightning_trainer(trainer),
        _lightning_module(module),
        None,
        _batch(end_of_round=True),
        2,
    )

    assert [version for _, version in data.published] == [5]
    assert module.actor_source_global_step == 99


def test_environment_stop_uses_collected_steps_not_optimizer_steps() -> None:
    """Baseline replay optimizer steps cannot stop training between rounds."""
    callback = EnvironmentStepStop(total_environment_steps=128)
    trainer, module, _ = _callback_fixture(environment_steps=127)

    callback.on_train_batch_end(
        _lightning_trainer(trainer),
        _lightning_module(module),
        None,
        _batch(end_of_round=True),
        9,
    )
    assert not trainer.should_stop

    module.environment_steps = 128
    callback.on_train_batch_end(
        _lightning_trainer(trainer),
        _lightning_module(module),
        None,
        _batch(end_of_round=False),
        10,
    )
    assert not trainer.should_stop

    callback.on_train_batch_end(
        _lightning_trainer(trainer),
        _lightning_module(module),
        None,
        _batch(end_of_round=True),
        11,
    )
    assert trainer.should_stop


def test_boundary_checkpoint_is_atomic_and_due_only_after_a_round(
    tmp_path: Path,
) -> None:
    """A due checkpoint must be one durable file with no surviving sibling temp."""
    callback = BoundaryCheckpoint(tmp_path, every_environment_steps=100)
    trainer, module, _ = _callback_fixture(environment_steps=0)
    callback.on_fit_start(_lightning_trainer(trainer), _lightning_module(module))

    module.environment_steps = 100
    callback.on_train_batch_end(
        _lightning_trainer(trainer),
        _lightning_module(module),
        None,
        _batch(end_of_round=False),
        0,
    )
    assert trainer.saved == []

    callback.on_train_batch_end(
        _lightning_trainer(trainer),
        _lightning_module(module),
        None,
        _batch(end_of_round=True),
        1,
    )

    checkpoints = list(tmp_path.glob("*.ckpt"))
    assert [path.name for path in checkpoints] == ["step-100.ckpt"]
    assert checkpoints[0].read_text() == "complete"
    assert not list(tmp_path.glob("*.tmp"))
    assert trainer.saved[0].parent == tmp_path
    assert trainer.saved[0] != checkpoints[0]

    callback.on_fit_end(_lightning_trainer(trainer), _lightning_module(module))

    assert len(trainer.saved) == 1
    assert callback.final_checkpoint == tmp_path / "step-100.ckpt"


def test_fit_end_saves_the_exact_terminal_boundary_below_cadence(
    tmp_path: Path,
) -> None:
    """A successful short phase must return current state, not an older file."""
    callback = BoundaryCheckpoint(tmp_path, every_environment_steps=1_000)
    trainer, module, _ = _callback_fixture(environment_steps=0)
    callback.on_fit_start(_lightning_trainer(trainer), _lightning_module(module))
    stale = tmp_path / "step-999.ckpt"
    stale.write_text("stale")
    module.environment_steps = 40

    callback.on_fit_end(_lightning_trainer(trainer), _lightning_module(module))

    assert (tmp_path / "step-40.ckpt").read_text() == "complete"
    assert callback.final_checkpoint == tmp_path / "step-40.ckpt"
    assert stale.read_text() == "stale"


def test_boundary_checkpoint_resume_waits_for_the_next_global_threshold(
    tmp_path: Path,
) -> None:
    """Restoring on a saved threshold must not immediately save it again."""
    callback = BoundaryCheckpoint(tmp_path, every_environment_steps=100)
    trainer, module, _ = _callback_fixture(environment_steps=100)
    callback.on_fit_start(_lightning_trainer(trainer), _lightning_module(module))

    module.environment_steps = 150
    callback.on_train_batch_end(
        _lightning_trainer(trainer),
        _lightning_module(module),
        None,
        _batch(end_of_round=True),
        0,
    )
    assert trainer.saved == []

    module.environment_steps = 205
    callback.on_train_batch_end(
        _lightning_trainer(trainer),
        _lightning_module(module),
        None,
        _batch(end_of_round=True),
        1,
    )
    assert (tmp_path / "step-205.ckpt").is_file()


def test_population_snapshot_is_boundary_only_and_resume_stable(
    tmp_path: Path,
) -> None:
    """A crossed cadence publishes once and resume schedules strictly ahead."""
    base = ToadConfig.control()
    config = base.model_copy(
        update={
            "population": base.population.model_copy(
                update={"snapshot_every_environment_steps": 100}
            ),
            "runtime": base.runtime.model_copy(update={"output_dir": tmp_path}),
        }
    )
    module = ToadLightningModule(config)
    data = _DataModule(config)
    trainer = _Trainer(data)
    callback = PopulationSnapshotCallback()
    callback.on_fit_start(
        _lightning_trainer(trainer), cast(lightning.LightningModule, module)
    )

    module.environment_steps = 100
    callback.on_train_batch_end(
        _lightning_trainer(trainer),
        cast(lightning.LightningModule, module),
        None,
        _batch(end_of_round=False),
        0,
    )
    assert not list((tmp_path / "population").glob("snapshot-*.pt"))

    callback.on_train_batch_end(
        _lightning_trainer(trainer),
        cast(lightning.LightningModule, module),
        None,
        _batch(end_of_round=True),
        1,
    )
    snapshots = list((tmp_path / "population").glob("snapshot-*.pt"))
    assert len(snapshots) == 1
    assert data.manifests[-1] == module.population_manifest

    checkpoint: dict[str, object] = {}
    module.on_save_checkpoint(checkpoint)
    stored = cast(dict[str, object], checkpoint["toad"])
    assert stored["population_manifest"] == module.population_manifest.model_dump(
        mode="json"
    )

    resumed = ToadLightningModule(config)
    resumed.on_load_checkpoint(checkpoint)
    resumed_data = _DataModule(config)
    resumed_trainer = _Trainer(resumed_data)
    resumed_callback = PopulationSnapshotCallback()
    resumed_callback.on_fit_start(
        _lightning_trainer(resumed_trainer),
        cast(lightning.LightningModule, resumed),
    )
    resumed.environment_steps = 205
    resumed_callback.on_train_batch_end(
        _lightning_trainer(resumed_trainer),
        cast(lightning.LightningModule, resumed),
        None,
        _batch(end_of_round=True),
        2,
    )

    assert len(list((tmp_path / "population").glob("snapshot-*.pt"))) == 2
    assert resumed.population_manifest.entries[-1].environment_steps == 205


def test_population_snapshot_at_start_is_published_before_collection(
    tmp_path: Path,
) -> None:
    """A configured bootstrap snapshot must be selectable in the first round."""
    base = ToadConfig.control()
    config = base.model_copy(
        update={
            "population": base.population.model_copy(
                update={"snapshot_at_start": True}
            ),
            "runtime": base.runtime.model_copy(update={"output_dir": tmp_path}),
        }
    )
    module = ToadLightningModule(config)
    data = _DataModule(config)
    trainer = _Trainer(data)

    PopulationSnapshotCallback().on_fit_start(
        _lightning_trainer(trainer), cast(lightning.LightningModule, module)
    )

    assert len(module.population_manifest.entries) == 1
    assert data.manifests == [module.population_manifest]


def test_collection_bootstrap_manifest_is_captured_at_first_boundary(
    tmp_path: Path,
) -> None:
    """Initial snapshots materialized by collection must enter the checkpoint."""
    base = ToadConfig.control()
    config = base.model_copy(
        update={"runtime": base.runtime.model_copy(update={"output_dir": tmp_path})}
    )
    module = ToadLightningModule(config)
    data = _DataModule(config)
    trainer = _Trainer(data)
    callback = PopulationSnapshotCallback()
    callback.on_fit_start(
        _lightning_trainer(trainer), cast(lightning.LightningModule, module)
    )
    store = SnapshotStore(
        tmp_path / "population",
        capacity=config.population.pool_capacity,
        structure=structural_fingerprint(config),
    )
    store.add(
        module.policy.state_dict(),
        environment_steps=0,
        round_id=0,
        run_id="initial",
    )

    callback.on_train_batch_end(
        _lightning_trainer(trainer),
        cast(lightning.LightningModule, module),
        None,
        _batch(end_of_round=True),
        0,
    )

    assert module.population_manifest == store.manifest
    assert data.manifests[-1] == store.manifest


def test_snapshot_only_cadence_writes_the_latest_authoritative_checkpoint(
    tmp_path: Path,
) -> None:
    """A crash between checkpoint cadences resumes the just-published manifest."""
    base = ToadConfig.control()
    config = base.model_copy(
        update={
            "population": base.population.model_copy(
                update={"snapshot_every_environment_steps": 100}
            ),
            "runtime": base.runtime.model_copy(
                update={
                    "output_dir": tmp_path,
                    "checkpoint_every_environment_steps": 1_000,
                }
            ),
        }
    )
    module = ToadLightningModule(config)
    data = _DataModule(config)

    class CheckpointingTrainer(_Trainer):
        def save_checkpoint(self, path: str) -> None:
            checkpoint: dict[str, object] = {}
            module.on_save_checkpoint(checkpoint)
            torch.save(checkpoint, path)
            self.saved.append(Path(path))

    trainer = CheckpointingTrainer(data)
    snapshot = PopulationSnapshotCallback()
    periodic = BoundaryCheckpoint(tmp_path)
    snapshot.on_fit_start(
        _lightning_trainer(trainer), cast(lightning.LightningModule, module)
    )
    periodic.on_fit_start(
        _lightning_trainer(trainer), cast(lightning.LightningModule, module)
    )
    module.environment_steps = 100
    module.collection_round = 1
    boundary = _batch(end_of_round=True)

    snapshot.on_train_batch_end(
        _lightning_trainer(trainer),
        cast(lightning.LightningModule, module),
        None,
        boundary,
        0,
    )
    periodic.on_train_batch_end(
        _lightning_trainer(trainer),
        cast(lightning.LightningModule, module),
        None,
        boundary,
        0,
    )

    checkpoint_path = tmp_path / "step-100.ckpt"
    assert checkpoint_path.is_file()
    assert len(trainer.saved) == 1
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    resumed = ToadLightningModule(config)
    resumed.on_load_checkpoint(checkpoint)
    assert resumed.environment_steps == 100
    assert resumed.population_manifest == module.population_manifest

    resumed_data = _DataModule(config)
    resumed_trainer = _Trainer(resumed_data)
    PopulationSnapshotCallback().on_fit_start(
        _lightning_trainer(resumed_trainer),
        cast(lightning.LightningModule, resumed),
    )
    assert resumed_data.manifests[-1] == resumed.population_manifest
