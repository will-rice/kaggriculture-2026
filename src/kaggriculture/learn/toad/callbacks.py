"""Collection-round callbacks for the native Toad trainer."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import lightning
from lightning.pytorch.utilities.types import STEP_OUTPUT

from kaggriculture.learn.toad.compile import policy_state_dict
from kaggriculture.learn.toad.config import structural_fingerprint
from kaggriculture.learn.toad.data import (
    LearnerBatch,
    ToadDataModule,
    _all_gather_objects,
)
from kaggriculture.learn.toad.lightning import ToadLightningModule
from kaggriculture.learn.toad.population import (
    SnapshotIntegrityError,
    SnapshotManifest,
    SnapshotStore,
)


class DistributedBoundaryError(RuntimeError):
    """A rank-zero durable action failed and every rank must stop together."""

    def __init__(self, message: str, details: object = None) -> None:
        self.distributed_details = details
        super().__init__(message)


def _save_authoritative_checkpoint(
    trainer: lightning.Trainer,
    module: ToadLightningModule,
    output_dir: Path,
) -> Path:
    """Atomically save the sole resumable state for this manifest generation."""
    identity = (
        module.environment_steps,
        module.population_manifest.model_dump_json(),
    )
    final_path = output_dir / f"step-{module.environment_steps}.ckpt"
    if (
        getattr(module, "_authoritative_checkpoint_identity", None) == identity
        and final_path.is_file()
    ):
        return final_path
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = final_path.with_suffix(".tmp")
    try:
        if int(getattr(trainer, "world_size", 1)) > 1:
            # ``Trainer.save_checkpoint`` ends with an unconditional strategy
            # barrier and therefore cannot live inside this rank-zero failure
            # envelope.  DDP's ordinary checkpoint dump is process-local; its
            # strategy writer already restricts filesystem I/O to global zero.
            checkpoint = trainer._checkpoint_connector.dump_checkpoint()
            trainer.strategy.save_checkpoint(checkpoint, temporary_path)
        else:
            trainer.save_checkpoint(str(temporary_path))
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    os.replace(temporary_path, final_path)  # noqa: PTH105
    module._authoritative_checkpoint_identity = identity
    return final_path


def _data_module(trainer: lightning.Trainer) -> ToadDataModule:
    """Return the Toad data module owned by ``trainer``."""
    data_module = getattr(trainer, "datamodule", None)
    if data_module is None:
        raise RuntimeError("Toad callbacks require a ToadDataModule")
    return cast(ToadDataModule, data_module)


def _boundary_payload(
    trainer: lightning.Trainer,
    module: ToadLightningModule,
    *,
    include_manifest: bool,
) -> dict[str, object]:
    """Serialize global clocks plus the source-owned next collection boundary."""
    source = _data_module(trainer).source
    payload: dict[str, object] = {
        "environment_steps": module.environment_steps,
        "collection_round": module.collection_round,
        "actor_version": module.actor_version,
        "actor_source_global_step": module.actor_source_global_step,
        "next_game_id": source.next_game_id,
        "next_round_id": source._next_round_id,
    }
    if include_manifest:
        payload["population_manifest"] = module.population_manifest.model_dump(
            mode="json"
        )
    return payload


def _apply_boundary_payload(
    trainer: lightning.Trainer,
    module: ToadLightningModule,
    payload: object,
    *,
    include_manifest: bool,
) -> None:
    """Install the rank-zero boundary before any rank starts collection again."""
    if not isinstance(payload, Mapping):
        raise DistributedBoundaryError("rank-zero boundary payload is malformed")
    payload_map = cast(Mapping[str, object], payload)

    def counter(name: str) -> int:
        value = payload_map.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise DistributedBoundaryError(
                f"rank-zero boundary counter {name!r} is malformed"
            )
        return value

    module.environment_steps = counter("environment_steps")
    module.collection_round = counter("collection_round")
    module.actor_version = counter("actor_version")
    module.actor_source_global_step = counter("actor_source_global_step")
    data = _data_module(trainer)
    data.source.next_game_id = counter("next_game_id")
    data.source._next_round_id = counter("next_round_id")
    if include_manifest:
        try:
            manifest = SnapshotManifest.model_validate(
                payload_map["population_manifest"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise DistributedBoundaryError(
                "rank-zero population manifest payload is malformed"
            ) from error
        data.publish_manifest(manifest)
        module.population_manifest = manifest


def _run_rank_zero_boundary(
    trainer: lightning.Trainer,
    module: ToadLightningModule,
    action: Callable[[], object],
    *,
    include_manifest: bool,
) -> None:
    """Run one durable action, propagate failure, then barrier and broadcast."""
    world_size = int(getattr(trainer, "world_size", 1))
    is_global_zero = bool(getattr(trainer, "is_global_zero", True))
    if world_size == 1:
        action()
        return
    status: dict[str, object] | None = None
    if is_global_zero:
        try:
            action()
        except Exception as error:  # rank peers need a serializable failure
            status = {
                "ok": False,
                "type": type(error).__name__,
                "message": str(error),
            }
        else:
            status = {"ok": True}
    status = cast(dict[str, object], trainer.strategy.broadcast(status, src=0))
    if not bool(status.get("ok")):
        raise DistributedBoundaryError(
            "rank-zero durable boundary failed: "
            f"{status.get('type')}: {status.get('message')}"
        )
    trainer.strategy.barrier()
    payload = (
        _boundary_payload(trainer, module, include_manifest=include_manifest)
        if is_global_zero
        else None
    )
    payload = trainer.strategy.broadcast(payload, src=0)
    install_error: Exception | None = None
    try:
        _apply_boundary_payload(
            trainer,
            module,
            payload,
            include_manifest=include_manifest,
        )
    except Exception as error:
        install_error = error
    install_status: dict[str, object] = {
        "ok": install_error is None,
        "rank": int(getattr(trainer, "global_rank", 0)),
        "type": None if install_error is None else type(install_error).__name__,
        "message": None if install_error is None else str(install_error),
    }
    gathered = _all_gather_objects(install_status, world_size)
    failures = tuple(
        cast(Mapping[str, object], item)
        for item in gathered
        if isinstance(item, Mapping) and not bool(item.get("ok"))
    )
    if failures:
        rendered = "; ".join(
            f"rank={failure.get('rank')} {failure.get('type')}: "
            f"{failure.get('message')}"
            for failure in failures
        )
        distributed = DistributedBoundaryError(
            "distributed boundary install failed: " + rendered,
            failures,
        )
        if install_error is not None:
            raise distributed from install_error
        raise distributed


class ActorSyncCallback(lightning.Callback):
    """Publish learner snapshots at eligible completed collection rounds."""

    def __init__(self, every_rounds: int) -> None:
        self.every_rounds = every_rounds

    def on_fit_start(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
    ) -> None:
        """Publish restored/current learner weights before collection begins."""
        module = cast(ToadLightningModule, pl_module)
        data = _data_module(trainer)
        consume_restored = getattr(data, "consume_restored_actor", None)
        if consume_restored is not None and consume_restored():
            return
        data.publish_actor(policy_state_dict(module), module.actor_version)

    def on_train_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        outputs: STEP_OUTPUT,
        batch: object,
        batch_idx: int,
    ) -> None:
        """Advance actor lag state only on configured non-warmup boundaries."""
        module = cast(ToadLightningModule, pl_module)
        learner_batch = cast(LearnerBatch, batch)
        if not learner_batch.end_of_round or module._round_started_warming:
            return
        if module.collection_round % self.every_rounds:
            return
        module.actor_version += 1
        module.actor_source_global_step = int(module.global_step)
        _data_module(trainer).publish_actor(
            policy_state_dict(module), module.actor_version
        )


class EnvironmentStepStop(lightning.Callback):
    """Stop only after a completed round reaches the collection budget."""

    def __init__(self, total_environment_steps: int) -> None:
        self.total_environment_steps = total_environment_steps

    def on_train_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        outputs: STEP_OUTPUT,
        batch: object,
        batch_idx: int,
    ) -> None:
        """Set Lightning's cooperative stop flag at an eligible boundary."""
        module = cast(ToadLightningModule, pl_module)
        learner_batch = cast(LearnerBatch, batch)
        if (
            learner_batch.end_of_round
            and module.environment_steps >= self.total_environment_steps
        ):
            trainer.should_stop = True


class PopulationSnapshotCallback(lightning.Callback):
    """Publish verified policy snapshots only at completed round boundaries."""

    def __init__(self) -> None:
        self.store: SnapshotStore | None = None
        self.next_snapshot_steps: int | None = None

    def on_fit_start(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
    ) -> None:
        """Open the durable pool and schedule strictly after restored progress."""
        module = cast(ToadLightningModule, pl_module)
        population = module.config.population

        def publish_start() -> None:
            previous_manifest = module.population_manifest
            self._reopen_store(trainer, module)
            assert self.store is not None
            if not previous_manifest.entries:
                _data_module(trainer).materialize_population_bootstrap(
                    self.store,
                    policy_state_dict(module),
                    environment_steps=module.environment_steps,
                    round_id=module.collection_round,
                    run_id=module.config.curriculum.phase,
                )
            self._publish(trainer, module)
            if module.population_manifest != previous_manifest:
                _save_authoritative_checkpoint(
                    trainer,
                    module,
                    module.config.runtime.output_dir,
                )

        _run_rank_zero_boundary(
            trainer,
            module,
            publish_start,
            include_manifest=True,
        )
        interval = population.snapshot_every_environment_steps
        self.next_snapshot_steps = (
            None
            if interval is None
            else (module.environment_steps // interval + 1) * interval
        )

    def on_train_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        outputs: STEP_OUTPUT,
        batch: object,
        batch_idx: int,
    ) -> None:
        """Write at most one snapshot when a completed round crosses cadence."""
        module = cast(ToadLightningModule, pl_module)
        learner_batch = cast(LearnerBatch, batch)
        if not learner_batch.end_of_round:
            return
        interval = module.config.population.snapshot_every_environment_steps

        def publish_boundary() -> None:
            previous_manifest = module.population_manifest
            self._reopen_store(trainer, module)
            if (
                interval is not None
                and self.next_snapshot_steps is not None
                and module.environment_steps >= self.next_snapshot_steps
            ):
                self._add_snapshot(module)
            self._publish(trainer, module)
            if module.population_manifest != previous_manifest:
                _save_authoritative_checkpoint(
                    trainer,
                    module,
                    module.config.runtime.output_dir,
                )

        _run_rank_zero_boundary(
            trainer,
            module,
            publish_boundary,
            include_manifest=True,
        )
        if (
            interval is not None
            and self.next_snapshot_steps is not None
            and module.environment_steps >= self.next_snapshot_steps
        ):
            self.next_snapshot_steps = (
                module.environment_steps // interval + 1
            ) * interval

    def _reopen_store(
        self, trainer: lightning.Trainer, module: ToadLightningModule
    ) -> None:
        """Refresh collection-created bootstrap entries without accepting drift."""
        data = _data_module(trainer)
        population_directory = getattr(data, "population_directory", None)
        directory = (
            population_directory()
            if population_directory is not None
            else (module.config.runtime.output_dir / "population").resolve()
        )
        open_population_store = getattr(data, "open_population_store", None)
        store = (
            open_population_store(module.population_manifest)
            if open_population_store is not None
            else SnapshotStore(
                directory,
                capacity=module.config.population.pool_capacity,
                structure=structural_fingerprint(module.config),
            )
        )
        if (
            module.population_manifest.entries
            and module.population_manifest != store.manifest
        ):
            raise SnapshotIntegrityError(
                "checkpoint population manifest does not match durable store"
            )
        self.store = store

    def _add_snapshot(self, module: ToadLightningModule) -> None:
        """Add the current policy to the opened verified store."""
        if self.store is None:
            raise RuntimeError("population snapshot store is not initialized")
        self.store.add(
            policy_state_dict(module),
            environment_steps=module.environment_steps,
            round_id=module.collection_round,
            run_id=module.config.curriculum.phase,
        )

    def _publish(self, trainer: lightning.Trainer, module: ToadLightningModule) -> None:
        """Publish the durable manifest before later boundary callbacks run."""
        if self.store is None:
            raise RuntimeError("population snapshot store is not initialized")
        data = _data_module(trainer)
        publish_manifest = getattr(data, "publish_manifest", None)
        if publish_manifest is None:
            raise RuntimeError("ToadDataModule does not support population manifests")
        publish_manifest(self.store.manifest)
        module.population_manifest = self.store.manifest


class BoundaryCheckpoint(lightning.Callback):
    """Atomically checkpoint the complete Lightning state on due boundaries."""

    def __init__(
        self,
        output_dir: Path,
        every_environment_steps: int | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.every_environment_steps = every_environment_steps
        self._next_environment_steps: int | None = None
        self.final_checkpoint: Path | None = None

    def on_fit_start(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
    ) -> None:
        """Schedule the first threshold strictly after restored progress."""
        module = cast(ToadLightningModule, pl_module)
        interval = self._interval(module)
        self._next_environment_steps = (
            module.environment_steps // interval + 1
        ) * interval

    def on_fit_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
    ) -> None:
        """Publish the exact terminal phase state, even below the cadence."""
        module = cast(ToadLightningModule, pl_module)
        final_path = self.output_dir / f"step-{module.environment_steps}.ckpt"
        _run_rank_zero_boundary(
            trainer,
            module,
            lambda: _save_authoritative_checkpoint(trainer, module, self.output_dir),
            include_manifest=False,
        )
        self.final_checkpoint = final_path

    def on_train_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        outputs: STEP_OUTPUT,
        batch: object,
        batch_idx: int,
    ) -> None:
        """Save one durable checkpoint when a global threshold is due."""
        module = cast(ToadLightningModule, pl_module)
        learner_batch = cast(LearnerBatch, batch)
        if not learner_batch.end_of_round:
            return
        interval = self._interval(module)
        if self._next_environment_steps is None:
            self._next_environment_steps = interval
        if module.environment_steps < self._next_environment_steps:
            return

        _run_rank_zero_boundary(
            trainer,
            module,
            lambda: _save_authoritative_checkpoint(trainer, module, self.output_dir),
            include_manifest=False,
        )
        self._next_environment_steps = (
            module.environment_steps // interval + 1
        ) * interval

    def _interval(self, module: ToadLightningModule) -> int:
        """Resolve the explicit test cadence or effective runtime cadence."""
        if self.every_environment_steps is not None:
            return self.every_environment_steps
        return module.config.runtime.checkpoint_every_environment_steps
