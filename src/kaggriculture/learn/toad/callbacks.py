"""Collection-round callbacks for the native Toad trainer."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import lightning
from lightning.pytorch.utilities.types import STEP_OUTPUT

from kaggriculture.learn.toad.data import LearnerBatch, ToadDataModule
from kaggriculture.learn.toad.lightning import ToadLightningModule


def _data_module(trainer: lightning.Trainer) -> ToadDataModule:
    """Return the Toad data module owned by ``trainer``."""
    data_module = getattr(trainer, "datamodule", None)
    if data_module is None:
        raise RuntimeError("Toad callbacks require a ToadDataModule")
    return cast(ToadDataModule, data_module)


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
        _data_module(trainer).publish_actor(
            module.policy.state_dict(), module.actor_version
        )

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
            module.policy.state_dict(), module.actor_version
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

        self.output_dir.mkdir(parents=True, exist_ok=True)
        final_path = self.output_dir / f"step-{module.environment_steps}.ckpt"
        temporary_path = final_path.with_suffix(".tmp")
        trainer.save_checkpoint(str(temporary_path))
        os.replace(temporary_path, final_path)  # noqa: PTH105
        self._next_environment_steps = (
            module.environment_steps // interval + 1
        ) * interval

    def _interval(self, module: ToadLightningModule) -> int:
        """Resolve the explicit test cadence or effective runtime cadence."""
        if self.every_environment_steps is not None:
            return self.every_environment_steps
        return module.config.runtime.checkpoint_every_environment_steps
