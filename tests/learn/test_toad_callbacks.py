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
)
from kaggriculture.learn.toad.config import ToadConfig


class _DataModule:
    """Record actor publications without replacing callback behavior."""

    def __init__(self) -> None:
        self.published: list[tuple[dict[str, torch.Tensor], int]] = []

    def publish_actor(self, state_dict: dict[str, torch.Tensor], version: int) -> None:
        self.published.append((dict(state_dict), version))


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
