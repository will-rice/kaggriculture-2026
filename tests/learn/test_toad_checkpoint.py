"""Round-trip test for phase-1 checkpointing.

Attempt one at this run carried no checkpointing and lost 2.3 hours of training
to an external kill. The property that matters is not merely that weights
reload, but that the learning-rate schedule continues from where it stopped: a
schedule silently restarted from step zero restores the initial rate and changes
the recipe for the remainder of the run without anything looking wrong.
"""

import itertools
import pathlib
import random
from typing import cast

import lightning
import pytest
import torch

from kaggriculture.learn.encoding import SCALARS, TILE_PLANES
from kaggriculture.learn.model import Policy
from kaggriculture.learn.rollout import Trajectory
from kaggriculture.learn.scripts import toad
from kaggriculture.learn.toad.callbacks import ActorSyncCallback, BoundaryCheckpoint
from kaggriculture.learn.toad.config import ToadConfig, structural_fingerprint
from kaggriculture.learn.toad.data import (
    CollectionAssignment,
    ReferenceRoundSource,
    ToadDataModule,
)
from kaggriculture.learn.toad.lightning import (
    ResumeConfigError,
    ToadLightningModule,
)
from kaggriculture.learn.toad_loss import (
    ADAM_EPS,
    LEARNING_RATE,
    MIN_LR_MOD,
    TOTAL_STEPS,
)
from tests.learn.test_toad_control_fixture import (
    _assert_state_equal,
    control_fixture_config,
    control_fixture_threads,
    load_control_fixture,
)

# The arm's own mix, so the schedule under test is the one that runs.
ECON_FRACTION = 0.5


def _fresh() -> tuple[
    Policy, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler
]:
    """Return a policy, optimizer and schedule as `main` builds them."""
    policy = Policy(blocks=1, channels=16, value_bound=toad.VALUE_BOUND)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, toad._decay(ECON_FRACTION))
    return policy, optimizer, schedule


@pytest.mark.parametrize(("econ_fraction", "seats"), [(0.0, 48), (0.5, 36), (0.25, 42)])
def test_the_schedule_reaches_its_floor_no_earlier_than_the_budget(
    econ_fraction: float, seats: int
) -> None:
    """The decay must span the budget, not floor a quarter of the way short.

    Their linear schedule reaches ``min_lr_mod`` at 99% of ``total_steps``. An
    arm that mixes opponents records fewer seats per round than the environment
    count suggests -- 36 rather than 48 at ``--econ-fraction 0.5`` -- so a
    schedule keyed on the naive count floors at 74% of the budget and the last
    quarter of the run trains at 1e-6. The seat counts are literals and the run
    is played forward a round at a time, so this fails on the real quantity
    rather than agreeing with the helper it checks.
    """
    decay = toad._decay(econ_fraction)
    steps, update = 0, 0
    while steps < TOTAL_STEPS:
        steps += seats * toad.TURNS
        update += 1
    floor = next(step for step in itertools.count() if decay(step) == MIN_LR_MOD)

    assert decay(0) == 1.0
    assert floor / update > 0.98


def test_resuming_continues_the_schedule_rather_than_restarting_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The LR after a resume must equal the uninterrupted run's LR.

    Broken by restarting the schedule, which is the natural way to get this
    wrong and is invisible except in this number.
    """
    monkeypatch.setattr(toad, "RUNS", tmp_path)

    # An uninterrupted run: 40 schedule steps.
    _, _, uninterrupted = _fresh()
    for _ in range(40):
        uninterrupted.step()
    expected = uninterrupted.get_last_lr()[0]

    # A run that stops at 25 and resumes.
    policy, optimizer, schedule = _fresh()
    for _ in range(25):
        schedule.step()
    path = toad._checkpoint(
        policy, optimizer, schedule, steps=863_000, update=25, prefix="phase1"
    )
    assert path.is_file()

    restored_policy, restored_optimizer, restored_schedule = _fresh()
    steps, update = toad._restore(
        path, restored_policy, restored_optimizer, restored_schedule, "cpu"
    )
    assert (steps, update) == (863_000, 25)
    for _ in range(15):
        restored_schedule.step()

    assert restored_schedule.get_last_lr()[0] == expected


def test_the_checkpoint_carries_weights_and_adam_moments(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Weights and optimizer state must both survive, not just weights.

    Adam's moments are as much of the training state as the parameters; dropping
    them restarts the optimizer cold and loses the run's accumulated scaling.
    """
    monkeypatch.setattr(toad, "RUNS", tmp_path)
    policy, optimizer, schedule = _fresh()
    # Take a real step so the moments are populated.
    policy(
        torch.randn(2, TILE_PLANES, 10, 10),
        torch.randn(2, SCALARS),
        torch.zeros(2, 20, dtype=torch.int64),
    )[2].sum().backward()
    optimizer.step()

    path = toad._checkpoint(
        policy, optimizer, schedule, steps=1, update=25, prefix="phase1"
    )
    restored_policy, restored_optimizer, restored_schedule = _fresh()
    toad._restore(path, restored_policy, restored_optimizer, restored_schedule, "cpu")

    for before, after in zip(
        policy.state_dict().values(), restored_policy.state_dict().values(), strict=True
    ):
        assert torch.equal(before, after)
    assert restored_optimizer.state_dict()["state"], "Adam moments were not restored"


def test_the_checkpoint_write_is_atomic(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No temporary file may survive, or a kill mid-write leaves a torn checkpoint."""
    monkeypatch.setattr(toad, "RUNS", tmp_path)
    policy, optimizer, schedule = _fresh()
    toad._checkpoint(policy, optimizer, schedule, steps=1, update=50, prefix="phase1b")
    assert list(tmp_path.glob("*.pt"))
    assert not list(tmp_path.glob("*.tmp"))


def test_native_checkpoint_extends_lightning_with_all_foundation_counters() -> None:
    """Dropping any Toad clock makes the next collection boundary diverge."""
    module = ToadLightningModule(ToadConfig.control())
    module.environment_steps = 321
    module.collection_round = 5
    module.actor_version = 3
    module.actor_source_global_step = 17
    module.warmup_remaining = 8
    checkpoint: dict[str, object] = {}

    module.on_save_checkpoint(checkpoint)

    state = cast(dict[str, object], checkpoint["toad"])
    assert state == {
        "config": module.config.model_dump(mode="json"),
        "fingerprint": structural_fingerprint(module.config),
        "environment_steps": 321,
        "collection_round": 5,
        "actor_version": 3,
        "actor_source_global_step": 17,
        "warmup_remaining": 8,
        "teacher": {"present": False, "blocks": None, "quantity": None},
    }


def test_native_resume_allows_operational_overrides_and_restores_counters(
    tmp_path: pathlib.Path,
) -> None:
    """Changing output plumbing cannot reset authoritative progress clocks."""
    stored = ToadLightningModule(ToadConfig.control())
    stored.environment_steps = 640
    stored.collection_round = 10
    stored.actor_version = 4
    stored.actor_source_global_step = 22
    stored.warmup_remaining = 7
    checkpoint: dict[str, object] = {}
    stored.on_save_checkpoint(checkpoint)
    effective_config = ToadConfig.control().model_copy(
        update={
            "runtime": ToadConfig.control().runtime.model_copy(
                update={"output_dir": tmp_path, "log_every_n_steps": 3}
            )
        }
    )
    restored = ToadLightningModule(effective_config)

    restored.on_load_checkpoint(checkpoint)

    assert restored.config.runtime.output_dir == tmp_path
    assert restored.environment_steps == 640
    assert restored.collection_round == 10
    assert restored.actor_version == 4
    assert restored.actor_source_global_step == 22
    assert restored.warmup_remaining == 7


def test_native_resume_reports_exact_structural_paths_and_values() -> None:
    """A shape mismatch must identify both stored and effective declarations."""
    stored_config = ToadConfig.control().model_copy(
        update={"model": ToadConfig.control().model.model_copy(update={"channels": 16})}
    )
    stored = ToadLightningModule(stored_config)
    checkpoint: dict[str, object] = {}
    stored.on_save_checkpoint(checkpoint)
    effective_config = stored_config.model_copy(
        update={
            "optimizer": stored_config.optimizer.model_copy(
                update={"batch_segments": 8}
            )
        }
    )
    restored = ToadLightningModule(effective_config)

    with pytest.raises(ResumeConfigError) as caught:
        restored.on_load_checkpoint(checkpoint)

    assert str(caught.value) == (
        "structural config mismatch: optimizer.batch_segments: stored=4, effective=8"
    )


def test_data_module_checkpoint_restores_the_collector_stream() -> None:
    """The next game, random draw, and published actor version must all resume."""
    config = ToadConfig.control()
    source = ReferenceRoundSource(config, assignments=())
    data = ToadDataModule(config, source)
    source.next_game_id = 41
    source.actor_version = 6
    source.rng.random()
    state = data.state_dict()
    expected_next_random = source.rng.random()
    expected = random.Random()
    expected.setstate(cast(tuple[object, ...], state["collector_rng"]))
    assert expected.random() == expected_next_random

    resumed_source = ReferenceRoundSource(config, assignments=())
    resumed = ToadDataModule(config, resumed_source)
    resumed.load_state_dict(state)

    assert resumed_source.next_game_id == 41
    assert resumed_source.actor_version == 6
    assert resumed_source.rng.random() == expected_next_random


def test_data_module_resume_reproduces_the_next_collection_boundary() -> None:
    """Resume must not repeat either a game ID or its logical round ID."""
    config = ToadConfig.control().model_copy(
        update={
            "population": ToadConfig.control().population.model_copy(
                update={"environments_per_rank": 1}
            )
        }
    )
    trajectory = cast(Trajectory, load_control_fixture()["trajectory"])
    source = ReferenceRoundSource(config, collect_assignment=lambda _: (trajectory,))
    data = ToadDataModule(config, source)
    data.publish_actor({"weight": torch.ones(1)}, version=3)
    first = next(iter(source))
    state = data.state_dict()
    uninterrupted_next = next(iter(source))

    resumed_source = ReferenceRoundSource(
        config, collect_assignment=lambda _: (trajectory,)
    )
    resumed = ToadDataModule(config, resumed_source)
    resumed.load_state_dict(state)
    resumed.publish_actor({"weight": torch.ones(1)}, version=3)
    resumed_next = next(iter(resumed_source))

    assert (first.round_id, first.game_ids) == (0, (0,))
    assert (
        resumed_next.round_id,
        resumed_next.game_ids,
        resumed_next.actor_version,
    ) == (
        uninterrupted_next.round_id,
        uninterrupted_next.game_ids,
        uninterrupted_next.actor_version,
    )


def _native_resume_config(output_dir: pathlib.Path) -> ToadConfig:
    """Return the one-game, one-batch config used by resume integration."""
    config = control_fixture_config()
    return config.model_copy(
        update={
            "population": config.population.model_copy(
                update={
                    "actor_sync_every_rounds": 100,
                    "environments_per_rank": 1,
                }
            ),
            "optimizer": config.optimizer.model_copy(update={"value_passes": 0}),
            "runtime": config.runtime.model_copy(
                update={
                    "checkpoint_every_environment_steps": 64,
                    "output_dir": output_dir,
                }
            ),
        }
    )


def _native_trainer(
    root: pathlib.Path,
    *,
    max_steps: int,
    callbacks: list[lightning.Callback],
) -> lightning.Trainer:
    """Build the production automatic-optimization boundary without logging."""
    return lightning.Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        max_steps=max_steps,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        default_root_dir=root,
        gradient_clip_val=control_fixture_config().optimizer.clip_grad_norm,
        gradient_clip_algorithm="norm",
        callbacks=callbacks,
        use_distributed_sampler=False,
    )


def test_lightning_resume_matches_uninterrupted_full_state(
    tmp_path: pathlib.Path,
) -> None:
    """Lightning and Toad state must resume into the identical second round."""
    fixture = load_control_fixture()
    trajectory = cast(Trajectory, fixture["trajectory"])
    initial_model = cast(dict[str, torch.Tensor], fixture["initial_model"])
    config = _native_resume_config(tmp_path / "checkpoints")

    uninterrupted_source = ReferenceRoundSource(
        config, collect_assignment=lambda _: (trajectory,)
    )
    uninterrupted_data = ToadDataModule(config, uninterrupted_source)
    uninterrupted_module = ToadLightningModule(config)
    uninterrupted_module.policy.load_state_dict(initial_model)
    uninterrupted_trainer = _native_trainer(
        tmp_path / "uninterrupted",
        max_steps=2,
        callbacks=[ActorSyncCallback(every_rounds=100)],
    )

    split_source = ReferenceRoundSource(
        config, collect_assignment=lambda _: (trajectory,)
    )
    split_data = ToadDataModule(config, split_source)
    split_module = ToadLightningModule(config)
    split_module.policy.load_state_dict(initial_model)
    split_trainer = _native_trainer(
        tmp_path / "split",
        max_steps=1,
        callbacks=[
            ActorSyncCallback(every_rounds=100),
            BoundaryCheckpoint(config.runtime.output_dir),
        ],
    )

    with control_fixture_threads():
        uninterrupted_trainer.fit(uninterrupted_module, datamodule=uninterrupted_data)
        split_trainer.fit(split_module, datamodule=split_data)

    checkpoint_path = config.runtime.output_dir / "step-64.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["global_step"] == 1
    assert checkpoint["optimizer_states"]
    assert checkpoint["lr_schedulers"]

    collected: list[tuple[int, int, bool]] = []
    resumed_source: ReferenceRoundSource

    def collect_resumed(assignment: CollectionAssignment) -> tuple[Trajectory]:
        actor_matches = all(
            torch.equal(resumed_source.actor_state[name], value)
            for name, value in initial_model.items()
        )
        collected.append(
            (assignment.game_id, resumed_source.actor_version, actor_matches)
        )
        return (trajectory,)

    resumed_source = ReferenceRoundSource(config, collect_assignment=collect_resumed)
    resumed_data = ToadDataModule(config, resumed_source)
    resumed_module = ToadLightningModule(config)
    resumed_trainer = _native_trainer(
        tmp_path / "resumed",
        max_steps=2,
        callbacks=[ActorSyncCallback(every_rounds=100)],
    )

    with control_fixture_threads():
        resumed_trainer.fit(
            resumed_module,
            datamodule=resumed_data,
            ckpt_path=checkpoint_path,
        )

    assert collected == [(1, 0, True)]
    assert resumed_trainer.global_step == uninterrupted_trainer.global_step == 2
    assert resumed_module.environment_steps == 128
    assert resumed_module.collection_round == 2
    assert resumed_source.next_game_id == 2
    for name, value in resumed_module.policy.state_dict().items():
        assert torch.equal(value, uninterrupted_module.policy.state_dict()[name])
    _assert_state_equal(
        resumed_trainer.optimizers[0].state_dict(),
        uninterrupted_trainer.optimizers[0].state_dict(),
    )
    resumed_scheduler = resumed_trainer.lr_scheduler_configs[0].scheduler
    uninterrupted_scheduler = uninterrupted_trainer.lr_scheduler_configs[0].scheduler
    _assert_state_equal(
        resumed_scheduler.state_dict(), uninterrupted_scheduler.state_dict()
    )
    assert resumed_scheduler.get_last_lr() == uninterrupted_scheduler.get_last_lr()
