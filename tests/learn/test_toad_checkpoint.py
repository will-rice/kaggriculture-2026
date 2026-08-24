"""Round-trip test for phase-1 checkpointing.

Attempt one at this run carried no checkpointing and lost 2.3 hours of training
to an external kill. The property that matters is not merely that weights
reload, but that the learning-rate schedule continues from where it stopped: a
schedule silently restarted from step zero restores the initial rate and changes
the recipe for the remainder of the run without anything looking wrong.
"""

import hashlib
import itertools
import pathlib
import random
from typing import cast

import lightning
import pytest
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT

from kaggriculture.learn.encoding import SCALARS, TILE_PLANES
from kaggriculture.learn.model import Policy
from kaggriculture.learn.rollout import Trajectory
from kaggriculture.learn.scripts import toad
from kaggriculture.learn.toad.callbacks import ActorSyncCallback, BoundaryCheckpoint
from kaggriculture.learn.toad.config import ToadConfig, structural_fingerprint
from kaggriculture.learn.toad.data import (
    CollectionAssignment,
    LearnerBatch,
    ReferenceRoundSource,
    ToadDataModule,
)
from kaggriculture.learn.toad.lightning import (
    ResumeConfigError,
    ToadLightningModule,
    load_checkpoint_policy,
)
from kaggriculture.learn.toad.model import StatefulPolicy
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
from tests.learn.test_toad_model_integration import (
    _all_feature_config,
    _initial_policy_state,
    _synthetic_trajectory,
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
        "teacher": {"present": False},
    }


def test_full_enabled_lightning_policy_extraction_is_structurally_strict(
    tmp_path: pathlib.Path,
) -> None:
    """Extraction must load every optional weight and reject partial/control sets."""
    config = _all_feature_config()
    expected = _initial_policy_state(config)
    exact_path = tmp_path / "all-feature.ckpt"
    torch.save(
        {"state_dict": {f"policy.{name}": value for name, value in expected.items()}},
        exact_path,
    )
    restored = StatefulPolicy(config.model)

    missing_path = tmp_path / "missing-enabled.ckpt"
    missing = dict(expected)
    del missing["interaction_value.value_projection.weight"]
    torch.save(
        {"state_dict": {f"policy.{name}": value for name, value in missing.items()}},
        missing_path,
    )
    control_path = tmp_path / "control-only.ckpt"
    control = Policy(blocks=1, channels=4, value_bound=1.0)
    torch.save(
        {
            "state_dict": {
                f"policy.{name}": value for name, value in control.state_dict().items()
            }
        },
        control_path,
    )

    assert load_checkpoint_policy(restored, exact_path) == []
    for name, value in expected.items():
        assert torch.equal(restored.state_dict()[name], value), name
    with pytest.raises(
        RuntimeError, match="interaction_value[.]value_projection[.]weight"
    ):
        load_checkpoint_policy(StatefulPolicy(config.model), missing_path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_checkpoint_policy(StatefulPolicy(config.model), control_path)


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


def test_recurrent_boundary_resume_reproduces_the_next_optimizer_update(  # noqa: C901
    tmp_path: pathlib.Path,
) -> None:
    """The authoritative all-feature boundary must replay the next update exactly."""
    teacher = Policy(blocks=1, channels=4, value_bound=1.0)
    teacher_path = tmp_path / "teacher.pt"
    torch.save(teacher.state_dict(), teacher_path)
    config = _all_feature_config(
        output_dir=tmp_path / "checkpoints",
        teacher_checkpoint=teacher_path,
        value_warmup_batches=8,
    )
    initial_model = _initial_policy_state(config)
    published_actor_version = 6
    trajectories = {
        0: _synthetic_trajectory(config, initial_model, game_id=0).trajectory
    }

    def copy_policy(module: ToadLightningModule) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().to("cpu", copy=True)
            for name, value in module.policy.state_dict().items()
        }

    def state_matches(
        actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
    ) -> bool:
        return set(actual) == set(expected) and all(
            torch.equal(actual[name], value) for name, value in expected.items()
        )

    def make_source(
        records: list[tuple[int, int, float, bool]],
        expected_actor_state: dict[str, torch.Tensor],
    ) -> ReferenceRoundSource:
        source: ReferenceRoundSource

        def collect(assignment: CollectionAssignment) -> tuple[Trajectory]:
            records.append(
                (
                    assignment.game_id,
                    source.actor_version,
                    source.rng.random(),
                    state_matches(source.actor_state, expected_actor_state),
                )
            )
            return (trajectories[assignment.game_id],)

        source = ReferenceRoundSource(config, collect_assignment=collect)
        return source

    class SeedActorPublisher(lightning.Callback):
        """Publish a real post-update snapshot at the first completed boundary."""

        def __init__(self, data: ToadDataModule) -> None:
            self.data = data
            self.actor_state: dict[str, torch.Tensor] = {}
            self.source_global_step: int | None = None

        def on_train_batch_end(
            self,
            trainer: lightning.Trainer,
            pl_module: lightning.LightningModule,
            outputs: STEP_OUTPUT,
            batch: object,
            batch_idx: int,
        ) -> None:
            learner_batch = cast(LearnerBatch, batch)
            if not learner_batch.end_of_round:
                return
            module = cast(ToadLightningModule, pl_module)
            assert self.source_global_step is None
            assert module.actor_version == published_actor_version - 1
            module.actor_version += 1
            module.actor_source_global_step = int(module.global_step)
            self.actor_state = copy_policy(module)
            self.source_global_step = module.actor_source_global_step
            self.data.publish_actor(self.actor_state, module.actor_version)

    class BatchEntryRecorder(lightning.Callback):
        def __init__(self) -> None:
            self.entries: list[
                tuple[int, tuple[int, ...], torch.Tensor, torch.Tensor, torch.Tensor]
            ] = []

        def on_train_batch_start(
            self,
            trainer: lightning.Trainer,
            pl_module: lightning.LightningModule,
            batch: object,
            batch_idx: int,
        ) -> None:
            learner_batch = cast(LearnerBatch, batch)
            if not learner_batch.first_of_round:
                return
            segment = learner_batch.segments[0]
            self.entries.append(
                (
                    learner_batch.round_id,
                    learner_batch.game_ids,
                    segment["initial_hidden"].clone(),
                    segment["initial_cell"].clone(),
                    segment["initial_belief"].clone(),
                )
            )

    def trainer(
        root: pathlib.Path,
        *,
        max_steps: int,
        callbacks: list[lightning.Callback],
    ) -> lightning.Trainer:
        return lightning.Trainer(
            accelerator="cpu",
            devices=1,
            precision="32-true",
            max_steps=max_steps,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            default_root_dir=root,
            gradient_clip_val=config.optimizer.clip_grad_norm,
            gradient_clip_algorithm="norm",
            deterministic=True,
            callbacks=callbacks,
            use_distributed_sampler=False,
        )

    seed_records: list[tuple[int, int, float, bool]] = []
    seed_source = make_source(seed_records, initial_model)
    seed_data = ToadDataModule(config, seed_source)
    seed_module = ToadLightningModule(config)
    seed_module.policy.load_state_dict(initial_model, strict=True)
    seed_module.actor_version = published_actor_version - 1
    seed_publisher = SeedActorPublisher(seed_data)
    seed_trainer = trainer(
        tmp_path / "seed-run",
        max_steps=2,
        callbacks=[
            ActorSyncCallback(every_rounds=1),
            seed_publisher,
            BoundaryCheckpoint(tmp_path / "seed-checkpoint"),
        ],
    )
    seed_trainer.fit(seed_module, datamodule=seed_data)

    seed_checkpoint_path = tmp_path / "seed-checkpoint" / "step-32.ckpt"
    seed_checkpoint = torch.load(
        seed_checkpoint_path, map_location="cpu", weights_only=False
    )
    seed_checkpoint_toad = cast(dict[str, object], seed_checkpoint["toad"])
    seed_learner_state = cast(dict[str, torch.Tensor], seed_checkpoint["state_dict"])
    seed_data_state = cast(dict[str, object], seed_checkpoint["ToadDataModule"])
    seed_checkpoint_actor = cast(
        dict[str, torch.Tensor], seed_data_state["published_actor_state"]
    )
    actor_source_global_step = cast(
        int, seed_checkpoint_toad["actor_source_global_step"]
    )
    assert [record[:2] for record in seed_records] == [(0, 5)]
    assert seed_records[0][3]
    assert seed_trainer.global_step == actor_source_global_step == 2
    assert seed_checkpoint["global_step"] == actor_source_global_step
    assert seed_checkpoint["optimizer_states"]
    assert seed_checkpoint["lr_schedulers"]
    assert seed_checkpoint_toad["environment_steps"] == 32
    assert seed_checkpoint_toad["collection_round"] == 1
    assert seed_checkpoint_toad["actor_version"] == published_actor_version
    assert seed_checkpoint_toad["warmup_remaining"] == 6
    assert seed_data_state["next_game_id"] == 1
    assert seed_data_state["published_actor_version"] == published_actor_version
    assert seed_publisher.source_global_step == actor_source_global_step
    assert state_matches(seed_publisher.actor_state, copy_policy(seed_module))
    assert state_matches(seed_source.actor_state, seed_publisher.actor_state)
    assert state_matches(seed_checkpoint_actor, seed_publisher.actor_state)
    assert all(
        torch.equal(seed_learner_state[f"policy.{name}"], value)
        for name, value in seed_publisher.actor_state.items()
    )
    assert any(
        not torch.equal(initial_model[name], value)
        for name, value in seed_publisher.actor_state.items()
    )

    trajectories.update(
        {
            game_id: _synthetic_trajectory(
                config,
                seed_publisher.actor_state,
                game_id=game_id,
            ).trajectory
            for game_id in (1, 2)
        }
    )

    uninterrupted_records: list[tuple[int, int, float, bool]] = []
    uninterrupted_source = make_source(
        uninterrupted_records, seed_publisher.actor_state
    )
    uninterrupted_data = ToadDataModule(config, uninterrupted_source)
    uninterrupted_module = ToadLightningModule(config)
    uninterrupted_entries = BatchEntryRecorder()
    uninterrupted_trainer = trainer(
        tmp_path / "uninterrupted",
        max_steps=5,
        callbacks=[
            uninterrupted_entries,
            ActorSyncCallback(every_rounds=1),
        ],
    )

    split_records: list[tuple[int, int, float, bool]] = []
    split_source = make_source(split_records, seed_publisher.actor_state)
    split_data = ToadDataModule(config, split_source)
    split_module = ToadLightningModule(config)
    split_trainer = trainer(
        tmp_path / "split",
        max_steps=4,
        callbacks=[
            ActorSyncCallback(every_rounds=1),
            BoundaryCheckpoint(config.runtime.output_dir),
        ],
    )

    uninterrupted_trainer.fit(
        uninterrupted_module,
        datamodule=uninterrupted_data,
        ckpt_path=seed_checkpoint_path,
    )
    split_trainer.fit(
        split_module,
        datamodule=split_data,
        ckpt_path=seed_checkpoint_path,
    )

    assert split_source.actor_version == published_actor_version
    assert state_matches(split_source.actor_state, seed_publisher.actor_state)
    assert any(
        not torch.equal(split_module.policy.state_dict()[name], value)
        for name, value in split_source.actor_state.items()
    )

    checkpoint_path = config.runtime.output_dir / "step-64.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_toad = cast(dict[str, object], checkpoint["toad"])
    checkpoint_data = cast(dict[str, object], checkpoint["ToadDataModule"])
    checkpoint_actor = cast(
        dict[str, torch.Tensor], checkpoint_data["published_actor_state"]
    )
    assert checkpoint["global_step"] == 4
    assert checkpoint["optimizer_states"]
    assert checkpoint["lr_schedulers"]
    assert checkpoint_toad["config"] == split_module.config.model_dump(mode="json")
    assert checkpoint_toad["fingerprint"] == structural_fingerprint(config)
    assert checkpoint_toad["environment_steps"] == 64
    assert checkpoint_toad["collection_round"] == 2
    assert checkpoint_toad["actor_version"] == published_actor_version
    checkpoint_actor_source_step = cast(
        int, checkpoint_toad["actor_source_global_step"]
    )
    assert checkpoint_actor_source_step == actor_source_global_step
    assert 0 < checkpoint_actor_source_step <= checkpoint["global_step"]
    assert checkpoint["global_step"] - checkpoint_actor_source_step == 2
    assert checkpoint_toad["warmup_remaining"] == 4
    assert checkpoint_toad["teacher"] == {
        "present": True,
        "checkpoint": str(teacher_path),
        "sha256": hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
        "topology": {
            "blocks": 1,
            "channels": 4,
            "kernel_size": 3,
            "activation": "relu",
            "value_bound": 1.0,
        },
        "declared_heads": {
            "operation": True,
            "quantity": False,
            "market": True,
            "value": False,
        },
        "actual_heads": ["operation", "market"],
    }
    assert checkpoint_data["next_game_id"] == 2
    assert checkpoint_data["published_actor_version"] == published_actor_version
    assert state_matches(checkpoint_actor, seed_publisher.actor_state)

    class RestoredStateRecorder(lightning.Callback):
        def __init__(self) -> None:
            self.observations: list[tuple[int, int, int, int, bool, bool]] = []

        def on_train_batch_start(
            self,
            trainer: lightning.Trainer,
            pl_module: lightning.LightningModule,
            batch: object,
            batch_idx: int,
        ) -> None:
            if self.observations:
                return
            module = cast(ToadLightningModule, pl_module)
            data = cast(ToadDataModule, getattr(trainer, "datamodule", None))
            actor_state = data.source.actor_state
            self.observations.append(
                (
                    int(module.global_step),
                    module.actor_version,
                    module.actor_source_global_step,
                    module.warmup_remaining,
                    state_matches(actor_state, seed_publisher.actor_state),
                    state_matches(actor_state, copy_policy(module)),
                )
            )

    resumed_records: list[tuple[int, int, float, bool]] = []
    resumed_source = make_source(resumed_records, seed_publisher.actor_state)
    resumed_data = ToadDataModule(config, resumed_source)
    resumed_module = ToadLightningModule(config)
    resumed_entries = BatchEntryRecorder()
    restored_state = RestoredStateRecorder()
    resumed_trainer = trainer(
        tmp_path / "resumed",
        max_steps=5,
        callbacks=[
            restored_state,
            resumed_entries,
            ActorSyncCallback(every_rounds=1),
        ],
    )

    resumed_trainer.fit(
        resumed_module,
        datamodule=resumed_data,
        ckpt_path=checkpoint_path,
    )

    assert [record[0] for record in uninterrupted_records] == [1, 2]
    assert [record[0] for record in split_records] == [1]
    assert [record[0] for record in resumed_records] == [2]
    assert split_records[0][2:] == uninterrupted_records[0][2:]
    assert resumed_records[0][1:] == uninterrupted_records[1][1:]
    assert all(record[3] for record in uninterrupted_records + resumed_records)
    assert all(
        record[1] == published_actor_version
        for record in uninterrupted_records + split_records + resumed_records
    )
    assert restored_state.observations == [(4, 6, 2, 4, True, False)]
    uninterrupted_next = next(
        entry for entry in uninterrupted_entries.entries if entry[0] == 2
    )
    assert resumed_entries.entries[0][:2] == uninterrupted_next[:2] == (2, (2,))
    for resumed_state, uninterrupted_state in zip(
        resumed_entries.entries[0][2:], uninterrupted_next[2:], strict=True
    ):
        assert torch.equal(resumed_state, uninterrupted_state)
        assert not torch.count_nonzero(resumed_state)

    assert resumed_trainer.global_step == uninterrupted_trainer.global_step == 5
    assert resumed_module.environment_steps == uninterrupted_module.environment_steps
    assert resumed_module.collection_round == uninterrupted_module.collection_round
    assert (
        resumed_module.actor_version
        == uninterrupted_module.actor_version
        == published_actor_version
    )
    assert (
        resumed_module.actor_source_global_step
        == uninterrupted_module.actor_source_global_step
        == actor_source_global_step
    )
    assert resumed_trainer.global_step - resumed_module.actor_source_global_step == 3
    assert resumed_module.warmup_remaining == uninterrupted_module.warmup_remaining == 3
    assert resumed_source.next_game_id == uninterrupted_source.next_game_id == 3
    assert (
        resumed_source.actor_version
        == uninterrupted_source.actor_version
        == published_actor_version
    )
    assert state_matches(resumed_source.actor_state, seed_publisher.actor_state)
    assert state_matches(uninterrupted_source.actor_state, seed_publisher.actor_state)
    for name, value in uninterrupted_source.actor_state.items():
        assert torch.equal(resumed_source.actor_state[name], value), name
    for name, value in uninterrupted_module.policy.state_dict().items():
        assert torch.equal(resumed_module.policy.state_dict()[name], value), name
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
