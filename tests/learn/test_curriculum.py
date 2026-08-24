"""Tests for the five-phase curriculum table and the runner that reads it.

Every prior arm ran phase 1 with phase 2's teacher cost, and nothing caught
it, because the numbers lived in a command line rather than in a table
anything could check. These tests are that check: the table itself, pinned
against the literal recipe, and the runner's wiring into ``toad``'s
own flags -- proven against a real checkpoint and a real ``Policy``, not a
namespace read back at itself.
"""

from pathlib import Path

import pytest
import torch

from kaggriculture.learn.model import Policy
from kaggriculture.learn.scripts import curriculum, toad
from kaggriculture.learn.toad.config import (
    CurriculumConfig,
    ModelConfig,
    OptimizerConfig,
    PopulationConfig,
    RuntimeConfig,
    ToadConfig,
)

# Transcribed independently of curriculum.PHASES, from the literal recipe
# table -- so this file fails if the module's own transcription drifts, not
# just if it disagrees with itself.
_TABLE: dict[str, dict[str, object]] = {
    "phase1": {
        "blocks": 8,
        "steps": int(2e7),
        "reward": "shaped_money",
        "teacher_kl_cost": 0.0,
        "lr": 1e-4,
        "entropy_cost": 1e-3,
        "lmb": 0.8,
        "teacher_from": None,
    },
    "phase2": {
        "blocks": 8,
        "steps": int(1e7),
        "reward": "sparse",
        "teacher_kl_cost": 0.005,
        "lr": 1e-4,
        "entropy_cost": 1e-3,
        "lmb": 0.8,
        "teacher_from": "phase1",
    },
    "phase3": {
        "blocks": 16,
        "steps": int(2e7),
        "reward": "sparse",
        "teacher_kl_cost": 0.01,
        "lr": 5e-5,
        "entropy_cost": 2e-4,
        "lmb": 0.8,
        "teacher_from": "phase1",
    },
    "phase4": {
        "blocks": 16,
        "steps": int(2e7),
        "reward": "sparse",
        "teacher_kl_cost": 0.001,
        "lr": 5e-5,
        "entropy_cost": 2e-4,
        "lmb": 0.8,
        "teacher_from": "phase1",
    },
    "phase5": {
        "blocks": 24,
        "steps": int(2e7),
        "reward": "sparse",
        "teacher_kl_cost": 0.005,
        "lr": 5e-5,
        "entropy_cost": 2e-4,
        "lmb": 0.9,
        "teacher_from": "phase3",
    },
}


def test_the_phase_table_matches_the_recipe() -> None:
    """The five phases are transcribed from Toad's config YAMLs.

    This test is the transcription's proof. Every prior arm ran phase 1 with
    phase 2's teacher cost, and nothing caught it, because the numbers lived
    in a command line rather than in a table anything could check.
    """
    assert [phase.name for phase in curriculum.PHASES] == list(_TABLE)
    for phase in curriculum.PHASES:
        expected = _TABLE[phase.name]
        assert phase.blocks == expected["blocks"], phase.name
        assert phase.steps == expected["steps"], phase.name
        assert phase.reward == expected["reward"], phase.name
        assert phase.teacher_kl_cost == pytest.approx(expected["teacher_kl_cost"]), (
            phase.name
        )
        assert phase.lr == pytest.approx(expected["lr"]), phase.name
        assert phase.entropy_cost == pytest.approx(expected["entropy_cost"]), phase.name
        assert phase.lmb == pytest.approx(expected["lmb"]), phase.name
        assert phase.teacher_from == expected["teacher_from"], phase.name


def test_phase_one_has_no_teacher() -> None:
    """The single most consequential number in this plan."""
    assert curriculum.PHASES[0].teacher_kl_cost == 0.0
    assert curriculum.PHASES[0].teacher_from is None


def test_every_curriculum_phase_is_a_valid_toad_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each table row must become the complete typed experiment contract."""
    monkeypatch.setattr(toad, "RUNS", tmp_path)
    phase1 = tmp_path / "phase1_000025.pt"
    phase3 = tmp_path / "phase3_000025.pt"
    phase1.touch()
    phase3.touch()

    for phase in curriculum.PHASES:
        config = curriculum.phase_config(phase)
        assert isinstance(config, ToadConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.optimizer, OptimizerConfig)
        assert isinstance(config.population, PopulationConfig)
        assert isinstance(config.runtime, RuntimeConfig)
        assert isinstance(config.curriculum, CurriculumConfig)
        assert config.curriculum.phase == phase.name
        assert config.curriculum.reward_field == phase.reward
        assert config.model.blocks == phase.blocks
        assert config.model.channels == toad.CHANNELS
        assert config.optimizer.lr == pytest.approx(phase.lr)
        assert config.optimizer.lmb == pytest.approx(phase.lmb)
        assert config.optimizer.entropy_cost == pytest.approx(phase.entropy_cost)
        assert config.optimizer.teacher_kl_cost == pytest.approx(phase.teacher_kl_cost)
        assert config.runtime.total_environment_steps == phase.steps
        expected_teacher = (
            None
            if phase.teacher_from is None
            else curriculum._checkpoint(phase.teacher_from)
        )
        assert (
            None
            if config.population.teacher is None
            else config.population.teacher.checkpoint
        ) == expected_teacher
        expected_teacher_blocks = (
            None if phase.teacher_from is None else curriculum._teacher_blocks(phase)
        )
        assert (
            None
            if config.population.teacher is None
            else config.population.teacher.blocks
        ) == expected_teacher_blocks


def test_curriculum_runtime_resume_override_reaches_native_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume stays explicit and never becomes an implicit phase-4 transition."""
    resume = tmp_path / "phase3.ckpt"
    resume.touch()
    seen: list[ToadConfig] = []
    monkeypatch.setattr(toad, "run", seen.append)

    curriculum.main(["phase1", "--set", f'runtime.resume="{resume}"'])

    assert len(seen) == 1
    assert seen[0].runtime.resume == resume


def test_a_phase_refuses_to_start_without_its_teacher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing teacher checkpoint must fail loudly, not train unanchored."""
    monkeypatch.setattr(toad, "RUNS", tmp_path)

    with pytest.raises(FileNotFoundError) as excinfo:
        curriculum._flags(curriculum._phase("phase2"))

    # Names the phase it wanted a teacher from and the path it looked for.
    assert "phase1" in str(excinfo.value)
    assert str(tmp_path) in str(excinfo.value)


def test_teachers_are_named_by_size_not_the_immediately_preceding_phase() -> None:
    """Phases 3 and 4 are taught by the 8-block net, phase 5 by the 16-block one.

    Not by whichever phase runs immediately before them: phase 4's own
    predecessor is phase 3, at 16 blocks, but its teacher is phase 1's
    8-block checkpoint, unchanged from phase 3's.
    """
    assert curriculum._teacher_blocks(curriculum._phase("phase3")) == 8
    assert curriculum._teacher_blocks(curriculum._phase("phase4")) == 8
    assert curriculum._teacher_blocks(curriculum._phase("phase5")) == 16


def test_checkpoint_resolves_the_latest_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The newest checkpoint must win, not the first glob match or an alphabetic one."""
    monkeypatch.setattr(toad, "RUNS", tmp_path)
    for update in (25, 100, 50, 75):
        (tmp_path / f"phase1_{update:06d}.pt").touch()

    assert curriculum._checkpoint("phase1") == tmp_path / "phase1_000100.pt"


def test_lightning_checkpoints_are_numeric_and_phase_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Native phase outputs cannot collide and step 10 must beat step 9."""
    monkeypatch.setattr(curriculum, "OUTPUT_ROOT", tmp_path)
    phase1_dir = tmp_path / "phase1"
    phase1_dir.mkdir()
    (phase1_dir / "step-9.ckpt").touch()
    (phase1_dir / "step-10.ckpt").touch()

    assert curriculum._checkpoint("phase1") == phase1_dir / "step-10.ckpt"
    assert curriculum.phase_config(curriculum._phase("phase1")).runtime.output_dir == (
        tmp_path / "phase1"
    )
    assert curriculum.phase_config(curriculum._phase("phase3")).runtime.output_dir == (
        tmp_path / "phase3"
    )


def test_phase_one_lightning_output_becomes_phase_two_teacher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The native checkpoint writer's envelope is readable by the next phase."""
    monkeypatch.setattr(curriculum, "OUTPUT_ROOT", tmp_path)
    phase1_dir = tmp_path / "phase1"
    phase1_dir.mkdir()
    source = Policy(blocks=8, channels=toad.CHANNELS, value_bound=toad.VALUE_BOUND)
    checkpoint = phase1_dir / "step-200.ckpt"
    torch.save(
        {
            "state_dict": {
                f"policy.{name}": value for name, value in source.state_dict().items()
            }
        },
        checkpoint,
    )

    config = curriculum.phase_config(curriculum._phase("phase2"))
    module = toad.ToadLightningModule(config)

    assert config.population.teacher is not None
    assert config.population.teacher.checkpoint == checkpoint
    assert module.teacher_policy is not None
    assert torch.equal(
        module.teacher_policy.state_dict()["stem.weight"],
        source.state_dict()["stem.weight"],
    )


def test_phase_one_has_no_sparse_flag() -> None:
    """Phase 1 trains on the dense shaped reward, not the terminal sparse one.

    Dense here is ``shaped_money``, not ``shaped``: in Lux their ``city``
    weight *is* the score, so the faithful component set carries the score
    constituent and ``shaped`` is the ablation that deletes it. See
    ``toad.REWARD_FIELD``.
    """
    flags = curriculum._flags(curriculum._phase("phase1"))
    assert "--sparse" not in flags
    assert "--no-money" not in flags
    arguments = toad._parser().parse_args(flags)
    assert toad._field(arguments) == "shaped_money"


def test_every_phase_reward_survives_toad_s_own_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The table's ``reward`` column must be what the learner actually reads.

    One phase selects its reward by naming no flag at all, which means the
    column and ``toad.REWARD_FIELD``'s default agree only by coincidence until
    something checks. That default has already moved once, silently retargeting
    phase 1, so every row is put through the real parser rather than the one
    row a spot check would cover.
    """
    monkeypatch.setattr(toad, "RUNS", tmp_path)
    for phase in curriculum.PHASES:
        (tmp_path / f"{phase.name}_000025.pt").touch()

    for phase in curriculum.PHASES:
        arguments = toad._parser().parse_args(curriculum._flags(phase))
        assert toad._field(arguments) == phase.reward, phase.name


def test_phase_two_reaches_sparse_through_the_real_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--sparse must actually reach _field, not just appear in the flag list."""
    monkeypatch.setattr(toad, "RUNS", tmp_path)
    (tmp_path / "phase1_000025.pt").touch()

    flags = curriculum._flags(curriculum._phase("phase2"))
    arguments = toad._parser().parse_args(flags)
    assert toad._field(arguments) == "sparse"


def test_flags_reach_a_real_teacher_through_toad_s_own_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--teacher and --teacher-blocks must resolve to a checkpoint toad can load.

    Proven against a real ``Policy`` and a real state dict, not a namespace
    read back at itself: a runner that pointed ``--teacher`` at the right
    path but the wrong ``--teacher-blocks`` would fail here with a shape
    mismatch, exactly as it would training for real.
    """
    monkeypatch.setattr(toad, "RUNS", tmp_path)
    torch.save(
        Policy(blocks=8, channels=16).state_dict(), tmp_path / "phase1_000025.pt"
    )

    flags = curriculum._flags(curriculum._phase("phase2"))
    arguments = toad._parser().parse_args([*flags, "--channels", "16"])
    teacher = toad._teacher(arguments, "cpu")

    assert teacher is not None
    assert len(teacher.policy.blocks) == 8


def test_phase_five_s_teacher_is_the_sixteen_block_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one phase whose teacher is not the 8-block net -- proven, not just tabled."""
    monkeypatch.setattr(toad, "RUNS", tmp_path)
    torch.save(
        Policy(blocks=16, channels=16).state_dict(), tmp_path / "phase3_000025.pt"
    )

    flags = curriculum._flags(curriculum._phase("phase5"))
    arguments = toad._parser().parse_args([*flags, "--channels", "16"])
    teacher = toad._teacher(arguments, "cpu")

    assert teacher is not None
    assert len(teacher.policy.blocks) == 16


def test_the_cli_names_every_phase() -> None:
    """``curriculum <phase-name>`` must accept exactly the five table entries."""
    parser = curriculum._parser()
    for phase in curriculum.PHASES:
        assert parser.parse_args([phase.name]).phase == phase.name
    with pytest.raises(SystemExit):
        parser.parse_args(["phase0"])


def test_unknown_phase_name_raises() -> None:
    """``_phase`` must name the phase it did not recognise, not raise blindly."""
    with pytest.raises(ValueError, match="phase0"):
        curriculum._phase("phase0")
