"""Tests for the serializable configuration of the native Toad trainer."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from kaggriculture.learn.scripts import toad
from kaggriculture.learn.toad.config import (
    TeacherSpec,
    ToadConfig,
    apply_overrides,
    load_config,
    resolved_round_geometry,
    structural_fingerprint,
    validate_stored_config,
)
from kaggriculture.learn.toad.data import expected_round_batch_counts
from kaggriculture.learn.toad.lightning import (
    ResumeConfigError,
    assert_resume_compatible,
)
from kaggriculture.learn.toad_loss import UNROLL_LENGTH


def test_control_config_reproduces_current_constants() -> None:
    """The native control starts from the current Toad recipe."""
    config = ToadConfig.control()
    assert config.model.blocks == toad.BLOCKS
    assert config.model.channels == toad.CHANNELS
    assert config.optimizer.unroll_length == UNROLL_LENGTH
    assert config.optimizer.batch_segments == toad.BATCH_SEGMENTS
    assert config.runtime.precision == "32-true"
    assert config.runtime.devices == 1
    assert config.population.selfplay + config.population.scripted == 1.0


def test_unroll_cannot_exceed_the_competition_decision_horizon() -> None:
    """An empty segment stream must be rejected before collection starts."""
    with pytest.raises(ValidationError, match="unroll_length.*719"):
        ToadConfig.model_validate({"optimizer": {"unroll_length": 720}})


def test_round_segments_must_fill_optimizer_batches_exactly() -> None:
    """A self-play seat cannot be truncated to fit the batch width."""
    with pytest.raises(ValidationError, match="segments.*divisible.*batch_segments"):
        ToadConfig.model_validate(
            {
                "population": {
                    "selfplay": 1.0,
                    "scripted": 0.0,
                    "environments_per_rank": 1,
                },
                "optimizer": {"batch_segments": 3},
            }
        )


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_round_geometry_is_identical_for_both_collection_backends(
    backend: str,
) -> None:
    """Backend selection cannot change policy or value replay batch quotas."""
    config = ToadConfig.model_validate(
        {
            "population": {
                "selfplay": 1.0,
                "scripted": 0.0,
                "environments_per_rank": 2,
            },
            "optimizer": {
                "unroll_length": 719,
                "batch_segments": 4,
                "value_passes": 2,
            },
            "runtime": {"rollout_backend": backend},
        }
    )

    counts = expected_round_batch_counts(config)

    assert counts.policy == 1
    assert counts.value == 2


def test_round_geometry_resolves_every_enabled_population_kind(
    tmp_path: Path,
) -> None:
    """Self-play's second seat and every single-seat kind are counted exactly."""
    frozen = tmp_path / "frozen.pt"
    teacher = tmp_path / "teacher.pt"
    frozen.touch()
    teacher.touch()
    config = ToadConfig.model_validate(
        {
            "population": {
                "selfplay": 0.25,
                "scripted": 0.25,
                "frozen_opponent": 0.25,
                "teacher_distill": 0.25,
                "environments_per_rank": 8,
                "initial_snapshots": [frozen],
                "teacher": {"checkpoint": teacher, "quantity": True},
            },
            "optimizer": {
                "unroll_length": 719,
                "batch_segments": 5,
                "value_passes": 3,
            },
        }
    )

    geometry = resolved_round_geometry(config.population, config.optimizer)

    assert geometry.environments_by_kind == {
        "selfplay": 2,
        "scripted": 2,
        "frozen_opponent": 2,
        "teacher_distill": 2,
    }
    assert geometry.trajectories_by_kind == {
        "selfplay": 4,
        "scripted": 2,
        "frozen_opponent": 2,
        "teacher_distill": 2,
    }
    assert geometry.segments_by_kind == geometry.trajectories_by_kind
    assert geometry.policy_batches == 2
    assert geometry.value_batches == 6


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lr", float("nan")),
        ("lr", float("inf")),
        ("gamma", -0.01),
        ("gamma", 1.01),
        ("gamma", float("nan")),
        ("lmb", -0.01),
        ("lmb", 1.01),
        ("entropy_cost", -0.01),
        ("teacher_kl_cost", -0.01),
        ("baseline_cost", float("inf")),
        ("clip_grad_norm", float("inf")),
    ],
)
def test_optimizer_rejects_nonfinite_or_mathematically_invalid_values(
    field: str, value: float
) -> None:
    """Invalid optimizer math cannot survive typed config resolution."""
    with pytest.raises(ValidationError):
        ToadConfig.model_validate({"optimizer": {field: value}})


@pytest.mark.parametrize(
    "payload",
    [
        {"model": {"value_bound": float("inf")}},
        {"population": {"selfplay": float("nan")}},
        {
            "optimizer": {
                "entropy": {
                    "operation": {
                        "initial_target": 0.0,
                        "initial_multiplier": 0.001,
                        "multiplier_change_per_step": 0.0,
                        "maximum": float("inf"),
                    }
                }
            }
        },
        {
            "curriculum": {
                "gate": {
                    "metric": "win_rate",
                    "minimum": float("nan"),
                    "opponent": "economic",
                    "seeds": 1,
                }
            }
        },
        {"curriculum": {"money_weight": float("inf")}},
    ],
)
def test_every_serialized_float_domain_rejects_nan_and_infinity(
    payload: dict[str, object],
) -> None:
    """Non-finite values cannot hide outside the optimizer's top-level fields."""
    with pytest.raises(ValidationError):
        ToadConfig.model_validate(payload)


def test_effective_resume_must_be_a_readable_lightning_checkpoint(
    tmp_path: Path,
) -> None:
    """Missing and directory resumes fail during effective config validation."""
    missing = tmp_path / "missing.ckpt"
    directory = tmp_path / "directory.ckpt"
    directory.mkdir()

    for path in (missing, directory):
        with pytest.raises(ValidationError, match="resume checkpoint is not readable"):
            ToadConfig.model_validate({"runtime": {"resume": path}})


def test_historical_resume_config_skips_external_checkpoint_readability(
    tmp_path: Path,
) -> None:
    """Stored schema remains valid after its original resume source is removed."""
    payload = ToadConfig.control().model_dump(mode="python")
    payload["runtime"]["resume"] = tmp_path / "removed.ckpt"

    stored = validate_stored_config(payload)

    assert stored.runtime.resume == tmp_path / "removed.ckpt"


def test_legacy_resume_message_names_supported_weight_only_paths(
    tmp_path: Path,
) -> None:
    """A legacy artifact cannot imply that full resume support is forthcoming."""
    checkpoint = tmp_path / "legacy.pt"
    checkpoint.touch()

    with pytest.raises(ValidationError) as excinfo:
        ToadConfig.model_validate({"runtime": {"resume": checkpoint}})

    message = str(excinfo.value)
    assert "full Lightning resume" in message
    assert "warm_start_checkpoint" in message
    assert "read-only" in message
    assert "Stage 9" not in message


def test_batch_probabilities_must_sum_to_one() -> None:
    """Invalid population weights cannot reach the trainer."""
    with pytest.raises(ValidationError, match="sum to one"):
        ToadConfig(population={"selfplay": 0.8, "scripted": 0.8})


def test_frozen_population_requires_a_first_round_snapshot_source() -> None:
    """A validated run cannot request frozen opponents from an empty first pool."""
    with pytest.raises(ValidationError, match="frozen_opponent.*first-round"):
        ToadConfig(
            population={
                "selfplay": 0.5,
                "scripted": 0.0,
                "frozen_opponent": 0.5,
            }
        )


def test_teacher_loss_requires_a_teacher() -> None:
    """A teacher loss cannot silently run without a teacher checkpoint."""
    with pytest.raises(ValidationError, match="teacher checkpoint"):
        ToadConfig(optimizer={"teacher_kl_cost": 0.1})


def test_teacher_checkpoint_must_be_a_readable_file(tmp_path: Path) -> None:
    """Teacher-enabled runs verify the checkpoint before they start."""
    checkpoint = tmp_path / "teacher.ckpt"
    checkpoint.touch()

    config = ToadConfig(
        population={"teacher": {"checkpoint": checkpoint}},
        optimizer={"teacher_kl_cost": 0.1},
    )

    assert config.population.teacher == TeacherSpec(checkpoint=checkpoint)


def test_legacy_teacher_path_and_blocks_migrate_atomically(tmp_path: Path) -> None:
    """One-release inputs become one immutable teacher rather than parallel fields."""
    checkpoint = tmp_path / "teacher.ckpt"
    checkpoint.touch()

    config = ToadConfig(
        population={"teacher_checkpoint": checkpoint, "teacher_blocks": 16},
        optimizer={"teacher_kl_cost": 0.1},
    )

    assert config.population.teacher == TeacherSpec(
        checkpoint=checkpoint,
        blocks=16,
    )
    assert "teacher_checkpoint" not in config.population.model_dump()
    assert "teacher_blocks" not in config.population.model_dump()


def test_new_and_legacy_teacher_contracts_conflict(tmp_path: Path) -> None:
    """Ambiguous dual teacher sources fail before either source can win."""
    checkpoint = tmp_path / "teacher.ckpt"
    checkpoint.touch()

    with pytest.raises(ValidationError, match="teacher.*conflicts.*teacher_checkpoint"):
        ToadConfig(
            population={
                "teacher": {"checkpoint": checkpoint},
                "teacher_checkpoint": checkpoint,
            },
            optimizer={"teacher_kl_cost": 0.1},
        )


def test_teacher_spec_is_frozen_and_forbids_unknown_semantics(tmp_path: Path) -> None:
    """The serialized teacher contract cannot drift after validation."""
    checkpoint = tmp_path / "teacher.ckpt"
    checkpoint.touch()
    spec = TeacherSpec(checkpoint=checkpoint)

    with pytest.raises(ValidationError, match="frozen"):
        spec.quantity = True  # ty: ignore[invalid-assignment]
    with pytest.raises(ValidationError, match="extra_forbidden"):
        TeacherSpec(checkpoint=checkpoint, unknown=True)  # ty: ignore[unknown-argument]


def test_teacher_value_cost_requires_declared_compatible_value_semantics(
    tmp_path: Path,
) -> None:
    """Value alignment cannot compare differently structured value functions."""
    checkpoint = tmp_path / "teacher.ckpt"
    checkpoint.touch()

    accepted = ToadConfig(
        population={"teacher": {"checkpoint": checkpoint, "value": True}},
        optimizer={"teacher_baseline_cost": 0.5},
    )
    assert accepted.optimizer.teacher_baseline_cost == 0.5

    with pytest.raises(ValidationError, match="requires a declared teacher value"):
        ToadConfig(
            population={"teacher": {"checkpoint": checkpoint}},
            optimizer={"teacher_baseline_cost": 0.5},
        )
    with pytest.raises(ValidationError, match="compatible control value"):
        ToadConfig(
            model={"interaction_value": True},
            population={"teacher": {"checkpoint": checkpoint, "value": True}},
            optimizer={"teacher_baseline_cost": 0.5},
        )


@pytest.mark.parametrize("model", [{"kernel_size": 5}, {"activation": "leaky_relu"}])
def test_control_teacher_does_not_inherit_student_trunk_semantics(
    tmp_path: Path, model: dict[str, object]
) -> None:
    """Explicit control teachers remain valid beside a non-control student."""
    checkpoint = tmp_path / "teacher.ckpt"
    checkpoint.touch()

    config = ToadConfig(
        model=model,
        population={"teacher": {"checkpoint": checkpoint}},
        optimizer={"teacher_kl_cost": 0.1},
    )

    assert config.population.teacher is not None
    assert config.population.teacher.checkpoint == checkpoint


@pytest.mark.parametrize("missing_head", ["operation", "quantity", "market"])
def test_teacher_distill_requires_every_action_head_before_file_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_head: str,
) -> None:
    """Rollout-incomplete teachers fail validation before touching their path."""
    teacher = {
        "checkpoint": tmp_path / "missing.pt",
        "operation": True,
        "quantity": True,
        "market": True,
        missing_head: False,
    }

    def unexpected_open(*args: object, **kwargs: object) -> object:
        raise AssertionError("teacher path was opened before head validation")

    monkeypatch.setattr(Path, "open", unexpected_open)

    with pytest.raises(
        ValidationError,
        match="teacher_distill requires operation, quantity, and market",
    ):
        ToadConfig(
            population={
                "selfplay": 0.0,
                "scripted": 0.0,
                "teacher_distill": 1.0,
                "teacher": teacher,
            }
        )


def test_warm_start_checkpoint_is_readable_and_excludes_resume(tmp_path: Path) -> None:
    """Warm-start weights are validated input and cannot masquerade as resume."""
    checkpoint = tmp_path / "warm.pt"
    checkpoint.touch()

    config = ToadConfig(curriculum={"warm_start_checkpoint": checkpoint})
    assert config.curriculum.warm_start_checkpoint == checkpoint

    with pytest.raises(ValidationError, match="warm start and resume"):
        ToadConfig(
            curriculum={"warm_start_checkpoint": checkpoint},
            runtime={"resume": tmp_path / "resume.ckpt"},
        )


def test_warm_start_checkpoint_must_be_a_readable_file(tmp_path: Path) -> None:
    """An invalid warm start fails during config validation, before runtime."""
    with pytest.raises(ValidationError, match="warm-start checkpoint is not readable"):
        ToadConfig(curriculum={"warm_start_checkpoint": tmp_path / "missing.pt"})


def test_warm_start_provenance_does_not_make_resume_structurally_incompatible(
    tmp_path: Path,
) -> None:
    """A stored initialization source is provenance, not model structure."""
    warm = tmp_path / "warm.pt"
    warm.touch()
    resume = tmp_path / "resume.ckpt"
    resume.touch()
    stored = ToadConfig(curriculum={"warm_start_checkpoint": warm})
    effective = ToadConfig(runtime={"resume": resume})

    assert_resume_compatible(effective, stored)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("runtime.seed", 17),
        ("population.population_seed", 19),
        ("population.actor_sync_every_rounds", 2),
        ("optimizer.gamma", 0.9),
        ("optimizer.lmb", 0.7),
        ("optimizer.entropy_cost", 0.25),
        ("curriculum.phase", "phase2"),
        ("curriculum.reward_field", "own"),
        ("curriculum.money_weight", 0.5),
        ("model.blocks", 4),
        ("runtime.precision", "bf16-mixed"),
        ("runtime.compile.enabled", True),
        ("runtime.rollout_backend", "native"),
    ],
)
def test_resume_rejects_every_behavior_changing_config_drift(
    path: str,
    value: object,
) -> None:
    """The next assignment and update are bound to the complete resolved config."""
    stored = ToadConfig.control()
    effective = apply_overrides(stored, [f"{path}={json.dumps(value)}"])

    with pytest.raises(ResumeConfigError, match="resume config mismatch"):
        assert_resume_compatible(effective, stored)


def test_resume_allows_only_explicit_operational_overrides(tmp_path: Path) -> None:
    """Placement, diagnostics, destination, and resume path do not alter learning."""
    stored = ToadConfig.control()
    effective = stored.model_copy(
        update={
            "runtime": stored.runtime.model_copy(
                update={
                    "accelerator": "cpu",
                    "devices": 2,
                    "num_nodes": 2,
                    "strategy": "ddp",
                    "log_every_n_steps": 9,
                    "profiler": "simple",
                    "output_dir": tmp_path / "resumed",
                    "resume": tmp_path / "step.ckpt",
                }
            )
        }
    )

    assert_resume_compatible(effective, stored)


def test_resume_rejects_unlisted_runtime_budget_and_numerics_drift() -> None:
    """Operational does not mean arbitrary runtime settings are update-neutral."""
    stored = ToadConfig.control()
    for field, value in (
        ("total_environment_steps", stored.runtime.total_environment_steps + 1),
        ("checkpoint_every_environment_steps", 17),
        ("deterministic", True),
    ):
        effective = stored.model_copy(
            update={"runtime": stored.runtime.model_copy(update={field: value})}
        )
        with pytest.raises(ResumeConfigError, match="resume config mismatch"):
            assert_resume_compatible(effective, stored)


def test_teacher_checkpoint_must_open_for_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A regular file that cannot be opened is not a usable teacher checkpoint."""
    checkpoint = tmp_path / "teacher.ckpt"
    checkpoint.touch()

    def fail_to_open(*args: object, **kwargs: object) -> None:
        raise OSError("simulated unreadable checkpoint")

    monkeypatch.setattr(Path, "open", fail_to_open)

    with pytest.raises(ValidationError, match="teacher checkpoint is not readable"):
        ToadConfig(
            population={"teacher_checkpoint": checkpoint},
            optimizer={"teacher_kl_cost": 0.1},
        )


def test_evaluation_gate_rejects_unknown_fields() -> None:
    """Evaluation gates cannot silently discard misspelled configuration."""
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ToadConfig(
            curriculum={
                "gate": {
                    "metric": "win_rate",
                    "minimum": 0.5,
                    "opponent": "economic",
                    "seeds": 1,
                    "unexpected": True,
                }
            }
        )


def test_evaluation_gate_is_frozen() -> None:
    """A validated gate remains immutable with the rest of the experiment."""
    config = ToadConfig(
        curriculum={
            "gate": {
                "metric": "win_rate",
                "minimum": 0.5,
                "opponent": "economic",
                "seeds": 1,
            }
        }
    )
    assert config.curriculum.gate is not None

    with pytest.raises(ValidationError, match="frozen"):
        config.curriculum.gate.metric = "loss"  # ty: ignore[invalid-assignment]


def test_load_config_applies_json_typed_overrides(tmp_path: Path) -> None:
    """JSON files and overrides reconstruct one fully validated config."""
    path = tmp_path / "toad.json"
    path.write_text(json.dumps({"optimizer": {"lr": 0.0002}}))

    config = load_config(path, ["runtime.seed=42", 'model.activation="leaky_relu"'])

    assert config.optimizer.lr == 0.0002
    assert config.runtime.seed == 42
    assert config.model.activation == "leaky_relu"


def test_overrides_reject_unknown_paths() -> None:
    """A misspelled override cannot be accepted and then ignored."""
    with pytest.raises(ValueError, match="unknown override path"):
        apply_overrides(ToadConfig.control(), ["optimizer.unknown=1"])


def test_structural_fingerprint_ignores_nonstructural_settings() -> None:
    """Only model shape and batch structure determine compatibility."""
    control = ToadConfig.control()
    changed_seed = apply_overrides(control, ["runtime.seed=42"])
    changed_shape = apply_overrides(control, ["model.channels=256"])

    assert structural_fingerprint(changed_seed) == structural_fingerprint(control)
    assert structural_fingerprint(changed_shape) != structural_fingerprint(control)


def test_native_toad_modules_are_not_in_the_vendored_exclusion() -> None:
    """The native config stays subject to the project quality checks."""
    precommit = Path(".pre-commit-config.yaml").read_text()
    exclude = next(
        line.strip() for line in precommit.splitlines() if line.startswith("exclude:")
    )
    assert exclude != "exclude: ^src/kaggriculture/learn/toad/"
    assert "config\\.py" not in exclude
