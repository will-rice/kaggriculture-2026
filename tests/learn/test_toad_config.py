"""Tests for the serializable configuration of the native Toad trainer."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from kaggriculture.learn.scripts import toad
from kaggriculture.learn.toad.config import (
    ToadConfig,
    apply_overrides,
    load_config,
    structural_fingerprint,
)
from kaggriculture.learn.toad.lightning import assert_resume_compatible
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


def test_batch_probabilities_must_sum_to_one() -> None:
    """Invalid population weights cannot reach the trainer."""
    with pytest.raises(ValidationError, match="sum to one"):
        ToadConfig(population={"selfplay": 0.8, "scripted": 0.8})


def test_teacher_loss_requires_a_teacher() -> None:
    """A teacher loss cannot silently run without a teacher checkpoint."""
    with pytest.raises(ValidationError, match="teacher checkpoint"):
        ToadConfig(optimizer={"teacher_kl_cost": 0.1})


def test_teacher_checkpoint_must_be_a_readable_file(tmp_path: Path) -> None:
    """Teacher-enabled runs verify the checkpoint before they start."""
    checkpoint = tmp_path / "teacher.ckpt"
    checkpoint.touch()

    config = ToadConfig(
        population={"teacher_checkpoint": checkpoint},
        optimizer={"teacher_kl_cost": 0.1},
    )

    assert config.population.teacher_checkpoint == checkpoint


@pytest.mark.parametrize("model", [{"kernel_size": 5}, {"activation": "leaky_relu"}])
def test_teacher_rejects_ambiguous_nondefault_trunk_semantics(
    tmp_path: Path, model: dict[str, object]
) -> None:
    """Teacher checkpoints cannot silently inherit undeclared trunk settings."""
    checkpoint = tmp_path / "teacher.ckpt"
    checkpoint.touch()

    with pytest.raises(ValidationError, match="teacher.*default kernel.*activation"):
        ToadConfig(
            model=model,
            population={"teacher_checkpoint": checkpoint},
            optimizer={"teacher_kl_cost": 0.1},
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
    stored = ToadConfig(curriculum={"warm_start_checkpoint": warm})
    effective = ToadConfig(runtime={"resume": tmp_path / "resume.ckpt"})

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
