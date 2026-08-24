"""Tests for the typed Toad CLI and native Lightning runner boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from kaggriculture.learn.scripts import toad
from kaggriculture.learn.toad.callbacks import (
    ActorSyncCallback,
    BoundaryCheckpoint,
    EnvironmentStepStop,
)
from kaggriculture.learn.toad.config import ToadConfig
from kaggriculture.learn.toad.lightning import compute_loss
from tests.learn.test_toad_control_fixture import (
    control_fixture_batch,
    load_control_fixture,
)


def test_cli_resolves_config_and_json_overrides(tmp_path: Path) -> None:
    """A dotted override must reach the one validated config object."""
    path = tmp_path / "toad.json"
    path.write_text(json.dumps(ToadConfig.control().model_dump(mode="json")))

    config = toad.parse_config(["--config", str(path), "--set", "optimizer.lr=5e-5"])

    assert config.optimizer.lr == 5e-5


def test_primary_parser_rejects_legacy_flags() -> None:
    """The documented argparse surface is config plus repeatable overrides."""
    with pytest.raises(SystemExit):
        toad.parse_config(["--lr", "5e-5"])


def test_main_help_exposes_only_the_primary_surface(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Production help must not advertise the one-release compatibility flags."""
    with pytest.raises(SystemExit):
        toad.main(["--help"])

    help_text = capsys.readouterr().out
    assert "--config" in help_text
    assert "--set" in help_text
    assert "--lr" not in help_text


def test_legacy_flags_translate_once_and_call_the_native_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A legacy invocation must warn and converge on the same typed run path."""
    seen: list[ToadConfig] = []
    monkeypatch.setattr(toad, "run", seen.append)

    with pytest.warns(DeprecationWarning, match="legacy Toad flags"):
        toad.main(
            [
                "--lr",
                "5e-5",
                "--sparse",
                "--name",
                "phase-old",
                "--total-steps",
                "1234",
            ]
        )

    assert len(seen) == 1
    config = seen[0]
    assert config.optimizer.lr == 5e-5
    assert config.curriculum.reward_field == "sparse"
    assert config.curriculum.phase == "phase-old"
    assert config.runtime.total_environment_steps == 1234


def test_all_active_legacy_semantics_reach_typed_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compatibility is semantic: no active legacy value is ignored or rejected."""
    warm_start = tmp_path / "clone.pt"
    teacher = tmp_path / "teacher.pt"
    torch.save(toad.Policy(blocks=1, channels=16).state_dict(), warm_start)
    teacher.touch()
    monkeypatch.setattr(toad, "CHECKPOINT", warm_start)

    config = toad._legacy_config(
        [
            "--margin",
            "--money-weight",
            "0.01",
            "--clone-init",
            "--blocks",
            "2",
            "--channels",
            "16",
            "--teacher",
            str(teacher),
            "--teacher-blocks",
            "1",
            "--teacher-quantity",
            "--teacher-kl-cost",
            "0.005",
        ]
    )

    assert config.curriculum.reward_field == "margin"
    assert config.curriculum.money_weight == pytest.approx(0.01)
    assert config.curriculum.warm_start_checkpoint == warm_start
    assert config.population.teacher is not None
    assert config.population.teacher.checkpoint == teacher
    assert config.population.teacher.blocks == 1
    assert config.population.teacher.quantity is True


@pytest.mark.parametrize(
    ("quantity_flag", "has_quantity"),
    [("--teacher-quantity", True), ("--no-teacher-quantity", False)],
)
def test_legacy_teacher_quantity_flag_controls_the_quantity_loss(
    tmp_path: Path,
    quantity_flag: str,
    has_quantity: bool,
) -> None:
    """Legacy compatibility declares quantity validity instead of inferring it."""
    checkpoint = tmp_path / "teacher.pt"
    state = toad.Policy(
        blocks=1,
        channels=16,
        value_bound=toad.VALUE_BOUND,
    ).state_dict()
    torch.save(
        {
            name: value
            for name, value in state.items()
            if has_quantity or not name.startswith("quantity_head.")
        },
        checkpoint,
    )
    config = toad._legacy_config(
        [
            "--blocks",
            "1",
            "--channels",
            "16",
            "--teacher",
            str(checkpoint),
            "--teacher-blocks",
            "1",
            quantity_flag,
            "--teacher-kl-cost",
            "0.25",
        ]
    )
    module = toad.ToadLightningModule(config)

    report = compute_loss(
        module.policy,
        control_fixture_batch(load_control_fixture()),
        module.config,
        module.teacher,
    )

    assert module.teacher is not None
    assert module.teacher.spec.quantity is has_quantity
    if has_quantity:
        assert report.terms["teacher/quantity_kl"].item() > 0.0
    else:
        assert report.terms["teacher/quantity_kl"].item() == 0.0


def test_legacy_teacher_requires_an_explicit_quantity_compatibility_flag(
    tmp_path: Path,
) -> None:
    """A historical invocation cannot silently default teacher head validity."""
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.touch()

    with pytest.raises(ValueError, match="teacher quantity compatibility"):
        toad._legacy_config(["--teacher", str(checkpoint)])


def test_build_trainer_uses_environment_budget_and_boundary_callbacks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Trainer settings must leave budget, clipping, and boundaries in one place."""
    captured: dict[str, Any] = {}
    logger = object()

    class _Trainer:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(toad.lightning, "Trainer", _Trainer)
    monkeypatch.setattr(toad, "build_wandb_logger", lambda config: logger)
    config = ToadConfig.control().model_copy(
        update={
            "runtime": ToadConfig.control().runtime.model_copy(
                update={
                    "output_dir": tmp_path,
                    "total_environment_steps": 321,
                }
            )
        }
    )

    toad.build_trainer(config)

    assert captured["gradient_clip_val"] == config.optimizer.clip_grad_norm
    assert captured["gradient_clip_algorithm"] == "norm"
    assert captured["max_steps"] == -1
    assert captured["max_epochs"] == -1
    assert captured["use_distributed_sampler"] is False
    assert captured["enable_checkpointing"] is False
    assert captured["logger"] is logger
    callbacks = captured["callbacks"]
    assert isinstance(callbacks, list)
    assert [type(callback) for callback in callbacks] == [
        ActorSyncCallback,
        EnvironmentStepStop,
        BoundaryCheckpoint,
    ]
    stop = callbacks[1]
    checkpoint = callbacks[2]
    assert isinstance(stop, EnvironmentStepStop)
    assert stop.total_environment_steps == 321
    assert isinstance(checkpoint, BoundaryCheckpoint)
    assert checkpoint.output_dir == tmp_path


def test_wandb_logger_receives_the_complete_resolved_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tracking receives the full JSON model, not a second hand-built surface."""
    captured: dict[str, object] = {}

    class _Logger:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(toad, "WandbLogger", _Logger)
    config = ToadConfig.control()

    toad.build_wandb_logger(config)

    assert captured["config"] == config.model_dump(mode="json")


def test_run_seeds_builds_native_components_and_passes_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Production execution delegates placement, optimization, and restore to fit."""
    events: list[tuple[object, ...]] = []
    config = ToadConfig.control().model_copy(
        update={
            "runtime": ToadConfig.control().runtime.model_copy(
                update={"resume": tmp_path / "resume.ckpt"}
            )
        }
    )
    effective = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(
                update={"output_dir": tmp_path / "confirmed"}
            )
        }
    )

    class _Module:
        def __init__(self, original: ToadConfig) -> None:
            assert original is config
            self.config = effective

    module = _Module(config)
    data = object()
    built: list[tuple[str, ToadConfig]] = []

    class _Trainer:
        def fit(self, fitted: object, **kwargs: object) -> None:
            events.append(("fit", fitted, kwargs))

    monkeypatch.setattr(
        toad,
        "seed_everything",
        lambda seed, workers: events.append(("seed", seed, workers)),
    )
    monkeypatch.setattr(toad, "ToadLightningModule", lambda original: module)
    monkeypatch.setattr(
        toad,
        "build_reference_data_module",
        lambda received: built.append(("data", received)) or data,
    )
    monkeypatch.setattr(
        toad,
        "build_trainer",
        lambda received: built.append(("trainer", received)) or _Trainer(),
    )

    toad.run(config)

    assert events == [
        ("seed", config.runtime.seed, True),
        ("fit", module, {"datamodule": data, "ckpt_path": effective.runtime.resume}),
    ]
    assert built == [("data", effective), ("trainer", effective)]
