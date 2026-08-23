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
            "--teacher-kl-cost",
            "0.005",
        ]
    )

    assert config.curriculum.reward_field == "margin"
    assert config.curriculum.money_weight == pytest.approx(0.01)
    assert config.model.warm_start_checkpoint == warm_start
    assert config.population.teacher_checkpoint == teacher
    assert config.population.teacher_blocks == 1


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
    module = object()
    data = object()
    resume = tmp_path / "resume.ckpt"

    class _Trainer:
        def fit(self, fitted: object, **kwargs: object) -> None:
            events.append(("fit", fitted, kwargs))

    monkeypatch.setattr(
        toad,
        "seed_everything",
        lambda seed, workers: events.append(("seed", seed, workers)),
    )
    monkeypatch.setattr(toad, "ToadLightningModule", lambda config: module)
    monkeypatch.setattr(toad, "build_reference_data_module", lambda config: data)
    monkeypatch.setattr(toad, "build_trainer", lambda config: _Trainer())
    config = ToadConfig.control().model_copy(
        update={
            "runtime": ToadConfig.control().runtime.model_copy(
                update={"resume": resume}
            )
        }
    )

    toad.run(config)

    assert events == [
        ("seed", config.runtime.seed, True),
        ("fit", module, {"datamodule": data, "ckpt_path": resume}),
    ]
