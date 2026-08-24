"""Tests for the typed Toad CLI and native Lightning runner boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from pydantic import ValidationError

from kaggriculture.learn.scripts import toad
from kaggriculture.learn.toad.callbacks import (
    ActorSyncCallback,
    BoundaryCheckpoint,
    EnvironmentStepStop,
    PopulationSnapshotCallback,
)
from kaggriculture.learn.toad.config import ToadConfig
from kaggriculture.learn.toad.lightning import compute_loss
from tests.learn.test_toad_control_fixture import (
    control_fixture_batch,
    load_control_fixture,
)


def _runtime_config(
    *,
    accelerator: str = "cpu",
    devices: int | str | list[int] = 1,
    num_nodes: int = 1,
    strategy: str = "auto",
    precision: str = "32-true",
    compile_enabled: bool = False,
    rollout_backend: str = "reference",
    rollout_device: str = "cpu",
    rollout_cuda_graph: bool = False,
    scripted: float = 0.5,
    scripted_opponent: str = "economic",
) -> ToadConfig:
    """Build one literal runtime-matrix row from the typed control contract."""
    payload = ToadConfig.control().model_dump(mode="python")
    runtime = payload["runtime"]
    assert isinstance(runtime, dict)
    runtime.update(
        accelerator=accelerator,
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategy,
        precision=precision,
        compile={"enabled": compile_enabled},
        rollout_backend=rollout_backend,
        rollout_device=rollout_device,
        rollout_cuda_graph=rollout_cuda_graph,
    )
    population = payload["population"]
    assert isinstance(population, dict)
    population.update(
        selfplay=1.0 - scripted,
        scripted=scripted,
        scripted_opponent=scripted_opponent,
    )
    return ToadConfig.model_validate(payload)


@pytest.mark.parametrize(
    "config",
    [
        _runtime_config(),
        _runtime_config(accelerator="gpu", precision="bf16-mixed"),
        _runtime_config(
            accelerator="gpu", precision="bf16-mixed", compile_enabled=True
        ),
        _runtime_config(
            accelerator="gpu", precision="bf16-mixed", rollout_backend="native"
        ),
        _runtime_config(devices=2, strategy="ddp"),
        _runtime_config(devices=2, strategy="ddp", rollout_backend="native"),
        _runtime_config(num_nodes=2, strategy="ddp"),
    ],
    ids=(
        "reference-eager-fp32-cpu",
        "reference-eager-bf16-gpu",
        "reference-compile-bf16-gpu",
        "native-eager-bf16-gpu",
        "reference-ddp",
        "native-ddp",
        "reference-multinode-ddp",
    ),
)
def test_supported_runtime_matrix(
    config: ToadConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every proved eager, BF16, compile, native, and DDP row is explicit."""
    monkeypatch.setattr(toad, "_cuda_available", lambda: True)
    monkeypatch.setattr(toad, "_cuda_device_count", lambda: 4)
    monkeypatch.setattr(toad, "_cuda_bf16_supported", lambda: True)
    monkeypatch.setattr(toad, "_cpu_bf16_supported", lambda: True)

    assert toad.runtime_preflight(config) is None


@pytest.mark.parametrize(
    ("config", "cuda_available", "cuda_devices", "cuda_bf16", "cpu_bf16", "error"),
    [
        (
            _runtime_config(accelerator="gpu"),
            False,
            0,
            False,
            True,
            "GPU accelerator requested but CUDA is unavailable",
        ),
        (
            _runtime_config(accelerator="gpu", precision="bf16-mixed"),
            True,
            1,
            False,
            True,
            "bf16-mixed requested but CUDA BF16 is unavailable",
        ),
        (
            _runtime_config(accelerator="gpu", devices=2, strategy="ddp"),
            True,
            1,
            True,
            True,
            "requested GPU devices are unavailable: requested=2, available=1",
        ),
        (
            _runtime_config(accelerator="gpu", devices=[0, 2], strategy="ddp"),
            True,
            2,
            True,
            True,
            "requested GPU device index is unavailable: available=2, devices=(0, 2)",
        ),
        (
            _runtime_config(accelerator="gpu", devices=[0, 0], strategy="ddp"),
            True,
            2,
            True,
            True,
            "requested GPU device indexes must be unique: devices=(0, 0)",
        ),
        (
            _runtime_config(devices=2),
            True,
            2,
            True,
            True,
            "resolved world size above one requires DDP strategy='ddp': "
            "world_size=2, strategy='auto'",
        ),
        (
            _runtime_config(num_nodes=2),
            True,
            2,
            True,
            True,
            "resolved world size above one requires DDP strategy='ddp': "
            "world_size=2, strategy='auto'",
        ),
        (
            _runtime_config(precision="bf16-mixed"),
            False,
            0,
            False,
            False,
            "bf16-mixed requested but CPU BF16 autocast is unavailable",
        ),
        (
            _runtime_config(compile_enabled=True, rollout_backend="native"),
            True,
            1,
            True,
            True,
            "native rollout with torch.compile is not proved; refusing eager fallback",
        ),
        (
            _runtime_config(rollout_cuda_graph=True),
            True,
            1,
            True,
            True,
            "CUDA-graph rollout requires rollout_backend='native'",
        ),
        (
            _runtime_config(rollout_backend="native", rollout_cuda_graph=True),
            True,
            1,
            True,
            True,
            "scripted CUDA-graph rollout is unsupported because the scripted "
            "opponent reads simulator rows on the host",
        ),
        (
            _runtime_config(
                rollout_backend="native", rollout_cuda_graph=True, scripted=0.0
            ),
            True,
            1,
            True,
            True,
            "CUDA-graph round collection is not yet selectable; refusing eager "
            "fallback",
        ),
        (
            _runtime_config(rollout_backend="native", scripted_opponent="starter"),
            True,
            1,
            True,
            True,
            "native scripted rollout supports only the verified 'economic' opponent",
        ),
        (
            _runtime_config(rollout_device="cuda"),
            True,
            1,
            True,
            True,
            "reference rollout does not consume rollout_device; refusing an ignored "
            "'cuda' request",
        ),
        (
            _runtime_config(rollout_backend="native", rollout_device="cuda"),
            False,
            0,
            False,
            True,
            "native CUDA rollout requested but CUDA is unavailable",
        ),
    ],
    ids=(
        "gpu-unavailable",
        "cuda-bf16-unavailable",
        "gpu-count-unavailable",
        "gpu-index-unavailable",
        "gpu-index-duplicate",
        "devices-require-ddp",
        "nodes-require-ddp",
        "cpu-bf16-unavailable",
        "native-compile-unproved",
        "reference-graph",
        "scripted-native-graph",
        "native-graph-unproved",
        "native-opponent-unproved",
        "reference-device-ignored",
        "native-device-unavailable",
    ),
)
def test_rejected_runtime_matrix_has_exact_preflight_errors(
    config: ToadConfig,
    cuda_available: bool,
    cuda_devices: int,
    cuda_bf16: bool,
    cpu_bf16: bool,
    error: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported runtime requests fail closed before Trainer construction."""
    monkeypatch.setattr(toad, "_cuda_available", lambda: cuda_available)
    monkeypatch.setattr(toad, "_cuda_device_count", lambda: cuda_devices)
    monkeypatch.setattr(toad, "_cuda_bf16_supported", lambda: cuda_bf16)
    monkeypatch.setattr(toad, "_cpu_bf16_supported", lambda: cpu_bf16)

    with pytest.raises(toad.RuntimePreflightError) as raised:
        toad.runtime_preflight(config)

    assert str(raised.value) == error


def test_runtime_metadata_is_frozen_and_records_resolved_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One run records precision, compiler, world size, and rollout backend."""
    monkeypatch.setattr(toad, "_cuda_available", lambda: False)
    config = _runtime_config(
        devices=2,
        strategy="ddp",
        precision="bf16-mixed",
        compile_enabled=True,
    )

    metadata = toad.runtime_metadata(config)

    assert metadata.model_dump(mode="json") == {
        "precision": "bf16-mixed",
        "compile": {
            "enabled": True,
            "mode": "default",
            "fullgraph": False,
            "dynamic": False,
        },
        "world_size": 2,
        "rollout_backend": "reference",
    }
    with pytest.raises(ValidationError, match="frozen"):
        metadata.world_size = 1  # ty: ignore[invalid-assignment]


def test_only_lightning_orchestration_remains_callable() -> None:
    """The retired learner/device/checkpoint/update path cannot be selected."""
    retired = {
        "_learner",
        "_optimizer",
        "_start_run",
        "_log_definitions",
        "_teacher",
        "_prefix",
        "_default_name",
        "_warm_start",
        "_checkpoint",
        "_restore",
        "_device",
        "_collect",
        "_play",
        "_update",
        "_value_passes",
        "_step",
    }

    assert not retired.intersection(vars(toad))


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
        PopulationSnapshotCallback,
        BoundaryCheckpoint,
    ]
    stop = callbacks[1]
    snapshot = callbacks[2]
    checkpoint = callbacks[3]
    assert isinstance(stop, EnvironmentStepStop)
    assert stop.total_environment_steps == 321
    assert isinstance(snapshot, PopulationSnapshotCallback)
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

    logged = cast(dict[str, object], captured["config"])
    resolved = config.model_dump(mode="json")
    assert captured["entity"] == toad.WANDB_ENTITY
    assert {key: logged[key] for key in resolved} == resolved
    assert logged["runtime_metadata"] == {
        "precision": "32-true",
        "compile": config.runtime.compile.model_dump(mode="json"),
        "world_size": 1,
        "rollout_backend": "reference",
    }
    assert logged["metric_definitions"] == toad.METRIC_DEFINITIONS


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
