"""Precision boundaries and finite diagnostics for the Toad learner."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import replace

import lightning
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from kaggriculture.learn.model import Policy
from kaggriculture.learn.scripts.toad import (
    RuntimePreflightError,
    runtime_preflight,
)
from kaggriculture.learn.toad.config import ToadConfig
from kaggriculture.learn.toad.data import LearnerBatch
from kaggriculture.learn.toad.lightning import (
    NonFiniteTrainingError,
    ToadLightningModule,
)
from tests.learn.test_toad_control_fixture import (
    control_fixture_batch,
    control_fixture_config,
    load_control_fixture,
)


def _bf16_config(accelerator: str = "cpu") -> ToadConfig:
    """Return a small resolved BF16 configuration suitable for unit tests."""
    control = control_fixture_config()
    return ToadConfig.model_validate(
        control.model_dump(mode="json")
        | {"runtime": {"accelerator": accelerator, "precision": "bf16-mixed"}}
    )


def test_bf16_is_rejected_when_accelerator_has_no_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A requested CUDA BF16 run must never silently fall back to FP32."""
    from kaggriculture.learn.scripts import toad

    monkeypatch.setattr(
        toad,
        "_cuda_capabilities",
        lambda: toad.CudaCapabilities(available=True, device_count=1, bf16=False),
    )

    with pytest.raises(RuntimePreflightError, match="bf16-mixed"):
        runtime_preflight(_bf16_config("gpu"))


def test_gpu_bf16_preflight_never_initializes_cuda_in_the_parent() -> None:
    """The capability probe may isolate CUDA work, but the parent stays clean."""
    source = """
import torch
from kaggriculture.learn.scripts import toad
from kaggriculture.learn.toad.config import ToadConfig

toad._cuda_capabilities = lambda: toad.CudaCapabilities(True, 1, False)
try:
    toad.runtime_preflight(
        ToadConfig(runtime={"accelerator": "gpu", "precision": "bf16-mixed"})
    )
except toad.RuntimePreflightError:
    pass
else:
    raise AssertionError("expected BF16 preflight rejection")
assert not torch.cuda.is_initialized()
"""
    completed = subprocess.run(
        [sys.executable, "-c", source],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_cuda_preflight_uses_one_disposable_capability_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Availability, count, and BF16 checks must never inspect parent CUDA."""
    from subprocess import CompletedProcess

    from kaggriculture.learn.scripts import toad

    calls: list[list[str]] = []

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("parent CUDA runtime inspection is forbidden")

    def probe(command: list[str], **_kwargs: object) -> CompletedProcess[str]:
        calls.append(command)
        return CompletedProcess(
            command, 0, '{"available":true,"device_count":2,"bf16":true}\n', ""
        )

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "device_count", forbidden)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", forbidden)
    monkeypatch.setattr(toad.subprocess, "run", probe)

    config = ToadConfig.model_validate(
        control_fixture_config().model_dump(mode="python")
        | {
            "runtime": {
                "accelerator": "gpu",
                "devices": [1],
                "precision": "bf16-mixed",
            }
        }
    )
    runtime_preflight(config)

    assert len(calls) == 1


def test_finite_provenance_defers_recurrent_norm_scalar_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Healthy forward provenance keeps state diagnostics device-resident."""
    from kaggriculture.learn.toad.lightning import NonFiniteProvenance
    from kaggriculture.learn.toad.model import PolicyState

    batch = control_fixture_batch(load_control_fixture())
    state = PolicyState(
        hidden=torch.ones(1, 2, 2),
        cell=torch.ones(1, 2, 2),
        prior_belief=torch.empty(1, 0),
    )

    def forbidden(_value: torch.Tensor) -> float:
        raise AssertionError("healthy provenance converted a tensor scalar")

    monkeypatch.setattr(torch.Tensor, "__float__", forbidden)

    provenance = NonFiniteProvenance.from_batch(batch, state, "32-true")

    assert provenance.game_ids == batch.game_ids


def test_healthy_finite_scan_materializes_one_scalar_for_the_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finite phase synchronizes once, independent of its tensor count."""
    from kaggriculture.learn.toad import lightning as toad_lightning

    original = torch.Tensor.__bool__
    scalar_reads: list[int] = []

    def counted(value: torch.Tensor) -> bool:
        scalar_reads.append(value.numel())
        return original(value)

    monkeypatch.setattr(torch.Tensor, "__bool__", counted)

    assert (
        toad_lightning._nonfinite_tensor_names(
            {
                "first": torch.ones(2),
                "second": torch.ones(3),
                "third": torch.ones(4),
            }
        )
        == []
    )
    assert scalar_reads == [1]


def test_healthy_gradient_phase_materializes_only_the_aggregate_scalar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gradient diagnostics defer local names until the synchronized flag fails."""
    from kaggriculture.learn.toad.lightning import NonFiniteProvenance

    module = ToadLightningModule(control_fixture_config())
    batch = control_fixture_batch(load_control_fixture())
    module._last_finite_provenance = NonFiniteProvenance.from_batch(
        batch, None, "32-true"
    )
    for parameter in module.policy.parameters():
        parameter.grad = torch.ones_like(parameter)

    original_bool = torch.Tensor.__bool__
    original_item = torch.Tensor.item
    scalar_reads: list[str] = []

    def counted_bool(value: torch.Tensor) -> bool:
        scalar_reads.append("bool")
        return original_bool(value)

    def counted_item(value: torch.Tensor, *args: object) -> object:
        scalar_reads.append("item")
        return original_item(value, *args)

    monkeypatch.setattr(torch.Tensor, "__bool__", counted_bool)
    monkeypatch.setattr(torch.Tensor, "item", counted_item)

    module.on_after_backward()

    assert scalar_reads == ["item"]


def test_sensitive_policy_math_is_fp32_under_autocast() -> None:
    """Loss/return math stays FP32 even when the network forward is autocast."""
    module = ToadLightningModule(_bf16_config())

    with torch.autocast("cpu", dtype=torch.bfloat16):
        report = module.compute_report(control_fixture_batch(load_control_fixture()))

    assert report.debug_dtypes["learner_log_probs"] is torch.float32
    assert report.debug_dtypes["importance_ratios"] is torch.float32
    assert report.debug_dtypes["value_targets"] is torch.float32
    assert report.debug_dtypes["entropy"] is torch.float32
    assert report.debug_dtypes["teacher"] is torch.float32


def test_nonfinite_error_keeps_batch_provenance() -> None:
    """A bad forward identifies the exact collection provenance and precision."""

    class NaNPolicy(Policy):
        def forward(
            self,
            board: torch.Tensor,
            scalars: torch.Tensor,
            positions: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            unit, quantity, market, values = super().forward(board, scalars, positions)
            return torch.full_like(unit, torch.nan), quantity, market, values

    config = control_fixture_config()
    module = ToadLightningModule(config)
    module.policy = NaNPolicy(
        blocks=1, channels=16, value_bound=config.model.value_bound
    )
    batch = control_fixture_batch(load_control_fixture())

    with pytest.raises(NonFiniteTrainingError) as raised:
        module.compute_report(batch)

    error = raised.value
    assert "unit_logits" in error.tensor_names
    assert error.batch_kind == "selfplay"
    assert error.game_ids == (0,)
    assert error.opponent_digests == ()
    assert error.actor_version == 0
    assert error.precision == "32-true"


def test_importance_ratio_overflow_is_not_hidden_by_vtrace_clipping() -> None:
    """Finite clipped returns cannot conceal an infinite raw importance ratio."""
    module = ToadLightningModule(control_fixture_config())
    batch = control_fixture_batch(load_control_fixture())
    overflowed = tuple(
        {**segment, "log_probs": torch.full_like(segment["log_probs"], -1000.0)}
        for segment in batch.segments
    )

    with pytest.raises(NonFiniteTrainingError) as raised:
        module.compute_report(replace(batch, segments=overflowed))

    assert "importance_ratios" in raised.value.tensor_names


def test_auto_cpu_rejects_explicit_device_topology_before_trainer_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU cannot reinterpret a GPU-index device tuple as a Lightning count."""
    from kaggriculture.learn.scripts import toad

    monkeypatch.setattr(
        toad,
        "_cuda_capabilities",
        lambda: toad.CudaCapabilities(False, 0, False),
    )
    config = ToadConfig.model_validate(
        control_fixture_config().model_dump(mode="json")
        | {"runtime": {"accelerator": "auto", "devices": [0]}}
    )

    with pytest.raises(RuntimePreflightError, match="CPU.*explicit device"):
        runtime_preflight(config)


def test_cpu_explicit_device_topology_is_rejected_by_config() -> None:
    """An explicitly CPU-bound device-index list fails during validation."""
    with pytest.raises(ValueError, match="CPU.*explicit device"):
        ToadConfig.model_validate(
            control_fixture_config().model_dump(mode="json")
            | {"runtime": {"accelerator": "cpu", "devices": [0]}}
        )


def test_nonfinite_initial_state_is_checked_before_recurrent_reset() -> None:
    """A reset mask cannot erase invalid recurrent input provenance."""
    from kaggriculture.learn.toad.data import BatchKind, LearnerBatch, segments
    from tests.learn.test_toad_data import _recurrent_trajectory

    config = ToadConfig.model_validate(
        {
            "model": {
                "blocks": 1,
                "channels": 16,
                "recurrent": True,
                "recurrent_channels": 3,
                "recurrent_layers": 2,
            }
        }
    )
    recurrent_segments = segments(_recurrent_trajectory(), 16)
    invalid = dict(recurrent_segments[0])
    invalid["initial_hidden"] = torch.full_like(invalid["initial_hidden"], torch.nan)
    invalid["dones"] = torch.ones_like(invalid["dones"], dtype=torch.bool)
    batch = LearnerBatch(
        segments=(invalid,),
        kind=BatchKind.SELFPLAY,
        baseline_only=False,
        first_of_round=True,
        end_of_round=True,
        collected_steps=16,
        round_id=0,
        actor_version=3,
        game_ids=(91,),
        opponent_ids=("self",),
    )

    with pytest.raises(NonFiniteTrainingError) as raised:
        ToadLightningModule(config).compute_report(batch)

    assert "input/initial_hidden" in raised.value.tensor_names
    assert raised.value.game_ids == (91,)


def test_after_backward_rejects_nonfinite_gradients_with_batch_provenance() -> None:
    """Finite losses cannot commit an optimizer step with infinite gradients."""

    class InfiniteGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx: object, value: torch.Tensor) -> torch.Tensor:
            return value

        @staticmethod
        def backward(
            ctx: torch.autograd.function.FunctionCtx, *gradients: object
        ) -> tuple[torch.Tensor]:
            del ctx
            (gradient,) = gradients
            assert isinstance(gradient, torch.Tensor)
            return (torch.full_like(gradient, torch.inf),)

    class InfiniteGradientPolicy(Policy):
        def forward(
            self,
            board: torch.Tensor,
            scalars: torch.Tensor,
            positions: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            unit, quantity, market, values = super().forward(board, scalars, positions)
            return InfiniteGradient.apply(unit), quantity, market, values

    config = control_fixture_config()
    module = ToadLightningModule(config)
    module.policy = InfiniteGradientPolicy(
        blocks=1, channels=16, value_bound=config.model.value_bound
    )
    batch = control_fixture_batch(load_control_fixture())
    loss = module.training_step(batch, 0)
    loss.backward()

    with pytest.raises(NonFiniteTrainingError) as raised:
        module.on_after_backward()

    assert any(name.startswith("grad/") for name in raised.value.tensor_names)
    assert raised.value.game_ids == batch.game_ids


def test_runtime_preflight_precedes_seed_and_trainer_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid runtime requests fail before the runner can cause side effects."""
    from kaggriculture.learn.scripts import toad

    events: list[str] = []
    config = control_fixture_config()
    monkeypatch.setattr(
        toad, "runtime_preflight", lambda received: events.append("preflight")
    )
    monkeypatch.setattr(
        toad, "seed_everything", lambda *_args, **_kwargs: events.append("seed")
    )
    monkeypatch.setattr(toad, "ToadLightningModule", lambda _config: None)

    with pytest.raises(AttributeError):
        toad.run(config)

    assert events == ["preflight", "seed"]


def test_explicit_device_topology_is_typed_and_normalized_for_lightning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The serializable tuple topology reaches Lightning's list-based API."""
    from kaggriculture.learn.scripts import toad

    captured: dict[str, object] = {}

    class Trainer:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    config = ToadConfig.model_validate(
        control_fixture_config().model_dump(mode="json")
        | {"runtime": {"accelerator": "gpu", "devices": [0]}}
    )
    monkeypatch.setattr(toad.lightning, "Trainer", Trainer)
    monkeypatch.setattr(toad, "build_wandb_logger", lambda _config: None)

    toad.build_trainer(config)

    assert config.runtime.devices == (0,)
    assert captured["devices"] == [0]


def test_multi_device_topology_fails_before_ddp_is_implemented() -> None:
    """Task 1 exposes topology without silently starting an unsupported DDP run."""
    config = ToadConfig.model_validate(
        control_fixture_config().model_dump(mode="json")
        | {"runtime": {"accelerator": "cpu", "devices": 2}}
    )

    with pytest.raises(RuntimePreflightError, match="DDP"):
        runtime_preflight(config)


@pytest.mark.cuda
def test_cuda_bf16_update_has_finite_loss_gradients_and_parameters() -> None:
    """A supported GPU performs one mixed-BF16 update without non-finite state."""
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 is unavailable")
    config = _bf16_config("gpu")
    module = ToadLightningModule(config).cuda()
    batch = module.transfer_batch_to_device(
        control_fixture_batch(load_control_fixture()), torch.device("cuda"), 0
    )
    optimizer = torch.optim.Adam(module.policy.parameters(), lr=config.optimizer.lr)

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        report = module.compute_report(batch)
    report.total.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in module.policy.parameters()
    )
    optimizer.step()
    assert all(
        torch.isfinite(parameter).all() for parameter in module.policy.parameters()
    )


@pytest.mark.cuda
def test_cuda_lightning_bf16_matches_fp32_automatic_optimization() -> None:
    """The production Trainer precision plugin keeps a short update near FP32."""
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 is unavailable")

    class CaptureLoss(lightning.Callback):
        def __init__(self) -> None:
            self.loss: float | None = None

        def on_train_batch_end(
            self,
            trainer: lightning.Trainer,
            pl_module: lightning.LightningModule,
            outputs: object,
            batch: object,
            batch_idx: int,
        ) -> None:
            del trainer, pl_module, batch, batch_idx
            loss = outputs.get("loss") if isinstance(outputs, Mapping) else outputs
            if not isinstance(loss, torch.Tensor):
                raise TypeError("Toad automatic optimization must return its loss")
            self.loss = float(loss.detach().float().cpu())

    fp32_config = control_fixture_config().model_copy(
        update={
            "runtime": control_fixture_config().runtime.model_copy(
                update={"accelerator": "gpu", "precision": "32-true"}
            )
        }
    )
    bf16_config = fp32_config.model_copy(
        update={
            "runtime": fp32_config.runtime.model_copy(
                update={"precision": "bf16-mixed"}
            )
        }
    )
    torch.manual_seed(1701)
    initial_module = ToadLightningModule(fp32_config)
    initial = {
        name: tensor.detach().cpu().clone()
        for name, tensor in initial_module.policy.state_dict().items()
    }
    batch = control_fixture_batch(load_control_fixture())

    class OneBatch(Dataset[LearnerBatch]):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> LearnerBatch:
            if index != 0:
                raise IndexError(index)
            return batch

    def fit(config: ToadConfig) -> tuple[float, dict[str, torch.Tensor]]:
        module = ToadLightningModule(config)
        module.policy.load_state_dict(initial, strict=True)
        capture = CaptureLoss()
        trainer = lightning.Trainer(
            accelerator="gpu",
            devices=1,
            precision=config.runtime.precision,
            max_steps=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            deterministic=True,
            gradient_clip_val=config.optimizer.clip_grad_norm,
            gradient_clip_algorithm="norm",
            callbacks=[capture],
            use_distributed_sampler=False,
        )
        trainer.fit(module, train_dataloaders=DataLoader(OneBatch(), batch_size=None))
        assert capture.loss is not None
        return capture.loss, {
            name: tensor.detach().cpu().float().clone()
            for name, tensor in module.policy.state_dict().items()
        }

    fp32_loss, fp32_state = fit(fp32_config)
    bf16_loss, bf16_state = fit(bf16_config)

    assert torch.isfinite(torch.tensor([fp32_loss, bf16_loss])).all()
    assert bf16_loss == pytest.approx(fp32_loss, rel=0.05, abs=0.02)
    fp32_delta = torch.cat(
        [(fp32_state[name] - initial[name].float()).flatten() for name in initial]
    )
    bf16_delta = torch.cat(
        [(bf16_state[name] - initial[name].float()).flatten() for name in initial]
    )
    assert torch.isfinite(fp32_delta).all() and torch.isfinite(bf16_delta).all()
    relative_update_drift = float(
        (bf16_delta - fp32_delta).norm() / fp32_delta.norm().clamp_min(1e-12)
    )
    assert relative_update_drift <= 0.15
