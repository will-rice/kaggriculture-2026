"""Compile ownership and canonical Toad policy export contracts."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch
from pydantic import ValidationError
from torch import nn

from kaggriculture.learn.toad.callbacks import ActorSyncCallback
from kaggriculture.learn.toad.compile import (
    CompileRequestedError,
    maybe_compile,
    policy_state_dict,
    unwrap_compiled,
)
from kaggriculture.learn.toad.config import (
    CompileConfig,
    ToadConfig,
    structural_fingerprint,
    validate_stored_config,
)
from kaggriculture.learn.toad.data import BatchKind, LearnerBatch, segments
from kaggriculture.learn.toad.lightning import ToadLightningModule, compute_loss
from kaggriculture.learn.toad.model import StatefulPolicy
from tests.learn.test_toad_control_fixture import (
    control_fixture_batch,
    control_fixture_config,
    load_control_fixture,
)
from tests.learn.test_toad_data import _recurrent_trajectory


class _FakeCompiled(nn.Module):
    """Torch-compile-shaped wrapper that deliberately prefixes ordinary keys."""

    def __init__(self, original: nn.Module) -> None:
        super().__init__()
        self._orig_mod = original

    def forward(self, *args: object, **kwargs: object) -> object:
        return self._orig_mod(*args, **kwargs)


def _compile_config() -> ToadConfig:
    control = control_fixture_config()
    return control.model_copy(
        update={
            "runtime": control.runtime.model_copy(
                update={
                    "compile": CompileConfig(
                        enabled=True,
                        mode="reduce-overhead",
                        fullgraph=True,
                        dynamic=True,
                    )
                }
            )
        }
    )


def _fake_compile_wrapper(module: nn.Module, **_kwargs: object) -> nn.Module:
    return _FakeCompiled(module)


def test_compile_disabled_returns_the_same_policy() -> None:
    """A disabled flag must preserve eager identity, hooks, and behavior."""
    module = ToadLightningModule(control_fixture_config())

    assert maybe_compile(module.policy, module.config) is module.policy


def test_compiled_policy_unwraps_and_exports_canonical_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A compiler wrapper must never leak its private state-dict prefix."""
    module = ToadLightningModule(_compile_config())
    monkeypatch.setattr(torch, "compile", _fake_compile_wrapper)
    module.policy = maybe_compile(module.policy, module.config)

    assert unwrap_compiled(module.policy).__class__.__name__ == "Policy"
    assert tuple(policy_state_dict(module)) == tuple(
        unwrap_compiled(module.policy).state_dict()
    )
    assert all(not key.startswith("_orig_mod.") for key in policy_state_dict(module))
    assert all("_orig_mod." not in key for key in module.state_dict())


def test_compiled_policy_checkpoint_restores_into_eager_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lightning checkpoint state remains portable after removing compilation."""
    compiled = ToadLightningModule(_compile_config())
    monkeypatch.setattr(torch, "compile", _fake_compile_wrapper)
    compiled.policy = maybe_compile(compiled.policy, compiled.config)
    checkpoint = compiled.state_dict()
    eager = ToadLightningModule(control_fixture_config())

    eager.load_state_dict(checkpoint)

    assert all("_orig_mod." not in key for key in checkpoint)
    assert tuple(eager.policy.state_dict()) == tuple(policy_state_dict(compiled))

    resumed = ToadLightningModule(_compile_config())
    resumed.policy = maybe_compile(resumed.policy, resumed.config)
    resumed.load_state_dict(checkpoint)
    assert tuple(policy_state_dict(resumed)) == tuple(policy_state_dict(compiled))


def test_actor_publication_exports_fake_compiled_policy_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actor workers receive the same canonical key layout as eager policies."""
    module = ToadLightningModule(_compile_config())
    monkeypatch.setattr(torch, "compile", _fake_compile_wrapper)
    module.policy = maybe_compile(module.policy, module.config)
    published: list[dict[str, torch.Tensor]] = []

    class Data:
        def publish_actor(self, state: dict[str, torch.Tensor], _version: int) -> None:
            published.append(state)

    trainer = type("Trainer", (), {"datamodule": Data()})()
    ActorSyncCallback(every_rounds=1).on_fit_start(
        trainer,  # ty: ignore[invalid-argument-type]
        module,
    )

    assert published
    assert tuple(published[0]) == tuple(unwrap_compiled(module.policy).state_dict())


def test_requested_compile_failure_aborts_without_eager_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A requested optimization cannot quietly change the experiment to eager."""
    module = ToadLightningModule(_compile_config())

    def fail_compile(*_args: object, **_kwargs: object) -> nn.Module:
        raise RuntimeError("compiler unavailable")

    monkeypatch.setattr(torch, "compile", fail_compile)

    with pytest.raises(CompileRequestedError, match="compiler unavailable"):
        maybe_compile(module.policy, module.config)


def test_compile_settings_are_immutable_resume_identity() -> None:
    """A resumed run cannot silently change compiler graph semantics."""
    eager = control_fixture_config()
    compiled = _compile_config()

    with pytest.raises(ValidationError):
        compiled.runtime.compile.enabled = False  # ty: ignore[invalid-assignment]
    assert structural_fingerprint(eager) != structural_fingerprint(compiled)


def test_historical_disabled_compile_checkpoint_resumes_as_eager() -> None:
    """The former boolean eager setting remains a valid stored checkpoint."""
    config = control_fixture_config()
    stored = ToadLightningModule(config)
    checkpoint: dict[str, object] = {}
    stored.on_save_checkpoint(checkpoint)
    metadata = cast(dict[str, object], checkpoint["toad"])
    stored_config = cast(dict[str, object], metadata["config"])
    stored_runtime = cast(dict[str, object], stored_config["runtime"])
    stored_runtime["compile"] = False
    resumed = ToadLightningModule(config)

    resumed.on_load_checkpoint(checkpoint)

    assert validate_stored_config(stored_config).runtime.compile == CompileConfig()
    assert resumed.environment_steps == stored.environment_steps


@pytest.mark.parametrize("legacy_value", [True, [], "false", 0])
def test_historical_compile_accepts_only_the_former_false_literal(
    legacy_value: object,
) -> None:
    """Malformed historical compile values cannot accidentally enable a mode."""
    payload = control_fixture_config().model_dump(mode="json")
    runtime = cast(dict[str, object], payload["runtime"])
    runtime["compile"] = legacy_value

    with pytest.raises(ValidationError):
        validate_stored_config(payload)


def _recurrent_fixture() -> tuple[ToadConfig, LearnerBatch]:
    config = ToadConfig.model_validate(
        {
            "model": {
                "blocks": 1,
                "channels": 16,
                "recurrent": True,
                "recurrent_channels": 3,
                "recurrent_layers": 2,
            },
            "optimizer": {"unroll_length": 2, "value_warmup_batches": 0},
            "runtime": {"compile": {"enabled": True}},
        }
    )
    batch = LearnerBatch(
        segments=tuple(segments(_recurrent_trajectory(turns=2), 2)),
        kind=BatchKind.SELFPLAY,
        baseline_only=False,
        first_of_round=True,
        end_of_round=True,
        collected_steps=2,
        round_id=0,
        actor_version=0,
        game_ids=(0,),
        opponent_ids=("self",),
    )
    return config, batch


@pytest.mark.compile
@pytest.mark.filterwarnings("ignore:Can't initialize NVML")
def test_compiled_recurrent_update_matches_eager_and_exports_eager_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compiled learner loss/update matches eager and checkpoint keys stay canonical."""
    config, batch = _recurrent_fixture()
    eager = ToadLightningModule(config)
    compiled = ToadLightningModule(config)
    compiled.load_state_dict(eager.state_dict())
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "1")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    compiled.policy = maybe_compile(compiled.policy, config)
    eager_optimizer = torch.optim.Adam(
        eager.policy.parameters(), lr=config.optimizer.lr
    )
    compiled_optimizer = torch.optim.Adam(
        compiled.policy.parameters(), lr=config.optimizer.lr
    )
    eager_optimizer.zero_grad(set_to_none=True)
    eager_report = eager.compute_report(batch)
    eager_report.total.backward()
    eager_optimizer.step()
    compiled_optimizer.zero_grad(set_to_none=True)
    compiled_report = compiled.compute_report(batch)
    compiled_report.total.backward()
    compiled_optimizer.step()

    assert compiled_report.total.item() == pytest.approx(
        eager_report.total.item(), rel=1e-5, abs=1e-6
    )
    assert compiled_report.debug_dtypes == eager_report.debug_dtypes
    assert all(
        dtype is torch.float32 for dtype in compiled_report.debug_dtypes.values()
    )
    for name, parameter in eager.policy.state_dict().items():
        assert torch.allclose(
            policy_state_dict(compiled)[name],
            parameter,
            rtol=1e-5,
            atol=1e-6,
        )

    checkpoint = compiled.state_dict()
    restored = ToadLightningModule(config)
    restored.load_state_dict(checkpoint)
    assert all("_orig_mod." not in key for key in checkpoint)
    assert policy_state_dict(restored).keys() == policy_state_dict(compiled).keys()

    if torch.amp.autocast_mode.is_autocast_available("cpu"):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            bf16_report = compiled.compute_report(batch)
        assert all(
            dtype is torch.float32 for dtype in bf16_report.debug_dtypes.values()
        )


def test_compiled_control_policy_runs_the_real_learner_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compiling the policy changes the function compute_loss actually calls."""
    config = _compile_config()
    module = ToadLightningModule(config)
    calls = 0

    class RecordingCompiled(_FakeCompiled):
        def forward(self, *args: object, **kwargs: object) -> object:
            nonlocal calls
            calls += 1
            return super().forward(*args, **kwargs)

    monkeypatch.setattr(
        torch, "compile", lambda module, **_kwargs: RecordingCompiled(module)
    )
    module.policy = maybe_compile(module.policy, config)
    report = module.compute_report(control_fixture_batch(load_control_fixture()))

    assert calls == 1
    assert isinstance(report.total, torch.Tensor)


def test_fake_compiled_recurrent_policy_preserves_stateful_learner_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wrapper must not make recurrent batches take the legacy forward branch."""
    config, batch = _recurrent_fixture()
    policy = StatefulPolicy(config.model)
    monkeypatch.setattr(torch, "compile", _fake_compile_wrapper)

    report = compute_loss(maybe_compile(policy, config), batch, config)

    assert torch.isfinite(report.total)
