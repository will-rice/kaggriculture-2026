"""Compile ownership and canonical Toad policy export contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from pydantic import ValidationError
from torch import nn

from kaggriculture.learn.encoding import MAX_UNITS, SCALARS, TILE_PLANES
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
            "optimizer": {"value_warmup_batches": 0},
            "runtime": {"compile": {"enabled": True}},
        }
    )
    batch = LearnerBatch(
        segments=tuple(segments(_recurrent_trajectory(), 16)),
        kind=BatchKind.SELFPLAY,
        baseline_only=False,
        first_of_round=True,
        end_of_round=True,
        collected_steps=32,
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
    """The compiled active recurrent forward updates like eager and saves cleanly."""
    config, _ = _recurrent_fixture()
    eager = StatefulPolicy(config.model)
    compiled = StatefulPolicy(config.model)
    compiled.load_state_dict(eager.state_dict())
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "1")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    compiled = maybe_compile(compiled, config)
    eager_optimizer = torch.optim.Adam(eager.parameters(), lr=config.optimizer.lr)
    compiled_optimizer = torch.optim.Adam(compiled.parameters(), lr=config.optimizer.lr)
    board = torch.zeros(2, 1, TILE_PLANES, 10, 10)
    scalars = torch.zeros(2, 1, SCALARS)
    positions = torch.zeros(2, 1, MAX_UNITS, dtype=torch.int64)
    dones = torch.tensor([[False], [True]])

    eager_output = eager(
        board, scalars, positions, state=eager.initial_state(1, like=board), dones=dones
    )
    eager_loss = (
        eager_output.unit_logits.sum()
        + eager_output.quantity_logits.sum()
        + eager_output.market_logits.sum()
        + eager_output.values.sum()
    )
    eager_loss.backward()
    eager_optimizer.step()
    compiled_stateful = unwrap_compiled(compiled)
    assert isinstance(compiled_stateful, StatefulPolicy)
    compiled_output = compiled(
        board,
        scalars,
        positions,
        state=compiled_stateful.initial_state(1, like=board),
        dones=dones,
    )
    assert hasattr(compiled_output, "unit_logits")
    compiled_loss = (
        compiled_output.unit_logits.sum()
        + compiled_output.quantity_logits.sum()
        + compiled_output.market_logits.sum()
        + compiled_output.values.sum()
    )
    compiled_loss.backward()
    compiled_optimizer.step()

    assert compiled_loss.item() == pytest.approx(eager_loss.item(), rel=1e-5, abs=1e-6)
    for name, parameter in eager.state_dict().items():
        assert torch.allclose(
            unwrap_compiled(compiled).state_dict()[name],
            parameter,
            rtol=1e-5,
            atol=1e-6,
        )

    path = tmp_path / "policy.pt"
    torch.save(policy_state_dict(compiled), path)
    restored = StatefulPolicy(config.model)
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert all(not key.startswith("_orig_mod.") for key in policy_state_dict(compiled))


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
