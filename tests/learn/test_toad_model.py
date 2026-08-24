"""Tests for the native Toad policy tensor contract."""

import pytest
import torch
from pydantic import ValidationError

from kaggriculture.learn.encoding import MAX_UNITS, SCALARS, TILE_PLANES
from kaggriculture.learn.model import Policy
from kaggriculture.learn.toad.config import ModelConfig, ToadConfig
from kaggriculture.learn.toad.model import PolicyState, StatefulPolicy


def model_inputs(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return deterministic inputs accepted by the control policy."""
    generator = torch.Generator().manual_seed(29)
    board = torch.randn(batch, TILE_PLANES, 10, 10, generator=generator)
    scalars = torch.randn(batch, SCALARS, generator=generator)
    positions = torch.randint(0, 100, (batch, MAX_UNITS), generator=generator)
    return board, scalars, positions


def test_disabled_features_match_the_current_policy() -> None:
    """The permanent control adapter must be tensor-for-tensor identical."""
    torch.manual_seed(11)
    old = Policy(blocks=1, channels=16, value_bound=1.0)
    new = StatefulPolicy(ModelConfig.control(blocks=1, channels=16))
    new.load_control_state_dict(old.state_dict())
    board, scalars, positions = model_inputs()

    old_output = old(board, scalars, positions)
    new_output = new(board, scalars, positions, state=None, dones=None)

    assert torch.equal(new_output.unit_logits, old_output[0])
    assert torch.equal(new_output.quantity_logits, old_output[1])
    assert torch.equal(new_output.market_logits, old_output[2])
    assert torch.equal(new_output.values, old_output[3])
    assert new_output.belief_logits is None
    assert new_output.state is None


def test_control_constructor_disables_every_optional_component() -> None:
    """One named constructor defines the permanent compatibility topology."""
    config = ModelConfig.control()

    assert (config.blocks, config.channels) == (8, 128)
    assert not config.recurrent
    assert not config.transformer
    assert not config.local_patch
    assert not config.belief
    assert not config.interaction_value


def test_initial_state_is_typed_and_detachable() -> None:
    """Enabled state is allocated from the caller's dtype and device."""
    policy = StatefulPolicy(
        ModelConfig.control(blocks=1, channels=16).model_copy(
            update={"recurrent": True, "recurrent_channels": 12, "belief_size": 9}
        )
    )
    like = torch.randn(2, dtype=torch.float64, requires_grad=True)

    state = policy.initial_state(3, like=like)

    assert isinstance(state, PolicyState)
    assert state.hidden.shape == (3, 12, 10, 10)
    assert state.cell.shape == (3, 12, 10, 10)
    assert state.prior_belief.shape == (3, 9)
    assert state.hidden.dtype == like.dtype

    source = torch.randn(2, requires_grad=True)
    tracked = PolicyState(
        hidden=source * 2,
        cell=source.square(),
        prior_belief=source.sigmoid(),
    )
    detached = tracked.detach()
    for tensor in (detached.hidden, detached.cell, detached.prior_belief):
        assert tensor.grad_fn is None
        assert not tensor.requires_grad


def test_loading_control_weights_cannot_initialize_enabled_components() -> None:
    """Bare control weights are never a partial enabled-model warm start."""
    old = Policy(blocks=1, channels=16, value_bound=1.0)
    enabled = StatefulPolicy(
        ModelConfig.control(blocks=1, channels=16).model_copy(
            update={"recurrent": True}
        )
    )

    with pytest.raises(ValueError, match="enabled optional components"):
        enabled.load_control_state_dict(old.state_dict())


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"belief_feedback": True}, "requires the belief head"),
        ({"belief_loss_weight": 0.5}, "requires the belief head"),
        (
            {"transformer": True, "channels": 10, "transformer_heads": 4},
            "divide evenly",
        ),
        ({"local_patch": True, "local_patch_size": 6}, "must be odd"),
        ({"recurrent_layers": 0}, "greater than 0"),
    ],
)
def test_optional_model_dimensions_are_validated(
    override: dict[str, object], message: str
) -> None:
    """Invalid optional paths fail before constructing any torch module."""
    with pytest.raises(ValidationError, match=message):
        ModelConfig.model_validate(override)


@pytest.mark.parametrize(
    "model",
    [
        {"recurrent": True},
        {"transformer": True},
        {"local_patch": True},
        {"belief": True},
        {"belief": True, "belief_loss_weight": 0.5},
        {"belief": True, "belief_feedback": True},
        {"interaction_value": True},
    ],
)
def test_active_trainer_rejects_unimplemented_optional_model_paths(
    model: dict[str, object],
) -> None:
    """Inactive architecture paths cannot be silently ignored by Lightning."""
    with pytest.raises(ValidationError, match="not implemented in the active trainer"):
        ToadConfig.model_validate({"model": model})
