"""Tests for the native Toad policy tensor contract."""

import pytest
import torch
from pydantic import ValidationError

from kaggriculture.constants import BOARD_SIZE
from kaggriculture.learn.encoding import (
    MARKET_SLOTS,
    MAX_UNITS,
    QUANTITIES,
    SCALARS,
    TILE_PLANES,
    UNIT_OPS,
)
from kaggriculture.learn.model import Policy
from kaggriculture.learn.toad.config import ModelConfig, ToadConfig
from kaggriculture.learn.toad.model import ConvLSTM, PolicyState, StatefulPolicy


def model_inputs(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return deterministic inputs accepted by the control policy."""
    generator = torch.Generator().manual_seed(29)
    board = torch.randn(batch, TILE_PLANES, BOARD_SIZE, BOARD_SIZE, generator=generator)
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


def test_terminal_reset_erases_previous_episode_memory_per_batch_row() -> None:
    """A terminal resets only that row before its next observation is processed."""
    torch.manual_seed(3)
    layer = ConvLSTM(input_channels=8, hidden_channels=8, kernel_size=3)
    first = torch.randn(3, 2, 8, BOARD_SIZE, BOARD_SIZE)
    second = torch.randn(2, 2, 8, BOARD_SIZE, BOARD_SIZE)
    carried = layer(
        first,
        None,
        torch.zeros(3, 2, dtype=torch.bool),
    ).state

    reset = torch.tensor([[True, False], [False, False]])
    output = layer(second, carried, reset)
    fresh = layer(second[:, :1], None, reset[:, :1])

    assert torch.allclose(
        output.hidden_sequence[:, :1],
        fresh.hidden_sequence,
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        output.cell_sequence[:, :1],
        fresh.cell_sequence,
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        output.state.hidden[:, :1], fresh.state.hidden, atol=1e-6, rtol=1e-6
    )
    assert torch.allclose(
        output.state.cell[:, :1], fresh.state.cell, atol=1e-6, rtol=1e-6
    )


def test_two_chunks_equal_one_sequence_without_a_terminal() -> None:
    """Carrying state across an unroll boundary preserves the full sequence."""
    torch.manual_seed(5)
    layer = ConvLSTM(input_channels=8, hidden_channels=8, kernel_size=3)
    first = torch.randn(3, 2, 8, BOARD_SIZE, BOARD_SIZE)
    second = torch.randn(2, 2, 8, BOARD_SIZE, BOARD_SIZE)
    dones = torch.zeros(5, 2, dtype=torch.bool)

    full = layer(torch.cat((first, second)), None, dones)
    left = layer(first, None, dones[: len(first)])
    right = layer(second, left.state, dones[len(first) :])

    assert torch.allclose(
        torch.cat((left.hidden_sequence, right.hidden_sequence)),
        full.hidden_sequence,
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(right.state.hidden, full.state.hidden, atol=1e-6, rtol=1e-6)
    assert torch.allclose(right.state.cell, full.state.cell, atol=1e-6, rtol=1e-6)


def test_convlstm_multi_layer_state_uses_one_state_per_layer() -> None:
    """Each configured ConvLSTM layer retains a separate hidden and cell map."""
    layer = ConvLSTM(
        input_channels=6,
        hidden_channels=4,
        kernel_size=3,
        layers=2,
    )
    inputs = torch.randn(4, 3, 6, BOARD_SIZE, BOARD_SIZE)

    output = layer(inputs, None, torch.zeros(4, 3, dtype=torch.bool))

    assert output.hidden_sequence.shape == (4, 3, 4, BOARD_SIZE, BOARD_SIZE)
    assert output.cell_sequence.shape == (4, 3, 4, BOARD_SIZE, BOARD_SIZE)
    assert output.state.hidden.shape == (2, 3, 4, BOARD_SIZE, BOARD_SIZE)
    assert output.state.cell.shape == (2, 3, 4, BOARD_SIZE, BOARD_SIZE)


def recurrent_policy() -> StatefulPolicy:
    """Build the smallest enabled policy used by sequence-contract tests."""
    return StatefulPolicy(
        ModelConfig.control(blocks=1, channels=16).model_copy(
            update={
                "recurrent": True,
                "recurrent_channels": 12,
                "recurrent_layers": 2,
            }
        )
    )


def recurrent_inputs(
    time: int = 3, batch: int = 2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return time-major inputs accepted by the enabled stateful policy."""
    generator = torch.Generator().manual_seed(47)
    board = torch.randn(
        time, batch, TILE_PLANES, BOARD_SIZE, BOARD_SIZE, generator=generator
    )
    scalars = torch.randn(time, batch, SCALARS, generator=generator)
    positions = torch.randint(0, 100, (time, batch, MAX_UNITS), generator=generator)
    return board, scalars, positions


def test_recurrent_policy_accepts_time_major_inputs_and_keeps_layered_state() -> None:
    """The enabled path preserves time and batch axes through every output head."""
    policy = recurrent_policy()
    board, scalars, positions = recurrent_inputs()

    output = policy(
        board, scalars, positions, dones=torch.zeros(3, 2, dtype=torch.bool)
    )

    assert output.unit_logits.shape == (3, 2, MAX_UNITS, len(UNIT_OPS))
    assert output.quantity_logits.shape == (3, 2, MAX_UNITS, len(QUANTITIES))
    assert output.market_logits.shape == (
        3,
        2,
        len(MARKET_SLOTS) + 2,
        len(QUANTITIES),
    )
    assert output.values.shape == (3, 2)
    assert output.state is not None
    assert output.state.hidden.shape == (2, 2, 12, BOARD_SIZE, BOARD_SIZE)
    assert output.state.cell.shape == (2, 2, 12, BOARD_SIZE, BOARD_SIZE)


def test_recurrent_policy_chunks_match_one_full_time_major_sequence() -> None:
    """Policy heads and final state are invariant to a nonterminal chunk boundary."""
    torch.manual_seed(13)
    policy = recurrent_policy()
    board, scalars, positions = recurrent_inputs(time=5)
    dones = torch.zeros(5, 2, dtype=torch.bool)

    full = policy(board, scalars, positions, dones=dones)
    left = policy(board[:3], scalars[:3], positions[:3], dones=dones[:3])
    right = policy(
        board[3:], scalars[3:], positions[3:], state=left.state, dones=dones[3:]
    )

    assert torch.allclose(
        torch.cat((left.unit_logits, right.unit_logits)),
        full.unit_logits,
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(right.state.hidden, full.state.hidden, atol=1e-6, rtol=1e-6)
    assert torch.allclose(right.state.cell, full.state.cell, atol=1e-6, rtol=1e-6)


def test_recurrent_policy_terminal_row_matches_a_fresh_policy_row() -> None:
    """The wrapper carries no hidden or cell memory across one row's terminal."""
    torch.manual_seed(23)
    policy = recurrent_policy()
    first_board, first_scalars, first_positions = recurrent_inputs(time=2)
    carried = policy(
        first_board,
        first_scalars,
        first_positions,
        dones=torch.zeros(2, 2, dtype=torch.bool),
    ).state
    assert carried is not None
    second_board, second_scalars, second_positions = recurrent_inputs(time=2)
    dones = torch.tensor([[True, False], [False, False]])

    output = policy(
        second_board,
        second_scalars,
        second_positions,
        state=carried,
        dones=dones,
    )
    fresh = policy(
        second_board[:, :1],
        second_scalars[:, :1],
        second_positions[:, :1],
        dones=dones[:, :1],
    )

    assert torch.allclose(
        output.unit_logits[:, :1], fresh.unit_logits, atol=1e-6, rtol=1e-6
    )
    assert output.state is not None
    assert fresh.state is not None
    assert torch.allclose(
        output.state.hidden[:, :1], fresh.state.hidden, atol=1e-6, rtol=1e-6
    )
    assert torch.allclose(
        output.state.cell[:, :1], fresh.state.cell, atol=1e-6, rtol=1e-6
    )


def test_recurrent_policy_resets_prior_belief_at_the_terminal_boundary() -> None:
    """Later belief feedback's state slot follows recurrent resets now."""
    policy = recurrent_policy()
    board, scalars, positions = recurrent_inputs(time=3)
    state = policy.initial_state(2, like=board)
    assert state is not None
    state = PolicyState(
        hidden=state.hidden,
        cell=state.cell,
        prior_belief=torch.tensor([[1.0] * 9, [2.0] * 9]),
    )

    output = policy(
        board,
        scalars,
        positions,
        state=state,
        dones=torch.tensor([[False, False], [True, False], [False, False]]),
    )

    assert output.state is not None
    assert torch.equal(output.state.prior_belief[0], torch.zeros(9))
    assert torch.equal(output.state.prior_belief[1], torch.full((9,), 2.0))


def test_policy_state_reset_clears_all_terminal_components_before_a_step() -> None:
    """One reset operation gives later belief feedback the recurrent boundary."""
    state = PolicyState(
        hidden=torch.ones(2, 3, BOARD_SIZE, BOARD_SIZE),
        cell=torch.full((2, 3, BOARD_SIZE, BOARD_SIZE), 2.0),
        prior_belief=torch.full((2, 9), 3.0),
    )

    reset = state.reset_rows(torch.tensor([True, False]))

    assert torch.equal(reset.hidden[0], torch.zeros(3, BOARD_SIZE, BOARD_SIZE))
    assert torch.equal(reset.cell[0], torch.zeros(3, BOARD_SIZE, BOARD_SIZE))
    assert torch.equal(reset.prior_belief[0], torch.zeros(9))
    assert torch.equal(reset.hidden[1], torch.ones(3, BOARD_SIZE, BOARD_SIZE))
    assert torch.equal(reset.cell[1], torch.full((3, BOARD_SIZE, BOARD_SIZE), 2.0))
    assert torch.equal(reset.prior_belief[1], torch.full((9,), 3.0))


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
        (
            {"belief": True, "belief_size": 9},
            "fixed private-state target schema",
        ),
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
        {"transformer": True},
        {"local_patch": True},
        {"interaction_value": True},
    ],
)
def test_active_trainer_rejects_unimplemented_optional_model_paths(
    model: dict[str, object],
) -> None:
    """Inactive architecture paths cannot be silently ignored by Lightning."""
    with pytest.raises(ValidationError, match="not implemented in the active trainer"):
        ToadConfig.model_validate({"model": model})


def test_active_trainer_accepts_recurrent_and_belief_but_keeps_later_paths_gated() -> (
    None
):
    """Task 4 relaxes belief only; later architecture stages remain errors."""
    config = ToadConfig.model_validate(
        {"model": {"recurrent": True, "belief": True, "belief_feedback": True}}
    )

    assert config.model.recurrent
    assert config.model.belief
    with pytest.raises(ValidationError, match="not implemented in the active trainer"):
        ToadConfig.model_validate({"model": {"recurrent": True, "transformer": True}})
