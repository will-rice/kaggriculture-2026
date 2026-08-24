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
from kaggriculture.learn.toad.model import (
    ConvLSTM,
    InteractionValueHead,
    LocalUnitHead,
    PolicyState,
    SpatialTransformer,
    StatefulPolicy,
    extract_unit_patches,
)


def model_inputs(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return deterministic inputs accepted by the control policy."""
    generator = torch.Generator().manual_seed(29)
    board = torch.randn(batch, TILE_PLANES, BOARD_SIZE, BOARD_SIZE, generator=generator)
    scalars = torch.randn(batch, SCALARS, generator=generator)
    positions = torch.randint(0, 100, (batch, MAX_UNITS), generator=generator)
    return board, scalars, positions


def test_unit_patch_is_centered_on_the_flat_position() -> None:
    """The middle of an odd window must be the indexed board cell."""
    features = torch.arange(BOARD_SIZE**2, dtype=torch.float32).view(
        1, 1, BOARD_SIZE, BOARD_SIZE
    )

    patch = extract_unit_patches(
        features,
        torch.tensor([[5 * BOARD_SIZE + 4]]),
        size=7,
    )

    assert patch.shape == (1, 1, 2, 7, 7)
    assert patch[0, 0, 0, 3, 3] == 54
    assert patch[0, 0, 0, 0, 0] == 21
    assert not patch[0, 0, 1].any()


@pytest.mark.parametrize(
    ("position", "feature_plane", "indicator"),
    [
        (
            0,
            [[0, 0, 0], [0, 0, 1], [0, 10, 11]],
            [[1, 1, 1], [1, 0, 0], [1, 0, 0]],
        ),
        (
            9,
            [[0, 0, 0], [8, 9, 0], [18, 19, 0]],
            [[1, 1, 1], [0, 0, 1], [0, 0, 1]],
        ),
        (
            90,
            [[0, 80, 81], [0, 90, 91], [0, 0, 0]],
            [[1, 0, 0], [1, 0, 0], [1, 1, 1]],
        ),
        (
            99,
            [[88, 89, 0], [98, 99, 0], [0, 0, 0]],
            [[0, 0, 1], [0, 0, 1], [1, 1, 1]],
        ),
    ],
)
def test_corner_patches_have_exact_padding(
    position: int,
    feature_plane: list[list[int]],
    indicator: list[list[int]],
) -> None:
    """Each corner must preserve orientation and mark only off-board cells."""
    features = torch.arange(BOARD_SIZE**2, dtype=torch.float32).view(
        1, 1, BOARD_SIZE, BOARD_SIZE
    )

    patch = extract_unit_patches(features, torch.tensor([[position]]), size=3)

    assert torch.equal(patch[0, 0, 0], torch.tensor(feature_plane, dtype=torch.float32))
    assert torch.equal(patch[0, 0, 1], torch.tensor(indicator, dtype=torch.float32))


def test_padding_indicator_is_independent_of_zero_features() -> None:
    """A real zero tile must remain distinguishable from an off-board zero."""
    features = torch.zeros(1, 2, BOARD_SIZE, BOARD_SIZE)

    edge = extract_unit_patches(features, torch.tensor([[0]]), size=7)
    center = extract_unit_patches(features, torch.tensor([[55]]), size=7)

    assert not edge[0, 0, :-1].any()
    assert edge[0, 0, -1, :3].all()
    assert edge[0, 0, -1, 3:, 3:].sum() == 0
    assert not center[0, 0, -1].any()


def test_patch_extraction_batches_multiple_units_without_cross_talk() -> None:
    """Every batch/unit pair must gather its own integer-indexed center."""
    features = torch.stack(
        (
            torch.arange(BOARD_SIZE**2).view(1, BOARD_SIZE, BOARD_SIZE),
            torch.arange(BOARD_SIZE**2, 2 * BOARD_SIZE**2).view(
                1, BOARD_SIZE, BOARD_SIZE
            ),
        )
    ).float()
    positions = torch.tensor([[0, 54, 99], [99, 45, 0]])

    patches = extract_unit_patches(features, positions, size=9)

    assert patches.shape == (2, 3, 2, 9, 9)
    assert torch.equal(
        patches[:, :, 0, 4, 4], torch.tensor([[0, 54, 99], [199, 145, 100]])
    )


@pytest.mark.parametrize(
    ("positions", "message"),
    [
        (torch.zeros(1, MAX_UNITS, 1, dtype=torch.int64), "rank 2"),
        (torch.zeros(2, MAX_UNITS, dtype=torch.int64), "batch dimension"),
        (torch.zeros(1, MAX_UNITS, dtype=torch.float32), "integer dtype"),
        (torch.tensor([[-1]]), "between 0 and 99"),
        (torch.tensor([[100]]), "between 0 and 99"),
    ],
)
def test_patch_extraction_rejects_invalid_positions(
    positions: torch.Tensor, message: str
) -> None:
    """Malformed gather indices must fail at the public helper boundary."""
    features = torch.zeros(1, 1, BOARD_SIZE, BOARD_SIZE)

    with pytest.raises(ValueError, match=message):
        extract_unit_patches(features, positions, size=7)


@pytest.mark.parametrize(
    ("size", "message"),
    [
        (7.0, "integer"),
        (0, "positive"),
        (2, "odd"),
        (11, "at most 9"),
    ],
)
def test_patch_extraction_rejects_unsupported_sizes(
    size: int | float, message: str
) -> None:
    """Only positive odd windows within the declared board ceiling are valid."""
    features = torch.zeros(1, 1, BOARD_SIZE, BOARD_SIZE)

    with pytest.raises(ValueError, match=message):
        extract_unit_patches(features, torch.zeros(1, 1, dtype=torch.int64), size)


def test_local_unit_head_outputs_encoding_derived_dimensions() -> None:
    """The two decisions retain their established independent option spaces."""
    head = LocalUnitHead(channels=8, blocks=1, patch_size=7)
    features = torch.randn(2, 8, BOARD_SIZE, BOARD_SIZE)
    positions = torch.tensor([[0, 54, 99], [99, 45, 0]])

    unit_logits, quantity_logits = head(features, positions)

    assert unit_logits.shape == (2, 3, len(UNIT_OPS))
    assert quantity_logits.shape == (2, 3, len(QUANTITIES))


def test_local_operation_and_quantity_projections_are_independent() -> None:
    """Changing either final projection must leave the other output untouched."""
    torch.manual_seed(53)
    head = LocalUnitHead(channels=8, blocks=0, patch_size=7)
    features = torch.randn(1, 8, BOARD_SIZE, BOARD_SIZE)
    positions = torch.tensor([[0, 54]])
    baseline_units, baseline_quantities = head(features, positions)

    with torch.no_grad():
        head.operation.bias.add_(1.0)
    changed_units, unchanged_quantities = head(features, positions)
    with torch.no_grad():
        head.quantity.bias.add_(2.0)
    unchanged_units, changed_quantities = head(features, positions)

    assert not torch.equal(changed_units, baseline_units)
    assert torch.equal(unchanged_quantities, baseline_quantities)
    assert torch.equal(unchanged_units, changed_units)
    assert not torch.equal(changed_quantities, unchanged_quantities)


def test_local_residual_preprocessing_receives_finite_gradients() -> None:
    """Both shared preprocessing and every local residual parameter must train."""
    head = LocalUnitHead(channels=8, blocks=2, patch_size=7)
    features = torch.randn(2, 8, BOARD_SIZE, BOARD_SIZE, requires_grad=True)
    positions = torch.tensor([[0, 54, 99], [99, 45, 0]])

    unit_logits, quantity_logits = head(features, positions)
    (unit_logits.square().mean() + quantity_logits.square().mean()).backward()

    shared_parameters = [
        parameter
        for name, parameter in head.named_parameters()
        if name.startswith(("preprocess.", "blocks."))
    ]
    assert shared_parameters
    assert all(parameter.grad is not None for parameter in shared_parameters)
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in shared_parameters
        if parameter.grad is not None
    )
    assert features.grad is not None
    assert torch.count_nonzero(features.grad)


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


def test_spatial_transformer_preserves_shape_and_trains_explicit_positions() -> None:
    """Removing spatial positions or either residual branch must break gradients."""
    layer = SpatialTransformer(channels=16, blocks=2, heads=4, mlp_ratio=2)
    features = torch.randn(3, 16, BOARD_SIZE, BOARD_SIZE, requires_grad=True)

    output = layer(features)
    output.square().mean().backward()

    assert output.shape == features.shape
    assert layer.position.shape == (1, BOARD_SIZE * BOARD_SIZE, 16)
    assert layer.position.grad is not None
    assert torch.count_nonzero(layer.position.grad)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in layer.parameters()
    )


def test_interaction_value_attends_to_remote_spatial_and_global_tokens() -> None:
    """The new head itself must read all 100 cells and projected global context."""
    torch.manual_seed(37)
    head = InteractionValueHead(
        channels=16,
        global_features=SCALARS,
        heads=4,
        mlp_ratio=2,
        value_bound=0.75,
    )
    features = torch.randn(2, 16, BOARD_SIZE, BOARD_SIZE, requires_grad=True)
    scalars = torch.randn(2, SCALARS, requires_grad=True)
    attended_shapes: list[tuple[torch.Size, torch.Size]] = []

    def capture_attention_inputs(
        _module: torch.nn.Module, args: tuple[torch.Tensor, ...]
    ) -> None:
        attended_shapes.append((args[0].shape, args[1].shape))

    hook = head.attention.register_forward_pre_hook(capture_attention_inputs)
    values = head(features, scalars)
    hook.remove()
    values.sum().backward()

    assert values.shape == (2,)
    assert torch.all(values >= -0.75)
    assert torch.all(values <= 0.75)
    assert attended_shapes == [
        (
            torch.Size((2, 1, 16)),
            torch.Size((2, BOARD_SIZE * BOARD_SIZE + 1, 16)),
        )
    ]
    assert features.grad is not None
    assert torch.count_nonzero(features.grad[:, :, -1, -1])
    assert scalars.grad is not None
    assert torch.count_nonzero(scalars.grad)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in head.parameters()
    )


def test_spatial_transformer_rejects_single_channel_layer_norm() -> None:
    """Direct spatial construction cannot create width-one token normalization."""
    with pytest.raises(ValueError, match="at least 2"):
        SpatialTransformer(channels=1, blocks=1, heads=1, mlp_ratio=2)


def test_interaction_value_rejects_single_channel_layer_norm() -> None:
    """Direct value-head construction cannot erase every context token."""
    with pytest.raises(ValueError, match="at least 2"):
        InteractionValueHead(
            channels=1,
            global_features=SCALARS,
            heads=1,
            mlp_ratio=2,
            value_bound=None,
        )


def test_width_two_interaction_responds_to_isolated_spatial_and_global_changes() -> (
    None
):
    """The smallest valid head retains both token sources after normalization."""
    head = InteractionValueHead(
        channels=2,
        global_features=SCALARS,
        heads=1,
        mlp_ratio=2,
        value_bound=None,
    )
    with torch.no_grad():
        head.value_token.copy_(torch.tensor([[[1.0, -1.0]]]))
        head.global_projection.weight.zero_()
        head.global_projection.weight[:, 0] = torch.tensor([1.0, -1.0])
        head.global_projection.bias.zero_()
        head.attention.in_proj_weight.zero_()
        identity = torch.eye(2)
        head.attention.in_proj_weight[:2].copy_(identity)
        head.attention.in_proj_weight[2:4].copy_(identity)
        head.attention.in_proj_weight[4:].copy_(identity)
        head.attention.in_proj_bias.zero_()
        head.attention.out_proj.weight.copy_(identity)
        head.attention.out_proj.bias.zero_()
        for parameter in head.mlp.parameters():
            parameter.zero_()
        head.value_projection.weight.copy_(torch.tensor([[1.0, 0.0]]))
        head.value_projection.bias.zero_()

    spatial = torch.zeros(1, 2, BOARD_SIZE, BOARD_SIZE)
    global_features = torch.zeros(1, SCALARS)
    remote = spatial.clone()
    remote[0, :, -1, -1] = torch.tensor([1.0, -1.0])
    changed_global = global_features.clone()
    changed_global[0, 0] = 1.0

    baseline = head(spatial, global_features)
    remote_value = head(remote, global_features)
    global_value = head(spatial, changed_global)

    assert remote_value.item() > baseline.item()
    assert global_value.item() > baseline.item()


def test_width_two_single_head_attention_config_is_valid() -> None:
    """The new lower bound must preserve the smallest nondegenerate topology."""
    config = ModelConfig.model_validate(
        {
            "channels": 2,
            "transformer": True,
            "transformer_blocks": 1,
            "transformer_heads": 1,
            "interaction_value": True,
        }
    )

    assert config.channels == 2
    assert config.transformer_heads == 1


@pytest.mark.parametrize(
    ("recurrent", "belief", "transformer", "local_patch", "interaction_value"),
    [
        (False, False, False, True, False),
        (False, False, True, False, False),
        (False, False, False, False, True),
        (False, True, True, True, True),
        (True, False, True, True, True),
        (True, True, True, True, True),
    ],
)
def test_optional_model_combinations_preserve_time_major_shapes(
    recurrent: bool,
    belief: bool,
    transformer: bool,
    local_patch: bool,
    interaction_value: bool,
) -> None:
    """Every supported optional composition must retain time and batch axes."""
    config = ModelConfig.model_validate(
        {
            "blocks": 1,
            "channels": 16,
            "recurrent": recurrent,
            "recurrent_channels": 4,
            "belief": belief,
            "transformer": transformer,
            "transformer_blocks": 1 if transformer else 0,
            "local_patch": local_patch,
            "local_patch_blocks": 1,
            "interaction_value": interaction_value,
        }
    )
    policy = StatefulPolicy(config)
    board, scalars, positions = recurrent_inputs(time=2, batch=1)

    output = policy(board, scalars, positions)

    assert output.unit_logits.shape == (2, 1, MAX_UNITS, len(UNIT_OPS))
    assert output.quantity_logits.shape == (2, 1, MAX_UNITS, len(QUANTITIES))
    assert output.market_logits.shape == (
        2,
        1,
        len(MARKET_SLOTS) + 2,
        len(QUANTITIES),
    )
    assert output.values.shape == (2, 1)
    assert (output.belief_logits is not None) is belief
    assert (output.state is not None) is (recurrent or belief)


def test_combined_attention_policy_trains_every_enabled_component() -> None:
    """Every optional component receives finite gradients in one composition."""
    config = ModelConfig.model_validate(
        {
            "blocks": 1,
            "channels": 16,
            "recurrent": True,
            "recurrent_channels": 4,
            "belief": True,
            "belief_feedback": True,
            "transformer": True,
            "transformer_blocks": 1,
            "local_patch": True,
            "local_patch_blocks": 1,
            "interaction_value": True,
        }
    )
    policy = StatefulPolicy(config)
    board, scalars, positions = recurrent_inputs(time=3, batch=2)

    output = policy(board, scalars, positions)
    assert output.belief_logits is not None
    loss = (
        output.unit_logits.square().mean()
        + output.quantity_logits.square().mean()
        + output.market_logits.square().mean()
        + output.values.square().mean()
        + output.belief_logits.square().mean()
    )
    loss.backward()

    for prefix in (
        "recurrent.",
        "transformer.",
        "belief_head.",
        "local_head.",
        "interaction_value.",
    ):
        gradients = [
            parameter.grad
            for name, parameter in policy.named_parameters()
            if name.startswith(prefix)
        ]
        assert gradients
        assert all(gradient is not None for gradient in gradients)
        assert all(
            torch.isfinite(gradient).all()
            for gradient in gradients
            if gradient is not None
        )


def test_local_patch_changes_only_unit_readouts_from_the_control_paths() -> None:
    """Market and control-value heads must remain byte-identical when local is on."""
    torch.manual_seed(71)
    control = Policy(blocks=1, channels=16, value_bound=1.0)
    config = ModelConfig.control(blocks=1, channels=16).model_copy(
        update={"local_patch": True, "local_patch_blocks": 1}
    )
    local = StatefulPolicy(config)
    local.control.load_state_dict(control.state_dict())
    board, scalars, positions = recurrent_inputs(time=2, batch=2)

    expected = control(
        board.flatten(0, 1), scalars.flatten(0, 1), positions.flatten(0, 1)
    )
    output = local(board, scalars, positions)

    torch.testing.assert_close(
        output.market_logits.flatten(0, 1), expected[2], atol=2e-7, rtol=0
    )
    assert torch.equal(output.values.flatten(), expected[3])


def test_local_patch_receives_post_transformer_features() -> None:
    """Patch extraction must follow, rather than bypass, spatial attention."""
    config = ModelConfig.model_validate(
        {
            "blocks": 1,
            "channels": 16,
            "transformer": True,
            "transformer_blocks": 1,
            "local_patch": True,
            "local_patch_blocks": 1,
        }
    )
    policy = StatefulPolicy(config)
    board, scalars, positions = recurrent_inputs(time=2, batch=1)
    transformed: list[torch.Tensor] = []
    local_inputs: list[torch.Tensor] = []

    transformer_hook = policy.transformer.register_forward_hook(
        lambda _module, _args, output: transformed.append(output.detach())
    )
    local_hook = policy.local_head.register_forward_pre_hook(
        lambda _module, args: local_inputs.append(args[0].detach())
    )
    policy(board, scalars, positions)
    transformer_hook.remove()
    local_hook.remove()

    assert torch.equal(torch.stack(transformed).flatten(0, 1), local_inputs[0])


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
            {
                "transformer": True,
                "transformer_blocks": 1,
                "channels": 10,
                "transformer_heads": 4,
            },
            "divide evenly",
        ),
        (
            {"interaction_value": True, "channels": 10, "transformer_heads": 4},
            "divide evenly",
        ),
        (
            {"transformer": True, "transformer_blocks": 0},
            "positive transformer_blocks",
        ),
        (
            {
                "transformer": True,
                "transformer_blocks": 1,
                "channels": 1,
                "transformer_heads": 1,
            },
            "at least 2",
        ),
        (
            {
                "interaction_value": True,
                "channels": 1,
                "transformer_heads": 1,
            },
            "at least 2",
        ),
        ({"local_patch": True, "local_patch_size": 6}, "must be odd"),
        ({"local_patch": True, "local_patch_size": 11}, "at most 9"),
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


def test_active_trainer_accepts_every_implemented_optional_model_path() -> None:
    """Task 6 relaxes the final architecture gate only after local integration."""
    config = ToadConfig.model_validate(
        {
            "model": {
                "recurrent": True,
                "belief": True,
                "belief_feedback": True,
                "transformer": True,
                "transformer_blocks": 1,
                "local_patch": True,
                "local_patch_blocks": 1,
                "interaction_value": True,
            }
        }
    )

    assert config.model.recurrent
    assert config.model.belief
    assert config.model.transformer
    assert config.model.local_patch
    assert config.model.interaction_value
