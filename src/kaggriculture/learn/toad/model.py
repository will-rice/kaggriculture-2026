"""Typed stateful policy contract for the native Toad learner.

The optional architecture implemented in this module is adapted from Tony
Kozlovsky's MIT-licensed ``tonykozlovsky/lux-ai3-pub`` repository at commit
``8001d70d939c78725d198a70586e8ba77efa2a24``, specifically
``final_versions/08_03_tune_against_mask_cont/lux_ai/nns/models.py`` and
``final_versions/08_03_tune_against_mask_cont/lux_ai/nns/transformer.py``.
See ``THIRD_PARTY_NOTICES.md`` for the complete upstream notice.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from kaggriculture.constants import BOARD_SIZE
from kaggriculture.learn.encoding import MARKET_SLOTS, QUANTITIES, SCALARS, UNIT_OPS
from kaggriculture.learn.model import Policy, Residual
from kaggriculture.learn.toad.config import MAX_LOCAL_PATCH_SIZE, ModelConfig


def _validate_local_patch_size(size: int) -> None:
    """Validate one public or configured local window size."""
    if isinstance(size, bool) or not isinstance(size, int):
        raise ValueError("local patch size must be an integer")
    if size <= 0:
        raise ValueError("local patch size must be positive")
    if size % 2 == 0:
        raise ValueError("local patch size must be odd")
    if size > MAX_LOCAL_PATCH_SIZE:
        raise ValueError(f"local patch size must be at most {MAX_LOCAL_PATCH_SIZE}")


def _validate_patch_inputs(features: torch.Tensor, positions: torch.Tensor) -> None:
    """Validate feature-map and integer-position tensor schemas."""
    if features.ndim != 4:
        raise ValueError("local patch features must have rank 4")
    batch, _channels, height, width = features.shape
    if (height, width) != (BOARD_SIZE, BOARD_SIZE):
        raise ValueError(
            "local patch features require a "
            f"{BOARD_SIZE}x{BOARD_SIZE} map, got {height}x{width}"
        )
    if positions.ndim != 2:
        raise ValueError("local patch positions must have rank 2")
    if positions.shape[0] != batch:
        raise ValueError("local patch positions batch dimension must match features")
    integer_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64)
    if positions.dtype not in integer_dtypes:
        raise ValueError("local patch positions must have an integer dtype")
    if positions.device != features.device:
        raise ValueError("local patch positions must be on the features device")
    if positions.numel() and (
        torch.any(positions < 0) or torch.any(positions >= BOARD_SIZE * BOARD_SIZE)
    ):
        raise ValueError(
            f"local patch positions must be between 0 and {BOARD_SIZE * BOARD_SIZE - 1}"
        )


def extract_unit_patches(
    features: torch.Tensor, positions: torch.Tensor, size: int
) -> torch.Tensor:
    """Gather centered odd local windows with an out-of-bounds indicator plane."""
    _validate_local_patch_size(size)
    _validate_patch_inputs(features, positions)
    batch, channels, _height, _width = features.shape

    radius = size // 2
    spatial = torch.nn.functional.pad(features, (radius,) * 4)
    indicator = torch.nn.functional.pad(
        features.new_zeros(batch, 1, BOARD_SIZE, BOARD_SIZE),
        (radius,) * 4,
        value=1,
    )
    padded = torch.cat((spatial, indicator), dim=1)
    windows = torch.nn.functional.unfold(padded, kernel_size=size)
    indices = positions.to(dtype=torch.int64)[:, None, :].expand(
        -1, windows.shape[1], -1
    )
    selected = windows.gather(2, indices).transpose(1, 2)
    return selected.reshape(batch, positions.shape[1], channels + 1, size, size)


class LocalUnitHead(torch.nn.Module):
    """Shared residual patch processing with independent unit decision heads."""

    def __init__(self, channels: int, blocks: int, patch_size: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("local patch channels must be positive")
        if blocks < 0:
            raise ValueError("local patch blocks must be nonnegative")
        _validate_local_patch_size(patch_size)
        self.channels = channels
        self.patch_size = patch_size
        self.preprocess = torch.nn.Sequential(
            torch.nn.Conv2d(channels + 1, channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        self.blocks = torch.nn.ModuleList(Residual(channels) for _ in range(blocks))
        self.operation = torch.nn.Linear(channels, len(UNIT_OPS))
        self.quantity = torch.nn.Linear(channels, len(QUANTITIES))

    def forward(
        self, features: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return operation and quantity logits for every supplied unit slot."""
        if features.ndim == 4 and features.shape[1] != self.channels:
            raise ValueError(
                f"local unit head expected {self.channels} feature channels, "
                f"got {features.shape[1]}"
            )
        patches = extract_unit_patches(features, positions, self.patch_size)
        batch, units = patches.shape[:2]
        local = self.preprocess(patches.flatten(0, 1))
        for block in self.blocks:
            local = block(local)
        pooled = local.mean(dim=(-2, -1)).view(batch, units, -1)
        return self.operation(pooled), self.quantity(pooled)


@dataclass(frozen=True)
class PolicyState:
    """Tensor state carried between consecutive policy observations."""

    hidden: torch.Tensor
    cell: torch.Tensor
    prior_belief: torch.Tensor

    def detach(self) -> "PolicyState":
        """Return the same state detached at a learner segment boundary."""
        return PolicyState(
            hidden=self.hidden.detach(),
            cell=self.cell.detach(),
            prior_belief=self.prior_belief.detach(),
        )

    def reset_rows(self, dones: torch.Tensor) -> "PolicyState":
        """Clear terminal rows before their next observation is consumed."""
        return PolicyState(
            hidden=_reset_spatial_rows(self.hidden, dones),
            cell=_reset_spatial_rows(self.cell, dones),
            prior_belief=self.prior_belief
            * (~dones)
            .to(device=self.prior_belief.device, dtype=self.prior_belief.dtype)
            .view(-1, 1),
        )


@dataclass(frozen=True)
class ConvLSTMState:
    """Hidden and cell maps for every ConvLSTM layer."""

    hidden: torch.Tensor
    cell: torch.Tensor

    def reset_rows(self, dones: torch.Tensor) -> "ConvLSTMState":
        """Clear terminal hidden and cell rows before an LSTM update."""
        return ConvLSTMState(
            hidden=_reset_spatial_rows(self.hidden, dones),
            cell=_reset_spatial_rows(self.cell, dones),
        )


@dataclass(frozen=True)
class ConvLSTMOutput:
    """All recurrent activations plus the state after the final time row."""

    hidden_sequence: torch.Tensor
    cell_sequence: torch.Tensor
    state: ConvLSTMState


class ConvLSTMCell(torch.nn.Module):
    """One spatial LSTM update preserving the input's board extent."""

    def __init__(
        self, input_channels: int, hidden_channels: int, kernel_size: int
    ) -> None:
        """Build the four jointly-projected LSTM gates."""
        super().__init__()
        self.gates = torch.nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=kernel_size // 2,
        )

    def forward(
        self, x: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one batch of spatial states."""
        input_gate, forget_gate, output_gate, candidate = self.gates(
            torch.cat((x, hidden), dim=1)
        ).chunk(4, dim=1)
        cell = torch.sigmoid(forget_gate) * cell + torch.sigmoid(
            input_gate
        ) * torch.tanh(candidate)
        hidden = torch.sigmoid(output_gate) * torch.tanh(cell)
        return hidden, cell


def _reset_spatial_rows(tensor: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
    """Multiply out terminal batch rows from either one or many LSTM layers."""
    batch_dimension = 1 if tensor.ndim == 5 else 0
    shape = [1] * tensor.ndim
    shape[batch_dimension] = -1
    keep = (~dones).to(device=tensor.device, dtype=tensor.dtype).view(shape)
    return tensor * keep


class ConvLSTM(torch.nn.Module):
    """A stacked ConvLSTM that resets terminal rows before their observation."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int,
        layers: int = 1,
    ) -> None:
        """Build a recurrent stack whose later layers consume prior hidden maps."""
        super().__init__()
        self.hidden_channels = hidden_channels
        self.layers = layers
        self.cells = torch.nn.ModuleList(
            ConvLSTMCell(
                input_channels if layer == 0 else hidden_channels,
                hidden_channels,
                kernel_size,
            )
            for layer in range(layers)
        )

    def initial_state(
        self, batch: int, height: int, width: int, *, like: torch.Tensor
    ) -> ConvLSTMState:
        """Allocate a layer-major zero state from the caller's tensor metadata."""
        state = like.new_zeros(self.layers, batch, self.hidden_channels, height, width)
        return ConvLSTMState(hidden=state, cell=torch.zeros_like(state))

    def forward(
        self,
        x: torch.Tensor,
        state: ConvLSTMState | PolicyState | None,
        dones: torch.Tensor,
    ) -> ConvLSTMOutput:
        """Return last-layer sequences and final state for time-major inputs."""
        time, batch, _, height, width = x.shape
        if state is None:
            state = self.initial_state(batch, height, width, like=x)
        elif isinstance(state, PolicyState):
            state = ConvLSTMState(hidden=state.hidden, cell=state.cell)

        hidden_steps: list[torch.Tensor] = []
        cell_steps: list[torch.Tensor] = []
        for step in range(time):
            state = state.reset_rows(dones[step])
            state = self.step(x[step], state)
            hidden_steps.append(state.hidden[-1])
            cell_steps.append(state.cell[-1])
        return ConvLSTMOutput(
            hidden_sequence=torch.stack(hidden_steps),
            cell_sequence=torch.stack(cell_steps),
            state=state,
        )

    def step(self, x: torch.Tensor, state: ConvLSTMState) -> ConvLSTMState:
        """Advance an already terminal-reset state by one observation."""
        hidden = state.hidden
        cell = state.cell
        if hidden.ndim == 4:
            hidden = hidden.unsqueeze(0)
            cell = cell.unsqueeze(0)
        layer_input = x
        next_hidden: list[torch.Tensor] = []
        next_cell: list[torch.Tensor] = []
        for layer, recurrent_cell in enumerate(self.cells):
            layer_hidden, layer_cell = recurrent_cell(
                layer_input, hidden[layer], cell[layer]
            )
            next_hidden.append(layer_hidden)
            next_cell.append(layer_cell)
            layer_input = layer_hidden
        return ConvLSTMState(
            hidden=torch.stack(next_hidden), cell=torch.stack(next_cell)
        )


class _TransformerBlock(torch.nn.Module):
    """One pre-normalized self-attention and MLP residual block."""

    def __init__(self, channels: int, heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.attention_norm = torch.nn.LayerNorm(channels)
        self.attention = torch.nn.MultiheadAttention(channels, heads, batch_first=True)
        self.mlp_norm = torch.nn.LayerNorm(channels)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(channels, channels * mlp_ratio),
            torch.nn.GELU(),
            torch.nn.Linear(channels * mlp_ratio, channels),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        tokens = tokens + attended
        return tokens + self.mlp(self.mlp_norm(tokens))


def _validate_attention_dimensions(channels: int, heads: int, mlp_ratio: int) -> None:
    """Reject malformed attention dimensions at the standalone module boundary."""
    if channels <= 0 or heads <= 0 or mlp_ratio <= 0:
        raise ValueError("attention channels, heads, and mlp_ratio must be positive")
    if channels < 2:
        raise ValueError("attention channels must be at least 2")
    if channels % heads:
        raise ValueError("attention channels must divide evenly across heads")


class SpatialTransformer(torch.nn.Module):
    """Pre-normalized attention over the explicit 10x10 spatial token grid."""

    def __init__(self, channels: int, blocks: int, heads: int, mlp_ratio: int) -> None:
        super().__init__()
        _validate_attention_dimensions(channels, heads, mlp_ratio)
        if blocks <= 0:
            raise ValueError("spatial transformer blocks must be positive")
        self.position = torch.nn.Parameter(
            torch.zeros(1, BOARD_SIZE * BOARD_SIZE, channels)
        )
        self.blocks = torch.nn.ModuleList(
            _TransformerBlock(channels, heads, mlp_ratio) for _ in range(blocks)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Preserve ``(batch, channels, 10, 10)`` around token attention."""
        batch, channels, height, width = features.shape
        if (height, width) != (BOARD_SIZE, BOARD_SIZE):
            raise ValueError(
                f"spatial transformer requires a 10x10 map, got {height}x{width}"
            )
        tokens = features.flatten(2).transpose(1, 2) + self.position
        for block in self.blocks:
            tokens = block(tokens)
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


def _bound_value(value: torch.Tensor, bound: float | None) -> torch.Tensor:
    """Apply the control policy's established optional scalar value bound."""
    if bound is None:
        return value
    return torch.sigmoid(value) * (2.0 * bound) - bound


class InteractionValueHead(torch.nn.Module):
    """Read value with one query over a global token and 100 spatial tokens."""

    def __init__(
        self,
        channels: int,
        global_features: int,
        heads: int,
        mlp_ratio: int,
        value_bound: float | None,
    ) -> None:
        super().__init__()
        _validate_attention_dimensions(channels, heads, mlp_ratio)
        if global_features <= 0:
            raise ValueError("interaction global_features must be positive")
        self.value_bound = value_bound
        self.value_token = torch.nn.Parameter(torch.zeros(1, 1, channels))
        self.global_projection = torch.nn.Linear(global_features, channels)
        self.query_norm = torch.nn.LayerNorm(channels)
        self.context_norm = torch.nn.LayerNorm(channels)
        self.attention = torch.nn.MultiheadAttention(channels, heads, batch_first=True)
        self.mlp_norm = torch.nn.LayerNorm(channels)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(channels, channels * mlp_ratio),
            torch.nn.GELU(),
            torch.nn.Linear(channels * mlp_ratio, channels),
        )
        self.value_projection = torch.nn.Linear(channels, 1)

    def forward(
        self, features: torch.Tensor, global_features: torch.Tensor
    ) -> torch.Tensor:
        """Attend before reducing, without pooling away spatial interactions."""
        batch, _channels, height, width = features.shape
        if (height, width) != (BOARD_SIZE, BOARD_SIZE):
            raise ValueError(
                f"interaction value requires a 10x10 map, got {height}x{width}"
            )
        spatial = features.flatten(2).transpose(1, 2)
        global_token = self.global_projection(global_features).unsqueeze(1)
        context = self.context_norm(torch.cat((global_token, spatial), dim=1))
        value = self.value_token.expand(batch, -1, -1)
        attended, _ = self.attention(
            self.query_norm(value), context, context, need_weights=False
        )
        value = value + attended
        value = value + self.mlp(self.mlp_norm(value))
        scalar = self.value_projection(value[:, 0]).squeeze(-1)
        return _bound_value(scalar, self.value_bound)


def uses_stateful_policy(config: ModelConfig) -> bool:
    """Return whether implemented optional components need typed time-major I/O."""
    return any(
        (
            config.recurrent,
            config.transformer,
            config.local_patch,
            config.belief,
            config.interaction_value,
        )
    )


@dataclass(frozen=True)
class PolicyOutput:
    """Every tensor emitted by a stateful Toad policy forward pass."""

    unit_logits: torch.Tensor
    quantity_logits: torch.Tensor
    market_logits: torch.Tensor
    values: torch.Tensor
    belief_logits: torch.Tensor | None
    state: PolicyState | None
    input_state: PolicyState | None = None


class StatefulPolicy(torch.nn.Module):
    """Stable stateful interface with an exact legacy-control delegation path."""

    def __init__(self, config: ModelConfig) -> None:
        """Build the configured policy without changing the control topology."""
        super().__init__()
        self.config = config
        self.control = Policy(config.blocks, config.channels, config.value_bound)
        if config.recurrent:
            self.recurrent = ConvLSTM(
                config.channels,
                config.recurrent_channels,
                config.recurrent_kernel_size,
                config.recurrent_layers,
            )
            self.merge = torch.nn.Conv2d(
                config.channels + 2 * config.recurrent_channels,
                config.channels,
                kernel_size=1,
            )
        if config.transformer:
            self.transformer = SpatialTransformer(
                config.channels,
                config.transformer_blocks,
                config.transformer_heads,
                config.transformer_mlp_ratio,
            )
        if config.belief:
            self.belief_head = torch.nn.Linear(config.channels, config.belief_size)
        if config.belief_feedback:
            self.feedback = torch.nn.Linear(config.belief_size, config.channels)
        if config.local_patch:
            self.local_head = LocalUnitHead(
                config.channels,
                config.local_patch_blocks,
                config.local_patch_size,
            )
        if config.interaction_value:
            self.interaction_value = InteractionValueHead(
                config.channels,
                SCALARS,
                config.transformer_heads,
                config.transformer_mlp_ratio,
                config.value_bound,
            )

    def _has_optional_components(self) -> bool:
        return any(
            (
                self.config.recurrent,
                self.config.transformer,
                self.config.local_patch,
                self.config.belief,
                self.config.interaction_value,
            )
        )

    def load_control_state_dict(self, weights: Mapping[str, torch.Tensor]) -> None:
        """Load an exact bare or wrapper control state dict.

        Control-only weights are deliberately rejected for enabled models so
        that new components can never retain random initialization unnoticed.
        """
        if self._has_optional_components():
            raise ValueError(
                "control weights cannot initialize enabled optional components"
            )
        keys = tuple(weights)
        prefixed = [key.startswith("control.") for key in keys]
        if any(prefixed) and not all(prefixed):
            raise ValueError("control checkpoint mixes bare and stateful key layouts")
        control_weights = (
            {key.removeprefix("control."): value for key, value in weights.items()}
            if prefixed
            else dict(weights)
        )
        self.control.load_state_dict(control_weights, strict=True)

    def initial_state(self, batch: int, *, like: torch.Tensor) -> PolicyState | None:
        """Allocate zero state from the caller's device and dtype when needed."""
        if not (self.config.recurrent or self.config.belief):
            return None
        recurrent_state = (
            self.recurrent.initial_state(batch, BOARD_SIZE, BOARD_SIZE, like=like)
            if self.config.recurrent
            else None
        )
        hidden = (
            recurrent_state.hidden
            if recurrent_state is not None
            else like.new_zeros(batch, 0, 0, 0)
        )
        cell = (
            recurrent_state.cell
            if recurrent_state is not None
            else like.new_zeros(batch, 0, 0, 0)
        )
        if recurrent_state is not None and self.config.recurrent_layers == 1:
            hidden = hidden.squeeze(0)
            cell = cell.squeeze(0)
        return PolicyState(
            hidden=hidden,
            cell=cell,
            prior_belief=like.new_zeros(batch, self.config.belief_size),
        )

    def _stateful_step(
        self,
        board: torch.Tensor,
        scalars: torch.Tensor,
        state: PolicyState | None,
    ) -> tuple[torch.Tensor, PolicyState | None, torch.Tensor | None]:
        """Advance optional recurrence and belief feedback for one time row."""
        scalar_features = self.control.market(scalars)
        if self.config.belief_feedback:
            assert state is not None
            scalar_features = scalar_features + self.feedback(state.prior_belief)
        features = self.control.stem(board) + scalar_features[:, :, None, None]
        for block in self.control.blocks:
            features = block(features)
        if self.config.recurrent:
            assert state is not None
            recurrent_state = self.recurrent.step(
                features, ConvLSTMState(hidden=state.hidden, cell=state.cell)
            )
            features = self.merge(
                torch.cat(
                    (
                        features,
                        recurrent_state.hidden[-1],
                        recurrent_state.cell[-1],
                    ),
                    dim=1,
                )
            )
            hidden = recurrent_state.hidden
            cell = recurrent_state.cell
        else:
            hidden = state.hidden if state is not None else None
            cell = state.cell if state is not None else None
        if self.config.transformer:
            features = self.transformer(features)
        belief = (
            self.belief_head(features.mean(dim=(2, 3))) if self.config.belief else None
        )
        if state is None:
            return features, None, belief
        assert hidden is not None
        assert cell is not None
        return (
            features,
            PolicyState(
                hidden,
                cell,
                belief if belief is not None else state.prior_belief,
            ),
            belief,
        )

    def _validate_state(
        self, state: PolicyState, batch: int, *, like: torch.Tensor
    ) -> None:
        """Reject state whose tensor schema does not match this topology."""
        if state.hidden.shape != state.cell.shape:
            raise ValueError("policy hidden and cell state shapes must match")
        if self.config.recurrent:
            expected_spatial = (
                (batch, self.config.recurrent_channels, BOARD_SIZE, BOARD_SIZE)
                if self.config.recurrent_layers == 1
                else (
                    self.config.recurrent_layers,
                    batch,
                    self.config.recurrent_channels,
                    BOARD_SIZE,
                    BOARD_SIZE,
                )
            )
            if tuple(state.hidden.shape) != expected_spatial:
                raise ValueError(
                    "recurrent policy state has the wrong spatial shape: "
                    f"{tuple(state.hidden.shape)} != {expected_spatial}"
                )
        elif state.hidden.numel() != 0 or tuple(state.hidden.shape) != (batch, 0, 0, 0):
            raise ValueError(
                "belief-only policy hidden and cell state must be zero-sized "
                f"batch-aligned tensors, got {tuple(state.hidden.shape)}"
            )
        expected_belief = (batch, self.config.belief_size)
        if self.config.belief and tuple(state.prior_belief.shape) != expected_belief:
            raise ValueError(
                "policy prior-belief state has the wrong shape: "
                f"{tuple(state.prior_belief.shape)} != {expected_belief}"
            )
        if not self.config.belief and (
            state.prior_belief.ndim != 2 or state.prior_belief.shape[0] != batch
        ):
            raise ValueError("policy prior-belief state must be batch-aligned")
        for name, tensor in (
            ("hidden", state.hidden),
            ("cell", state.cell),
            ("prior belief", state.prior_belief),
        ):
            if tensor.device != like.device or tensor.dtype != like.dtype:
                raise ValueError(
                    f"policy {name} state must match the input device and dtype"
                )

    def _unit_readouts(
        self, features: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use either exact control columns or configured local patches."""
        if self.config.local_patch:
            return self.local_head(features, positions)
        columns = features.flatten(2)
        wanted = positions[:, None, :].tile(1, columns.shape[1], 1)
        gathered = columns.gather(2, wanted).transpose(1, 2)
        return self.control.head(gathered), self.control.quantity_head(gathered)

    def forward(
        self,
        board: torch.Tensor,
        scalars: torch.Tensor,
        positions: torch.Tensor,
        state: PolicyState | None = None,
        dones: torch.Tensor | None = None,
    ) -> PolicyOutput:
        """Run exact control delegation or the enabled time-major recurrence."""
        if not uses_stateful_policy(self.config):
            unit, quantity, market, values = self.control(board, scalars, positions)
            return PolicyOutput(unit, quantity, market, values, None, None)

        time, batch = board.shape[:2]
        if dones is None:
            dones = torch.zeros(time, batch, dtype=torch.bool, device=board.device)
        if self.config.recurrent or self.config.belief:
            state = self.initial_state(batch, like=board) if state is None else state
            assert state is not None
            self._validate_state(state, batch, like=board)
        elif state is not None:
            raise ValueError("attention-only policy does not accept recurrent state")
        feature_steps: list[torch.Tensor] = []
        belief_steps: list[torch.Tensor] = []
        input_state: PolicyState | None = None
        for step in range(time):
            if state is not None:
                state = state.reset_rows(dones[step])
                if step == 0:
                    input_state = state
            features, state, belief = self._stateful_step(
                board[step], scalars[step], state
            )
            feature_steps.append(features)
            if belief is not None:
                belief_steps.append(belief)

        flat_features = torch.stack(feature_steps).flatten(0, 1)
        flat_positions = positions.flatten(0, 1)
        flat_units, flat_quantities = self._unit_readouts(flat_features, flat_positions)
        units = flat_units.view(time, batch, -1, len(UNIT_OPS))
        quantities = flat_quantities.view(time, batch, -1, len(QUANTITIES))
        pooled = flat_features.mean(dim=(2, 3))
        market = self.control.trade_head(pooled).view(
            time, batch, len(MARKET_SLOTS) + 2, len(QUANTITIES)
        )
        values = (
            self.interaction_value(flat_features, scalars.flatten(0, 1))
            if self.config.interaction_value
            else _bound_value(
                self.control.value(pooled).squeeze(-1), self.control.value_bound
            )
        )
        output_state = state
        if (
            state is not None
            and self.config.recurrent
            and self.config.recurrent_layers == 1
        ):
            output_state = PolicyState(
                state.hidden.squeeze(0), state.cell.squeeze(0), state.prior_belief
            )
        return PolicyOutput(
            units,
            quantities,
            market,
            values.view(time, batch),
            torch.stack(belief_steps) if belief_steps else None,
            output_state,
            input_state,
        )
