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
from kaggriculture.learn.encoding import MARKET_SLOTS, QUANTITIES, UNIT_OPS
from kaggriculture.learn.model import Policy
from kaggriculture.learn.toad.config import ModelConfig


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
        if config.belief:
            self.belief_head = torch.nn.Linear(config.channels, config.belief_size)
        if config.belief_feedback:
            self.feedback = torch.nn.Linear(config.belief_size, config.channels)

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
        state: PolicyState,
    ) -> tuple[torch.Tensor, PolicyState, torch.Tensor | None]:
        """Advance optional recurrence and belief feedback for one time row."""
        scalar_features = self.control.market(scalars)
        if self.config.belief_feedback:
            scalar_features = scalar_features + self.feedback(state.prior_belief)
        features = self.control.stem(board) + scalar_features[:, :, None, None]
        for block in self.control.blocks:
            features = block(features)
        if self.config.recurrent:
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
            hidden = state.hidden
            cell = state.cell
        belief = (
            self.belief_head(features.mean(dim=(2, 3))) if self.config.belief else None
        )
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

    def forward(
        self,
        board: torch.Tensor,
        scalars: torch.Tensor,
        positions: torch.Tensor,
        state: PolicyState | None = None,
        dones: torch.Tensor | None = None,
    ) -> PolicyOutput:
        """Run exact control delegation or the enabled time-major recurrence."""
        if not (self.config.recurrent or self.config.belief):
            unit, quantity, market, values = self.control(board, scalars, positions)
            return PolicyOutput(unit, quantity, market, values, None, None)

        time, batch = board.shape[:2]
        if dones is None:
            dones = torch.zeros(time, batch, dtype=torch.bool, device=board.device)
        state = self.initial_state(batch, like=board) if state is None else state
        assert state is not None
        self._validate_state(state, batch, like=board)
        feature_steps: list[torch.Tensor] = []
        belief_steps: list[torch.Tensor] = []
        input_state: PolicyState | None = None
        for step in range(time):
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
        columns = flat_features.flatten(2)
        wanted = flat_positions[:, None, :].tile(1, columns.shape[1], 1)
        gathered = columns.gather(2, wanted).transpose(1, 2)
        units = self.control.head(gathered).view(time, batch, -1, len(UNIT_OPS))
        quantities = self.control.quantity_head(gathered).view(
            time, batch, -1, len(QUANTITIES)
        )
        pooled = flat_features.mean(dim=(2, 3))
        market = self.control.trade_head(pooled).view(
            time, batch, len(MARKET_SLOTS) + 2, len(QUANTITIES)
        )
        values = self.control.value(pooled).squeeze(-1)
        if self.control.value_bound is not None:
            values = (
                torch.sigmoid(values) * (2.0 * self.control.value_bound)
                - self.control.value_bound
            )
        hidden, cell = state.hidden, state.cell
        if self.config.recurrent and self.config.recurrent_layers == 1:
            hidden = hidden.squeeze(0)
            cell = cell.squeeze(0)
        return PolicyOutput(
            units,
            quantities,
            market,
            values.view(time, batch),
            torch.stack(belief_steps) if belief_steps else None,
            PolicyState(hidden, cell, state.prior_belief),
            input_state,
        )
