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


@dataclass(frozen=True)
class PolicyOutput:
    """Every tensor emitted by a stateful Toad policy forward pass."""

    unit_logits: torch.Tensor
    quantity_logits: torch.Tensor
    market_logits: torch.Tensor
    values: torch.Tensor
    belief_logits: torch.Tensor | None
    state: PolicyState | None


class StatefulPolicy(torch.nn.Module):
    """Stable stateful interface with an exact legacy-control delegation path."""

    def __init__(self, config: ModelConfig) -> None:
        """Build the configured policy without changing the control topology."""
        super().__init__()
        self.config = config
        self.control = Policy(config.blocks, config.channels, config.value_bound)

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
        return PolicyState(
            hidden=like.new_zeros(
                batch,
                self.config.recurrent_channels,
                BOARD_SIZE,
                BOARD_SIZE,
            ),
            cell=like.new_zeros(
                batch,
                self.config.recurrent_channels,
                BOARD_SIZE,
                BOARD_SIZE,
            ),
            prior_belief=like.new_zeros(batch, self.config.belief_size),
        )

    def forward(
        self,
        board: torch.Tensor,
        scalars: torch.Tensor,
        positions: torch.Tensor,
        state: PolicyState | None = None,
        dones: torch.Tensor | None = None,
    ) -> PolicyOutput:
        """Return the exact control tensors through the new typed interface."""
        unit, quantity, market, values = self.control(board, scalars, positions)
        return PolicyOutput(unit, quantity, market, values, None, None)
