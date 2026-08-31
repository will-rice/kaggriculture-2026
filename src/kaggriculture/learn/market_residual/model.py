"""The Torch half of the market residual: one small recurrent head.

The architecture is deliberately the smallest thing that can express the
decision the residual is for. A market event is not independent of the ones
before it -- whether to sell now depends on whether we already sold, on how the
book has moved since we last looked, and on how much season is left -- so the
head is recurrent over events rather than a function of one row. Everything
else is a linear readout of that state: which mode to play, how much of each
allowed slot, what the paired outcome is worth, and the auxiliary next-event
predictions the offline phase regresses against. It comes to 415,173 parameters
against a budget of a million, and ``tests/market_residual/test_model.py`` pins
that it stays under.

Two decisions here are not free choices.

**Everything runs in FP32, autocast or not.** The rest of this project trains
under BF16 autocast, and BF16 carries about three significant decimal digits.
The quantities this head produces -- a mode logit near a tie, a joint log
probability, a value regressed against paired win-point deltas of order 0.01 --
do not survive that, and neither would export parity: the served artifact is
float32, so a head that trained in BF16 would be gated as one thing and
submitted as another. The whole forward is therefore wrapped in a disabled
autocast region rather than only the heads, because this net is small enough
that the mixed-precision speedup it forgoes is not measurable, and a boundary
drawn around "the sensitive parts" is a boundary someone later moves.

**Illegal quantities are masked to the finite float32 minimum**, the constant
the NumPy runtime uses, imported rather than restated. ``-inf`` is what the
rest of this codebase masks with, and it is wrong here: an allowed slot with no
legal bucket is an ordinary position, and an all-``-inf`` row makes
``log_softmax`` NaN and poisons the update. The finite minimum makes that row
uniform and keeps every logit the merge boundary inspects finite.

``ModelConfig`` is imported from the runtime rather than declared here, so one
description of the architecture sizes the Torch parameters, writes the export,
and validates it at load.
"""

from dataclasses import dataclass

import torch
from torch import nn

from kaggriculture.market_residual.actions import ResidualDecision, ResidualMode
from kaggriculture.market_residual.numpy_policy import (
    MASK_FILL,
    MODES,
    ModelConfig,
)

__all__ = [
    "MarketResidualNet",
    "ModelConfig",
    "PolicyHeads",
    "entropy",
    "greedy_decision",
    "log_probability",
    "mask_quantity_logits",
]

REPLACE_INDEX = MODES.index(ResidualMode.REPLACE)


@dataclass(frozen=True)
class PolicyHeads:
    """One forward pass's outputs, before any of them has been decided from.

    Args:
        mode_logits: ``(steps, batch, modes)`` preference for deferring to the
            frozen controller versus replacing its commodity orders.
        quantity_logits: ``(steps, batch, slots, buckets)`` quantity preference
            per allowed market slot, unmasked.
        value: ``(steps, batch)`` estimate of the paired incremental outcome.
        auxiliary: ``(steps, batch, auxiliary_size)`` next-event predictions.
    """

    mode_logits: torch.Tensor
    quantity_logits: torch.Tensor
    value: torch.Tensor
    auxiliary: torch.Tensor


class MarketResidualNet(nn.Module):
    """A projection, a single-layer GRU over market events, and four readouts."""

    def __init__(self, config: ModelConfig) -> None:
        """Build the head at one declared shape.

        Args:
            config: The architecture the runtime will also be told to expect.
        """
        super().__init__()
        self.config = config
        self.projection = nn.Linear(config.input_size, config.projection_size)
        self.gru = nn.GRU(config.projection_size, config.hidden_size)
        self.mode = nn.Linear(config.hidden_size, len(MODES))
        self.quantities = nn.Linear(config.hidden_size, config.slots * config.buckets)
        self.value = nn.Linear(config.hidden_size, 1)
        self.auxiliary = nn.Linear(config.hidden_size, config.auxiliary_size)

    def forward(
        self, features: torch.Tensor, state: torch.Tensor | None
    ) -> tuple[PolicyHeads, torch.Tensor]:
        """Run a sequence of market events and return the state it ended in.

        Args:
            features: ``(steps, batch, input_size)`` canonical feature rows, in
                the order the events happened.
            state: ``(1, batch, hidden_size)`` state carried from the previous
                call, or ``None`` to start an episode.

        Returns:
            This sequence's head outputs and the state to carry forward. The
            state is returned rather than kept on the module so that one head
            can serve many episodes at once and none of them can leak into
            another.
        """
        with torch.autocast(device_type=features.device.type, enabled=False):
            projected = nn.functional.silu(self.projection(features.float()))
            outputs, current = self.gru(
                projected, None if state is None else state.float()
            )
            steps, batch, _hidden = outputs.shape
            return (
                PolicyHeads(
                    mode_logits=self.mode(outputs),
                    quantity_logits=self.quantities(outputs).reshape(
                        steps, batch, self.config.slots, self.config.buckets
                    ),
                    value=self.value(outputs)[..., 0],
                    auxiliary=self.auxiliary(outputs),
                ),
                current,
            )

    def exported_arrays(self) -> dict[str, torch.Tensor]:
        """Return the parameters under the names the runtime artifact uses.

        Torch's own layouts are kept, so the export is a transcription: linear
        weights stay ``(out, in)`` and the GRU's gates stay packed reset, update
        then new, which is the order the NumPy cell slices them in.

        Returns:
            Every array in ``ModelConfig.array_shapes`` order, detached on the
            CPU.
        """
        recurrent = dict(self.gru.named_parameters())
        named: dict[str, torch.Tensor] = {
            "projection.weight": self.projection.weight,
            "projection.bias": self.projection.bias,
            "gru.weight_ih": recurrent["weight_ih_l0"],
            "gru.weight_hh": recurrent["weight_hh_l0"],
            "gru.bias_ih": recurrent["bias_ih_l0"],
            "gru.bias_hh": recurrent["bias_hh_l0"],
            "mode.weight": self.mode.weight,
            "mode.bias": self.mode.bias,
            "quantities.weight": self.quantities.weight,
            "quantities.bias": self.quantities.bias,
            "value.weight": self.value.weight,
            "value.bias": self.value.bias,
            "auxiliary.weight": self.auxiliary.weight,
            "auxiliary.bias": self.auxiliary.bias,
        }
        if tuple(named) != tuple(self.config.array_shapes()):
            raise RuntimeError("exported array names do not match the declared model")
        return {
            name: parameter.detach().to(device="cpu", dtype=torch.float32)
            for name, parameter in named.items()
        }


def mask_quantity_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return quantity logits with illegal buckets pushed to a finite minimum.

    Args:
        logits: Quantity logits whose last axis is the bucket vocabulary.
        mask: A boolean tensor of the same shape, true where a bucket is legal.

    Returns:
        The logits with every illegal entry replaced by ``MASK_FILL``, which is
        finite, so a slot with nothing legal becomes uniform instead of NaN.
    """
    return logits.masked_fill(~mask, MASK_FILL)


def log_probability(
    heads: PolicyHeads,
    mask: torch.Tensor,
    modes: torch.Tensor,
    buckets: torch.Tensor,
) -> torch.Tensor:
    """Return the joint log probability of the actions actually taken.

    The action is factorised: a mode, and -- only when that mode replaces --
    one quantity per allowed slot. Deferring therefore costs the mode term
    alone, which is what makes ``USE_KAITO`` a single decision rather than one
    the quantity heads are also scored on.

    Args:
        heads: One forward pass's outputs.
        mask: ``(steps, batch, slots, buckets)`` legality.
        modes: ``(steps, batch)`` indices into ``MODES``.
        buckets: ``(steps, batch, slots)`` chosen quantity buckets.

    Returns:
        ``(steps, batch)`` joint log probabilities.
    """
    mode_terms = torch.log_softmax(heads.mode_logits, dim=-1).gather(
        -1, modes[..., None]
    )[..., 0]
    quantity_terms = (
        torch.log_softmax(mask_quantity_logits(heads.quantity_logits, mask), dim=-1)
        .gather(-1, buckets[..., None])[..., 0]
        .sum(dim=-1)
    )
    return mode_terms + torch.where(modes == REPLACE_INDEX, quantity_terms, 0.0)


def entropy(heads: PolicyHeads, mask: torch.Tensor) -> torch.Tensor:
    """Return the entropy of the factorised action distribution.

    The quantity heads are counted only as often as they are used: a head that
    is certain it will defer has the entropy of its mode alone, so an entropy
    bonus cannot pay for exploration in slots the policy never reaches.

    Args:
        heads: One forward pass's outputs.
        mask: ``(steps, batch, slots, buckets)`` legality.

    Returns:
        ``(steps, batch)`` entropies.
    """
    mode_log = torch.log_softmax(heads.mode_logits, dim=-1)
    mode_entropy = -(mode_log.exp() * mode_log).sum(dim=-1)
    quantity_log = torch.log_softmax(
        mask_quantity_logits(heads.quantity_logits, mask), dim=-1
    )
    quantity_entropy = -(quantity_log.exp() * quantity_log).sum(dim=-1).sum(dim=-1)
    return mode_entropy + mode_log[..., REPLACE_INDEX].exp() * quantity_entropy


def greedy_decision(
    mode_logits: torch.Tensor, quantity_logits: torch.Tensor
) -> ResidualDecision:
    """Return the argmax action for one market event.

    The Torch twin of ``numpy_policy.greedy_decision``, and the reason it
    exists: the parity check compares two implementations of the decode as well
    as two implementations of the arithmetic, because a runtime that agreed on
    every logit and disagreed on the argmax would play a different season.

    Args:
        mode_logits: ``(modes,)`` logits for one event.
        quantity_logits: ``(slots, buckets)`` logits for one event, already
            masked.

    Returns:
        The decision to hand the merge boundary, with ``finite`` read off the
        logits rather than assumed.
    """
    finite = bool(
        torch.isfinite(mode_logits).all() and torch.isfinite(quantity_logits).all()
    )
    return ResidualDecision(
        mode=MODES[int(mode_logits.argmax())],
        buckets=tuple(
            int(bucket) for bucket in quantity_logits.argmax(dim=-1).tolist()
        ),
        finite=finite,
    )
