"""Batched simulator stepping and action tensors."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from kaggriculture.constants import MARKET_PARAMS, MAX_MARKET_ORDERS_PER_TURN
from kaggriculture.learn.encoding import MAX_UNITS
from kaggriculture.sim.config import Config
from kaggriculture.sim.rng import rng_words
from kaggriculture.sim.state import (
    PRODUCT_NAMES,
    SimState,
    UnsupportedConfiguration,
    empty_state,
)

ORDER_NONE = 0
ORDER_SELL = 1
ORDER_BUY_SEED = 2
ORDER_BUY_PRODUCT = 3
ORDER_BUY_ANIMAL = 4
ORDER_HIRE = 5
ORDER_BUY_LAND = 6
ORDER_MALFORMED = 7

_DEFAULT_CONFIG = Config()


@dataclass(frozen=True)
class MarketActions:
    """Expanded market orders for both seats in a batch."""

    order_type: torch.Tensor
    order_item: torch.Tensor
    order_qty: torch.Tensor

    @classmethod
    def empty(cls, batch: int, device: torch.device | str = "cpu") -> MarketActions:
        """Return an all-empty order batch."""
        shape = (batch, 2, MAX_MARKET_ORDERS_PER_TURN)
        return cls(
            order_type=torch.zeros(shape, dtype=torch.int8, device=device),
            order_item=torch.full(shape, -1, dtype=torch.int8, device=device),
            order_qty=torch.zeros(shape, dtype=torch.int32, device=device),
        )

    def compacted(self) -> MarketActions:
        """Return the same orders left-aligned into the reference's queue.

        The reference engine reads each seat's market action as a list with no
        holes and resolves the ``i``-th entry of one seat against the ``i``-th
        entry of the other, so queue position is what couples two seats into a
        lockstep round. A slot typed ``ORDER_NONE`` means the seat submitted no
        order there, not that it submitted a dead one, so it must not consume a
        position. Sliding the present orders down restores that alignment; an
        order that is merely dead -- malformed, zero quantity, or unaffordable
        -- still holds its position, exactly as it does in the reference list.

        Returns:
            Orders whose non-empty slots occupy the low indices in their
            original relative order, padded with empty slots.
        """
        width = self.order_type.shape[-1]
        present = self.order_type != ORDER_NONE
        rank = present.to(torch.int64).cumsum(dim=-1) - 1
        destination = torch.where(present, rank, torch.full_like(rank, width))
        padded = (*self.order_type.shape[:-1], width + 1)
        device = self.order_type.device
        return MarketActions(
            order_type=torch.zeros(padded, dtype=self.order_type.dtype, device=device)
            .scatter_(2, destination, self.order_type)[..., :width]
            .contiguous(),
            order_item=torch.full(
                padded, -1, dtype=self.order_item.dtype, device=device
            )
            .scatter_(2, destination, self.order_item)[..., :width]
            .contiguous(),
            order_qty=torch.zeros(padded, dtype=self.order_qty.dtype, device=device)
            .scatter_(2, destination, self.order_qty)[..., :width]
            .contiguous(),
        )


def reset(
    config: Config, seeds: torch.Tensor, device: torch.device | str | None = None
) -> SimState:
    """Create a batch of initial states for the supplied episode seeds."""
    if config != _DEFAULT_CONFIG:
        differences = [
            name
            for name in config.__dataclass_fields__
            if getattr(config, name) != getattr(_DEFAULT_CONFIG, name)
        ]
        raise UnsupportedConfiguration(
            f"unsupported configuration fields: {', '.join(differences)}"
        )
    if seeds.ndim != 1:
        raise ValueError(f"seeds must be one-dimensional, got {tuple(seeds.shape)}")
    target = seeds.device if device is None else device
    resolved_seeds = seeds.to(target)
    state = empty_state(len(seeds), target)
    state.kind[..., 5:] = 1
    state.kind[..., 5:, :5] = 1
    state.money.fill_(3_000)
    state.unit_x[..., 0] = 4
    state.unit_y[..., 0] = 4
    state.alive[..., 0] = True
    state.inventory.copy_(
        torch.tensor(
            [int(MARKET_PARAMS[name]["I0"]) for name in PRODUCT_NAMES],
            dtype=torch.int64,
            device=target,
        )[None, :].expand(len(seeds), -1)
    )
    state.prices.copy_(
        torch.tensor(
            [int(MARKET_PARAMS[name]["base"]) for name in PRODUCT_NAMES],
            dtype=torch.int64,
            device=target,
        )[None, :].expand(len(seeds), -1)
    )
    state.seed.copy_(resolved_seeds)
    state.rng_words.copy_(rng_words(resolved_seeds))
    return state


def unit_quantity_ones(batch: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Return the all-ones unit-quantity tensor.

    The many callers that mean "single-item transfers, exactly the old
    behaviour" say so with this one word.
    """
    return torch.ones((batch, 2, MAX_UNITS), dtype=torch.int16, device=device)


def step(
    state: SimState,
    unit_actions: torch.Tensor,
    market_actions: MarketActions,
    unit_quantities: torch.Tensor,
) -> SimState:
    """Advance every environment through all seven pure-Torch phases."""
    expected_units = (state.batch_size, 2, MAX_UNITS)
    if tuple(unit_actions.shape) != expected_units:
        raise ValueError(
            f"unit_actions has {tuple(unit_actions.shape)}; expected {expected_units}"
        )
    if tuple(unit_quantities.shape) != expected_units:
        raise ValueError(
            f"unit_quantities has {tuple(unit_quantities.shape)}; "
            f"expected {expected_units}"
        )
    from kaggriculture.sim.day import apply_day_phases  # noqa: PLC0415
    from kaggriculture.sim.market import apply_market_phase  # noqa: PLC0415
    from kaggriculture.sim.units import apply_unit_phases  # noqa: PLC0415

    after_units = apply_unit_phases(state, unit_actions, unit_quantities)
    after_market = apply_market_phase(after_units, market_actions)
    return apply_day_phases(after_market)
