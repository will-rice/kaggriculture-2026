"""Thin Torch compatibility adapters for canonical Python legality masks."""

from typing import Any, Mapping

import torch

from kaggriculture.features import encode_observation
from kaggriculture.learn.encoding import to_torch


def unit_mask(observation: Mapping[str, Any], seat: int) -> torch.Tensor:
    """Return the legacy batched unit-operation mask."""
    return to_torch(encode_observation(observation, seat)).unit_mask


def unit_quantity_mask(observation: Mapping[str, Any], seat: int) -> torch.Tensor:
    """Return the legacy batched per-unit transfer-quantity mask."""
    return to_torch(encode_observation(observation, seat)).quantity_mask


def market_mask(observation: Mapping[str, Any], seat: int) -> torch.Tensor:
    """Return the legacy batched market-quantity mask."""
    return to_torch(encode_observation(observation, seat)).market_mask
