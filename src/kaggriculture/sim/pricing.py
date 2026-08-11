"""Batched market-price calculations with exact reference semantics."""

import torch

from kaggriculture.constants import MARKET_PARAMS
from kaggriculture.sim.state import PRODUCT_NAMES
from kaggriculture.sim.tensors import tensor_constant

PRICE_FLOOR = 1


def _shape(name: str, value: torch.Tensor) -> torch.Tensor:
    """Apply one reference price-curve shape in float64."""
    value = torch.clamp_min(value, 0.0)
    if name == "linear":
        return value
    if name == "sq":
        return value * value
    if name == "sqrt":
        return torch.sqrt(value)
    if name == "log":
        return torch.log1p(value)
    if name == "log10":
        return torch.log10(1.0 + value)
    return value


def market_prices(inventory: torch.Tensor, product: int) -> torch.Tensor:
    """Return exact integer prices for one product over any batch shape.

    Args:
        inventory: Integral inventory levels of arbitrary shape.
        product: Index into ``PRODUCT_NAMES``.

    Returns:
        An int64 tensor with the same shape and device as ``inventory``.
    """
    if not 0 <= product < len(PRODUCT_NAMES):
        raise IndexError(product)
    params = MARKET_PARAMS[PRODUCT_NAMES[product]]
    values = inventory.to(torch.float64)
    base = float(params["base"])
    initial = float(params["I0"])
    threshold = tensor_constant(
        float(params["T"]), dtype=torch.float64, device=inventory.device
    )

    below_shape = str(params["below_func"])
    below_amp = float(params["below_target"]) * base / _shape(below_shape, threshold)
    below = base + below_amp * _shape(below_shape, initial - values)

    above_shape = str(params["above_func"])
    above_amp = float(params["above_target"]) * base / _shape(above_shape, threshold)
    above = base - above_amp * _shape(above_shape, values - initial)
    prices = torch.where(values < initial, below, above)
    return torch.clamp_min(torch.round(prices).to(torch.int64), PRICE_FLOOR)


def refresh_prices(inventory: torch.Tensor) -> torch.Tensor:
    """Price a ``(..., products)`` inventory tensor product by product."""
    if inventory.shape[-1] != len(PRODUCT_NAMES):
        raise ValueError(
            f"last dimension is {inventory.shape[-1]}; expected {len(PRODUCT_NAMES)}"
        )
    return torch.stack(
        [
            market_prices(inventory[..., index], index)
            for index in range(len(PRODUCT_NAMES))
        ],
        dim=-1,
    )
