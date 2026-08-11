"""Batched market-price calculations with exact reference semantics."""

from functools import lru_cache

import torch

from kaggriculture.constants import MARKET_PARAMS
from kaggriculture.sim.state import PRODUCT_NAMES
from kaggriculture.sim.tensors import tensor_constant

PRICE_FLOOR = 1

# Every shape the reference curve can name, in the order the gathered shape
# codes index them.
_SHAPES = ("linear", "sq", "sqrt", "log", "log10")

# An inventory this large prices every product at the floor, so it bounds the
# search for the level where each curve bottoms out.
_UNREACHABLE_LEVEL = 1 << 62


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


def _shaped(value: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
    """Apply the curve shape each element names, without a runtime name scan."""
    shapes = torch.stack([_shape(name, value) for name in _SHAPES])
    return shapes.gather(0, code.expand(value.shape)[None]).squeeze(0)


@lru_cache(maxsize=None)
def _curve(device_type: str, device_index: int | None) -> tuple[torch.Tensor, ...]:
    """Materialize the per-product curve parameters on one device.

    Amplitudes are evaluated with the same float64 tensor operations the
    reference-equivalent scalar path used, so the gathered table is bitwise
    identical to computing them inside the quote.

    Args:
        device_type: Torch device type, such as ``"cuda"``.
        device_index: Device ordinal, or ``None`` for the default device.

    Returns:
        Base, initial level, below and above amplitudes, and below and above
        shape codes, each a tensor indexed by product.
    """
    device = (
        torch.device(device_type)
        if device_index is None
        else torch.device(device_type, device_index)
    )
    below_amplitude = []
    above_amplitude = []
    for name in PRODUCT_NAMES:
        params = MARKET_PARAMS[name]
        base = float(params["base"])
        threshold = tensor_constant(
            float(params["T"]), dtype=torch.float64, device=device
        )
        below_amplitude.append(
            float(params["below_target"])
            * base
            / _shape(str(params["below_func"]), threshold)
        )
        above_amplitude.append(
            float(params["above_target"])
            * base
            / _shape(str(params["above_func"]), threshold)
        )
    scalar = {"dtype": torch.float64, "device": device}
    return (
        tensor_constant(
            [float(MARKET_PARAMS[n]["base"]) for n in PRODUCT_NAMES], **scalar
        ),
        tensor_constant(
            [float(MARKET_PARAMS[n]["I0"]) for n in PRODUCT_NAMES], **scalar
        ),
        torch.cat(below_amplitude),
        torch.cat(above_amplitude),
        tensor_constant(
            [_SHAPES.index(str(MARKET_PARAMS[n]["below_func"])) for n in PRODUCT_NAMES],
            dtype=torch.int64,
            device=device,
        ),
        tensor_constant(
            [_SHAPES.index(str(MARKET_PARAMS[n]["above_func"])) for n in PRODUCT_NAMES],
            dtype=torch.int64,
            device=device,
        ),
    )


def _tables(device: torch.device) -> tuple[torch.Tensor, ...]:
    resolved = torch.device(device)
    index = resolved.index
    if resolved.type == "cuda" and index is None:
        index = torch.cuda.current_device()
    return _curve(resolved.type, index)


def prices_for(inventory: torch.Tensor, product: torch.Tensor) -> torch.Tensor:
    """Return exact integer prices for a gathered product index.

    Args:
        inventory: Integral inventory levels of arbitrary shape.
        product: Product indices broadcastable to ``inventory``'s shape.

    Returns:
        An int64 tensor with the same shape and device as ``inventory``.
    """
    base, initial, below_amplitude, above_amplitude, below_code, above_code = _tables(
        inventory.device
    )
    index = product.expand(inventory.shape)
    values = inventory.to(torch.float64)
    start = initial[index]
    below = base[index] + below_amplitude[index] * _shaped(
        start - values, below_code[index]
    )
    above = base[index] - above_amplitude[index] * _shaped(
        values - start, above_code[index]
    )
    prices = torch.where(values < start, below, above)
    return torch.clamp_min(torch.round(prices).to(torch.int64), PRICE_FLOOR)


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
    return prices_for(
        inventory,
        tensor_constant(product, dtype=torch.int64, device=inventory.device),
    )


def refresh_prices(inventory: torch.Tensor) -> torch.Tensor:
    """Price a ``(..., products)`` inventory tensor in one batched pass."""
    if inventory.shape[-1] != len(PRODUCT_NAMES):
        raise ValueError(
            f"last dimension is {inventory.shape[-1]}; expected {len(PRODUCT_NAMES)}"
        )
    catalogue = tensor_constant(
        tuple(range(len(PRODUCT_NAMES))), dtype=torch.int64, device=inventory.device
    )
    return prices_for(inventory, catalogue)


@lru_cache(maxsize=None)
def _floor(device_type: str, device_index: int | None) -> torch.Tensor:
    device = (
        torch.device(device_type)
        if device_index is None
        else torch.device(device_type, device_index)
    )
    catalogue = torch.arange(len(PRODUCT_NAMES), device=device)
    low = torch.zeros_like(catalogue)
    high = torch.full_like(catalogue, _UNREACHABLE_LEVEL)
    for _ in range(_UNREACHABLE_LEVEL.bit_length()):
        middle = low + (high - low) // 2
        at_floor = prices_for(middle, catalogue) <= PRICE_FLOOR
        low = torch.where(at_floor, low, middle + 1)
        high = torch.where(at_floor, middle, high)
    return low


def floor_levels(device: torch.device) -> torch.Tensor:
    """Return the lowest inventory level that prices each product at the floor.

    Prices are non-increasing in inventory, so a sale adds supply exactly while
    the level is below this bound. The market phase uses it to advance the book
    across a whole quantity axis without re-quoting each unit-round.

    Args:
        device: Device the returned table must live on.

    Returns:
        An int64 tensor indexed by product.
    """
    resolved = torch.device(device)
    index = resolved.index
    if resolved.type == "cuda" and index is None:
        index = torch.cuda.current_device()
    return _floor(resolved.type, index)
