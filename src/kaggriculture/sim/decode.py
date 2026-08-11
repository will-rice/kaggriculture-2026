"""Pure-tensor decoding from policy heads to simulator actions."""

import torch

from kaggriculture.learn.encoding import (
    HIRE_SLOT,
    LAND_SLOT,
    MARKET_SLOTS,
    QUANTITIES,
)
from kaggriculture.sim.engine import MarketActions
from kaggriculture.sim.state import ANIMAL_NAMES, CROP_NAMES, PRODUCT_NAMES
from kaggriculture.sim.tensors import tensor_constant

_TYPE = {
    "SELL": 1,
    "BUY_SEED": 2,
    "BUY_PRODUCT": 3,
    "BUY_ANIMAL": 4,
}
_CATALOGUES = {
    "SELL": PRODUCT_NAMES,
    "BUY_SEED": CROP_NAMES,
    "BUY_PRODUCT": ("WHEAT", "FERTILIZER"),
    "BUY_ANIMAL": ANIMAL_NAMES,
}


def decode_unit_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Select the highest-scoring legal unit op for every batch slot."""
    if logits.shape != mask.shape:
        raise ValueError(f"logits shape {logits.shape} != mask shape {mask.shape}")
    return logits.masked_fill(~mask, -torch.inf).argmax(dim=-1).to(torch.int16)


def decode_market_logits(logits: torch.Tensor, mask: torch.Tensor) -> MarketActions:
    """Mask market logits and expand their winning quantity buckets."""
    if logits.shape != mask.shape:
        raise ValueError(f"logits shape {logits.shape} != mask shape {mask.shape}")
    buckets = logits.masked_fill(~mask, -torch.inf).argmax(dim=-1)
    return decode_market_buckets(buckets)


def decode_market_buckets(buckets: torch.Tensor) -> MarketActions:
    """Expand ``(B,2,21)`` quantity buckets into at most ten ordered orders."""
    expected = len(MARKET_SLOTS) + 2
    if buckets.ndim != 3 or buckets.shape[1:] != (2, expected):
        raise ValueError(
            f"expected (B, 2, {expected}) buckets, got {tuple(buckets.shape)}"
        )
    device = buckets.device
    quantities = tensor_constant(QUANTITIES, dtype=torch.int32, device=device)[
        buckets.to(torch.int64)
    ]
    regular_types = tensor_constant(
        [_TYPE[operation] for operation, _ in MARKET_SLOTS],
        dtype=torch.int8,
        device=device,
    )
    regular_items = tensor_constant(
        [_CATALOGUES[operation].index(item) for operation, item in MARKET_SLOTS],
        dtype=torch.int8,
        device=device,
    )
    batch = buckets.shape[0]
    regular_qty = quantities[..., : len(MARKET_SLOTS)]
    regular_mask = regular_qty > 0
    regular_types = regular_types.expand(batch, 2, -1)
    regular_items = regular_items.expand(batch, 2, -1)

    hire_number = torch.arange(64, device=device).reshape(1, 1, 64)
    hire_mask = hire_number < quantities[..., HIRE_SLOT, None]
    hire_types = torch.full((batch, 2, 64), 5, dtype=torch.int8, device=device)
    hire_items = torch.full((batch, 2, 64), -1, dtype=torch.int8, device=device)
    hire_qty = torch.ones((batch, 2, 64), dtype=torch.int32, device=device)

    land_mask = buckets[..., LAND_SLOT, None] != 0
    land_types = torch.full((batch, 2, 1), 6, dtype=torch.int8, device=device)
    land_items = torch.full((batch, 2, 1), -1, dtype=torch.int8, device=device)
    land_qty = torch.ones((batch, 2, 1), dtype=torch.int32, device=device)

    present = torch.cat((regular_mask, hire_mask, land_mask), dim=-1)
    source_types = torch.cat((regular_types, hire_types, land_types), dim=-1)
    source_items = torch.cat((regular_items, hire_items, land_items), dim=-1)
    source_qty = torch.cat((regular_qty, hire_qty, land_qty), dim=-1)
    ranks = present.to(torch.int64).cumsum(dim=-1) - 1
    destination = torch.where(present & (ranks < 10), ranks, torch.full_like(ranks, 10))

    output_shape = (batch, 2, 11)
    types = torch.zeros(output_shape, dtype=torch.int8, device=device)
    items = torch.full(output_shape, -1, dtype=torch.int8, device=device)
    qty = torch.zeros(output_shape, dtype=torch.int32, device=device)
    types.scatter_(2, destination, torch.where(present, source_types, 0))
    items.scatter_(2, destination, torch.where(present, source_items, -1))
    qty.scatter_(2, destination, torch.where(present, source_qty, 0))
    return MarketActions(types[..., :10], items[..., :10], qty[..., :10])
