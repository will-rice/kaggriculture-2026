"""Head-wise entropy accounting for the native Toad learner."""

import math

import torch

from kaggriculture.learn.encoding import IGNORE, TRANSFER_OPS
from kaggriculture.learn.toad.lightning import HeadEntropy, compute_head_entropy


def _log_probs(shape: tuple[int, ...]) -> torch.Tensor:
    """Return differentiable, uniform two-option log probabilities."""
    return torch.full(shape, -math.log(2), dtype=torch.float32, requires_grad=True)


def _entropy(
    *,
    unit_actions: torch.Tensor,
    unit_mask: torch.Tensor | None = None,
    quantity_mask: torch.Tensor | None = None,
    market_mask: torch.Tensor | None = None,
) -> tuple[HeadEntropy, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    leading = unit_actions.shape
    units = _log_probs((*leading, 2))
    quantities = _log_probs((*leading, 2))
    market = _log_probs((*leading[:-1], 3, 2))
    return (
        compute_head_entropy(
            unit_log_probs=units,
            quantity_log_probs=quantities,
            market_log_probs=market,
            unit_masks=(
                torch.ones_like(units, dtype=torch.bool)
                if unit_mask is None
                else unit_mask
            ),
            quantity_masks=(
                torch.ones_like(quantities, dtype=torch.bool)
                if quantity_mask is None
                else quantity_mask
            ),
            market_masks=(
                torch.ones_like(market, dtype=torch.bool)
                if market_mask is None
                else market_mask
            ),
            unit_actions=unit_actions,
        ),
        (units, quantities, market),
    )


def test_quantity_entropy_counts_only_transfer_decisions() -> None:
    """A quantity bucket ignored by every selected operation has no gradient."""
    entropy, (_units, quantities, _market) = _entropy(
        unit_actions=torch.tensor([[0, 1], [2, IGNORE]])
    )

    assert entropy.quantity.valid.item() == 0
    assert entropy.quantity.sum.item() == 0.0
    entropy.quantity.sum.backward()
    assert quantities.grad is not None
    assert not quantities.grad.any()


def test_padded_unit_slots_do_not_count_toward_operation_or_quantity_entropy() -> None:
    """Placeholder logits and a placeholder transfer label are not decisions."""
    transfer = TRANSFER_OPS.index(True)
    entropy, (units, quantities, _market) = _entropy(
        unit_actions=torch.tensor([[0, IGNORE], [0, IGNORE]])
    )
    padded_actions = torch.tensor([[0, transfer], [0, transfer]])
    padded_entropy, padded_logits = _entropy(unit_actions=padded_actions)

    assert entropy.operation.valid.item() == 2
    assert entropy.quantity.valid.item() == 0
    torch.testing.assert_close(entropy.operation.sum, torch.tensor(2 * math.log(2)))
    assert units.grad is None
    entropy.operation.sum.backward()
    assert units.grad is not None
    assert not units.grad[:, 1].any()
    assert quantities.grad is None
    # This is the counterfactual that would be wrong if IGNORE were treated as
    # a transfer; a real transfer slot, unlike the padded one above, counts.
    assert padded_entropy.quantity.valid.item() == 2
    assert padded_logits[1].grad is None


def test_all_padding_and_empty_market_have_exact_safe_zero_statistics() -> None:
    """Empty reductions neither leak NaNs nor manufacture valid decisions."""
    unit_actions = torch.full((2, 3), IGNORE, dtype=torch.int64)
    units = _log_probs((2, 3, 2))
    quantities = _log_probs((2, 3, 2))
    market = _log_probs((2, 0, 2))
    entropy = compute_head_entropy(
        unit_log_probs=units,
        quantity_log_probs=quantities,
        market_log_probs=market,
        unit_masks=torch.zeros_like(units, dtype=torch.bool),
        quantity_masks=torch.zeros_like(quantities, dtype=torch.bool),
        market_masks=torch.zeros_like(market, dtype=torch.bool),
        unit_actions=unit_actions,
    )

    for stat in (entropy.operation, entropy.quantity, entropy.market):
        assert stat.valid.item() == 0
        assert stat.sum.item() == 0.0
        assert torch.isfinite(stat.sum)


def test_market_entropy_counts_each_real_market_slot_once() -> None:
    """Market has no unit-style padding: every present slot is a decision."""
    entropy, _logits = _entropy(unit_actions=torch.tensor([[0, 0], [0, 0]]))

    assert entropy.market.valid.item() == 6
    torch.testing.assert_close(entropy.market.sum, torch.tensor(6 * math.log(2)))


def test_masked_illegal_logits_are_irrelevant_to_entropy_and_gradients() -> None:
    """Changing forbidden logits must not change a valid row's entropy."""
    unit_actions = torch.tensor([[0]])
    units = torch.tensor([[[0.0, 123.0]]], dtype=torch.float32, requires_grad=True)
    quantities = _log_probs((1, 1, 2))
    market = _log_probs((1, 1, 2))
    entropy = compute_head_entropy(
        unit_log_probs=units,
        quantity_log_probs=quantities,
        market_log_probs=market,
        unit_masks=torch.tensor([[[True, False]]]),
        quantity_masks=torch.ones_like(quantities, dtype=torch.bool),
        market_masks=torch.ones_like(market, dtype=torch.bool),
        unit_actions=unit_actions,
    )

    assert entropy.operation.sum.item() == 0.0
    entropy.operation.sum.backward()
    assert units.grad is not None
    assert units.grad[..., 1].item() == 0.0


def test_entropy_flattens_batch_and_time_dimensions_in_fp32() -> None:
    """All leading batch dimensions count, and reductions stay FP32 in autocast."""
    transfer = TRANSFER_OPS.index(True)
    actions = torch.tensor([[[transfer, 0]], [[transfer, IGNORE]]])
    units = _log_probs((2, 1, 2, 2)).to(dtype=torch.bfloat16).detach().requires_grad_()
    quantities = (
        _log_probs((2, 1, 2, 2)).to(dtype=torch.bfloat16).detach().requires_grad_()
    )
    market = _log_probs((2, 1, 3, 2)).to(dtype=torch.bfloat16).detach().requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        entropy = compute_head_entropy(
            unit_log_probs=units,
            quantity_log_probs=quantities,
            market_log_probs=market,
            unit_masks=torch.ones_like(units, dtype=torch.bool),
            quantity_masks=torch.ones_like(quantities, dtype=torch.bool),
            market_masks=torch.ones_like(market, dtype=torch.bool),
            unit_actions=actions,
        )

    assert entropy.operation.valid.item() == 3
    assert entropy.quantity.valid.item() == 2
    assert entropy.market.valid.item() == 6
    assert entropy.operation.sum.dtype == torch.float32
    assert entropy.quantity.sum.dtype == torch.float32
    assert entropy.market.sum.dtype == torch.float32
