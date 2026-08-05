"""Tests for the reference policy: shape, size and the inference budget."""

import time

import torch

from kaggriculture.learn.encoding import (
    BOARD,
    IGNORE,
    MAX_UNITS,
    SCALARS,
    TILE_PLANES,
    UNIT_OPS,
)
from kaggriculture.learn.model import Policy


def test_forward_returns_one_distribution_per_unit() -> None:
    """Each unit picks its own op, so the head is per-unit, not per-board."""
    model = Policy()
    board = torch.zeros(2, TILE_PLANES, BOARD, BOARD)
    scalars = torch.zeros(2, SCALARS)

    logits = model(board, scalars)

    assert logits.shape == (2, MAX_UNITS, len(UNIT_OPS))


def test_the_model_fits_the_size_every_winner_used() -> None:
    """Verified winners span 1M-20M parameters; nothing above 20M is supported."""
    parameters = sum(p.numel() for p in Policy().parameters())

    assert 1e6 < parameters < 20e6


def test_a_turn_fits_the_sandbox_budget_at_two_threads() -> None:
    """The sandbox has two cores and one second a turn, not this workstation.

    Measured at two threads because timing on 64 cores flatters the sandbox by
    an order of magnitude. The Phase 0 probe measured a 10M-parameter trunk at
    ~45ms of the 1000ms budget; this asserts a wide margin, not that figure.
    """
    torch.set_num_threads(2)
    model = Policy().eval()
    board = torch.zeros(1, TILE_PLANES, BOARD, BOARD)
    scalars = torch.zeros(1, SCALARS)

    with torch.no_grad():
        model(board, scalars)
        start = time.perf_counter()
        for _ in range(10):
            model(board, scalars)
        elapsed = (time.perf_counter() - start) / 10

    assert elapsed < 0.25, f"{elapsed * 1000:.0f}ms per turn leaves no margin"


def test_the_market_reaches_the_trunk() -> None:
    """Prices decide this game; a scalar branch that is ignored is a silent bug."""
    model = Policy().eval()
    board = torch.zeros(1, TILE_PLANES, BOARD, BOARD)

    with torch.no_grad():
        cheap = model(board, torch.zeros(1, SCALARS))
        rich = model(board, torch.ones(1, SCALARS))

    assert not torch.allclose(cheap, rich)


def test_ignored_label_slots_do_not_affect_the_loss() -> None:
    """Most unit slots are padding; cross_entropy must not learn from them.

    A batch pairs the model's logits with labels where every slot past the
    first three is ``IGNORE`` (unhired hands this turn). Perturbing the
    logits only at those padded positions must leave the cross-entropy loss
    bit-identical -- if it moved, the model would be trained to predict
    padding rather than real unit choices.
    """
    torch.manual_seed(0)
    model = Policy().eval()
    board = torch.randn(2, TILE_PLANES, BOARD, BOARD)
    scalars = torch.randn(2, SCALARS)

    with torch.no_grad():
        logits = model(board, scalars)

    labels = torch.full((2, MAX_UNITS), IGNORE, dtype=torch.int64)
    labels[:, :3] = 0

    perturbed = logits.clone()
    perturbed[:, 3:, :] = torch.randn(2, MAX_UNITS - 3, len(UNIT_OPS))

    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, len(UNIT_OPS)), labels.reshape(-1), ignore_index=IGNORE
    )
    perturbed_loss = torch.nn.functional.cross_entropy(
        perturbed.reshape(-1, len(UNIT_OPS)), labels.reshape(-1), ignore_index=IGNORE
    )

    assert torch.equal(loss, perturbed_loss)
