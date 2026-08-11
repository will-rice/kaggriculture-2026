"""Tests that go through the runner, not just the loss it calls.

The value warmup shipped with ``baseline_only`` passed to a ``_step`` that did
not accept it. Every loss test still passed, because they call ``losses``
directly and never cross the runner boundary -- so arm C' launched, crashed on
its first batch, and burned the slot. These call what the runner calls.
"""

from typing import Any

import pytest
import torch

from kaggriculture.learn.encoding import (
    MARKET_SLOTS,
    MAX_UNITS,
    QUANTITIES,
    SCALARS,
    TILE_PLANES,
    UNIT_OPS,
)
from kaggriculture.learn.model import Policy
from kaggriculture.learn.scripts import toad_phase1
from kaggriculture.learn.toad_loss import ADAM_EPS, LEARNING_RATE


def _segment(turns: int = 16) -> dict[str, torch.Tensor]:
    """Return one synthetic unroll shaped exactly as ``_segments`` produces.

    The observed fields carry one row more than the acted ones. That extra state
    is the one the value target bootstraps from -- it is deliberately outside the
    segment's own decisions -- so a stand-in built at equal lengths would not be
    the shape the runner is handed. See ``toad_phase1._segments``.
    """
    slots = len(MARKET_SLOTS) + 2
    return {
        "board": torch.randn(turns + 1, TILE_PLANES, 10, 10),
        "scalars": torch.randn(turns + 1, SCALARS),
        "positions": torch.zeros(turns + 1, MAX_UNITS, dtype=torch.int64),
        "unit_actions": torch.zeros(turns, MAX_UNITS, dtype=torch.int64),
        "market_actions": torch.zeros(turns, slots, dtype=torch.int64),
        "unit_masks": torch.ones(turns, MAX_UNITS, len(UNIT_OPS), dtype=torch.bool),
        "market_masks": torch.ones(turns, slots, len(QUANTITIES), dtype=torch.bool),
        "log_probs": torch.full((turns,), -1.0),
        "shaped": torch.randn(turns) * 0.01,
        "shaped_money": torch.randn(turns) * 0.01,
        "dones": torch.zeros(turns, dtype=torch.bool),
    }


def _policy() -> tuple[Policy, torch.optim.Optimizer]:
    """Return a tiny policy and its optimizer."""
    policy = Policy(blocks=1, channels=16, value_bound=toad_phase1.VALUE_BOUND)
    return policy, torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS)


def test_the_runner_can_take_a_warmup_step() -> None:
    """During warmup the runner's own step must optimise the baseline alone."""
    policy, optimizer = _policy()
    segments = [_segment() for _ in range(toad_phase1.BATCH_SEGMENTS)]
    warm = toad_phase1._step(
        policy, optimizer, segments, "cpu", "shaped_money", baseline_only=True
    )
    assert warm["total"] == warm["baseline"]


def test_the_runner_takes_a_full_step_after_warmup() -> None:
    """Outside warmup the policy terms must be back in the total."""
    policy, optimizer = _policy()
    segments = [_segment() for _ in range(toad_phase1.BATCH_SEGMENTS)]
    full = toad_phase1._step(
        policy, optimizer, segments, "cpu", "shaped_money", baseline_only=False
    )
    assert full["total"] != full["baseline"]


def test_the_warmup_budget_is_counted_down_in_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_update` must report how many batches it consumed, or warmup never ends."""
    policy, optimizer = _policy()

    class _Fake:
        dones = torch.zeros(64, dtype=torch.bool)

    segments = [_segment() for _ in range(8)]
    # `monkeypatch` rather than assign-and-restore: it undoes the patch even if
    # the assertions below raise, which the hand-rolled `try/finally` did too but
    # only as long as nobody edited it.
    monkeypatch.setattr(toad_phase1, "_segments", lambda trajectory: segments)  # noqa: ARG005
    # `_update` reads only `dones` off each trajectory, which is the whole point
    # of the stand-in; it is not a `Trajectory` and does not need to be.
    batch: list[Any] = [_Fake()]
    terms, consumed = toad_phase1._update(
        policy, optimizer, batch, "cpu", "shaped_money", warmup_left=10**9
    )
    assert consumed == len(segments) // toad_phase1.BATCH_SEGMENTS
    # Every batch was inside the warmup budget, so the mean total is the mean baseline.
    assert terms["total"] == terms["baseline"]
