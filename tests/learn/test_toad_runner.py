"""Compatibility tests for Toad's retained read-only metric helpers."""

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
from kaggriculture.learn.rollout import Trajectory
from kaggriculture.learn.scripts import toad


def _trajectory(turns: int = 4, final_margin: float = 1.0) -> Trajectory:
    """Return the minimal recorded seat consumed by the metric helpers."""
    slots = len(MARKET_SLOTS) + 2
    dones = torch.zeros(turns, dtype=torch.bool)
    dones[-1] = True
    rewards = torch.full((turns,), 0.1)
    return Trajectory(
        board=torch.zeros(turns, TILE_PLANES, 10, 10),
        scalars=torch.zeros(turns, SCALARS),
        positions=torch.zeros(turns, MAX_UNITS, dtype=torch.int64),
        unit_actions=torch.zeros(turns, MAX_UNITS, dtype=torch.int64),
        unit_quantities=torch.ones(turns, MAX_UNITS, dtype=torch.int64),
        market_actions=torch.zeros(turns, slots, dtype=torch.int64),
        unit_masks=torch.ones(turns, MAX_UNITS, len(UNIT_OPS), dtype=torch.bool),
        unit_quantity_masks=torch.ones(
            turns, MAX_UNITS, len(QUANTITIES), dtype=torch.bool
        ),
        market_masks=torch.ones(turns, slots, len(QUANTITIES), dtype=torch.bool),
        log_probs=torch.zeros(turns),
        values=torch.zeros(turns),
        rewards=rewards,
        own=rewards,
        shaped=rewards,
        shaped_money=rewards,
        margin=rewards,
        sparse=rewards,
        potentials=torch.zeros(turns, 1),
        dones=dones,
        final_margin=final_margin,
        final_bank=10.0,
        final_capital=2.0,
        illegal=0,
        sales=1.0,
        units_sold=1.0,
        mean_sale_price=1.0,
        realisation=1.0,
        bought=1.0,
    )


def _record(
    mirror_batch: list[Trajectory], econ_batch: list[Trajectory]
) -> dict[str, object]:
    """Call the read-only metric adapter with fixed clocks and loss terms."""
    terms = {
        "vtrace_pg": 0.0,
        "upgo_pg": 0.0,
        "baseline": 0.0,
        "entropy": 0.0,
        "teacher": 0.0,
        "total": 0.0,
        "baseline_passes": 0.0,
    }
    return toad._record(
        mirror_batch,
        econ_batch,
        "shaped",
        update=1,
        steps=10,
        hours=0.1,
        lr=1e-4,
        warming=False,
        warmup_left=0,
        terms=terms,
    )


def test_every_logged_metric_carries_a_role_prefix_and_definition() -> None:
    """Every retained metric belongs to a dashboard role and is documented."""
    record = _record(
        mirror_batch=[_trajectory(), _trajectory()], econ_batch=[_trajectory()]
    )
    assert record
    assert all(key.startswith(toad.METRIC_PREFIXES) for key in record)
    assert set(record) == set(toad.METRIC_DEFINITIONS)


def test_mirror_decisive_rate_is_the_raw_win_rate_doubled() -> None:
    """Two recorded self-play seats make all-decisive read as one, not one-half."""
    decisive = [_trajectory(final_margin=1.0), _trajectory(final_margin=-1.0)]
    assert _record(decisive, [])["diag/mirror_decisive_rate"] == pytest.approx(1.0)

    tied = [_trajectory(final_margin=0.0), _trajectory(final_margin=0.0)]
    assert _record(tied, [])["diag/mirror_decisive_rate"] == pytest.approx(0.0)
