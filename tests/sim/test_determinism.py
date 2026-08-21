"""Determinism and batch-independence tests."""
# ruff: noqa: ANN001, D103

from dataclasses import fields

import torch

from kaggriculture.learn.encoding import MAX_UNITS
from kaggriculture.sim.config import Config
from kaggriculture.sim.engine import MarketActions, reset, step, unit_quantity_ones


def _assert_tensor_identical(left, right) -> None:
    for field in fields(left):
        assert torch.equal(getattr(left, field.name), getattr(right, field.name)), (
            field.name
        )


def test_reset_and_step_are_self_reproducible() -> None:
    seeds = torch.tensor([223, 227])
    left = reset(Config(), seeds)
    right = reset(Config(), seeds)
    units = torch.zeros(2, 2, MAX_UNITS, dtype=torch.int16)
    markets = MarketActions.empty(2)

    left = step(left, units, markets, unit_quantity_ones(2))
    right = step(right, units, markets, unit_quantity_ones(2))

    _assert_tensor_identical(left, right)


def test_batch_members_match_their_singleton_trajectories() -> None:
    seeds = torch.tensor([229, 233])
    batched = reset(Config(), seeds)
    units = torch.zeros(2, 2, MAX_UNITS, dtype=torch.int16)
    batched = step(batched, units, MarketActions.empty(2), unit_quantity_ones(2))

    for batch, seed in enumerate(seeds):
        singleton = reset(Config(), seed.reshape(1))
        singleton = step(
            singleton,
            torch.zeros(1, 2, MAX_UNITS, dtype=torch.int16),
            MarketActions.empty(1),
            unit_quantity_ones(1),
        )
        for field in fields(singleton):
            assert torch.equal(
                getattr(batched, field.name)[batch : batch + 1],
                getattr(singleton, field.name),
            ), field.name
