"""Simulator reset tests."""
# ruff: noqa: D103

import pytest
import torch
from kaggle_environments import make

from kaggriculture.sim.config import Config
from kaggriculture.sim.engine import reset
from kaggriculture.sim.state import UnsupportedConfiguration, unpack


def test_reset_matches_reference_for_each_seed() -> None:
    seeds = torch.tensor([5, 31], dtype=torch.int64)

    state = reset(Config(), seeds)

    for batch, seed in enumerate(seeds.tolist()):
        environment = make("kaggriculture", configuration={"seed": seed}, debug=True)
        environment.reset(2)
        for seat in range(2):
            expected = dict(environment.state[seat].observation)
            expected.pop("remainingOverageTime", None)
            assert unpack(state, batch, seat) == expected


def test_reset_rejects_non_competition_configuration() -> None:
    with pytest.raises(UnsupportedConfiguration, match="turns_per_day"):
        reset(Config(turns_per_day=12), torch.tensor([1]))
