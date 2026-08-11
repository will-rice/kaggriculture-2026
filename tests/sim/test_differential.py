"""Differential assertion behavior tests."""
# ruff: noqa: D103

from dataclasses import replace

import pytest
from kaggle_environments import make

from kaggriculture.sim.state import pack
from tests.sim.conftest import assert_identical


def test_assert_identical_accepts_an_exact_codec_round_trip() -> None:
    environment = make("kaggriculture", configuration={"seed": 163}, debug=True)
    environment.reset(2)

    assert_identical(environment, pack([environment]), 0)


def test_assert_identical_names_the_first_corrupted_field() -> None:
    environment = make("kaggriculture", configuration={"seed": 167}, debug=True)
    environment.reset(2)
    state = pack([environment])
    money = state.money.clone()
    money[0, 0] += 1
    corrupted = replace(state, money=money)

    with pytest.raises(AssertionError, match=r"seat\[0\]\.farms\[0\]\.money"):
        assert_identical(environment, corrupted, 0)
