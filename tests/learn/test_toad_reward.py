"""Tests for Toad's shaped reward mapped onto our farm economy.

These guard the three transcription details that would corrupt a whole training
run silently: the /500 normaliser the briefing analysis omitted, the
non-negative clamp on the stock term, and the 10x terminal result riding inside
the shaped reward rather than replacing it.
"""

from collections.abc import Mapping
from typing import Any

import pytest
from kaggle_environments import make

from kaggriculture.constants import ENVIRONMENT
from kaggriculture.learn import toad_reward


@pytest.fixture(name="observation")
def _observation() -> Mapping[str, Any]:
    """Return seat 0's opening observation from a real episode."""
    environment = make(ENVIRONMENT, debug=True)
    environment.reset(2)
    return environment.state[0].observation


def test_counts_read_the_opening_position(observation: Mapping[str, Any]) -> None:
    """The five counts must match what the engine actually starts a farm with."""
    opening = toad_reward.counts(observation, 0)
    # One quadrant of a 10x10 board is unlocked at the start, and nothing is
    # planted yet, so `city` is the unlocked-tile count alone.
    assert opening.city == 25
    assert opening.unit == 1
    assert opening.fuel == 0
    assert opening.money > 0


def test_the_weights_are_the_published_ones() -> None:
    """Pin every coefficient to its literal published value.

    Written as literals on purpose. Every other test in this file states its
    expectation in terms of these constants, which makes them agree with the
    module no matter what it says -- changing NORMALISER to 1.0 left the whole
    file green until this test existed. These are the numbers read out of
    reward_spaces_lux.py:166-177 and :227, and nothing else here checks them.
    """
    assert toad_reward.GAME_RESULT_WEIGHT == 10.0
    assert toad_reward.CITY_WEIGHT == 1.0
    assert toad_reward.UNIT_WEIGHT == 0.5
    assert toad_reward.RESEARCH_WEIGHT == 0.1
    assert toad_reward.FUEL_WEIGHT == 0.005
    assert toad_reward.STEP_WEIGHT == 0.005
    assert toad_reward.FULL_WORKERS_WEIGHT == 0.0
    assert toad_reward.NORMALISER == 500.0


def test_the_opening_turn_scores_only_the_step_reward(
    observation: Mapping[str, Any],
) -> None:
    """With no counts changed, only `step` fires -- and it fires through /500.

    The literal 0.00001 is 0.005/500. Stating it as a bare number rather than
    as STEP_WEIGHT/NORMALISER is what makes this fail if either constant drifts.
    """
    reward = toad_reward.StatefulMultiReward(observation, 0)
    assert reward.step(observation, done=False) == pytest.approx(0.00001)


def test_selling_stock_is_not_punished(observation: Mapping[str, Any]) -> None:
    """The stock term is clamped at zero, as their fuel term is.

    Selling is how our game is won, and the sale is already paid for through
    `game_result`. An unclamped delta would make the shaped reward fight the
    objective every time the agent banks anything.
    """
    reward = toad_reward.StatefulMultiReward(observation, 0)
    reward.previous = toad_reward.Counts(
        city=25, unit=1, research=0, fuel=40, money=100.0
    )
    emptied = _with_shed(observation, 0)
    # Only `step` survives: the -40 stock delta is clamped away.
    assert reward.step(emptied, done=False) == pytest.approx(
        toad_reward.STEP_WEIGHT / toad_reward.NORMALISER
    )


def test_gaining_stock_is_rewarded(observation: Mapping[str, Any]) -> None:
    """The clamp must not flatten the term in the direction that counts."""
    reward = toad_reward.StatefulMultiReward(observation, 0)
    filled = _with_shed(observation, 20)
    expected = (
        20 * toad_reward.FUEL_WEIGHT + toad_reward.STEP_WEIGHT
    ) / toad_reward.NORMALISER
    assert reward.step(filled, done=False) == pytest.approx(expected)


def test_the_terminal_result_rides_inside_the_shaped_reward(
    observation: Mapping[str, Any],
) -> None:
    """Winning must pay 10x on the last turn, through the same normaliser.

    Their `game_result` is a component of the shaped reward from turn one, not a
    separate sparse phase. A reproduction that only pays it after the phase
    switch has changed the method.
    """
    reward = toad_reward.StatefulMultiReward(observation, 0)
    won = _with_money(observation, ours=9_000.0, theirs=1_000.0)
    lost = _with_money(observation, ours=1_000.0, theirs=9_000.0)
    win = toad_reward.StatefulMultiReward(observation, 0).step(won, done=True)
    loss = reward.step(lost, done=True)
    assert win - loss == pytest.approx(
        2 * toad_reward.GAME_RESULT_WEIGHT / toad_reward.NORMALISER
    )


def _with_shed(observation: Mapping[str, Any], units: int) -> dict[str, Any]:
    """Return a copy of the observation holding `units` of one good."""
    shed = dict.fromkeys(observation["private"]["shed"], 0)
    shed[next(iter(shed))] = units
    private = {**observation["private"], "shed": shed}
    return {**observation, "private": private}


def _with_money(
    observation: Mapping[str, Any], ours: float, theirs: float
) -> dict[str, Any]:
    """Return a copy of the observation with both farms' banks set."""
    farms = [dict(farm) for farm in observation["farms"]]
    farms[0]["money"] = ours
    farms[1]["money"] = theirs
    return {**observation, "farms": farms}
