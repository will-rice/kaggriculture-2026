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


def _series(**deltas: float) -> list[toad_reward.Counts]:
    """Return a two-state counts series differing by the given deltas."""
    base = toad_reward.Counts(city=25, unit=1, research=0, fuel=10, money=1000.0)
    after = toad_reward.Counts(
        city=25 + int(deltas.get("city", 0)),
        unit=1 + int(deltas.get("unit", 0)),
        research=0 + int(deltas.get("research", 0)),
        fuel=10 + int(deltas.get("fuel", 0)),
        money=1000.0 + deltas.get("money", 0.0),
    )
    return [base, after]


def test_the_money_component_is_off_by_default() -> None:
    """The baseline reward must be exactly Toad's five components.

    The running baseline was measured under this default, and a money term
    leaking into it would silently invalidate the comparison phase-1b exists to
    make.
    """
    # A pure sale: 648 coins in, 8 units of stock out, nothing else changed.
    series = _series(money=648.0, fuel=-8)
    baseline = toad_reward.shaped(series, won=0.0)
    # step alone: 0.005 / 500. The stock delta is clamped away and money is off.
    assert float(baseline[0]) == pytest.approx(0.00001)


def test_a_sale_turn_scores_positive_under_phase_1b() -> None:
    """Selling must pay, and pay distinguishably -- not merely fail to hurt.

    This is the whole point of phase-1b. Under the baseline the same turn is
    worth 1e-05, which is the step reward and nothing else: Toad's components
    are blind to the act that banks the coins.

      money 648 * 0.001 = 0.648
      fuel  max(-8, 0)  = 0
      step              = 0.005
      total 0.653 / 500 = 0.001306
    """
    series = _series(money=648.0, fuel=-8)
    phase_1b = toad_reward.shaped(
        series, won=0.0, money_weight=toad_reward.MONEY_WEIGHT
    )
    assert float(phase_1b[0]) == pytest.approx(0.001306)
    # Strictly greater than the baseline, by a hundredfold, on the same turn.
    assert float(phase_1b[0]) > float(toad_reward.shaped(series, won=0.0)[0])


def test_spending_coins_is_not_punished() -> None:
    """The money delta is clamped, as their fuel delta is.

    Buying seeds, hands or land is how the chain advances, and the return on it
    is already paid when the goods are sold. An unclamped term would make the
    shaped reward fight investment.
    """
    series = _series(money=-500.0)
    reward = toad_reward.shaped(series, won=0.0, money_weight=toad_reward.MONEY_WEIGHT)
    assert float(reward[0]) == pytest.approx(0.00001)


def test_the_money_weight_is_the_derived_one() -> None:
    """Pin the weight as a literal, for the reason the other constants are pinned."""
    assert toad_reward.MONEY_WEIGHT == 0.001


def test_the_clamped_money_term_pays_for_a_losing_round_trip(monkeypatch) -> None:
    """The clamp is what made arm W pump, and this pins the mechanism.

    Buy 648 coins of stock, sell it back for 600. The farm is 48 coins poorer,
    but the clamp forgives the purchase and pays the sale, so the shaped reward
    is positive. Arm W did exactly this: money_term rose 0.0064 -> 0.028 with
    gross_purchases tracking it and bank_mean pinned at zero.
    """
    monkeypatch.delenv(toad_reward.MONEY_SIGNED_ENV, raising=False)
    buy = toad_reward.Counts(city=25, unit=1, research=0, fuel=18, money=352.0)
    sell = toad_reward.Counts(city=25, unit=1, research=0, fuel=10, money=952.0)
    opening = toad_reward.Counts(city=25, unit=1, research=0, fuel=10, money=1000.0)
    total = float(toad_reward.shaped([opening, buy, sell], won=0.0, money_weight=0.01).sum())
    # Net -48 coins, yet the reward is positive:
    #   sale 600 * 0.01 = 6.0 | stock +8 * 0.005 = 0.04 | 2 steps = 0.01
    #   6.05 / 500 = 0.0121
    assert total > 0.0
    assert total == pytest.approx(0.0121)


def test_the_signed_money_term_punishes_a_losing_round_trip(monkeypatch) -> None:
    """Unclamped, the deltas telescope to net coins, so the pump cannot pay.

    Same round trip, now worth the net -48 coins it actually cost:
      net -48 * 0.01 = -0.48 | stock +8 * 0.005 = 0.04 | 2 steps = 0.01
      -0.43 / 500 = -0.00086
    """
    monkeypatch.setenv(toad_reward.MONEY_SIGNED_ENV, "1")
    buy = toad_reward.Counts(city=25, unit=1, research=0, fuel=18, money=352.0)
    sell = toad_reward.Counts(city=25, unit=1, research=0, fuel=10, money=952.0)
    opening = toad_reward.Counts(city=25, unit=1, research=0, fuel=10, money=1000.0)
    total = float(toad_reward.shaped([opening, buy, sell], won=0.0, money_weight=0.01).sum())
    assert total < 0.0
    assert total == pytest.approx(-0.00086)
