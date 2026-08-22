"""Tests for Toad's shaped reward mapped onto our farm economy.

These guard the three transcription details that would corrupt a whole training
run silently: the /500 normaliser the briefing analysis omitted, the
non-negative clamp on the stock term, and the 10x terminal result riding inside
the shaped reward rather than replacing it.
"""

import dataclasses
from collections.abc import Mapping
from typing import Any

import pytest
import torch
from kaggle_environments import make

from kaggriculture.constants import ENVIRONMENT
from kaggriculture.learn import toad_reward
from kaggriculture.learn.model import Policy
from kaggriculture.learn.rollout import rollout


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
        city=25, unit=1, research=0, fuel=40, capital=0, money=100.0, opponent=0.0
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
    base = toad_reward.Counts(
        city=25, unit=1, research=0, fuel=10, capital=0, money=1000.0, opponent=0.0
    )
    after = toad_reward.Counts(
        capital=0 + int(deltas.get("capital", 0)),
        city=25 + int(deltas.get("city", 0)),
        unit=1 + int(deltas.get("unit", 0)),
        research=0 + int(deltas.get("research", 0)),
        fuel=10 + int(deltas.get("fuel", 0)),
        money=1000.0 + deltas.get("money", 0.0),
        opponent=0.0 + deltas.get("opponent", 0.0),
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


def test_the_clamped_money_term_pays_for_a_losing_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The clamp is what made arm W pump, and this pins the mechanism.

    Buy 648 coins of stock, sell it back for 600. The farm is 48 coins poorer,
    but the clamp forgives the purchase and pays the sale, so the shaped reward
    is positive. Arm W did exactly this: money_term rose 0.0064 -> 0.028 with
    gross_purchases tracking it and bank_mean pinned at zero.
    """
    monkeypatch.delenv(toad_reward.MONEY_SIGNED_ENV, raising=False)
    buy = toad_reward.Counts(
        city=25, unit=1, research=0, fuel=18, capital=0, money=352.0, opponent=0.0
    )
    sell = toad_reward.Counts(
        city=25, unit=1, research=0, fuel=10, capital=0, money=952.0, opponent=0.0
    )
    opening = toad_reward.Counts(
        city=25, unit=1, research=0, fuel=10, capital=0, money=1000.0, opponent=0.0
    )
    total = float(
        toad_reward.shaped([opening, buy, sell], won=0.0, money_weight=0.01).sum()
    )
    # Net -48 coins, yet the reward is positive:
    #   sale 600 * 0.01 = 6.0 | stock +8 * 0.005 = 0.04 | 2 steps = 0.01
    #   6.05 / 500 = 0.0121
    assert total > 0.0
    assert total == pytest.approx(0.0121)


def test_the_signed_money_term_punishes_a_losing_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unclamped, the deltas telescope to net coins, so the pump cannot pay.

    Same round trip, now worth the net -48 coins it actually cost:
      net -48 * 0.01 = -0.48 | stock +8 * 0.005 = 0.04 | 2 steps = 0.01
      -0.43 / 500 = -0.00086
    """
    monkeypatch.setenv(toad_reward.MONEY_SIGNED_ENV, "1")
    buy = toad_reward.Counts(
        city=25, unit=1, research=0, fuel=18, capital=0, money=352.0, opponent=0.0
    )
    sell = toad_reward.Counts(
        city=25, unit=1, research=0, fuel=10, capital=0, money=952.0, opponent=0.0
    )
    opening = toad_reward.Counts(
        city=25, unit=1, research=0, fuel=10, capital=0, money=1000.0, opponent=0.0
    )
    total = float(
        toad_reward.shaped([opening, buy, sell], won=0.0, money_weight=0.01).sum()
    )
    assert total < 0.0
    assert total == pytest.approx(-0.00086)


# The reference operating point on kaggle-environments 1.32.6, measured over
# economic_policy mirror seasons at seeds 7 and 11. Both seats end holding
# fifteen animals; the `fuel` term totals 0.0092-0.0098 for the produce those
# fields and that herd actually yielded, and the shaped reward excluding capital
# totals 0.146-0.157. Written as literals because they are what sizes
# CAPITAL_WEIGHT, and a weight sized against numbers nobody can see is a weight
# nobody can check.
REFERENCE_ANIMALS = 15
REFERENCE_FUEL_TERM = 0.0092
# startingMoney 3000 over GOOSE at 300: the most capital reachable without
# selling anything at all.
FLOAT_ANIMALS = 10


def test_the_capital_weight_stays_under_the_produce_it_enables() -> None:
    """Cap the weight below the produce it enables, with no margin term to help.

    THE INEQUALITY THE CAPITAL WEIGHT RESTS ON. In the margin arm this term was
    safe because the margin charged the animal's full 300-500 coins, six to nine
    times what capital paid. This arm has no
    margin and no money term, so coins carry no reward at all and that guard is
    simply absent. The replacement is a magnitude bound: acquiring the asset must
    pay less than operating it, or the reward points at buying rather than at
    farming -- the shape behind all four of this project's pump defects.

    Toad's published 1.0 for this slot fails the bound by 3x, which is why the
    slot's own number is not reused.
    """
    ceiling = REFERENCE_FUEL_TERM * toad_reward.NORMALISER / REFERENCE_ANIMALS
    assert ceiling == pytest.approx(0.3067, abs=1e-4)
    assert toad_reward.CAPITAL_WEIGHT == 0.05
    assert toad_reward.CAPITAL_WEIGHT < ceiling
    # An order of magnitude inside it, not a hair inside it.
    assert toad_reward.CAPITAL_WEIGHT * 5 < ceiling
    # And the number actually reused for this slot would not have cleared it.
    assert toad_reward.CITY_WEIGHT > ceiling


def test_spending_the_whole_opening_float_on_animals_cannot_rival_winning() -> None:
    """A season of buying and never selling must stay far below the terminal result.

    Capital is bounded by coins irreversibly spent, so the reachable total
    without ever completing a sale is the opening float divided by the cheapest
    animal. That number has to be small against `game_result`, or a policy can
    out-score winning by liquidating its float into livestock on day one.
    """
    hoarded = float(
        toad_reward.shaped(_series(capital=FLOAT_ANIMALS), won=0.0).sum()
    ) - float(toad_reward.shaped(_series(), won=0.0).sum())
    terminal = toad_reward.GAME_RESULT_WEIGHT / toad_reward.NORMALISER

    assert hoarded == pytest.approx(0.001)
    assert terminal == pytest.approx(0.02)
    # At most a twentieth of what winning the match is worth. Stated on the
    # constants rather than on the float32 rewards, which land a few ulps over
    # the boundary and would make the guard a coin toss rather than a bound.
    assert (
        FLOAT_ANIMALS * toad_reward.CAPITAL_WEIGHT * 20
        <= toad_reward.GAME_RESULT_WEIGHT
    )


def test_carrying_an_animal_out_of_the_shed_and_onto_a_tile_scores_zero() -> None:
    """The one capital cycle the engine permits must be exactly reward-neutral.

    PICKUP takes the animal out of the shed and PLACE puts it on its structure;
    `capital` counts both halves, so the pair telescopes. This is only true
    because the term is UNCLAMPED. Clamped like `fuel`, the negative leg would be
    forgiven and one animal would pay forever -- so the test drives the cycle
    rather than asserting on the source, and a clamp added later breaks it.
    """
    shed = toad_reward.Counts(
        city=25, unit=1, research=0, fuel=10, capital=1, money=1000.0, opponent=0.0
    )
    carried = dataclasses.replace(shed, capital=0)
    placed = dataclasses.replace(shed, capital=1)

    cycle = toad_reward.shaped([shed, carried, placed], won=0.0)
    steps = 2 * toad_reward.STEP_WEIGHT / toad_reward.NORMALISER

    # An absolute tolerance, because the two legs are float32 and cancel to a
    # number far smaller than themselves; a relative tolerance on 2e-05 sits
    # below float32's own resolution at 1e-04 and would fail on rounding alone.
    assert float(cycle.sum()) == pytest.approx(steps, abs=1e-9)
    # The negative leg lands first, so the cycle can never be entered for profit.
    assert float(cycle[0]) < float(cycle[1])


def test_an_empty_structure_earns_nothing_under_the_shaped_reward() -> None:
    """BUILD_COOP and BUILD_PASTURE are free, so they must not reach the reward.

    Verified against the engine: both ops write a tile and charge nothing. At
    this weight a count of structures rather than animals would pay 0.01 a season
    off roughly a hundred buildable tiles -- half the terminal result, for coins
    nobody spent.
    """
    environment = make(ENVIRONMENT, debug=True)
    environment.reset(2)
    observation = environment.state[0].observation
    tiles = [list(row) for row in observation["farms"][0]["tiles"]]
    built = 0
    for row in tiles:
        for index, tile in enumerate(row):
            if tile is None:
                row[index] = {"kind": "PASTURE"}
                built += 1
    assert built > 0
    farms = [dict(farm) for farm in observation["farms"]]
    farms[0] = {**farms[0], "tiles": tiles}

    before = toad_reward.counts(observation, 0)
    after = toad_reward.counts({**observation, "farms": farms}, 0)

    assert after.capital == before.capital
    assert float(toad_reward.shaped([before, after], won=0.0).sum()) == pytest.approx(
        toad_reward.STEP_WEIGHT / toad_reward.NORMALISER
    )


def test_sparse_is_zero_everywhere_but_the_final_turn() -> None:
    """Phases 2+ train on the game result alone.

    Sparse means sparse: a non-zero mid-episode entry would be a shaped
    reward wearing the name, and the phase boundary would stop meaning what
    the recipe says. Played through a real episode, via ``rollout``, rather
    than a hand-built series -- ``toad_reward.sparse`` takes the same
    ``series``/``won`` shape ``margin`` and ``shaped`` do, and the wiring that
    hands it a real one is exactly what a hand-built series cannot exercise.
    """
    torch.manual_seed(0)
    policy = Policy(blocks=1, channels=32).eval()
    trajectory = rollout(policy, "starter", seed=0)

    theirs = trajectory.final_bank - trajectory.final_margin
    expected = toad_reward.rank(trajectory.final_bank, theirs)

    assert torch.all(trajectory.sparse[:-1] == 0.0)
    assert float(trajectory.sparse[-1]) == expected
    assert float(trajectory.sparse.sum()) == expected
