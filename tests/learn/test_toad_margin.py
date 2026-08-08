"""Tests for the margin reward -- the competition's win condition as a reward.

Every previous arm this project ran maximised its own bank, and corpus mining
then measured bank against ladder rating at Pearson -0.043. The reward under
test here counts a different coin, so these guards are about the three ways
that substitution goes silently wrong:

* the margin failing to telescope to the terminal margin, which would mean the
  reward is a proxy for the objective rather than the objective;
* the absolute term vanishing, which turns the reward into a zero-sum game
  whose mirror self-play fixed point is measured at 4 coins of 200,000;
* a term that pays for a *state* surviving into turn 719, which is the
  un-zeroed potential (Grzes, AAMAS 2017) that already cost this project a run
  by making refusal to sell shaped-optimal.

Every expectation below is written against a hand-computed number rather than
against the module's own constants. A round of reward tests here previously
passed with the constant under test set to 1.0, because each test stated its
expectation in terms of the thing it was checking.
"""

import pytest
import torch

from kaggriculture.constants import ANIMALS
from kaggriculture.learn import toad_reward


def _counts(money: float, opponent: float, **rest: int) -> toad_reward.Counts:
    """Return a counts state, defaulting every non-monetary field to a constant.

    The defaults matter: they are held equal across a series on purpose, so that
    any test whose numbers move has moved them through money alone.
    """
    return toad_reward.Counts(
        city=rest.get("city", 25),
        unit=rest.get("unit", 1),
        research=rest.get("research", 0),
        fuel=rest.get("fuel", 0),
        capital=rest.get("capital", 0),
        money=money,
        opponent=opponent,
    )


def test_the_weights_are_the_chosen_ones() -> None:
    """Pin both coefficients to their literal values.

    Written as literals, and the only test in this file that mentions the
    constants at all. Everything else states hand-computed numbers, so this is
    the single place a silent re-weighting has to get past.
    """
    assert toad_reward.MARGIN_WEIGHT == 0.001
    assert toad_reward.ABSOLUTE_WEIGHT == 0.05
    assert toad_reward.GAME_RESULT_WEIGHT == 10.0
    assert toad_reward.NORMALISER == 500.0


def test_the_margin_term_telescopes_to_the_terminal_margin() -> None:
    """The 719 per-turn values must sum to the final margin, exactly.

    This is the property that makes the reward the objective rather than a
    proxy for it. The series wanders -- both farms earn and spend on different
    turns -- so a reward that tracked anything other than the difference would
    land somewhere else.
    """
    ours = [3000.0, 3400.0, 2900.0, 5000.0, 4800.0, 9000.0]
    theirs = [3000.0, 3100.0, 4000.0, 3800.0, 6000.0, 5500.0]
    series = [_counts(o, t) for o, t in zip(ours, theirs, strict=True)]
    rewards = toad_reward.margin(series, won=0.0)

    final_margin = ours[-1] - theirs[-1]
    # No capital is acquired anywhere in this series, so the margin term is the
    # whole reward: 3,500 coins at 0.001, over 500.
    assert final_margin == 3500.0
    assert float(rewards.sum()) == pytest.approx(0.001 * final_margin / 500.0)
    assert float(rewards.sum()) == pytest.approx(0.007)
    assert rewards.shape == (5,)


def test_a_coin_denied_is_worth_exactly_a_coin_earned() -> None:
    """The win condition is relative, so the two directions must price alike.

    "A coin denied to the opponent is worth exactly a coin earned" is the whole
    content of maximising a difference, and the absolute term is a COUNT
    precisely so that it does not quietly tilt this back toward our own bank.
    Both directions are checked, because a reward that had lost the opponent's
    bank entirely would still pass a test that only looked at ours.
    """
    earned = toad_reward.margin(
        [_counts(3000.0, 3000.0), _counts(4000.0, 3000.0)], won=0.0
    )
    denied = toad_reward.margin(
        [_counts(3000.0, 3000.0), _counts(3000.0, 2000.0)], won=0.0
    )
    # 0.001 * 1000 / 500, whichever farm the coin moved on or off.
    assert float(earned.sum()) == pytest.approx(0.002)
    assert float(denied.sum()) == pytest.approx(0.002)


def test_the_absolute_term_survives_a_dead_heat() -> None:
    """At margin zero the reward must still distinguish farming from idling.

    THE FIXED POINT THIS EXISTS TO ESCAPE: two copies of a policy that banks
    nothing bankrupt each other identically, so a purely differential reward is
    zero on every turn and no gradient separates any action from any other.
    Measured over 30 iterations of mirror self-play: best episode 4 coins of a
    possible 200,000.

    Both seats here move coin for coin -- a perfect dead heat, margin zero
    throughout -- so the margin term contributes nothing and only the farm that
    built capital scores at all.
    """
    idle = toad_reward.margin(
        [_counts(3000.0, 3000.0), _counts(23000.0, 23000.0)], won=0.0
    )
    productive = toad_reward.margin(
        [_counts(3000.0, 3000.0), _counts(23000.0, 23000.0, capital=6)], won=0.0
    )
    assert float(idle.sum()) == pytest.approx(0.0)
    # Six animals at 0.05, over 500. The banks are identical in both episodes,
    # which is exactly the point: the floor is not made of coins.
    assert float(productive.sum()) == pytest.approx(0.0006)
    assert float(productive.sum()) > float(idle.sum())


def test_the_terminal_result_rides_on_the_last_turn_alone() -> None:
    """A win is worth 0.02 and a loss -0.02, added to the last turn only."""
    series = [_counts(3000.0, 3000.0), _counts(3000.0, 3000.0), _counts(3000.0, 3000.0)]
    won = toad_reward.margin(series, won=1.0)
    lost = toad_reward.margin(series, won=-1.0)
    drawn = toad_reward.margin(series, won=0.0)
    # 10.0 / 500.
    assert float(won[-1]) == pytest.approx(0.02)
    assert float(lost[-1]) == pytest.approx(-0.02)
    assert float(won[0]) == pytest.approx(0.0)
    assert float(drawn.sum()) == pytest.approx(0.0)


def test_a_near_parity_win_is_dominated_by_winning_not_by_the_margin() -> None:
    """At the corpus's median margin, the discrete win outweighs its size.

    The paired corpus comparison puts the median winning margin at 2,561 coins.
    That reads 2,561 * 0.001 / 500 = 0.0051 against the terminal result's 0.02,
    so what the reward is mostly paying for at the ladder's actual operating
    point is *winning*, which is what the objective says.
    """
    series = [_counts(3000.0, 3000.0), _counts(3000.0, 439.0)]
    rewards = toad_reward.margin(series, won=1.0)
    margin_part = 0.001 * 2561.0 / 500.0
    assert margin_part == pytest.approx(0.005122)
    assert float(rewards.sum()) == pytest.approx(margin_part + 0.02)
    assert 0.02 > margin_part


def test_holding_produce_to_the_horizon_scores_below_selling_it() -> None:
    """No potential survives into the stopping state. The Grzes condition.

    This project shipped the un-zeroed version once: the shaped reward carried
    the value of held produce into turn 719, and at gamma 0.999 holding from
    turn 619 returned 0.905 of it, so in a market both farms were pushing below
    base, REFUSING TO SELL WAS SHAPED-OPTIMAL. "Shed fills, bank flat" was built
    into the reward.

    The guard is behavioural rather than algebraic on purpose: two episodes
    identical except that one converted its shed into coins and the other did
    not. A reward denominated only in banked coins cannot rank them the wrong
    way round, and this is the test that says so.
    """
    opening = _counts(3000.0, 3000.0, fuel=0)
    # Harvested 40 units and still holding them at the horizon.
    held = _counts(3000.0, 3000.0, fuel=40)
    # Sold the same 40 units for 2,400 coins, shed empty at the horizon.
    sold = _counts(5400.0, 3000.0, fuel=0)

    hoarding = float(toad_reward.margin([opening, held], won=0.0).sum())
    selling = float(toad_reward.margin([opening, sold], won=1.0).sum())

    assert hoarding == pytest.approx(0.0)
    assert selling > hoarding
    # And the shed is worth exactly nothing: the fuel delta of +40 changes the
    # reward not at all, which is what "no state is paid for" means.
    assert hoarding == float(
        toad_reward.margin([opening, _counts(3000.0, 3000.0, fuel=0)], won=0.0).sum()
    )


@pytest.mark.parametrize("animal", ["GOOSE", "COW", "SHEEP"])
def test_buying_an_animal_never_pays_for_itself(animal: str) -> None:
    """THE STRUCTURAL GUARANTEE THE CAPITAL WEIGHT RESTS ON.

    An un-priced count is the shape that gave this project three "shaped up,
    bank down" failures. This one is safe only because the margin term charges
    the animal's full purchase price, which must exceed what the capital term
    pays. Checked against the engine's own price table, per animal, so that a
    rules change or a weight change breaks it rather than quietly re-opening
    the pump.

    This is also what discharges the Grzes condition for the capital term: if
    acquiring capital never pays on its own, there is no turn late enough to
    buy on, and the state carried into turn 719 cannot have been bought for the
    reward.
    """
    cost = ANIMALS[animal]["cost"]
    before = _counts(3000.0, 3000.0, capital=0)
    after = _counts(3000.0 - cost, 3000.0, capital=1)
    total = float(toad_reward.margin([before, after], won=0.0).sum())

    assert total == pytest.approx((0.05 - 0.001 * cost) / 500.0)
    assert total < 0.0
    # And with room to spare: the charge is at least six times the credit, so
    # the weight would have to rise above 0.30 -- the cheapest animal's charge --
    # before a spree paid for itself. GOOSE is the binding case at exactly 6x.
    assert 0.001 * cost / 0.05 == pytest.approx(cost / 50.0)
    assert cost / 50.0 >= 6.0


def test_a_buying_spree_costs_what_it_cost() -> None:
    """Ten animals bought outright is a loss, not a harvest of reward."""
    before = _counts(8000.0, 3000.0, capital=0, fuel=0)
    after = _counts(3000.0, 3000.0, capital=10, fuel=60)
    rewards = toad_reward.margin([before, after], won=0.0)
    # -5,000 of margin at 0.001 against ten animals at 0.05, over 500.
    assert float(rewards.sum()) == pytest.approx((-5.0 + 0.5) / 500.0)
    assert float(rewards.sum()) < 0.0


def test_the_reward_stays_inside_the_value_head_bound() -> None:
    """A blowout episode must not peg the bounded critic to its rail.

    The value head is confined to [-1, +1], which is faithful to Toad once
    BaselineLayer expands the spec by MAX_DAYS. The reference banks ~161,730
    against our ~3, so the worst realistic episode return is a margin of about
    -160,000 -- and it has to land well inside the bound, or every value target
    saturates and the critic learns nothing.
    """
    thrashed = toad_reward.margin(
        [_counts(3000.0, 3000.0), _counts(3.0, 161730.0)], won=-1.0
    )
    total = float(thrashed.sum())
    assert total == pytest.approx((0.001 * -161727.0) / 500.0 - 0.02)
    assert -0.4 < total < 0.0


def test_a_series_shorter_than_two_states_raises() -> None:
    """One state is no decision, and a silent empty tensor would be worse."""
    with pytest.raises(ValueError, match="one state per turn"):
        toad_reward.margin([_counts(3000.0, 3000.0)], won=0.0)


def test_counts_reads_the_opponents_bank_from_either_seat() -> None:
    """`opponent` must be the other farm's money, seat by seat.

    Reading our own bank into both fields would make every margin identically
    zero and the arm would train on the absolute term alone while every metric
    stayed finite -- which is precisely the class of silent failure that has
    cost this project two runs.
    """
    observation = {
        "farms": [
            {"money": 4000.0, "tiles": [["FIELD"]], "hands": []},
            {"money": 7000.0, "tiles": [["FIELD"]], "hands": []},
        ],
        "private": {"shed": {}},
        "town": {"unlocked_shops": []},
    }
    assert toad_reward.counts(observation, 0).money == 4000.0
    assert toad_reward.counts(observation, 0).opponent == 7000.0
    assert toad_reward.counts(observation, 1).money == 7000.0
    assert toad_reward.counts(observation, 1).opponent == 4000.0


def test_an_empty_structure_is_not_capital() -> None:
    """Building is free, so a structure count would be free reward.

    ``BUILD_COOP`` and ``BUILD_PASTURE`` take no coins in this engine -- they
    need only an empty tile. A capital term that counted structures would
    therefore pay roughly a hundred times 0.05 a season for nothing, which after
    the normaliser is 0.01 against an objective worth about 0.025 at the
    competitive point. That is not a small leak; it is most of the reward.

    So the count is of animals. An empty COOP and an empty PASTURE must both
    score zero, and the assertion is written against the engine's own spelling
    of a built-but-unoccupied tile.
    """
    observation = {
        "farms": [
            {
                "money": 1000.0,
                "hands": [],
                "tiles": [
                    [{"kind": "COOP"}, {"kind": "PASTURE"}],
                    [{"kind": "PASTURE"}, {"kind": "COOP"}],
                ],
            },
            {"money": 1000.0, "hands": [], "tiles": [[None]]},
        ],
        "private": {"shed": {}},
        "town": {"unlocked_shops": []},
    }
    assert toad_reward.counts(observation, 0).capital == 0


def test_capital_counts_animals_wherever_they_stand() -> None:
    """A bought animal counts in the shed and still counts once placed.

    The engine replaces ``{"kind": "PASTURE"}`` wholesale with a dict carrying
    an ``"animal"`` key when one is placed, so the occupied and empty states are
    disjoint and an animal is counted exactly once however it is held.
    """
    observation = {
        "farms": [
            {
                "money": 1000.0,
                "hands": [],
                "tiles": [
                    [{"kind": "PASTURE", "animal": "COW", "yield_units": 0}, None],
                    [{"kind": "COOP", "animal": "GOOSE", "yield_units": 2}, "LOCKED"],
                ],
            },
            {"money": 1000.0, "hands": [], "tiles": [[None]]},
        ],
        # Two more still boxed up, plus stock that is not capital.
        "private": {"shed": {"SHEEP": 2, "WHEAT": 9}},
        "town": {"unlocked_shops": []},
    }
    reading = toad_reward.counts(observation, 0)
    assert reading.capital == 4
    assert reading.fuel == 9


def test_placing_an_animal_is_capital_neutral() -> None:
    """Moving an animal out of the shed onto its structure must not score.

    Acquisition pays once. If the shed count and the placed count did not hand
    the point cleanly between them, a farm could pick an animal up and put it
    down repeatedly for reward -- and if they overlapped, it would be paid twice
    for owning one bird.
    """
    boxed = {
        "farms": [
            {"money": 1000.0, "hands": [], "tiles": [[{"kind": "COOP"}]]},
            {"money": 1000.0, "hands": [], "tiles": [[None]]},
        ],
        "private": {"shed": {"GOOSE": 1}},
        "town": {"unlocked_shops": []},
    }
    placed = {
        **boxed,
        "farms": [
            {
                "money": 1000.0,
                "hands": [],
                "tiles": [[{"kind": "COOP", "animal": "GOOSE", "yield_units": 0}]],
            },
            {"money": 1000.0, "hands": [], "tiles": [[None]]},
        ],
        "private": {"shed": {"GOOSE": 0}},
    }
    before = toad_reward.counts(boxed, 0)
    after = toad_reward.counts(placed, 0)

    assert before.capital == 1
    assert after.capital == 1
    assert float(toad_reward.margin([before, after], won=0.0).sum()) == pytest.approx(
        0.0
    )


def test_the_two_seats_of_a_mirror_are_not_exact_negations() -> None:
    """The symmetry that cancels the advantage must be broken, and by how much.

    With one policy in both seats a purely differential reward gives
    R_1 = -R_0 identically, the advantage vanishes, and there is no gradient.
    The absolute term is what breaks it. The sum of the two seats' rewards is
    the tell: zero means the reward is still zero-sum.
    """
    ours = [3000.0, 6000.0, 11000.0]
    theirs = [3000.0, 4000.0, 9000.0]
    mine = [0, 2, 3]
    yours = [0, 1, 4]
    seat_zero = toad_reward.margin(
        [_counts(o, t, capital=c) for o, t, c in zip(ours, theirs, mine, strict=True)],
        won=1.0,
    )
    seat_one = toad_reward.margin(
        [_counts(t, o, capital=c) for o, t, c in zip(ours, theirs, yours, strict=True)],
        won=-1.0,
    )
    combined = float(seat_zero.sum()) + float(seat_one.sum())
    # The margin terms and the terminal ranks cancel exactly; what is left is
    # the capital term on both farms: 0.05 * (3 + 4) / 500. The tolerance is
    # float32's, because that is the dtype the learner reads.
    assert combined == pytest.approx(0.0007, rel=1e-5)
    assert combined != 0.0


def test_the_margin_reward_ignores_every_non_monetary_count() -> None:
    """City, units, research, fuel and capital must not reach this reward.

    Toad's components are still recorded on every trajectory, and a margin arm
    that quietly summed them too would be the opponent-mix arm again with a new
    name. Moving city, units, research and fuel at once must change nothing.
    `capital` is excluded here because it is the one that IS read -- see the
    dead-heat test.
    """
    before = _counts(3000.0, 3000.0)
    flat = _counts(4000.0, 3000.0)
    busy = _counts(4000.0, 3000.0, city=99, unit=8, research=5, fuel=77)
    assert float(toad_reward.margin([before, busy], won=0.0).sum()) == pytest.approx(
        float(toad_reward.margin([before, flat], won=0.0).sum())
    )


def test_rewards_are_float32_and_one_per_decision() -> None:
    """Shape and dtype, because a length off by one shifts every attribution."""
    series = [_counts(3000.0 + 100.0 * i, 3000.0) for i in range(9)]
    rewards = toad_reward.margin(series, won=0.0)
    assert rewards.shape == (8,)
    assert rewards.dtype == torch.float32
