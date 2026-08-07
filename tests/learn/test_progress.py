"""Tests for the progress potential, driven through the real engine.

Nothing here is synthetic and nothing here mocks a tile. The whole claim of
``learn/progress.py`` is that its numbers are a reading of the engine's own
state transitions rather than of somebody's intuition about farming, and the
only way to hold it to that is to play the chain -- buy a seed, plant it, water
it across five days, harvest it, drop it, sell it -- against
``kaggle_environments`` and watch the potential at every link. A handwritten
tile dict would pass whatever the engine actually does.

The chain is played at the farmer's spawn, ``(4, 4)``, which is both an unlocked
NW tile and one of the four shed-access tiles. That is not a trick: it lets the
same square be planted, harvested and dropped from, so the walk between them
does not have to be scripted, and ``_end_of_day`` returns the farmer to exactly
that square every night so the multi-day part of the chain needs no movement
either.

No policy is loaded and no network runs, so this file costs about a second.
"""

import pytest
from kaggle_environments import make
from kaggle_environments.core import Environment

from kaggriculture.constants import ANIMALS, ENVIRONMENT, EPISODE_STEPS, MARKET_PARAMS
from kaggriculture.learn.progress import (
    GROWING,
    POTENTIAL_COMPONENTS,
    VALUE,
    potential,
)

PASS = {"farmer": ["PASS"], "hands": [], "market": []}

# What the opponent does for the whole of every episode here. Seat 1 is not
# under test and an idle seat keeps the market moving only by the town's own
# consumption schedule, which is the point of the last assertion below.
IDLE = PASS

# Wheat: seed 10, first yield on day 2, bonus watering window days 2 to 4,
# base price 25. Read from the tables rather than typed, so a rules change
# upstream fails this file loudly instead of quietly asserting old arithmetic.
SEED = 10.0
PRICE = float(MARKET_PARAMS["WHEAT"]["base"])


def _step(environment: Environment, action: dict) -> float:
    """Apply one seat-0 action against an idle seat 1 and return the new potential."""
    environment.step([action, IDLE])
    return sum(potential(environment.state[0].observation))


def _advance(environment: Environment, day: int) -> None:
    """Pass turns until the season reaches ``day``."""
    while environment.state[0].observation["day"] < day:
        _step(environment, PASS)


# How long to sit on a full shed waiting for the town to move the wheat quote.
# It is slow: `_town_consume` takes one unit per product per twelve turns plus
# whatever the unlocked shops eat, against a market inventory of 10,000, and
# `market_price` rounds to whole coins -- so a handful of turns leaves the quote
# where it was and a test written against a handful passes under live prices.
# Bounded rather than fixed, and the fixture records whether the bound was
# reached, so a rules change that freezes the market fails the test loudly
# instead of making it vacuous.
IDLE_LIMIT = 400


@pytest.fixture(scope="module")
def chain() -> dict[str, float]:
    """Return the potential at every link of one complete wheat season.

    One farm, one tile, one crop, played to a sale. The keys are the links; the
    values are the total potential immediately after each one.
    """
    environment = make(
        ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS, "seed": 3}
    )
    environment.reset(2)
    reached = {"opening": sum(potential(environment.state[0].observation))}
    reached["bought"] = _step(
        environment, {**PASS, "market": [["BUY_SEED", "WHEAT", 1]]}
    )
    reached["planted"] = _step(environment, {**PASS, "farmer": ["PLANT", "WHEAT"]})
    reached["passed"] = _step(environment, PASS)
    reached["watered/0"] = _step(environment, {**PASS, "farmer": ["WATER"]})
    for day in range(1, 5):
        _advance(environment, day)
        reached[f"watered/{day}"] = _step(environment, {**PASS, "farmer": ["WATER"]})
    reached["harvested"] = _step(environment, {**PASS, "farmer": ["HARVEST"]})
    reached["dropped"] = _step(environment, {**PASS, "farmer": ["DROP"]})
    reached["quoted"] = _quote(environment)
    for _turn in range(IDLE_LIMIT):
        reached["idled"] = _step(environment, PASS)
        if _quote(environment) != reached["quoted"]:
            break
    reached["requoted"] = _quote(environment)
    reached["sold"] = _step(environment, {**PASS, "market": [["SELL", "WHEAT", 4]]})
    return reached


def _quote(environment: Environment) -> float:
    """Return what the market is currently paying for a unit of wheat."""
    return float(environment.state[0].observation["market"]["prices"]["WHEAT"])


def test_an_empty_farm_holds_nothing(chain: dict[str, float]) -> None:
    """Both farms open identical and empty, so the potential opens at exactly zero.

    Worth an assertion of its own because a potential with a constant offset
    would satisfy every difference below -- the shaped reward only ever sees
    differences -- while making ``potential/total`` unreadable as "coins in the
    pipeline", which is the number a run is charted on.
    """
    assert chain["opening"] == 0.0


def test_every_link_of_the_chain_pays(chain: dict[str, float]) -> None:
    """The whole point: no step of the seven is invisible to the gradient.

    The plateau this replaces paid for banked coins alone, so a policy that
    planted, watered and harvested correctly and then failed to sell received
    exactly what a policy that passed for 719 turns received. Each assertion
    below is one link that used to be worth nothing.

    ``BUY_SEED`` and ``DROP`` are asserted *neutral* rather than positive, and
    deliberately. A seed is bought with money, so paying for the purchase as
    well would be paying twice for one conversion; and the engine's
    ``_end_of_day`` empties every farmer's arms into the shed for free, so
    carrying produce there is not a link the agent has to be taught.

    Every earning link is asserted twice: once against the arithmetic the module
    intends, and once as a bare ``> 0``. The second is not redundant. The first
    is written in terms of ``GROWING``, so it goes vacuously true at
    ``GROWING == 0`` -- both sides become zero, the chain pays nothing, and a
    test that only ever compared the two would report the whole change working
    while the reward was flat again.
    """
    assert chain["bought"] - chain["opening"] == pytest.approx(SEED)
    assert chain["planted"] - chain["bought"] == pytest.approx(GROWING * PRICE)
    assert chain["watered/2"] - chain["watered/1"] == pytest.approx(GROWING * PRICE)
    assert chain["watered/3"] - chain["watered/2"] == pytest.approx(GROWING * PRICE)
    assert chain["watered/4"] - chain["watered/3"] == pytest.approx(GROWING * PRICE)
    assert chain["harvested"] - chain["watered/4"] == pytest.approx(
        (1.0 - GROWING) * 4 * PRICE - SEED
    )
    assert chain["dropped"] == chain["harvested"]

    assert chain["planted"] > chain["bought"]
    assert chain["watered/2"] > chain["watered/1"]
    assert chain["watered/3"] > chain["watered/2"]
    assert chain["watered/4"] > chain["watered/3"]
    assert chain["harvested"] > chain["watered/4"]


def test_the_harvest_is_the_largest_single_payment(chain: dict[str, float]) -> None:
    """The link random exploration is least likely to find is the best paid.

    ``GROWING`` is the only free parameter in the module and this is what it was
    chosen for. Below 0.5 the majority of a crop's value lands on HARVEST, which
    needs the agent to walk back to a tile it planted five days earlier; above
    it, the same crop is worth more standing in the field than in a farmer's
    arms, and there is a version of this reward that pays best for planting and
    walking away.
    """
    links = [chain["planted"] - chain["bought"]] + [
        chain[f"watered/{day}"] - chain[f"watered/{day - 1}"] for day in (2, 3, 4)
    ]

    assert chain["harvested"] - chain["watered/4"] > max(links)


def test_watering_outside_the_bonus_window_pays_nothing(
    chain: dict[str, float],
) -> None:
    """The potential reads the engine, not a belief that watering is good.

    Wheat's window is days 2 to 4 (``water_bonus_window``), and a watering on
    day 0 or 1 adds no yield -- ``_apply_unit_action`` checks the window before
    touching ``yield_units``. It is still necessary, because two consecutive
    unwatered days turn the tile into a WEED, but the necessity shows up as the
    loss that is avoided rather than as a payment, which is the difference
    between shaping the state and paying for verbs.
    """
    assert chain["watered/0"] == chain["planted"]
    assert chain["watered/1"] == chain["watered/0"]


def test_selling_hands_the_potential_back_and_takes_money_instead(
    chain: dict[str, float],
) -> None:
    """A season that sells everything ends where it started, and that is correct.

    The shaped return over a whole chain is therefore zero and the profit is
    entirely in the bank -- which is the property that makes this a curriculum
    rather than a second objective. A reward whose shaped total grew with every
    completed chain would eventually be larger than the money it was meant to be
    teaching the agent to earn.
    """
    assert chain["sold"] == 0.0
    assert chain["dropped"] == pytest.approx(4 * PRICE)


def test_an_idle_turn_moves_nothing_even_though_the_market_did(
    chain: dict[str, float],
) -> None:
    """The potential is a function of *our* state, and nothing else.

    The town consumes on a fixed schedule every four and twelve turns and moves
    every quoted price with it, so a potential valued at ``market["prices"]``
    would pay and charge us on turns where we did nothing at all -- rewarding
    the agent for the town's appetite and, in a self-play batch, for whatever
    the opponent chose to dump. The engine constants in ``VALUE`` are what stop
    that, and this is where a switch to live quotes fails.

    Checked while the shed is *full*, which is the only state where the two
    implementations differ: an empty farm is worth zero at any price, so the
    same assertion taken on an idle turn before the harvest passes under live
    quotes as readily as under base prices. The quote is asserted to have
    actually moved across those turns, so the test cannot pass because nothing
    happened.
    """
    assert chain["passed"] == chain["planted"]
    assert chain["requoted"] != chain["quoted"]
    assert chain["idled"] == chain["dropped"]


def test_a_plant_dug_up_gives_back_only_what_a_plant_is_worth() -> None:
    """Plant, dig, replant is a treadmill and it must not be a source of reward.

    The specific way a shaped reward gets farmed: find the cheapest transition
    that pays and repeat it. Planting pays, ``DIG`` costs one turn, and wheat
    seed is ten coins against a starting bank of three thousand, so an agent
    with a per-plant bonus could plant three hundred times and never farm
    anything.

    Under a potential it cannot: the cycle returns the tile to bare soil, so the
    potential returns to what it was before the seed was bought and the whole
    round trip pays exactly zero -- while the money is gone. Asserted against
    the *opening* potential rather than against zero, so a potential that
    happened to be zero everywhere would not pass.
    """
    environment = make(
        ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS, "seed": 5}
    )
    environment.reset(2)
    opening = sum(potential(environment.state[0].observation))
    opening_money = float(environment.state[0].observation["farms"][0]["money"])

    for _cycle in range(5):
        _step(environment, {**PASS, "market": [["BUY_SEED", "WHEAT", 1]]})
        planted = _step(environment, {**PASS, "farmer": ["PLANT", "WHEAT"]})
        dug = _step(environment, {**PASS, "farmer": ["DIG"]})

    assert planted > opening
    assert dug == opening
    assert float(
        environment.state[0].observation["farms"][0]["money"]
    ) == pytest.approx(opening_money - 5 * SEED)


def test_a_plant_left_unwatered_loses_everything_it_was_holding() -> None:
    """Two dry days is a WEED, and the potential has to say so on the night it happens.

    ``_daily_refresh_plants`` replaces the tile outright once
    ``consecutive_unwatered`` reaches two, and ``_new_plant`` starts that
    counter at one -- so a crop planted and ignored is gone by the end of its
    own first day. That loss is the entire signal for watering, since watering
    outside the bonus window pays nothing, and a potential that valued a WEED
    the way it values a PLANT would leave the agent no reason ever to carry a
    watering can.
    """
    environment = make(
        ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS, "seed": 7}
    )
    environment.reset(2)
    _step(environment, {**PASS, "market": [["BUY_SEED", "WHEAT", 1]]})
    planted = _step(environment, {**PASS, "farmer": ["PLANT", "WHEAT"]})
    _advance(environment, 1)
    withered = sum(potential(environment.state[0].observation))

    assert planted == pytest.approx(SEED + GROWING * PRICE)
    assert withered == 0.0


def test_a_bought_animal_is_worth_what_it_cost_wherever_it_is_standing() -> None:
    """Buying and placing livestock must not read as an 800-coin mistake.

    ``BUY_ANIMAL`` takes 400 coins for a cow and puts it in the shed;
    ``PLACE`` moves it out of the shed onto a pasture. Neither is production and
    neither should pay, but both are steps a producing farm has to take, and a
    potential that valued only *harvested* produce would charge the agent 400 to
    buy the cow and another 400 to put it out to grass. Carrying capital at cost
    is what makes every irreversible purchase in this game neutral rather than
    punished.

    The pasture is built at the spawn square, which is bare NW soil.
    """
    environment = make(
        ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS, "seed": 11}
    )
    environment.reset(2)
    opening = sum(potential(environment.state[0].observation))
    built = _step(environment, {**PASS, "farmer": ["BUILD_PASTURE"]})
    bought = _step(environment, {**PASS, "market": [["BUY_ANIMAL", "COW", 1]]})
    picked = _step(environment, {**PASS, "farmer": ["PICKUP", "COW", 1]})
    placed = _step(environment, {**PASS, "farmer": ["PLACE", "COW"]})

    assert built == opening
    assert bought - opening == pytest.approx(ANIMALS["COW"]["cost"])
    assert picked == bought
    assert placed == bought


def test_every_component_is_reported_and_they_sum_to_the_whole(
    chain: dict[str, float],
) -> None:
    """Six numbers, because the total hides which failure a run is having.

    ``growing`` alone is a farm that plants and never harvests; ``stored`` alone
    is one that harvests and never sells. Both are a rising total with a flat
    bank, and the training loop charts them individually for exactly that
    reason.
    """
    environment = make(
        ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS, "seed": 3}
    )
    environment.reset(2)
    values = potential(environment.state[0].observation)

    assert len(values) == len(POTENTIAL_COMPONENTS)
    assert sum(values) == pytest.approx(chain["opening"])
    assert set(VALUE) >= set(ANIMALS)
