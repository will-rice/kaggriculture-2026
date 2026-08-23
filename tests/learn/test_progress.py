"""Tests for the progress potential, driven through the real engine.

Nothing here is synthetic and nothing here mocks a tile. The whole claim of
``learn/progress.py`` is that its numbers are a reading of the engine's own
state transitions rather than of somebody's intuition about farming, and the
only way to hold it to that is to play the chain -- buy a seed, plant it, water
it across five days, harvest it, drop it, sell it -- against
``kaggle_environments`` and watch the potential at every link. A handwritten
tile dict would pass whatever the engine actually does.

The potential is the seat's **net worth**, bank included, so every link is read
here as a change in net worth: a purchase at the quoted price moves value
between two components and changes nothing, a watering inside the bonus window
creates value, and a sale realises it. Each link therefore records the whole
component vector rather than its sum, because two of the claims -- that a fair
trade is neutral, and that a growing crop gains without any money moving -- are
claims about *which* components moved and are invisible in the total.

The chain is played at the farmer's spawn, ``(4, 4)``, which is both an unlocked
NW tile and one of the four shed-access tiles. That is not a trick: it lets the
same square be planted, harvested and dropped from, so the walk between them
does not have to be scripted, and ``_end_of_day`` returns the farmer to exactly
that square every night so the multi-day part of the chain needs no movement
either.

No policy is loaded and no network runs, so this file costs about a second.
"""

from dataclasses import dataclass

import pytest
from kaggle_environments import make
from kaggle_environments.core import Environment

from kaggriculture.constants import (
    ANIMALS,
    ENVIRONMENT,
    EPISODE_STEPS,
    MARKET_PARAMS,
    STARTING_MONEY,
    TURNS_PER_DAY,
)
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

# Where the bank sits in a potential vector. Everything before it is the
# pipeline -- what the farm is holding on its way to the bank, which is what
# this potential was on its own before the bank joined it.
MONEY = POTENTIAL_COMPONENTS.index("money")


def _pipeline(components: list[float]) -> float:
    """Return one potential vector's coins, with the bank taken back out."""
    return sum(components) - components[MONEY]


def _step(environment: Environment, action: dict) -> list[float]:
    """Apply one seat-0 action against an idle seat 1 and return the new potential."""
    environment.step([action, IDLE])
    return potential(environment.state[0].observation)


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

# How much of a traded position the market may take in fees and impact before
# this file calls it something other than a fair price. `BUY_PRODUCT` is quoted
# at `market_price(inventory - 1)` and `SELL` at `market_price(inventory)`, both
# against an inventory of 10,000 that a handful of units barely moves, so the
# realised premium over the `base` mark is single-digit coins on hundreds
# traded. 20% is loose enough that a rules change to the price curve does not
# fail this file spuriously and tight enough that a potential which paid out the
# *whole* traded value on one leg -- which is what a pipeline-only potential
# does -- cannot pass.
SPREAD_LIMIT = 0.2


@dataclass(frozen=True)
class Chain:
    """One complete wheat season, read at every link.

    Attributes:
        links: The whole component vector immediately after each link, keyed by
            the link. Vectors rather than sums because several of the claims
            here are about *which* component moved, and a sum cannot say.
        quoted: What the market paid for wheat once the shed was full.
        requoted: What it paid after the idle turns the fixture then takes,
            which it keeps taking until the two differ.
    """

    links: dict[str, list[float]]
    quoted: float
    requoted: float


@pytest.fixture(scope="module")
def chain() -> Chain:
    """Return the potential at every link of one complete wheat season.

    One farm, one tile, one crop, played to a sale.
    """
    environment = make(
        ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS, "seed": 3}
    )
    environment.reset(2)
    reached: dict[str, list[float]] = {
        "opening": potential(environment.state[0].observation)
    }
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
    quoted = _quote(environment)
    for _turn in range(IDLE_LIMIT):
        reached["idled"] = _step(environment, PASS)
        if _quote(environment) != quoted:
            break
    requoted = _quote(environment)
    reached["sold"] = _step(environment, {**PASS, "market": [["SELL", "WHEAT", 4]]})
    return Chain(links=reached, quoted=quoted, requoted=requoted)


def _quote(environment: Environment) -> float:
    """Return what the market is currently paying for a unit of wheat."""
    return float(environment.state[0].observation["market"]["prices"]["WHEAT"])


def test_an_empty_farm_is_worth_the_bank_it_opens_with(
    chain: Chain,
) -> None:
    """Nothing in the fields, nothing in the shed, three thousand coins.

    Worth an assertion of its own because a potential with a constant offset
    would satisfy every difference below -- the shaped reward only ever sees
    differences -- while making ``potential/total`` unreadable as "what this
    seat is worth", which is the number a run is charted on. The two halves are
    asserted separately: the pipeline opens at exactly zero, and the whole
    opens at exactly the bank both seats are given.
    """
    assert _pipeline(chain.links["opening"]) == 0.0
    assert sum(chain.links["opening"]) == float(STARTING_MONEY)


def test_every_link_of_the_chain_pays(chain: Chain) -> None:
    """The whole point: no step of the seven is invisible to the gradient.

    The plateau this replaces paid for banked coins alone, so a policy that
    planted, watered and harvested correctly and then failed to sell received
    exactly what a policy that passed for 719 turns received. Each assertion
    below is one link that used to be worth nothing.

    ``BUY_SEED`` and ``DROP`` are asserted *neutral* rather than positive, and
    deliberately. The seed is quoted at ``CROPS["WHEAT"]["seed"]``, which is the
    literal constant the potential prices a stored seed at, so ten coins leave
    the bank and ten coins of seed arrive and the seat is worth exactly what it
    was; and the engine's ``_end_of_day`` empties every farmer's arms into the
    shed for free, so carrying produce there is not a link the agent has to be
    taught. The purchase is checked to have actually happened -- the pipeline
    half moves by the whole seed -- so a neutral total cannot mean a rejected
    order.

    Every earning link is asserted twice: once against the arithmetic the module
    intends, and once as a bare ``> 0``. The second is not redundant. The first
    is written in terms of ``GROWING``, so it goes vacuously true at
    ``GROWING == 0`` -- both sides become zero, the chain pays nothing, and a
    test that only ever compared the two would report the whole change working
    while the reward was flat again.
    """
    assert sum(chain.links["bought"]) == pytest.approx(sum(chain.links["opening"]))
    assert _pipeline(chain.links["bought"]) - _pipeline(
        chain.links["opening"]
    ) == pytest.approx(SEED)
    assert sum(chain.links["planted"]) - sum(chain.links["bought"]) == pytest.approx(
        GROWING * PRICE
    )
    assert sum(chain.links["watered/2"]) - sum(
        chain.links["watered/1"]
    ) == pytest.approx(GROWING * PRICE)
    assert sum(chain.links["watered/3"]) - sum(
        chain.links["watered/2"]
    ) == pytest.approx(GROWING * PRICE)
    assert sum(chain.links["watered/4"]) - sum(
        chain.links["watered/3"]
    ) == pytest.approx(GROWING * PRICE)
    assert sum(chain.links["harvested"]) - sum(
        chain.links["watered/4"]
    ) == pytest.approx((1.0 - GROWING) * 4 * PRICE - SEED)
    assert sum(chain.links["dropped"]) == sum(chain.links["harvested"])

    assert sum(chain.links["planted"]) > sum(chain.links["bought"])
    assert sum(chain.links["watered/2"]) > sum(chain.links["watered/1"])
    assert sum(chain.links["watered/3"]) > sum(chain.links["watered/2"])
    assert sum(chain.links["watered/4"]) > sum(chain.links["watered/3"])
    assert sum(chain.links["harvested"]) > sum(chain.links["watered/4"])


def test_the_harvest_is_the_largest_single_payment(chain: Chain) -> None:
    """The link random exploration is least likely to find is the best paid.

    ``GROWING`` is the only free parameter in the module and this is what it was
    chosen for. Below 0.5 the majority of a crop's value lands on HARVEST, which
    needs the agent to walk back to a tile it planted five days earlier; above
    it, the same crop is worth more standing in the field than in a farmer's
    arms, and there is a version of this reward that pays best for planting and
    walking away.
    """
    links = [sum(chain.links["planted"]) - sum(chain.links["bought"])] + [
        sum(chain.links[f"watered/{day}"]) - sum(chain.links[f"watered/{day - 1}"])
        for day in (2, 3, 4)
    ]

    assert sum(chain.links["harvested"]) - sum(chain.links["watered/4"]) > max(links)


def test_watering_outside_the_bonus_window_pays_nothing(
    chain: Chain,
) -> None:
    """The potential reads the engine, not a belief that watering is good.

    Wheat's window is days 2 to 4 (``water_bonus_window``), and a watering on
    day 0 or 1 adds no yield -- ``_apply_unit_action`` checks the window before
    touching ``yield_units``. It is still necessary, because two consecutive
    unwatered days turn the tile into a WEED, but the necessity shows up as the
    loss that is avoided rather than as a payment, which is the difference
    between shaping the state and paying for verbs.
    """
    assert chain.links["watered/0"] == chain.links["planted"]
    assert chain.links["watered/1"] == chain.links["watered/0"]


def test_a_growing_crop_gains_value_while_the_bank_does_not_move(
    chain: Chain,
) -> None:
    """A crop is worth more on every turn the engine puts yield on it.

    This is the transition the whole shaped term exists for and the one a
    money-only reward cannot see: across the three days of wheat's bonus window
    the seat is worth ``3 * GROWING * PRICE`` more than it was, and not one coin
    changed hands to do it. The bank is asserted flat across the same span, so
    the gain cannot be some purchase being credited twice.
    """
    assert chain.links["watered/4"][MONEY] == chain.links["watered/1"][MONEY]
    assert sum(chain.links["watered/4"]) - sum(
        chain.links["watered/1"]
    ) == pytest.approx(3 * GROWING * PRICE)
    assert sum(chain.links["watered/4"]) > sum(chain.links["watered/1"])


def test_selling_realises_the_mark_instead_of_forfeiting_it(
    chain: Chain,
) -> None:
    """A sale moves value between components; it does not leave the potential.

    This is what putting the bank in the potential is for. Under a pipeline-only
    potential this sale handed back every coin of the produce it converted --
    the largest single loss in the season, applied to the one action the season
    is played for -- and the agent's shaped gradient pointed away from selling.
    Here the shed empties and the bank fills, and the whole potential moves only
    by what the market paid over the ``base`` mark: strictly less than
    ``SPREAD_LIMIT`` of what was traded, against 100% of it before.

    The season's produce is still asserted to have been created, so a potential
    that valued nothing at all would not pass.
    """
    assert _pipeline(chain.links["sold"]) == 0.0
    assert _pipeline(chain.links["dropped"]) == pytest.approx(4 * PRICE)
    assert chain.links["sold"][MONEY] > chain.links["dropped"][MONEY]
    assert (
        abs(sum(chain.links["sold"]) - sum(chain.links["dropped"]))
        < SPREAD_LIMIT * 4 * PRICE
    )
    assert sum(chain.links["sold"]) - sum(chain.links["opening"]) == pytest.approx(
        chain.links["sold"][MONEY] - float(STARTING_MONEY)
    )


def test_an_idle_turn_moves_nothing_even_though_the_market_did(
    chain: Chain,
) -> None:
    """The potential is a function of *our* state, and nothing else.

    The town consumes on a fixed schedule every four and twelve turns and moves
    every quoted price with it, so a potential valued at ``market["prices"]``
    would pay and charge us on turns where we did nothing at all -- rewarding
    the agent for the town's appetite and, in a self-play batch, for whatever
    the opponent chose to dump. The engine constants in ``VALUE`` are what stop
    that, and this is where a switch to live quotes fails.

    Checked while the shed is *full*, which is the only state where the two
    implementations differ: an empty farm is worth its bank at any price, so the
    same assertion taken on an idle turn before the harvest passes under live
    quotes as readily as under base prices. The quote is asserted to have
    actually moved across those turns, so the test cannot pass because nothing
    happened.
    """
    assert chain.links["passed"] == chain.links["planted"]
    assert chain.requoted != chain.quoted
    assert chain.links["idled"] == chain.links["dropped"]


def test_doing_nothing_is_flat() -> None:
    """Three days of PASS and the seat is worth exactly what it opened with.

    The behaviour thirteen RL arms converged on, and the one the shaped reward
    has to be neutral about rather than opposed to: if idling drifted, the
    shaping would be paying or charging for the passage of time, which is not a
    decision. Three days rather than three turns because ``_end_of_day`` is
    where a drift would come from -- it pays farm hands, drops every inventory
    into the shed and refreshes every plant and animal -- and a test that never
    crossed a night would not see one.
    """
    environment = make(
        ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS, "seed": 13}
    )
    environment.reset(2)
    opening = potential(environment.state[0].observation)
    for _turn in range(3 * TURNS_PER_DAY):
        idled = _step(environment, PASS)

    assert environment.state[0].observation["day"] == 3
    assert idled == opening
    assert sum(idled) == float(STARTING_MONEY)


def test_buying_at_the_quoted_price_is_neutral_to_the_coin() -> None:
    """The two purchases the engine quotes off a constant cost the seat nothing.

    ``_commit_unit`` charges ``CROPS[item]["seed"]`` for a ``BUY_SEED`` and
    ``ANIMALS[item]["cost"]`` for a ``BUY_ANIMAL``, and those are the same two
    constants ``progress.VALUE`` and ``_standing`` price the results at. So the
    tolerance here is not a judgement about how fair the market is -- the
    numbers are equal by construction and the only slack is float
    representation, which is why this asserts equality rather than
    ``pytest.approx``.

    That is the property the whole change turns on. Every purchase a farm has to
    make is an expense, and a potential that did not carry what the expense
    bought would read each one as a loss the size of its price -- 3,000 coins of
    "mistakes" before a fresh seat has grown anything.
    """
    environment = make(
        ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS, "seed": 17}
    )
    environment.reset(2)
    opening = potential(environment.state[0].observation)
    seeds = _step(environment, {**PASS, "market": [["BUY_SEED", "WHEAT", 10]]})
    cow = _step(environment, {**PASS, "market": [["BUY_ANIMAL", "COW", 1]]})

    assert sum(seeds) == sum(opening)
    assert sum(cow) == sum(opening)
    assert _pipeline(seeds) == pytest.approx(10 * SEED)
    assert _pipeline(cow) == pytest.approx(10 * SEED + float(ANIMALS["COW"]["cost"]))


def test_a_market_round_trip_pays_the_spread_and_pumps_nothing() -> None:
    """Churning produce through the book cannot be a source of reward.

    The specific failure a potential that valued goods but not coins invites:
    ``BUY_PRODUCT`` converts coins the potential cannot see into produce it
    can, so buying wheat reads as free money -- 250 coins of shaped reward for
    ten units -- and selling it back reads as a loss. That is a pump on a loop
    a fresh agent can execute from turn 0, and it is the shape the reward must
    not have.

    Under net worth the buy is *negative*, by exactly the premium the book
    charges: ``_commit_unit`` quotes ``BUY_PRODUCT`` at
    ``market_price(inventory - 1)``, which is strictly above the ``base`` the
    potential marks at, and each further unit is quoted a step deeper. The sale
    back recovers it -- the engine quotes the buy at the post-buy inventory
    precisely so that a round trip against an unchanged market nets zero -- so
    the whole loop is a wash rather than a payout, and what is left of it is the
    town's consumption between the two turns.

    Both legs are asserted, because only the first discriminates: a
    pipeline-only potential passes the round-trip assertion (it opens and closes
    at zero) and fails the buy outright.
    """
    environment = make(
        ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS, "seed": 19}
    )
    environment.reset(2)
    opening = potential(environment.state[0].observation)
    bought = _step(environment, {**PASS, "market": [["BUY_PRODUCT", "WHEAT", 10]]})
    sold = _step(environment, {**PASS, "market": [["SELL", "WHEAT", 10]]})

    assert _pipeline(bought) == pytest.approx(10 * PRICE)
    assert sum(bought) < sum(opening)
    assert sum(opening) - sum(bought) < SPREAD_LIMIT * 10 * PRICE
    assert _pipeline(sold) == 0.0
    assert abs(sum(sold) - sum(opening)) < SPREAD_LIMIT * 10 * PRICE


def test_a_plant_dug_up_gives_back_only_what_a_plant_is_worth() -> None:
    """Plant, dig, replant is a treadmill and it must not be a source of reward.

    The specific way a shaped reward gets farmed: find the cheapest transition
    that pays and repeat it. Planting pays, ``DIG`` costs one turn, and wheat
    seed is ten coins against a starting bank of three thousand, so an agent
    with a per-plant bonus could plant three hundred times and never farm
    anything.

    Under a potential it cannot: the cycle returns the tile to bare soil, so the
    round trip pays back exactly what planting paid -- and the seed it consumed
    is gone from the bank, so five cycles leave the seat worth five seeds less
    than it started. That charge is the whole difference between this and a
    pipeline-only potential, which scored the treadmill at exactly zero and let
    the money leak out where the reward could not see it.
    """
    environment = make(
        ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS, "seed": 5}
    )
    environment.reset(2)
    opening = potential(environment.state[0].observation)

    for cycle in range(5):
        _step(environment, {**PASS, "market": [["BUY_SEED", "WHEAT", 1]]})
        planted = _step(environment, {**PASS, "farmer": ["PLANT", "WHEAT"]})
        dug = _step(environment, {**PASS, "farmer": ["DIG"]})
        assert sum(dug) == pytest.approx(sum(opening) - (cycle + 1) * SEED)

    assert sum(planted) > sum(dug)
    assert _pipeline(dug) == 0.0
    assert dug[MONEY] == pytest.approx(float(STARTING_MONEY) - 5 * SEED)


def test_a_plant_left_unwatered_loses_everything_it_was_holding() -> None:
    """Two dry days is a WEED, and the potential has to say so on the night it happens.

    ``_daily_refresh_plants`` replaces the tile outright once
    ``consecutive_unwatered`` reaches two, and ``_new_plant`` starts that
    counter at one -- so a crop planted and ignored is gone by the end of its
    own first day. That loss is the entire signal for watering, since watering
    outside the bonus window pays nothing, and a potential that valued a WEED
    the way it values a PLANT would leave the agent no reason ever to carry a
    watering can.

    What the loss comes to is the seed, exactly: the standing yield the plant
    was carrying was never bought, and the ten coins that were are gone.
    """
    environment = make(
        ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS, "seed": 7}
    )
    environment.reset(2)
    opening = potential(environment.state[0].observation)
    _step(environment, {**PASS, "market": [["BUY_SEED", "WHEAT", 1]]})
    planted = _step(environment, {**PASS, "farmer": ["PLANT", "WHEAT"]})
    _advance(environment, 1)
    withered = potential(environment.state[0].observation)

    assert sum(planted) == pytest.approx(sum(opening) + GROWING * PRICE)
    assert sum(withered) == pytest.approx(sum(opening) - SEED)
    assert _pipeline(withered) == 0.0


def test_a_bought_animal_is_worth_what_it_cost_wherever_it_is_standing() -> None:
    """Buying and placing livestock must not read as an 800-coin mistake.

    ``BUY_ANIMAL`` takes 400 coins for a cow and puts it in the shed;
    ``PLACE`` moves it out of the shed onto a pasture. Neither is production and
    neither should pay, but both are steps a producing farm has to take, and a
    potential that valued only *harvested* produce would charge the agent 400 to
    buy the cow and another 400 to put it out to grass. Carrying capital at cost
    is what makes every irreversible purchase in this game neutral rather than
    punished -- and carrying the bank is what makes the purchase itself neutral
    rather than a 400-coin payout.

    The pasture is built at the spawn square, which is bare NW soil.
    """
    environment = make(
        ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS, "seed": 11}
    )
    environment.reset(2)
    opening = potential(environment.state[0].observation)
    built = _step(environment, {**PASS, "farmer": ["BUILD_PASTURE"]})
    bought = _step(environment, {**PASS, "market": [["BUY_ANIMAL", "COW", 1]]})
    picked = _step(environment, {**PASS, "farmer": ["PICKUP", "COW", 1]})
    placed = _step(environment, {**PASS, "farmer": ["PLACE", "COW"]})

    assert sum(built) == sum(opening)
    assert sum(bought) == sum(opening)
    assert sum(picked) == sum(bought)
    assert sum(placed) == sum(bought)
    assert _pipeline(placed) == pytest.approx(ANIMALS["COW"]["cost"])


def test_every_component_is_reported_and_they_sum_to_the_whole(
    chain: Chain,
) -> None:
    """Seven numbers, because the total hides which failure a run is having.

    ``growing`` alone is a farm that plants and never harvests; ``stored`` alone
    is one that harvests and never sells. Both are a rising total with a flat
    bank, and the training loop charts them individually for exactly that
    reason -- which it can only do while ``money`` is one of the seven rather
    than the thing the other six are compared against.
    """
    environment = make(
        ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS, "seed": 3}
    )
    environment.reset(2)
    values = potential(environment.state[0].observation)

    assert len(values) == len(POTENTIAL_COMPONENTS)
    assert sum(values) == pytest.approx(sum(chain.links["opening"]))
    assert values[MONEY] == float(STARTING_MONEY)
    assert set(VALUE) >= set(ANIMALS)
