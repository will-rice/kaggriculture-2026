"""Unit tests for the vendored agent's model of the game.

The economic policy carries its own reimplementation of several rules the engine
also implements: the market price curve, the hire cost ladder, the board
geometry. Those reimplementations are where its strength comes from — pricing
from the live curve rather than a base table is the thing our own heuristic
never did — and they are also the most dangerous code in the file, because a
model that disagrees with the engine is confidently wrong rather than obviously
broken. Nothing raises when an agent mis-prices a crop; it just plays badly.

So these tests do not assert the agent's arithmetic against itself. They assert
it against `kaggle_environments`, which is the authority. If one fails, the
agent's picture of the game has drifted from the game.

They also serve the second purpose that the whole-episode characterization test
in `test_economic_policy.py` cannot: that test proves a refactor changed
nothing, but teaches nothing about what any of this means. These document what
the individual pieces are for, which is what makes the file safe to change
rather than merely safe to leave alone.
"""

import pytest

from kaggriculture import economic_policy as ep
from kaggriculture.constants import (
    MARKET_PARAMS,
    PRODUCTS,
    hire_cost,
    market_price,
    quadrant_of,
    shed_access_tiles,
)

BOARD_SIZE = 10
# The market baseline the engine starts every product at.
EQUILIBRIUM = 10000


@pytest.mark.parametrize("item", PRODUCTS)
@pytest.mark.parametrize("offset", [-4000, -500, -50, 0, 50, 500, 4000])
def test_price_model_agrees_with_the_engine(item: str, offset: int) -> None:
    """The agent prices every product exactly as the engine would.

    This is the single most load-bearing assertion about the vendored code. Its
    edge is that it prices from live inventory, so a drift here would make every
    sell and buy decision confidently wrong across the whole product table.
    """
    inventory = EQUILIBRIUM + offset

    assert ep._price_at(item, inventory, None) == market_price(item, inventory, None)


@pytest.mark.parametrize("name", ["linear", "sq", "sqrt", "log", "log10"])
@pytest.mark.parametrize("value", [0.0, 1.0, 7.5, 400.0])
def test_shape_functions_match_the_engine_curves(name: str, value: float) -> None:
    """The five price-curve shapes are the engine's, not approximations of them."""
    from kaggle_environments.envs.kaggriculture import kaggriculture as engine

    assert ep._shape(name, value) == pytest.approx(engine._shape(name, value))


@pytest.mark.parametrize("hires", range(12))
def test_hire_cost_ladder_matches_the_engine(hires: int) -> None:
    """Hiring is priced fib(n) and resets daily; mis-modelling it mis-sizes the crew."""
    assert ep._fib(hires) == hire_cost(hires)


@pytest.mark.parametrize("position", [(0, 0), (4, 4), (5, 4), (4, 5), (9, 9), (5, 5)])
def test_quadrant_geometry_matches_the_engine(position: tuple[int, int]) -> None:
    """Land unlocks by quadrant, so a disagreement here misroutes every unit."""
    assert ep._quadrant_of(position, BOARD_SIZE) == quadrant_of(*position, BOARD_SIZE)


def test_shed_tiles_are_the_engine_s_shed_access_tiles() -> None:
    """Produce only banks from the four tiles touching the shed."""
    tiles = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]

    found = ep._shed_tiles(BOARD_SIZE, tiles)

    assert sorted(tuple(tile) for tile in found) == sorted(
        shed_access_tiles(BOARD_SIZE)
    )


def test_nearest_shed_picks_the_closest_access_tile() -> None:
    """A haul goes to the nearest drop, or the crew walks further than it must."""
    tiles = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]

    nearest = ep._nearest_shed((0, 0), BOARD_SIZE, tiles)

    assert tuple(nearest) == (4, 4)


def test_distance_is_manhattan() -> None:
    """Units move one orthogonal step per turn, so distance is steps, not euclidean."""
    assert ep._distance((0, 0), (3, 4)) == 7
    assert ep._distance((2, 2), (2, 2)) == 0


def test_town_demand_grows_at_the_documented_day_thresholds() -> None:
    """Town centre demand steps up on day 10 and again on day 20."""
    early = ep._town_demand_per_day({"day": 0, "town": {"unlocked_shops": []}}, "MELON")
    middle = ep._town_demand_per_day(
        {"day": 10, "town": {"unlocked_shops": []}}, "MELON"
    )
    late = ep._town_demand_per_day({"day": 20, "town": {"unlocked_shops": []}}, "MELON")

    assert (early, middle, late) == (2, 4, 8)


def test_fertilizer_has_no_town_centre_demand() -> None:
    """Only shops absorb fertilizer, which is why it stays cheap enough to buy."""
    demand = ep._town_demand_per_day(
        {"day": 25, "town": {"unlocked_shops": []}}, "FERTILIZER"
    )

    assert demand == 0


def test_a_single_product_shop_absorbs_double_a_mixed_one() -> None:
    """Shop absorption is what stops a glut, so its size drives every sell decision."""
    from kaggriculture.constants import SHOPS

    single = next(name for name, items in SHOPS.items() if len(items) == 1)
    mixed = next(name for name, items in SHOPS.items() if len(items) > 1)
    item = SHOPS[single][0]

    alone = ep._town_demand_per_day(
        {"day": 0, "town": {"unlocked_shops": [single]}}, item
    )
    baseline = ep._town_demand_per_day({"day": 0, "town": {"unlocked_shops": []}}, item)
    mixed_item = SHOPS[mixed][0]
    together = ep._town_demand_per_day(
        {"day": 0, "town": {"unlocked_shops": [mixed]}}, mixed_item
    )

    assert alone - baseline == 12
    assert (
        together
        - ep._town_demand_per_day(
            {"day": 0, "town": {"unlocked_shops": []}}, mixed_item
        )
        == 6
    )


@pytest.mark.parametrize("item", PRODUCTS)
def test_price_never_falls_below_the_engine_floor(item: str) -> None:
    """A price of zero would let the agent give produce away; the engine floors at 1."""
    flooded = ep._price_at(item, EQUILIBRIUM + 100_000, None)

    assert flooded >= 1
    assert flooded == market_price(item, EQUILIBRIUM + 100_000, None)


def test_market_parameters_cover_every_tradeable_product() -> None:
    """A missing product would silently price at defaults rather than fail."""
    for item in MARKET_PARAMS:
        base = ep._market_parameters(None, item)[0]

        assert base == MARKET_PARAMS[item]["base"]
