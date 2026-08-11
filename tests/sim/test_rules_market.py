"""Tensor market-phase differential tests."""
# ruff: noqa: ANN001, D103

import torch
from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as reference_engine

from kaggriculture.sim.day import apply_day_phases
from kaggriculture.sim.engine import MarketActions
from kaggriculture.sim.market import apply_market_phase
from kaggriculture.sim.state import (
    CROP_NAMES,
    PRODUCT_NAMES,
    pack,
    unpack,
)
from tests.sim.states import hire_ladder, price_at_floor, shed_one_below_capacity


def _assert_observations(environment, state) -> None:
    for seat in range(2):
        expected = dict(environment.state[seat].observation)
        expected.pop("remainingOverageTime", None)
        assert unpack(state, 0, seat) == expected


def test_tensor_market_matches_sale_funding_later_purchase_and_lockstep() -> None:
    environment = make("kaggriculture", configuration={"seed": 151}, debug=True)
    environment.reset(2)
    observation = environment.state[0].observation
    observation.farms[0].money = 0.0
    observation.farms[1].money = 0.0
    environment.state[0].observation.private.shed["MELON"] = 2
    environment.state[1].observation.private.shed["MELON"] = 2
    state = pack([environment])
    actions = MarketActions.empty(1)
    for seat in range(2):
        actions.order_type[0, seat, 0] = 1
        actions.order_item[0, seat, 0] = PRODUCT_NAMES.index("MELON")
        actions.order_qty[0, seat, 0] = 2
    actions.order_type[0, 0, 1] = 2
    actions.order_item[0, 0, 1] = CROP_NAMES.index("WHEAT")
    actions.order_qty[0, 0, 1] = 1
    reference = [
        {
            "farmer": ["PASS"],
            "hands": [],
            "market": [["SELL", "MELON", 2], ["BUY_SEED", "WHEAT", 1]],
        },
        {"farmer": ["PASS"], "hands": [], "market": [["SELL", "MELON", 2]]},
    ]

    environment.step(reference)
    actual = apply_day_phases(apply_market_phase(state, actions))

    _assert_observations(environment, actual)


def test_tensor_market_matches_hire_land_and_capacity_guards() -> None:
    environment = make("kaggriculture", configuration={"seed": 157}, debug=True)
    environment.reset(2)
    environment.state[0].observation.private.shed["CARROT"] = 100
    state = pack([environment])
    actions = MarketActions.empty(1)
    actions.order_type[0, 0, 0] = 3
    actions.order_item[0, 0, 0] = 0  # WHEAT
    actions.order_qty[0, 0, 0] = 2
    actions.order_type[0, 0, 1] = 5
    actions.order_type[0, 1, 0] = 6
    reference = [
        {
            "farmer": ["PASS"],
            "hands": [],
            "market": [["BUY_PRODUCT", "WHEAT", 2], ["HIRE"]],
        },
        {"farmer": ["PASS"], "hands": [], "market": [["BUY_LAND"]]},
    ]

    environment.step(reference)
    actual = apply_day_phases(apply_market_phase(state, actions))

    _assert_observations(environment, actual)


def test_price_floor_sale_does_not_add_supply() -> None:
    environment = price_at_floor("WHEAT")
    environment.state[0].observation.private.shed["WHEAT"] = 1
    state = pack([environment])
    actions = MarketActions.empty(1)
    actions.order_type[0, 0, 0] = 1
    actions.order_item[0, 0, 0] = PRODUCT_NAMES.index("WHEAT")
    actions.order_qty[0, 0, 0] = 1
    reference = [
        {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", 1]]},
        {"farmer": ["PASS"], "hands": [], "market": []},
    ]

    environment.step(reference)
    actual = apply_day_phases(apply_market_phase(state, actions))

    _assert_observations(environment, actual)


def test_capacity_is_rechecked_for_each_bought_unit() -> None:
    environment = shed_one_below_capacity()
    state = pack([environment])
    actions = MarketActions.empty(1)
    actions.order_type[0, 0, 0] = 3
    actions.order_item[0, 0, 0] = 0
    actions.order_qty[0, 0, 0] = 2
    reference = [
        {
            "farmer": ["PASS"],
            "hands": [],
            "market": [["BUY_PRODUCT", "WHEAT", 2]],
        },
        {"farmer": ["PASS"], "hands": [], "market": []},
    ]

    environment.step(reference)
    actual = apply_day_phases(apply_market_phase(state, actions))

    _assert_observations(environment, actual)


def test_buy_product_quote_at_inventory_minus_one_nets_zero_round_trip() -> None:
    environment = make("kaggriculture", configuration={"seed": 163}, debug=True)
    environment.reset(2)
    state = pack([environment])
    actions = MarketActions.empty(1)
    actions.order_type[0, 0, :2] = torch.tensor([3, 1], dtype=torch.int8)
    actions.order_item[0, 0, :2] = torch.tensor(
        [0, PRODUCT_NAMES.index("WHEAT")], dtype=torch.int8
    )
    actions.order_qty[0, 0, :2] = 1
    reference = [
        {
            "farmer": ["PASS"],
            "hands": [],
            "market": [
                ["BUY_PRODUCT", "WHEAT", 1],
                ["SELL", "WHEAT", 1],
            ],
        },
        {"farmer": ["PASS"], "hands": [], "market": []},
    ]

    environment.step(reference)
    actual = apply_day_phases(apply_market_phase(state, actions))

    _assert_observations(environment, actual)


def test_hire_cost_uses_current_hires_today_fibonacci_index() -> None:
    environment = hire_ladder(hires_today=2, money=2)
    state = pack([environment])
    actions = MarketActions.empty(1)
    actions.order_type[0, 0, 0] = 5
    reference = [
        {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]]},
        {"farmer": ["PASS"], "hands": [], "market": []},
    ]

    environment.step(reference)
    actual = apply_day_phases(apply_market_phase(state, actions))

    _assert_observations(environment, actual)


def test_market_phase_refreshes_stored_prices_before_the_day_phase() -> None:
    environment = make("kaggriculture", configuration={"seed": 167}, debug=True)
    environment.reset(2)
    environment.state[0].observation.private.shed["MELON"] = 8
    state = pack([environment])
    actions = MarketActions.empty(1)
    actions.order_type[0, 0, 0] = 1
    actions.order_item[0, 0, 0] = PRODUCT_NAMES.index("MELON")
    actions.order_qty[0, 0, 0] = 8
    environment.state[0].action = {
        "farmer": ["PASS"],
        "hands": [],
        "market": [["SELL", "MELON", 8]],
    }
    environment.state[1].action = {"farmer": ["PASS"], "hands": [], "market": []}

    reference_engine._process_market(environment.state, environment)
    environment.state[1].observation.market = environment.state[0].observation.market
    environment.state[1].observation.farms = environment.state[0].observation.farms
    actual = apply_market_phase(state, actions)

    _assert_observations(environment, actual)
