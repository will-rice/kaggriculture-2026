"""End-to-end simulator step parity tests."""
# ruff: noqa: ANN001, ANN202, D103

import torch
from kaggle_environments import make

from kaggriculture.learn.encoding import MAX_UNITS, UNIT_OPS
from kaggriculture.sim.engine import MarketActions, step, unit_quantity_ones
from kaggriculture.sim.state import pack, unpack


def _actions(unit_names, market_orders):
    units = torch.full((1, 2, MAX_UNITS), UNIT_OPS.index("PASS"), dtype=torch.int16)
    for seat, names in enumerate(unit_names):
        for slot, name in enumerate(names):
            units[0, seat, slot] = UNIT_OPS.index(name)
    market = MarketActions.empty(1)
    types = {
        "SELL": 1,
        "BUY_SEED": 2,
        "BUY_PRODUCT": 3,
        "BUY_ANIMAL": 4,
        "HIRE": 5,
        "BUY_LAND": 6,
    }
    catalogues = {
        "SELL": tuple(
            sorted(
                (
                    "WHEAT",
                    "CARROT",
                    "TOMATO",
                    "STRAWBERRY",
                    "MELON",
                    "EGG",
                    "MILK",
                    "WOOL",
                    "FERTILIZER",
                )
            )
        ),
        "BUY_PRODUCT": ("WHEAT", "FERTILIZER"),
        "BUY_SEED": tuple(sorted(("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"))),
        "BUY_ANIMAL": tuple(sorted(("GOOSE", "COW", "SHEEP"))),
    }
    for seat, orders in enumerate(market_orders):
        for slot, order in enumerate(orders):
            op = order[0]
            market.order_type[0, seat, slot] = types[op]
            if len(order) > 1:
                market.order_item[0, seat, slot] = catalogues[op].index(order[1])
            market.order_qty[0, seat, slot] = 1 if len(order) < 3 else order[2]
    return units, market


def test_step_matches_reference_for_unit_and_market_actions() -> None:
    environment = make("kaggriculture", configuration={"seed": 17}, debug=True)
    environment.reset(2)
    state = pack([environment])
    reference_actions = [
        {"farmer": ["EAST"], "hands": [], "market": [["BUY_SEED", "WHEAT", 2]]},
        {"farmer": ["SOUTH"], "hands": [], "market": [["HIRE"]]},
    ]
    units, market = _actions(
        [["EAST"], ["SOUTH"]],
        [[["BUY_SEED", "WHEAT", 2]], [["HIRE"]]],
    )

    environment.step(reference_actions)
    actual = step(state, units, market, unit_quantity_ones(1))

    for seat in range(2):
        expected = dict(environment.state[seat].observation)
        expected.pop("remainingOverageTime", None)
        assert unpack(actual, 0, seat) == expected


def test_step_matches_reference_across_an_end_of_day() -> None:
    environment = make("kaggriculture", configuration={"seed": 29}, debug=True)
    environment.reset(2)
    passes = [{"farmer": ["PASS"], "hands": [], "market": []}] * 2
    for _ in range(23):
        environment.step(passes)
    state = pack([environment])
    units, market = _actions([["PASS"], ["PASS"]], [[], []])

    environment.step(passes)
    actual = step(state, units, market, unit_quantity_ones(1))

    for seat in range(2):
        expected = dict(environment.state[seat].observation)
        expected.pop("remainingOverageTime", None)
        assert unpack(actual, 0, seat) == expected
