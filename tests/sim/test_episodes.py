"""Whole-trajectory differential campaigns."""
# ruff: noqa: D103

import pytest
import torch
from kaggle_environments import make

from kaggriculture.learn.encoding import MAX_UNITS, UNIT_OPS
from kaggriculture.sim.config import Config
from kaggriculture.sim.engine import MarketActions, reset, step, unit_quantity_ones
from kaggriculture.sim.state import ANIMAL_NAMES, CROP_NAMES, PRODUCT_NAMES
from tests.sim.conftest import assert_identical


def _unit_list(index: int) -> list[object]:
    verb, _, item = UNIT_OPS[index].partition(":")
    if not item:
        return [verb]
    return [verb, item] if verb == "PLANT" else [verb, item, 1]


def _market_list(actions: MarketActions, batch: int, seat: int) -> list[list[object]]:
    catalogues = {
        1: ("SELL", PRODUCT_NAMES),
        2: ("BUY_SEED", CROP_NAMES),
        3: ("BUY_PRODUCT", ("WHEAT", "FERTILIZER")),
        4: ("BUY_ANIMAL", ANIMAL_NAMES),
    }
    orders: list[list[object]] = []
    for slot in range(10):
        kind = int(actions.order_type[batch, seat, slot])
        if kind == 0:
            continue
        if kind == 5:
            orders.append(["HIRE"])
        elif kind == 6:
            orders.append(["BUY_LAND"])
        elif kind in catalogues:
            verb, items = catalogues[kind]
            item = int(actions.order_item[batch, seat, slot])
            orders.append(
                [verb, items[item], int(actions.order_qty[batch, seat, slot])]
            )
        else:
            orders.append(["MALFORMED"])
    return orders


@pytest.mark.slow
def test_random_full_vocabulary_campaign_has_zero_divergences(
    request: pytest.FixtureRequest,
) -> None:
    episode_count = int(request.config.getoption("--sim-episodes"))
    turn_count = int(request.config.getoption("--sim-turns"))
    seeds = torch.arange(episode_count, dtype=torch.int64) * 18 + 173
    environments = []
    for seed in seeds.tolist():
        environment = make("kaggriculture", configuration={"seed": seed}, debug=True)
        environment.reset(2)
        environments.append(environment)
    state = reset(Config(), seeds)
    generator = torch.Generator().manual_seed(20260810)

    for _turn in range(turn_count):
        units = torch.randint(
            0,
            len(UNIT_OPS),
            (len(seeds), 2, MAX_UNITS),
            generator=generator,
            dtype=torch.int16,
        )
        markets = MarketActions.empty(len(seeds))
        for batch in range(len(seeds)):
            for seat in range(2):
                for slot in range(2):
                    if int(torch.randint(0, 4, (), generator=generator)) != 0:
                        continue
                    kind = int(torch.randint(1, 7, (), generator=generator))
                    markets.order_type[batch, seat, slot] = kind
                    catalogue_size = {
                        1: len(PRODUCT_NAMES),
                        2: len(CROP_NAMES),
                        3: 2,
                        4: len(ANIMAL_NAMES),
                    }.get(kind, 1)
                    markets.order_item[batch, seat, slot] = int(
                        torch.randint(0, catalogue_size, (), generator=generator)
                    )
                    markets.order_qty[batch, seat, slot] = int(
                        torch.randint(1, 4, (), generator=generator)
                    )
        for batch, environment in enumerate(environments):
            reference_actions = []
            for seat in range(2):
                count = 1 + len(environment.state[0].observation.farms[seat].hands)
                reference_actions.append(
                    {
                        "farmer": _unit_list(int(units[batch, seat, 0])),
                        "hands": [
                            _unit_list(int(units[batch, seat, unit]))
                            for unit in range(1, count)
                        ],
                        "market": _market_list(markets, batch, seat),
                    }
                )
            environment.step(reference_actions)
        state = step(state, units, markets, unit_quantity_ones(len(seeds)))
        for batch, environment in enumerate(environments):
            assert_identical(environment, state, batch)
