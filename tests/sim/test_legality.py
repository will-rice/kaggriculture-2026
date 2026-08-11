"""Batched legality parity tests."""
# ruff: noqa: D103

import torch
from kaggle_environments import make

from kaggriculture.learn.mask import market_mask, unit_mask
from kaggriculture.sim.legality import legal
from kaggriculture.sim.state import pack


def test_legal_matches_reference_masks_for_both_seats_and_a_batch() -> None:
    environments = []
    for seed in (11, 23):
        environment = make("kaggriculture", configuration={"seed": seed}, debug=True)
        environment.reset(2)
        environments.append(environment)
    state = pack(environments)

    for seat in range(2):
        units, market = legal(state, seat)
        for batch, environment in enumerate(environments):
            observation = environment.state[seat].observation
            assert torch.equal(units[batch : batch + 1], unit_mask(observation, seat))
            assert torch.equal(
                market[batch : batch + 1], market_mask(observation, seat)
            )


def test_legal_matches_reference_on_plant_animal_shed_and_market_branches() -> None:
    environment = make("kaggriculture", configuration={"seed": 37}, debug=True)
    environment.reset(2)
    observation = environment.state[0].observation
    farm = observation.farms[0]
    private = environment.state[0].observation.private
    farm.farmer = [2, 2]
    farm.hands = [[4, 4], [3, 3]]
    farm.tiles[2][2] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 0,
        "watered_today": False,
        "consecutive_unwatered": 1,
        "yield_units": 2,
        "max_lifespan_step": 100,
        "fertilized_until_day": -1,
    }
    farm.tiles[3][3] = {
        "kind": "COOP",
        "animal": "GOOSE",
        "placed_day": 0,
        "yield_units": 1,
        "consecutive_unfed": 0,
        "fed_today": False,
        "cared_today": False,
        "fertilizer_available": True,
        "pending_care_bonus": 0,
    }
    private.shed["WHEAT"] = 12
    private.seeds["CARROT"] = 3
    private.inventories = [
        {"FERTILIZER": 1},
        {"MILK": 2},
        {"WHEAT": 1},
    ]
    farm.money = 4_321.0
    state = pack([environment])

    units, market = legal(state, 0)

    assert torch.equal(units, unit_mask(observation, 0))
    assert torch.equal(market, market_mask(observation, 0))
