"""Tensor unit-phase differential tests."""
# ruff: noqa: ANN001, ANN202, D103

import torch
from kaggle_environments import make

from kaggriculture.learn.encoding import MAX_UNITS, UNIT_OPS
from kaggriculture.sim.day import apply_day_phases
from kaggriculture.sim.state import pack, unpack
from kaggriculture.sim.units import apply_unit_phases


def _unit_tensor(names):
    actions = torch.full((1, 2, MAX_UNITS), UNIT_OPS.index("PASS"), dtype=torch.int16)
    for seat, seat_names in enumerate(names):
        for unit, name in enumerate(seat_names):
            actions[0, seat, unit] = UNIT_OPS.index(name)
    return actions


def _assert_observations(environment, state) -> None:
    for seat in range(2):
        expected = dict(environment.state[seat].observation)
        expected.pop("remainingOverageTime", None)
        assert unpack(state, 0, seat) == expected


def test_tensor_units_match_movement_build_and_atomic_plant_guard() -> None:
    environment = make("kaggriculture", configuration={"seed": 131}, debug=True)
    environment.reset(2)
    observation = environment.state[0].observation
    for seat in range(2):
        observation.farms[seat].hands = [[3, 4], [2, 4]]
        environment.state[seat].observation.private.inventories = [{}, {}, {}]
    environment.state[0].observation.private.seeds["WHEAT"] = 2
    state = pack([environment])
    reference = [
        {
            "farmer": ["PLANT", "WHEAT"],
            "hands": [["PLANT", "WHEAT"], ["PLANT", "WHEAT"]],
            "market": [],
        },
        {"farmer": ["EAST"], "hands": [["BUILD_COOP"], ["PASS"]], "market": []},
    ]
    actions = _unit_tensor(
        [
            ["PLANT:WHEAT", "PLANT:WHEAT", "PLANT:WHEAT"],
            ["EAST", "BUILD_COOP", "PASS"],
        ]
    )

    environment.step(reference)
    actual = apply_day_phases(apply_unit_phases(state, actions))

    _assert_observations(environment, actual)


def test_tensor_units_match_locked_shed_transfers_and_inventory_order() -> None:
    environment = make("kaggriculture", configuration={"seed": 137}, debug=True)
    environment.reset(2)
    observation = environment.state[0].observation
    observation.farms[0].farmer = [5, 4]
    private = environment.state[0].observation.private
    private.shed["CARROT"] = 98
    private.inventories = [{"WHEAT": 2, "MILK": 1}]
    observation.farms[1].farmer = [4, 4]
    environment.state[1].observation.private.shed["WOOL"] = 1
    state = pack([environment])
    reference = [
        {"farmer": ["DROP"], "hands": [], "market": []},
        {"farmer": ["PICKUP", "WOOL", 1], "hands": [], "market": []},
    ]
    actions = _unit_tensor([["DROP"], ["PICKUP:WOOL"]])

    environment.step(reference)
    actual = apply_day_phases(apply_unit_phases(state, actions))

    _assert_observations(environment, actual)


def test_tensor_units_match_water_feed_care_and_fertilizer_collection() -> None:
    environment = make("kaggriculture", configuration={"seed": 139}, debug=True)
    environment.reset(2)
    observation = environment.state[0].observation
    observation.farms[0].farmer = [1, 1]
    observation.farms[0].hands = [[1, 1]]
    observation.farms[0].tiles[1][1] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 0,
        "watered_today": False,
        "consecutive_unwatered": 1,
        "yield_units": 1,
        "max_lifespan_step": 120,
        "fertilized_until_day": 0,
    }
    environment.state[0].observation.private.inventories = [{}, {}]
    observation.farms[1].farmer = [2, 2]
    observation.farms[1].hands = [[2, 2], [2, 2]]
    observation.farms[1].tiles[2][2] = {
        "kind": "COOP",
        "animal": "GOOSE",
        "placed_day": 0,
        "yield_units": 0,
        "consecutive_unfed": 0,
        "fed_today": False,
        "cared_today": False,
        "fertilizer_available": True,
        "pending_care_bonus": 0,
    }
    environment.state[1].observation.private.inventories = [
        {"WHEAT": 1},
        {},
        {},
    ]
    state = pack([environment])
    reference = [
        {"farmer": ["WATER"], "hands": [["WATER"]], "market": []},
        {
            "farmer": ["FEED"],
            "hands": [["CARE"], ["COLLECT_FERTILIZER"]],
            "market": [],
        },
    ]
    actions = _unit_tensor([["WATER", "WATER"], ["FEED", "CARE", "COLLECT_FERTILIZER"]])

    environment.step(reference)
    actual = apply_day_phases(apply_unit_phases(state, actions))

    _assert_observations(environment, actual)


def test_tensor_units_match_animal_placement_and_mature_harvest() -> None:
    environment = make("kaggriculture", configuration={"seed": 149}, debug=True)
    environment.reset(2)
    observation = environment.state[0].observation
    observation.farms[0].farmer = [1, 1]
    observation.farms[0].tiles[1][1] = {"kind": "PASTURE"}
    environment.state[0].observation.private.inventories = [{"COW": 1}]
    observation.farms[1].farmer = [2, 2]
    observation.farms[1].tiles[2][2] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": -5,
        "watered_today": True,
        "consecutive_unwatered": 0,
        "yield_units": 2,
        "max_lifespan_step": 200,
        "fertilized_until_day": -1,
    }
    state = pack([environment])
    reference = [
        {"farmer": ["PLACE", "COW", 1], "hands": [], "market": []},
        {"farmer": ["HARVEST"], "hands": [], "market": []},
    ]
    actions = _unit_tensor([["PLACE:COW"], ["HARVEST"]])

    environment.step(reference)
    actual = apply_day_phases(apply_unit_phases(state, actions))

    _assert_observations(environment, actual)


def test_fertilizer_bonus_includes_its_final_day() -> None:
    environment = make("kaggriculture", configuration={"seed": 173}, debug=True)
    environment.reset(2)
    passes = [{"farmer": ["PASS"], "hands": [], "market": []}] * 2
    for _ in range(48):
        environment.step(passes)
    observation = environment.state[0].observation
    observation.farms[0].farmer = [1, 1]
    observation.farms[0].tiles[1][1] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 0,
        "watered_today": False,
        "consecutive_unwatered": 0,
        "yield_units": 1,
        "max_lifespan_step": 120,
        "fertilized_until_day": 2,
    }
    state = pack([environment])
    reference = [
        {"farmer": ["WATER"], "hands": [], "market": []},
        {"farmer": ["PASS"], "hands": [], "market": []},
    ]
    actions = _unit_tensor([["WATER"], ["PASS"]])

    environment.step(reference)
    actual = apply_day_phases(apply_unit_phases(state, actions))

    _assert_observations(environment, actual)


def test_harvest_refuses_a_crop_before_first_yield_day() -> None:
    environment = make("kaggriculture", configuration={"seed": 179}, debug=True)
    environment.reset(2)
    observation = environment.state[0].observation
    observation.farms[0].farmer = [1, 1]
    observation.farms[0].tiles[1][1] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 0,
        "watered_today": True,
        "consecutive_unwatered": 0,
        "yield_units": 1,
        "max_lifespan_step": 120,
        "fertilized_until_day": -1,
    }
    state = pack([environment])
    reference = [
        {"farmer": ["HARVEST"], "hands": [], "market": []},
        {"farmer": ["PASS"], "hands": [], "market": []},
    ]
    actions = _unit_tensor([["HARVEST"], ["PASS"]])

    environment.step(reference)
    actual = apply_day_phases(apply_unit_phases(state, actions))

    _assert_observations(environment, actual)


def test_dig_refuses_occupied_structure_and_place_requires_matching_kind() -> None:
    environment = make("kaggriculture", configuration={"seed": 181}, debug=True)
    environment.reset(2)
    observation = environment.state[0].observation
    observation.farms[0].farmer = [1, 1]
    observation.farms[0].tiles[1][1] = {
        "kind": "COOP",
        "animal": "GOOSE",
        "placed_day": 0,
        "yield_units": 0,
        "consecutive_unfed": 0,
        "fed_today": False,
        "cared_today": False,
        "fertilizer_available": False,
        "pending_care_bonus": 0,
    }
    observation.farms[1].farmer = [2, 2]
    observation.farms[1].tiles[2][2] = {"kind": "COOP"}
    environment.state[1].observation.private.inventories = [{"COW": 1}]
    state = pack([environment])
    reference = [
        {"farmer": ["DIG"], "hands": [], "market": []},
        {"farmer": ["PLACE", "COW", 1], "hands": [], "market": []},
    ]
    actions = _unit_tensor([["DIG"], ["PLACE:COW"]])

    environment.step(reference)
    actual = apply_day_phases(apply_unit_phases(state, actions))

    _assert_observations(environment, actual)
