"""Verify every unit verb plus storage, land, and daily labour economics."""

from __future__ import annotations

from copy import deepcopy

from kaggle_environments.envs.kaggriculture import kaggriculture as engine

from _common import PASS, assert_exact_engine, emit, new_env


BOARD = 10
MONEY = 3000
DAY = 2
TPD = 24
CAP = 100


def fresh():
    return engine._new_farm(BOARD, MONEY), engine._new_private()


def apply(farm, private, action, *, day=DAY, idx=0):
    engine._apply_unit_action(farm, private, idx, action, BOARD, day, TPD, CAP)


def unit_actions() -> dict[str, object]:
    effects: dict[str, object] = {}

    farm, private = fresh()
    origin = list(farm["farmer"])
    positions = {}
    for op in ("NORTH", "SOUTH", "EAST", "WEST"):
        farm["farmer"] = origin.copy()
        apply(farm, private, [op])
        positions[op] = list(farm["farmer"])
    assert positions == {
        "NORTH": [4, 3], "SOUTH": [4, 5], "EAST": [5, 4], "WEST": [3, 4]
    }
    assert farm["money"] == MONEY
    effects["movement"] = {"origin": origin, "destinations": positions, "money_cost": 0}

    farm, private = fresh()
    before = deepcopy((farm, private))
    apply(farm, private, ["PASS"])
    assert (farm, private) == before
    effects["PASS"] = {"state_changes": 0, "money_cost": 0}

    farm, private = fresh()
    private["inventories"][0]["WHEAT"] = 101
    apply(farm, private, ["DROP"])
    assert private["shed"]["WHEAT"] == 100
    assert private["inventories"][0] == {}
    effects["DROP"] = {
        "carried_before": 101, "shed_after": 100, "carried_after": 0,
        "overflow_destroyed": 1,
    }

    farm, private = fresh()
    private["shed"]["WHEAT"] = 5
    apply(farm, private, ["PICKUP", "WHEAT", 3])
    assert private["shed"]["WHEAT"] == 2
    assert private["inventories"][0]["WHEAT"] == 3
    farm2, private2 = fresh()
    private2["shed"]["WHEAT"] = 2
    apply(farm2, private2, ["PICKUP", "WHEAT"])
    assert private2["inventories"][0]["WHEAT"] == 1
    effects["PICKUP"] = {
        "requested": 3, "picked_up": 3, "shed_remaining": 2, "default_quantity": 1
    }

    farm, private = fresh()
    private["shed"]["FERTILIZER"] = 99
    private["inventories"][0]["WHEAT"] = 5
    apply(farm, private, ["PLACE", "WHEAT", 3])
    assert private["shed"]["WHEAT"] == 1
    assert private["inventories"][0]["WHEAT"] == 4
    farm2, private2 = fresh()
    private2["inventories"][0]["WHEAT"] = 2
    apply(farm2, private2, ["PLACE", "WHEAT"])
    assert private2["shed"]["WHEAT"] == 1 and private2["inventories"][0]["WHEAT"] == 1
    effects["PLACE_shed"] = {
        "requested": 3, "placed_with_one_slot": 1, "still_carried": 4,
        "overflow_destroyed": 0, "default_quantity": 1,
    }

    farm, private = fresh()
    apply(farm, private, ["BUILD_COOP"])
    assert farm["tiles"][4][4] == {"kind": "COOP"}
    private["inventories"][0]["GOOSE"] = 1
    apply(farm, private, ["PLACE", "GOOSE"])
    assert farm["tiles"][4][4]["animal"] == "GOOSE"
    assert "GOOSE" not in private["inventories"][0]
    effects["BUILD_COOP_and_PLACE_animal"] = {
        "money_cost": 0, "animals_consumed": 1, "animal": "GOOSE"
    }

    farm, private = fresh()
    apply(farm, private, ["BUILD_PASTURE"])
    assert farm["tiles"][4][4] == {"kind": "PASTURE"}
    effects["BUILD_PASTURE"] = {"money_cost": 0}

    farm, private = fresh()
    private["seeds"]["WHEAT"] = 1
    apply(farm, private, ["PLANT", "WHEAT"], day=0)
    tile = farm["tiles"][4][4]
    assert tile["crop"] == "WHEAT" and private["seeds"]["WHEAT"] == 0
    assert tile["yield_units"] == 1 and tile["consecutive_unwatered"] == 1
    effects["PLANT"] = {
        "seeds_consumed": 1, "initial_yield_units": 1,
        "initial_consecutive_unwatered": 1,
    }

    apply(farm, private, ["WATER"], day=2)
    assert tile["watered_today"] and tile["yield_units"] == 2
    effects["WATER"] = {"base_yield_added_in_window": 1, "uses_item": False}

    farm, private = fresh()
    private["seeds"]["WHEAT"] = 1
    apply(farm, private, ["PLANT", "WHEAT"], day=0)
    private["inventories"][0]["FERTILIZER"] = 1
    apply(farm, private, ["FERTILIZE"], day=2)
    tile = farm["tiles"][4][4]
    assert tile["fertilized_until_day"] == 4
    assert "FERTILIZER" not in private["inventories"][0]
    apply(farm, private, ["WATER"], day=2)
    assert tile["yield_units"] == 3
    effects["FERTILIZE"] = {
        "fertilizer_consumed": 1, "active_days": [2, 3, 4],
        "water_yield_added_while_active": 2,
    }

    farm, private = fresh()
    farm["tiles"][4][4] = engine._new_plant("WHEAT", 0, TPD)
    farm["tiles"][4][4]["yield_units"] = 3
    apply(farm, private, ["HARVEST"], day=2)
    assert private["inventories"][0]["WHEAT"] == 3
    assert farm["tiles"][4][4] is None
    effects["HARVEST_one_time"] = {"yield_transferred": 3, "tile_cleared": True}

    farm, private = fresh()
    farm["tiles"][4][4] = engine._new_plant("TOMATO", 0, TPD)
    farm["tiles"][4][4]["yield_units"] = 2
    apply(farm, private, ["HARVEST"], day=8)
    assert private["inventories"][0]["TOMATO"] == 2
    assert farm["tiles"][4][4]["yield_units"] == 0
    effects["HARVEST_ongoing"] = {"yield_transferred": 2, "tile_cleared": False}

    farm, private = fresh()
    farm["tiles"][4][4] = {"kind": "WEED"}
    apply(farm, private, ["DIG"])
    assert farm["tiles"][4][4] is None
    farm["tiles"][4][4] = {"kind": "COOP"}
    apply(farm, private, ["DIG"])
    assert farm["tiles"][4][4] is None
    farm["tiles"][4][4] = engine._new_animal("GOOSE", 0)
    apply(farm, private, ["DIG"])
    assert farm["tiles"][4][4]["animal"] == "GOOSE"
    effects["DIG"] = {
        "clears_weed_plant_or_empty_structure": True,
        "clears_occupied_structure": False,
        "money_cost": 0,
    }

    farm, private = fresh()
    farm["tiles"][4][4] = engine._new_animal("GOOSE", 0)
    private["inventories"][0]["WHEAT"] = 2
    apply(farm, private, ["FEED"])
    apply(farm, private, ["FEED"])
    assert farm["tiles"][4][4]["fed_today"]
    assert private["inventories"][0]["WHEAT"] == 1
    effects["FEED"] = {"wheat_consumed_at_most_once_per_day": 1}

    farm["tiles"][4][4]["fertilizer_available"] = True
    apply(farm, private, ["COLLECT_FERTILIZER"])
    apply(farm, private, ["COLLECT_FERTILIZER"])
    assert private["inventories"][0]["FERTILIZER"] == 1
    effects["COLLECT_FERTILIZER"] = {"collected_when_available": 1}

    apply(farm, private, ["CARE"])
    apply(farm, private, ["CARE"])
    assert farm["tiles"][4][4]["cared_today"]
    effects["CARE"] = {"uses_item": False, "effective_at_most_once_per_day": 1}

    farm, private = fresh()
    apply(farm, private, ["EAST"])
    assert farm["tiles"][4][5] == "LOCKED"
    private["inventories"][0]["WHEAT"] = 1
    apply(farm, private, ["DROP"])
    assert private["shed"]["WHEAT"] == 1
    private["seeds"]["WHEAT"] = 1
    apply(farm, private, ["PLANT", "WHEAT"])
    assert farm["tiles"][4][5] == "LOCKED" and private["seeds"]["WHEAT"] == 1
    effects["locked_tiles"] = {
        "movement_allowed": True, "shed_access_allowed_from_inner_corner": True,
        "tile_mutation_allowed": False,
    }

    expected_verbs = {
        "NORTH", "SOUTH", "EAST", "WEST", "PASS", "DROP", "PICKUP", "PLACE",
        "PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG", "BUILD_COOP",
        "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE",
    }
    tested_verbs = {
        "NORTH", "SOUTH", "EAST", "WEST", "PASS", "DROP", "PICKUP", "PLACE",
        "PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG", "BUILD_COOP",
        "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE",
    }
    assert tested_verbs == expected_verbs and len(tested_verbs) == 18
    effects["tested_verbs"] = sorted(tested_verbs)
    return effects


def plant_atomicity_and_ordering() -> dict[str, object]:
    env = new_env(seed=1, weedSpawnChance=0)
    env.reset(2)
    env.step([
        {"farmer": ["PASS"], "hands": [], "market": [
            ["HIRE"], ["BUY_LAND"], ["BUY_SEED", "WHEAT", 1]
        ]},
        PASS,
    ])
    obs = env.state[0].observation
    assert obs.private.seeds.WHEAT == 1
    assert obs.farms[0].money == 1989
    env.step([
        {"farmer": ["PLANT", "WHEAT"], "hands": [["PLANT", "WHEAT"]], "market": []},
        PASS,
    ])
    obs = env.state[0].observation
    assert obs.private.seeds.WHEAT == 1
    assert obs.farms[0].tiles[4][4] is None
    assert obs.farms[0].tiles[4][5] is None
    return {
        "available_seeds": 1,
        "simultaneous_plant_requests": 2,
        "requests_executed_when_demand_exceeds_stock": 0,
        "market_bought_seed_available_on_next_action_step": True,
    }


def shed_end_of_day() -> dict[str, object]:
    private = engine._new_private()
    private["shed"]["WHEAT"] = 99
    private["inventories"] = [{"WHEAT": 3}]
    private["seeds"]["MELON"] = 500
    engine._drop_inventories_to_shed(private, CAP)
    assert private["shed"]["WHEAT"] == 100
    assert private["inventories"] == [{}]
    assert private["seeds"]["MELON"] == 500
    return {
        "capacity": 100,
        "shed_before": 99,
        "carried_before": 3,
        "shed_after": 100,
        "overflow_destroyed": 2,
        "seeds_outside_capacity": 500,
    }


def land_and_hiring() -> dict[str, object]:
    farm, private = fresh()
    initial_owned = sum(tile != "LOCKED" for row in farm["tiles"] for tile in row)
    initial_locked = sum(tile == "LOCKED" for row in farm["tiles"] for tile in row)
    assert (initial_owned, initial_locked) == (25, 75)
    land_banks = [farm["money"]]
    unlocked = [list(farm["unlocked_quadrants"])]
    for _ in range(3):
        engine._do_buy_land(farm, BOARD)
        land_banks.append(farm["money"])
        unlocked.append(list(farm["unlocked_quadrants"])
        )
    assert land_banks == [3000.0, 2000.0, 0.0, 0.0]
    # The third parcel costs 4000, so it cannot be bought from this bank.
    farm["money"] = 4000
    engine._do_buy_land(farm, BOARD)
    assert farm["money"] == 0 and farm["unlocked_quadrants"] == ["NW", "NE", "SW", "SE"]
    assert sum(tile == "LOCKED" for row in farm["tiles"] for tile in row) == 0

    farm, private = fresh()
    hire_costs = [engine._hire_cost(i) for i in range(10)]
    assert hire_costs == [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    before = farm["money"]
    for _ in range(7):
        engine._do_hire(farm, private, BOARD)
    assert before - farm["money"] == 33
    assert len(farm["hands"]) == 7 and len(private["inventories"]) == 8
    assert farm["hands"][:4] == [[5, 4], [4, 5], [5, 5], [4, 4]]
    return {
        "board": [10, 10],
        "quadrant": [5, 5],
        "initial_owned_and_locked_tiles": [initial_owned, initial_locked],
        "unlock_order": ["NE", "SW", "SE"],
        "land_prices": [1000, 2000, 4000],
        "all_unlocked_tiles": 100,
        "hire_costs_first_ten": hire_costs,
        "first_seven_total": 33,
        "first_four_hand_spawns": farm["hands"][:4],
        "hands_are_day_labor": True,
    }


if __name__ == "__main__":
    emit({
        "engine": assert_exact_engine(),
        "unit_actions": unit_actions(),
        "plant_atomicity": plant_atomicity_and_ordering(),
        "shed": shed_end_of_day(),
        "land_and_hiring": land_and_hiring(),
    })
