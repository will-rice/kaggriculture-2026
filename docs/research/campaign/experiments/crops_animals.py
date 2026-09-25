"""Verify crop lifecycles and animal production directly against the engine."""

from __future__ import annotations

from kaggle_environments.envs.kaggriculture import kaggriculture as engine

from _common import assert_exact_engine, emit


TPD = 24


def crop_constants() -> dict[str, dict[str, object]]:
    expected = {
        "WHEAT": {
            "seed": 10,
            "first_yield_day": 2,
            "max_yield_day": 4,
            "interval": 0,
            "max_yield": 6,
            "ongoing": False,
        },
        "CARROT": {
            "seed": 20,
            "first_yield_day": 2,
            "max_yield_day": 3,
            "interval": 0,
            "max_yield": 4,
            "ongoing": False,
        },
        "TOMATO": {
            "seed": 50,
            "first_yield_day": 8,
            "max_yield_day": 8,
            "interval": 1,
            "max_yield": 4,
            "ongoing": True,
        },
        "STRAWBERRY": {
            "seed": 100,
            "first_yield_day": 10,
            "max_yield_day": 10,
            "interval": 2,
            "max_yield": 4,
            "ongoing": True,
        },
        "MELON": {
            "seed": 80,
            "first_yield_day": 10,
            "max_yield_day": 12,
            "interval": 0,
            "max_yield": 6,
            "ongoing": False,
        },
    }
    assert engine.CROPS == expected
    return expected


def one_time_water_windows() -> dict[str, object]:
    expected = {
        "WHEAT": {
            "productive_water_ages": [2, 3, 4],
            "plain_yield_by_age": [1, 1, 2, 3, 4],
            "fertilized_cap_age": 4,
        },
        "CARROT": {
            "productive_water_ages": [2, 3],
            "plain_yield_by_age": [1, 1, 2, 3],
            "fertilized_cap_age": 3,
        },
        "MELON": {
            "productive_water_ages": [6, 7, 8, 9, 10],
            "plain_yield_by_age": [1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 6, 6],
            "fertilized_cap_age": 8,
        },
    }
    result: dict[str, object] = {}
    for crop in ("WHEAT", "CARROT", "MELON"):
        data = engine.CROPS[crop]
        farm = engine._new_farm(10, 3000)
        private = engine._new_private()
        private["seeds"][crop] = 1
        engine._apply_unit_action(farm, private, 0, ["PLANT", crop], 10, 0, TPD, 100)
        tile = farm["tiles"][4][4]
        yields = []
        productive = []
        previous = tile["yield_units"]
        for age in range(data["max_yield_day"] + 1):
            tile["watered_today"] = False
            engine._apply_unit_action(farm, private, 0, ["WATER"], 10, age, TPD, 100)
            if tile["yield_units"] > previous:
                productive.append(age)
            previous = tile["yield_units"]
            yields.append(tile["yield_units"])

        farm = engine._new_farm(10, 3000)
        private = engine._new_private()
        private["seeds"][crop] = 1
        engine._apply_unit_action(farm, private, 0, ["PLANT", crop], 10, 0, TPD, 100)
        tile = farm["tiles"][4][4]
        window_start = (data["max_yield_day"] + 1) // 2
        private["inventories"][0]["FERTILIZER"] = 1
        engine._apply_unit_action(
            farm, private, 0, ["FERTILIZE"], 10, window_start, TPD, 100
        )
        cap_age = None
        for age in range(window_start, data["max_yield_day"] + 1):
            tile["watered_today"] = False
            engine._apply_unit_action(farm, private, 0, ["WATER"], 10, age, TPD, 100)
            if tile["yield_units"] == data["max_yield"] and cap_age is None:
                cap_age = age
        result[crop] = {
            "productive_water_ages": productive,
            "plain_yield_by_age": yields,
            "fertilized_cap_age": cap_age,
            "lifespan_decay_starts_step": (data["max_yield_day"] + 1) * TPD,
            "decay_every_steps": 2,
        }
        assert productive == expected[crop]["productive_water_ages"]
        assert yields == expected[crop]["plain_yield_by_age"]
        assert cap_age == expected[crop]["fertilized_cap_age"]
    return result


def ongoing_schedules() -> dict[str, object]:
    expected_days = {"TOMATO": [8, 9, 10, 11], "STRAWBERRY": [10, 12, 14, 16]}
    result = {}
    for crop, days in expected_days.items():
        farm = engine._new_farm(10, 3000)
        farm["tiles"][4][4] = engine._new_plant(crop, 0, TPD)
        tile = farm["tiles"][4][4]
        produced = []
        prior = 0
        for current_day in range(days[-1] + 1):
            tile = farm["tiles"][4][4]
            tile["watered_today"] = True
            engine._daily_refresh_plants(farm, current_day, TPD)
            tile = farm["tiles"][4][4]
            if tile["yield_units"] > prior:
                produced.append(current_day + 1)
            prior = tile["yield_units"]
        assert produced == days
        assert tile["yield_units"] == 4
        expected_decay = (days[-1] + 1) * TPD
        assert tile["max_lifespan_step"] == expected_decay
        for step in range(expected_decay, expected_decay + 8, 2):
            engine._decay_plants(farm, step)
        assert farm["tiles"][4][4] == {"kind": "WEED"}
        result[crop] = {
            "production_days_after_planting": produced,
            "total_productions": len(produced),
            "decay_starts_step": expected_decay,
            "full_unharvested_decay_ticks": 4,
            "production_continues_after_harvest": True,
        }
    return result


def watering_and_decay() -> dict[str, object]:
    farm = engine._new_farm(10, 3000)
    farm["tiles"][4][4] = engine._new_plant("WHEAT", 0, TPD)
    engine._daily_refresh_plants(farm, 0, TPD)
    assert farm["tiles"][4][4] == {"kind": "WEED"}

    farm = engine._new_farm(10, 3000)
    farm["tiles"][4][4] = engine._new_plant("WHEAT", 0, TPD)
    farm["tiles"][4][4]["watered_today"] = True
    engine._daily_refresh_plants(farm, 0, TPD)
    assert farm["tiles"][4][4]["consecutive_unwatered"] == 0
    engine._daily_refresh_plants(farm, 1, TPD)
    assert farm["tiles"][4][4]["consecutive_unwatered"] == 1
    engine._daily_refresh_plants(farm, 2, TPD)
    assert farm["tiles"][4][4] == {"kind": "WEED"}
    return {
        "planting_day_initial_unwatered_count": 1,
        "unwatered_seed_weeds_on_first_night": True,
        "consecutive_unwatered_nights_to_weed_after_watering": 2,
        "fertilizer_duration_days_inclusive": 3,
        "fertilized_water_increment": 2,
        "plain_water_increment": 1,
    }


def animal_constants() -> dict[str, dict[str, object]]:
    expected = {
        "GOOSE": {
            "cost": 300,
            "structure": "COOP",
            "first_yield_day": 4,
            "interval": 1,
            "max_held": 4,
            "product": "EGG",
        },
        "COW": {
            "cost": 400,
            "structure": "PASTURE",
            "first_yield_day": 8,
            "interval": 2,
            "max_held": 6,
            "product": "MILK",
        },
        "SHEEP": {
            "cost": 500,
            "structure": "PASTURE",
            "first_yield_day": 6,
            "interval": 3,
            "max_held": 6,
            "product": "WOOL",
        },
    }
    assert engine.ANIMALS == expected
    return expected


def animal_schedules() -> dict[str, object]:
    expected = {
        "GOOSE": [4, 5, 6, 7],
        "COW": [8, 10, 12, 14],
        "SHEEP": [6, 9, 12, 15],
    }
    result = {}
    for animal, production_days in expected.items():
        farm = engine._new_farm(10, 3000)
        farm["tiles"][4][4] = engine._new_animal(animal, 0)
        produced = []
        previous = 0
        for current_day in range(production_days[-1]):
            tile = farm["tiles"][4][4]
            tile["fed_today"] = True
            engine._daily_refresh_animals(farm, current_day)
            tile = farm["tiles"][4][4]
            if tile["yield_units"] > previous:
                produced.append(current_day + 1)
            previous = tile["yield_units"]
        assert produced == production_days
        result[animal] = {
            "first_four_production_days": produced,
            "units_per_uncared_production": 1,
            "max_held": engine.ANIMALS[animal]["max_held"],
            "ongoing_while_animal_remains": True,
        }
    return result


def animal_care_feed_and_fertilizer() -> dict[str, object]:
    farm = engine._new_farm(10, 3000)
    farm["tiles"][4][4] = engine._new_animal("GOOSE", 0)
    tile = farm["tiles"][4][4]
    tile["fed_today"] = True
    tile["cared_today"] = True
    engine._daily_refresh_animals(farm, 0)
    assert tile["pending_care_bonus"] == 1 and tile["fertilizer_available"]
    for current_day in (1, 2):
        tile["fed_today"] = True
        if current_day == 1:
            tile["cared_today"] = True
        engine._daily_refresh_animals(farm, current_day)
    assert tile["pending_care_bonus"] == 2
    tile["fed_today"] = True
    engine._daily_refresh_animals(farm, 3)
    assert tile["yield_units"] == 3 and tile["pending_care_bonus"] == 0

    unfed_producer = engine._new_farm(10, 3000)
    unfed_producer["tiles"][4][4] = engine._new_animal("GOOSE", 0)
    unfed_producer["tiles"][4][4]["placed_day"] = 0
    # Pretend it was fed through the prior night, then miss its first-yield night.
    unfed_producer["tiles"][4][4]["consecutive_unfed"] = 0
    engine._daily_refresh_animals(unfed_producer, 3)
    assert unfed_producer["tiles"][4][4]["yield_units"] == 1
    assert unfed_producer["tiles"][4][4]["consecutive_unfed"] == 1

    escape = engine._new_farm(10, 3000)
    escape["tiles"][4][4] = engine._new_animal("GOOSE", 0)
    engine._daily_refresh_animals(escape, 0)
    assert escape["tiles"][4][4]["consecutive_unfed"] == 1
    engine._daily_refresh_animals(escape, 1)
    assert escape["tiles"][4][4] == {"kind": "COOP"}

    fertilizer = engine._new_farm(10, 3000)
    fertilizer["tiles"][4][4] = engine._new_animal("COW", 0)
    for day in (0, 1):
        fertilizer["tiles"][4][4]["fed_today"] = True
        engine._daily_refresh_animals(fertilizer, day)
    assert fertilizer["tiles"][4][4]["fertilizer_available"] is True
    return {
        "feed_cost_wheat_per_day": 1,
        "consecutive_unfed_nights_before_escape": 2,
        "structure_remains_after_escape": True,
        "care_actions_banked_in_example": 2,
        "later_fed_goose_production_with_bonus": 3,
        "care_applies_to_same_production": False,
        "base_units_on_first_unfed_production_night": 1,
        "fertilizer_available_per_surviving_night": 1,
        "uncollected_fertilizer_stacks": False,
    }


if __name__ == "__main__":
    emit(
        {
            "engine": assert_exact_engine(),
            "crops": crop_constants(),
            "one_time_crops": one_time_water_windows(),
            "ongoing_crops": ongoing_schedules(),
            "plant_health": watering_and_decay(),
            "animals": animal_constants(),
            "animal_schedules": animal_schedules(),
            "animal_care_feed_fertilizer": animal_care_feed_and_fertilizer(),
        }
    )
