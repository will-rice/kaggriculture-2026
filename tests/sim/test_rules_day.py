"""Tensor day-phase differential tests."""
# ruff: noqa: ANN001, D103

from kaggle_environments import make

from kaggriculture.sim.day import apply_day_phases
from kaggriculture.sim.state import pack, unpack


def _assert_observations(environment, state) -> None:
    for seat in range(2):
        expected = dict(environment.state[seat].observation)
        expected.pop("remainingOverageTime", None)
        assert unpack(state, 0, seat) == expected


def test_tensor_day_phases_match_town_consumption_and_clock() -> None:
    environment = make("kaggriculture", configuration={"seed": 101}, debug=True)
    environment.reset(2)
    state = pack([environment])
    passes = [{"farmer": ["PASS"], "hands": [], "market": []}] * 2

    environment.step(passes)
    actual = apply_day_phases(state)

    _assert_observations(environment, actual)


def test_tensor_day_phases_match_end_of_day_rng_and_reset() -> None:
    environment = make("kaggriculture", configuration={"seed": 7}, debug=True)
    environment.reset(2)
    passes = [{"farmer": ["PASS"], "hands": [], "market": []}] * 2
    for _ in range(23):
        environment.step(passes)
    observation = environment.state[0].observation
    observation.farms[0].hands = [[4, 4], [5, 4]]
    environment.state[0].observation.private.inventories = [
        {"WHEAT": 2, "MILK": 3},
        {"WOOL": 4},
        {"EGG": 5},
    ]
    state = pack([environment])

    environment.step(passes)
    actual = apply_day_phases(state)

    _assert_observations(environment, actual)


def test_tensor_day_phases_match_production_escape_weeds_and_shop_draw() -> None:
    environment = make("kaggriculture", configuration={"seed": 7}, debug=True)
    environment.reset(2)
    passes = [{"farmer": ["PASS"], "hands": [], "market": []}] * 2
    for _ in range(71):
        environment.step(passes)
    observation = environment.state[0].observation
    observation.farms[0].tiles[0][0] = {
        "kind": "PLANT",
        "crop": "TOMATO",
        "planted_day": -2,
        "watered_today": True,
        "consecutive_unwatered": 1,
        "yield_units": 1,
        "max_lifespan_step": -1,
        "fertilized_until_day": 2,
    }
    observation.farms[0].tiles[0][1] = {
        "kind": "COOP",
        "animal": "GOOSE",
        "placed_day": -2,
        "yield_units": 1,
        "consecutive_unfed": 0,
        "fed_today": True,
        "cared_today": True,
        "fertilizer_available": False,
        "pending_care_bonus": 2,
    }
    observation.farms[1].tiles[0][0] = {
        "kind": "PASTURE",
        "animal": "COW",
        "placed_day": 0,
        "yield_units": 0,
        "consecutive_unfed": 1,
        "fed_today": False,
        "cared_today": False,
        "fertilizer_available": False,
        "pending_care_bonus": 3,
    }
    state = pack([environment])

    environment.step(passes)
    actual = apply_day_phases(state)

    _assert_observations(environment, actual)


def test_plant_decay_skips_the_odd_lifespan_offset() -> None:
    environment = make("kaggriculture", configuration={"seed": 109}, debug=True)
    environment.reset(2)
    passes = [{"farmer": ["PASS"], "hands": [], "market": []}] * 2
    environment.step(passes)
    observation = environment.state[0].observation
    observation.farms[0].tiles[0][0] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 0,
        "watered_today": False,
        "consecutive_unwatered": 0,
        "yield_units": 2,
        "max_lifespan_step": 0,
        "fertilized_until_day": -1,
    }
    state = pack([environment])

    environment.step(passes)
    actual = apply_day_phases(state)

    _assert_observations(environment, actual)


def test_unfed_production_wipes_a_banked_care_bonus() -> None:
    environment = make("kaggriculture", configuration={"seed": 113}, debug=True)
    environment.reset(2)
    passes = [{"farmer": ["PASS"], "hands": [], "market": []}] * 2
    for _ in range(95):
        environment.step(passes)
    observation = environment.state[0].observation
    observation.farms[0].tiles[0][0] = {
        "kind": "COOP",
        "animal": "GOOSE",
        "placed_day": 0,
        "yield_units": 0,
        "consecutive_unfed": 0,
        "fed_today": False,
        "cared_today": False,
        "fertilizer_available": False,
        "pending_care_bonus": 3,
    }
    state = pack([environment])

    environment.step(passes)
    actual = apply_day_phases(state)

    _assert_observations(environment, actual)


def test_shop_draw_can_repeat_an_existing_shop() -> None:
    environment = make("kaggriculture", configuration={"seed": 7}, debug=True)
    environment.reset(2)
    passes = [{"farmer": ["PASS"], "hands": [], "market": []}] * 2
    for _ in range(71):
        environment.step(passes)
    observation = environment.state[0].observation
    observation.town.unlocked_shops = ["BAKERY"]
    state = pack([environment])

    environment.step(passes)
    actual = apply_day_phases(state)

    _assert_observations(environment, actual)


def test_ongoing_crop_uses_the_next_day_for_production() -> None:
    environment = make("kaggriculture", configuration={"seed": 127}, debug=True)
    environment.reset(2)
    passes = [{"farmer": ["PASS"], "hands": [], "market": []}] * 2
    for _ in range(191):
        environment.step(passes)
    observation = environment.state[0].observation
    observation.farms[0].tiles[0][0] = {
        "kind": "PLANT",
        "crop": "TOMATO",
        "planted_day": 0,
        "watered_today": True,
        "consecutive_unwatered": 0,
        "yield_units": 0,
        "max_lifespan_step": -1,
        "fertilized_until_day": -1,
    }
    state = pack([environment])

    environment.step(passes)
    actual = apply_day_phases(state)

    _assert_observations(environment, actual)
