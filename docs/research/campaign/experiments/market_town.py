"""Verify market curves, transaction mechanics, and shared town demand."""

from __future__ import annotations

from kaggle_environments.envs.kaggriculture import kaggriculture as engine

from _common import PASS, assert_exact_engine, emit, new_env


EXPECTED_PARAMS = {
    "WHEAT": {"base": 25, "I0": 10000, "T": 400, "below_func": "sqrt", "below_target": 0.80, "above_func": "log", "above_target": 0.20},
    "CARROT": {"base": 35, "I0": 10000, "T": 450, "below_func": "hinge", "below_target": 1.00, "above_func": "sqrt", "above_target": 0.70},
    "TOMATO": {"base": 60, "I0": 10000, "T": 200, "below_func": "hinge", "below_target": 0.40, "above_func": "sqrt", "above_target": 0.60},
    "STRAWBERRY": {"base": 120, "I0": 10000, "T": 100, "below_func": "sqrt", "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON": {"base": 250, "I0": 10000, "T": 300, "below_func": "log", "below_target": 0.20, "above_func": "sq", "above_target": 3.60},
    "EGG": {"base": 50, "I0": 10000, "T": 332, "below_func": "hinge", "below_target": 0.40, "above_func": "log", "above_target": 0.20},
    "MILK": {"base": 160, "I0": 10000, "T": 122, "below_func": "sqrt", "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL": {"base": 200, "I0": 10000, "T": 105, "below_func": "log", "below_target": 0.20, "above_func": "sq", "above_target": 3.20},
    "FERTILIZER": {"base": 100, "I0": 10000, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}


def market_parameters_and_curves() -> dict[str, object]:
    assert engine.MARKET_PARAMS == EXPECTED_PARAMS
    assert engine.MARKET_I0 == 10000
    assert engine.PRICE_FLOOR == 1
    assert engine.HINGE_GAIN == 8.0
    samples = {}
    for item, params in EXPECTED_PARAMS.items():
        samples[item] = {
            "at_I0_minus_T": engine.market_price(item, params["I0"] - params["T"]),
            "at_I0": engine.market_price(item, params["I0"]),
            "at_I0_plus_T": engine.market_price(item, params["I0"] + params["T"]),
        }
    assert samples == {
        "WHEAT": {"at_I0_minus_T": 45, "at_I0": 25, "at_I0_plus_T": 20},
        "CARROT": {"at_I0_minus_T": 70, "at_I0": 35, "at_I0_plus_T": 10},
        "TOMATO": {"at_I0_minus_T": 84, "at_I0": 60, "at_I0_plus_T": 24},
        "STRAWBERRY": {"at_I0_minus_T": 204, "at_I0": 120, "at_I0_plus_T": 1},
        "MELON": {"at_I0_minus_T": 300, "at_I0": 250, "at_I0_plus_T": 1},
        "EGG": {"at_I0_minus_T": 70, "at_I0": 50, "at_I0_plus_T": 40},
        "MILK": {"at_I0_minus_T": 256, "at_I0": 160, "at_I0_plus_T": 1},
        "WOOL": {"at_I0_minus_T": 240, "at_I0": 200, "at_I0_plus_T": 1},
        "FERTILIZER": {"at_I0_minus_T": 140, "at_I0": 100, "at_I0_plus_T": 60},
    }
    hinge = {}
    for item in ("CARROT", "TOMATO", "EGG"):
        params = EXPECTED_PARAMS[item]
        hinge[item] = {
            "price_at_half_T_scarcity": engine.market_price(item, params["I0"] - params["T"] / 2),
            "price_at_double_T_scarcity": engine.market_price(item, params["I0"] - 2 * params["T"]),
        }
    assert hinge == {
        "CARROT": {"price_at_half_T_scarcity": 52, "price_at_double_T_scarcity": 385},
        "TOMATO": {"price_at_half_T_scarcity": 72, "price_at_double_T_scarcity": 300},
        "EGG": {"price_at_half_T_scarcity": 60, "price_at_double_T_scarcity": 250},
    }
    return {
        "parameters": EXPECTED_PARAMS,
        "price_floor": 1,
        "hinge_gain": 8.0,
        "shape_names": ["linear", "sq", "sqrt", "log", "log10", "hinge"],
        "samples": samples,
        "hinge_samples": hinge,
    }


def simultaneous_and_sequential_sales() -> dict[str, object]:
    env = new_env(seed=1, weedSpawnChance=0)
    env.reset(2)
    env.step([PASS, PASS])
    market = env.state[0].observation.market
    market.inventory.CARROT = 9550
    engine._refresh_prices(market)
    for state in env.state:
        state.observation.private.shed.CARROT = 1
    before = [state.observation.farms[i].money for i, state in enumerate(env.state)]
    sell_one = {"farmer": ["PASS"], "hands": [], "market": [["SELL", "CARROT", 1]]}
    env.step([sell_one, sell_one])
    after = [state.observation.farms[i].money for i, state in enumerate(env.state)]
    assert before == [3000.0, 3000.0]
    assert after == [3070.0, 3070.0]
    assert env.state[0].observation.market.inventory.CARROT == 9552

    env = new_env(seed=1, weedSpawnChance=0)
    env.reset(2)
    env.step([PASS, PASS])
    market = env.state[0].observation.market
    market.inventory.CARROT = 9100
    engine._refresh_prices(market)
    env.state[0].observation.private.shed.CARROT = 3
    quotes = [engine.market_price("CARROT", 9100 + i) for i in range(3)]
    assert quotes == [385, 384, 382]
    env.step([
        {"farmer": ["PASS"], "hands": [], "market": [["SELL", "CARROT", 3]]},
        PASS,
    ])
    assert env.state[0].observation.farms[0].money == 3000 + sum(quotes)
    assert env.state[0].observation.market.inventory.CARROT == 9103
    return {
        "simultaneous_start_inventory": 9550,
        "simultaneous_sellers": 2,
        "same_quote_each": 70,
        "inventory_after": 9552,
        "sequential_start_inventory": 9100,
        "three_sequential_quotes": quotes,
        "sequential_inventory_after": 9103,
    }


def sale_floor_and_purchase_rules() -> dict[str, object]:
    env = new_env(seed=1, weedSpawnChance=0)
    env.reset(2)
    env.step([PASS, PASS])
    market = env.state[0].observation.market
    market.inventory.MELON = 10300
    engine._refresh_prices(market)
    assert market["prices"]["MELON"] == 1
    env.state[0].observation.private.shed.MELON = 1
    env.step([
        {"farmer": ["PASS"], "hands": [], "market": [["SELL", "MELON", 1]]},
        PASS,
    ])
    assert env.state[0].observation.farms[0].money == 3001
    assert env.state[0].observation.market.inventory.MELON == 10300

    env = new_env(seed=1, weedSpawnChance=0)
    env.reset(2)
    env.step([PASS, PASS])
    market = env.state[0].observation.market
    market.inventory.WHEAT = 10000
    engine._refresh_prices(market)
    env.step([
        {"farmer": ["PASS"], "hands": [], "market": [
            ["BUY_PRODUCT", "WHEAT", 1], ["SELL", "WHEAT", 1]
        ]},
        PASS,
    ])
    assert env.state[0].observation.farms[0].money == 3000
    assert env.state[0].observation.market.inventory.WHEAT == 10000
    assert env.state[0].observation.private.shed.WHEAT == 0
    before = env.state[0].observation.farms[0].money
    env.step([
        {"farmer": ["PASS"], "hands": [], "market": [["BUY_PRODUCT", "CARROT", 1]]},
        PASS,
    ])
    assert env.state[0].observation.farms[0].money == before

    env.state[0].observation.private.inventories[0]["WOOL"] = 2
    before = env.state[0].observation.farms[0].money
    env.step([
        {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WOOL", 2]]},
        PASS,
    ])
    assert env.state[0].observation.farms[0].money == before
    assert env.state[0].observation.private.inventories[0].WOOL == 2
    return {
        "floor_price": 1,
        "floor_sale_units": 1,
        "floor_sale_inventory_increase": 0,
        "buyable_products": ["WHEAT", "FERTILIZER"],
        "round_trip_units": 1,
        "unchanged_market_round_trip_net_cost": 0,
        "SELL_source": "shed only",
        "carried_wool_not_sold": 2,
        "seed_prices": {crop: data["seed"] for crop, data in engine.CROPS.items()},
        "animal_prices": {animal: data["cost"] for animal, data in engine.ANIMALS.items()},
    }


def order_limit_and_catalogue() -> dict[str, object]:
    env = new_env(seed=1, weedSpawnChance=0)
    env.reset(2)
    eleven_hires = [["HIRE"] for _ in range(11)]
    env.step([
        {"farmer": ["PASS"], "hands": [], "market": eleven_hires},
        PASS,
    ])
    farm = env.state[0].observation.farms[0]
    assert len(farm.hands) == 10
    assert farm.money == 2857
    return {
        "max_orders_per_player_turn": 10,
        "submitted_orders": 11,
        "processed_orders": 10,
        "market_verbs": ["BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND"],
        "quantified_verbs": ["BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"],
        "atomic_verbs": ["HIRE", "BUY_LAND"],
    }


def town_center() -> dict[str, object]:
    env = new_env(seed=1, weedSpawnChance=0, townShopUnlockInterval=1000)
    env.run([lambda o, c: PASS, lambda o, c: PASS])
    inventory = dict(env.state[0].observation.market.inventory)
    for item in engine.TOWN_CENTER_PRODUCTS:
        assert inventory[item] == 9970
    assert inventory["FERTILIZER"] == 10000
    return {
        "consumption_interval_action_steps": 24,
        "first_consumption_action_step": 0,
        "ticks_in_episode": 30,
        "units_per_non_fertilizer_product_per_tick": 1,
        "non_fertilizer_products": list(engine.TOWN_CENTER_PRODUCTS),
        "ending_inventory_without_sales_or_shops": inventory,
    }


def shops_and_duplicates() -> dict[str, object]:
    expected_shops = {
        "BAKERY": ["EGG", "WHEAT"],
        "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
        "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"],
        "YARN_STORE": ["WOOL"],
        "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"],
        "PET_CAFE": ["CARROT"],
        "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"],
        "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
    }
    assert engine.SHOPS == expected_shops
    env = new_env(seed=1, weedSpawnChance=0, townShopUnlockInterval=1000)
    env.reset(2)
    env.step([PASS, PASS])
    env.state[0].observation.town.unlocked_shops = ["YARN_STORE", "BAKERY", "YARN_STORE"]
    for _ in range(3):
        env.step([PASS, PASS])
    before = dict(env.state[0].observation.market.inventory)
    env.step([PASS, PASS])
    after = dict(env.state[0].observation.market.inventory)
    assert before["WOOL"] - after["WOOL"] == 4
    assert before["EGG"] - after["EGG"] == 1
    assert before["WHEAT"] - after["WHEAT"] == 1
    return {
        "catalogue": expected_shops,
        "consumption_interval_action_steps": 4,
        "multi_product_units_each": 1,
        "single_product_units_each": 2,
        "duplicate_yarn_instances_in_probe": 2,
        "wool_consumed_in_tick": 4,
        "bakery_egg_and_wheat_each": 1,
    }


def unlock_sequence(seed: int) -> dict[str, object]:
    env = new_env(seed=seed, weedSpawnChance=0)
    env.reset(2)
    seen = 0
    events = []
    while env.state[0].status == "ACTIVE" and len(events) < 8:
        env.step([PASS, PASS])
        shops = list(env.state[0].observation.town.unlocked_shops)
        if len(shops) > seen:
            obs = env.state[0].observation
            events.append({"day": obs.day, "recorded_step": obs.step, "shop": shops[-1]})
            seen = len(shops)
    return {
        "input_seed": seed,
        "resolved_seed_in_env_info": env.info["seed"],
        "configuration_seed_after_reset": env.configuration.seed,
        "resolved_seed_visible_to_agent": False,
        "events": events,
    }


def shop_unlocks_and_seed() -> dict[str, object]:
    seed1a = unlock_sequence(1)
    seed1b = unlock_sequence(1)
    seed2 = unlock_sequence(2)
    assert seed1a == seed1b
    assert seed1a["events"] != seed2["events"]
    expected_days = [3, 6, 9, 12, 15, 18, 21, 24]
    assert [event["day"] for event in seed1a["events"]] == expected_days
    assert [event["recorded_step"] for event in seed1a["events"]] == [72, 144, 216, 288, 360, 432, 504, 576]
    assert seed1a["configuration_seed_after_reset"] is None
    return {
        "unlock_every_days": 3,
        "maximum_instances": 8,
        "unlock_days": expected_days,
        "seed_1": seed1a,
        "seed_2": seed2,
        "same_seed_repeats_sequence": True,
        "draws_with_replacement": True,
        "rng_key_formula": "(resolved_seed * 1000003) XOR current_day",
    }


if __name__ == "__main__":
    emit({
        "engine": assert_exact_engine(),
        "market": market_parameters_and_curves(),
        "sales": simultaneous_and_sequential_sales(),
        "purchase_and_floor_rules": sale_floor_and_purchase_rules(),
        "orders": order_limit_and_catalogue(),
        "town_center": town_center(),
        "shops": shops_and_duplicates(),
        "shop_unlocks": shop_unlocks_and_seed(),
    })
