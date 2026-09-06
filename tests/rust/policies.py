"""Action generators that drive both engines through the same tape.

Both read the reference engine's observation for a seat and return the
action list the reference accepts. The biased random policy leans toward
legal, state-changing actions so a long episode reaches the late rules
(lifespans, escapes, shop unlocks); the scripted husbandry policy walks the
animal, fertilizer and hand paths that a random walk rarely strings together.
"""

import random
from typing import Any

from kaggle_environments.envs.kaggriculture import kaggriculture as engine

CROPS = list(engine.CROPS)
ANIMALS = list(engine.ANIMALS)
PRODUCTS = list(engine.PRODUCTS)
SHED_ITEMS = PRODUCTS + ANIMALS
SHED_TILES = {(4, 4), (5, 4), (4, 5), (5, 5)}
MOVES = ["NORTH", "SOUTH", "EAST", "WEST"]

MALFORMED_UNIT: list[Any] = [
    [],
    "WATER",
    ["PLANT"],
    ["PLANT", "GOLD"],
    ["PICKUP"],
    ["PICKUP", "GOLD", 3],
    ["PLACE"],
    ["PLACE", "GOLD"],
    ["PICKUP", "WHEAT", 0],
    ["PICKUP", "WHEAT", -2],
    ["PLACE", "WHEAT", 0],
    ["JUMP"],
    [7],
    None,
]

MALFORMED_ORDER: list[Any] = [
    [],
    ["SELL"],
    ["SELL", "WHEAT"],
    ["SELL", "WHEAT", 0],
    ["SELL", "WHEAT", -1],
    ["SELL", "WHEAT", "three"],
    ["SELL", "WHEAT", None],
    ["SELL", "WHEAT", 2.7],
    ["SELL", "WHEAT", "2"],
    ["SELL", "WHEAT", True],
    ["BUY_SEED", "GOLD", 1],
    ["BUY_PRODUCT", "MELON", 1],
    ["BUY_ANIMAL", "DRAGON", 1],
    ["SELL", "GOOSE", 1],
    ["ROB_BANK"],
    "HIRE",
    None,
]


def _tile(farm: Any, pos: Any) -> Any:  # noqa: ANN401
    x, y = pos
    return farm["tiles"][y][x]


def _random_unit(rng: random.Random, farm: Any, private: Any, pos: Any) -> Any:  # noqa: ANN401, C901
    tile = _tile(farm, pos)
    if rng.random() < 0.02:
        return rng.choice(MALFORMED_UNIT)
    if tuple(pos) in SHED_TILES and rng.random() < 0.45:
        roll = rng.random()
        if roll < 0.4:
            return ["PICKUP", rng.choice(SHED_ITEMS), rng.choice([1, 1, 2, 5, 100])]
        if roll < 0.7:
            return ["DROP"]
        return ["PLACE", rng.choice(SHED_ITEMS), rng.choice([1, 1, 2, 3])]
    if isinstance(tile, dict):
        kind = tile.get("kind")
        roll = rng.random()
        if kind == "PLANT":
            if not tile["watered_today"] and roll < 0.6:
                return ["WATER"]
            if roll < 0.75:
                return ["HARVEST"]
            if roll < 0.85:
                return ["FERTILIZE"]
            if roll < 0.88:
                return ["DIG"]
        elif "animal" in tile:
            return rng.choice(
                [
                    ["FEED"],
                    ["CARE"],
                    ["COLLECT_FERTILIZER"],
                    ["HARVEST"],
                    ["FEED"],
                    ["DIG"],
                ]
            )
        elif kind in ("COOP", "PASTURE"):
            if roll < 0.7:
                return ["PLACE", rng.choice(ANIMALS)]
            if roll < 0.85:
                return ["DIG"]
        elif kind == "WEED" and roll < 0.7:
            return ["DIG"]
    elif tile is None:
        roll = rng.random()
        seeds = [crop for crop, count in private["seeds"].items() if count > 0]
        if seeds and roll < 0.5:
            return ["PLANT", rng.choice(seeds)]
        if roll < 0.55:
            return ["BUILD_COOP"]
        if roll < 0.6:
            return ["BUILD_PASTURE"]
        if roll < 0.63:
            return ["PLANT", rng.choice(CROPS)]
    return [rng.choice([*MOVES, "PASS", "WATER", "HARVEST"])]


def _random_market(rng: random.Random, farm: Any, private: Any) -> list[Any]:  # noqa: ANN401
    orders: list[Any] = []
    shed = private["shed"]
    for _ in range(rng.choice([0, 0, 1, 1, 2, 3])):
        roll = rng.random()
        if roll < 0.25:
            orders.append(["BUY_SEED", rng.choice(CROPS), rng.choice([1, 2, 5, 10])])
        elif roll < 0.38:
            orders.append(["BUY_ANIMAL", rng.choice(ANIMALS), 1])
        elif roll < 0.5:
            orders.append(
                [
                    "BUY_PRODUCT",
                    rng.choice(["WHEAT", "FERTILIZER"]),
                    rng.choice([1, 3, 10]),
                ]
            )
        elif roll < 0.75:
            held = [item for item, count in shed.items() if count > 0]
            if held and rng.random() < 0.9:
                item = rng.choice(held)
            else:
                item = rng.choice([*PRODUCTS, "GOOSE", "GOLD"])
            orders.append(["SELL", item, rng.choice([1, 2, 5, 1000])])
        elif roll < 0.86:
            orders.append(["HIRE"])
        elif roll < 0.93:
            orders.append(["BUY_LAND"])
        else:
            orders.append(rng.choice(MALFORMED_ORDER))
    if rng.random() < 0.01:
        orders.extend([["HIRE"]] * 12)  # past maxMarketOrdersPerTurn
    return orders


def biased_random(rng: random.Random, observation: Any) -> Any:  # noqa: ANN401
    """A random policy that prefers legal, state-changing actions."""
    farm = observation["farms"][observation["player"]]
    private = observation["private"]
    farmer = _random_unit(rng, farm, private, farm["farmer"])
    hands = [_random_unit(rng, farm, private, pos) for pos in farm["hands"]]
    roll = rng.random()
    if roll < 0.03:
        hands.append([rng.choice(MOVES)])  # one more hand than exists
    elif roll < 0.06 and hands:
        hands.pop()  # one fewer
    action = {
        "farmer": farmer,
        "hands": hands,
        "market": _random_market(rng, farm, private),
    }
    if rng.random() < 0.005:
        return rng.choice(
            [None, [], "PASS", {"farmer": ["WATER"], "hands": 3, "market": 4}]
        )
    return action


def _walk(pos: Any, target: tuple[int, int]) -> Any:  # noqa: ANN401
    x, y = pos
    if x < target[0]:
        return ["EAST"]
    if x > target[0]:
        return ["WEST"]
    if y < target[1]:
        return ["SOUTH"]
    if y > target[1]:
        return ["NORTH"]
    return None


def _carry(private: Any, unit: int, item: str) -> int:  # noqa: ANN401
    inventories = private["inventories"]
    if unit >= len(inventories):
        return 0
    return inventories[unit].get(item, 0)


def husbandry(observation: Any) -> Any:  # noqa: ANN401, C901
    """Keep a goose in a coop on the farmer's spawn tile and a hand on a tomato.

    The farmer builds the coop, places the goose, then feeds, cares, collects
    fertilizer and harvests eggs every day. The first hand picks up wheat and
    fertilizer from a locked shed-access tile, walks to (3, 4), and plants,
    waters and fertilizes a tomato. Everything in the shed is sold each
    morning and land is bought when affordable.
    """
    player = observation["player"]
    farm = observation["farms"][player]
    private = observation["private"]
    shed = private["shed"]
    seeds = private["seeds"]
    day = observation["day"]
    hour = observation["hour"]

    market: list[Any] = []
    if hour == 0:
        for item in ("EGG", "TOMATO", "MELON", "WHEAT"):
            if shed.get(item, 0) > (5 if item == "WHEAT" else 0):
                market.append(
                    ["SELL", item, shed[item] - (5 if item == "WHEAT" else 0)]
                )
        if shed.get("GOOSE", 0) == 0 and day == 0:
            market.append(["BUY_ANIMAL", "GOOSE", 1])
        if shed.get("WHEAT", 0) < 5:
            market.append(["BUY_PRODUCT", "WHEAT", 6])
        if shed.get("FERTILIZER", 0) < 2:
            market.append(["BUY_PRODUCT", "FERTILIZER", 2])
        if seeds.get("TOMATO", 0) == 0:
            market.append(["BUY_SEED", "TOMATO", 1])
        if day in (6, 12) and farm["money"] > 2500:
            market.append(["BUY_LAND"])
        market.append(["HIRE"])

    farmer: Any = ["PASS"]
    tile = _tile(farm, farm["farmer"])
    if tuple(farm["farmer"]) != (4, 4):
        farmer = _walk(farm["farmer"], (4, 4))
    elif tile is None:
        farmer = ["BUILD_COOP"]
    elif isinstance(tile, dict) and tile.get("kind") == "COOP" and "animal" not in tile:
        if _carry(private, 0, "GOOSE"):
            farmer = ["PLACE", "GOOSE"]
        elif shed.get("GOOSE", 0):
            farmer = ["PICKUP", "GOOSE", 1]
    elif isinstance(tile, dict) and "animal" in tile:
        if not tile["fed_today"]:
            farmer = ["FEED"] if _carry(private, 0, "WHEAT") else ["PICKUP", "WHEAT", 3]
        elif not tile["cared_today"]:
            farmer = ["CARE"]
        elif tile["fertilizer_available"]:
            farmer = ["COLLECT_FERTILIZER"]
        elif tile["yield_units"] > 0:
            farmer = ["HARVEST"]
        elif _carry(private, 0, "EGG") or _carry(private, 0, "FERTILIZER"):
            farmer = ["DROP"]
    elif isinstance(tile, dict) and tile.get("kind") == "WEED":
        farmer = ["DIG"]

    hands: list[Any] = []
    for index, pos in enumerate(farm["hands"]):
        unit = index + 1
        if index == 0:
            if tuple(pos) in SHED_TILES and not _carry(private, unit, "WHEAT"):
                hands.append(["PICKUP", "WHEAT", 1])
                continue
            if tuple(pos) in SHED_TILES and not _carry(private, unit, "FERTILIZER"):
                hands.append(["PICKUP", "FERTILIZER", 1])
                continue
            step = _walk(pos, (3, 4))
            if step is not None:
                hands.append(step)
                continue
            plot = _tile(farm, pos)
            if plot is None:
                hands.append(
                    ["PLANT", "TOMATO"] if seeds.get("TOMATO", 0) else ["PASS"]
                )
            elif isinstance(plot, dict) and plot.get("kind") == "PLANT":
                if not plot["watered_today"]:
                    hands.append(["WATER"])
                elif plot["yield_units"] > 0:
                    hands.append(["HARVEST"])
                elif plot["fertilized_until_day"] < day and _carry(
                    private, unit, "FERTILIZER"
                ):
                    hands.append(["FERTILIZE"])
                else:
                    hands.append(["PASS"])
            elif isinstance(plot, dict) and plot.get("kind") == "WEED":
                hands.append(["DIG"])
            else:
                hands.append(["PASS"])
        else:
            # Later hands wander the unlocked quadrants, watering what they find.
            plot = _tile(farm, pos)
            if (
                isinstance(plot, dict)
                and plot.get("kind") == "PLANT"
                and not plot["watered_today"]
            ):
                hands.append(["WATER"])
            else:
                hands.append([MOVES[(day + hour + index) % 4]])
    return {"farmer": farmer, "hands": hands, "market": market}
