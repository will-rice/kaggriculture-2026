"""Deliberate rare reference states used by differential rule tests."""
# ruff: noqa: D103

from collections.abc import Mapping, Sequence
from typing import Any

from kaggle_environments import make
from kaggle_environments.core import Environment

from kaggriculture.constants import ANIMALS, BOARD_SIZE, CROPS, LAND_ORDER, market_price


def _fresh(seed: int = 251) -> Environment:
    environment = make("kaggriculture", configuration={"seed": seed}, debug=True)
    environment.reset(2)
    return environment


def _sync(environment: Environment) -> Environment:
    public = environment.state[0].observation
    target = environment.state[1].observation
    for name in ("farms", "market", "town", "day", "hour"):
        target[name] = public[name]
    return environment


def shed_at_capacity(*, holding: Mapping[str, int]) -> Environment:
    environment = _fresh()
    shed = environment.state[0].observation.private.shed
    for name, count in holding.items():
        shed[name] = count
    shed["CARROT"] += 100 - sum(shed.values())
    return environment


def shed_one_below_capacity() -> Environment:
    return shed_at_capacity(holding={"CARROT": 99})


def money_exactly(price: int, *, for_order: list[Any]) -> Environment:
    environment = _fresh()
    environment.state[0].observation.farms[0].money = float(price)
    environment.state[0].action = {
        "farmer": ["PASS"],
        "hands": [],
        "market": [for_order],
    }
    return _sync(environment)


def book_at(product: str, *, inventory: int, money: Sequence[int]) -> Environment:
    """Place one product's book at a level and set both seats' coin.

    Args:
        product: Market product whose book level is set.
        inventory: Level the book sits at before the market phase.
        money: Coin for seat zero and seat one.

    Returns:
        An environment with both observations synchronized.
    """
    environment = _fresh()
    public = environment.state[0].observation
    public.market.inventory[product] = inventory
    public.market.prices[product] = market_price(product, inventory)
    for seat, amount in enumerate(money):
        public.farms[seat].money = float(amount)
    return _sync(environment)


def price_at_floor(product: str) -> Environment:
    environment = _fresh()
    inventory = 10_000
    stride = 1
    while market_price(product, inventory) > 1:
        inventory += stride
        stride *= 2
    public = environment.state[0].observation
    public.market.inventory[product] = inventory
    public.market.prices[product] = 1
    return _sync(environment)


def unit_on_locked_shed_tile() -> Environment:
    environment = _fresh()
    environment.state[0].observation.farms[0].farmer = [5, 4]
    return _sync(environment)


def fully_masked_unit() -> Environment:
    return unit_on_locked_shed_tile()


def occupied_structure(animal: str, *, fed: bool, cared: bool) -> Environment:
    environment = _fresh()
    farm = environment.state[0].observation.farms[0]
    farm.farmer = [1, 1]
    farm.tiles[1][1] = {
        "kind": ANIMALS[animal]["structure"],
        "animal": animal,
        "placed_day": 0,
        "yield_units": 1,
        "consecutive_unfed": 0,
        "fed_today": fed,
        "cared_today": cared,
        "fertilizer_available": True,
        "pending_care_bonus": 1,
    }
    return _sync(environment)


def crop_at_age(crop: str, age: int, *, fertilized: bool, watered: bool) -> Environment:
    environment = _fresh()
    public = environment.state[0].observation
    public.day, public.hour, public.step = age, 0, age * 24
    farm = public.farms[0]
    farm.farmer = [1, 1]
    data = CROPS[crop]
    farm.tiles[1][1] = {
        "kind": "PLANT",
        "crop": crop,
        "planted_day": 0,
        "watered_today": watered,
        "consecutive_unwatered": 0 if watered else 1,
        "yield_units": 0 if data["ongoing"] else 1,
        "max_lifespan_step": -1,
        "fertilized_until_day": age if fertilized else -1,
    }
    return _sync(environment)


def hire_ladder(*, hires_today: int, money: int) -> Environment:
    environment = _fresh()
    farm = environment.state[0].observation.farms[0]
    farm.hires_today, farm.money = hires_today, float(money)
    return _sync(environment)


def land_at_stage(stage: int) -> Environment:
    environment = _fresh()
    farm = environment.state[0].observation.farms[0]
    farm.unlocked_quadrants = ["NW", *LAND_ORDER[:stage]]
    half = BOARD_SIZE // 2
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            quadrant = ("N" if y < half else "S") + ("W" if x < half else "E")
            farm.tiles[y][x] = None if quadrant in farm.unlocked_quadrants else "LOCKED"
    return _sync(environment)


def town_with_shops(names: Sequence[str]) -> Environment:
    environment = _fresh()
    environment.state[0].observation.town.unlocked_shops = list(names)
    return _sync(environment)


def last_turn() -> Environment:
    environment = _fresh()
    public = environment.state[0].observation
    public.step, public.day, public.hour = 718, 29, 22
    return _sync(environment)


def multi_item_inventory(order: Sequence[str]) -> Environment:
    environment = _fresh()
    environment.state[0].observation.private.inventories = [
        {name: index + 1 for index, name in enumerate(order)}
    ]
    return environment
