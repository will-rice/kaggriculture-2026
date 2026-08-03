"""Baseline Kaggriculture turn policy.

The farm is worked by a pool of interchangeable units (the main farmer plus the
hands hired that day). Every turn the policy lists the jobs the farm needs done,
sorts them by urgency, and hands each job to the nearest idle unit. Market orders
are planned separately and in money order: sell first so later orders can spend
the proceeds.

Labour is the cheap resource in this game — the n-th hire of a day costs
``fib(n)`` coins while a crop tile returns tens of coins a day — so the policy
scales hands and land with the tile count rather than hoarding cash.
"""

from dataclasses import dataclass
from typing import Any, Mapping

from kaggriculture.actions import PASS, Op, Turn, distance, step_toward
from kaggriculture.constants import (
    CROPS,
    LAND_PRICES,
    LAST_DAY,
    MARKET_PARAMS,
    PRODUCTS,
    hire_cost,
    shed_access_tiles,
    water_bonus_window,
)
from kaggriculture.observation import (
    Observation,
    Position,
    is_harvestable,
    is_plant,
    is_weed,
)

# Job priorities, lowest first.
HAUL_FINAL = 0
WATER = 1
HARVEST = 2
PLANT = 3
DIG = 4
HAUL = 5


@dataclass(frozen=True)
class Strategy:
    """Tunable knobs for the baseline policy."""

    crop: str = "MELON"
    max_hands: int = 8
    tiles_per_unit: int = 4
    hire_before_hour: int = 4
    wage_share: float = 0.1
    cash_reserve: int = 300
    land_reserve: int = 800
    last_land_day: int = 20
    max_sell_per_turn: int = 2
    min_price_ratio: float = 0.6
    haul_threshold: int = 12
    flush_hour: int = 12


STRATEGY = Strategy()


@dataclass(frozen=True)
class Job:
    """A single unit-turn of work at a tile."""

    priority: int
    pos: Position
    op: Op


def agent(raw_obs: Mapping[str, Any]) -> dict[str, Any]:
    """Return this turn's action dict for the environment."""
    return plan(Observation.parse(raw_obs), STRATEGY).to_action()


def plan(obs: Observation, strategy: Strategy) -> Turn:
    """Return the full turn plan for a player."""
    unit_ops = plan_units(obs, strategy)
    return Turn(
        farmer=unit_ops[0],
        hands=unit_ops[1:],
        market=plan_market(obs, strategy),
    )


def plan_units(obs: Observation, strategy: Strategy) -> list[Op]:
    """Assign every unit an op, nearest-unit-first over jobs in priority order."""
    units = obs.units
    assigned: dict[int, Job] = {}
    for job in collect_jobs(obs, strategy):
        idle = [i for i in range(len(units)) if i not in assigned]
        if not idle:
            break
        assigned[min(idle, key=lambda unit: distance(units[unit], job.pos))] = job

    ops: list[Op] = []
    for unit, pos in enumerate(units):
        assignment = assigned.get(unit)
        if assignment is None:
            ops.append(list(PASS))
        elif pos == assignment.pos:
            ops.append(assignment.op)
        else:
            ops.append(step_toward(pos, assignment.pos))
    return ops


def collect_jobs(obs: Observation, strategy: Strategy) -> list[Job]:
    """Return every job the farm wants done this turn, most urgent first."""
    final_day = obs.day >= LAST_DAY
    jobs: list[Job] = []
    for y, row in enumerate(obs.tiles):
        for x, tile in enumerate(row):
            pos = (x, y)
            if is_plant(tile):
                if needs_water(tile, obs.day):
                    jobs.append(Job(WATER, pos, ["WATER"]))
                elif is_harvestable(tile, obs.day) or (final_day and tile["yield_units"] > 0):
                    jobs.append(Job(HARVEST, pos, ["HARVEST"]))
            elif is_weed(tile):
                jobs.append(Job(DIG, pos, ["DIG"]))

    jobs.extend(planting_jobs(obs, strategy))
    jobs.extend(haul_jobs(obs, strategy))
    jobs.sort(key=lambda job: job.priority)
    return jobs


def needs_water(tile: Mapping[str, Any], day: int) -> bool:
    """Return whether a plant must be watered today.

    A plant dies after two consecutive dry days and gains yield for every watered
    day inside its bonus window; watering outside those cases is a wasted
    unit-turn.
    """
    if tile["watered_today"]:
        return False
    if tile["consecutive_unwatered"] >= 1:
        return True
    start, end = water_bonus_window(tile["crop"])
    return start <= day - tile["planted_day"] <= end


def plantable(obs: Observation, strategy: Strategy) -> int:
    """Return how many more tiles are worth sowing right now.

    A crop that misses two days of water becomes a weed, so planting past what
    the crew can walk to every day destroys tiles instead of earning from them.
    The area is therefore capped by the crew a full day can field, not by the
    land owned.
    """
    if obs.day + CROPS[strategy.crop]["first_yield_day"] > LAST_DAY:
        return 0
    serviceable = (1 + strategy.max_hands) * strategy.tiles_per_unit
    return min(len(obs.open_tiles()), serviceable - obs.plant_count())


def planting_jobs(obs: Observation, strategy: Strategy) -> list[Job]:
    """Return plant jobs for open tiles, capped by seeds on hand.

    The environment drops *every* ``PLANT`` op for a crop when a turn requests
    more of a crop than there are seeds, so the seed cap is a correctness
    requirement and not just an optimisation.
    """
    crop = strategy.crop
    wanted = min(plantable(obs, strategy), obs.seeds.get(crop, 0))
    if wanted <= 0:
        return []
    return [Job(PLANT, pos, ["PLANT", crop]) for pos in obs.open_tiles()[:wanted]]


def haul_jobs(obs: Observation, strategy: Strategy) -> list[Job]:
    """Return jobs moving carried produce into the shed, where it can be sold.

    Units drop their load into the shed for free at the end of every day, so
    hauling mid-day only pays once a unit carries a real load — or on the last
    day, whose free drop lands after the final scored turn.
    """
    drops = [pos for pos in shed_access_tiles() if obs.tile(pos) != "LOCKED"]
    final_flush = obs.day >= LAST_DAY and obs.hour >= strategy.flush_hour
    priority = HAUL_FINAL if final_flush else HAUL
    threshold = 1 if final_flush else strategy.haul_threshold

    jobs: list[Job] = []
    for unit, pos in enumerate(obs.units):
        if sum(obs.inventory(unit).values()) < threshold:
            continue
        jobs.append(Job(priority, min(drops, key=lambda drop: distance(pos, drop)), ["DROP"]))
    return jobs


def plan_market(obs: Observation, strategy: Strategy) -> list[Op]:
    """Return this turn's market orders, ordered so sales fund later purchases."""
    orders = sell_orders(obs, strategy)
    orders.extend(hire_orders(obs, strategy))
    orders.extend(land_orders(obs, strategy))
    orders.extend(seed_orders(obs, strategy))
    return orders


def sell_orders(obs: Observation, strategy: Strategy) -> list[Op]:
    """Return sell orders for shed produce.

    Sales are trickled out and held back when a product has been driven well
    under its base price, except on the final day when unsold stock scores
    nothing.
    """
    final_day = obs.day >= LAST_DAY
    orders: list[Op] = []
    for item in PRODUCTS:
        quantity = obs.shed.get(item, 0)
        if quantity <= 0:
            continue
        if final_day:
            orders.append(["SELL", item, quantity])
        elif obs.prices[item] >= int(MARKET_PARAMS[item]["base"]) * strategy.min_price_ratio:
            orders.append(["SELL", item, min(quantity, strategy.max_sell_per_turn)])
    return orders


def hire_orders(obs: Observation, strategy: Strategy) -> list[Op]:
    """Return hire orders bringing the day's crew up to the size the crops need.

    Hands are re-hired every day and priced ``fib(n)``, so the crew is sized to
    the tiles actually growing rather than to the land owned, and the day's wage
    bill is held to a share of the bank — the fibonacci tail can outrun a young
    farm's income before its first harvest lands.
    """
    if obs.hour > strategy.hire_before_hour or obs.day >= LAST_DAY:
        return []
    work = obs.plant_count() + obs.seeds.get(strategy.crop, 0)
    target = min(strategy.max_hands, -(-work // strategy.tiles_per_unit))
    budget = obs.money * strategy.wage_share
    orders: list[Op] = []
    for hires in range(obs.farm["hires_today"], target):
        budget -= hire_cost(hires)
        if budget < 0:
            break
        orders.append(["HIRE"])
    return orders


def land_orders(obs: Observation, strategy: Strategy) -> list[Op]:
    """Return a land order once the crew has outgrown the land it already owns."""
    bought = len(obs.farm["unlocked_quadrants"]) - 1
    if bought >= len(LAND_PRICES) or obs.day > strategy.last_land_day:
        return []
    if obs.unlocked_tile_count() >= (1 + strategy.max_hands) * strategy.tiles_per_unit:
        return []
    if obs.money < LAND_PRICES[bought] + strategy.land_reserve:
        return []
    return [["BUY_LAND"]]


def seed_orders(obs: Observation, strategy: Strategy) -> list[Op]:
    """Return a seed order sized to the tiles the crew can still take on."""
    crop = strategy.crop
    wanted = plantable(obs, strategy) - obs.seeds.get(crop, 0)
    affordable = int((obs.money - strategy.cash_reserve) // CROPS[crop]["seed"])
    quantity = min(wanted, affordable)
    if quantity <= 0:
        return []
    return [["BUY_SEED", crop, quantity]]
