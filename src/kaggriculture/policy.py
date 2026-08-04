"""Baseline Kaggriculture turn policy.

The farm is worked by a pool of interchangeable units (the main farmer plus the
hands hired that day). Every turn the policy lists the jobs the farm needs done,
sorts them by urgency, and hands each job to the nearest idle unit. Market orders
are planned separately, sell first, and each planner is handed the balance the
planners before it left rather than the turn's opening balance.

Labour is the cheap resource in this game — the n-th hire of a day costs
``fib(n)`` coins while a crop tile returns tens of coins a day — so the policy
scales hands and land with the tile count rather than hoarding cash.
"""

import collections
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from kaggriculture.actions import PASS, Op, Turn, distance, step_toward
from kaggriculture.constants import (
    ANIMALS,
    CROPS,
    LAND_PRICES,
    LAST_DAY,
    MAX_MARKET_ORDERS_PER_TURN,
    PRODUCTS,
    TURNS_PER_DAY,
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

# A planner's market orders paired with what they cost, so the next planner can
# be told what is left to spend.
Order = tuple[list[Op], float]

# Job priorities, lowest first. Housing and placing an animal come before
# watering: they are one-off turns that put a bought asset to work, where an
# unwatered plant has a day of slack before it takes harm.
HAUL_FINAL = 0
FEED = 1
STOCK = 2
BUILD = 3
WATER = 4
HARVEST = 5
CARE = 6
PLANT = 7
COLLECT = 8
DIG = 9
HAUL = 10


@dataclass(frozen=True)
class Strategy:
    """Tunable knobs for the baseline policy."""

    crop: str = "STRAWBERRY"
    herd: Mapping[str, int] = field(default_factory=lambda: {"COW": 8, "SHEEP": 2})
    max_hands: int = 12
    tiles_per_unit: int = 4
    hire_before_hour: int = 4
    wage_share: float = 0.1
    cash_reserve: int = 300
    land_reserve: int = 800
    last_land_day: int = 20
    animals_per_turn: int = 2
    working_capital: int = 400
    feed_days: int = 3
    sell_rate: int = 2
    buy_slots: int = 4
    haul_threshold: int = 12
    flush_hour: int = 12


STRATEGY = Strategy()


@dataclass(frozen=True)
class Job:
    """A single unit-turn of work at a tile.

    ``unit`` binds the job to one unit, used when the job depends on what that
    unit is already carrying — only the farmer holding a cow can place it.
    """

    priority: int
    pos: Position
    op: Op
    unit: Optional[int] = None


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
        if job.unit is not None:
            if job.unit not in assigned:
                assigned[job.unit] = job
            continue
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
                elif is_harvestable(tile, obs.day) or (
                    final_day and tile["yield_units"] > 0
                ):
                    jobs.append(Job(HARVEST, pos, ["HARVEST"]))
            elif is_weed(tile):
                jobs.append(Job(DIG, pos, ["DIG"]))

    jobs.extend(animal_jobs(obs, strategy))
    jobs.extend(carried_jobs(obs))
    jobs.extend(fetch_jobs(obs))
    jobs.extend(planting_jobs(obs, strategy))
    jobs.extend(haul_jobs(obs, strategy))
    jobs.sort(key=lambda job: job.priority)
    return jobs


def pasture_sites(obs: Observation, strategy: Strategy) -> list[Position]:
    """Return the tiles the herd holds or is about to hold, nearest the shed first.

    Animals are fed from the shed every single day, so keeping the herd close to
    it is worth more than the crop yield of those tiles. Only the animals we own
    plus a day's purchases are sited: reserving the whole target herd up front
    sterilised fourteen tiles from day zero for animals that arrived over the
    following three weeks, if at all.
    """
    built = [
        (x, y)
        for y, row in enumerate(obs.tiles)
        for x, tile in enumerate(row)
        if isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE")
    ]
    wanted = min(
        sum(owned_animals(obs).values()) + strategy.animals_per_turn,
        sum(strategy.herd.values()),
    )
    if len(built) >= wanted:
        return built
    home = shed_access_tiles()[0]
    free = sorted(obs.open_tiles(), key=lambda pos: distance(pos, home))
    return built + free[: wanted - len(built)]


def structures(obs: Observation, kind: str, occupied: bool) -> list[Position]:
    """Return coop or pasture tiles, either holding an animal or waiting for one."""
    return [
        (x, y)
        for y, row in enumerate(obs.tiles)
        for x, tile in enumerate(row)
        if isinstance(tile, dict)
        and tile.get("kind") == kind
        and (tile.get("animal") is not None) == occupied
    ]


def livestock(obs: Observation) -> list[tuple[Position, Mapping[str, Any]]]:
    """Return every placed animal with its tile position."""
    return [
        ((x, y), tile)
        for y, row in enumerate(obs.tiles)
        for x, tile in enumerate(row)
        if isinstance(tile, dict) and tile.get("animal")
    ]


def owned_animals(obs: Observation) -> collections.Counter[str]:
    """Return how many of each animal we own, placed, shed-bound or in hand.

    A bought animal lands in the shed and only earns once a unit has carried it
    to a structure, so every count that decides spending or building has to see
    all three places at once.
    """
    owned = collections.Counter(tile["animal"] for _, tile in livestock(obs))
    for animal in ANIMALS:
        carried = sum(
            obs.inventory(unit).get(animal, 0) for unit in range(len(obs.units))
        )
        owned[animal] += obs.shed.get(animal, 0) + carried
    return owned


def homeless_animals(obs: Observation) -> collections.Counter[str]:
    """Return how many owned animals of each structure kind have nowhere to live."""
    homeless: collections.Counter[str] = collections.Counter()
    for animal, count in owned_animals(obs).items():
        kind = str(ANIMALS[animal]["structure"])
        homeless[kind] += count
    for kind in list(homeless):
        homeless[kind] -= len(structures(obs, kind, occupied=True))
        homeless[kind] -= len(structures(obs, kind, occupied=False))
    return homeless


def animal_jobs(obs: Observation, strategy: Strategy) -> list[Job]:
    """Return the daily work a placed herd needs, plus the pastures to build.

    An animal that misses two consecutive feeds escapes and is gone for good, so
    feeding outranks everything except the final-day liquidation.
    """
    # Feeding and care only buy a tomorrow, and on the last day there is none:
    # the animal's next production lands after the final scored turn, so the
    # wheat is worth more sold than eaten.
    keeping = obs.day < LAST_DAY
    jobs: list[Job] = []
    for pos, tile in livestock(obs):
        if tile["yield_units"] > 0:
            jobs.append(Job(HARVEST, pos, ["HARVEST"]))
        if keeping and not tile["cared_today"]:
            jobs.append(Job(CARE, pos, ["CARE"]))
        if tile["fertilizer_available"]:
            jobs.append(Job(COLLECT, pos, ["COLLECT_FERTILIZER"]))

    # A structure is built only once an animal is waiting for it. Building the
    # target herd's housing up front costs no coins but takes the tiles out of
    # cultivation for the whole season.
    sites = [pos for pos in pasture_sites(obs, strategy) if obs.tile(pos) is None]
    for kind, homeless in homeless_animals(obs).items():
        wanted = min(max(homeless, 0), len(sites))
        jobs.extend(Job(BUILD, pos, [f"BUILD_{kind}"]) for pos in sites[:wanted])
        sites = sites[wanted:]
    return jobs


def unfed_animals(obs: Observation) -> list[Position]:
    """Return the animals still owed feed today, none of them on the last day.

    An animal is fed to keep it alive for tomorrow, and the last day has no
    tomorrow — feeding it there burns a unit-turn and buries wheat that is about
    to be sold in an animal's stomach.
    """
    if obs.day >= LAST_DAY:
        return []
    return [pos for pos, tile in livestock(obs) if not tile["fed_today"]]


def carried_jobs(obs: Observation) -> list[Job]:
    """Return jobs for units already carrying something that must be delivered."""
    jobs: list[Job] = []
    unfed = unfed_animals(obs)
    for unit, pos in enumerate(obs.units):
        inventory = obs.inventory(unit)
        animal = next((name for name in ANIMALS if inventory.get(name, 0) > 0), None)
        if animal is not None:
            homes = structures(obs, str(ANIMALS[animal]["structure"]), occupied=False)
            if homes:
                target = min(homes, key=lambda home: distance(pos, home))
                jobs.append(Job(STOCK, target, ["PLACE", animal], unit))
            continue
        if inventory.get("WHEAT", 0) > 0 and unfed:
            target = min(unfed, key=lambda animal_pos: distance(pos, animal_pos))
            jobs.append(Job(FEED, target, ["FEED"], unit))
    return jobs


def fetch_jobs(obs: Observation) -> list[Job]:
    """Return trips to the shed for animals waiting to be placed and for feed."""
    drops = [pos for pos in shed_access_tiles() if obs.tile(pos) != "LOCKED"]
    if not drops:
        return []
    drop = drops[0]

    jobs: list[Job] = []
    for animal in ANIMALS:
        homes = len(structures(obs, str(ANIMALS[animal]["structure"]), occupied=False))
        carried = sum(
            obs.inventory(unit).get(animal, 0) for unit in range(len(obs.units))
        )
        for _ in range(min(obs.shed.get(animal, 0), homes - carried)):
            jobs.append(Job(STOCK, drop, ["PICKUP", animal, 1]))

    # The day's feed goes to one carrier, sized to the animals still unfed rather
    # than to the herd. Splitting it across a carrier per few animals was measured
    # worse over sixteen seeds: pastures are sited together beside the shed, so
    # one loaded unit walks them cheaply, while every extra carrier is a unit
    # taken off watering for a whole round trip.
    carried_wheat = sum(
        obs.inventory(unit).get("WHEAT", 0) for unit in range(len(obs.units))
    )
    shortfall = min(len(unfed_animals(obs)) - carried_wheat, obs.shed.get("WHEAT", 0))
    if shortfall > 0:
        jobs.append(Job(FEED, drop, ["PICKUP", "WHEAT", shortfall]))
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


def crop_capacity(obs: Observation, strategy: Strategy) -> int:
    """Return how many crop tiles a full day's crew can still keep watered.

    The herd draws on the same crew as the crops — feeding, care and collection
    are unit-turns like watering — so every animal owned is charged against the
    day's capacity. Ignoring that is how a farm with fourteen animals ends the
    season holding sixteen weeds and two sheep it never had a spare hand to
    place.
    """
    crew = (1 + strategy.max_hands) * strategy.tiles_per_unit
    return crew - sum(owned_animals(obs).values())


def plantable(obs: Observation, strategy: Strategy) -> int:
    """Return how many more tiles are worth sowing right now.

    A crop that misses two days of water becomes a weed, so planting past what
    the crew can walk to every day destroys tiles instead of earning from them.
    The area is therefore capped by the crew a full day can field, not by the
    land owned.
    """
    if obs.day + CROPS[strategy.crop]["first_yield_day"] > LAST_DAY:
        return 0
    return min(len(obs.open_tiles()), crop_capacity(obs, strategy) - obs.plant_count())


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
    reserved = set(pasture_sites(obs, strategy))
    free = [pos for pos in obs.open_tiles() if pos not in reserved]
    return [Job(PLANT, pos, ["PLANT", crop]) for pos in free[:wanted]]


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
    keeping = obs.day < LAST_DAY
    for unit, pos in enumerate(obs.units):
        inventory = obs.inventory(unit)
        carrying_stock = inventory.get("WHEAT", 0) or any(
            inventory.get(name, 0) for name in ANIMALS
        )
        if carrying_stock and keeping:
            continue  # feed and livestock are in hand for a reason, not produce to bank
        if sum(inventory.values()) < threshold:
            continue
        jobs.append(
            Job(priority, min(drops, key=lambda drop: distance(pos, drop)), ["DROP"])
        )
    return jobs


def plan_market(obs: Observation, strategy: Strategy) -> list[Op]:
    """Return this turn's market orders, spent in the order the farm needs them.

    The environment charges each order against the balance left by the ones
    before it, but every planner below sees only the turn's opening balance. So
    the balance is threaded through them explicitly and each returns what it
    spent: a planner is told what is left, never what the turn started with.
    Budgeting independently is how the day-zero turn came to queue two cows, a
    sheep, a land purchase and 25 melon seeds that each fitted inside $3000 and
    together did not — the farm woke up on day one with $58, could not feed the
    cows it had just bought, and lost them.

    Sale proceeds are deliberately not counted, though sales are queued first:
    the price a sale realises walks as it fills, so treating it as spendable is
    how a budget goes negative.
    """
    money = obs.money
    orders = sell_orders(obs, strategy)
    # Feed first: an unfed animal is a bought asset walking off the farm. Then
    # labour and seed, which are what the crops already in the ground need to
    # pay for the herd, and only then the herd itself and more land.
    for planner in (
        feed_orders,
        hire_orders,
        seed_orders,
        animal_orders,
        land_orders,
    ):
        planned, spent = planner(obs, strategy, money)
        orders.extend(planned)
        money -= spent
    return orders


def feed_orders(obs: Observation, strategy: Strategy, money: float) -> Order:
    """Return wheat purchases keeping the shed stocked with feed.

    Buying feed is cheaper than growing it: a cow eats one wheat a day and
    returns milk worth several times the wheat price. Stock runs ahead of the
    herd by one day's purchases, because an animal is only bought once the feed
    it will eat is already in the shed.
    """
    if obs.day >= LAST_DAY:
        return [], 0
    mouths = min(
        sum(owned_animals(obs).values()) + strategy.animals_per_turn,
        sum(strategy.herd.values()),
    )
    price = max(1, obs.prices["WHEAT"])
    wanted = mouths * strategy.feed_days - obs.shed.get("WHEAT", 0)
    quantity = min(wanted, int((money - strategy.cash_reserve) // price))
    if quantity <= 0:
        return [], 0
    return [["BUY_PRODUCT", "WHEAT", quantity]], quantity * price


def animal_orders(obs: Observation, strategy: Strategy, money: float) -> Order:
    """Return livestock purchases, gated on feed in the shed and season left.

    An animal that misses two consecutive feeds escapes and takes its whole
    purchase price with it, so one is bought only when the shed already holds
    the feed to keep it alive — an order that cannot be honoured until the wheat
    lands is a turn's delay, where a starved cow is 400 coins gone.

    Buying also stops once the season is too short for the animal to yield
    twice, which is roughly what it takes to repay its cost.
    """
    owned = owned_animals(obs)
    if obs.shed.get("WHEAT", 0) < (sum(owned.values()) + 1) * strategy.feed_days:
        return [], 0

    orders: list[Op] = []
    spent = 0
    for animal, count in strategy.herd.items():
        data = ANIMALS[animal]
        payback_day = obs.day + int(data["first_yield_day"]) + int(data["interval"])
        if payback_day > LAST_DAY:
            continue
        cost = int(data["cost"])
        for _ in range(count - owned[animal]):
            if len(orders) >= strategy.animals_per_turn:
                return orders, spent
            if money - spent - cost < strategy.working_capital:
                return orders, spent
            spent += cost
            orders.append(["BUY_ANIMAL", animal, 1])
    return orders, spent


def turns_remaining(obs: Observation) -> int:
    """Return how many turns are left in the season, this one included."""
    return (LAST_DAY - obs.day) * TURNS_PER_DAY + (TURNS_PER_DAY - obs.hour)


def sell_orders(obs: Observation, strategy: Strategy) -> list[Op]:
    """Return sell orders that clear the shed by season's end, no faster.

    Every unit sold pushes its own price down, and the town shops drain
    inventory between turns, so a trickle realises close to the peak price on
    every unit while a dump realises the average of a falling curve. Measured:
    selling everything the market would absorb at a 10% impact cap banked 41.2k
    where a slow trickle banked 49.1k on the same seeds.

    So the rate is the smallest that still empties the shed in time —
    ``held / turns_remaining`` — with a floor so a small stock still moves. That
    single expression also replaces the old final-day dump: as the last turn
    approaches the divisor falls to one and the rate rises to the whole holding,
    which liquidates smoothly over the closing turns instead of cratering the
    price in a single one.

    What is deliberately gone is the old gate that refused to sell below a
    fraction of the *base* price. Base price says nothing here — the shops keep
    goods scarce, so strawberry trades near $207 against a base of $120 — and
    refusing to sell when a price was depressed simply hoarded stock into the
    worst possible moment.

    Orders are ranked by value and capped so purchases still fit in the turn's
    ten market slots.
    """
    final_day = obs.day >= LAST_DAY
    feed_reserve = len(livestock(obs)) * strategy.feed_days
    left = turns_remaining(obs)
    candidates: list[tuple[float, str, int]] = []
    for item in PRODUCTS:
        held = obs.shed.get(item, 0)
        if item == "WHEAT" and not final_day:
            held -= feed_reserve  # the herd eats before the market does
        if held <= 0:
            continue
        quantity = min(held, max(strategy.sell_rate, -(-held // left)))
        candidates.append((quantity * obs.prices[item], item, quantity))

    candidates.sort(key=lambda candidate: -candidate[0])
    slots = MAX_MARKET_ORDERS_PER_TURN - (0 if final_day else strategy.buy_slots)
    return [["SELL", item, quantity] for _, item, quantity in candidates[:slots]]


def hire_orders(obs: Observation, strategy: Strategy, money: float) -> Order:
    """Return hire orders bringing the day's crew up to the size the crops need.

    Hands are re-hired every day and priced ``fib(n)``, so the crew is sized to
    the tiles actually growing rather than to the land owned, and the day's wage
    bill is held to a share of the bank — the fibonacci tail can outrun a young
    farm's income before its first harvest lands.
    """
    if obs.hour > strategy.hire_before_hour or obs.day >= LAST_DAY:
        return [], 0
    work = (
        obs.plant_count()
        + obs.seeds.get(strategy.crop, 0)
        + sum(owned_animals(obs).values())
    )
    target = min(strategy.max_hands, -(-work // strategy.tiles_per_unit))
    budget = money * strategy.wage_share
    orders: list[Op] = []
    spent = 0
    for hires in range(obs.farm["hires_today"], target):
        cost = hire_cost(hires)
        if spent + cost > budget:
            break
        spent += cost
        orders.append(["HIRE"])
    return orders, spent


def land_orders(obs: Observation, strategy: Strategy, money: float) -> Order:
    """Return a land order once the crew has outgrown the land it already owns."""
    bought = len(obs.farm["unlocked_quadrants"]) - 1
    if bought >= len(LAND_PRICES) or obs.day > strategy.last_land_day:
        return [], 0
    if obs.unlocked_tile_count() >= crop_capacity(obs, strategy):
        return [], 0
    if money < LAND_PRICES[bought] + strategy.land_reserve:
        return [], 0
    return [["BUY_LAND"]], LAND_PRICES[bought]


def seed_orders(obs: Observation, strategy: Strategy, money: float) -> Order:
    """Return a seed order sized to the tiles the crew can still take on."""
    crop = strategy.crop
    price = int(CROPS[crop]["seed"])
    wanted = plantable(obs, strategy) - obs.seeds.get(crop, 0)
    quantity = min(wanted, int((money - strategy.cash_reserve) // price))
    if quantity <= 0:
        return [], 0
    return [["BUY_SEED", crop, quantity]], quantity * price
