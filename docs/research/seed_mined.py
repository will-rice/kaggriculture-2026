"""A seed built from the public corpus rather than from a harvested kernel.

Every champion this campaign has produced descends from `thomastschinkel_router`,
one public agent, sixty-nine generations deep. That is the monoculture at its
root, and no instruction escapes it, because a round is handed the champion and
asked to improve it.

This starts somewhere else: from what 17,617 public games say the ladder's
strongest opening does. Those three agents share a signature the rest of the top
twelve do not -- second quadrant on day three, third on day eight -- and they
sit 0.9 log-odds clear of fourth place. Mined as orders rather than as averaged
holdings, because an average over a group asked for 1.2 quadrants and nobody can
buy a fifth of one:

    day 0   7 x HIRE, 3 x BUY_PRODUCT WHEAT x2, BUY_ANIMAL COW x5
    day 1   4 x HIRE, 4 x SELL FERTILIZER x2
    day 2   7 x HIRE, 3 x SELL FERTILIZER x2, BUY_PRODUCT WHEAT x6
    day 3   7 x HIRE, 3 x SELL FERTILIZER x2, 1 x BUY_LAND
    day 4   7 x HIRE, 3 x SELL WHEAT x6, 2 x BUY_SEED MELON

The husbandry underneath is deliberately plain. What is being seeded is the
opening, and a clever policy on top of it would make it impossible to tell which
half mattered.
"""

GRID = 10
# Wheat: $10 a seed, first yield at age 2, capped at 6, and the only crop that
# also feeds animals. Everything here is wheat for that second reason.
SEED_COST = 10.0
HIRES = {0: 7, 1: 4, 2: 7, 3: 7, 4: 7}
LATER_HIRES = 7
LAND_DAY = 3
LAND_PRICE = (1000.0, 2000.0, 4000.0)
# No herd, though the mined opening buys one. `BUY_ANIMAL` puts the animal in
# the shed, and it earns nothing until a unit builds a pasture, carries it
# there and places it. Bought without that, five cows and a sheep is $2,500 of
# the $3,000 both sides start with, spent on livestock that never leaves
# storage -- which is exactly what the first version of this file did, and it
# scored nought. Animals are the obvious first thing for a round to add.
KEEP = 60.0


def _where(hand):
    """A hand's position.

    A hand is a plain `[row, column]`, like the farmer. Guessed as a dict with
    a "position" key the first time, which sent every hand to the same square
    and had seven of them repeat one move all season.
    """
    return hand if isinstance(hand, (list, tuple)) else [4, 4]


def _jobs(tiles):
    """Every tile worth walking to, as (kind, row, column).

    Ripe before dry before empty: a harvest banks what is already grown, water
    saves what would otherwise weed, and planting is what to do with a spare
    unit once nothing is at risk.
    """
    ripe, dry, empty = [], [], []
    for row in range(GRID):
        for column in range(GRID):
            tile = tiles[row][column]
            if tile is None:
                empty.append(("plant", row, column))
            elif isinstance(tile, dict) and tile.get("crop"):
                if float(tile.get("yield_units") or 0) > 0:
                    ripe.append(("harvest", row, column))
                elif not tile.get("watered_today"):
                    dry.append(("water", row, column))
            elif isinstance(tile, dict) and tile.get("animal"):
                if not tile.get("fed_today"):
                    dry.append(("feed", row, column))
    return ripe + dry + empty


def _unit(position, tiles, jobs, private):
    """What one unit does this step: act where it stands, or walk to work."""
    row, column = int(position[0]), int(position[1])
    tile = tiles[row][column] if 0 <= row < GRID and 0 <= column < GRID else "LOCKED"
    if isinstance(tile, dict):
        if tile.get("crop"):
            if float(tile.get("yield_units") or 0) > 0:
                return ["HARVEST"]
            if not tile.get("watered_today"):
                return ["WATER"]
        elif tile.get("animal") and not tile.get("fed_today"):
            return ["FEED"]
    if tile is None and float((private.get("seeds") or {}).get("WHEAT") or 0) > 0:
        return ["PLANT", "WHEAT"]
    # Nothing to do here, so walk to the nearest job. Carried harvest needs no
    # DROP: the day-end refresh deposits inventory to the shed, and dropping
    # unconditionally at a corner left the farmer standing on one all season
    # repeating the same move.
    target = _nearest(row, column, jobs)
    return _step(row, column, target) if target else ["PASS"]


def _nearest(row, column, jobs):
    """The closest job by walking distance, or None when there is none."""
    best = None
    for _, target_row, target_column in jobs:
        span = abs(target_row - row) + abs(target_column - column)
        if best is None or span < best[0]:
            best = (span, target_row, target_column)
    return None if best is None else (best[1], best[2])


def _step(row, column, target):
    """One move toward `target`; rows first, so the path is deterministic."""
    target_row, target_column = target
    if target_row < row:
        return ["NORTH"]
    if target_row > row:
        return ["SOUTH"]
    if target_column < column:
        return ["WEST"]
    if target_column > column:
        return ["EAST"]
    return ["PASS"]


def _market(farm, private, day, hour, tiles):
    """The day's orders, sent once at its first hour."""
    if hour:
        return []
    money = float(farm["money"])
    orders = []
    # Hire to the mined floor. Labour is the cheapest thing on the board and
    # the opening's first move is seven of them.
    want = HIRES.get(day, LATER_HIRES)
    for _ in range(max(0, want - int(farm.get("hires_today") or 0))):
        orders.append(["HIRE"])
    # The land, on the day they take it.
    quadrants = len(farm.get("unlocked_quadrants") or [])
    if day >= LAND_DAY and 1 <= quadrants <= 3:
        price = LAND_PRICE[quadrants - 1]
        if money >= price:
            orders.append(["BUY_LAND"])
            money -= price
    # Seed for the empty ground, keeping a float back for feed and hires.
    empty = sum(1 for row in tiles for tile in row if tile is None)
    held = float((private.get("seeds") or {}).get("WHEAT") or 0)
    short = min(empty - held, int((money - KEEP) / SEED_COST))
    if short > 0:
        orders.append(["BUY_SEED", "WHEAT", int(short)])
    # And sell what the shed has accumulated. Harvest lands in carried
    # inventory and the day-end refresh deposits it, so by the first hour of
    # the next day it is in the shed and sellable.
    for item, count in (private.get("shed") or {}).items():
        if int(count or 0) > 0 and item != "FERTILIZER":
            orders.append(["SELL", item, int(count)])
        elif item == "FERTILIZER" and int(count or 0) > 0:
            orders.append(["SELL", item, int(count)])
    return orders


# `agent` is defined last, and that is load-bearing rather than tidy. The Kaggle
# runner takes `[v for v in env.values() if callable(v)][-1]` -- the last
# callable by insertion order -- so a helper defined below this one is served
# as the agent instead. It was, the first time this file was written: the
# runner picked `_market` and every game died on a TypeError.
def agent(observation, configuration=None):
    """One move: every unit acts, then the day's market orders go out."""
    me = observation["player"]
    farm = observation["farms"][me]
    day = int(observation["day"])
    hour = int(observation["hour"])
    private = observation.get("private") or {}
    tiles = farm["tiles"]
    hands = farm.get("hands") or []

    jobs = _jobs(tiles)
    farmer = _unit(farm["farmer"], tiles, jobs, private)
    crew = [_unit(_where(hand), tiles, jobs, private) for hand in hands]
    return {
        "farmer": farmer,
        "hands": crew,
        "market": _market(farm, private, day, hour, tiles),
    }
