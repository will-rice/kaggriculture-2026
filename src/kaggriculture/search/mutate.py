"""Single-edit mutations over a route.

Five edits, each confined to one turn: replace a unit's op, insert a new
market order, retime two of a turn's orders against each other, resize one,
or retarget one. Compound edits are excluded deliberately -- the fitness is a
win rate over a few hundred noisy games, and a candidate differing in several
places tells us nothing about which edit paid.

Legality is not checked here. The engine masks an op the board does not permit
and partially fills an order larger than the shed holds, so an illegal edit
simply scores badly and the search discards it. Checking here would mean a
second implementation of the rules, which is the defect this project has hit
most often.
"""

import copy
import random

from kaggriculture.search.route import Route

TURNS_PER_DAY = 24
SELLABLE = (
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
)
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
# Engine grammar, not our internal op names. `encode_turn` would accept
# ["PLANT:TOMATO"] because that is how UNIT_OPS spells it, but the reference
# engine reads ["PLANT", "TOMATO"] and a route has to replay there -- the final
# gate is on the reference engine, not on the simulator.
#
# The PLANT entries are in the pool on purpose. Without them the search cannot
# put a tomato in the ground, and the whole reason for building it is that the
# town drains 264 tomato a season into a market the entire field supplies with
# 15 units. A mutation set that cannot represent the answer would hill-climb
# forever and return an honest-looking negative.
UNIT_OPS_POOL = (
    ("PASS",),
    ("WATER",),
    ("HARVEST",),
    ("COLLECT_FERTILIZER",),
    ("NORTH",),
    ("SOUTH",),
    ("EAST",),
    ("WEST",),
    ("PLANT", "WHEAT"),
    ("PLANT", "CARROT"),
    ("PLANT", "TOMATO"),
    ("PLANT", "STRAWBERRY"),
    ("PLANT", "MELON"),
)

# Retargeting and inserting both cover BUY_SEED and BUY_ANIMAL as well as
# SELL, for the same reason the PLANT ops are in the pool: buying tomato seed
# and buying a goose are the two market edits that make the hinge products
# reachable at all, and a SELL-only catalogue can never emit either.
MARKET_CATALOGUES = {"SELL": SELLABLE, "BUY_SEED": CROPS, "BUY_ANIMAL": ANIMALS}
INSERT_QUANTITY = 4


def mutate(route: Route, rng: random.Random) -> tuple[Route, str]:
    """Return a copy of ``route`` with exactly one turn changed.

    Args:
        route: The parent route, left untouched.
        rng: The stream deciding the edit, so a run is reproducible.

    Returns:
        The mutated route and a description of the edit.
    """
    mutated = copy.deepcopy(route)

    # Order slots and multi-order turns are gathered from real market verbs
    # only. A route with 100+ SELL orders and a single BUY_SEED order would
    # otherwise drown a uniform-over-existing-orders retarget: the chance of
    # landing on the one BUY_SEED slot, then re-choosing TOMATO from it, is
    # too small to clear inside a hill-climb's budget. `insert` sidesteps
    # that by landing on any of the 720 turns, independent of what is
    # already there -- the same freedom the PLANT unit-ops already have.
    order_slots = [
        (turn, slot)
        for turn, action in enumerate(route)
        for slot, order in enumerate(action["market"])
        if len(order) >= 3 and order[0] in MARKET_CATALOGUES
    ]
    multi_order_turns = [
        turn for turn, action in enumerate(route) if len(action["market"]) >= 2
    ]

    edits = ["unit", "insert"]
    if order_slots:
        edits += ["resize", "retarget"]
    if multi_order_turns:
        edits.append("retime")
    choice = rng.choice(edits)

    if choice == "unit":
        turn = rng.randrange(len(route))
        current = tuple(route[turn]["farmer"])
        choices = [op for op in UNIT_OPS_POOL if op != current]
        op = list(rng.choice(choices))
        mutated[turn]["farmer"] = op
        return mutated, f"turn {turn}: farmer -> {' '.join(op)}"

    if choice == "insert":
        turn = rng.randrange(len(route))
        verb = rng.choice(list(MARKET_CATALOGUES))
        item = rng.choice(MARKET_CATALOGUES[verb])
        mutated[turn]["market"].append([verb, item, INSERT_QUANTITY])
        return mutated, f"turn {turn}: insert {verb} {item} {INSERT_QUANTITY}"

    if choice == "retime":
        # A reorder within the turn, not a move to another turn. Moving an
        # order across turns would change two turns at once, and then a
        # fitness difference could not be attributed to one edit. Queue
        # position is what pairs the two seats' orders in the reference
        # engine, so a reorder inside one turn is a real edit, not a no-op.
        turn = rng.choice(multi_order_turns)
        slots = mutated[turn]["market"]
        slot, other = rng.sample(range(len(slots)), 2)
        slots[slot], slots[other] = slots[other], slots[slot]
        return mutated, f"turn {turn}: slots {slot} and {other} swapped"

    turn, slot = rng.choice(order_slots)
    order = list(mutated[turn]["market"][slot])

    if choice == "resize":
        current_quantity = int(order[2])
        deltas = [
            d
            for d in (-4, -2, -1, 1, 2, 4)
            if max(1, min(64, current_quantity + d)) != current_quantity
        ]
        quantity = max(1, min(64, current_quantity + rng.choice(deltas)))
        order[2] = quantity
        mutated[turn]["market"][slot] = order
        return mutated, f"turn {turn} slot {slot}: quantity -> {quantity}"

    # retarget
    choices = [item for item in MARKET_CATALOGUES[order[0]] if item != order[1]]
    item = rng.choice(choices)
    order[1] = item
    mutated[turn]["market"][slot] = order
    return mutated, f"turn {turn} slot {slot}: item -> {item}"
