"""Single-edit mutations over a route.

Four edits, each confined to one turn: replace a unit's op, retime two of a
turn's orders against each other, resize one, or retarget one. Compound edits
are excluded deliberately -- the fitness is a win rate over a few hundred
noisy games, and a candidate differing in several places tells us nothing
about which edit paid.

Whole-order insertion was considered and left out on purpose. Every hinge
product the search needs is already reachable by rewriting an order that
exists: a harvested route carries roughly 92 BUY_SEED orders and 103 SELL
orders, so retargeting one of those reaches BUY_SEED:TOMATO and SELL:TOMATO,
and PLANT:TOMATO is already in the unit-op pool. Inserting a whole new order
would only buy market activity on turns that currently have none -- and
harvested routes already carry orders on 282 of 720 turns. Evaluation costs
about a minute a candidate, so a move type with no identified use is not
free; do not add it back without a hinge product it is the only way to reach.

Legality is not checked here. The engine masks an op the board does not permit
and partially fills an order larger than the shed holds, so an illegal edit
simply scores badly and the search discards it. Checking here would mean a
second implementation of the rules, which is the defect this project has hit
most often.
"""

import copy
import random

from kaggriculture.search.route import Route

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

# Retargeting covers BUY_SEED and BUY_ANIMAL as well as SELL, for the same
# reason the PLANT ops are in the pool: buying tomato seed and buying a goose
# are the two market edits that make the hinge products reachable at all, and
# a SELL-only catalogue can never emit either.
MARKET_CATALOGUES = {"SELL": SELLABLE, "BUY_SEED": CROPS, "BUY_ANIMAL": ANIMALS}


def mutate(route: Route, rng: random.Random) -> tuple[Route, str]:
    """Return a copy of ``route`` with exactly one turn changed.

    Args:
        route: The parent route, left untouched.
        rng: The stream deciding the edit, so a run is reproducible.

    Returns:
        The mutated route and a description of the edit.
    """
    mutated = copy.deepcopy(route)

    # Order slots grouped by verb. A route typically carries far more SELL
    # orders than BUY_SEED or BUY_ANIMAL ones, so retarget chooses a verb
    # first, uniformly, and only then an order of that verb -- a buy-side
    # retarget is exactly as likely as a sell-side one regardless of how
    # lopsided the counts are. Choosing uniformly over every order instead
    # would drown the one BUY_SEED order among a hundred SELL orders and
    # make BUY_SEED:TOMATO unreachable inside a hill-climb's budget.
    kind_slots: dict[str, list[tuple[int, int]]] = {
        kind: [] for kind in MARKET_CATALOGUES
    }
    for turn, action in enumerate(route):
        for slot, order in enumerate(action["market"]):
            if len(order) >= 3 and order[0] in kind_slots:
                kind_slots[order[0]].append((turn, slot))
    order_slots = [pair for slots in kind_slots.values() for pair in slots]
    # A turn only admits a real retime if two of its orders differ in
    # content. Two `SELL WHEAT 3` orders queued back to back swap into an
    # identical list -- a route a real season plausibly produces -- so a
    # turn where every order is the same as every other cannot retime at
    # all, not even by picking a different pair.
    retimable_turns = [
        turn
        for turn, action in enumerate(route)
        if len({tuple(order) for order in action["market"]}) >= 2
    ]

    edits = ["unit"]
    if order_slots:
        edits += ["resize", "retarget"]
    if retimable_turns:
        edits.append("retime")
    choice = rng.choice(edits)

    if choice == "unit":
        turn = rng.randrange(len(route))
        current = tuple(route[turn]["farmer"])
        choices = [op for op in UNIT_OPS_POOL if op != current]
        op = list(rng.choice(choices))
        mutated[turn]["farmer"] = op
        return mutated, f"turn {turn}: farmer -> {' '.join(op)}"

    if choice == "retime":
        # A reorder within the turn, not a move to another turn. Moving an
        # order across turns would change two turns at once, and then a
        # fitness difference could not be attributed to one edit. Queue
        # position is what pairs the two seats' orders in the reference
        # engine, so a reorder inside one turn is a real edit, not a no-op
        # -- provided the two slots swapped actually differ in content. The
        # turn was chosen because it holds at least two distinct orders, so
        # every slot has at least one differing partner to swap with.
        turn = rng.choice(retimable_turns)
        slots = mutated[turn]["market"]
        slot = rng.randrange(len(slots))
        candidates = [i for i in range(len(slots)) if slots[i] != slots[slot]]
        other = rng.choice(candidates)
        slots[slot], slots[other] = slots[other], slots[slot]
        return mutated, f"turn {turn}: slots {slot} and {other} swapped"

    if choice == "resize":
        turn, slot = rng.choice(order_slots)
        order = list(mutated[turn]["market"][slot])
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

    # retarget: verb first, then an order of that verb, then a new item.
    kind = rng.choice([k for k, slots in kind_slots.items() if slots])
    turn, slot = rng.choice(kind_slots[kind])
    order = list(mutated[turn]["market"][slot])
    choices = [item for item in MARKET_CATALOGUES[kind] if item != order[1]]
    item = rng.choice(choices)
    order[1] = item
    mutated[turn]["market"][slot] = order
    return mutated, f"turn {turn} slot {slot}: item -> {item}"
