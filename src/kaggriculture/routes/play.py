"""Replay the nearest recorded route, and price this turn's market live.

The store's median route banks 154,636 where the vendored ``economic_policy``
banks about 118,000, so the value here is in replaying what a strong seat
actually did rather than re-deciding every turn from scratch. Phase 3's
behaviour clone banked nothing for exactly that reason: it re-decided each
turn and hired on nearly every one, where its teacher hired on 14.6% of turns
and only in a day's first four hours. A route carries the sequence a per-turn
policy cannot represent.

Three divisions make that replay survive contact with a live game.

**The whole turn is replayed, market included, and then clamped.** This
started out the other way round: the plan replayed production and recomputed
every market order from ``economic_policy``, on the reasoning that a recorded
``SELL WHEAT 40`` was priced against an inventory that no longer exists. That
reasoning is wrong, and one episode says so -- 10,312 banked with the market
recomputed against 181,321 with it replayed. A route's buys and its production
are one plan: it buys the cow on the turn its units are walking to the pasture
that will hold it. Substituting an independent market policy buys livestock
the replayed units never place and seeds they never sow, so money leaves and
nothing is produced; the recomputed episode ended with four cows and four
sheep stranded in the shed. What a live board really invalidates is not the
*intent* of a recorded order but its *size*, so orders are replayed and then
cut down to what our own shed and bank support. See ``clamp``.

**Orders are realigned onto the units we actually have.** A recorded action is
an instruction to the unit standing on a particular tile. Our hand count comes
from our own live hiring, so it routinely differs from the route's, and a
recorded list of hand orders applied positionally would hand a harvest order
to a unit standing somewhere else entirely. See ``realign``.

**Nothing matches forever, so the fallback is counted.** When the nearest
route is further away than ``MATCH_THRESHOLD`` this turn is played by
``economic_policy.agent`` whole, which makes this design's floor the agent we
already ship rather than zero. ``RouteAgent.fallbacks`` counts those turns and
the count is logged at the end of the episode: a route memory that falls back
on 90% of turns is ``economic_policy`` with extra steps, and no win rate would
ever say so.

This module is on the agent path, so it imports no torch, no wandb and nothing
under a ``scripts`` package -- importing torch alone costs 10.7 seconds of the
submission sandbox's 60-second overage pool, and ``package.py`` excludes
``learn/corpus.py`` from the archive entirely, so an import that reached it
would raise ``ModuleNotFoundError`` on turn zero. ``harvest`` was moved out of
``routes/store.py`` for that second reason; ``tests/routes/test_play.py`` pins
both.

``agent`` is deliberately the last callable defined, the same arrangement
``learn/play.py`` uses: ``kaggle_environments`` execs an agent file and takes
the last callable in the resulting namespace, so this module stays safe to
name as an entrypoint directly.
"""

import functools
import logging
from pathlib import Path
from typing import Any, Mapping

from kaggriculture import economic_policy
from kaggriculture.constants import (
    ANIMALS,
    BOARD_SIZE,
    CROPS,
    EPISODE_STEPS,
    LAND_PRICES,
    MOVES,
    SHED_CAPACITY,
    TURNS_PER_DAY,
    hire_cost,
    quadrant_of,
    shed_access_tiles,
)
from kaggriculture.routes.signature import (
    HANDS_SCALE,
    SIGNATURE_FIELDS,
    distance,
    signature,
)
from kaggriculture.routes.store import Prototype, load

LOGGER = logging.getLogger(__name__)

Position = tuple[int, int]

# Beside this file, so one path serves both the local league and the unpacked
# submission archive -- the same reasoning `learn/__init__.py` gives for the
# checkpoint. `routes/scripts/harvest.py` writes the store under `/data`, and
# `package.py` is what copies it here.
STORE = Path(__file__).parent / "prototypes.json.gz"

# How far a route's recorded signature may sit from the live board before this
# turn is played by `economic_policy` instead. Set from where the distances
# actually fall in the two regimes, never from which bank came out largest.
#
# When a replay is tracking, the live board is nearly the recorded one: over a
# full episode against `starter` with every turn replayed, the nearest route
# sits at p50 0.02 and never exceeds 1.27, and against `economic_policy` on a
# seed where the replay holds, never above 0.94. When a replay has derailed
# the number is an order of magnitude larger -- p90 7.9 and a maximum of 57.4
# on the seed where it does derail, and 25-47 routinely under the older,
# broken market split. There is a clear gap between "tracking" and "lost", and
# 4.0 sits in it: roughly three times the worst tracking error seen, and well
# under the range that means the board has left the route behind.
#
# UNMEASURED against outcomes -- that is Task 5's sweep, and this needs to be
# a value the sweep can move in both directions rather than an infinity that
# can only come down. The earlier value of 1.5 was chosen when replay was
# harmful and was really protecting us from a bug in the market split; with
# that fixed it would leave almost no headroom above the observed maximum,
# and a single spurious fallback compounds -- one turn off the route pushes
# the board further from it, which makes the next turn likelier to fall back
# too.
#
# Note that `distance` is not scale-stable across the season: its composition
# weight climbs from 0.5 on day 0 to 40 on day 30, so one flat constant is a
# far stricter test late than early. Recorded here for the sweep rather than
# quietly patched with a second guess.
MATCH_THRESHOLD = 4.0

# How much closer a challenger route must be before it displaces the route we
# are already following. Each route is individually coherent -- it hires, buys
# and plants in a sequence that pays off later -- and alternating between two
# of them yields a sequence neither would ever play, so a challenger has to
# win by a margin rather than by a hair. UNMEASURED, like the threshold above:
# Task 5 sweeps it. Evidence that it is currently too loose rather than too
# tight: over one local episode's 87 replayed turns this still followed 11
# distinct routes and switched 17 times.
HYSTERESIS_MARGIN = 0.25

_HANDS_INDEX = SIGNATURE_FIELDS.index("HANDS")

# Where the engine puts a farmer at the start of every day: the first
# shed-access tile inside the quadrant every farm starts with. Derived from
# `constants` rather than written as a literal so it cannot drift from the
# engine's own `_default_spawn`.
SPAWN: Position = next(
    tile for tile in shed_access_tiles() if quadrant_of(*tile) == "NW"
)


class RouteAgent:
    """Plays a season by replaying the nearest recorded route, turn by turn.

    Carries the two pieces of state a single ``act`` call cannot hold: which
    route is currently being followed, so that hysteresis can prefer it, and
    how many turns have been played by the fallback instead.

    One instance plays one episode. ``current`` is an index into
    ``prototypes``, or ``None`` before the first match and after any turn that
    fell back.
    """

    def __init__(self, prototypes: list[Prototype]) -> None:
        self.prototypes = prototypes
        self.current: int | None = None
        self.fallbacks = 0

    def act(
        self, observation: Mapping[str, Any], config: Mapping[str, int] | None = None
    ) -> dict[str, Any]:
        """Return this turn's farmer, hand and market actions.

        The turn index comes from ``day`` and ``hour``, never from
        ``observation["step"]``: ``kaggle_environments`` writes ``step`` onto
        agent 0's observation only, so seat 1 has no such key -- the same
        reason ``routes.signature`` reads the phase the way it does.

        Args:
            observation: One turn's observation, as the environment hands it
                over.
            config: The episode configuration, or ``None`` for the defaults.

        Returns:
            The action dict the environment consumes: one op for the farmer,
            one per hired hand, and this turn's market orders.
        """
        seat = int(observation.get("player", 0) or 0)
        day = int(observation["day"])
        step = day * TURNS_PER_DAY + int(observation["hour"])
        chosen = self.choose(signature(observation, seat), step, day)
        if step == EPISODE_STEPS - 1:
            LOGGER.info(
                "route memory fell back to the economic policy on %d of %d turns",
                self.fallbacks + (1 if chosen is None else 0),
                EPISODE_STEPS,
            )
        if chosen is None:
            self.fallbacks += 1
            self.current = None
            return economic_policy.agent(observation, config)

        self.current = chosen
        recorded = self.prototypes[chosen].actions[step]
        farm = observation["farms"][seat]
        private = observation["private"]
        ours = [_position(farm["farmer"])] + [_position(hand) for hand in farm["hands"]]
        plan = realign(recorded, route_units(self.prototypes[chosen], step), ours)

        # The shed as it will stand when the market runs: the engine applies
        # unit actions before market orders, so a DROP this turn is already in
        # the shed by the time a SELL is quoted, and clamping against the
        # observed shed would cut away sales the engine would have honoured.
        # `_post_field_storage` reads `farmer` and `hands` and never
        # `liquidation`, which the vendored TypedDict requires and which is
        # inert here.
        field: economic_policy.FieldPlan = {
            "farmer": plan["farmer"],
            "hands": plan["hands"],
            "liquidation": False,
        }
        capacity = SHED_CAPACITY if config is None else config["shedCapacity"]
        shed, _ = economic_policy._post_field_storage(private, field, capacity)
        plan["market"] = clamp(
            recorded["market"],
            shed,
            float(farm["money"]),
            int(farm["hires_today"]),
            len(farm["unlocked_quadrants"]),
            observation["market"]["prices"],
        )
        return plan

    def choose(self, key: tuple[float, ...], step: int, day: int) -> int | None:
        """Return which route to follow this turn, or ``None`` to fall back.

        The nearest route wins unless one is already being followed, in which
        case the challenger has to beat it by ``HYSTERESIS_MARGIN``. Routes
        that have no action left for this turn are not candidates: a season's
        final observation is followed by no action at all, so the last turn of
        every episode falls back.

        Args:
            key: The live board's signature.
            step: The turn index, ``day * TURNS_PER_DAY + hour``.
            day: The season day, which sets how ``distance`` weights phase
                against composition.

        Returns:
            An index into ``prototypes``, or ``None`` when the nearest route
            is further away than ``MATCH_THRESHOLD``.
        """
        distances = {
            index: distance(key, prototype.signatures[step], day)
            for index, prototype in enumerate(self.prototypes)
            if step < len(prototype.actions)
        }
        if not distances:
            return None
        best = min(distances, key=lambda index: (distances[index], index))
        incumbent = self.current
        if incumbent is not None and incumbent in distances:
            if distances[best] > distances[incumbent] - HYSTERESIS_MARGIN:
                best = incumbent
        return best if distances[best] <= MATCH_THRESHOLD else None


def realign(
    action: Mapping[str, Any],
    prototype_units: list[Position],
    our_units: list[Position],
) -> dict[str, Any]:
    """Map a route's unit orders onto the units standing on our board.

    Index 0 of both lists is the farmer -- the engine orders units that way
    and there is exactly one farmer per farm, so the farmer's order is never
    in question. The hands are the problem: our hand count comes from our own
    live hiring, so a recorded list of hand orders applied positionally would
    send our second hand to do what the route's second hand did, wherever the
    two happen to be standing. Instead each of our hands takes the order of
    the nearest route hand, nearest pair first, and route hands left over --
    hands the route hired and we did not -- have their orders dropped rather
    than issued to somebody else. Our own hands left over pass.

    An order is only issued verbatim once our unit stands where the route's
    unit stood. Until then it is a step toward that tile, because a stationary
    order executed one tile away operates on the wrong tile: ``HARVEST`` next
    to the crop the route meant harvests whatever is underfoot, or nothing.

    Args:
        action: The recorded action, carrying ``farmer`` and ``hands``.
        prototype_units: Where the route's units stood, farmer first.
        our_units: Where our units stand, farmer first.

    Returns:
        ``{"farmer": ..., "hands": [...]}`` with exactly one order per unit we
        have. The market half is not this function's business.
    """
    offers = list(zip(prototype_units[1:], action["hands"], strict=False))
    pairs = sorted(
        (_manhattan(ours, position), index, offer)
        for index, ours in enumerate(our_units[1:])
        for offer, (position, _) in enumerate(offers)
    )
    taken: dict[int, int] = {}
    claimed: set[int] = set()
    for _, index, offer in pairs:
        if index in taken or offer in claimed:
            continue
        taken[index] = offer
        claimed.add(offer)
    hands = [
        _translate(offers[taken[index]][1], offers[taken[index]][0], ours)
        if index in taken
        else ["PASS"]
        for index, ours in enumerate(our_units[1:])
    ]
    return {
        "farmer": _translate(action["farmer"], prototype_units[0], our_units[0]),
        "hands": hands,
    }


def clamp(
    orders: list[list[Any]],
    shed: Mapping[str, int],
    money: float,
    hires_today: int,
    quadrants: int,
    prices: Mapping[str, float],
) -> list[list[Any]]:
    """Cut a route's recorded orders down to what our own state supports.

    A recorded order is a statement about the farm that recorded it. We reach
    the same turn with a different shed and a different bank, so ``SELL WHEAT
    40`` against five sacks of wheat, or ``BUY_ANIMAL COW 2`` against 400
    coins, is a claim our board cannot back.

    What the engine does with such an order, read from ``_process_market`` and
    ``_commit_unit`` rather than assumed: orders are truncated to the first
    ten, then committed one unit at a time, and the first unit that fails --
    an empty shed, an unaffordable price -- abandons the rest of *that* order
    and moves on to the next one. So an over-large order is not rejected
    outright, it part-fills. Clamping is therefore behaviour-preserving at the
    engine level, not a rescue: its value is that the action we emit is a true
    statement of what will happen, which is what the fallback count, the
    replays and any later audit are read against. (Measured: no route in the
    store ever records more than ten orders, so dropping an order that clamps
    to nothing never buys a slot back either.)

    The money accounting deliberately errs high -- sale proceeds are credited
    at the price quoted now, and purchases are costed at it too, where the
    engine will re-quote per unit as the inventory moves. That asymmetry is
    the safe one: an order we keep and the engine rejects costs nothing, while
    an order we drop is gone. The shed is not estimated at all; it is our own
    private state and is exact.

    Args:
        orders: The route's recorded market orders for this turn.
        shed: Our shed as it will stand when the market runs -- after this
            turn's field plan has dropped into it.
        money: Our bank before any of these orders.
        hires_today: Hands hired so far today, which sets the next hire's cost.
        quadrants: How many quadrants we have unlocked, which sets the next
            land price.
        prices: This turn's quoted market prices.

    Returns:
        The orders we can actually back, in the recorded order, with
        quantities reduced and impossible orders dropped.

    Raises:
        ValueError: If an order carries a verb the engine does not define.
            Every one of the 136,610 actions in the store was checked against
            this set; a new verb means the store or the engine has changed
            under us, and playing on regardless would hide it.
    """
    remaining = dict(shed)
    budget = money
    hires = hires_today
    land = quadrants - 1
    kept: list[list[Any]] = []
    for order in orders:
        verb = order[0]
        if verb == "HIRE":
            cost = float(hire_cost(hires))
            if budget < cost:
                continue
            budget -= cost
            hires += 1
            kept.append(["HIRE"])
            continue
        if verb == "BUY_LAND":
            if land >= len(LAND_PRICES) or budget < LAND_PRICES[land]:
                continue
            budget -= LAND_PRICES[land]
            land += 1
            kept.append(["BUY_LAND"])
            continue
        item = order[1]
        if verb == "SELL":
            quantity = min(int(order[2]), int(remaining.get(item, 0)))
            if quantity <= 0:
                continue
            remaining[item] -= quantity
            budget += quantity * float(prices[item])
        else:
            unit = _unit_cost(verb, item, prices)
            quantity = min(int(order[2]), int(budget // unit))
            if quantity <= 0:
                continue
            budget -= quantity * unit
        kept.append([verb, item, quantity])
    return kept


def _unit_cost(verb: str, item: str, prices: Mapping[str, float]) -> float:
    """Return what one unit of a purchase costs, from the engine's own tables."""
    if verb == "BUY_SEED":
        return float(CROPS[item]["seed"])
    if verb == "BUY_ANIMAL":
        return float(ANIMALS[item]["cost"])
    if verb == "BUY_PRODUCT":
        return float(prices[item])
    raise ValueError(f"unknown market verb {verb!r}")


def route_units(prototype: Prototype, step: int) -> list[Position]:
    """Return where the route's farmer and hands stood at ``step``, farmer first.

    A prototype records what its units did, never where they were, so replay
    has to derive the positions. They are fully determined by the recording:
    the engine puts every unit back on the spawn tile at the end of each day,
    only the four movement verbs move a unit, and a hired hand appears on the
    least-occupied shed-access tile. Folding one day's recorded actions
    forward from the spawn therefore reconstructs the exact positions the
    engine had -- no approximation and no need to re-harvest the store.

    How many hands stood there comes from the signature rather than from the
    length of the recorded hand list: the signature's ``HANDS`` field is read
    straight off the observation the engine produced, where the recorded list
    is whatever the recorded agent chose to send, which the engine truncates
    to the hands that exist. Hires are applied after the turn's unit actions
    because that is the order the engine applies them in, and a hand spawning
    before its turn's moves would land on a different tile.

    Args:
        prototype: The route being replayed.
        step: The turn index to reconstruct, ``day * TURNS_PER_DAY + hour``.

    Returns:
        One position per unit the route had at ``step``, farmer first.
    """
    opening = step - step % TURNS_PER_DAY
    positions = _spawn_to([SPAWN], _hand_count(prototype, opening))
    for index in range(opening, step):
        positions = _advance(
            positions, prototype.actions[index], _hand_count(prototype, index + 1)
        )
    return positions


def _advance(
    positions: list[Position], action: Mapping[str, Any], hands: int
) -> list[Position]:
    """Return the unit positions one turn after ``action`` was played."""
    orders = [action["farmer"], *action["hands"]]
    moved = [
        _step(position, order)
        for position, order in zip(positions, orders, strict=False)
    ]
    moved += positions[len(moved) :]
    return _spawn_to(moved, hands)


def _step(position: Position, order: list[Any]) -> Position:
    """Return where a unit stands after being given ``order``."""
    if order[0] not in MOVES:
        return position
    dx, dy = MOVES[order[0]]
    moved = (position[0] + dx, position[1] + dy)
    if not (0 <= moved[0] < BOARD_SIZE and 0 <= moved[1] < BOARD_SIZE):
        return position
    return moved


def _spawn_to(positions: list[Position], hands: int) -> list[Position]:
    """Return ``positions`` grown to ``hands`` hands on the engine's spawn tiles.

    Mirrors the engine's own hand spawn: the first shed-access tile in NWSE
    order, ties broken by how many units already stand on it, recomputed after
    each hire because hands hired in one turn stack outward.
    """
    grown = list(positions)
    tiles = shed_access_tiles()
    while len(grown) - 1 < hands:
        grown.append(
            min(tiles, key=lambda tile: (grown.count(tile), tiles.index(tile)))
        )
    return grown


def _hand_count(prototype: Prototype, step: int) -> int:
    """Return how many hands the route had at ``step``, read from its signature."""
    return round(prototype.signatures[step][_HANDS_INDEX] * HANDS_SCALE)


def _manhattan(a: Position, b: Position) -> int:
    """Return the walking distance between two tiles."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _translate(order: list[Any], source: Position, destination: Position) -> list[Any]:
    """Return ``order`` as the unit at ``destination`` can carry it out."""
    if source == destination:
        return list(order)
    if source[0] != destination[0]:
        return ["EAST" if source[0] > destination[0] else "WEST"]
    return ["SOUTH" if source[1] > destination[1] else "NORTH"]


def _position(raw: list[int]) -> Position:
    """Return a unit's position as a tuple the rest of this module can compare."""
    return (int(raw[0]), int(raw[1]))


@functools.lru_cache(maxsize=1)
def route_agent() -> RouteAgent:
    """Return the episode's agent, loading the store once per process.

    The store is 3.6 MiB gzipped and takes about four seconds to decode, which
    is charged to the sandbox's overage pool exactly once rather than on every
    turn. Cached publicly rather than in a module global so a test can point
    ``STORE`` at its own file and drop what a previous test loaded.

    Returns:
        A fresh ``RouteAgent`` over the shipped store.
    """
    return RouteAgent(load(STORE))


def agent(
    observation: Mapping[str, Any], config: Mapping[str, int] | None = None
) -> dict[str, Any]:
    """Return this turn's action, replaying a route or falling back.

    Args:
        observation: One turn's observation, as the environment hands it over.
        config: The episode configuration, or ``None`` for the defaults.

    Returns:
        The action dict the environment consumes.
    """
    return route_agent().act(observation, config)
