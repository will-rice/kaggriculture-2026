"""Which actions the engine will actually act on, per unit and per market slot.

Most of ``UNIT_OPS`` is illegal on any given turn: you cannot ``HARVEST`` bare
soil, ``PLANT`` without owning the seed, ``PICKUP`` what the shed does not hold,
or ``DROP`` away from a shed-access tile. The engine says so by doing nothing --
``_apply_unit_action``
is a chain of guards each of which ``return``s silently, and ``_commit_unit``
aborts an order by returning ``False``. An unmasked policy therefore spends its
training budget rediscovering legality instead of strategy, and never gets a
gradient signal to tell it which of the two it is failing at.

The two ways a mask can be wrong are not symmetric, and only one of them is
visible. A mask that is **too permissive** costs a turn: the engine no-ops the
action and the agent learns more slowly. A mask that is **too strict** deletes a
legal move from the action space permanently, and *nothing anywhere raises* --
the policy simply never learns the move exists. Every guard below is therefore
mirrored from the engine's source rather than from the rules documentation, and
``tests/learn/test_mask.py`` replays both masks against the engine on real
corpus observations and counts the disagreements in each direction separately.

The documentation is not usable for this. It has been wrong three times in this
project: it omits ``DROP``, which ``_apply_unit_action`` implements; it implies
``BUY_PRODUCT`` accepts any product when ``_process_market`` gates it to
``("WHEAT", "FERTILIZER")``; and it says nothing about ``observation["step"]``
being written onto agent 0 only. The functions mirrored here are
``_apply_unit_action`` (kaggriculture.py:297-506) for the unit mask and
``_process_market`` (520-603), ``_parse_order`` (606-624), ``_commit_unit``
(627-656), ``_do_hire`` (671-678) and ``_do_buy_land`` (681-694) for the market
one.

Two facts about the engine's geometry are load-bearing. A unit's position is
``[x, y]`` and the grid is ``tiles[y][x]`` -- ``_apply_unit_action`` reads
``fx, fy = pos[0], pos[1]`` and then ``farm["tiles"][fy][fx]``. And a unit
standing on a ``"LOCKED"`` tile can move and pass and do nothing else, which is
a state the game reaches on its own: ``_spawn_hand`` puts a new hand on the
first shed-access tile whether or not that quadrant has been bought.

``observation["private"]`` belongs to whoever's observation it is and the
opponent's is hidden, so a mask can only ever be computed for our own seat and
``private`` is never indexed by seat. The shed, the seeds and the per-unit
inventories are all read straight off it.

Both masks are returned at the shape of the logits they gate, batch dimension
included, so a rollout can set the masked positions to ``-inf`` before the
softmax without reshaping anything. No row of either mask is ever entirely
False: an all-``-inf`` row softmaxes to NaN, so the padded unit slots keep
``PASS`` and every market slot keeps bucket 0, "trade nothing", which the
engine always permits because it emits no order at all.

What this cannot express is the handful of constraints that couple slots
together, because the heads sample each slot independently:

* The turn's money is shared. Each market slot is masked against the full
  balance, so ten separately-affordable orders can still overdraw and the
  engine will abort the later ones.
* ``interpreter`` drops *every* ``PLANT`` for a crop whose seeds the turn's
  units jointly overspend. The mask permits ``PLANT`` per unit whenever a seed
  exists, so three units planting wheat against two seeds lose all three.
* ``decode_market`` truncates to ``MAX_ORDERS``, so an eleventh order never
  reaches the game.

Each of those wastes budget rather than deleting a move, which is the cheaper
of the two failures.
"""

from typing import Any, Mapping

import torch

from kaggriculture.constants import (
    ANIMALS,
    BOARD_SIZE,
    CROPS,
    LAND_ORDER,
    LAND_PRICES,
    MOVES,
    SHED_CAPACITY,
    hire_cost,
    market_price,
    shed_access_tiles,
)
from kaggriculture.learn.encoding import (
    CROP_NAMES,
    HIRE_SLOT,
    LAND_SLOT,
    MARKET_SLOTS,
    MAX_ORDERS,
    MAX_UNITS,
    QUANTITIES,
    SHED_NAMES,
    UNIT_OPS,
    TooManyUnitsError,
    quantity_of,
    unit_count,
)
from kaggriculture.observation import Tile

_OP_INDEX = {op: index for index, op in enumerate(UNIT_OPS)}
_SHED_ACCESS = frozenset(shed_access_tiles())

# The largest quantity any bucket stands for. Nothing above it can be ordered,
# so the affordability walks below stop there.
_MAX_QUANTITY = max(QUANTITIES)


def unit_mask(observation: Mapping[str, Any], seat: int) -> torch.Tensor:
    """Return which op each of ``seat``'s units may play this turn.

    Slot ``k`` is the ``k``-th unit in the engine's own order,
    ``[farmer, *farm["hands"]]``, which is the order ``encode_positions`` and
    ``encode_units`` use. Slots past the crew carry ``PASS`` alone: they are
    never decoded -- ``decode_units`` reads only the first ``unit_count`` of
    them -- but a row of all-False would become a row of all ``-inf`` and
    softmax to NaN.

    ``PICKUP:<item>`` and ``PLACE:<item>`` are masked per item, which is the
    whole reason ``UNIT_OPS`` carries the item at all: while both were bare
    verbs this mask forbade them outright, because ``_apply_unit_action``
    returns on ``len(action) < 2`` and neither could ever land. They are now
    masked from the engine's own guards -- what the shed holds for ``PICKUP``,
    what the unit holds and where it stands for ``PLACE``.

    Args:
        observation: One turn's observation, whose ``private`` mapping is
            ``seat``'s own.
        seat: Which player's units to mask. Must be the seat the observation
            belongs to, since ``private`` is not knowable for the other one.

    Returns:
        A ``(1, MAX_UNITS, len(UNIT_OPS))`` bool tensor, True where the engine
        would act on the op.

    Raises:
        TooManyUnitsError: If the crew exceeds ``MAX_UNITS``.
    """
    units = unit_count(observation, seat)
    if units > MAX_UNITS:
        raise TooManyUnitsError(f"{units} acting units exceeds MAX_UNITS={MAX_UNITS}")
    farm = observation["farms"][seat]
    private = observation["private"]
    mask = torch.zeros(1, MAX_UNITS, len(UNIT_OPS), dtype=torch.bool)
    mask[0, :, _OP_INDEX["PASS"]] = True
    for unit, position in enumerate([farm["farmer"], *farm["hands"]]):
        for op in _legal_ops(farm, private, unit, position, observation["day"]):
            mask[0, unit, _OP_INDEX[op]] = True
    return mask


def _legal_ops(
    farm: Mapping[str, Any],
    private: Mapping[str, Any],
    unit: int,
    position: list[int],
    day: int,
) -> set[str]:
    """Return the ops ``_apply_unit_action`` would act on for one unit.

    Follows the engine's own dispatch order: the moves and ``PASS`` are decided
    before the tile is even read, the ``"LOCKED"`` check gates everything after,
    and the three shed transfers -- ``DROP``, ``PICKUP`` and ``PLACE`` -- are
    settled from the shed geometry rather than from the tile. What is left is a
    question about the tile alone, which ``_tile_ops`` answers.

    ``PLACE`` is the one op both halves have a say in, and the union of the two
    is exact rather than approximate. ``_apply_unit_action`` tries the animal
    branch first -- an animal item onto a matching unoccupied structure, from
    this unit's own hands -- and only falls through to the shed drop when that
    branch's *condition* fails, never when its ``_inv_take`` does. So a bare
    coop takes a goose wherever it stands (``_tile_ops``), a bare pasture beside
    the shed puts that same goose back into the shed instead (here), and neither
    branch can offer an item the other should have refused.

    Args:
        farm: ``seat``'s farm.
        private: ``seat``'s private mapping, never indexed by seat.
        unit: Index into ``[farmer, *hands]``.
        position: That unit's ``[x, y]``.
        day: The observation's day, for the crop maturity gate.

    Returns:
        The names of the legal ops, always including ``PASS``.
    """
    x, y = position
    legal = {"PASS"}
    for op, (dx, dy) in MOVES.items():
        if 0 <= x + dx < BOARD_SIZE and 0 <= y + dy < BOARD_SIZE:
            legal.add(op)

    tile = farm["tiles"][y][x]
    if tile == "LOCKED":
        return legal

    # `_farmer_inventory` indexes `[farmer, *hands]` exactly as the units are
    # indexed, because `_do_hire` appends an inventory as it appends a hand.
    inventory = private["inventories"][unit]
    if (x, y) in _SHED_ACCESS:
        if inventory:
            # DROP empties the whole inventory. A full shed makes it destructive
            # rather than illegal: `room` clamps to zero and the engine deletes
            # the carried item either way, so the only guard is having something
            # to drop.
            legal.add("DROP")
        # PICKUP moves `min(n, shed[item])` and returns when that is zero, so
        # one item in the shed is the whole guard at TRANSFER_QUANTITY of 1.
        legal.update(
            f"PICKUP:{item}" for item in SHED_NAMES if private["shed"][item] > 0
        )
        # PLACE's shed drop is the mirror, and adds the capacity check DROP does
        # not have: `room` clamps to zero and the engine returns on `n <= 0`
        # instead of deleting the item, so a full shed makes this one illegal
        # where it makes DROP destructive.
        if sum(private["shed"].values()) < SHED_CAPACITY:
            legal.update(
                f"PLACE:{item}" for item in SHED_NAMES if inventory.get(item, 0) > 0
            )
    return legal | _tile_ops(tile, private, inventory, day)


def _tile_ops(
    tile: Tile, private: Mapping[str, Any], inventory: Mapping[str, int], day: int
) -> set[str]:
    """Return the ops legal on one unlocked tile, by what is standing on it.

    The engine's own tile taxonomy, in four branches: open ground, an occupied
    structure, a plant, and everything else -- a weed or a structure nobody has
    moved an animal into, which can be dug and, if it is the right kind of
    structure, filled. Written as a dispatch rather than as one chain of guards
    so that a tile kind added upstream lands in exactly one place instead of
    falling through to whichever branch happens to accept it.

    This answers the tile alone. The shed transfers that depend on where the
    unit is standing rather than on what it is standing on are ``_legal_ops``'
    business, and ``PLACE`` is in both -- see there for why the union is exact.

    Args:
        tile: The tile under the unit. Never ``"LOCKED"``; the caller returns
            first on that.
        private: ``seat``'s private mapping, for the seed counts.
        inventory: What this unit is carrying.
        day: The observation's day, for the crop maturity gate.

    Returns:
        The names of the legal ops.
    """
    if tile is None:
        legal = {"BUILD_COOP", "BUILD_PASTURE"}
        legal.update(
            f"PLANT:{crop}" for crop in CROP_NAMES if private["seeds"][crop] > 0
        )
        return legal
    if "animal" in tile:
        return _animal_ops(tile, inventory)
    if tile["kind"] == "PLANT":
        return _plant_ops(tile, inventory, day)
    # A weed or a bare coop or pasture. DIG clears all three and refuses only
    # an occupied structure, which the branch above has already taken.
    #
    # A bare structure is also the one place an animal can be PLACEd out of a
    # unit's hands, and the engine pairs them by `ANIMALS[item]["structure"]`
    # rather than by a shared "structure" kind -- a cow will not go in a coop.
    # Read off the rules table so an animal added upstream with a new structure
    # kind gets its own pairing instead of silently matching nothing.
    legal = {"DIG"}
    legal.update(
        f"PLACE:{animal}"
        for animal, data in ANIMALS.items()
        if tile["kind"] == data["structure"] and inventory.get(animal, 0) > 0
    )
    return legal


def _plant_ops(tile: Tile, inventory: Mapping[str, int], day: int) -> set[str]:
    """Return the ops legal on a growing crop.

    Args:
        tile: A ``kind == "PLANT"`` tile.
        inventory: What this unit is carrying, for the fertilizer bag.
        day: The observation's day.

    Returns:
        The names of the legal ops.
    """
    legal = {"DIG"}
    if not tile["watered_today"]:
        legal.add("WATER")
    if inventory.get("FERTILIZER", 0) > 0:
        legal.add("FERTILIZE")
    # `_new_plant` gives a one-time crop a yield unit on the day it goes in, so
    # the count alone says nothing; the engine also requires the crop's
    # `first_yield_day` to have passed.
    mature = day - tile["planted_day"] >= CROPS[tile["crop"]]["first_yield_day"]
    if tile["yield_units"] > 0 and mature:
        legal.add("HARVEST")
    return legal


def _animal_ops(tile: Tile, inventory: Mapping[str, int]) -> set[str]:
    """Return the ops legal on an occupied coop or pasture.

    ``DIG`` is absent on purpose: the engine refuses to clear a structure with
    an animal in it, which is the one thing that separates this branch from a
    bare structure's.

    Args:
        tile: A tile carrying an ``animal`` key.
        inventory: What this unit is carrying, for the wheat that FEED consumes.

    Returns:
        The names of the legal ops.
    """
    legal = set()
    if not tile["fed_today"] and inventory.get("WHEAT", 0) > 0:
        legal.add("FEED")
    if not tile["cared_today"]:
        legal.add("CARE")
    if tile["fertilizer_available"]:
        legal.add("COLLECT_FERTILIZER")
    if tile["yield_units"] > 0:
        legal.add("HARVEST")
    return legal


def market_mask(observation: Mapping[str, Any], seat: int) -> torch.Tensor:
    """Return which quantity bucket each market slot may carry this turn.

    A bucket is permitted only when the engine fills the whole order. The
    engine part-fills instead of refusing -- ``_commit_unit`` returns ``False``
    on the unit it cannot pay for and ``_process_market`` then abandons the
    rest -- so a twelve-sack sell order against five sacks does move state, and
    wastes seven of the ten orders a turn is allowed doing it.

    Bucket 0 is always permitted on every slot: it emits no order at all, and
    the corpus trades nothing on 46.3% of turns.

    ``BUY_LAND`` is masked to bucket 0 or 1. ``decode_market`` emits one
    ``BUY_LAND`` order for any non-zero bucket and ``encode_market`` labels the
    slot 0 or 1, so the fifteen larger buckets are fifteen spellings of one
    decision and would only spread the head's probability mass.

    ``HIRE`` is the one slot whose bucket is not a quantity on a single order.
    ``_parse_order`` returns ``HIRE`` bare, so one order hires one hand and
    ``decode_market`` repeats the order by count -- which means a ``HIRE``
    bucket spends that many of the turn's ``MAX_ORDERS`` and anything past the
    tenth is cut by ``_process_market``'s ``q[:max_orders]`` before it is ever
    read.

    ``HIRE`` is additionally capped at the crew ``encode_positions`` can
    represent. That cap is the mask's one deliberate departure from pure engine
    legality: the engine keeps hiring while the money lasts, but a crew wider
    than ``MAX_UNITS`` raises ``TooManyUnitsError`` on the next encode, so
    permitting the order makes a crash reachable from an ordinary rollout.

    Args:
        observation: One turn's observation, whose ``private`` mapping is
            ``seat``'s own.
        seat: Which player's orders to mask.

    Returns:
        A ``(1, len(MARKET_SLOTS) + 2, len(QUANTITIES))`` bool tensor.
    """
    farm = observation["farms"][seat]
    private = observation["private"]
    market = observation["market"]
    mask = torch.zeros(1, len(MARKET_SLOTS) + 2, len(QUANTITIES), dtype=torch.bool)
    mask[0, :, 0] = True

    money = farm["money"]
    for slot, (verb, item) in enumerate(MARKET_SLOTS):
        fillable = _fillable(verb, item, money, private, market)
        for bucket in range(1, len(QUANTITIES)):
            mask[0, slot, bucket] = quantity_of(bucket) <= fillable

    hires = _hireable(farm, money, MAX_UNITS - unit_count(observation, seat))
    for bucket in range(1, len(QUANTITIES)):
        mask[0, HIRE_SLOT, bucket] = quantity_of(bucket) <= hires

    bought = len(farm["unlocked_quadrants"]) - 1
    mask[0, LAND_SLOT, 1] = bought < len(LAND_ORDER) and money >= LAND_PRICES[bought]
    return mask


def _fillable(
    verb: str,
    item: str,
    money: float,
    private: Mapping[str, Any],
    market: Mapping[str, Any],
) -> int:
    """Return how many units of one (verb, item) order the engine would commit.

    ``_commit_unit``'s four branches, in its own order. ``SELL`` is bounded by
    the shed and never by money; the seed and animal purchases are priced from
    the rules tables and so divide evenly; ``BUY_PRODUCT`` is the one that has
    to be walked, because ``_process_market`` quotes it at ``inventory - 1``
    and ``_commit_unit`` then decrements the inventory, so each successive sack
    costs at least as much as the last and a flat ``money // price`` bound buys
    one more than the market will sell.

    Args:
        verb: The order's verb, one of ``MARKET_SLOTS``' four.
        item: The order's item.
        money: The farm's balance, shared with every other slot this turn.
        private: Our own private mapping.
        market: The public market.

    Returns:
        The largest quantity the engine would fill completely, capped at the
        largest quantity any bucket stands for.
    """
    if verb == "SELL":
        return private["shed"][item]
    if verb == "BUY_SEED":
        return int(money // int(CROPS[item]["seed"]))
    if verb == "BUY_ANIMAL":
        return int(money // int(ANIMALS[item]["cost"]))

    inventory = market["inventory"][item]
    params = market.get("params")
    filled = 0
    while filled < _MAX_QUANTITY:
        price = market_price(item, inventory - 1 - filled, params)
        if money < price:
            break
        money -= price
        filled += 1
    return filled


def _hireable(farm: Mapping[str, Any], money: float, room: int) -> int:
    """Return how many hands the farm could hire this turn.

    ``_do_hire`` charges ``fib(hires_today)`` and increments the counter on
    every success, so the ladder steepens within the turn and the count has to
    be walked rather than divided.

    Bounded by ``MAX_ORDERS`` as well as by the money: a ``HIRE`` bucket
    becomes that many separate one-hand orders, and ``_process_market`` cuts
    the queue at ``maxMarketOrdersPerTurn`` before reading any of it, so the
    eleventh hire of a turn cannot happen however rich the farm is.

    Args:
        farm: ``seat``'s farm, for ``hires_today``.
        money: The farm's balance.
        room: How many more units ``MAX_UNITS`` leaves space for.

    Returns:
        The number of hires the engine would complete, capped at ``room`` and
        at the turn's order budget.
    """
    hired = 0
    while hired < min(room, MAX_ORDERS):
        cost = hire_cost(farm["hires_today"] + hired)
        if money < cost:
            break
        money -= cost
        hired += 1
    return hired
