"""Tests for the legal-action masks.

A mask that is too permissive costs training budget: the engine no-ops the
action and the agent learns slowly. A mask that is too strict deletes a legal
move from the action space forever and *nothing raises* -- the policy simply
never discovers the move exists. The two failures are not symmetric, so every
check here asks both questions, and the corpus tests at the bottom ask them
against the engine itself rather than against a fixture that can only ever
agree with whatever the mask already does.
"""

import copy
import json
import zipfile
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from kaggle_environments.envs.kaggriculture import kaggriculture as engine

from kaggriculture.constants import (
    BOARD_SIZE,
    LAND_ORDER,
    LAND_PRICES,
    PRODUCTS,
    SHED_CAPACITY,
    TURNS_PER_DAY,
    hire_cost,
)
from kaggriculture.learn.corpus import CORPUS
from kaggriculture.learn.encoding import (
    BOARD,
    HIRE_SLOT,
    LAND_SLOT,
    MARKET_SLOTS,
    MAX_ORDERS,
    MAX_UNITS,
    QUANTITIES,
    UNIT_OPS,
    decode_units,
    quantity_of,
    unit_count,
)
from kaggriculture.learn.mask import market_mask, unit_mask
from kaggriculture.observation import Tile


def _empty_farm() -> dict[str, Any]:
    """Return a fresh farm with an unlocked, empty board and a farmer by the shed.

    The tile grid is rebuilt per call so the two farms in one observation never
    share a mutable list. The farmer stands on ``_default_spawn``, the engine's
    own opening tile, which is shed-adjacent -- so ``DROP`` and the shed
    branches are reachable from the fixture without moving anybody.
    """
    return {
        "tiles": [[None] * BOARD for _ in range(BOARD)],
        "money": 3000.0,
        "farmer": list(engine._default_spawn(BOARD_SIZE)),
        "hands": [],
        "unlocked_quadrants": ["NW"],
        "hires_today": 0,
    }


def empty_observation(seat: int = 0) -> dict[str, Any]:
    """Return a minimal well-formed observation with an empty unlocked board.

    ``private`` comes from ``engine._new_private`` rather than being typed out,
    so the shed and seed mappings are dense exactly as the real game writes
    them and a mask that reaches for a key the game does not write cannot pass
    here and fail on a real observation.
    """
    return {
        "player": seat,
        "day": 0,
        "hour": 0,
        "farms": [_empty_farm(), _empty_farm()],
        "market": {
            "prices": {item: engine.MARKET_PARAMS[item]["base"] for item in PRODUCTS},
            "inventory": dict.fromkeys(PRODUCTS, engine.MARKET_I0),
        },
        "town": {"unlocked_shops": []},
        "private": engine._new_private(),
    }


def _plant(crop: str = "WHEAT", day: int = 0) -> dict[str, Any]:
    """Return a freshly planted tile built by the engine's own constructor."""
    return engine._new_plant(crop, day, TURNS_PER_DAY)


def _animal(animal: str = "GOOSE", day: int = 0) -> dict[str, Any]:
    """Return a freshly placed animal tile built by the engine's own constructor."""
    return engine._new_animal(animal, day)


def _stand_on(observation: dict[str, Any], tile: Tile, seat: int = 0) -> dict[str, Any]:
    """Put ``tile`` under ``seat``'s farmer and return the observation.

    The farmer's position is ``[x, y]`` and the grid is ``tiles[y][x]`` --
    ``_apply_unit_action`` reads ``fx, fy = pos[0], pos[1]`` and then
    ``farm["tiles"][fy][fx]``. Writing that transposition out once, here, keeps
    every test below off it.
    """
    x, y = observation["farms"][seat]["farmer"]
    observation["farms"][seat]["tiles"][y][x] = tile
    return observation


def _op_mask(observation: dict[str, Any], op: str, seat: int = 0) -> bool:
    """Return whether the farmer may play ``op`` from this observation."""
    return bool(unit_mask(observation, seat)[0, 0, UNIT_OPS.index(op)])


# --------------------------------------------------------------------------
# The brief's tests, verbatim.
# --------------------------------------------------------------------------


def test_a_unit_may_always_pass() -> None:
    """PASS is the one op the engine never refuses; a mask that forbids it deadlocks."""
    mask = unit_mask(empty_observation(), seat=0)

    assert mask[0, 0, UNIT_OPS.index("PASS")]


def test_planting_needs_a_seed() -> None:
    """PLANT consumes stock the engine checks before it does anything."""
    without = empty_observation()
    with_seed = empty_observation()
    with_seed["private"]["seeds"]["MELON"] = 3

    assert not unit_mask(without, 0)[0, 0, UNIT_OPS.index("PLANT:MELON")]
    assert unit_mask(with_seed, 0)[0, 0, UNIT_OPS.index("PLANT:MELON")]


def test_selling_needs_stock_in_the_shed() -> None:
    """SELL draws from the shed; the engine refuses it empty."""
    empty = empty_observation()
    stocked = empty_observation()
    stocked["private"]["shed"]["WHEAT"] = 5
    slot = MARKET_SLOTS.index(("SELL", "WHEAT"))

    assert not market_mask(empty, 0)[0, slot, 1]
    assert market_mask(stocked, 0)[0, slot, 1]


def test_a_quantity_beyond_the_shed_is_masked() -> None:
    """Bucket 12 against five sacks is an order the engine part-fills and wastes."""
    observation = empty_observation()
    observation["private"]["shed"]["WHEAT"] = 5
    slot = MARKET_SLOTS.index(("SELL", "WHEAT"))

    mask = market_mask(observation, 0)

    assert mask[0, slot, QUANTITIES.index(4)]
    assert not mask[0, slot, QUANTITIES.index(12)]


# --------------------------------------------------------------------------
# Shape, dtype and the padding contract.
# --------------------------------------------------------------------------


def test_the_masks_have_the_shapes_of_the_logits_they_gate() -> None:
    """They are multiplied against logits, so a shape drift is a silent misgate."""
    observation = empty_observation()

    units = unit_mask(observation, 0)
    market = market_mask(observation, 0)

    assert units.shape == (1, MAX_UNITS, len(UNIT_OPS))
    assert units.dtype == torch.bool
    assert market.shape == (1, len(MARKET_SLOTS) + 2, len(QUANTITIES))
    assert market.dtype == torch.bool


def test_no_row_of_either_mask_is_entirely_forbidden() -> None:
    """An all-forbidden row becomes all -inf, and softmax over that is NaN.

    Checked on every row including the padded unit slots, which carry no unit
    at all. ``decode_units`` never reads them, but a NaN in the logits
    propagates through the loss whether the slot was decoded or not.
    """
    observation = empty_observation()

    assert unit_mask(observation, 0).any(dim=-1).all()
    assert market_mask(observation, 0).any(dim=-1).all()


def test_a_slot_with_no_unit_in_it_may_only_pass() -> None:
    """A padded slot has no position and no inventory; PASS is its only honest op."""
    observation = empty_observation()

    mask = unit_mask(observation, 0)
    padded = mask[0, unit_count(observation, 0) :]

    assert padded.sum() == padded.shape[0]
    assert padded[:, UNIT_OPS.index("PASS")].all()


def test_trading_nothing_is_always_available() -> None:
    """Bucket 0 is the engine's default: it emits no order at all."""
    observation = empty_observation()
    observation["farms"][0]["money"] = 0.0

    assert market_mask(observation, 0)[0, :, 0].all()


# --------------------------------------------------------------------------
# `_apply_unit_action`'s guards, one at a time.
# --------------------------------------------------------------------------


def test_a_unit_on_a_locked_tile_away_from_the_shed_may_only_move_or_pass() -> None:
    """The engine returns on `tile == "LOCKED"` before every op that reads it.

    Stood at ``[1, 1]``: an interior tile, so all four moves are in bounds and
    the only thing being tested is the guard, and not one of the four
    shed-access tiles, which the test below covers separately.
    """
    observation = empty_observation()
    observation["private"]["seeds"]["WHEAT"] = 9
    observation["private"]["inventories"][0] = {"WHEAT": 9, "FERTILIZER": 9}
    observation["farms"][0]["farmer"] = [1, 1]
    _stand_on(observation, "LOCKED")

    mask = unit_mask(observation, 0)[0, 0]
    allowed = {UNIT_OPS[index] for index in mask.nonzero().flatten().tolist()}

    assert allowed == {"PASS", "NORTH", "SOUTH", "EAST", "WEST"}


def test_a_locked_shed_access_tile_still_reaches_the_shed() -> None:
    """kaggle-environments 1.32.6 resolves the shed transfers before the guard.

    Three of the four shed-access tiles start ``"LOCKED"`` and a hand can spawn
    on one, so guarding the tile first would make the shed unreachable from
    them -- which is why the engine moved ``DROP``, ``PICKUP`` and ``PLACE``
    above the guard. They use the tile only as a standing position; the shed
    itself is always owned.

    Everything that mutates the tile stays forbidden, which is what separates
    this from simply dropping the guard. ``PLACE`` is the op that proves it:
    its shed-drop branch is permitted here, while its animal branch cannot be,
    because that branch needs a structure dict and a locked tile is the string
    ``"LOCKED"``.
    """
    observation = empty_observation()
    observation["private"]["seeds"]["WHEAT"] = 9
    observation["private"]["inventories"][0] = {"WHEAT": 9, "FERTILIZER": 9}
    observation["private"]["shed"]["WHEAT"] = 3
    # The default spawn is shed-adjacent, so locking it is the real state.
    _stand_on(observation, "LOCKED")

    mask = unit_mask(observation, 0)[0, 0]
    allowed = {UNIT_OPS[index] for index in mask.nonzero().flatten().tolist()}

    assert allowed == {
        "PASS",
        "NORTH",
        "SOUTH",
        "EAST",
        "WEST",
        "DROP",
        "PICKUP:WHEAT",
        "PLACE:WHEAT",
        "PLACE:FERTILIZER",
    }


def test_a_move_off_the_board_is_masked() -> None:
    """The engine bounds-checks the destination and no-ops rather than clamping."""
    observation = empty_observation()
    observation["farms"][0]["farmer"] = [0, 0]

    mask = unit_mask(observation, 0)[0, 0]

    assert not mask[UNIT_OPS.index("WEST")]
    assert not mask[UNIT_OPS.index("NORTH")]
    assert mask[UNIT_OPS.index("EAST")]
    assert mask[UNIT_OPS.index("SOUTH")]


def test_moving_onto_a_locked_tile_is_allowed() -> None:
    """The engine permits it on purpose: blocking it would strand a spawned hand."""
    observation = empty_observation()
    observation["farms"][0]["farmer"] = [4, 4]
    observation["farms"][0]["tiles"][4][5] = "LOCKED"

    assert _op_mask(observation, "EAST")


def test_watering_is_masked_once_it_has_been_watered() -> None:
    """A second WATER the same day is a no-op the engine returns on immediately."""
    dry = _stand_on(empty_observation(), _plant())
    wet = _stand_on(empty_observation(), _plant())
    wet["farms"][0]["tiles"][4][4]["watered_today"] = True

    assert _op_mask(dry, "WATER")
    assert not _op_mask(wet, "WATER")


def test_harvesting_an_immature_plant_is_masked() -> None:
    """A one-time crop carries a yield unit from the day it is planted.

    ``_new_plant`` sets ``yield_units`` to 1 immediately, so the yield count
    alone says nothing; the engine also requires ``day - planted_day`` to have
    reached the crop's ``first_yield_day``. A mask that checked only the yield
    would offer HARVEST on every wheat tile from the hour it went in.
    """
    young = _stand_on(empty_observation(), _plant("WHEAT", day=0))
    mature = _stand_on(empty_observation(), _plant("WHEAT", day=0))
    mature["day"] = engine.CROPS["WHEAT"]["first_yield_day"]

    assert not _op_mask(young, "HARVEST")
    assert _op_mask(mature, "HARVEST")


def test_harvesting_an_animal_needs_something_to_collect() -> None:
    """An animal has no maturity gate, only a yield count."""
    empty = _stand_on(empty_observation(), _animal("GOOSE"))
    laid = _stand_on(empty_observation(), _animal("GOOSE"))
    laid["farms"][0]["tiles"][4][4]["yield_units"] = 2

    assert not _op_mask(empty, "HARVEST")
    assert _op_mask(laid, "HARVEST")


def test_digging_refuses_an_occupied_structure_but_allows_a_bare_one() -> None:
    """`DIG` clears a plant, a weed and a bare structure and never an animal."""
    bare = _stand_on(empty_observation(), {"kind": "COOP"})
    occupied = _stand_on(empty_observation(), _animal("GOOSE"))
    weeds = _stand_on(empty_observation(), {"kind": "WEED"})
    open_ground = empty_observation()

    assert _op_mask(bare, "DIG")
    assert _op_mask(weeds, "DIG")
    assert not _op_mask(occupied, "DIG")
    assert not _op_mask(open_ground, "DIG")


def test_building_needs_open_ground() -> None:
    """Both builds require `tile is None`; neither replaces what is there."""
    open_ground = empty_observation()
    weeds = _stand_on(empty_observation(), {"kind": "WEED"})

    assert _op_mask(open_ground, "BUILD_COOP")
    assert _op_mask(open_ground, "BUILD_PASTURE")
    assert not _op_mask(weeds, "BUILD_COOP")
    assert not _op_mask(weeds, "BUILD_PASTURE")


def test_feeding_needs_wheat_in_this_unit_s_own_hands() -> None:
    """`FEED` takes from the unit's inventory, not from the shed.

    The shed is stocked in the empty-handed case to pin that down: a mask that
    read the shed would offer FEED to a unit standing empty-handed beside a
    full barn, and the engine would refuse every one of them.
    """
    empty_handed = _stand_on(empty_observation(), _animal("GOOSE"))
    empty_handed["private"]["shed"]["WHEAT"] = 50
    carrying = _stand_on(empty_observation(), _animal("GOOSE"))
    carrying["private"]["inventories"][0] = {"WHEAT": 1}

    assert not _op_mask(empty_handed, "FEED")
    assert _op_mask(carrying, "FEED")


def test_feeding_is_masked_once_the_animal_has_eaten() -> None:
    """`fed_today` short-circuits before the inventory is even touched."""
    observation = _stand_on(empty_observation(), _animal("COW"))
    observation["private"]["inventories"][0] = {"WHEAT": 4}
    observation["farms"][0]["tiles"][4][4]["fed_today"] = True

    assert not _op_mask(observation, "FEED")


def test_caring_is_masked_once_the_animal_has_been_cared_for() -> None:
    """The second CARE of a day banks no bonus."""
    fresh = _stand_on(empty_observation(), _animal("SHEEP"))
    done = _stand_on(empty_observation(), _animal("SHEEP"))
    done["farms"][0]["tiles"][4][4]["cared_today"] = True

    assert _op_mask(fresh, "CARE")
    assert not _op_mask(done, "CARE")


def test_collecting_fertilizer_needs_it_to_be_available() -> None:
    """`fertilizer_available` is set by the daily refresh and cleared on collection."""
    fresh = _stand_on(empty_observation(), _animal("COW"))
    ready = _stand_on(empty_observation(), _animal("COW"))
    ready["farms"][0]["tiles"][4][4]["fertilizer_available"] = True

    assert not _op_mask(fresh, "COLLECT_FERTILIZER")
    assert _op_mask(ready, "COLLECT_FERTILIZER")


def test_fertilizing_needs_a_plant_and_a_bag_in_hand() -> None:
    """`FERTILIZE` takes a FERTILIZER from the unit's inventory onto a PLANT tile."""
    bare = empty_observation()
    bare["private"]["inventories"][0] = {"FERTILIZER": 1}
    plant_no_bag = _stand_on(empty_observation(), _plant())
    both = _stand_on(empty_observation(), _plant())
    both["private"]["inventories"][0] = {"FERTILIZER": 1}

    assert not _op_mask(bare, "FERTILIZE")
    assert not _op_mask(plant_no_bag, "FERTILIZE")
    assert _op_mask(both, "FERTILIZE")


def test_dropping_needs_a_shed_tile_and_something_in_hand() -> None:
    """`DROP` empties the unit's whole inventory into the shed.

    It is masked empty-handed because the engine's loop then has nothing to
    iterate: the turn is spent and no state moves.
    """
    away = empty_observation()
    away["farms"][0]["farmer"] = [0, 0]
    away["private"]["inventories"][0] = {"WHEAT": 3}
    empty_handed = empty_observation()
    carrying = empty_observation()
    carrying["private"]["inventories"][0] = {"WHEAT": 3}

    assert not _op_mask(away, "DROP")
    assert not _op_mask(empty_handed, "DROP")
    assert _op_mask(carrying, "DROP")


def test_dropping_into_a_full_shed_still_moves_state() -> None:
    """The engine deletes the carried item whether or not the shed had room.

    A full shed makes DROP destructive rather than illegal, and masking it out
    would forbid a move the engine performs. Modelled here rather than argued:
    ``room`` clamps to zero and ``del inv[item]`` runs unconditionally.
    """
    observation = empty_observation()
    observation["private"]["shed"]["WHEAT"] = SHED_CAPACITY
    observation["private"]["inventories"][0] = {"MELON": 2}

    assert _op_mask(observation, "DROP")


def test_picking_up_needs_a_shed_tile_and_stock_of_that_item() -> None:
    """``PICKUP`` moves ``min(n, shed[item])`` and returns when that is zero.

    Item by item: an empty barn offers nothing, and a barn holding wheat offers
    wheat and not the melons it does not have. The away case pins the geometry
    guard, which is the same ``_is_shed_adjacent`` check ``DROP`` carries.
    """
    empty = empty_observation()
    stocked = empty_observation()
    stocked["private"]["shed"]["WHEAT"] = 20
    away = empty_observation()
    away["farms"][0]["farmer"] = [0, 0]
    away["private"]["shed"]["WHEAT"] = 20

    assert not _op_mask(empty, "PICKUP:WHEAT")
    assert _op_mask(stocked, "PICKUP:WHEAT")
    assert not _op_mask(stocked, "PICKUP:MELON")
    assert not _op_mask(away, "PICKUP:WHEAT")


def _on_a_coop_away_from_the_shed(carrying: dict[str, int]) -> dict[str, Any]:
    """Return an observation with the farmer on a bare coop in the far corner.

    Away from the shed on purpose. ``_default_spawn`` is shed-adjacent, so a
    structure built there offers every animal in hand through ``PLACE``'s shed
    branch -- the engine really would put a cow standing on a coop back into the
    barn -- and the animal-to-structure pairing this pins would be invisible.
    """
    observation = empty_observation()
    observation["farms"][0]["farmer"] = [0, 0]
    observation["farms"][0]["tiles"][0][0] = {
        "kind": str(engine.ANIMALS["GOOSE"]["structure"])
    }
    observation["private"]["inventories"][0] = carrying
    return observation


def test_placing_an_animal_needs_its_own_kind_of_structure() -> None:
    """The engine pairs an animal with ``ANIMALS[item]["structure"]``, not "any".

    A goose goes in a coop and a cow does not, so a mask that offered every
    animal on every bare structure would spend a turn per refusal. The animal
    branch works anywhere on the farm, unlike every other shed transfer, which
    is why these stand in the corner rather than at the spawn.
    """
    both = _on_a_coop_away_from_the_shed({"GOOSE": 1, "COW": 1})
    empty_handed = _on_a_coop_away_from_the_shed({})

    assert _op_mask(both, "PLACE:GOOSE")
    assert not _op_mask(both, "PLACE:COW")
    assert not _op_mask(empty_handed, "PLACE:GOOSE")


def test_an_animal_beside_the_shed_may_go_back_into_the_barn() -> None:
    """``PLACE``'s two branches, and which one the engine picks between them.

    ``_apply_unit_action`` tries the animal branch first and falls through to
    the shed drop only when its *condition* fails -- the wrong structure kind
    here. So a cow standing on a coop beside the shed is a legal ``PLACE``, and
    the same cow standing on a coop in the corner is not. The mask is the union
    of the two branches and this is the case that separates them.
    """
    beside = _stand_on(
        empty_observation(), {"kind": str(engine.ANIMALS["GOOSE"]["structure"])}
    )
    beside["private"]["inventories"][0] = {"COW": 1}
    corner = _on_a_coop_away_from_the_shed({"COW": 1})

    assert _op_mask(beside, "PLACE:COW")
    assert not _op_mask(corner, "PLACE:COW")


def test_placing_into_the_shed_needs_room_where_dropping_does_not() -> None:
    """The two shed transfers differ exactly here, and the engine says so.

    ``DROP`` clamps ``room`` to zero and deletes the carried item anyway, so a
    full shed makes it destructive rather than illegal. ``PLACE``'s shed branch
    returns on ``n <= 0`` instead, so the same full shed makes it a no-op.
    Masking them alike in either direction would be wrong in both.
    """
    room = empty_observation()
    room["private"]["inventories"][0] = {"MELON": 2}
    full = empty_observation()
    full["private"]["shed"]["WHEAT"] = SHED_CAPACITY
    full["private"]["inventories"][0] = {"MELON": 2}

    assert _op_mask(room, "PLACE:MELON")
    assert not _op_mask(full, "PLACE:MELON")
    assert _op_mask(full, "DROP")


# --------------------------------------------------------------------------
# `_process_market`'s guards.
# --------------------------------------------------------------------------


def test_buying_a_seed_is_bounded_by_money() -> None:
    """`_commit_unit` refuses a unit it cannot pay for and aborts the order."""
    observation = empty_observation()
    observation["farms"][0]["money"] = float(3 * engine.CROPS["WHEAT"]["seed"])
    slot = MARKET_SLOTS.index(("BUY_SEED", "WHEAT"))

    mask = market_mask(observation, 0)

    assert mask[0, slot, QUANTITIES.index(3)]
    assert not mask[0, slot, QUANTITIES.index(4)]


def test_buying_an_animal_is_bounded_by_money() -> None:
    """Animals are priced from the rules table, not the market."""
    observation = empty_observation()
    observation["farms"][0]["money"] = float(engine.ANIMALS["COW"]["cost"])
    slot = MARKET_SLOTS.index(("BUY_ANIMAL", "COW"))

    mask = market_mask(observation, 0)

    assert mask[0, slot, 1]
    assert not mask[0, slot, QUANTITIES.index(2)]


def test_buying_a_product_prices_each_unit_against_a_shrinking_inventory() -> None:
    """The quote moves per unit, so a flat `money // price` bound is wrong.

    ``_process_market`` quotes ``BUY_PRODUCT`` at ``inventory - 1`` and
    ``_commit_unit`` then decrements the inventory, so the second unit costs at
    least as much as the first. Priced flat at the opening quote, this budget
    buys one more sack than the engine will actually sell.
    """
    observation = empty_observation()
    inventory = observation["market"]["inventory"]["WHEAT"]
    quote = engine.market_price("WHEAT", inventory - 1)
    observation["farms"][0]["money"] = float(4 * quote)
    slot = MARKET_SLOTS.index(("BUY_PRODUCT", "WHEAT"))

    mask = market_mask(observation, 0)
    affordable = max(
        n for n in QUANTITIES if bool(mask[0, slot, QUANTITIES.index(n)]) or n == 0
    )
    exact = 0
    budget = 4.0 * quote
    while budget >= engine.market_price("WHEAT", inventory - 1 - exact):
        budget -= engine.market_price("WHEAT", inventory - 1 - exact)
        exact += 1

    assert affordable == exact


def test_hiring_is_bounded_by_the_fibonacci_ladder() -> None:
    """The n-th hire of a day costs fib(n), so the ladder bounds the bucket."""
    observation = empty_observation()
    observation["farms"][0]["hires_today"] = 4
    observation["farms"][0]["money"] = float(
        hire_cost(4) + hire_cost(5) + hire_cost(6) - 1
    )

    mask = market_mask(observation, 0)

    assert mask[0, HIRE_SLOT, QUANTITIES.index(2)]
    assert not mask[0, HIRE_SLOT, QUANTITIES.index(3)]


def test_hiring_never_exceeds_the_crew_the_encoders_can_represent() -> None:
    """``encode_positions`` raises past ``MAX_UNITS``; the mask must not steer there.

    This is the mask's one deliberate departure from pure engine legality. The
    engine will keep hiring for as long as the money lasts, but a crew wider
    than ``MAX_UNITS`` cannot be encoded at all, so permitting the order makes
    a crash reachable from an ordinary rollout.
    """
    observation = empty_observation()
    observation["farms"][0]["money"] = 1_000_000.0
    observation["farms"][0]["hands"] = [[4, 4]] * (MAX_UNITS - 3)
    observation["private"]["inventories"] = [{} for _ in range(MAX_UNITS - 2)]

    mask = market_mask(observation, 0)

    assert mask[0, HIRE_SLOT, QUANTITIES.index(2)]
    assert not mask[0, HIRE_SLOT, QUANTITIES.index(3)]


def test_hiring_never_exceeds_the_turn_s_order_budget() -> None:
    """One HIRE order hires one hand, so the bucket spends that many orders.

    ``_parse_order`` returns ``HIRE`` bare and ``decode_market`` repeats the
    order by count, so a bucket of 16 queues sixteen orders and
    ``_process_market`` cuts the queue at ``maxMarketOrdersPerTurn`` before
    reading any of it. Every bucket above ``MAX_ORDERS`` is therefore an order
    the engine can only part-fill, however rich the farm is.
    """
    observation = empty_observation()
    observation["farms"][0]["money"] = 1_000_000.0

    mask = market_mask(observation, 0)

    assert mask[0, HIRE_SLOT, QUANTITIES.index(MAX_ORDERS)]
    assert not mask[0, HIRE_SLOT, QUANTITIES.index(MAX_ORDERS) + 1 :].any()


def test_land_is_a_flag_and_every_larger_bucket_is_masked() -> None:
    """``decode_market`` emits one ``BUY_LAND`` for any non-zero bucket.

    Buckets 2 and up decode to the same single order as bucket 1, so leaving
    them open would spread the head's probability mass across sixteen spellings
    of one decision. ``encode_market`` labels this slot 0 or 1 and nothing
    else.
    """
    observation = empty_observation()
    observation["farms"][0]["money"] = float(LAND_PRICES[0])

    mask = market_mask(observation, 0)

    assert mask[0, LAND_SLOT, 1]
    assert not mask[0, LAND_SLOT, 2:].any()


def test_land_is_masked_when_it_is_unaffordable_or_all_bought() -> None:
    """`_do_buy_land` refuses on price and on there being no quadrant left."""
    poor = empty_observation()
    poor["farms"][0]["money"] = float(LAND_PRICES[0] - 1)
    complete = empty_observation()
    complete["farms"][0]["money"] = 1_000_000.0
    complete["farms"][0]["unlocked_quadrants"] = ["NW", *LAND_ORDER]

    assert not market_mask(poor, 0)[0, LAND_SLOT, 1]
    assert not market_mask(complete, 0)[0, LAND_SLOT, 1]


def test_the_mask_reads_the_seat_it_is_asked_for() -> None:
    """`private` belongs to the observation's owner and is never indexed by seat.

    Seat 1's farm is what changes here while ``private`` stays shared, which is
    exactly the real shape: both farms are public, one private mapping is in
    hand, and reading ``private[seat]`` would raise rather than mislead only
    because ``private`` is a mapping of the wrong shape.
    """
    observation = empty_observation(seat=1)
    observation["farms"][1]["farmer"] = [0, 0]

    ours = unit_mask(observation, 1)[0, 0]
    theirs = unit_mask(observation, 0)[0, 0]

    assert not ours[UNIT_OPS.index("WEST")]
    assert theirs[UNIT_OPS.index("WEST")]


# --------------------------------------------------------------------------
# Step 5: assert both masks against the engine, on real observations.
# --------------------------------------------------------------------------

# One turn out of every `_UNIT_STRIDE` of each archive's first episode, both
# seats. The unit probe applies every one of `UNIT_OPS` per unit per turn
# through a fresh copy of the engine and the market probe applies 21 slots x 16
# buckets, so the strides buy coverage across all seven archives -- the ladder's
# agent mix changes daily and this repo has twice shipped a claim that came from
# generalising one archive -- at a runtime measured in tens of seconds. The unit
# stride was widened when PICKUP and PLACE gained an item each and doubled the
# vocabulary.
_UNIT_STRIDE = 90
_MARKET_STRIDE = 120

_needs_corpus = pytest.mark.skipif(
    not CORPUS.is_dir(), reason="needs the replay corpus"
)


def _corpus_turns(stride: int) -> list[tuple[str, dict[str, Any], int]]:
    """Return ``(archive, observation, seat)`` sampled from every archive on disk.

    Streamed from the zips: an episode is ~27 MB decoded and the corpus is
    ~107 GB, so nothing here is ever extracted.
    """
    turns = []
    for archive in sorted(CORPUS.glob("*.zip")):
        with zipfile.ZipFile(archive) as bundle:
            name = next(n for n in bundle.namelist() if n.endswith(".json"))
            with bundle.open(name) as member:
                steps = json.load(member)["steps"]
        for index in range(0, len(steps), stride):
            for seat in (0, 1):
                turns.append((archive.name, steps[index][seat]["observation"], seat))
    return turns


def _decoded_op(unit: int, op_index: int, units: int) -> list[Any]:
    """Return the op list the policy would actually emit for one (unit, op) pair.

    Routed through ``decode_units`` rather than rebuilt from ``UNIT_OPS`` so the
    probe measures what the rollout emits -- ``TRANSFER_QUANTITY`` and all --
    instead of a more generous op the decoder cannot produce. That distinction
    is what made this probe agree with a mask that forbade ``PICKUP`` outright
    while the vocabulary carried no item: the decoder really could not express
    a landable one.

    The mask handed to ``decode_units`` is all-True on purpose, and this is the
    one place in the project that should hand it one. The probe exists to ask
    the engine whether ``unit_mask`` was right about an op; passing
    ``unit_mask`` itself would make the decoder refuse exactly the ops the mask
    forbids, so the over-permissive count would read zero however wrong the
    mask was and the test would be asserting its own premise.
    """
    logits = torch.zeros(1, MAX_UNITS, len(UNIT_OPS))
    logits[0, :, UNIT_OPS.index("PASS")] = 1.0
    logits[0, unit, op_index] = 2.0
    action = decode_units(logits, units, torch.ones_like(logits, dtype=torch.bool))
    return [action["farmer"], *action["hands"]][unit]


def _engine_acts_on(
    observation: dict[str, Any], seat: int, unit: int, op: list[Any]
) -> bool:
    """Return whether ``_apply_unit_action`` moves any state for one unit's op.

    The engine reports nothing: an illegal op is a silent ``return``. So
    legality is read off the state instead -- copy the farm and the private
    mapping, apply the op, and ask whether anything moved.

    ``PASS`` is the one op for which "nothing moved" means accepted rather than
    refused, and it is the one op the engine documents as always legal, so it
    is answered directly.
    """
    if op[0] == "PASS":
        return True
    farm = copy.deepcopy(observation["farms"][seat])
    private = copy.deepcopy(observation["private"])
    before = (copy.deepcopy(farm), copy.deepcopy(private))
    engine._apply_unit_action(
        farm,
        private,
        unit,
        op,
        BOARD_SIZE,
        observation["day"],
        TURNS_PER_DAY,
        SHED_CAPACITY,
    )
    return (farm, private) != before


@pytest.mark.slow
@_needs_corpus
def test_the_unit_mask_agrees_with_the_engine_on_real_observations() -> None:
    """Every (unit, op) pair, applied through the engine, on real turns.

    Both directions are counted and both are reported, because they fail
    differently. A permitted op the engine refuses costs a turn of the training
    budget. A forbidden op the engine accepts is deleted from the action space
    for good and raises nothing anywhere -- the policy simply never learns the
    move exists. Widening the mask until the numbers agree would fit it to
    whatever this probe happens to do; the numbers are asserted at zero
    instead.

    Zero disagreements is also what a probe that exercises nothing reports, so
    the verbs whose masks are newest are counted as well. While ``PICKUP`` and
    ``PLACE`` were bare verbs, both sides of every comparison involving them was
    ``False`` on every turn and this test passed as loudly as it does now. The
    counts are asserted non-zero so that can never be true again silently.
    """
    turns = _corpus_turns(_UNIT_STRIDE)
    permissive: list[str] = []
    strict: list[str] = []
    exercised: dict[str, int] = {"PICKUP": 0, "PLACE": 0}

    assert turns, "no archives on disk, so this proved nothing"
    for archive, observation, seat in turns:
        units = unit_count(observation, seat)
        mask = unit_mask(observation, seat)
        for unit in range(units):
            for op_index, name in enumerate(UNIT_OPS):
                op = _decoded_op(unit, op_index, units)
                accepted = _engine_acts_on(observation, seat, unit, op)
                allowed = bool(mask[0, unit, op_index])
                where = f"{archive} seat {seat} day {observation['day']} unit {unit}"
                if allowed and not accepted:
                    permissive.append(f"{where}: mask allows {name}, engine refuses it")
                if accepted and not allowed:
                    strict.append(f"{where}: engine accepts {name}, mask forbids it")
                if accepted and op[0] in exercised:
                    exercised[op[0]] += 1

    assert not permissive, f"{len(permissive)} over-permissive: {permissive[:5]}"
    assert not strict, f"{len(strict)} over-strict: {strict[:5]}"
    assert all(exercised.values()), (
        f"the engine accepted no {[k for k, v in exercised.items() if not v]} on any "
        "sampled turn, so their masks were never tested"
    )


def _committed(observation: dict[str, Any], seat: int, slot: int, quantity: int) -> int:
    """Return how many units of one market order the engine actually commits.

    Run through ``_process_market`` itself rather than through ``_commit_unit``
    in a loop, so the per-unit lockstep, the re-quoting and the abort-on-refusal
    are the engine's own. The opponent is given a fresh private mapping and no
    orders: their real one is hidden, and an idle opponent is the only opponent
    a single-seat probe can honestly model.

    Counted per verb from the state it moves, because a part-filled order is
    the failure this is looking for and "something changed" cannot see the
    difference between one unit sold and twelve.
    """
    farms = copy.deepcopy(observation["farms"])
    market = copy.deepcopy(observation["market"])
    privates = [engine._new_private(), engine._new_private()]
    privates[seat] = copy.deepcopy(observation["private"])
    orders = _orders(slot, quantity)
    state = [
        SimpleNamespace(
            observation=SimpleNamespace(
                farms=farms, market=market, private=privates[player]
            ),
            action={"market": orders if player == seat else []},
        )
        for player in (0, 1)
    ]
    engine._process_market(state, SimpleNamespace(configuration={}))

    farm, private = farms[seat], privates[seat]
    if slot == HIRE_SLOT:
        return len(farm["hands"]) - len(observation["farms"][seat]["hands"])
    if slot == LAND_SLOT:
        return len(farm["unlocked_quadrants"]) - len(
            observation["farms"][seat]["unlocked_quadrants"]
        )
    verb, item = MARKET_SLOTS[slot]
    before = observation["private"]
    if verb == "SELL":
        return before["shed"][item] - private["shed"][item]
    if verb == "BUY_SEED":
        return private["seeds"][item] - before["seeds"][item]
    return private["shed"][item] - before["shed"][item]


def _orders(slot: int, quantity: int) -> list[list[Any]]:
    """Return the orders ``decode_market`` emits for one slot and quantity.

    ``HIRE`` carries no quantity -- ``_parse_order`` returns it bare -- so one
    order hires exactly one hand and ``decode_market`` repeats the order by
    count. Everything else is a single order carrying its quantity, which is
    why the list is what this returns rather than an order.
    """
    if slot == HIRE_SLOT:
        return [["HIRE"]] * quantity
    if slot == LAND_SLOT:
        return [["BUY_LAND"]]
    verb, item = MARKET_SLOTS[slot]
    return [[verb, item, quantity]]


@pytest.mark.slow
@_needs_corpus
def test_the_market_mask_agrees_with_the_engine_on_real_observations() -> None:
    """Every (slot, bucket), pushed through `_process_market`, on real turns.

    A market order is legal here only if the engine fills it *completely*: a
    twelve-sack sell order against five sacks moves state, so a probe that only
    asked whether anything changed would call it accepted, and the mask would
    learn to queue orders that waste seven of the ten a turn is allowed.

    ``HIRE`` is compared against the engine capped at ``MAX_UNITS``, the crew
    ``encode_positions`` can represent. That cap is the mask's one deliberate
    departure from the engine and it is counted here rather than hidden, so a
    change to ``MAX_UNITS`` shows up as a disagreement instead of as nothing.
    """
    turns = _corpus_turns(_MARKET_STRIDE)
    permissive: list[str] = []
    strict: list[str] = []

    assert turns, "no archives on disk, so this proved nothing"
    for archive, observation, seat in turns:
        mask = market_mask(observation, seat)
        room = MAX_UNITS - unit_count(observation, seat)
        for slot in range(len(MARKET_SLOTS) + 2):
            for bucket in range(1, len(QUANTITIES)):
                quantity = quantity_of(bucket)
                if slot == LAND_SLOT and bucket > 1:
                    accepted = False
                else:
                    filled = _committed(observation, seat, slot, quantity)
                    wanted = 1 if slot == LAND_SLOT else quantity
                    accepted = filled == wanted
                    if slot == HIRE_SLOT and quantity > room:
                        accepted = False
                allowed = bool(mask[0, slot, bucket])
                where = f"{archive} seat {seat} slot {slot} bucket {bucket}"
                if allowed and not accepted:
                    permissive.append(
                        f"{where}: mask allows {quantity}, engine did not"
                    )
                if accepted and not allowed:
                    strict.append(f"{where}: engine filled {quantity}, mask forbids it")

    assert not permissive, f"{len(permissive)} over-permissive: {permissive[:5]}"
    assert not strict, f"{len(strict)} over-strict: {strict[:5]}"
