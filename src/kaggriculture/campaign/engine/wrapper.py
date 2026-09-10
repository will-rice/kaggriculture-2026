"""Drive the engine port from Python and speak the reference engine's dialect.

Rendering lives here rather than in C++ because the reference engine's
observation is a Python dict with a specific shape, and the differential test
compares dicts. Every shape below is read from ``kaggriculture.py``:
``_new_farm`` (farm), ``_new_plant`` and ``_new_animal`` and the
BUILD_COOP/BUILD_PASTURE branches of ``_apply_unit_action`` (tiles),
``_spawn_weeds`` (weeds), ``_new_private`` (private), ``_new_market``,
``_new_town``.
"""

import ctypes
import functools
from typing import Any

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    CROPS,
    LAND_ORDER,
    PRODUCTS,
)

from kaggriculture.campaign import config
from kaggriculture.campaign.engine import build

BOARD = 10
MAX_UNITS = 40
MAX_ORDERS = 16
MAX_SHOPS = 8
N_ITEMS = len(config.ITEMS)
N_CROPS = len(CROPS)
N_PRODUCTS = len(PRODUCTS)
INT16_MAX = 2**15 - 1
INT32_MAX = 2**31 - 1
ITEM_INDEX = {name: i for i, name in enumerate(config.ITEMS)}
UNIT_OP_INDEX = {name: i for i, name in enumerate(config.UNIT_OPS)}
MARKET_OP_INDEX = {name: i for i, name in enumerate(config.MARKET_OPS)}
# sim.hpp TileKind: T_EMPTY, T_LOCKED, T_WEED, T_COOP, T_PASTURE, T_PLANT
T_EMPTY, T_LOCKED, T_WEED, T_COOP, T_PASTURE, T_PLANT = range(6)
# sim.hpp packs the tile's booleans into one byte, in this order.
HAS_ANIMAL, WATERED, FED, CARED, FERTILIZER_AVAILABLE = 1, 2, 4, 8, 16
QUANTIFIED_OPS = {"PICKUP", "DROP", "PLACE"}
ITEM_OPS = {"PICKUP", "DROP", "PLACE", "PLANT"}
QUANTIFIED_ORDERS = {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"}


class PackedTile(ctypes.Structure):
    """Mirror of ``PackedTile`` in bridge.cpp."""

    _pack_ = 1
    _fields_ = [
        ("kind", ctypes.c_uint8),
        ("what", ctypes.c_uint8),
        ("flags", ctypes.c_uint8),
        ("consecutive_dry", ctypes.c_int8),
        ("yield_units", ctypes.c_int8),
        ("pending_care_bonus", ctypes.c_int8),
        ("planted_day", ctypes.c_int16),
        ("max_lifespan_step", ctypes.c_int32),
        ("fertilized_until_day", ctypes.c_int16),
    ]


class PackedFarm(ctypes.Structure):
    """Mirror of ``PackedFarm`` in bridge.cpp."""

    _pack_ = 1
    _fields_ = [
        ("money", ctypes.c_double),
        ("tiles", PackedTile * BOARD * BOARD),
        ("pos_x", ctypes.c_int8 * MAX_UNITS),
        ("pos_y", ctypes.c_int8 * MAX_UNITS),
        ("n_units", ctypes.c_int32),
        ("n_quadrants", ctypes.c_int32),
        ("hires_today", ctypes.c_int32),
        ("shed", ctypes.c_int16 * N_ITEMS),
        ("seeds", ctypes.c_int16 * N_CROPS),
        ("inv", ctypes.c_int16 * N_ITEMS * MAX_UNITS),
        ("inv_keys", ctypes.c_uint8 * N_ITEMS * MAX_UNITS),
        ("inv_nkeys", ctypes.c_uint8 * MAX_UNITS),
    ]


class PackedState(ctypes.Structure):
    """Mirror of ``PackedState`` in bridge.cpp."""

    _pack_ = 1
    _fields_ = [
        ("step", ctypes.c_int32),
        ("day", ctypes.c_int32),
        ("hour", ctypes.c_int32),
        ("done", ctypes.c_int32),
        ("n_shops", ctypes.c_int32),
        ("market_inventory", ctypes.c_int32 * N_PRODUCTS),
        ("market_prices", ctypes.c_int32 * N_PRODUCTS),
        ("shops", ctypes.c_uint8 * MAX_SHOPS),
        ("farms", PackedFarm * 2),
    ]


class PackedAction(ctypes.Structure):
    """Mirror of ``PackedAction`` in bridge.cpp."""

    _pack_ = 1
    _fields_ = [
        ("unit_ops", ctypes.c_uint8 * MAX_UNITS),
        ("unit_args", ctypes.c_uint8 * MAX_UNITS),
        ("unit_ns", ctypes.c_int16 * MAX_UNITS),
        ("n_units", ctypes.c_int32),
        ("order_ops", ctypes.c_uint8 * MAX_ORDERS),
        ("order_items", ctypes.c_uint8 * MAX_ORDERS),
        ("order_ns", ctypes.c_int32 * MAX_ORDERS),
        ("n_orders", ctypes.c_int32),
    ]


@functools.cache
def library() -> ctypes.CDLL:
    """Load the engine library once, by absolute path, with its signatures bound."""
    lib = ctypes.CDLL(str(build.build()))
    lib.kag_new.restype = ctypes.c_void_p
    lib.kag_new.argtypes = [ctypes.c_uint64, ctypes.c_int32]
    lib.kag_free.argtypes = [ctypes.c_void_p]
    lib.kag_clone.restype = ctypes.c_void_p
    lib.kag_clone.argtypes = [ctypes.c_void_p]
    lib.kag_step.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PackedAction),
        ctypes.POINTER(PackedAction),
    ]
    lib.kag_export.argtypes = [ctypes.c_void_p, ctypes.POINTER(PackedState)]
    return lib


def pack_units(packed: PackedAction, farmer: object, hands: object) -> None:
    """Encode the farmer's and the hands' ops the way ``_apply_unit_action`` reads them.

    An unknown op becomes PASS and an unknown item becomes an out-of-range id;
    the port no-ops both exactly where the reference falls through its branches.

    Args:
        packed: The action being filled in.
        farmer: The ``farmer`` entry of the action dict.
        hands: The ``hands`` entry of the action dict.
    """
    units: list[Any] = [farmer, *(hands if isinstance(hands, list) else [])][:MAX_UNITS]
    packed.n_units = len(units)
    for i, unit in enumerate(units):
        op = unit[0] if isinstance(unit, list) and unit else "PASS"
        packed.unit_ops[i] = UNIT_OP_INDEX.get(op, 0)
        if op in ITEM_OPS:
            packed.unit_args[i] = ITEM_INDEX.get(
                unit[1] if len(unit) > 1 else None, N_ITEMS
            )
        n = int(unit[2]) if op in QUANTIFIED_OPS and len(unit) > 2 else 1
        packed.unit_ns[i] = max(-1, min(n, INT16_MAX))


def pack_orders(packed: PackedAction, market: object) -> None:
    """Encode the market queue the way ``_parse_order`` reads it.

    A malformed order becomes an empty slot rather than being removed, because
    ``_process_market`` walks both players' queues by index: compacting one
    queue would pair its orders against the wrong opponent orders. Trailing
    empty slots are trimmed, since an index neither player fills does nothing.

    Args:
        packed: The action being filled in.
        market: The ``market`` entry of the action dict.
    """
    slots: list[Any] = (market if isinstance(market, list) else [])[:MAX_ORDERS]
    packed.n_orders = len(slots)
    for i, order in enumerate(slots):
        op = order[0] if isinstance(order, list) and order else "NONE"
        if op not in MARKET_OP_INDEX or op == "NONE":
            continue
        if op in QUANTIFIED_ORDERS:
            if len(order) < 3 or order[1] not in ITEM_INDEX:
                continue
            try:
                n = int(order[2])  # `_parse_order` drops what int() rejects
            except (TypeError, ValueError):
                continue
            if n <= 0:
                continue
            packed.order_items[i] = ITEM_INDEX[order[1]]
            packed.order_ns[i] = min(n, INT32_MAX)
        packed.order_ops[i] = MARKET_OP_INDEX[op]
    while packed.n_orders and not packed.order_ops[packed.n_orders - 1]:
        packed.n_orders -= 1


def pack_action(action: object) -> PackedAction:
    """Encode one player's action dict for the bridge.

    Args:
        action: The action the agent returned, in the reference engine's dialect.

    Returns:
        The packed action the bridge unpacks into ``kag::Action``.
    """
    fields = action if isinstance(action, dict) else {}
    packed = PackedAction()
    pack_units(packed, fields.get("farmer", ["PASS"]), fields.get("hands", []))
    pack_orders(packed, fields.get("market", []))
    return packed


def render_tile(tile: PackedTile) -> dict[str, Any] | str | None:
    """Return one tile as the reference engine stores it in ``farm["tiles"]``."""
    if tile.kind == T_EMPTY:
        return None
    if tile.kind == T_LOCKED:
        return "LOCKED"
    if tile.kind == T_WEED:
        return {"kind": "WEED"}
    if tile.kind == T_PLANT:
        return {
            "kind": "PLANT",
            "crop": config.ITEMS[tile.what],
            "planted_day": tile.planted_day,
            "watered_today": bool(tile.flags & WATERED),
            "consecutive_unwatered": tile.consecutive_dry,
            "yield_units": tile.yield_units,
            "max_lifespan_step": tile.max_lifespan_step,
            "fertilized_until_day": tile.fertilized_until_day,
        }
    structure = "COOP" if tile.kind == T_COOP else "PASTURE"
    # BUILD_COOP / BUILD_PASTURE and the escape branch of `_daily_refresh_animals`
    # all write a bare `{"kind": structure}`; the animal keys appear only once
    # `_new_animal` has run.
    if not tile.flags & HAS_ANIMAL:
        return {"kind": structure}
    return {
        "kind": structure,
        "animal": config.ITEMS[tile.what],
        "placed_day": tile.planted_day,
        "yield_units": tile.yield_units,
        "consecutive_unfed": tile.consecutive_dry,
        "fed_today": bool(tile.flags & FED),
        "cared_today": bool(tile.flags & CARED),
        "fertilizer_available": bool(tile.flags & FERTILIZER_AVAILABLE),
        "pending_care_bonus": tile.pending_care_bonus,
    }


def render_farm(farm: PackedFarm) -> dict[str, Any]:
    """Return the public farm dict `_new_farm` builds and the interpreter mutates."""
    return {
        "money": float(farm.money),
        "tiles": [
            [render_tile(farm.tiles[y][x]) for x in range(BOARD)] for y in range(BOARD)
        ],
        "farmer": [int(farm.pos_x[0]), int(farm.pos_y[0])],
        "hands": [
            [int(farm.pos_x[u]), int(farm.pos_y[u])] for u in range(1, farm.n_units)
        ],
        "unlocked_quadrants": ["NW", *LAND_ORDER[: farm.n_quadrants - 1]],
        "hires_today": int(farm.hires_today),
    }


def render_private(farm: PackedFarm) -> dict[str, Any]:
    """Return the private dict `_new_private` builds, one per player."""
    inventories = []
    for u in range(farm.n_units):
        keys = [farm.inv_keys[u][k] for k in range(farm.inv_nkeys[u])]
        inventories.append({config.ITEMS[i]: int(farm.inv[u][i]) for i in keys})
    return {
        "shed": {name: int(farm.shed[i]) for i, name in enumerate(config.ITEMS)},
        "seeds": {crop: int(farm.seeds[i]) for i, crop in enumerate(CROPS)},
        "inventories": inventories,
    }


def render(state: PackedState, player: int) -> dict[str, Any]:
    """Return the observation the reference engine hands ``player`` at this step.

    Only player 0 carries ``step``: the framework writes the step counter into
    the shared observation, and the interpreter copies just ``farms``,
    ``market``, ``town``, ``day`` and ``hour`` onto the others.
    """
    observation = {
        "player": player,
        "private": render_private(state.farms[player]),
        "market": {
            "inventory": {
                name: int(state.market_inventory[i]) for i, name in enumerate(PRODUCTS)
            },
            "prices": {
                name: int(state.market_prices[i]) for i, name in enumerate(PRODUCTS)
            },
        },
        "town": {
            "unlocked_shops": [
                config.SHOP_NAMES[state.shops[i]] for i in range(state.n_shops)
            ]
        },
        "farms": [render_farm(state.farms[0]), render_farm(state.farms[1])],
        "day": int(state.day),
        "hour": int(state.hour),
    }
    if player == 0:
        observation["step"] = int(state.step)
    return observation


class Engine:
    """One episode of the port, stepped by both players' action dicts."""

    def __init__(self, seed: int, episode_steps: int = 720) -> None:
        """Start an episode.

        Args:
            seed: The episode seed, as ``resolve_episode_seed`` would resolve it.
            episode_steps: Steps in the episode, as ``episodeSteps`` configures it.
        """
        self.library = library()
        self.sim = self.library.kag_new(seed, episode_steps)
        self.state = PackedState()
        self.export()

    def export(self) -> None:
        """Refresh the packed state from the simulation."""
        self.library.kag_export(self.sim, ctypes.byref(self.state))

    def step(self, action0: object, action1: object) -> None:
        """Advance one step with both players' action dicts."""
        a0, a1 = pack_action(action0), pack_action(action1)
        self.library.kag_step(self.sim, ctypes.byref(a0), ctypes.byref(a1))
        self.export()

    def observation(self, player: int) -> dict[str, Any]:
        """Return the observation the reference engine would hand ``player``."""
        return render(self.state, player)

    def bank(self, player: int) -> float:
        """Return ``player``'s money, the quantity the competition scores."""
        return float(self.state.farms[player].money)

    def fork(self) -> "Engine":
        """Return an independent episode at this exact position.

        A plan search tries many continuations from one position, and
        replaying from step 0 for each of them is the search's whole cost.
        The two engines share nothing after this returns: stepping one leaves
        the other where it was.
        """
        twin = Engine.__new__(Engine)
        twin.library = self.library
        twin.sim = self.library.kag_clone(self.sim)
        twin.state = PackedState()
        twin.export()
        return twin

    @property
    def done(self) -> bool:
        """Whether the episode has reached its final recorded step."""
        return bool(self.state.done)

    @property
    def step_index(self) -> int:
        """The step counter the reference engine reports as ``observation.step``."""
        return int(self.state.step)

    def __del__(self) -> None:
        """Free the simulation, tolerating a constructor that failed early."""
        if getattr(self, "sim", None):
            self.library.kag_free(self.sim)
            self.sim = None
