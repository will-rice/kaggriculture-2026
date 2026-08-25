"""Pure canonical observation features shared by policies and Torch adapters.

This module is deliberately standard-library-only apart from the packaged game
rules.  It owns every interpretation of a raw reference observation; consumers
read the immutable result or convert it without parsing the mapping again.
"""

import struct
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from kaggriculture import action_codec
from kaggriculture.constants import (
    ANIMALS,
    BOARD_SIZE,
    CROPS,
    EPISODE_STEPS,
    LAND_ORDER,
    LAND_PRICES,
    MARKET_PARAMS,
    MOVES,
    SEASON_DAYS,
    SHED_CAPACITY,
    SHOPS,
    TURNS_PER_DAY,
    hire_cost,
    market_price,
    shed_access_tiles,
)
from kaggriculture.observation import Tile

ANIMAL_NAMES = action_codec.ANIMAL_NAMES
CROP_NAMES = action_codec.CROP_NAMES
HIRE_SLOT = action_codec.HIRE_SLOT
LAND_SLOT = action_codec.LAND_SLOT
MARKET_SLOTS = action_codec.MARKET_SLOTS
MAX_ORDERS = action_codec.MAX_ORDERS
MAX_TRANSFER = action_codec.MAX_TRANSFER
PRODUCT_NAMES = action_codec.PRODUCT_NAMES
QUANTITIES = action_codec.QUANTITIES
SHED_NAMES = action_codec.SHED_NAMES
UNIT_OPS = action_codec.UNIT_OPS
SHOP_NAMES = tuple(sorted(SHOPS))

BOARD = BOARD_SIZE
MAX_UNITS = 20
UNIT_CARRIED_SCALE = 16.0
CARE_BONUS_SCALE = 8.0
NO_DEATH_SCHEDULED = 1.0
SEED_SCALE = 32.0
CARRIED_SCALE = 32.0

STRUCTURE_KINDS = tuple(sorted({str(data["structure"]) for data in ANIMALS.values()}))
TILE_STATES = ("WEED", "LOCKED", "EMPTY") + STRUCTURE_KINDS
TILE_FEATURES = (
    "ACTIVE_TODAY",
    "CARED_TODAY",
    "CARE_BONUS",
    "DISTRESS",
    "YIELD_FRACTION",
    "BONUS_READY",
    "AGE",
    "LIFESPAN_LEFT",
)
UNIT_PLANES = ("FARMER", "HANDS", "CARRIED")

PLANE_NAMES = (
    *(f"crop:{name}" for name in CROP_NAMES),
    *(f"animal:{name}" for name in ANIMAL_NAMES),
    *(f"state:{name}" for name in TILE_STATES),
    *(f"feature:{name}" for name in TILE_FEATURES),
    *(f"unit:{name}" for name in UNIT_PLANES),
)
if len(set(PLANE_NAMES)) != len(PLANE_NAMES):
    raise AssertionError("canonical plane names must be unique")
PLANE_INDEX = {name: index for index, name in enumerate(PLANE_NAMES)}
PER_FARM_PLANES = len(PLANE_NAMES)
TILE_PLANES = 2 * PER_FARM_PLANES

_PHASE_NAMES = (
    "our_money",
    "opponent_money",
    "day",
    "hour",
    "step",
    "town_shop_count",
    "our_land",
    "opponent_land",
    "our_hands",
    "opponent_hands",
    "our_hires_today",
    "opponent_hires_today",
)
SCALAR_NAMES = (
    *(f"price:{name}" for name in PRODUCT_NAMES),
    *(f"inventory:{name}" for name in PRODUCT_NAMES),
    *_PHASE_NAMES,
    *(f"shed:{name}" for name in SHED_NAMES),
    *(f"seed:{name}" for name in CROP_NAMES),
    *(f"carried:{name}" for name in PRODUCT_NAMES),
    *(f"shop:{name}" for name in SHOP_NAMES),
)
if len(set(SCALAR_NAMES)) != len(SCALAR_NAMES):
    raise AssertionError("canonical scalar names must be unique")
SCALAR_INDEX = {name: index for index, name in enumerate(SCALAR_NAMES)}
SCALARS = len(SCALAR_NAMES)

ENCODED_FIELDS: dict[str, frozenset[str]] = {
    "observation": frozenset(
        {"day", "hour", "player", "farms", "market", "town", "private"}
    ),
    "farm": frozenset(
        {"tiles", "farmer", "hands", "money", "unlocked_quadrants", "hires_today"}
    ),
    "market": frozenset({"prices", "inventory"}),
    "town": frozenset({"unlocked_shops"}),
    "private": frozenset({"shed", "seeds", "inventories"}),
    "tile": frozenset(
        {
            "kind",
            "crop",
            "animal",
            "planted_day",
            "placed_day",
            "watered_today",
            "fed_today",
            "cared_today",
            "pending_care_bonus",
            "consecutive_unwatered",
            "consecutive_unfed",
            "yield_units",
            "fertilized_until_day",
            "fertilizer_available",
            "max_lifespan_step",
        }
    ),
}
NOT_ENCODED = frozenset({"step", "remainingOverageTime"})


class TooManyUnitsError(ValueError):
    """Raised when a crew cannot fit the fixed policy slots."""


def unit_count(observation: Mapping[str, Any], seat: int) -> int:
    """Return the farmer plus hands acting for ``seat``."""
    return 1 + len(observation["farms"][seat]["hands"])


@dataclass(frozen=True)
class EncodedObservation:
    """One immutable, fully interpreted policy observation."""

    board: tuple[tuple[tuple[float, ...], ...], ...]
    scalars: tuple[float, ...]
    positions: tuple[int, ...]
    unit_mask: tuple[tuple[bool, ...], ...]
    quantity_mask: tuple[tuple[bool, ...], ...]
    market_mask: tuple[tuple[bool, ...], ...]
    carried_items: tuple[float, ...]
    units: int
    unit_carried_items: tuple[tuple[float, ...], ...] = ()

    def scalar(self, name: str) -> float:
        """Return a normalized scalar by stable schema name."""
        return self.scalars[SCALAR_INDEX[name]]

    def plane(self, name: str, x: int, y: int, *, opponent: bool = False) -> float:
        """Return a spatial feature by stable schema name and engine position."""
        index = PLANE_INDEX[name] + (PER_FARM_PLANES if opponent else 0)
        return self.board[index][y][x]

    def money(self, *, opponent: bool = False) -> float:
        """Return normalized public money for either farm."""
        return self.scalar("opponent_money" if opponent else "our_money")

    def phase(self) -> tuple[float, float, float]:
        """Return normalized day, hour, and episode step."""
        return (self.scalar("day"), self.scalar("hour"), self.scalar("step"))

    def day_count(self) -> int:
        """Recover the current integer season day from its normalized scalar."""
        return round(self.scalar("day") * SEASON_DAYS)

    def hand_count(self) -> int:
        """Recover the current number of hired hands from its normalized scalar."""
        return round(self.scalar("our_hands") * 8)

    def quadrant_count(self) -> int:
        """Recover the number of currently unlocked owned quadrants."""
        return round(self.scalar("our_land") * (1 + len(LAND_ORDER)))

    def live_price(self, product: str) -> float:
        """Return the existing normalized live-price feature."""
        return self.scalar(f"price:{product}")

    def live_inventory(self, product: str) -> float:
        """Return the existing normalized market-inventory feature."""
        return self.scalar(f"inventory:{product}")

    def seed_count(self, crop: str) -> int:
        """Recover the exact seed count represented by its normalized scalar."""
        return round(self.scalar(f"seed:{crop}") * SEED_SCALE)

    def shed_count(self, item: str) -> int:
        """Recover the exact shed count represented by its normalized scalar."""
        return round(self.scalar(f"shed:{item}") * SHED_CAPACITY)

    def carried_count(self, product: str) -> int:
        """Recover the crew-wide carried product count."""
        return round(self.scalar(f"carried:{product}") * CARRIED_SCALE)

    def carried_item_count(self, item: str) -> int:
        """Recover the crew-wide carried count for any canonical shed item."""
        return round(self.carried_items[SHED_NAMES.index(item)] * CARRIED_SCALE)

    def unit_carried_count(self, unit: int, item: str) -> int:
        """Recover one acting unit's exact carried count for a canonical item."""
        if not 0 <= unit < self.units:
            raise IndexError(f"unit {unit} outside acting crew of {self.units}")
        if not self.unit_carried_items:
            return 0
        return round(
            self.unit_carried_items[unit][SHED_NAMES.index(item)] * CARRIED_SCALE
        )

    def unit_position(self, unit: int) -> tuple[int, int]:
        """Return an acting unit's ``(x, y)`` position."""
        if not 0 <= unit < self.units:
            raise IndexError(f"unit {unit} outside acting crew of {self.units}")
        flat = self.positions[unit]
        return flat % BOARD, flat // BOARD

    def crop_at(self, x: int, y: int, *, opponent: bool = False) -> str | None:
        """Return the crop identity on a tile, if any."""
        return next(
            (
                name
                for name in CROP_NAMES
                if self.plane(f"crop:{name}", x, y, opponent=opponent) != 0.0
            ),
            None,
        )

    def animal_at(self, x: int, y: int, *, opponent: bool = False) -> str | None:
        """Return the animal identity on a tile, if any."""
        return next(
            (
                name
                for name in ANIMAL_NAMES
                if self.plane(f"animal:{name}", x, y, opponent=opponent) != 0.0
            ),
            None,
        )

    def distress_at(self, x: int, y: int, *, opponent: bool = False) -> float:
        """Return the normalized crop/animal distress feature on a tile."""
        return self.plane("feature:DISTRESS", x, y, opponent=opponent)

    def crop_count(self, crop: str, *, opponent: bool = False) -> int:
        """Return how many public tiles carry ``crop``."""
        index = PLANE_INDEX[f"crop:{crop}"] + (PER_FARM_PLANES if opponent else 0)
        return round(sum(sum(row) for row in self.board[index]))

    def animal_count(self, animal: str, *, opponent: bool = False) -> int:
        """Return public placed animals plus all private owned stock when ours."""
        index = PLANE_INDEX[f"animal:{animal}"] + (PER_FARM_PLANES if opponent else 0)
        placed = round(sum(sum(row) for row in self.board[index]))
        if opponent:
            return placed
        return placed + self.shed_count(animal) + self.carried_item_count(animal)

    def structure_count(self, structure: str, *, opponent: bool = False) -> int:
        """Return built structures, including structures currently holding animals."""
        offset = PER_FARM_PLANES if opponent else 0
        bare_index = offset + PLANE_INDEX[f"state:{structure}"]
        bare = round(sum(sum(row) for row in self.board[bare_index]))
        occupied = sum(
            round(
                sum(
                    sum(row)
                    for row in self.board[offset + PLANE_INDEX[f"animal:{animal}"]]
                )
            )
            for animal in ANIMAL_NAMES
            if ANIMALS[animal]["structure"] == structure
        )
        return bare + occupied

    def opponent_public_supply(self, product: str) -> int:
        """Return currently visible held yield on the opponent's matching tiles."""
        animal = next(
            (name for name, data in ANIMALS.items() if data["product"] == product),
            None,
        )
        total = 0.0
        for y in range(BOARD):
            for x in range(BOARD):
                if product in CROPS and self.crop_at(x, y, opponent=True) == product:
                    total += self.plane(
                        "feature:YIELD_FRACTION", x, y, opponent=True
                    ) * int(CROPS[product]["max_yield"])
                elif animal and self.animal_at(x, y, opponent=True) == animal:
                    total += self.plane(
                        "feature:YIELD_FRACTION", x, y, opponent=True
                    ) * int(ANIMALS[animal]["max_held"])
        return round(total)


def _empty_board() -> list[list[list[float]]]:
    return [[[0.0] * BOARD for _ in range(BOARD)] for _ in range(TILE_PLANES)]


def _float32(value: float) -> float:
    """Round like assignment into the legacy float32 Torch tensors."""
    return struct.unpack("f", struct.pack("f", value))[0]


def _put(
    board: list[list[list[float]]], base: int, name: str, x: int, y: int, value: float
) -> None:
    board[base + PLANE_INDEX[name]][y][x] = _float32(value)


def _write_features(
    board: list[list[list[float]]],
    base: int,
    x: int,
    y: int,
    values: Sequence[float],
) -> None:
    for name, value in zip(TILE_FEATURES, values, strict=True):
        _put(board, base, f"feature:{name}", x, y, value)


def _write_tile(
    board: list[list[list[float]]],
    base: int,
    tile: Tile,
    x: int,
    y: int,
    day: int,
    step: int,
) -> None:
    if tile is None:
        _put(board, base, "state:EMPTY", x, y, 1.0)
        return
    if tile == "LOCKED":
        _put(board, base, "state:LOCKED", x, y, 1.0)
        return
    kind = tile["kind"]
    if kind == "WEED":
        _put(board, base, "state:WEED", x, y, 1.0)
        return
    if kind == "PLANT":
        crop = tile["crop"]
        death = tile["max_lifespan_step"]
        _put(board, base, f"crop:{crop}", x, y, 1.0)
        _write_features(
            board,
            base,
            x,
            y,
            (
                float(tile["watered_today"]),
                0.0,
                0.0,
                tile["consecutive_unwatered"] / 2.0,
                tile["yield_units"] / int(CROPS[crop]["max_yield"]),
                float(tile["fertilized_until_day"] >= day),
                (day - tile["planted_day"]) / SEASON_DAYS,
                NO_DEATH_SCHEDULED if death < 0 else (death - step) / EPISODE_STEPS,
            ),
        )
        return
    animal = tile.get("animal")
    if animal is None:
        _put(board, base, f"state:{kind}", x, y, 1.0)
        return
    _put(board, base, f"animal:{animal}", x, y, 1.0)
    _write_features(
        board,
        base,
        x,
        y,
        (
            float(tile["fed_today"]),
            float(tile["cared_today"]),
            tile["pending_care_bonus"] / CARE_BONUS_SCALE,
            tile["consecutive_unfed"] / 2.0,
            tile["yield_units"] / int(ANIMALS[animal]["max_held"]),
            float(tile["fertilizer_available"]),
            (day - tile["placed_day"]) / SEASON_DAYS,
            0.0,
        ),
    )


def _encode_board_values(
    observation: Mapping[str, Any], seat: int
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    day = observation["day"]
    step = day * TURNS_PER_DAY + observation["hour"]
    farms = observation["farms"]
    board = _empty_board()
    for block, farm in enumerate((farms[seat], farms[1 - seat])):
        base = block * PER_FARM_PLANES
        for y, row in enumerate(farm["tiles"]):
            for x, tile in enumerate(row):
                _write_tile(board, base, tile, x, y, day, step)
        farmer_x, farmer_y = farm["farmer"]
        _put(board, base, "unit:FARMER", farmer_x, farmer_y, 1.0)
        for x, y in farm["hands"]:
            plane = board[base + PLANE_INDEX["unit:HANDS"]]
            plane[y][x] = _float32(plane[y][x] + 1.0 / MAX_UNITS)
    carried = board[PLANE_INDEX["unit:CARRIED"]]
    for index, (x, y) in enumerate([farms[seat]["farmer"], *farms[seat]["hands"]]):
        carried[y][x] = _float32(
            carried[y][x]
            + sum(observation["private"]["inventories"][index].values())
            / UNIT_CARRIED_SCALE
        )
    return tuple(tuple(tuple(row) for row in plane) for plane in board)


def _encode_scalar_values(
    observation: Mapping[str, Any], seat: int
) -> tuple[float, ...]:
    market = observation["market"]
    farms = observation["farms"]
    ours, theirs = farms[seat], farms[1 - seat]
    private = observation["private"]
    unlocked_shops = observation["town"]["unlocked_shops"]
    carried: Counter[str] = Counter()
    for inventory in private["inventories"]:
        carried.update(inventory)
    values = [
        (market["prices"][item] - MARKET_PARAMS[item]["base"])
        / MARKET_PARAMS[item]["base"]
        for item in PRODUCT_NAMES
    ]
    values += [
        (market["inventory"][item] - MARKET_PARAMS[item]["I0"])
        / MARKET_PARAMS[item]["T"]
        for item in PRODUCT_NAMES
    ]
    values += [
        ours["money"] / 10_000.0,
        theirs["money"] / 10_000.0,
        observation["day"] / SEASON_DAYS,
        observation["hour"] / TURNS_PER_DAY,
        (observation["day"] * TURNS_PER_DAY + observation["hour"]) / EPISODE_STEPS,
        len(unlocked_shops) / len(SHOP_NAMES),
        len(ours["unlocked_quadrants"]) / (1 + len(LAND_ORDER)),
        len(theirs["unlocked_quadrants"]) / (1 + len(LAND_ORDER)),
        len(ours["hands"]) / 8.0,
        len(theirs["hands"]) / 8.0,
        ours["hires_today"] / MAX_UNITS,
        theirs["hires_today"] / MAX_UNITS,
    ]
    values += [private["shed"][item] / SHED_CAPACITY for item in SHED_NAMES]
    values += [private["seeds"][crop] / SEED_SCALE for crop in CROP_NAMES]
    values += [carried[item] / CARRIED_SCALE for item in PRODUCT_NAMES]
    values += [float(shop in unlocked_shops) for shop in SHOP_NAMES]
    return tuple(_float32(value) for value in values)


def _encode_carried_item_values(observation: Mapping[str, Any]) -> tuple[float, ...]:
    """Encode private carried shed items without changing legacy tensor scalars."""
    carried: Counter[str] = Counter()
    for inventory in observation["private"]["inventories"]:
        carried.update(inventory)
    return tuple(_float32(carried[item] / CARRIED_SCALE) for item in SHED_NAMES)


def _encode_unit_carried_item_values(
    observation: Mapping[str, Any], seat: int
) -> tuple[tuple[float, ...], ...]:
    """Encode exact per-unit inventories as named non-tensor policy facts."""
    units = unit_count(observation, seat)
    if units > MAX_UNITS:
        raise TooManyUnitsError(f"{units} acting units exceeds MAX_UNITS={MAX_UNITS}")
    inventories = observation["private"]["inventories"]
    return tuple(
        tuple(
            _float32(inventories[unit].get(item, 0) / CARRIED_SCALE)
            for item in SHED_NAMES
        )
        for unit in range(units)
    )


def _encode_position_values(
    observation: Mapping[str, Any], seat: int
) -> tuple[int, ...]:
    units = unit_count(observation, seat)
    if units > MAX_UNITS:
        raise TooManyUnitsError(f"{units} acting units exceeds MAX_UNITS={MAX_UNITS}")
    farm = observation["farms"][seat]
    values = [0] * MAX_UNITS
    for index, (x, y) in enumerate([farm["farmer"], *farm["hands"]]):
        values[index] = y * BOARD + x
    return tuple(values)


_OP_INDEX = {op: index for index, op in enumerate(UNIT_OPS)}
_SHED_ACCESS = frozenset(shed_access_tiles())
_MAX_QUANTITY = max(QUANTITIES)


def _unit_mask_values(
    observation: Mapping[str, Any], seat: int
) -> tuple[tuple[bool, ...], ...]:
    units = unit_count(observation, seat)
    if units > MAX_UNITS:
        raise TooManyUnitsError(f"{units} acting units exceeds MAX_UNITS={MAX_UNITS}")
    farm = observation["farms"][seat]
    private = observation["private"]
    rows = [[False] * len(UNIT_OPS) for _ in range(MAX_UNITS)]
    for row in rows:
        row[_OP_INDEX["PASS"]] = True
    for unit, position in enumerate([farm["farmer"], *farm["hands"]]):
        for op in _legal_ops(farm, private, unit, position, observation["day"]):
            rows[unit][_OP_INDEX[op]] = True
    return tuple(tuple(row) for row in rows)


def _legal_ops(
    farm: Mapping[str, Any],
    private: Mapping[str, Any],
    unit: int,
    position: list[int],
    day: int,
) -> set[str]:
    x, y = position
    legal = {"PASS"}
    for op, (dx, dy) in MOVES.items():
        if 0 <= x + dx < BOARD and 0 <= y + dy < BOARD:
            legal.add(op)
    inventory = private["inventories"][unit]
    if (x, y) in _SHED_ACCESS:
        if inventory:
            legal.add("DROP")
        legal.update(
            f"PICKUP:{item}" for item in SHED_NAMES if private["shed"][item] > 0
        )
        if sum(private["shed"].values()) < SHED_CAPACITY:
            legal.update(
                f"PLACE:{item}" for item in SHED_NAMES if inventory.get(item, 0) > 0
            )
    tile = farm["tiles"][y][x]
    if tile == "LOCKED":
        return legal
    return legal | _tile_ops(tile, private, inventory, day)


def _tile_ops(  # noqa: C901 - mirrors the reference engine's tile dispatch
    tile: Tile, private: Mapping[str, Any], inventory: Mapping[str, int], day: int
) -> set[str]:
    if tile is None:
        legal = {"BUILD_COOP", "BUILD_PASTURE"}
        legal.update(
            f"PLANT:{crop}" for crop in CROP_NAMES if private["seeds"][crop] > 0
        )
        return legal
    if "animal" in tile:
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
    if tile["kind"] == "PLANT":
        legal = {"DIG"}
        if not tile["watered_today"]:
            legal.add("WATER")
        if inventory.get("FERTILIZER", 0) > 0:
            legal.add("FERTILIZE")
        mature = day - tile["planted_day"] >= CROPS[tile["crop"]]["first_yield_day"]
        if tile["yield_units"] > 0 and mature:
            legal.add("HARVEST")
        return legal
    legal = {"DIG"}
    legal.update(
        f"PLACE:{animal}"
        for animal, data in ANIMALS.items()
        if tile["kind"] == data["structure"] and inventory.get(animal, 0) > 0
    )
    return legal


def _quantity_mask_values(
    observation: Mapping[str, Any], seat: int
) -> tuple[tuple[bool, ...], ...]:
    units = unit_count(observation, seat)
    if units > MAX_UNITS:
        raise TooManyUnitsError(f"{units} acting units exceeds MAX_UNITS={MAX_UNITS}")
    live = tuple(1 <= quantity <= MAX_TRANSFER for quantity in QUANTITIES)
    dead = tuple(quantity == 1 for quantity in QUANTITIES)
    return tuple(live if index < units else dead for index in range(MAX_UNITS))


def _market_mask_values(
    observation: Mapping[str, Any], seat: int
) -> tuple[tuple[bool, ...], ...]:
    farm = observation["farms"][seat]
    private = observation["private"]
    market = observation["market"]
    rows = [[False] * len(QUANTITIES) for _ in range(len(MARKET_SLOTS) + 2)]
    for row in rows:
        row[0] = True
    money = farm["money"]
    for slot, (verb, item) in enumerate(MARKET_SLOTS):
        fillable = _fillable(verb, item, money, private, market)
        for bucket in range(1, len(QUANTITIES)):
            rows[slot][bucket] = action_codec.quantity_of(bucket) <= fillable
    hires = _hireable(farm, money, MAX_UNITS - unit_count(observation, seat))
    for bucket in range(1, len(QUANTITIES)):
        rows[HIRE_SLOT][bucket] = action_codec.quantity_of(bucket) <= hires
    bought = len(farm["unlocked_quadrants"]) - 1
    rows[LAND_SLOT][1] = bought < len(LAND_ORDER) and money >= LAND_PRICES[bought]
    return tuple(tuple(row) for row in rows)


def _fillable(
    verb: str,
    item: str,
    money: float,
    private: Mapping[str, Any],
    market: Mapping[str, Any],
) -> int:
    if verb == "SELL":
        return private["shed"][item]
    if verb == "BUY_SEED":
        return int(money // int(CROPS[item]["seed"]))
    room = SHED_CAPACITY - sum(private["shed"].values())
    if verb == "BUY_ANIMAL":
        return min(int(money // int(ANIMALS[item]["cost"])), room)
    inventory = market["inventory"][item]
    params = market.get("params")
    filled = 0
    while filled < min(_MAX_QUANTITY, room):
        price = market_price(item, inventory - 1 - filled, params)
        if money < price:
            break
        money -= price
        filled += 1
    return filled


def _hireable(farm: Mapping[str, Any], money: float, room: int) -> int:
    hired = 0
    while hired < min(room, MAX_ORDERS):
        cost = hire_cost(farm["hires_today"] + hired)
        if money < cost:
            break
        money -= cost
        hired += 1
    return hired


def encode_observation(observation: Mapping[str, Any], seat: int) -> EncodedObservation:
    """Interpret a raw observation exactly once into immutable policy features."""
    if seat not in (0, 1):
        raise ValueError(f"seat must be 0 or 1, got {seat}")
    units = unit_count(observation, seat)
    return EncodedObservation(
        board=_encode_board_values(observation, seat),
        scalars=_encode_scalar_values(observation, seat),
        positions=_encode_position_values(observation, seat),
        unit_mask=_unit_mask_values(observation, seat),
        quantity_mask=_quantity_mask_values(observation, seat),
        market_mask=_market_mask_values(observation, seat),
        carried_items=_encode_carried_item_values(observation),
        units=units,
        unit_carried_items=_encode_unit_carried_item_values(observation, seat),
    )
