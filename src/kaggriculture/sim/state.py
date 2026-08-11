"""Lossless tensor representation of reference Kaggriculture state."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Sequence

import torch

from kaggriculture.constants import (
    ANIMALS,
    BOARD_SIZE,
    CROPS,
    EPISODE_STEPS,
    LAND_ORDER,
    MAX_MARKET_ORDERS_PER_TURN,
    PRODUCTS,
    SHED_CAPACITY,
    SHOPS,
    STARTING_MONEY,
    TURNS_PER_DAY,
)
from kaggriculture.learn.encoding import MAX_UNITS

CROP_NAMES = tuple(sorted(CROPS))
ANIMAL_NAMES = tuple(sorted(ANIMALS))
PRODUCT_NAMES = tuple(sorted(PRODUCTS))
SHED_NAMES = tuple(sorted(set(PRODUCTS) | set(ANIMALS)))
SHOP_NAMES = tuple(sorted(SHOPS))
STRUCTURE_NAMES = tuple(sorted({str(data["structure"]) for data in ANIMALS.values()}))
TILE_KINDS = ("EMPTY", "LOCKED", "WEED", "PLANT", *STRUCTURE_NAMES)
MAX_SHOP_INSTANCES = 8
RNG_WORDS_PER_DAY = 512
SEASON_DAYS = EPISODE_STEPS // TURNS_PER_DAY

_CROP_INDEX = {name: i + 1 for i, name in enumerate(CROP_NAMES)}
_ANIMAL_INDEX = {name: i + 1 for i, name in enumerate(ANIMAL_NAMES)}
_KIND_INDEX = {name: i for i, name in enumerate(TILE_KINDS)}
_PRODUCT_INDEX = {name: i for i, name in enumerate(PRODUCT_NAMES)}
_SHED_INDEX = {name: i for i, name in enumerate(SHED_NAMES)}
_SHOP_INDEX = {name: i for i, name in enumerate(SHOP_NAMES)}


class UnsupportedConfiguration(ValueError):  # noqa: N818 - public name is specified
    """Raised when a reference environment is not competition-compatible."""


@dataclass(frozen=True)
class SimState:
    """A batch of complete game states represented only by tensors."""

    kind: torch.Tensor
    occupant: torch.Tensor
    crop: torch.Tensor
    planted_day: torch.Tensor
    placed_day: torch.Tensor
    yield_units: torch.Tensor
    watered_today: torch.Tensor
    consecutive_unwatered: torch.Tensor
    fertilized_until_day: torch.Tensor
    max_lifespan_step: torch.Tensor
    fed_today: torch.Tensor
    cared_today: torch.Tensor
    consecutive_unfed: torch.Tensor
    fertilizer_available: torch.Tensor
    pending_care_bonus: torch.Tensor
    money: torch.Tensor
    hires_today: torch.Tensor
    quadrants: torch.Tensor
    hand_count: torch.Tensor
    unit_x: torch.Tensor
    unit_y: torch.Tensor
    alive: torch.Tensor
    shed: torch.Tensor
    seeds: torch.Tensor
    inv_count: torch.Tensor
    inv_seq: torch.Tensor
    inv_tick: torch.Tensor
    inventory: torch.Tensor
    prices: torch.Tensor
    shops: torch.Tensor
    shop_count: torch.Tensor
    shop_drain: torch.Tensor
    step: torch.Tensor
    day: torch.Tensor
    hour: torch.Tensor
    done: torch.Tensor
    reward: torch.Tensor
    seed: torch.Tensor
    rng_words: torch.Tensor

    @property
    def batch_size(self) -> int:
        """Return the number of environments in the batch."""
        return int(self.step.shape[0])

    def to(self, device: torch.device | str) -> SimState:
        """Move every state plane to ``device`` without changing its values."""
        return SimState(
            **{
                field.name: getattr(self, field.name).to(device)
                for field in fields(self)
            }
        )


def empty_state(batch: int, device: torch.device | str) -> SimState:
    """Allocate a canonical zero-valued state batch on ``device``."""
    tile = (batch, 2, BOARD_SIZE, BOARD_SIZE)
    farm = (batch, 2)
    units = (batch, 2, MAX_UNITS)
    return SimState(
        kind=torch.zeros(tile, dtype=torch.int8, device=device),
        occupant=torch.zeros(tile, dtype=torch.int8, device=device),
        crop=torch.zeros(tile, dtype=torch.int8, device=device),
        planted_day=torch.zeros(tile, dtype=torch.int16, device=device),
        placed_day=torch.zeros(tile, dtype=torch.int16, device=device),
        yield_units=torch.zeros(tile, dtype=torch.int16, device=device),
        watered_today=torch.zeros(tile, dtype=torch.bool, device=device),
        consecutive_unwatered=torch.zeros(tile, dtype=torch.int16, device=device),
        fertilized_until_day=torch.full(tile, -1, dtype=torch.int32, device=device),
        max_lifespan_step=torch.full(tile, -1, dtype=torch.int32, device=device),
        fed_today=torch.zeros(tile, dtype=torch.bool, device=device),
        cared_today=torch.zeros(tile, dtype=torch.bool, device=device),
        consecutive_unfed=torch.zeros(tile, dtype=torch.int16, device=device),
        fertilizer_available=torch.zeros(tile, dtype=torch.bool, device=device),
        pending_care_bonus=torch.zeros(tile, dtype=torch.int16, device=device),
        money=torch.zeros(farm, dtype=torch.int64, device=device),
        hires_today=torch.zeros(farm, dtype=torch.int16, device=device),
        quadrants=torch.ones(farm, dtype=torch.int8, device=device),
        hand_count=torch.zeros(farm, dtype=torch.int16, device=device),
        unit_x=torch.zeros(units, dtype=torch.int8, device=device),
        unit_y=torch.zeros(units, dtype=torch.int8, device=device),
        alive=torch.zeros(units, dtype=torch.bool, device=device),
        shed=torch.zeros((batch, 2, len(SHED_NAMES)), dtype=torch.int16, device=device),
        seeds=torch.zeros(
            (batch, 2, len(CROP_NAMES)), dtype=torch.int16, device=device
        ),
        inv_count=torch.zeros(
            (batch, 2, MAX_UNITS, len(SHED_NAMES)), dtype=torch.int16, device=device
        ),
        inv_seq=torch.zeros(
            (batch, 2, MAX_UNITS, len(SHED_NAMES)), dtype=torch.int32, device=device
        ),
        inv_tick=torch.zeros(units, dtype=torch.int32, device=device),
        inventory=torch.zeros(
            (batch, len(PRODUCT_NAMES)), dtype=torch.int64, device=device
        ),
        prices=torch.zeros(
            (batch, len(PRODUCT_NAMES)), dtype=torch.int64, device=device
        ),
        shops=torch.full(
            (batch, MAX_SHOP_INSTANCES), -1, dtype=torch.int8, device=device
        ),
        shop_count=torch.zeros(batch, dtype=torch.int8, device=device),
        shop_drain=torch.zeros(
            (batch, len(PRODUCT_NAMES)), dtype=torch.int16, device=device
        ),
        step=torch.zeros(batch, dtype=torch.int32, device=device),
        day=torch.zeros(batch, dtype=torch.int32, device=device),
        hour=torch.zeros(batch, dtype=torch.int32, device=device),
        done=torch.zeros(batch, dtype=torch.bool, device=device),
        reward=torch.zeros((batch, 2), dtype=torch.float64, device=device),
        seed=torch.zeros(batch, dtype=torch.int64, device=device),
        rng_words=torch.zeros(
            (batch, SEASON_DAYS, RNG_WORDS_PER_DAY), dtype=torch.uint32, device=device
        ),
    )


def _reference(value: Any) -> tuple[Sequence[Any], int, Any | None]:  # noqa: ANN401
    if hasattr(value, "state") and hasattr(value, "info"):
        return value.state, int(value.info.get("seed", 0)), value.configuration
    if isinstance(value, tuple) and len(value) == 2:
        return value[0], int(value[1]), None
    return value, 0, None


def _validate_configuration(configuration: Any | None) -> None:  # noqa: ANN401
    if configuration is None:
        return
    expected = {
        "episodeSteps": EPISODE_STEPS,
        "boardSize": BOARD_SIZE,
        "startingMoney": STARTING_MONEY,
        "maxMarketOrdersPerTurn": MAX_MARKET_ORDERS_PER_TURN,
        "turnsPerDay": TURNS_PER_DAY,
        "shedCapacity": SHED_CAPACITY,
        "weedSpawnChance": 0.005,
        "townShopUnlockInterval": 3,
        "townShopSellInterval": 4,
        "townCenterSellInterval": 24,
        "farmHandCostMult": 1,
        "marketParams": {},
    }
    for key, wanted in expected.items():
        actual = configuration.get(key)
        if actual != wanted:
            raise UnsupportedConfiguration(f"{key}={actual!r}; expected {wanted!r}")


def pack(
    states: Sequence[Any],
    device: torch.device | str = "cpu",
    *,
    materialize_rng: bool = True,
) -> SimState:
    """Pack reference environments into one lossless tensor batch."""
    result = empty_state(len(states), device)
    for b, source in enumerate(states):
        agents, episode_seed, configuration = _reference(source)
        _validate_configuration(configuration)
        if len(agents) != 2:
            raise ValueError(f"expected two agents, got {len(agents)}")
        obs = agents[0].observation
        result.seed[b] = episode_seed
        result.step[b] = int(obs.get("step", 0))
        result.day[b] = int(obs.day)
        result.hour[b] = int(obs.hour)
        result.done[b] = all(str(agent.status) == "DONE" for agent in agents)
        for p, agent in enumerate(agents):
            reward = agent.reward
            result.reward[b, p] = 0.0 if reward is None else float(reward)
            _pack_farm(result, b, p, obs.farms[p], agent.observation.private)
        for name, value in obs.market.inventory.items():
            result.inventory[b, _PRODUCT_INDEX[name]] = int(value)
        for name, value in obs.market.prices.items():
            result.prices[b, _PRODUCT_INDEX[name]] = int(value)
        shops = list(obs.town.unlocked_shops)
        if len(shops) > MAX_SHOP_INSTANCES:
            raise ValueError(f"{len(shops)} shops exceeds {MAX_SHOP_INSTANCES}")
        result.shop_count[b] = len(shops)
        for slot, name in enumerate(shops):
            result.shops[b, slot] = _SHOP_INDEX[name]
            drain = 2 if len(SHOPS[name]) == 1 else 1
            for product in SHOPS[name]:
                result.shop_drain[b, _PRODUCT_INDEX[product]] += drain
    # Imported lazily to keep the state schema independent of RNG internals.
    from kaggriculture.sim.rng import rng_words  # noqa: PLC0415

    if materialize_rng:
        result.rng_words.copy_(rng_words(result.seed))
    return result


def _pack_farm(
    result: SimState,
    b: int,
    p: int,
    farm: Any,  # noqa: ANN401
    private: Any,  # noqa: ANN401
) -> None:
    money = float(farm["money"])
    if not money.is_integer():
        raise ValueError(f"farm money must be integral, got {money}")
    result.money[b, p] = int(money)
    result.hires_today[b, p] = int(farm["hires_today"])
    result.quadrants[b, p] = len(farm["unlocked_quadrants"])
    positions = [farm["farmer"], *farm["hands"]]
    if len(positions) > MAX_UNITS:
        raise ValueError(f"{len(positions)} units exceeds MAX_UNITS={MAX_UNITS}")
    result.hand_count[b, p] = len(farm["hands"])
    for unit, (x, y) in enumerate(positions):
        result.unit_x[b, p, unit] = int(x)
        result.unit_y[b, p, unit] = int(y)
        result.alive[b, p, unit] = True
    for y, row in enumerate(farm["tiles"]):
        for x, tile in enumerate(row):
            _pack_tile(result, b, p, y, x, tile)
    for name, value in private["shed"].items():
        result.shed[b, p, _SHED_INDEX[name]] = int(value)
    for name, value in private["seeds"].items():
        result.seeds[b, p, _CROP_INDEX[name] - 1] = int(value)
    for unit, inventory in enumerate(private["inventories"]):
        tick = 0
        for name, value in inventory.items():
            tick += 1
            item = _SHED_INDEX[name]
            result.inv_count[b, p, unit, item] = int(value)
            result.inv_seq[b, p, unit, item] = tick
        result.inv_tick[b, p, unit] = tick


def _pack_tile(
    result: SimState,
    b: int,
    p: int,
    y: int,
    x: int,
    tile: Any,  # noqa: ANN401
) -> None:
    if tile is None:
        return
    if tile == "LOCKED":
        result.kind[b, p, y, x] = _KIND_INDEX["LOCKED"]
        return
    kind = str(tile["kind"])
    result.kind[b, p, y, x] = _KIND_INDEX[kind]
    result.yield_units[b, p, y, x] = int(tile.get("yield_units", 0))
    if kind == "PLANT":
        result.crop[b, p, y, x] = _CROP_INDEX[str(tile["crop"])]
        result.planted_day[b, p, y, x] = int(tile["planted_day"])
        result.watered_today[b, p, y, x] = bool(tile["watered_today"])
        result.consecutive_unwatered[b, p, y, x] = int(tile["consecutive_unwatered"])
        result.fertilized_until_day[b, p, y, x] = int(
            tile.get("fertilized_until_day", -1)
        )
        result.max_lifespan_step[b, p, y, x] = int(tile.get("max_lifespan_step", -1))
    elif "animal" in tile:
        result.occupant[b, p, y, x] = _ANIMAL_INDEX[str(tile["animal"])]
        result.placed_day[b, p, y, x] = int(tile["placed_day"])
        result.fed_today[b, p, y, x] = bool(tile["fed_today"])
        result.cared_today[b, p, y, x] = bool(tile["cared_today"])
        result.consecutive_unfed[b, p, y, x] = int(tile["consecutive_unfed"])
        result.fertilizer_available[b, p, y, x] = bool(tile["fertilizer_available"])
        result.pending_care_bonus[b, p, y, x] = int(tile["pending_care_bonus"])


def unpack(state: SimState, b: int, seat: int) -> dict[str, Any]:
    """Decode one seat's observation, preserving sparse dictionary order."""
    if not 0 <= b < state.batch_size or seat not in (0, 1):
        raise IndexError((b, seat))
    farms = [_unpack_farm(state, b, p) for p in range(2)]
    private = _unpack_private(state, b, seat)
    market = {
        "inventory": {
            name: int(state.inventory[b, i]) for i, name in enumerate(PRODUCT_NAMES)
        },
        "prices": {
            name: int(state.prices[b, i]) for i, name in enumerate(PRODUCT_NAMES)
        },
    }
    shops = [
        SHOP_NAMES[int(state.shops[b, slot])]
        for slot in range(int(state.shop_count[b]))
    ]
    observation = {
        "player": seat,
        "farms": farms,
        "private": private,
        "market": market,
        "town": {"unlocked_shops": shops},
        "day": int(state.day[b]),
        "hour": int(state.hour[b]),
    }
    # The Kaggle framework owns ``step`` and writes it onto agent 0 only.
    if seat == 0:
        observation = {"step": int(state.step[b]), **observation}
    return observation


def _unpack_farm(state: SimState, b: int, p: int) -> dict[str, Any]:
    hands = [
        [int(state.unit_x[b, p, unit]), int(state.unit_y[b, p, unit])]
        for unit in range(1, int(state.hand_count[b, p]) + 1)
    ]
    return {
        "money": float(state.money[b, p]),
        "tiles": [
            [_unpack_tile(state, b, p, y, x) for x in range(BOARD_SIZE)]
            for y in range(BOARD_SIZE)
        ],
        "farmer": [int(state.unit_x[b, p, 0]), int(state.unit_y[b, p, 0])],
        "hands": hands,
        "unlocked_quadrants": ["NW", *LAND_ORDER[: int(state.quadrants[b, p]) - 1]],
        "hires_today": int(state.hires_today[b, p]),
    }


def _unpack_tile(state: SimState, b: int, p: int, y: int, x: int) -> Any:  # noqa: ANN401
    kind = TILE_KINDS[int(state.kind[b, p, y, x])]
    if kind == "EMPTY":
        return None
    if kind == "LOCKED":
        return "LOCKED"
    tile: dict[str, Any] = {"kind": kind}
    if kind == "WEED":
        return tile
    tile["yield_units"] = int(state.yield_units[b, p, y, x])
    if kind == "PLANT":
        tile.update(
            crop=CROP_NAMES[int(state.crop[b, p, y, x]) - 1],
            planted_day=int(state.planted_day[b, p, y, x]),
            watered_today=bool(state.watered_today[b, p, y, x]),
            consecutive_unwatered=int(state.consecutive_unwatered[b, p, y, x]),
            max_lifespan_step=int(state.max_lifespan_step[b, p, y, x]),
            fertilized_until_day=int(state.fertilized_until_day[b, p, y, x]),
        )
    elif int(state.occupant[b, p, y, x]) > 0:
        tile.update(
            animal=ANIMAL_NAMES[int(state.occupant[b, p, y, x]) - 1],
            placed_day=int(state.placed_day[b, p, y, x]),
            consecutive_unfed=int(state.consecutive_unfed[b, p, y, x]),
            fed_today=bool(state.fed_today[b, p, y, x]),
            cared_today=bool(state.cared_today[b, p, y, x]),
            fertilizer_available=bool(state.fertilizer_available[b, p, y, x]),
            pending_care_bonus=int(state.pending_care_bonus[b, p, y, x]),
        )
    else:
        tile.pop("yield_units")
    return tile


def _unpack_private(state: SimState, b: int, p: int) -> dict[str, Any]:
    inventories = []
    for unit in range(int(state.hand_count[b, p]) + 1):
        ordered = sorted(
            (
                (
                    int(state.inv_seq[b, p, unit, item]),
                    name,
                    int(state.inv_count[b, p, unit, item]),
                )
                for item, name in enumerate(SHED_NAMES)
                if int(state.inv_seq[b, p, unit, item]) > 0
            )
        )
        inventories.append({name: count for _, name, count in ordered})
    return {
        "shed": {name: int(state.shed[b, p, i]) for i, name in enumerate(SHED_NAMES)},
        "seeds": {name: int(state.seeds[b, p, i]) for i, name in enumerate(CROP_NAMES)},
        "inventories": inventories,
    }
