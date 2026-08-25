"""Behavior contracts for the reactive hybrid market selector."""
# ruff: noqa: D103

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from kaggle_environments.envs.kaggriculture import kaggriculture as engine

from kaggriculture.action_codec import HIRE_SLOT, LAND_SLOT, MARKET_SLOTS, QUANTITIES
from kaggriculture.constants import MARKET_PARAMS, market_price
from kaggriculture.features import (
    ANIMAL_NAMES,
    CROP_NAMES,
    PRODUCT_NAMES,
    STRUCTURE_KINDS,
    EncodedObservation,
    encode_observation,
)
from kaggriculture.hybrid.config import HybridConfig, to_runtime
from kaggriculture.hybrid.market import select_market
from kaggriculture.hybrid.opening import OpeningTargets
from kaggriculture.hybrid.runtime import RuntimeConfig
from tests.feature_golden_generator import rich_observation


def runtime_config() -> RuntimeConfig:
    return to_runtime(HybridConfig.default())


def targets_fixture(
    *,
    hands_needed: int = 0,
    quadrants_needed: int = 0,
    crop_deficits: dict[str, int] | None = None,
    animal_deficits: dict[str, int] | None = None,
    protected_cash: int = 0,
    protected_inventory: dict[str, int] | None = None,
    liquidating: bool = False,
) -> OpeningTargets:
    return OpeningTargets(
        phase_start_day=0,
        hands_needed=hands_needed,
        quadrants_needed=quadrants_needed,
        crop_deficits={name: (crop_deficits or {}).get(name, 0) for name in CROP_NAMES},
        animal_deficits={
            name: (animal_deficits or {}).get(name, 0) for name in ANIMAL_NAMES
        },
        structure_deficits=dict.fromkeys(STRUCTURE_KINDS, 0),
        protected_cash=protected_cash,
        protected_inventory={
            name: (protected_inventory or {}).get(name, 0) for name in PRODUCT_NAMES
        },
        liquidating=liquidating,
    )


def encoded_fixture(
    *,
    day: int = 3,
    money: int = 3_000,
    shed: dict[str, int] | None = None,
    seeds: dict[str, int] | None = None,
    inventory_offsets: dict[str, int] | None = None,
    opponent_tomatoes: int = 0,
    unlocked_shops: list[str] | None = None,
) -> EncodedObservation:
    observation = deepcopy(rich_observation())
    observation["day"] = day
    observation["hour"] = 0
    observation["town"]["unlocked_shops"] = unlocked_shops or []
    farm = observation["farms"][0]
    farm["money"] = money
    farm["hands"] = []
    farm["hires_today"] = 0
    farm["tiles"] = [[None] * 10 for _ in range(10)]
    opponent = observation["farms"][1]
    opponent["tiles"] = [[None] * 10 for _ in range(10)]
    for index in range(opponent_tomatoes):
        tile = engine._new_plant("TOMATO", 0, 0)
        tile["yield_units"] = 4
        opponent["tiles"][index // 10][index % 10] = tile
    private = engine._new_private()
    private["inventories"] = [{}]
    for item, quantity in (shed or {}).items():
        private["shed"][item] = quantity
    for crop, quantity in (seeds or {}).items():
        private["seeds"][crop] = quantity
    observation["private"] = private
    for item, offset in (inventory_offsets or {}).items():
        inventory = int(MARKET_PARAMS[item]["I0"]) + offset
        observation["market"]["inventory"][item] = inventory
        observation["market"]["prices"][item] = market_price(item, inventory)
    return encode_observation(observation, 0)


def test_market_preserves_cash_before_buying_and_hiring() -> None:
    encoded = encoded_fixture(money=350)
    targets = targets_fixture(
        protected_cash=300,
        hands_needed=2,
        crop_deficits={"WHEAT": 5},
        animal_deficits={"GOOSE": 1},
    )

    selected = select_market(encoded, targets, runtime_config())

    assert selected.total_cost <= 50


def test_live_price_and_opponent_supply_change_the_sale_ranking() -> None:
    scarce = encoded_fixture(shed={"TOMATO": 8}, inventory_offsets={"TOMATO": -200})
    flooded = encoded_fixture(
        shed={"TOMATO": 8},
        inventory_offsets={"TOMATO": 800},
        opponent_tomatoes=8,
    )

    assert select_market(scarce, targets_fixture(), runtime_config()).indices != (
        select_market(flooded, targets_fixture(), runtime_config()).indices
    )


def test_final_liquidation_sells_available_stock_despite_normal_reserves() -> None:
    encoded = encoded_fixture(day=29, shed={"WHEAT": 12})
    selected = select_market(
        encoded,
        targets_fixture(protected_inventory={"WHEAT": 12}, liquidating=True),
        runtime_config(),
    )

    wheat_slot = MARKET_SLOTS.index(("SELL", "WHEAT"))
    assert selected.indices[wheat_slot] != 0


def test_late_sale_quantity_uses_configured_liquidation_urgency() -> None:
    encoded = encoded_fixture(
        day=25,
        shed={"WHEAT": 12},
        inventory_offsets={"WHEAT": 400},
    )
    urgent = runtime_config()
    patient = replace(
        urgent,
        market=replace(urgent.market, liquidation_urgency=0.0),
    )

    assert select_market(encoded, targets_fixture(), urgent).indices != (
        select_market(encoded, targets_fixture(), patient).indices
    )


def test_normal_sales_keep_the_protected_inventory_reserve() -> None:
    encoded = encoded_fixture(day=20, shed={"WHEAT": 12})
    selected = select_market(
        encoded,
        targets_fixture(protected_inventory={"WHEAT": 12}),
        runtime_config(),
    )

    wheat_slot = MARKET_SLOTS.index(("SELL", "WHEAT"))
    assert selected.indices[wheat_slot] == 0


def test_acquisitions_only_fill_live_opening_deficits() -> None:
    encoded = encoded_fixture(money=10_000)

    selected = select_market(encoded, targets_fixture(), runtime_config())

    assert selected.indices[HIRE_SLOT] == 0
    assert selected.indices[LAND_SLOT] == 0
    assert all(
        selected.indices[slot] == 0
        for slot, (verb, _) in enumerate(MARKET_SLOTS)
        if verb.startswith("BUY")
    )


def test_purchase_slots_share_the_remaining_shed_capacity() -> None:
    encoded = encoded_fixture(money=10_000, shed={"CARROT": 99})
    selected = select_market(
        encoded,
        targets_fixture(
            crop_deficits={"WHEAT": 4},
            animal_deficits={"GOOSE": 1},
            protected_inventory={"CARROT": 99},
        ),
        runtime_config(),
    )
    shed_purchases = sum(
        QUANTITIES[selected.indices[slot]]
        for slot, (verb, _) in enumerate(MARKET_SLOTS)
        if verb in {"BUY_PRODUCT", "BUY_ANIMAL"}
    )

    assert shed_purchases <= 1


def test_each_selected_bucket_is_legal_in_the_encoded_market_mask() -> None:
    encoded = encoded_fixture(
        money=2_500,
        shed={"TOMATO": 12},
        seeds={"WHEAT": 1},
        inventory_offsets={"TOMATO": -200},
    )
    targets = targets_fixture(
        hands_needed=3,
        quadrants_needed=1,
        crop_deficits={"WHEAT": 8},
        protected_cash=300,
    )

    selected = select_market(encoded, targets, runtime_config())

    assert len(selected.indices) == len(MARKET_SLOTS) + 2
    assert all(
        encoded.market_mask[slot][bucket]
        for slot, bucket in enumerate(selected.indices)
    )
    assert all(0 <= bucket < len(QUANTITIES) for bucket in selected.indices)
