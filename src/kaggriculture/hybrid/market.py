"""Reactive, mask-legal market selection for the hybrid rule agent."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from kaggriculture.action_codec import (
    HIRE_SLOT,
    LAND_SLOT,
    MARKET_SLOTS,
    QUANTITIES,
)
from kaggriculture.constants import (
    ANIMALS,
    CROPS,
    LAND_PRICES,
    MARKET_PARAMS,
    SHED_CAPACITY,
    SHOPS,
    hire_cost,
    market_price,
)
from kaggriculture.features import MAX_UNITS, SHED_NAMES, SHOP_NAMES, EncodedObservation
from kaggriculture.hybrid.opening import OpeningTargets
from kaggriculture.hybrid.runtime import RuntimeConfig


@dataclass(frozen=True)
class MarketSelection:
    """One legal quantity-vocabulary index per canonical market slot."""

    indices: tuple[int, ...]
    total_cost: float


def _book_inventory(encoded: EncodedObservation, product: str) -> int:
    params = MARKET_PARAMS[product]
    return round(encoded.live_inventory(product) * int(params["T"]) + int(params["I0"]))


def _live_price(encoded: EncodedObservation, product: str) -> int:
    base = int(MARKET_PARAMS[product]["base"])
    return max(1, round((1.0 + encoded.live_price(product)) * base))


def _town_demand(encoded: EncodedObservation, product: str) -> float:
    open_shops = tuple(
        shop for shop in SHOP_NAMES if encoded.scalar(f"shop:{shop}") != 0.0
    )
    represented = sum(
        (2 if len(SHOPS[shop]) == 1 else 1)
        for shop in open_shops
        if product in SHOPS[shop]
    )
    total_open = round(encoded.scalar("town_shop_count") * len(SHOP_NAMES))
    duplicate_scale = total_open / len(open_shops) if open_shops else 1.0
    town_center = 0.0 if product == "FERTILIZER" else 1.0
    return town_center + represented * duplicate_scale


def _largest_legal_at_most(mask: Sequence[bool], wanted: int) -> int:
    candidates = (
        bucket
        for bucket, legal in enumerate(mask)
        if legal and QUANTITIES[bucket] <= wanted
    )
    return max(candidates, key=lambda bucket: QUANTITIES[bucket])


def _sale_quantity(
    encoded: EncodedObservation,
    targets: OpeningTargets,
    config: RuntimeConfig,
    product: str,
) -> int:
    stock = encoded.shed_count(product)
    reserve = targets.protected_inventory[product]
    available = stock if targets.liquidating else max(stock - reserve, 0)
    if available == 0:
        return 0
    if targets.liquidating:
        return available

    weights = config.market
    base = float(MARKET_PARAMS[product]["base"])
    price_ratio = _live_price(encoded, product) / base
    flooded_book = max(encoded.live_inventory(product), 0.0)
    opponent_supply = encoded.opponent_public_supply(product)
    supply_scale = opponent_supply / max(float(available), 1.0)
    reserve_share = reserve / max(float(stock), 1.0)
    days_to_liquidation = max(
        config.liquidation_start_day - encoded.day_count(),
        0,
    )
    liquidation_proximity = max(0.0, 1.0 - days_to_liquidation / 10.0)
    score = (
        weights.live_price * price_ratio
        + weights.town_demand * (_town_demand(encoded, product) / 4.0)
        - weights.opponent_supply * (flooded_book + supply_scale)
        - weights.reserve_penalty * reserve_share
        + weights.liquidation_urgency * liquidation_proximity
    )
    if score <= 0.0:
        return 0
    scale = max(
        weights.live_price + weights.town_demand + weights.opponent_supply,
        1.0,
    )
    scale += weights.liquidation_urgency * liquidation_proximity
    return min(available, max(1, math.ceil(available * min(score / scale, 1.0))))


def _acquisition_need(
    slot: int, encoded: EncodedObservation, targets: OpeningTargets
) -> int:
    if slot == HIRE_SLOT:
        return targets.hands_needed
    if slot == LAND_SLOT:
        return int(targets.quadrants_needed > 0)
    verb, item = MARKET_SLOTS[slot]
    if verb == "BUY_SEED":
        return max(targets.crop_deficits[item] - encoded.seed_count(item), 0)
    if verb == "BUY_ANIMAL":
        return targets.animal_deficits[item]
    if verb == "BUY_PRODUCT":
        reserve_need = max(
            targets.protected_inventory[item] - encoded.shed_count(item), 0
        )
        if item == "WHEAT":
            reserve_need = max(reserve_need, sum(targets.animal_deficits.values()))
        elif item == "FERTILIZER":
            reserve_need = max(reserve_need, sum(targets.crop_deficits.values()))
        return reserve_need
    return 0


def _quoted_cost(slot: int, quantity: int, encoded: EncodedObservation) -> float:
    if quantity == 0:
        return 0.0
    if slot == HIRE_SLOT:
        already = round(encoded.scalar("our_hires_today") * MAX_UNITS)
        return float(sum(hire_cost(already + offset) for offset in range(quantity)))
    if slot == LAND_SLOT:
        bought = encoded.quadrant_count() - 1
        return float(LAND_PRICES[bought])
    verb, item = MARKET_SLOTS[slot]
    if verb == "BUY_SEED":
        return float(quantity * int(CROPS[item]["seed"]))
    if verb == "BUY_ANIMAL":
        return float(quantity * int(ANIMALS[item]["cost"]))
    if verb == "BUY_PRODUCT":
        inventory = _book_inventory(encoded, item)
        return float(
            sum(
                market_price(item, inventory - offset)
                for offset in range(1, quantity + 1)
            )
        )
    return 0.0


def _affordable_bucket(
    slot: int,
    wanted: int,
    encoded: EncodedObservation,
    available_cash: float,
) -> tuple[int, float]:
    legal = encoded.market_mask[slot]
    candidates = sorted(
        (
            bucket
            for bucket, enabled in enumerate(legal)
            if enabled and QUANTITIES[bucket] <= wanted
        ),
        key=lambda bucket: QUANTITIES[bucket],
        reverse=True,
    )
    for bucket in candidates:
        cost = _quoted_cost(slot, QUANTITIES[bucket], encoded)
        if cost <= available_cash:
            return bucket, cost
    return 0, 0.0


def select_market(
    encoded: EncodedObservation,
    targets: OpeningTargets,
    config: RuntimeConfig,
) -> MarketSelection:
    """Select canonical market buckets using only live encoded facts."""
    cash = encoded.scalar("our_money") * 10_000.0
    shed_room = SHED_CAPACITY - sum(encoded.shed_count(item) for item in SHED_NAMES)
    selected: list[int] = []
    total_cost = 0.0
    for slot in range(len(MARKET_SLOTS) + 2):
        if slot < len(MARKET_SLOTS) and MARKET_SLOTS[slot][0] == "SELL":
            product = MARKET_SLOTS[slot][1]
            wanted = _sale_quantity(encoded, targets, config, product)
            bucket = _largest_legal_at_most(encoded.market_mask[slot], wanted)
            selected.append(bucket)
            shed_room += QUANTITIES[bucket]
            continue

        wanted = 0 if targets.liquidating else _acquisition_need(slot, encoded, targets)
        if slot < len(MARKET_SLOTS) and MARKET_SLOTS[slot][0] in {
            "BUY_PRODUCT",
            "BUY_ANIMAL",
        }:
            wanted = min(wanted, shed_room)
        spendable = max(cash - targets.protected_cash - total_cost, 0.0)
        bucket, cost = _affordable_bucket(slot, wanted, encoded, spendable)
        selected.append(bucket)
        total_cost += cost
        if slot < len(MARKET_SLOTS) and MARKET_SLOTS[slot][0] in {
            "BUY_PRODUCT",
            "BUY_ANIMAL",
        }:
            shed_room -= QUANTITIES[bucket]
    return MarketSelection(tuple(selected), total_cost)
