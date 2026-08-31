"""Contracts for the one canonical market feature vector.

Every assertion below compares a column against the same fact read off the raw
engine observation the row was built from. A row that agrees with a second copy
of the encoder's own arithmetic would agree with its bugs too.
"""
# ruff: noqa: D103

from typing import Any

import pytest

from kaggriculture.action_codec import MARKET_SLOTS, QUANTITIES, bucket_of
from kaggriculture.constants import SHED_CAPACITY
from kaggriculture.features import (
    PRODUCT_NAMES,
    SHED_NAMES,
    SHOP_NAMES,
    EncodedObservation,
    encode_observation,
)
from kaggriculture.market_residual.features import market_feature_vector
from kaggriculture.market_residual.schema import ALLOWED_SLOTS, MarketFeatureSchema

NO_KAITO_ORDERS = (0,) * len(ALLOWED_SLOTS)


def test_integer_and_categorical_features_are_one_hot(
    encoded: EncodedObservation,
) -> None:
    row = market_feature_vector(encoded, NO_KAITO_ORDERS, None)

    assert sum(row.values[row.schema.day_slice]) == 1.0
    assert sum(row.values[row.schema.hour_slice]) == 1.0
    assert sum(row.values[row.schema.kaito_bucket_slices[0]]) == 1.0
    assert row.schema.sha256 == MarketFeatureSchema.current().sha256


def test_the_day_and_hour_columns_index_the_real_engine_turn(
    episode: list[Any],
) -> None:
    """Set the wrong column and the model learns the season backwards."""
    for step in (0, 137, 500, 600, 719):
        observation = episode[step][0]["observation"]
        row = market_feature_vector(
            encode_observation(observation, 0), NO_KAITO_ORDERS, None
        )
        day = row.values[row.schema.day_slice]
        hour = row.values[row.schema.hour_slice]

        assert day.index(1.0) == observation["day"], step
        assert hour.index(1.0) == observation["hour"], step
        assert sum(day) == 1.0 and sum(hour) == 1.0, step


def test_the_open_shop_bits_are_the_town_the_engine_reports(
    encoded: EncodedObservation, observation: dict[str, Any]
) -> None:
    row = market_feature_vector(encoded, NO_KAITO_ORDERS, None)
    open_shops = set(observation["town"]["unlocked_shops"])

    assert open_shops, "the fixture step must have opened some shops"
    assert open_shops != set(SHOP_NAMES), "and must leave some shut"
    for shop in SHOP_NAMES:
        assert row.block(f"shop_open:{shop}") == (float(shop in open_shops),), shop


def test_prices_and_inventories_are_the_live_market(
    encoded: EncodedObservation, observation: dict[str, Any]
) -> None:
    """Only the live market can price a sale; a stale copy prices yesterday's."""
    row = market_feature_vector(encoded, NO_KAITO_ORDERS, None)

    assert len(set(observation["market"]["prices"].values())) > 1
    for product in PRODUCT_NAMES:
        assert row.block(f"price:{product}") == (encoded.live_price(product),)
        assert row.block(f"inventory:{product}") == (encoded.live_inventory(product),)


def test_held_quantities_are_the_shed_the_engine_reports(
    encoded: EncodedObservation, observation: dict[str, Any]
) -> None:
    shed = observation["private"]["shed"]
    row = market_feature_vector(encoded, NO_KAITO_ORDERS, None)

    assert any(shed[product] > 0 for product in PRODUCT_NAMES), (
        "the fixture step must hold something sellable"
    )
    for product in PRODUCT_NAMES:
        held = row.block(f"held:{product}")
        assert held.index(1.0) == bucket_of(shed[product]), product
        assert QUANTITIES[held.index(1.0)] == shed[product], product
    free = SHED_CAPACITY - sum(shed[item] for item in SHED_NAMES)
    assert row.block("shed_free").index(1.0) == bucket_of(free)


def test_opponent_supply_is_the_visible_yield_on_their_farm(
    encoded: EncodedObservation,
) -> None:
    row = market_feature_vector(encoded, NO_KAITO_ORDERS, None)
    supply = {
        product: encoded.opponent_public_supply(product) for product in PRODUCT_NAMES
    }

    assert any(supply.values()), "the fixture step must see opponent supply"
    for product, visible in supply.items():
        column = row.block(f"opponent_supply:{product}")
        assert column.index(1.0) == bucket_of(visible), product


def test_the_kaito_proposal_lands_on_its_own_slot(
    encoded: EncodedObservation,
) -> None:
    """Off-by-one here trains the residual against a proposal nobody made."""
    proposal = tuple(index % len(QUANTITIES) for index in range(len(ALLOWED_SLOTS)))
    row = market_feature_vector(encoded, proposal, None)

    for position, slot in enumerate(ALLOWED_SLOTS):
        verb, item = MARKET_SLOTS[slot]
        column = row.block(f"kaito:{verb}:{item}")
        assert column.index(1.0) == proposal[position], (verb, item)
        assert row.values[row.schema.kaito_bucket_slices[position]] == column


def test_the_first_event_of_an_episode_has_no_deltas(
    encoded: EncodedObservation,
) -> None:
    row = market_feature_vector(encoded, NO_KAITO_ORDERS, None)

    assert row.block("has_previous_event") == (0.0,)
    assert row.block("delta_cash") == (0.0,)
    for product in PRODUCT_NAMES:
        assert row.block(f"delta_price:{product}") == (0.0,)
        assert row.block(f"delta_inventory:{product}") == (0.0,)


def test_deltas_measure_the_move_between_two_real_events(
    encoded: EncodedObservation, earlier_encoded: EncodedObservation
) -> None:
    earlier = market_feature_vector(earlier_encoded, NO_KAITO_ORDERS, None)
    row = market_feature_vector(encoded, NO_KAITO_ORDERS, earlier)

    assert row.block("has_previous_event") == (1.0,)
    assert row.block("delta_cash") == (encoded.money() - earlier_encoded.money(),)
    assert row.block("delta_cash") != (0.0,), "the fixture pair must move cash"
    moved = 0
    for product in PRODUCT_NAMES:
        price_move = encoded.live_price(product) - earlier_encoded.live_price(product)
        inventory_move = encoded.live_inventory(
            product
        ) - earlier_encoded.live_inventory(product)
        assert row.block(f"delta_price:{product}") == (price_move,), product
        assert row.block(f"delta_inventory:{product}") == (inventory_move,), product
        moved += price_move != 0.0
    assert moved >= 2, "the fixture pair must move at least two live prices"


def test_a_wrong_length_kaito_proposal_is_rejected(
    encoded: EncodedObservation,
) -> None:
    with pytest.raises(ValueError, match="Kaito buckets"):
        market_feature_vector(encoded, NO_KAITO_ORDERS[:-1], None)


def test_a_quantity_bucket_outside_the_vocabulary_is_rejected(
    encoded: EncodedObservation,
) -> None:
    with pytest.raises(ValueError, match=f"outside .0, {len(QUANTITIES)}."):
        market_feature_vector(encoded, (len(QUANTITIES),) + NO_KAITO_ORDERS[1:], None)


def test_a_previous_row_from_another_schema_is_rejected(
    encoded: EncodedObservation,
) -> None:
    """Two schemas in one episode is exactly the drift the digest exists to stop."""
    from kaggriculture.market_residual.features import MarketFeatureVector

    schema = MarketFeatureSchema.current()
    stale = MarketFeatureVector(
        schema=MarketFeatureSchema(schema.blocks[:-1]),
        values=(0.0,) * (schema.width - schema.blocks[-1][1]),
    )

    with pytest.raises(ValueError, match="another schema"):
        market_feature_vector(encoded, NO_KAITO_ORDERS, stale)
