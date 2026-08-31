"""Contracts for the frozen market feature layout and its seed boundaries.

Two things in here are load-bearing for every later task. The layout digest is
part of run identity: a model trained under one column order and served under
another does not fail, it plays a different game. And the seed banks decide
which episodes a promotion decision is allowed to read; a training bank that
reaches into the held-out gate turns the exam into part of the search.
"""
# ruff: noqa: D103

import pytest

from kaggriculture.action_codec import HIRE_SLOT, LAND_SLOT, MARKET_SLOTS
from kaggriculture.features import EncodedObservation
from kaggriculture.market_residual.features import market_feature_vector
from kaggriculture.market_residual.schema import (
    ALLOWED_SLOTS,
    CONSUMED_PROMOTION_SEEDS,
    HELD_OUT_GATE_SEEDS,
    SEED_BANKS,
    MarketFeatureSchema,
    check_seed_banks,
    validate_seed_bank,
)
from kaggriculture.search.scripts.hillclimb import SEED_POOL
from kaggriculture.search.scripts.holdout import GATE_SEEDS

# The layout this project trains and serves under. It changes only when the
# columns change, and when it changes every artifact built under the old one
# has to be retrained, not re-labelled.
SCHEMA_SHA256 = "e84619fe8e4a8d4b4fafc3df8653a0947da972f8aeafb0488197f58c5689e200"


def test_only_commodity_slots_are_learnable() -> None:
    assert {MARKET_SLOTS[index][0] for index in ALLOWED_SLOTS} == {
        "SELL",
        "BUY_PRODUCT",
    }


def test_every_commodity_slot_is_learnable() -> None:
    """A missing SELL slot would be invisible: it just never gets replaced."""
    assert len(ALLOWED_SLOTS) == sum(
        1 for verb, _item in MARKET_SLOTS if verb in {"SELL", "BUY_PRODUCT"}
    )
    assert HIRE_SLOT not in ALLOWED_SLOTS
    assert LAND_SLOT not in ALLOWED_SLOTS


def test_block_names_are_unique_and_spans_tile_the_row() -> None:
    schema = MarketFeatureSchema.current()
    names = [name for name, _width in schema.blocks]

    assert len(set(names)) == len(names)
    end = 0
    for name, width in schema.blocks:
        span = schema.span(name)
        assert (span.start, span.stop) == (end, end + width), name
        end = span.stop
    assert end == schema.width


def test_the_encoder_emits_exactly_the_declared_width(
    encoded: EncodedObservation,
) -> None:
    row = market_feature_vector(encoded, (0,) * len(ALLOWED_SLOTS), None)

    assert len(row.values) == MarketFeatureSchema.current().width
    assert row.schema.sha256 == MarketFeatureSchema.current().sha256


def test_the_layout_digest_is_pinned() -> None:
    assert MarketFeatureSchema.current().sha256 == SCHEMA_SHA256


def test_the_digest_reads_names_and_widths_not_object_identity() -> None:
    """A digest over a Python repr would miss a rename and see a rebuild."""
    schema = MarketFeatureSchema.current()
    last_name, last_width = schema.blocks[-1]
    renamed = MarketFeatureSchema(
        schema.blocks[:-1] + ((last_name + "_x", last_width),)
    )
    widened = MarketFeatureSchema(schema.blocks[:-1] + ((last_name, last_width + 1),))
    reordered = MarketFeatureSchema(schema.blocks[-2:] + schema.blocks[:-2])

    assert MarketFeatureSchema(schema.blocks).sha256 == schema.sha256
    assert len({schema.sha256, renamed.sha256, widened.sha256, reordered.sha256}) == 4


def test_the_declared_banks_are_the_ones_the_plan_fixed() -> None:
    assert {
        name: (min(seeds), max(seeds), len(seeds)) for name, seeds in SEED_BANKS.items()
    } == {
        "counterfactual_train": (860_000, 860_511, 512),
        "counterfactual_temporal": (861_000, 861_127, 128),
        "online_train": (870_000, 871_023, 1024),
        "online_gate": (872_000, 872_063, 64),
        "temporal_frontier": (880_000, 880_127, 128),
        "export_determinism": (895_000, 895_007, 8),
        "promotion": (890_000, 890_127, 128),
    }


def test_the_banks_are_pairwise_disjoint() -> None:
    for name, seeds in SEED_BANKS.items():
        for other, other_seeds in SEED_BANKS.items():
            if other != name:
                assert not set(seeds) & set(other_seeds), f"{name} touches {other}"


def test_overlapping_banks_are_refused() -> None:
    """The rule the import-time check runs, exercised where it can go red."""
    with pytest.raises(RuntimeError, match="must be disjoint"):
        check_seed_banks({"train": (870_000, 870_001), "gate": (870_001,)})


def test_a_bank_built_on_a_spent_seed_is_refused() -> None:
    with pytest.raises(RuntimeError, match="consumed promotion seeds"):
        check_seed_banks({"train": (840_000,)})
    with pytest.raises(RuntimeError, match="held-out gate seeds"):
        check_seed_banks({"train": (GATE_SEEDS[-1],)})


def test_the_declared_banks_pass_their_own_check() -> None:
    check_seed_banks(SEED_BANKS)


def test_no_bank_touches_the_route_search_or_its_exam() -> None:
    """The 700000 bank is the exam every promotion this project made was graded on."""
    assert HELD_OUT_GATE_SEEDS == frozenset(GATE_SEEDS)
    for name, seeds in SEED_BANKS.items():
        assert not set(seeds) & frozenset(GATE_SEEDS), name
        assert not any(seed in SEED_POOL for seed in seeds), name


def test_consumed_promotion_seeds_are_rejected() -> None:
    with pytest.raises(ValueError, match="consumed promotion seed"):
        validate_seed_bank("counterfactual_train", (840000,))


def test_every_consumed_promotion_seed_is_rejected() -> None:
    assert CONSUMED_PROMOTION_SEEDS == frozenset(range(840_000, 840_128))
    for seed in sorted(CONSUMED_PROMOTION_SEEDS):
        with pytest.raises(ValueError, match="consumed promotion seed"):
            validate_seed_bank("online_train", (seed,))


def test_a_held_out_gate_seed_is_rejected() -> None:
    with pytest.raises(ValueError, match="held-out gate seed"):
        validate_seed_bank("online_train", (GATE_SEEDS[0],))


def test_a_seed_from_another_bank_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside the online_train bank"):
        validate_seed_bank("online_train", (860_000,))


def test_an_unknown_bank_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown seed bank"):
        validate_seed_bank("counterfactual", (860_000,))


def test_every_declared_bank_validates_itself() -> None:
    for name, seeds in SEED_BANKS.items():
        assert validate_seed_bank(name, seeds) == seeds
