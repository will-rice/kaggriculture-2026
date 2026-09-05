"""The constants every other campaign module trusts."""

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    ANIMALS,
    CROPS,
    PRODUCTS,
    SHOPS,
)

from kaggriculture.campaign import config


def test_exam_seeds_are_the_sealed_block() -> None:
    """The deep evaluation plays these seeds and nothing else may."""
    assert config.EXAM_SEEDS == tuple(range(700_000, 700_064))


def test_item_order_matches_the_engine_port_enum() -> None:
    """sim.hpp's Item enum is WHEAT..FERTILIZER then GOOSE, COW, SHEEP."""
    assert config.ITEMS == [
        "WHEAT",
        "CARROT",
        "TOMATO",
        "STRAWBERRY",
        "MELON",
        "EGG",
        "MILK",
        "WOOL",
        "FERTILIZER",
        "GOOSE",
        "COW",
        "SHEEP",
    ]
    assert config.ITEMS[: len(PRODUCTS)] == PRODUCTS
    assert config.ITEMS[len(PRODUCTS) :] == list(ANIMALS)
    assert list(CROPS) == config.ITEMS[: len(CROPS)]


def test_shop_names_are_sorted_like_the_engine_unlocks_them() -> None:
    """The engine unlocks shops with ``rng.choice(sorted(SHOPS))``."""
    assert config.SHOP_NAMES == sorted(SHOPS)


def test_unit_ops_match_the_port_enum_order() -> None:
    """The index into these lists is the wire value sim.hpp's Op/MOp enums use."""
    assert config.UNIT_OPS == [
        "PASS",
        "NORTH",
        "SOUTH",
        "EAST",
        "WEST",
        "PICKUP",
        "DROP",
        "PLACE",
        "PLANT",
        "WATER",
        "HARVEST",
        "FERTILIZE",
        "DIG",
        "BUILD_COOP",
        "BUILD_PASTURE",
        "FEED",
        "COLLECT_FERTILIZER",
        "CARE",
    ]
    assert config.MARKET_OPS == [
        "NONE",
        "HIRE",
        "BUY_LAND",
        "BUY_SEED",
        "BUY_PRODUCT",
        "BUY_ANIMAL",
        "SELL",
    ]


def test_core_budget_leaves_headroom() -> None:
    """Eight cores stay free for codex sessions, the loop, and the box's tenants."""
    import os

    assert 1 <= config.CORE_BUDGET <= max(1, (os.cpu_count() or 1) - 8)


def test_fast_seed_range_never_touches_the_exam_block() -> None:
    """The fast seed range never overlaps with the exam seed block."""
    assert not set(config.FAST_SEED_RANGE) & set(config.EXAM_SEEDS)


def test_hyperparameters_match_the_spec_table() -> None:
    """Evolution hyperparameters match Spec §3."""
    assert (config.ISLANDS, config.ISLAND_SIZE) == (4, 12)
    assert (config.MIGRATION_INTERVAL, config.MIGRANTS, config.RESET_INTERVAL) == (
        10,
        2,
        40,
    )
    assert config.UCB_C == 0.5 and config.CROSS_PROBABILITY == 0.3
    assert (config.FAST_SEEDS, config.EPOCH_INTERVAL, config.DEEP_TOP_K) == (4, 25, 3)
    assert (
        config.CHAMPION_WEIGHT,
        config.POOL_CAP,
        config.RETIRE_THRESHOLD,
        config.WEAKNESS_CAP,
    ) == (0.20, 10, 0.95, 0.5)
    assert (
        config.CODEX_CONCURRENCY,
        config.MUTATION_TIMEOUT_SECONDS,
        config.DAILY_CALL_BUDGET,
    ) == (8, 600, 200)


def test_runtime_paths_live_under_run_campaign() -> None:
    """All runtime paths live under the run campaign directory."""
    for path in (
        config.ARCHIVE,
        config.PROGRAMS,
        config.SANDBOXES,
        config.POOL,
        config.EPOCHS,
        config.FLOOR,
        config.CALLS,
    ):
        assert config.RUN in path.parents
