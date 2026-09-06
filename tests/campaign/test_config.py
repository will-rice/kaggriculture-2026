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


def test_constants_match_the_spec_table() -> None:
    """Spec section 8: the campaign's constants, and only these."""
    assert config.SESSIONS == 8
    assert config.ROUNDS_PER_SESSION == 5
    assert config.ROUND_LIMIT_SECONDS == 3600
    # Derived, and asserted as the derivation: a round cap raised without the
    # session budget following would end every session after its first round.
    assert config.SESSION_LIMIT_SECONDS == 5 * 3600
    assert config.GAME_LIMIT_SECONDS == 120
    assert config.FAST_SEEDS == 4
    assert len(config.EXAM_SEEDS) == 64
    assert (config.DEEP_TOP_K, config.DEEP_CONCURRENCY) == (3, 2)
    assert (config.POOL_CAP, config.RETIRE_THRESHOLD) == (10, 0.95)
    assert config.STAGNATION_SESSIONS == 40
    assert config.CODEX_MODEL == "gpt-5.6-luna"
    assert config.CODEX_FALLBACK_MODEL == "gpt-5.6-sol"


def test_the_deleted_constants_are_gone() -> None:
    """Islands, epochs, the call cap, crossover, weights and the sandboxes.

    `SANDBOXES` went with the workspace a model used to be given: it gets a
    temporary directory holding one file now, and nothing under `run/` is a
    model's to write.
    """
    for name in (
        "SANDBOXES",
        "ISLANDS",
        "ISLAND_SIZE",
        "MIGRANTS",
        "MIGRATION_INTERVAL",
        "RESET_INTERVAL",
        "UCB_C",
        "EPOCH_INTERVAL",
        "DAILY_CALL_BUDGET",
        "CODEX_CONCURRENCY",
        "MUTATION_TIMEOUT_SECONDS",
        "CHECK_TIMEOUT_SECONDS",
        "WANDB_RUN_ID",
        "CROSS_PROBABILITY",
        "CHAMPION_WEIGHT",
        "WEAKNESS_CAP",
    ):
        assert not hasattr(config, name), name


def test_runtime_paths_live_under_run_campaign() -> None:
    """All runtime paths live under the run campaign directory."""
    for path in (
        config.ARCHIVE,
        config.PROGRAMS,
        config.FLOOR,
        config.CHAMPIONS,
        config.CHAMPION,
    ):
        assert config.RUN in path.parents


def test_the_pool_file_is_not_a_sibling_of_the_run_directory() -> None:
    """The one file listing opponent paths lives away from everything else."""
    assert config.RUN not in config.POOL.parents
    assert config.POOL == config.OPPONENTS.parent / "campaign" / "pool.json"


def test_each_champion_keeps_its_own_file_apart_from_the_floor() -> None:
    """A pool of N champions must be able to hold N different programs."""
    assert config.CHAMPIONS != config.FLOOR
    assert config.FLOOR not in config.CHAMPIONS.parents
