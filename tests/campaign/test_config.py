"""The constants every other campaign module trusts."""

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    ANIMALS,
    CROPS,
    PRODUCTS,
    SHOPS,
)

from kaggriculture.campaign import config


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


def test_constants_match_the_spec_table() -> None:
    """Spec section 8: the campaign's constants, and only these."""
    assert config.SESSIONS == 8
    assert config.ROUNDS_PER_SESSION == 5
    # Sixteen seeds over sixteen opponents, which is the same 512 games the
    # gate cost at thirty-two seeds over eight. The budget went into
    # opponents because a rating's precision comes from the whole graph: one
    # more opponent is a whole new comparison, one more seed a slightly
    # tighter old one.
    assert config.GATE_SEEDS == 16
    assert config.GATE_OPPONENTS == 16
    assert 2 * config.GATE_SEEDS * config.GATE_OPPONENTS == 512
    # Every anchor is played by every gate, so they cannot fill the draw --
    # there has to be room for the contenders and the random remainder that
    # finds a counter.
    assert len(config.GATE_ANCHORS) + config.GATE_CONTENDERS < config.GATE_OPPONENTS
    # A bar with units in it. A rank had none, and promoted on a hair.
    assert config.PROMOTION_MARGIN > 0
    assert config.STAGNATION_SESSIONS == 40
    assert config.CODEX_MODEL == "gpt-5.6-luna"
    assert config.CODEX_FALLBACK_MODEL == "gpt-5.6-sol"


def test_the_deleted_constants_are_gone() -> None:
    """Islands, epochs, the call cap, crossover, weights, sandboxes, clocks.

    `SANDBOXES` went with the workspace a model used to be given: it gets a
    temporary directory holding one file now, and nothing under `run/` is a
    model's to write. The three clocks went because the work they bounded
    takes as long as it takes: the round cap only ever cut a call off before
    it had written anything. The one limit left is `harness.LATENCY_BUDGET`,
    which is not the campaign's -- it is half of Kaggle's own per-call
    `actTimeout`, and a program over it cannot compete.
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
        "ROUND_LIMIT_SECONDS",
        "SESSION_LIMIT_SECONDS",
        "GAME_LIMIT_SECONDS",
        # Went with the retirement rule the pool's `trim` replaced.
        "POOL_CAP",
        "RETIRE_THRESHOLD",
    ):
        assert not hasattr(config, name), name


def test_runtime_paths_live_under_run_campaign() -> None:
    """All runtime paths live under the run campaign directory."""
    for path in (
        config.LIVE.archive,
        config.LIVE.programs,
        config.LIVE.floor,
        config.LIVE.champions,
        config.LIVE.champion,
        config.LIVE.field,
        config.LIVE.state,
    ):
        assert config.LIVE.root in path.parents


def test_the_pool_file_is_not_a_sibling_of_the_run_directory() -> None:
    """The one file listing opponent paths lives away from everything else."""
    assert config.RUN not in config.POOL.parents
    assert config.POOL == config.OPPONENTS.parent / "campaign" / "pool.json"


def test_each_champion_keeps_its_own_file_apart_from_the_floor() -> None:
    """A pool of N champions must be able to hold N different programs."""
    assert config.LIVE.champions != config.LIVE.floor
    assert config.LIVE.floor not in config.LIVE.champions.parents
