"""The constants every other campaign module trusts."""

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    ANIMALS,
    CROPS,
    PRODUCTS,
    SHOPS,
)

from kaggriculture.campaign import config, loop, mutate, pool, roster


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
    """Enough cores stay free for codex, the loop, and the box's other tenants.

    Sized against a measurement rather than a guess. On 2026-09-09 the loop drew
    60 cores against a budget of 56 while other tenants took 13.3, and `vmstat`
    reported 89 to 151 runnable on 64 cores with zero blocked and zero iowait --
    oversubscribed by half again, at sixty thousand context switches a second.
    The reservation has to cover the tenants and the codex sessions, and eight
    did not.
    """
    import os

    cores = os.cpu_count() or 1
    assert 1 <= config.CORE_BUDGET
    assert config.CORE_BUDGET <= max(1, cores - 24), (
        "the arena's budget has to leave room for the box's other tenants and "
        "the codex sessions, or the shortfall is paid as contention instead"
    )


def test_constants_match_the_spec_table() -> None:
    """Spec section 8: the campaign's constants, and only these."""
    # Two, not the spec's original eight. Eight was chosen when a round cost a
    # codex call and the only limit was the machine; the limit is now a five-hour
    # quota window, and eight sessions drained one in twenty minutes and then
    # walked into the wall mid-round eight times over -- a window spent on eight
    # abandoned rounds and no verdict. Concurrency cannot buy rounds the quota
    # does not hold; it only decides how many are unfinished when the wall lands,
    # and how few cores each has to measure with.
    assert config.SESSIONS == 2
    # Twelve consecutive attempts against one opponent, and a session works
    # through every opponent in the pool -- so a session's length is the pool's
    # rather than a constant, and there is no ROUNDS_PER_SESSION.
    assert loop.ROUNDS_PER_OPPONENT == 12
    assert not hasattr(config, "ROUNDS_PER_SESSION")
    # Sixteen seeds over twenty-four opponents: 768 games a candidate, up from
    # the 512 that sixteen opponents cost. The budget goes into opponents
    # because a rating's precision comes from the whole graph -- one more
    # opponent is a whole new comparison, one more seed a slightly tighter old
    # one -- and the eight added slots are what let every vendored agent be
    # played every gate rather than drawn by luck.
    assert loop.GATE_SEEDS == 16
    assert pool.GATE_OPPONENTS == 24
    assert 2 * loop.GATE_SEEDS * pool.GATE_OPPONENTS == 768
    # The draw has to hold every vendored opponent, the anchors that are not
    # themselves vendored, the leader and floor, and the contenders -- with
    # room left over for the random remainder that finds a counter. This is
    # the arithmetic that decides `GATE_OPPONENTS`, so it is asserted rather
    # than left to be rediscovered when the roster next grows.
    vendored = len(roster.TRAINING)
    fixed = vendored + len({*config.GATE_ANCHORS} - {*roster.TRAINING}) + 2
    assert fixed + pool.GATE_CONTENDERS < pool.GATE_OPPONENTS, (
        f"{vendored} vendored + anchors + leader/floor + "
        f"{pool.GATE_CONTENDERS} contenders does not fit in "
        f"{pool.GATE_OPPONENTS}: the draw would truncate the vendored set, "
        "which is the only cross-population evidence the gate gets"
    )
    # There is deliberately no promotion margin: a fixed bar could not
    # answer a noise level that varies, and `gate.promotion` asks for a
    # margin beyond twice its own error instead.
    assert not hasattr(config, "PROMOTION_MARGIN")
    assert loop.STAGNATION_SESSIONS == 40
    assert config.CODEX_MODEL == "gpt-5.6-luna"
    assert config.CODEX_FALLBACK_MODEL == "gpt-5.6-sol"
    assert mutate.CODEX_REASONING == "max"


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


def test_importing_config_pins_the_blas_pool_to_one_thread() -> None:
    """A worker must not carry a thread pool sized to the whole machine.

    Measured 2026-09-09: a bare interpreter holds one thread and `import numpy`
    alone takes it to sixty-four, because OpenBLAS sizes its pool to the box.
    Every evaluation worker imports numpy, so forty of them carried some two
    and a half thousand threads across sixty-four cores -- a run queue of 134
    and sixty thousand context switches a second, with each worker drawing 1.4
    cores where the budget had counted it as one.

    Run in a subprocess because the pin has to land before numpy is imported,
    and by the time this test runs the suite has long since imported it.
    """
    import subprocess
    import sys

    probe = (
        "import pathlib\n"
        "from kaggriculture.campaign import config\n"
        "import numpy\n"
        "print([l for l in pathlib.Path('/proc/self/status').read_text()"
        ".splitlines() if l.startswith('Threads:')][0].split()[1])\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert out.stdout.strip() == "1", (
        f"a worker importing numpy spawned {out.stdout.strip()} threads; the "
        "pool pin in config is not landing before numpy is imported"
    )
