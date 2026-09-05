"""Every path and constant the campaign shares.

Opponent paths and the exam seeds live here and nowhere else, so the one
place a sandbox could learn either is a module it never imports.
"""

import os
from pathlib import Path

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    ANIMALS,
    PRODUCTS,
    SHOPS,
)

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "run" / "campaign"
OPPONENTS = Path("/data/kaggriculture/opponents")
EPISODES = Path("/data/kaggriculture/episodes")
ENGINE_LIBRARY = Path(__file__).parent / "engine" / "kaggriculture_engine.so"

# Measurement only. The deep evaluation plays these; nothing else may.
EXAM_SEEDS: tuple[int, ...] = tuple(range(700_000, 700_064))

# Cores the arena and the engine may use between them. Eight are left for
# codex sessions, the loop, and the box's own tenants.
CORE_BUDGET = max(1, (os.cpu_count() or 1) - 8)

# Item order is sim.hpp's `Item` enum: the nine products, then the animals.
ITEMS: list[str] = list(PRODUCTS) + list(ANIMALS)
# The engine unlocks shops with `rng.choice(sorted(SHOPS))`.
SHOP_NAMES: list[str] = sorted(SHOPS)
# sim.hpp's `Op` and `MOp` enums, in order; the index is the wire value.
UNIT_OPS: list[str] = [
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
MARKET_OPS: list[str] = [
    "NONE",
    "HIRE",
    "BUY_LAND",
    "BUY_SEED",
    "BUY_PRODUCT",
    "BUY_ANIMAL",
    "SELL",
]

# Spec §3. One iteration is one mutation per island.
ISLANDS = 4
ISLAND_SIZE = 12
MIGRATION_INTERVAL = 10
MIGRANTS = 2
RESET_INTERVAL = 40
UCB_C = 0.5
CROSS_PROBABILITY = 0.3
FAST_SEEDS = 4  # x both seats x every pool opponent
FAST_SEED_RANGE = range(1, 600_000)  # never the exam block
EPOCH_INTERVAL = 25
DEEP_TOP_K = 3
CHAMPION_WEIGHT = 0.20
POOL_CAP = 10
RETIRE_THRESHOLD = 0.95
WEAKNESS_CAP = 0.5
CODEX_CONCURRENCY = 8
MUTATION_TIMEOUT_SECONDS = 600
DAILY_CALL_BUDGET = 200

ARCHIVE = RUN / "archive.jsonl"
PROGRAMS = RUN / "programs"
SANDBOXES = RUN / "sandboxes"
POOL = RUN / "pool.json"
EPOCHS = RUN / "epochs.jsonl"
FLOOR = RUN / "floor" / "agent"
CALLS = RUN / "calls.jsonl"
