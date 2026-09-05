"""Every path and constant the campaign shares.

Opponent paths and the exam seeds are written here and nowhere else. That
does not hide them -- ``codex exec -s workspace-write`` restricts writes, not
reads, and a round's directory sits in the system temporary tree with the
repository a walk away. What keeps a candidate from copying an opponent is
the doctrine in the prompt and the copy-check gate that rejects a candidate
resembling one; what this module buys is that nothing the loop composes has
to *print* a path to do its job.
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
# Where an opponent whose source predates the vendored drop still lives; the
# roster names one. Both roots are stated here so no other module spells out
# a path under /data.
AGENTS = Path("/data/kaggriculture/agents")
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

# One codex session per worker; eight fit the machine beside their
# evaluations. Spec section 8.
SESSIONS = 8
# Codex calls in one session, each continuing from the program the last one
# produced. A session ends at whichever of this count and
# `SESSION_LIMIT_SECONDS` comes first: five rounds of codex plus a minute of
# scoring each fits inside 1500 seconds.
ROUNDS_PER_SESSION = 5
FAST_SEEDS = 4  # x both seats x every pool opponent
FAST_SEED_RANGE = range(1, 600_000)  # never the exam block
DEEP_TOP_K = 3
# Deep evaluations in flight at once. Each is about ten minutes of games.
DEEP_CONCURRENCY = 2
POOL_CAP = 10
# A joining champion retires the opponent it beats at least this
# decisively, if the pool is full.
RETIRE_THRESHOLD = 0.95
# Sessions without a promotion before a session starts from a program
# drawn from the database's top ten instead of the champion.
STAGNATION_SESSIONS = 40
# Probed 2026-09-05 on codex 0.153: the ChatGPT login accepts `gpt-6-astra`
# ("our most capable model for complex, demanding work") but not `gpt-5.6-astra`.
CODEX_MODEL = "gpt-6-astra"
# Astra answers "Selected model is at capacity" some of the time (two calls
# in the first live hour). A call that fails on the first model is retried
# once on this one, in the same directory, so the round still gets a child.
CODEX_FALLBACK_MODEL = "gpt-5.6-sol"
# The session's own budget, checked between rounds; the round that overruns
# it is the last one. Measured on the first live iteration (2026-09-05).
SESSION_LIMIT_SECONDS = 1500
# One round's share of that budget, and the cap on a single codex call.
# Whatever `child.py` holds when it fires is what the loop scores, so the cap
# bounds a round's wall clock, not whether it yields a program.
ROUND_LIMIT_SECONDS = SESSION_LIMIT_SECONDS // ROUNDS_PER_SESSION
# Liveness only, never speed: a 720-turn game of rule-based policies takes
# well under a second, so one still running after two minutes is stuck.
GAME_LIMIT_SECONDS = 120

ARCHIVE = RUN / "archive.jsonl"
PROGRAMS = RUN / "programs"
# The current floor, the copy that ships, and every champion ever promoted.
# `FLOOR/main.py` is overwritten each promotion; `CHAMPIONS/<name>.py` is
# written once and is what the pool points at, so a pool of N champions holds
# N different programs rather than N references to the newest one.
FLOOR = RUN / "floor" / "agent"
CHAMPIONS = RUN / "champions"
SERVED = ROOT / "src" / "kaggriculture" / "served" / "main.py"
# The promoted champion, written atomically by the gate before it returns, so
# a kill between the promotion and the next `state.json` write cannot lose it.
CHAMPION = RUN / "champion.json"
# Metrics go to one wandb run per campaign, resumed across restarts by its
# fixed id. Starting a fresh campaign (a new `run/campaign`) means a new id
# here, or its curves land on top of the old run's.
WANDB_ENTITY = "will-rice"
WANDB_PROJECT = "kaggriculture-2026"
# The only file that lists opponent paths, kept out of `run/campaign/` so it
# is not a sibling of anything a codex call is given.
POOL = OPPONENTS.parent / "campaign" / "pool.json"
