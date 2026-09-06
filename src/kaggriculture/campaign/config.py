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
# produced, and the only thing that ends a session besides a round clearing the
# bar or failing. There is no clock on a round or on a session: a round runs the
# skills this login has installed, which take as long as they take, and a cap
# only ever cut one off before it had written anything.
ROUNDS_PER_SESSION = 5
# Seeds a round is scored on, played in both seats against every pool
# opponent. Four made a rate out of eight games, which is a standard error of
# about 0.18: enough noise that a lucky program tops the draw, becomes what
# the next session starts from, and passes its luck on rather than its
# quality. The candidate that led the fast ranking at -1.65 came 13 of 13 at
# -3.02 on the sealed block, a gap of 1.4 rating points that is what this
# looks like. Eight halves the standard error and doubles a round's scoring to
# about 90 seconds, against a codex call of two to five minutes.
FAST_SEEDS = 8
FAST_SEED_RANGE = range(1, 600_000)  # never the exam block
DEEP_TOP_K = 3
# Deep evaluations in flight at once. Each is about ten minutes of games.
DEEP_CONCURRENCY = 2
# Opponents the pool keeps: the top this many by Bradley-Terry rating, the
# weakest making way as champions out-rate them. Measured on 2026-09-06 over
# the campaign's own programs, a pool of the top six separated them almost
# twice as widely as all twelve did and at half the games -- a weak opponent
# every candidate already beats tells two candidates apart no better than a
# coin. Eight rather than six for headroom: the sixth-ranked agent today is
# one our programs beat outright, and losing it would cost the search the one
# rate it can move.
POOL_SIZE = 8
# Sessions without a promotion before a session starts from a program
# drawn from the database's top ten instead of the champion.
STAGNATION_SESSIONS = 40
# Calls in a row that may run to no verdict before the campaign stops. A call
# that never reached the model is nobody's failure and writes nothing, so
# without this the loop spins at full rate on an expired login, a withdrawn
# model or a provider outage, looking busy and producing nothing. Eight is
# one per worker: a single bad call is noise, eight is the machine.
NO_VERDICT_LIMIT = 8
# The first campaign spent nearly all its quota on gpt-6-astra and produced a
# champion that beats one of six opponents; everything since is about the
# setup, not the model, so which is the constraint is unknown. This block is
# a calibration run on a lighter model: a session takes 8-25 minutes whatever
# model runs it, so if luna reaches about the same field rate, the bottleneck
# is the setup and the quota should buy volume instead of capability. Not the
# smallest model on offer -- a session has to run the harness, read a JSONL
# game log and edit a several-hundred-line program, and the smallest models
# are likelier to fail at the mechanics than at the strategy, which would
# answer the wrong question. `mutate.validate_model` checks this against the
# login's own catalog at startup, because a typo here is hundreds of failed
# sessions discovered one at a time -- not hypothetical: this login accepts
# `gpt-6-astra` but refuses `gpt-5.6-astra` (probed 2026-09-05 on codex 0.153).
CODEX_MODEL = "gpt-5.6-luna"
# The model retried once, in the same directory, when the first model's call
# fails without a verdict -- a provider refusal or a codex crash -- so the
# round still gets a child. Astra answered "Selected model is at capacity"
# some of the time (two calls in the first live hour), which is why this
# exists at all.
CODEX_FALLBACK_MODEL = "gpt-5.6-sol"
# The pool's own pairings, kept between gates because they are constants: the
# opponents are fixed files, the games are seeded, and none of them draws on
# randomness. Derived state, not a committed artifact -- a pool that gains a
# champion has that champion's pairings measured and added, and one that loses
# an opponent simply stops asking for its row.
FIELD = RUN / "field.json"

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
