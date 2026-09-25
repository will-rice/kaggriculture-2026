"""Every path and constant the campaign shares.

Opponent paths are written here and nowhere else. That
does not hide them -- ``codex exec -s workspace-write`` restricts writes, not
reads, and a round's directory sits in the system temporary tree with the
repository a walk away. What keeps a candidate from copying an opponent is
the doctrine in the prompt and the copy-check gate that rejects a candidate
resembling one; what this module buys is that nothing the loop composes has
to *print* a path to do its job.
"""

import dataclasses
import os
from pathlib import Path

# One BLAS thread per process, set before anything can import numpy.
#
# Measured 2026-09-09: a bare interpreter holds one thread and `import numpy`
# alone takes it to sixty-four, one per core, because OpenBLAS sizes its pool
# to the machine. Every evaluation worker imports numpy -- the agents do, if
# nothing else -- so forty workers were carrying about two thousand five
# hundred threads across sixty-four cores. `vmstat` showed sixty thousand
# context switches a second and a run queue of 134, and each worker process was
# drawing 1.4 cores on average and as much as 3.3, which is why a budget
# counting *processes* could not keep the box from being oversubscribed.
#
# Nothing here wants BLAS parallelism. A game is one sequential simulation and
# the parallelism that matters is across games, which the process pool already
# provides; threads inside a worker only contend with the other workers. This
# is set in `config` because every campaign module imports it first, and it
# must land before numpy is imported rather than after, since OpenBLAS reads
# the environment once at import and never again. Workers are spawned fresh
# with `max_tasks_per_child=1`, so they inherit this and read it on their own
# import.
for _pool in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_pool, "1")

from kaggle_environments.envs.kaggriculture.kaggriculture import (  # noqa: E402 - the thread pins above must land before anything imports numpy
    ANIMALS,
    PRODUCTS,
    SHOPS,
)

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "run" / "campaign"
OPPONENTS = Path("/data/kaggriculture/opponents")
# The one database. The nightly extraction writes the recorded ladder into it
# and the loop writes every game it plays into it, because the questions worth
# asking span both -- is this lineage converging on what the field does, or
# somewhere else -- and that is only askable while both are rows in one table.
# `episodes.source` says which writer owns a row, so the rebuild replaces the
# ladder and never touches a game the campaign played.
# Where the one games database answers. ClickHouse on localhost, started by
# the repository's `docker-compose.yml`: the recorded ladder and every game the
# campaign plays, in one store, because the questions worth asking span both.
GAMES_URL = os.environ.get("KAGGRICULTURE_GAMES_URL", "http://127.0.0.1:8123")
# The database inside that server. Here rather than in `games` because
# `dataset` names it too, and the two modules importing each other to agree
# on a string is how they come to disagree on one.
GAMES_DB = "games"
# Skills copied into every round's workspace. They live in the repository, beside
# the code whose schema they describe, rather than in the host's codex
# configuration: a skill that documents `browse`'s tables and is kept somewhere
# the tests cannot reach is a skill that goes stale the first time a column is
# renamed, and goes stale silently.
SKILLS = ROOT / ".agents" / "skills"
EPISODES = Path("/data/kaggriculture/episodes")

# Cores the arena and the engine may use between them.
#
# Twenty-four are left over, not eight. Measured 2026-09-09 on the 64-core box:
# the loop was drawing 60 cores against a budget of 56 while the box's other
# tenants took 13.3 and 945 processes of their own. `vmstat` showed 89 to 151
# runnable against 64 cores, zero blocked, zero iowait, and around sixty
# thousand context switches a second -- entirely CPU-bound and oversubscribed
# by half again, so a large share of the machine was spent scheduling rather
# than playing games.
#
# The old reservation was not wrong when it was written, it was sized for a box
# we had to ourselves. Eight cores cannot cover thirteen of other people's work
# plus eight codex sessions, and the shortfall comes out of the arena either
# way -- as contention rather than as a smaller pool, which is the same cost
# paid less efficiently.
CORE_BUDGET = max(1, (os.cpu_count() or 1) - 24)

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
# The five plantable ones: `tables.rs`'s `Crop` enum, which is the first five
# items. `PLANT` and `BUY_SEED` take one of these and nothing else.
CROPS: list[str] = list(PRODUCTS[:5])
MARKET_OPS: list[str] = [
    "NONE",
    "HIRE",
    "BUY_LAND",
    "BUY_SEED",
    "BUY_PRODUCT",
    "BUY_ANIMAL",
    "SELL",
]

# One session per worker. Eight fit the machine beside their evaluations, and
# eight is what this was while codex billed per call and the only limit was the
# machine.
#
# Two, because the limit now is a quota window rather than a core. `agy` refills
# on a five-hour clock, and on 2026-09-17 eight sessions took the Gemini pool
# from 94% to zero in twenty minutes -- then every one of the eight walked into
# the wall mid-round, so the window bought eight abandoned rounds and not one
# verdict. Concurrency cannot buy more rounds than the quota holds; all it
# decides is how many are in flight, unfinished, when the wall arrives.
#
# It also decides how fast each one measures. `--workers` is
# `CORE_BUDGET // SESSIONS`, so two sessions give a round twenty cores instead
# of five, and a round that can play its seasons four times as fast is a round
# that measures before it edits rather than guessing because measuring was slow.
SESSIONS = 2
# The champion pairing alone, which the gate gives a veto no other opponent
# has. The sweep above plays every pairing at `GATE_SEEDS`, and against an
# opponent already beaten 1.000 that is ample -- more games buy nothing once
# the interval is pinned at the ceiling. The head-to-head is the one pairing
# whose games decide something, and it was being asked with too few to answer.
#
# Measured 2026-09-21. At 32 games a Wilson lower bound above 0.5 needs a rate
# near 0.675 -- 22 wins of 32 -- so condition 3 was demanding a candidate beat
# its parent seven games in ten, which is a rout rather than an improvement.
# `pfe2ed71a20b0` won 0.594 of 32 and was refused at a lower bound of 0.423:
# a real edge turned away for want of games, not for want of strength.
#
# Sixty-four seeds is 128 games and brings the rate needed down to about 0.59,
# which is the size of edge an incremental improvement actually has. The bar
# itself does not move -- still a 95% lower bound above 0.5 -- so this buys
# evidence rather than lowering the standard, and a one-sided bound holds its
# 5% false-promotion rate at any depth. It costs the 48 seeds the sweep did
# not already play: 96 games against the sweep's ~5,800, about 1.7%.
DUEL_SEEDS = 64
# The whole space. Nothing is reserved any more: a set held back exists to
# give a number the search cannot steer, and drawing fresh seeds every
# evaluation already does that -- no program is ever measured on maps it or
# its ancestors were selected on. The reserved block was a held-out set for a
# search that is always, structurally, held out.
GATE_SEED_RANGE = range(1, 1_000_000)
# How many of our own champions stay in the pool. Harvested agents are never
# trimmed and these are, because the two are different kinds of evidence: a
# published agent says something about the field we are scored against and
# the campaign cannot produce another one, while the tenth-best rung of our
# own ladder says nothing the best rung does not.
#
# Eight, because that is roughly a third of a gate's draw -- enough that a
# candidate must beat its recent ancestry rather than only the current
# champion, and few enough that the other two thirds stay outside agents.
# Unbounded is what produced the monoculture: sixty-nine champions holding ten
# of twenty-four slots, so nearly half of every gate replayed our own lineage.
# The pool key our champion occupies. One key, overwritten on every promotion,
# because there is one champion.
#
# Not `champion_N`. `POOL_CHAMPIONS = 8` used to trim "our" champions by the
# `champion_` prefix, and that prefix is overloaded: the pool also holds
# `champion_1` and `champion_65` from the abandoned tape lineage, which are
# opponents rather than rungs of ours. On 2026-09-12 a promotion numbered itself
# from this run's empty champions directory, picked `champion_1`, and silently
# replaced the tape agent of that name -- an agent the pool was keeping
# specifically because it counters our lineage.
#
# A key that cannot be produced by harvesting or by the old lineage ends both
# problems: nothing to collide with, and nothing to scan a prefix for.
POOL_CHAMPION = "ours"
# What a retired champion is called, numbered from the promotion that retired
# it: `ours_16`, `ours_17`. Not `champion_N`, which is the prefix that caused
# the collision above -- `ours_` cannot be produced by harvesting, whose names
# are an author and a kernel slug, nor by the tape lineage, whose agents are
# `champion_N`.
POOL_ANCESTOR = "ours_{number}"


# The cheap model, and the campaign is back on it.
#
# `gpt-6-astra` ran here for three hours on 2026-09-09 and the trial is recorded
# rather than left implied. It produced one promotion -- champion_65, decisive
# over champion_64 on 28 of 32 games -- about fifty minutes in. That is not
# evidence it beats this model: the vendored-opponent draw landed in the same
# restart, so the promotion has two candidate causes, and the gate rate cannot
# separate them because it is bounded by the arena rather than by the model.
#
# Quota is why it was held back originally and why it is put down again. The
# first campaign spent nearly all of one on astra, `terra` was tried and
# reverted for the same reason, and three hours bought no result that argues for
# the cost. The comparison it was meant to inform had model, reasoning depth,
# sampler, gate rule and rating window all moving at once; removing the most
# expensive variable costs the least.
#
# The condition on trying it again is unchanged and is closer to met: fix the
# search first, because a stronger model climbing a broken hill only costs more
# to learn the same thing. What would justify a second trial is a measurement
# that can attribute -- promotions per session against a luna baseline on the
# same code, rather than against a memory of one.
#
# `mutate.validate_model` checks this against the login's own catalog at
# startup, because a typo here is hundreds of failed sessions discovered one
# at a time -- not hypothetical: this login accepts `gpt-6-astra` but refuses
# `gpt-5.6-astra` (probed 2026-09-05 on codex 0.153).
CODEX_MODEL = "gpt-5.6-luna"
# The model retried once, in the same directory, when the first model's call
# fails without a verdict -- a provider refusal or a codex crash -- so the
# round still gets a child. Astra answered "Selected model is at capacity" some
# of the time (two calls in the first live hour), which is why this exists at
# all -- and which matters more now that astra is the primary rather than the
# thing being fallen back from.
CODEX_FALLBACK_MODEL = "gpt-5.6-sol"


# The database id of the program a cold start seeds itself from. The copy the
# cold start writes under a run's `programs` is the campaign's lineage: every
# program descends from it, and it cannot change once written, which the file
# it was read from can -- harvesting an opponent's author again rewrites that
# file in place. So the copy check exempts that copy, not `SEED`, and the seed
# and the exemption cannot drift apart however a run was started.
SEED_ID = "seed"


@dataclasses.dataclass(frozen=True)
class Run:
    """Every file one campaign writes, derived from the directory it owns.

    These were module constants, and a test isolated itself by reaching into
    this module and swapping them -- which holds only while every reader looks
    the value up at call time, and a default argument does not. A run is
    constructed instead, and passed. There is no global left to patch, so
    there is none to patch wrongly, and a test gets a whole campaign of its
    own by naming a directory.

    `pool` is given rather than derived: the live one deliberately sits
    outside the run directory, so that it is not a sibling of anything a codex
    call is handed.

    Attributes:
        root: The directory this campaign owns.
        pool: The file listing the opponents, wherever it lives.
    """

    root: Path
    pool: Path

    @property
    def archive(self) -> Path:
        """The append-only log of every program and failure."""
        return self.root / "archive.jsonl"

    @property
    def programs(self) -> Path:
        """Where a program's source is stored, by id."""
        return self.root / "programs"

    @property
    def rounds(self) -> Path:
        """Where a round's own transcript is kept, by program id.

        The workspace a round works in is temporary and takes the transcript
        with it, which leaves the child and its score as the only record. That
        cannot tell a round which never opened `plan.json` from one which
        edited it, measured the edit worse and backed it out -- and the second
        is the round doing exactly what it was told. A transcript runs to
        hundreds of kilobytes, so this grows; it is worth the disk while what a
        round does with the plan is the open question.
        """
        return self.root / "rounds"

    @property
    def seed_program(self) -> Path:
        """The cold start's own copy of the seed: the campaign's lineage."""
        return self.programs / f"{SEED_ID}.py"

    @property
    def floor(self) -> Path:
        """The current floor, overwritten every promotion."""
        return self.root / "floor" / "agent"

    @property
    def champions(self) -> Path:
        """Every champion ever promoted, each written once.

        The pool points at these rather than at the floor, so a pool holding N
        champions holds N different programs rather than N references to the
        newest one.
        """
        return self.root / "champions"

    @property
    def champion(self) -> Path:
        """The promoted champion, written before the gate returns.

        Preferred over `state` on restart: a kill between the promotion and
        the next state write would otherwise lose it.
        """
        return self.root / "champion.json"

    @property
    def state(self) -> Path:
        """What a restart resumes: the session count and the champion."""
        return self.root / "state.json"


# The only file that lists opponent paths, kept out of `run/campaign/` so it
# is not a sibling of anything a codex call is given.
POOL = OPPONENTS.parent / "campaign" / "pool.json"
# The campaign this checkout runs. Everything that writes takes a `Run`, so
# this is the only place the live one is named -- a dry run and a test each
# construct their own and nothing has to be swapped out from under anyone.
LIVE = Run(root=RUN, pool=POOL)
