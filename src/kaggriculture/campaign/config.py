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
# Seeds a program is scored on: drawn fresh every evaluation and played in
# both seats against every pool opponent. This is the whole measurement -- one
# gate, one number, and promotion decided on it.
#
# There used to be two, a cheap ranking on eight seeds and a sealed block of
# sixty-four that promoted. The cheap one did not work. Over 471 programs it
# called 78 of them the best in the tournament and the block promoted none of
# them: its rating climbed 1.43 -> 2.16 while the block's score sat flat
# between 0.798 and 0.859, and the best block score belonged to the very first
# program measured.
#
# That gap was never overfitting -- the seeds are redrawn every call, so there
# is nothing to fit. It is the winner's curse. A rate over sixteen games has a
# standard error of 0.125, and taking the maximum over hundreds of such
# estimates returns the luckiest program rather than the best one. Selection
# on a noisy estimator is biased upward by construction, and no second
# measurement fixes that; only games do.
#
# Thirty-two seeds is 64 games a pairing and a standard error of 0.062, a
# quarter of the variance the eight-seed ranking selected on. It costs
# throughput -- about 31 programs an hour against 60 -- and that is the trade
# being made deliberately: 471 programs at the old depth bought no measurable
# improvement at all.
# Sixteen now rather than thirty-two, because the same budget buys twice as
# many opponents and a rating is fitted over all of a candidate's edges. Eight
# opponents at 64 games each and sixteen at 32 each are the same 512 games; the
# second tells you more, because a rating's precision comes from the whole
# graph and one more opponent is a whole new comparison where one more seed is
# a slightly tighter old one.
GATE_SEEDS = 16
# The whole space. Nothing is reserved any more: a set held back exists to
# give a number the search cannot steer, and drawing fresh seeds every
# evaluation already does that -- no program is ever measured on maps it or
# its ancestors were selected on. The reserved block was a held-out set for a
# search that is always, structurally, held out.
GATE_SEED_RANGE = range(1, 1_000_000)
# Opponents drawn for one gate. The pool itself is now everything the campaign
# has ever produced or harvested and nothing leaves it, so this is a sample
# and not the pool: a rating is fitted over every pairing anyone has ever
# played, and each candidate only has to add its own edges to that graph.
#
# The pool used to keep the top eight by rating and drop the rest, on the
# reasoning that an opponent every candidate beats separates two candidates no
# better than a coin. That is true of a *win rate* and false of a rating, and
# it cost us: champion_1 was trimmed out long ago, and champion_37 -- thirty
# promotions later, rated five log-odds above it -- beats it only 0.729 of the
# time. A field this non-transitive keeps its counters or walks past them.
GATE_OPPONENTS = 16
# How the sixteen are chosen. Anchors are played every single gate: they span
# the strength range and they are what keeps the graph connected, so a new
# champion is never rated through a chain of thirty overlapping pool eras.
# Measured 2026-09-07, that chain predicted champion_37 would beat champion_1
# 0.994 of the time; it beats it 0.729.
#
# Contenders are the highest rated, because topping the field still means
# beating the best of it. The remainder is drawn at random from everything
# else, which is both coverage -- the graph would otherwise go stale
# everywhere except the top -- and how a counter gets found rather than
# quietly discarded.
GATE_ANCHORS = (
    "champion_1",
    "champion_10",
    "champion_20",
    "champion_30",
    "thomastschinkel_router",
    "router_v1",
)
GATE_CONTENDERS = 6
# How far above the champion's rating a candidate must sit to replace it, in
# log-odds. Promotion used to mean topping a tournament of eight, which is a
# rank and so has no margin in it at all: a candidate a hair above the
# champion promoted, and the hair was usually noise.
#
# A rating has units, so the bar can be stated. 0.15 log-odds is about a 54%
# head-to-head, and at sixteen seeds a pairing the standard error on a
# candidate's rating is well inside that -- which is the point. Selecting the
# maximum of a noisy estimator is biased upward by construction, and this is
# the guard: 78 of 471 programs once topped a noisy gate and none of them
# survived a deeper measurement.
PROMOTION_MARGIN = 0.15
# The fewest games a candidate must actually decide against the floor before
# the margin above means anything.
#
# Measured 2026-09-08. champion_55 was promoted over champion_54 on a rating
# gap that the bar above reads as "about a 54% head-to-head". Their thirty-two
# games were two wins by five units and thirty exact draws: the two programs
# play the same game. A draw scores as half a win, so thirty draws and two
# wins come to 0.53125 -- the same number as seventeen wins and fifteen
# losses, which is two agents genuinely trading games rather than one agent
# and a copy of itself. Bradley-Terry cannot tell those apart, and the Wilson
# guard beside it was claiming thirty-two games of confidence for a pairing
# that decided two.
#
# Eight of thirty-two is a quarter. Below that the two programs are the same
# program and there is nothing to promote; above it the margin above is being
# read on games that happened.
DECISIVE_GAMES = 8
# How many of the database's best a session may start from, and how sharply
# the draw favours the better ones: weight `PARENT_DECAY ** rank`, so the best
# is taken about half the time, the second a quarter, and the tenth almost
# never.
#
# It was a uniform draw over ten, which is barely selection at all. With no
# champion there is nothing else deciding where a session begins, so nine
# sessions in ten started from something worse than the best program the
# campaign had -- while only one child in ten improves on its parent and one
# in five is worse. The population drifted down faster than selection pulled
# it up: over 259 rated programs the best rating peaked at the fiftieth and
# every cohort after was worse.
#
# Not fully greedy, because a parent is only half the move: the five
# instructions and the model's own sampling are the other half, and a search
# that always started from one program would explore with one hand.
PARENT_POOL = 10
PARENT_DECAY = 0.5

# Sessions without a promotion before a session starts from a program
# drawn from the database's top ten instead of the champion.
STAGNATION_SESSIONS = 40
# Calls in a row that may run to no verdict before the campaign stops. A call
# that never reached the model is nobody's failure and writes nothing, so
# without this the loop spins at full rate on an expired login, a withdrawn
# model or a provider outage, looking busy and producing nothing. Eight is
# one per worker: a single bad call is noise, eight is the machine.
NO_VERDICT_LIMIT = 8
# The cheap model, on purpose. `gpt-6-astra` is the strongest this login has
# and is not used: the first campaign spent nearly all of a quota on it, and
# quota is the constraint that binds here rather than capability. `terra` was
# tried briefly and reverted for the same reason.
#
# The 381-program block that ran on this model is not evidence against it. A
# session drew its starting program uniformly from the best ten for the whole
# of it, which is barely selection, and 95% of every rate measured was
# saturated at nought or one. That search ran on a broken hill, so "setup or
# model?" is still open -- and it is the search that is being fixed first,
# because a stronger model climbing the same broken hill would only cost more
# to learn the same thing.
#
# `mutate.validate_model` checks this against the login's own catalog at
# startup, because a typo here is hundreds of failed sessions discovered one
# at a time -- not hypothetical: this login accepts `gpt-6-astra` but refuses
# `gpt-5.6-astra` (probed 2026-09-05 on codex 0.153).
CODEX_MODEL = "gpt-5.6-luna"
# The model retried once, in the same directory, when the first model's call
# fails without a verdict -- a provider refusal or a codex crash -- so the
# round still gets a child. Astra answered "Selected model is at capacity"
# some of the time (two calls in the first live hour), which is why this
# exists at all.
CODEX_FALLBACK_MODEL = "gpt-5.6-sol"
SERVED = ROOT / "src" / "kaggriculture" / "served" / "main.py"
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
    this module and swapping them. That works only while every reader looks
    the value up at call time, and one did not: `gate.refresh` took
    `kept: Path = FIELD`, a default evaluated once at definition, so no patch
    ever moved it and a dry run wrote its champions into the live campaign's
    pairings. The live `field.json` ended up holding `champion_1` through
    `champion_9` and an opponent called `pass`.

    A run is constructed instead, and passed. There is no global left to
    patch, so there is none to patch wrongly, and a test gets a whole campaign
    of its own by naming a directory.

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
    def seed_program(self) -> Path:
        """The cold start's own copy of the seed: the campaign's lineage."""
        return self.programs / f"{SEED_ID}.py"

    @property
    def field(self) -> Path:
        """The pool's own pairings, kept between gates.

        They are constants: the opponents are fixed files and none of them
        draws on randomness. Derived state, not a committed artifact -- a pool
        that gains a champion has that champion's pairings measured and added,
        and one that loses an opponent stops asking for its row.
        """
        return self.root / "field.json"

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


# Metrics go to one wandb run per campaign, resumed across restarts by its
# fixed id. Starting a fresh campaign (a new `run/campaign`) means a new id
# here, or its curves land on top of the old run's.
WANDB_ENTITY = "will-rice"
WANDB_PROJECT = "kaggriculture-2026"
# The only file that lists opponent paths, kept out of `run/campaign/` so it
# is not a sibling of anything a codex call is given.
POOL = OPPONENTS.parent / "campaign" / "pool.json"
# The campaign this checkout runs. Everything that writes takes a `Run`, so
# this is the only place the live one is named -- a dry run and a test each
# construct their own and nothing has to be swapped out from under anyone.
LIVE = Run(root=RUN, pool=POOL)
# The published agent a cold start begins from, and the one number that says
# why it rather than another: fit over `FIELD` on 2026-09-06 it rated +1.78
# against +0.64 for the next opponent and -2.67 for the last, so it is the
# top of the field by a clear point of rating.
#
# The campaign spent 381 programs evolving upward from a thirty-line skeleton
# and never once topped the tournament -- 95% of every rate it measured was a
# shutout, because a program that loses every game to eleven of twelve
# opponents has no gradient to climb. Starting from the strongest published
# agent starts the search where the gradient is: every opponent is within
# reach of it, so an edit that helps shows up as a rate that moves.
#
# What makes this legitimate is the competition's own sharing rule -- this
# agent is published, and published code is the field's to build on -- and
# what makes it worth anything is the gate, which is unchanged: a candidate
# is promoted only when it tops the tournament, and this agent is *in* that
# tournament. Tying the seed does not promote. Only beating it does.
#
# It is 619 lines in one file, which matters: a round hands the model the
# whole program, and the 3,778-line `shopforge` or the 316KB tuned kernels
# would spend a call being read rather than improved.
SEED = OPPONENTS / "thomastschinkel_router" / "main.py"
