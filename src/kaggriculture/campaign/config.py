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
# Where an opponent whose source predates the vendored drop still lives; the
# roster names one. Both roots are stated here so no other module spells out
# a path under /data.
AGENTS = Path("/data/kaggriculture/agents")
EPISODES = Path("/data/kaggriculture/episodes")
ENGINE_LIBRARY = Path(__file__).parent / "engine" / "kaggriculture_engine.so"

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
# Consecutive codex calls against one opponent before the session moves to the
# next, and a session works through every opponent in the pool. So its length is
# the pool's: 41 opponents is 492 rounds, and what ends a session in practice is
# a promotion, a call that ran to no verdict, or the run stopping.
#
# There is no fixed cap any more. `ROUNDS_PER_SESSION = 12` was one, and it made
# the session length arbitrary -- twelve rounds covered one opponent or twelve
# depending on how the games happened to be walked. Both ways of walking them
# shipped on 2026-09-12 and both were wrong: the flat list gave twelve seasons
# against whichever agent `games.ordered` puts first, since it is matchup-major
# and that is the one the program loses to hardest; striding matchups gave one
# game against each of twelve, which is twelve first impressions.
#
# Twelve attempts is what learning an opponent takes, and the season advances
# inside the block, so they are twelve maps against the same agent rather than
# twelve tries at one game.
ROUNDS_PER_OPPONENT = 12
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
# Candidates that share one block of seasons before a fresh block is drawn.
#
# Sharing is what makes two candidates comparable at all. The seeds used to be
# drawn per call, so no two programs were ever ranked on the same seasons --
# and a season is most of what the rating measures. One unchanged agent
# through the gate five times, opponents held fixed and only the seeds moving,
# gave fitted ratings from -3.466 to -2.226: a standard deviation of 0.491,
# against a promotion bar that used to be 0.15. Holding the seeds and varying
# the opponents instead moved it 0.070, so the maps are seven times the draw.
#
# Rotating is what keeps the note above this constant true -- a block that
# never moved would be one the search gets selected against, which is exactly
# what the reserved held-out set existed to prevent. Sixty-four is a few
# generations of eight concurrent sessions: long enough that the candidates
# being compared share a block, short enough that no lineage lives in one.
SEED_ROTATION = 64
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
#
# Twenty-four rather than sixteen, because the draw now includes every vendored
# opponent: about a dozen of those, plus the leader and floor, plus the four
# anchors that are not themselves vendored, plus `GATE_CONTENDERS`. Sixteen
# would have truncated exactly the agents the change exists to include.
#
# The cost is real and was weighed against playing the pool entire. That is
# roughly 75 opponents today, 2,400 games a candidate against 512, and it grows
# with every promotion -- while buying no extra *share* of cross-population
# evidence, since the pool is itself 84% champions. Drawing all the vendored
# agents and sampling the rest lifts that share from about a sixth to about a
# half for half the added cost, and does not grow.
GATE_OPPONENTS = 24
# How the twenty-four are chosen. Anchors are played every single gate: they span
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
    "champion_65",
    "thomastschinkel_router",
    "router_v1",
)
# Four, and every one of them a different agent. It was six, and four of those
# were champion_1, _10, _20 and _30 -- which read as four points spanning the
# strength range and were four ages of one recording. The 720-step table
# underneath that lineage is byte-identical from its seed through champion_69,
# sha ef59f6f4a545d342, and 86.5% of every action any of them emits comes
# straight out of it; what differs between them is the repair layer over the
# other 13%.
#
# An anchor's whole job is to be a fixed point a rating is calibrated against,
# so anchoring the scale to one agent at four ages is the failure that
# calibration exists to prevent. champion_65 replaces the three: it is the
# strongest of that lineage and the bar a submission has to clear, so it earns
# a slot on its own account rather than as a reference point.
#
# The rest of that lineage left the pool with them. Sixty-nine champions held
# ten of a gate's twenty-four slots, which bought ten readings of one
# recording; its run is kept whole under `run/campaign-tape-lineage/` and its
# pairings stay in `field.json`, so the fit still places it.
# Highest-rated agents drawn beyond the anchors and the vendored set. Four
# rather than six because the leader is already drawn through `always` and the
# vendored opponents now take a dozen slots: the contenders were competing for
# room with the only cross-population evidence the gate gets.
GATE_CONTENDERS = 4
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
# There is no promotion margin any more, and this note is here so nobody adds
# one back.
#
# It was `PROMOTION_MARGIN = 0.15` of rating, about 26 Elo, and its job was to
# stop the winner's curse: selecting the maximum of a noisy estimator is
# biased upward by construction, and 78 of 471 programs once topped a noisy
# gate with none surviving a deeper look. The reasoning was right and the
# instrument was never checked against it. Measured 2026-09-11: one unchanged
# agent's fitted rating moves with a standard deviation of 0.745 across draws
# -- 129 Elo -- so the bar sat a fifth of a standard deviation out and
# filtered almost none of the noise it was aimed at. What it did filter
# reliably was a real improvement too small to clear it.
#
# A fixed size cannot be the answer to a quantity that varies with the draw,
# the candidate's strength and how many games were decided. `gate.promotion`
# asks for significance instead: the gate already plays the candidate against
# the champion over every gate seed in both seats, which is one set of seasons
# played twice and therefore paired, and a margin larger than twice its own
# error is a demonstration. That bar tightens when the measurement is good and
# refuses when it is not, which is the whole of what the constant was for.
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

# How often a session begins from nothing instead of from the champion.
#
# Every champion this campaign has produced is an edit of an edit of `SEED`,
# which is a harvested public agent -- one program's descendants, sixty-five
# generations deep. That is the monoculture at its root, and no instruction
# escapes it, because a round is handed the champion and asked to improve it.
#
# The case is structural, not empirical, and it is worth being exact about
# which. It is tempting to point at champion_1 scoring 2086.8 and champion_48
# scoring 1963.3 and say sixty-five generations bought nothing -- but those are
# different days, and the same bytes have scored 2386.8 and 1555.8 five days
# apart, so that comparison measures the field moving rather than the lineage
# standing still. Even the same-day pair, champion_47 at 2005.9 and
# champion_48 at 1963.3, sits inside a noise floor where identical agents have
# landed 455 and 512 apart.
#
# What is established is narrower and does not need the leaderboard: every
# champion is an edit of an edit of one harvested public agent, and neither
# signal we have can currently tell us whether that is working. The ladder
# score moves with the field; the gate's own rating over-claims by about five
# points of win rate against exactly the opponents that predict the ladder,
# because most of its evidence is the lineage measuring itself. Diversity here
# is a hedge against being stuck without being able to see it, which is a
# weaker claim than "the lineage is stuck" and the one the evidence supports.
#
# One session in eight. It is a real cost -- an eighth of the quota, on
# programs that begin unable to play -- so it is written here as a number to
# turn down rather than buried in the loop.
SCRATCH_CHANCE = 0.125
# The name the blank slate goes by, and the root every scratch lineage is
# traced back to.
SCRATCH_ID = "scratch"
# The blank slate itself: a policy that passes every turn. Not an empty file,
# which nothing downstream can score, and not `SEED`, which is the ancestry
# being escaped. It loses every game it plays, which is the point -- what it
# has that a champion does not is no commitments.
SCRATCH_AGENT = (
    "def agent(observation, configuration=None):\n"
    "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
)

# Sessions without a promotion before a session starts from a program
# drawn from the database's top ten instead of the champion.
STAGNATION_SESSIONS = 40
# Calls in a row that may run to no verdict before the campaign stops. A call
# that never reached the model is nobody's failure and writes nothing, so
# without this the loop spins at full rate on an expired login, a withdrawn
# model or a provider outage, looking busy and producing nothing. Eight is
# one per worker: a single bad call is noise, eight is the machine.
NO_VERDICT_LIMIT = 8
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
# How hard the model is asked to think, passed on every call.
#
# Astra offers low, medium, high, xhigh, max and ultra, and defaults to medium.
# The campaign was not running at medium, though, and not at anything it chose:
# `~/.codex/config.toml` sets `model_reasoning_effort = "high"` for the host's
# own interactive use, and every campaign call inherited it. That is the same
# shape of coupling as a round inheriting the host's skills -- the loop's
# behaviour changing because a file it does not own changed -- and it is worth
# closing whatever the value is.
#
# `max` is "maximum reasoning depth for the hardest problems". Above it sits
# `ultra`, which adds automatic task delegation; that is a different execution
# shape rather than more thinking, and a round already has a shape.
CODEX_REASONING = "max"
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
# How often the campaign takes newly published kernels into its pool, and how
# many refs it considers each time.
#
# `harvest` was written as the other half of the ratchet -- champions join on
# every promotion and nothing else does, so left alone the pool becomes the
# campaign playing itself. It ran on 2026-09-01 and 2026-09-06 and then
# nothing ran it, which is the whole of why the pool reached 69 champions
# against 12 published agents, all of them frozen at the older of those dates.
# A field that turns over in days was being gated against a five-day-old
# snapshot of itself.
#
# So the loop harvests rather than a person remembering to. Hourly because
# that is the rate the competition publishes at and a kernel costs one
# 720-step game to check; forty refs because that is roughly five days of
# publications, so a restart after an outage catches up in one pass.
HARVEST_INTERVAL_SECONDS = 3600
HARVEST_LIMIT = 40
# The campaign this checkout runs. Everything that writes takes a `Run`, so
# this is the only place the live one is named -- a dry run and a test each
# construct their own and nothing has to be swapped out from under anyone.
LIVE = Run(root=RUN, pool=POOL)
# `pb75e380571fc`, the best program the campaign's own rule-based lineage ever
# wrote: 491 lines that decide the season turn by turn from the observation --
# analytic market prices, crop and livestock forecasts on actual production
# dates, workers assigned by value. No recorded actions anywhere in it.
#
# It replaces `thomastschinkel_router`, which was adopted on 2026-09-07 on the
# reading that the rule-based search had no gradient: 95% of every rate it
# measured was a shutout. That reading was of the wrong number. Win rate was
# flat because a young lineage beats nobody, while the mean bank margin
# underneath it ran from -179,647 to -8,444 -- 171,000 coins of clean,
# well-ordered signal, already recorded on every program, already the
# tie-break `Database.top` sorts on. The record over that run went -119,258,
# then -10,446, then -8,444, the last of them 585 seconds before the run was
# stopped. It was accelerating when we read it as dead.
#
# What we adopted instead turned out to be a 720-step recording with a repair
# layer around it: 86.5% of champion_69's actions came out of the table
# verbatim, the table was byte-identical from the seed through 69 promotions,
# and emptying it dropped the agent to 3,000 -- what passing every turn banks.
# The search never touched the policy because it never could; 29,820
# characters of base64 do not fit in a prompt.
#
# So the lineage starts from a program that plays, and `ALLOWED_IMPORTS` no
# longer admits `base64` or `zlib`, which is what a recording needs to travel.
SEED = ROOT / "src" / "kaggriculture" / "seed" / "main.py"
