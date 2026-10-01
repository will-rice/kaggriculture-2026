"""The controller loop: sample, prompt, generate, evaluate, add.

Spec section 1, four steps: name the run, start the event loop, spin off
``config.SESSIONS`` workers, and gate what they produce. A worker is section
3 -- take the champion, then round after round compose a message, have codex
write one file, validate it, play it against every opponent, insert it, and
send the result back for another round -- and eight of them run that against
one database, one pool and one champion, so a promotion by any of them
changes what the other seven start from next. That is the only structure
there is.

The model is a mutation operator. It measures nothing, owns nothing and runs
nothing: the loop plays every game, which is what makes the verdict ours and
the cores accounted for. That is AlphaEvolve's controller loop
(``database.sample``, ``prompt_sampler.build``, ``llm.generate``,
``evaluator.execute``, ``database.add``) with FAMOU's evaluator, which
"prompts the LLM to produce complete modified Python files rather than
patches, validates the generated files, evaluates valid candidates in the
game simulator, and feeds the resulting performance summary back into the
next mutation prompt".

One event loop on one thread owns the database, the pool, the state and the
wandb run, so none of them needs a lock. Everything that blocks is awaited
elsewhere: a codex call is a child process in its own group, and validation,
the evaluation, a promotion's file work and a round's directory removal go
to threads. Each of those touches only what is its own -- `gate.promote`
writes files nothing else has -- while the pool that promotion joins is
changed here, on the loop, where the other seven workers are reading it.

Every evaluation forks a process pool (spawn, because the harness caps tasks
per child), so every caller of ``run`` must sit behind an
``if __name__ == "__main__":`` guard: ``python -m kaggriculture.campaign.loop``,
the ``campaign loop`` entry point and pytest all do.
"""

import argparse
import asyncio
import logging
import random
import signal
import tempfile
import time
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import wandb
from pydantic import BaseModel

# `gate` is also the name of a `Campaign` method, so annotations in the class
# body cannot see this module -- hence `Champion` imported by name below.
from kaggriculture.campaign import (
    archive,
    config,
    evaluator,
    games,
    gate,
    harvest,
    losses,
    plan,
    prompt,
    telemetry,
    validate,
    workspace,
)
from kaggriculture.campaign.evaluator import Result
from kaggriculture.campaign.gate import Champion
from kaggriculture.campaign.harness import OpponentCrash
from kaggriculture.campaign.mutate import (
    FakeMutator,
    Mutation,
    Mutator,
    asked_model,
    build,
    validate_models,
)
from kaggriculture.campaign.pool import Pool

LOGGER = logging.getLogger(__name__)

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

# How often the loop reloads the games this lineage really lost. An hour, like
# the harvest, and for the same reason: both keep a measurement from drifting
# away from the competition while the campaign optimises against it.
#
# It costs little after the first pass: `losses._held` asks the database which
# episodes it already holds, so an hour later only the games played since are
# fetched -- and a submission plays a few an hour, not a few hundred.
LOSSES_INTERVAL_SECONDS = 3600

# Calls in a row that may run to no verdict before the campaign stops. A call
# that never reached the model is nobody's failure and writes nothing, so
# without this the loop spins at full rate on an expired login, a withdrawn
# model or a provider outage, looking busy and producing nothing. Eight is
# one per worker: a single bad call is noise, eight is the machine.
NO_VERDICT_LIMIT = 8

PARENT_DECAY = 0.5

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

# The blank slate itself: a policy that passes every turn. Not an empty file,
# which nothing downstream can score, and not `SEED`, which is the ancestry
# being escaped. It loses every game it plays, which is the point -- what it
# has that a champion does not is no commitments.
SCRATCH_AGENT = (
    "def agent(observation, configuration=None):\n"
    "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
)

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
SEED = config.ROOT / "src" / "kaggriculture" / "seed" / "main.py"

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

# Sessions without a promotion before a session starts from a program
# drawn from the database's top ten instead of the champion.
STAGNATION_SESSIONS = 40

# Threads the loop needs at once: one per session for whichever blocking step
# it is on -- scoring a program, validating one, reading a transcript, and
# never two at once within a session -- plus the one a promotion packages on.
THREADS = config.SESSIONS + 1

# The database id of the program a cold start seeds itself from. Until the
# first promotion there is no champion, so this is the name the first
# sessions are told they are editing and the bar their verdict quotes. It
# lives in `config` so that the stored copy -- which is what the campaign's
# lineage actually is -- is named in one place.
SEED_ID = config.SEED_ID

# What the wandb run records as its configuration: spec section 8's table.
# The two model slugs are not here: they are chosen in `.env` at the call, so
# the constant beside them is a default rather than a fact about this run.
# `telemetry.open_run` reads them through the accessors instead.


def hyperparameters() -> dict[str, int]:
    """What the wandb run records as its configuration: spec section 8's table.

    Read at the call rather than held in a tuple of names, because each
    constant lives in the module that reads it and a test that moves one moves
    it there -- a name looked up on `config` would report the value the run
    did not use.
    """
    return {
        "SESSIONS": config.SESSIONS,
        "ROUNDS_PER_OPPONENT": ROUNDS_PER_OPPONENT,
        "GATE_SEEDS": GATE_SEEDS,
        "STAGNATION_SESSIONS": STAGNATION_SESSIONS,
    }


# Prepended to the instruction under stagnation, so the message says that this
# session did not start from the champion, and why.
STAGNATION_NOTE = """This session did not start from the champion. {sessions} \
sessions have passed with no promotion, so this lineage began at a program \
drawn from the database's top ten: the champion's line is stuck and this one \
may not be.

"""


def main(argv: list[str] | None = None) -> None:
    """``campaign loop``: eight workers, each mutating the champion."""
    args = _arguments(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    # A dry run is the whole campaign with the call faked, so it writes
    # everything a real one does -- into a directory of its own. It used
    # to get there by reassigning this module's constants, which is the
    # same global mutation the tests used and failed at in the same
    # place: a path captured in a default argument never moved.
    root = args.run_root / "dry-run" if args.dry_run else args.run_root
    paths = config.Run(
        root=root, pool=root / "pool.json" if args.dry_run else args.pool
    )
    if args.dry_run:
        LOGGER.info("dry run: every write goes under %s", paths.root)
    else:
        # A typo'd model is hundreds of failed sessions discovered one at a
        # time; caught here, before the run opens or a call is ever made.
        # Through the accessors, because `.env` is where the slug is chosen
        # now: validating the constant would pass a run that never uses it.
        # Against the catalog of whichever program is driving, because the two
        # share no vocabulary at all.
        validate_models()
    # 1. wandb, named for the revision of the code that produced the run.
    log = telemetry.open_run(args.dry_run, args.tag, hyperparameters())
    mutator: Mutator = (
        FakeMutator(edit=lambda source: source + "\n# dry-run mutation\n")
        if args.dry_run
        else build()
    )
    try:
        # 2-4. the event loop, the workers, and the gate they fire.
        run(
            args.sessions,
            mutator,
            args.workers,
            args.seed_agent,
            random.Random(),
            log,
            paths,
        )
    finally:
        log.finish()


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    """Parse the loop's command line; ``sys.argv[1:]`` when ``argv`` is None."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=10**9)
    # Section 7: every session in flight can be evaluating at once.
    parser.add_argument("--workers", type=int, default=config.GATE_CORES)
    # Whatever this names is copied to the run's `seed_program`, and that copy is
    # what `copycheck` exempts, so a run started from a snapshot exempts the
    # snapshot and a run started from the default exempts the default. There
    # is nothing here for the gate to disagree with.
    parser.add_argument("--seed-agent", type=Path, default=SEED)
    # The directory this campaign owns. A flag rather than a constant so a
    # dry run and a test each name their own, instead of reaching into this
    # module to move one -- which is what used to happen, and what silently
    # failed against a path captured in a default argument.
    parser.add_argument("--run-root", type=Path, default=config.RUN)
    # The pool this campaign measures against. A flag because two lineages
    # can now run at once and a pool is read-modify-written on every
    # promotion: sharing one would have them racing, and both writing the
    # single `POOL_CHAMPION` key and their own `ours_N` series into it.
    parser.add_argument("--pool", type=Path, default=config.POOL)
    # Distinguishes this run's wandb id from another on the same revision.
    # The id is the commit sha, which is the right name for one lineage and
    # the same name for two.
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="fake calls, no log, every write under run/campaign/dry-run",
    )
    parsed = parser.parse_args(argv)
    # Absolute from here on. An evaluation worker is given a directory of its
    # own and `chdir`s into it, so a relative path stops resolving the moment a
    # game starts -- which is minutes after the mistake was made, and reads as
    # a crashed agent rather than a bad argument. `--run-root` is resolved for
    # a slower version of the same fault: `gate.promote` writes the champion's
    # path into the pool as text, so a relative root seeds relative paths into
    # a file every later gate reads.
    for flag in ("seed_agent", "run_root", "pool"):
        setattr(parsed, flag, getattr(parsed, flag).resolve())
    return parsed


def state_file(paths: config.Run) -> Path:
    """Where the state a restart resumes from lives, beside the database."""
    return paths.state


class Kept(NamedTuple):
    """What a round produced, once it has been scored and put to the gate.

    One object rather than a widening tuple, because every field of it is
    something a caller was recomputing.

    Attributes:
        source: The stored program, on disk.
        name: Its database id.
        result: What the loop's evaluation of it said.
        cleared: Whether the gate promoted it.
    """

    source: Path
    name: str
    result: Result
    cleared: bool


class State(BaseModel):
    """What survives a restart, written after every session.

    ``champion`` is the whole floor in one field -- the file every session
    starts from, the name the pool knows it by, and its gate result on the
    pool as that promotion left it -- so a champion cannot be half present.
    """

    sessions: int = 0
    calls: int = 0
    champion: Champion | None = None
    sessions_since_promotion: int = 0


def run(
    sessions: int,
    mutator: Mutator,
    workers: int,
    seed_agent: Path,
    rng: random.Random,
    log: wandb.Run,
    paths: config.Run,
    opponents: Path = config.OPPONENTS,
) -> State:
    """Load what a restart resumes, seed an empty database, drive the workers.

    ``sessions`` is how many this run starts; what is in flight when the last
    is taken is drained. Returns the state as of the last completed session.

    ``opponents`` is the directory pool membership is read from, an argument
    rather than a global so that what a run measures against is something the
    caller states. A test that builds a two-opponent pool and then silently
    plays the machine's whole data directory is not testing what it says.
    """
    resume = state_file(paths)
    state = (
        State.model_validate_json(resume.read_text(encoding="utf-8"))
        if resume.exists()
        else State()
    )
    # `state.json` is written after every session, `champion.json` the moment
    # a champion is in the pool. A kill in between leaves the first stale, so
    # the second wins: otherwise the gate would restart with no baseline.
    champion = gate.load_champion(paths)
    if champion is not None:
        state.champion = champion
        LOGGER.info("resuming on champion %s from champion.json", champion.name)
    pool = Pool.load(paths.pool) if paths.pool.exists() else Pool.initial()
    # Whatever has been vendored since the last launch joins here rather than
    # waiting for the script that wrote it to remember the pool. Two writers
    # fill the opponents directory and only `harvest` ever wrote membership,
    # so sixty-one playable agents -- thirty-six of them families rebuilt from
    # recorded ladder episodes -- had never been played by anything.
    adopted = pool.adopt(opponents) if opponents.exists() else []
    if adopted:
        LOGGER.info("adopted %d opponents from %s", len(adopted), opponents)
    # A champion's name resolves through the pool file, so the pool on disk
    # must be current before anything plays a game.
    pool.save(paths.pool)
    # No anchor pairings are played here. They existed so a Bradley-Terry fit
    # would have a connected graph before the first gate ran, and no decision is
    # fitted any more: promotion is a win rate on shared opponents and a
    # head-to-head, both measured in the candidate's own evaluation. The fifteen
    # pairings cost about ten minutes of every startup.
    database = archive.Database(paths.archive, paths.programs)
    scored: Result | None = None
    if not database.programs:
        scored = evaluator.score(
            seed_agent,
            SEED_ID,
            pool,
            rng,
            rng.sample(config.GATE_SEED_RANGE, GATE_SEEDS),
            workers,
            paths.pool,
        )
        stored = database.store(seed_agent.read_text(encoding="utf-8"), SEED_ID)
        database.add(archive.measured(SEED_ID, stored, "", "seed", "", scored))
        LOGGER.info("seeded from %s at %.3f over the pool", seed_agent, scored.fitness)
    # Champion zero, so there is no pre-champion regime: every session starts
    # from a champion and every candidate plays one head-to-head, which makes the
    # promotion bar the same single condition from the first round instead of a
    # branch for having nothing to beat.
    if state.champion is None:
        first = database.top(1)[0]
        if scored is None or scored.program_id != first.id:
            # Resuming a run that has programs but no champion: the best of them
            # has to be measured against the pool as it now stands before it can
            # be the thing others are asked to beat.
            scored = evaluator.score(
                Path(first.source_path),
                first.id,
                pool,
                rng,
                rng.sample(config.GATE_SEED_RANGE, GATE_SEEDS),
                workers,
                paths.pool,
            )
        # On the record like every other evaluation. A session hands its rounds
        # a game read back from the database, so a champion whose games were
        # never written would give the first sessions nothing to show -- which
        # was the case until 2026-09-25, when the seed's games lived only in
        # memory and were lost with the process.
        games.record(first.id, games.played(scored, first.id), games.DATABASE)
        champion = gate.promote(first, scored, paths, package=False)
        gate.enroll(champion, pool, paths)
        state.champion = gate.record(champion, paths)
        database.promoted_as(first.id, champion.name)
        LOGGER.info(
            "champion zero: %s from %s at %.3f", champion.name, first.id, scored.fitness
        )
    campaign = Campaign(state, database, pool, mutator, workers, rng, log, paths)
    asyncio.run(campaign.drive(sessions))
    return campaign.state


class Campaign:
    """One run: the shared state, and the coroutines that advance it.

    Every attribute here is read and written only from coroutines on the one
    event-loop thread, which is what makes it all safe without a lock.
    """

    def __init__(
        self,
        state: State,
        database: archive.Database,
        pool: Pool,
        mutator: Mutator,
        workers: int,
        rng: random.Random,
        log: wandb.Run,
        paths: config.Run,
    ) -> None:
        self.state = state
        self.database = database
        self.pool = pool
        self.mutator = mutator
        self.workers = workers
        self.rng = rng
        self.log = log
        # Every file this campaign writes, so nothing here reaches for a
        # module-level path and no test has to swap one out from under it.
        self.paths = paths
        self.remaining = 0
        # The seasons every candidate in the current block is measured on, and
        # how many have been measured since it was drawn. Both live here
        # rather than in a session, because the eight run at once and a block
        # only makes candidates comparable if they share it.
        self.block: list[int] = []
        # The extra seasons the champion pairing is played on, drawn with the
        # block and disjoint from it. `seasons` explains why it is a slice of
        # the same sample rather than a second draw.
        self.duel: list[int] = []
        self.measured = 0
        # Calls in a row that ran to no verdict. A call that never reached the
        # model is nobody's failure, so it writes nothing and the worker
        # simply starts another -- which, when the cause is the login, the
        # model slug or the provider rather than one call, is a spin at full
        # rate that leaves the database empty while every counter advances.
        # It happened on this campaign's first launch: sixteen sessions in
        # forty-seven seconds over a missing flag.
        self.no_verdict = 0
        # Which champion file each candidate actually played, taken from the
        # pool snapshot its evaluation kept. The pool key is fixed, so the key
        # alone cannot tell the gate that the champion moved mid-evaluation --
        # the file can.
        self.champion_played: dict[str, str] = {}
        # The champion as measured on a given block, by the file measured and
        # the seasons it was measured on. The gate compares a candidate against
        # the champion, and the champion in `champion.json` carries the result
        # it was promoted on -- a different block, and now a different pool.
        #
        # That comparison is not one. Measured 2026-09-14: champion_12 scored
        # 0.454 over fifty shared opponents when it was promoted and 0.205 over
        # the same fifty on the block drawn eight hours later, a drift of 0.249
        # against a promotion bar of 0.035. Every one of the twenty-two
        # candidates the gate turned away that morning had beaten it -- they
        # ranged 0.234 to 0.366 -- and were refused for failing to reach a
        # number the champion itself could no longer reach. The bar is sized
        # for the sampling noise inside a block; the seasons are worth an order
        # of magnitude more than that, which `GATE_SEED_ROTATION`'s own note
        # records as an unchanged agent moving 1.24 log-odds across five
        # blocks.
        #
        # A session measures the program it starts from, so the champion is
        # re-measured on the current block eight times a generation for free.
        # This keeps the latest of those.
        # The champion's own measurement on one block, which is what a
        # candidate on that block can fairly be compared against. Written when
        # a session starts from the champion and when a promotion makes one,
        # since a promoted candidate was measured on the current block and its
        # result is the new champion's.
        self.champion_baseline: tuple[str, tuple[int, ...], Result] | None = None
        # A promotion writes the floor and swaps the champion's pool slot; eight
        # sessions promoting at once would race on both.
        self.promotions = asyncio.Lock()
        self.group = asyncio.TaskGroup()

    async def drive(self, sessions: int) -> None:
        """Steps 2-4: the workers and their gates, stopping cleanly on a signal.

        SIGINT and SIGTERM cancel the work, and a cancelled round kills its
        own process group, so a restart never finds one still alive.

        Raises:
            OpponentCrash: A pool opponent failed in its own seat, in a round
                or a gate. The task group has cancelled everything else by the
                time it arrives here.
        """
        self.remaining = sessions
        running = asyncio.get_running_loop()
        # `asyncio.to_thread` otherwise borrows the default executor, which is
        # sized `cpu_count + 4`. Every thread this loop asks for is waiting on
        # a subprocess or a file, never computing, so the core count is the
        # wrong basis: on a four-core machine that pool is eight threads
        # against eight sessions, and the sessions
        # that cannot get one simply do not run -- silently, because nothing
        # fails, they just queue. Sized here to what the loop actually asks
        # for at once, so the same campaign runs the same way on any machine.
        running.set_default_executor(
            ThreadPoolExecutor(
                max_workers=THREADS, thread_name_prefix="campaign-blocking"
            )
        )
        work = asyncio.ensure_future(self.work())
        # Beside the work rather than inside its task group, so that it ends
        # when the sessions do: a group waits for every task it holds, and a
        # harvester that sleeps for an hour would hold a finished run open.
        harvesting = asyncio.ensure_future(self.harvesting())
        losing = asyncio.ensure_future(self.losing())
        for number in (signal.SIGINT, signal.SIGTERM):
            running.add_signal_handler(number, work.cancel)
        try:
            await work
        except asyncio.CancelledError:
            LOGGER.warning("stopped on a signal at %d sessions", self.state.sessions)
        finally:
            harvesting.cancel()
            losing.cancel()
            for number in (signal.SIGINT, signal.SIGTERM):
                running.remove_signal_handler(number)

    async def losing(self) -> None:
        """Keep the games this lineage really lost current, for as long as it runs.

        The same reasoning as `harvesting`, one level out. A pool left alone
        becomes this campaign playing itself; a set of losses left alone becomes
        the *previous* champion's failures, and the search then studies games the
        standing program never played. Champions 31 to 39 all arrived inside a
        day, so by hand this is stale within the hour.

        Everything goes to a thread: the listing, the downloads and the parse are
        all slow and none of them touches the pool or the state the loop owns.
        Only the games database is written, and it takes parallel writers.

        A failed refresh is not a failed campaign, for the same reason a failed
        harvest is not: the competition's API is somebody else's uptime, and a
        run that has been evaluating for hours must not end because a listing
        timed out.
        """
        while True:
            await asyncio.sleep(LOSSES_INTERVAL_SECONDS)
            try:
                loaded = await asyncio.to_thread(losses.refresh)
                # Every pass, not only the ones that loaded something. The
                # figures describe the corpus rather than the delta, so a run
                # that promotes a champion and loads nothing for an hour still
                # has a series to read the promotion against.
                where = await asyncio.to_thread(losses.deficit)
            except Exception:
                LOGGER.exception("loss refresh failed; the campaign continues")
                continue
            if loaded:
                LOGGER.info("losses: %d new game(s) the lineage lost", loaded)
            if where:
                self.log.log(where)
                LOGGER.info(
                    "losses: %.0f games, gap %s at day 10, %s over days 20-29",
                    where["losses/games"],
                    f"{where['losses/gap_day10']:+,.0f}",
                    f"{where['losses/gap_days20_29']:+,.0f}",
                )

    async def harvesting(self) -> None:
        """Take newly published kernels into the pool, for as long as the run lasts.

        The other half of the ratchet. Promotions add champions and nothing
        else adds anything, so a pool left alone becomes this campaign's own
        lineage playing itself -- which it did: 69 champions against 12
        published agents, and those frozen on the day someone last ran the
        harvest by hand.

        Discovery, the download, the build and the 720-step check all go to a
        thread, because each is slow and none of them is the pool's. The pool
        is changed here, on the loop, where `gate.promote` also changes it and
        nothing runs at the same time. A load-modify-save from that thread
        would quietly drop any champion promoted while it was downloading.

        A failed harvest is not a failed campaign: the competition's API is
        somebody else's uptime, and a run that has been evaluating for hours
        must not end because a listing timed out.
        """
        while True:
            await asyncio.sleep(HARVEST_INTERVAL_SECONDS)
            try:
                found = await asyncio.to_thread(
                    harvest.vendored, HARVEST_LIMIT, set(self.pool.opponents)
                )
            except Exception:
                LOGGER.exception("harvest failed; the campaign continues")
                continue
            if not found:
                continue
            self.pool.opponents.update(found)
            self.pool.save(self.paths.pool)
            LOGGER.info(
                "harvest: %d new opponent(s) (%s), pool now %d",
                len(found),
                ", ".join(sorted(found)),
                len(self.pool.opponents),
            )

    async def work(self) -> None:
        """``config.SESSIONS`` workers in one task group, and every gate they fire."""
        try:
            async with self.group as group:
                for _ in range(config.SESSIONS):
                    group.create_task(self.worker())
        except BaseExceptionGroup as failures:
            raise _first(failures) from failures

    async def worker(self) -> None:
        """Spec section 3: sessions, one after another, until the run is done."""
        while self.remaining > 0:
            self.remaining -= 1
            await self.session()

    async def session(self) -> None:
        """One lineage, and the rounds that take it as far as it goes.

        The loop scores the starting program before the first round, so the
        first round is composed from the same verdict and the same day table
        as every later one and there is no opening special case. A round
        continues from its own previous program; a new session starts again
        from the champion. Depth within, breadth across.

        The session ends when a round clears the bar, when a call never ran to
        a verdict, or once it has spent ``ROUNDS_PER_OPPONENT`` rounds on every
        opponent in the pool. A round that was
        rejected is
        not the end of one: the reason goes in the ledger and the next round's
        message carries it back, which is what "your program did not parse" is
        worth.

        Raises:
            RuntimeError: The program this session starts from cannot be
                scored. It is the floor, and the floor is a pool opponent, so
                there is nothing left to run.
        """
        # A champion is what stagnation is measured against: with none, every
        # session already starts from the draw, and the note below would tell
        # a model that a line it never left is stuck.
        stagnant = (
            self.state.champion is not None
            and self.state.sessions_since_promotion >= STAGNATION_SESSIONS
        )
        source, name = self.start(stagnant)
        # Not caught: a starting program that raises is a broken floor, and
        # the floor is in the pool, so the next evaluation of anything would
        # raise too. The run stops rather than mutating what cannot play.
        result = await self.measure(source, name)
        # A session that starts from the champion has just measured it on the
        # current block, which is the only thing a candidate on that block can
        # fairly be compared against. `gate` reads it; `champion.json` carries
        # the result the champion was promoted on and cannot be repeated.
        standing = self.state.champion
        if standing is not None and str(source) == standing.path:
            self.champion_baseline = (standing.path, tuple(result.seeds), result)
        # One instruction, the same every round, in the template; what a
        # session can add is the note that it did not start from the champion.
        drawn = prompt.INSTRUCTION_NAME
        note = (
            STAGNATION_NOTE.format(sessions=self.state.sessions_since_promotion)
            if stagnant
            else ""
        )
        rounds = 0
        # A different game every round, so a session's rounds see as many maps as
        # it has rounds and no change gets twelve consecutive attempts at
        # entrenching on one. The campaign's feedback buys variety; verification
        # is the round's own, with `measure.py` in its directory.
        #
        # The seed was held across a session for a while, which gave the scarce
        # channel to depth the round can get for itself.
        # Every opponent, `ROUNDS_PER_OPPONENT` consecutive rounds each, so the
        # session's length is the pool's rather than a constant. A promotion ends
        # it long before the pool is exhausted in practice.
        # The games this program played, off the record rather than out of the
        # result: a result carries no games once it has been through the disk,
        # and every game the campaign scores is written the moment it is
        # scored, so the database is the one place they always are. Keyed the
        # way `record` wrote them, name included; `compose` has the name.
        scored = await asyncio.to_thread(games.recorded, name, games.DATABASE)
        # Only the games whose opponent this campaign can still put in the box.
        # The record outlives the pool: it holds every game ever played, and an
        # opponent can leave under a key nothing resolves any more, while a
        # round is handed its opponent as a file.
        scored = [one for one in scored if one[1].opponent in self.pool.opponents]
        if not scored:
            raise RuntimeError(f"{name} has no games on the record to hand a round")
        # Blocked by opponent rather than by the matchup number `record`
        # assigned, because those numbers are assigned afresh every time a
        # program is measured and mean nothing across two measurements. The
        # opponent is the thing a block of rounds is about.
        matchups = sorted({one[1].opponent for one in scored})
        planned = ROUNDS_PER_OPPONENT * len(matchups)
        for turn in range(planned):
            # `ROUNDS_PER_OPPONENT` consecutive rounds on one opponent,
            # then the next, with the season advancing inside the block. Four
            # attempts against one agent on four maps: enough to learn it,
            # without four attempts at the same game.
            block, attempt = divmod(turn, ROUNDS_PER_OPPONENT)
            against = [
                one
                for one in scored
                if one[1].opponent == matchups[block % len(matchups)]
            ]
            playing = against[attempt % len(against)]
            message = prompt.compose(name, playing, note)
            # The opponent's program goes into the round's directory as a
            # file, so the message names it and carries no path to it.
            opponent = Path(self.pool.opponents[playing[1].opponent])
            outcome = await self.round(source, name, result, message, drawn, opponent)
            rounds += 1
            if outcome is None:
                break
            source, name, result, cleared = outcome
            # The gate already ran, inside the round, under the promotions
            # lock, and said so.
            if cleared:
                LOGGER.info("%s promoted: the session is done", name)
                break
        self.finish(rounds)

    async def round(
        self,
        source: Path,
        name: str,
        result: Result,
        message: str,
        drawn: str,
        opponent: Path,
    ) -> tuple[Path, str, Result, bool] | None:
        """One round: one codex call on one file, and the loop's verdict on it.

        The call is given a directory holding ``child.py`` and nothing else --
        a copy of the program to improve -- and the whole message on standard
        input. The directory goes away as soon as what it holds is in the
        database.

        Args:
            source: The program this round starts from.
            name: What that program is called, for the failures it earns.
            result: The loop's verdict on that program, carried through so a
                rejected round hands the same program to the next one.
            message: The composed prompt for this round.
            opponent: This round's opponent, copied into the directory as
                ``opponent.py``.
            drawn: The name of the drawn instruction, recorded on the program.

        Returns:
            The program the next round continues from, its id, its verdict, and
            whether the gate promoted it. That is what this round wrote, or what
            it started from when the round was rejected or lost ground. None
            when the call never ran to a verdict, and the session ends there.
        """
        program_id = f"p{uuid.uuid4().hex[:12]}"
        # The directory owns its own removal. Written as `mkdtemp` and a
        # `finally`, the cleanup was a line somebody had to remember to put in
        # the right place, and twice it was not: first below the awaits, where
        # a cancelled codex call unwound straight past it and five hundred and
        # forty-eight workspaces collected in /tmp, and again as a `finally`
        # that had to be argued about -- synchronous, because an `await` in a
        # finally during cancellation is how a cleanup gets cancelled too.
        # `__exit__` runs on every one of those paths without being asked, so
        # the question stops being one anybody can get wrong.
        #
        # `ignore_cleanup_errors` keeps what `rmtree(ignore_errors=True)` gave
        # us: a call is free to leave a read-only file or a directory it
        # cannot remove behind, and a workspace that will not delete is not a
        # reason to lose the round that ran in it.
        with tempfile.TemporaryDirectory(
            prefix="campaign-round-", ignore_cleanup_errors=True
        ) as scratch:
            box = Path(scratch)
            workspace.prepare(
                box, source, self.database, self.mutator.SKILLS_DIRS, opponent
            )
            # What the campaign's own source says before this round runs, so
            # that what it says afterwards can be put back. `workspace.restored`
            # carries why that is necessary.
            before = workspace.kept()
            # A round that writes a `plan.json` the schema refuses is a round
            # that produced nothing runnable, which is an ordinary outcome of
            # asking a model to edit a file -- and until 2026-09-20 it was a
            # crash. `gather` validates, nothing caught what it raised, and the
            # error went up through the driver, the worker and the task group
            # and took both lineages down with it.
            #
            # It surfaced on the retry path, which is how a provider outage
            # became a campaign outage: the first call failed on an exhausted
            # quota, the fallback re-entered the driver, and `gather` read the
            # workspace the first model had already written a broken plan into.
            #
            # Caught here rather than in each driver because every round goes
            # through this function, so a fourth driver cannot forget it.
            try:
                mutation = await self.mutator(box, message, program_id)
            except plan.BrokenPlanError as broken:
                LOGGER.warning("%s wrote an unusable plan: %s", program_id, broken)
                mutation = Mutation(
                    program_id=program_id,
                    child=None,
                    status="no_output",
                    reason=str(broken),
                    seconds=0.0,
                    input_tokens=0,
                    output_tokens=0,
                    model=asked_model(),
                )
            workspace.transcripts(
                box, self.mutator.TRANSCRIPTS, self.paths.rounds / program_id
            )
            mutation = workspace.restored(before, mutation, program_id)
            mutation = workspace.packed(mutation, box, program_id)
            kept = await self.keep(mutation, name, drawn, program_id, source)
        self.state.calls += 1
        # Section 10, on the `calls` axis: one line per codex call.
        record: dict[str, float | str] = {
            "calls": self.state.calls,
            "calls/ok": int(mutation.child is not None),
            "calls/fallback": int(mutation.fallback),
            "calls/model": mutation.model,
            "calls/input_tokens": mutation.input_tokens,
            "calls/output_tokens": mutation.output_tokens,
            "calls/seconds": mutation.seconds,
            "database/programs": len(self.database.programs),
            "database/top": self.database.top(1)[0].fitness,
            # What the best program banks against the opponents it sweeps,
            # which is the axis `database/top` cannot show: that rate sits
            # near 0.99 whatever happens to the economy underneath it.
            "database/top_margin": archive.swept_margin(self.database.top(1)[0]),
        }
        if kept is not None:
            # Named apart from `result`, which is the program this round was
            # given and is what the comparison below is against. Rebinding it
            # here made that comparison the child against itself.
            scored = kept.result
            record["calls/fitness"] = scored.fitness
            record["calls/pool"] = len(scored.rates)
        self.log.log(record)
        if mutation.status == "exec_error":
            # No verdict, and no failure on the lineage either: there is
            # nothing to tell a next round, so the worker starts a new
            # session rather than spending the rest of this one's budget.
            return None
        # A rejected round continues from what it started from. Its reason is
        # in the ledger and the next message carries it back, which is worth
        # more than throwing away the rounds that remain.
        if kept is None:
            return source, name, result, False
        # Neither does a round that made the program worse. A session used to
        # continue from whatever its last round wrote, which makes a sequence
        # of rounds a random walk rather than a climb: on 2026-09-13 one
        # session wrote a program scoring 0.000, then spent three more rounds
        # editing that, four of the run's thirteen rounds spent below 0.14
        # while the program they started from scored 0.316.
        #
        # A promotion is taken whatever the comparison says. The gate is a
        # higher bar than this one and it has already run.
        if not kept.cleared and not kept.result.beats(result):
            LOGGER.info(
                "%s scored %.3f where %s scored %.3f: the next round starts "
                "from %s again",
                kept.name,
                kept.result.fitness,
                name,
                result.fitness,
                name,
            )
            return source, name, result, False
        return kept.source, kept.name, kept.result, kept.cleared

    async def keep(
        self,
        mutation: Mutation,
        started_from: str,
        drawn: str,
        program_id: str,
        parent: Path,
    ) -> Kept | None:
        """Validate what a call wrote, score it, insert it, and gate the top K.

        A call that never ran to a verdict -- the provider refused, or codex
        died -- leaves no failure behind: nothing about the lineage caused it.
        An `OpponentCrash` is not its failure either, and is not caught
        anywhere below `drive`: the pool is broken, so the run stops.

        Args:
            mutation: What the call produced.
            started_from: The id or name the call was editing.
            drawn: The name of the drawn instruction.
            program_id: The child program id.
            parent: The program this was edited from, so the record can carry
                what the edit did and not only what it scored. Both plans are
                files here and nowhere later.

        Returns:
            What the round produced and what the gate said about it, or None.
        """
        if mutation.status == "exec_error":
            LOGGER.warning("%s: no verdict (%s)", program_id, mutation.reason[:200])
            self.no_verdict += 1
            if self.no_verdict >= NO_VERDICT_LIMIT:
                raise SystemExit(
                    f"{self.no_verdict} calls in a row ran to no verdict, the "
                    f"last on {mutation.model}: {mutation.reason[:200]}"
                )
            return None
        self.no_verdict = 0
        if mutation.child is None:
            self.fail(started_from, drawn, f"{mutation.status}: {mutation.reason}")
            return None
        verdict = await asyncio.to_thread(validate.validate, mutation.child, 720)
        if verdict.status != "ok":
            self.fail(started_from, drawn, f"{verdict.status}: {verdict.reason}")
            return None
        stored = self.database.store(
            mutation.child.read_text(encoding="utf-8"), program_id
        )
        try:
            # The cheap half first. A sweep is 184 opponents where 126 sit at
            # 1.000 against the champion and cannot move any verdict, and
            # condition 1 is already read over the 58 that can -- so a
            # candidate that condition turns away was decided by a third of
            # the games. Both gates run on 2026-09-22 were exactly that, an
            # hour each to refuse on a number settled in the first twenty
            # minutes.
            #
            # The same test on the same seasons, so it cannot refuse a
            # candidate the full gate would promote; anything it passes plays
            # the whole pool and carries the per-opponent regression check
            # over every opponent, the swept ones included.
            against = self.screening()
            if against is not None:
                names = gate.contested(
                    against, sorted(set(against.result.rates) - {against.name})
                )
                screen = await self.measure(stored, program_id, only=names, keep=False)
                held, why = gate.screened(screen, against)
                if not held:
                    # Recorded the way a refusal is, not the way a failure is:
                    # the round ran, produced a program and was judged, and the
                    # next round is shown what it scored.
                    self.database.add(
                        archive.measured(
                            program_id,
                            stored,
                            started_from,
                            drawn,
                            mutation.model,
                            screen,
                            parent,
                        )
                    )
                    LOGGER.info("%s %s screened: %s", program_id, drawn, why)
                    return Kept(stored, program_id, screen, False)
            result = await self.measure(stored, program_id)
        except OpponentCrash:
            raise
        except RuntimeError as error:
            self.fail(started_from, drawn, f"gate: {error}")
            return None
        self.database.add(
            archive.measured(
                program_id,
                stored,
                started_from,
                drawn,
                mutation.model,
                result,
                parent,
            )
        )
        LOGGER.info("%s %s gate %.3f", program_id, drawn, result.fitness)
        cleared = await self.consider(program_id, result)
        return Kept(stored, program_id, result, cleared)

    async def consider(self, program_id: str, result: Result) -> bool:
        """Promote ``program_id`` if it topped the tournament it was just in.

        This is the whole gate. There was a second one -- the best three
        programs went on to a sealed block and only that could promote -- and
        it is gone, because the ranking that chose those three was noise: 78
        of 471 programs topped it and none of them survived the block. One
        measurement deep enough to select on decides, and it decides here,
        against the pool this program actually played.

        Under `promotions` because a promotion renumbers the pool, writes the
        floor and enrolls a champion, and eight sessions reach this line.
        """
        async with self.promotions:
            baseline = self.state.champion
            # The pool is copied per evaluation and eight run at once, so one
            # can promote while another is mid-game, and a result measured
            # against a floor that has since moved cannot be promoted over the
            # new one: the bar is a rating gap against a named agent, and a
            # gap against an agent this program never played is not a
            # measurement. The next round measures against the floor as it now
            # stands.
            #
            # It used to demand every pool opponent, which was right while the
            # pool *was* the tournament and a candidate played all eight of
            # it. Under a sampled draw it is unsatisfiable by construction --
            # sixteen opponents drawn from sixty-one leaves forty-five
            # "missing" every time -- and it silently blocked every promotion
            # for seven hours, invisibly, because the record below is the only
            # thing that logs a gate and this returned above it.
            # The champion as it stands, not its pool key. The key is fixed at
            # `config.POOL_CHAMPION` so that a promotion cannot collide with a
            # harvested name, and that made the staleness check below dead: a
            # candidate measured against the previous champion still finds the
            # key present and was credited with beating the current one.
            # `champion_4` was promoted on 2026-09-13 for beating a program it
            # never played. The file it played is what identifies it.
            standing = self.state.champion
            floor = standing.name if standing is not None else None
            measured_against = self.champion_played.get(program_id)
            if (
                standing is not None
                and measured_against is not None
                and measured_against != standing.path
            ):
                LOGGER.info(
                    "%s: measured against %s, and the champion is now %s; not promoted",
                    program_id,
                    Path(measured_against).stem,
                    Path(standing.path).stem,
                )
                self.log.log(
                    {
                        **self.promotion_record(result, False, baseline),
                        "gate/stale": 1,
                    }
                )
                return False
            if floor is not None and floor not in result.rates:
                LOGGER.info(
                    "%s: measured before %s became the floor; not promoted",
                    program_id,
                    floor,
                )
                # Marked, so a gate that turned a program away for being stale
                # is distinguishable in the record from one that judged it and
                # said no. Told apart nowhere, seven hours of rejections read
                # exactly like seven hours of candidates that were not good
                # enough.
                self.log.log(
                    {
                        **self.promotion_record(result, False, baseline),
                        "gate/stale": 1,
                    }
                )
                return False
            standing = self.paired(standing, result)
            if standing is None and self.state.champion is not None:
                LOGGER.info(
                    "%s: the champion has not been measured on these seasons; "
                    "not promoted",
                    program_id,
                )
                # A different series from `gate/stale`, because it is a
                # different fault with a different owner. `gate/stale` is about
                # the candidate: it played a champion that has since been
                # replaced, or never played the floor, and nothing can be done
                # for it. This is about us: the candidate is fine and we are
                # holding no measurement of the champion on its block, so the
                # comparison cannot be made for want of bookkeeping.
                #
                # They were one marker until 2026-09-26, and that is why five
                # candidates in five and a half hours were refused unjudged --
                # two at 0.978 -- while the record read exactly like a run of
                # candidates that were not good enough.
                self.log.log(
                    {
                        **self.promotion_record(result, False, baseline),
                        "gate/unpaired": 1,
                    }
                )
                return False
            verdict, why = gate.promotion(result, standing)
            # Logged either way. `why` is the only account of what the gate
            # decided and it used to be written only when the answer was yes:
            # 79 programs were turned away with a precise reason -- "25 of 28
            # at -1.983, below <agent> at +1.634" -- and every copy of it was
            # discarded, so the absence of a champion had no explanation
            # anywhere in the log or in wandb.
            LOGGER.info("%s %s", program_id, why)
            if verdict:
                # The file work in a thread; the pool it joins on the loop,
                # where the other seven workers are reading it.
                program = self.database.get(program_id)
                champion = await asyncio.to_thread(
                    gate.promote, program, result, self.paths
                )
                # Nothing leaves. The pool kept the highest-rated eight until
                # now, and it had discarded champion_1 -- which counters our
                # current champion at 0.729 where its rating says 0.994. In a
                # field this non-transitive, rating low against everyone and
                # beating *us* are different facts, and the trim could only
                # see the first.
                gate.enroll(champion, self.pool, self.paths)
                LOGGER.info("pool: %d opponents", len(self.pool.names()))
                self.state.champion = gate.record(champion, self.paths)
                self.database.promoted_as(program_id, champion.name)
                self.state.sessions_since_promotion = 0
                # The new champion's measurement on this block, which is the
                # result it was just promoted on: the file `gate.promote`
                # copied is these bytes, and they were measured minutes ago on
                # `self.block` against the pool as it then stood.
                #
                # Without this the gate stops judging. A baseline was set in
                # one place -- a session that happens to start from the
                # champion -- and `paired` refuses any candidate whose block or
                # champion it does not match, so between a promotion and the
                # next session that starts from the new champion, every
                # candidate is turned away unjudged however good it is.
                # Measured live on 2026-09-26: the last session to start from
                # the champion did so at 03:33, champion_43 took the slot at
                # 05:55, and the five candidates gated over the following five
                # and a half hours were all refused, two of them at 0.978. A
                # session is thousands of rounds long and ends on a promotion,
                # so "the next session to start from the champion" can be
                # hours away or never.
                #
                # Free and exact, which is why it goes here rather than in a
                # re-measurement: the candidate's own evaluation is the
                # champion's evaluation, the same bytes on the same seeds.
                self.champion_baseline = (
                    self.state.champion.path,
                    tuple(result.seeds),
                    result,
                )
                # No field refresh. It played the new champion's pairings so a
                # Bradley-Terry fit would have edges for it, and nothing reads
                # that fit any more: the bar is a win rate and a head-to-head,
                # both measured in the candidate's own evaluation.
                #
                # It cost 8.5 minutes of this lock per promotion, measured
                # 2026-09-13 -- 24 pairings at 768 games -- and every other
                # session's gate queued behind it. Three candidates were waiting
                # when the first promotion of that run finished, one of them a
                # better program than the one that had just promoted.
                artifact = wandb.Artifact(
                    champion.name, "champion", metadata=result.model_dump()
                )
                artifact.add_file(champion.tarball)
                self.log.log_artifact(artifact)
            self.log.log(self.promotion_record(result, verdict, baseline))
            self.champion_played.pop(program_id, None)
            return verdict

    def paired(self, standing: "Champion | None", result: Result) -> "Champion | None":
        """The champion as measured on the seasons this candidate played.

        The gate's question is whether a candidate beat the champion, and the
        two have to have been measured on the same world for that to be a
        question at all. The seasons are most of the world: an unchanged agent
        through five blocks fitted ratings from -3.466 to -2.226, and
        champion_12 scored 0.454 over fifty shared opponents on the block it
        was promoted on and 0.205 over the same fifty eight hours later. The
        promotion bar is 0.035, so a rotation swamps it by a factor of seven.

        Args:
            standing: The champion, carrying the result it was promoted on.
            result: The candidate's evaluation, whose seeds say which block it
                played.

        Returns:
            The champion with its measurement on those seasons, or None when
            there is no such measurement and the comparison cannot be made.
            None with no champion at all is champion zero's own gate and is
            not a refusal; the caller tells the two apart.
        """
        if standing is None or self.champion_baseline is None:
            return None
        path, seeds, measured = self.champion_baseline
        if path != standing.path or seeds != tuple(result.seeds):
            return None
        return standing.model_copy(update={"result": measured})

    def screening(self) -> "Champion | None":
        """The champion the screen compares against, or None to skip it.

        `paired` asks the same question of a finished evaluation and answers it
        from that evaluation's seeds. The screen runs before there is one, so
        this takes the baseline directly and checks only that it belongs to the
        champion now standing -- the block is the current one by construction,
        because the screen and the sweep that follows it share `seasons`.

        Returns:
            The champion carrying the measurement to compare against, or None
            when there is nothing comparable and the sweep should just run.
        """
        if self.state.champion is None or self.champion_baseline is None:
            return None
        path, _, measured = self.champion_baseline
        if path != self.state.champion.path:
            return None
        return self.state.champion.model_copy(update={"result": measured})

    def floor(self) -> str | None:
        """The champion's name, or None before there is one.

        The bar a candidate has to beat by a margin it can show. Read
        from state rather than passed down, because eight workers reach the
        gate concurrently and the floor may have moved since a round began --
        which is the correct behaviour: a candidate is judged against the
        champion that stands when it is judged.
        """
        return self.state.champion.name if self.state.champion else None

    async def measure(
        self,
        source: Path,
        program_id: str,
        only: Sequence[str] | None = None,
        keep: bool = True,
    ) -> Result:
        """Play ``source`` against the pool as it stands, off the loop thread.

        This is the campaign's only measurement of a round, and the only one
        a model is ever shown: the model plays nothing, so the seeds, the
        seating and the reading of "won" are all ours.

        """
        pool = self.snapshot()
        if only is not None:
            # The screen's pool: the opponents that can still move, without
            # the 126 at 1.000 whose games cannot change the answer it asks.
            wanted = set(only)
            pool = Pool(
                opponents={
                    name: path
                    for name, path in pool.opponents.items()
                    if name in wanted
                }
            )
        # Recorded before a game is played, so the gate can tell afterwards
        # whether the champion it is being compared against is the one it met.
        self.champion_played[program_id] = pool.opponents.get(config.POOL_CHAMPION, "")
        # `seasons` first and on its own line, because it is what rotates the
        # blocks: read inside the call's argument list it would depend on
        # left-to-right evaluation order to leave `self.duel` current.
        block = self.seasons()
        result = await asyncio.to_thread(
            evaluator.score,
            source,
            program_id,
            pool,
            random.Random(self.rng.random()),
            block,
            self.workers,
            self.paths.pool,
            self.duel,
        )
        # Every game of it, into the one database the nightly extraction also
        # writes to. Every candidate, not only the ones that survive: what
        # separates a program that promoted from one that did not is a
        # question about the ones that did not, and it cannot be asked of
        # games nobody kept. Measured at 9.6 MB and 1.3s an evaluation, which
        # is about 11 GB a day at eight sessions -- affordable against the
        # competition's remaining weeks, and off the loop thread either way.
        if keep:
            await asyncio.to_thread(
                games.record,
                program_id,
                games.played(result, program_id),
                games.DATABASE,
            )
        return result

    def seasons(self) -> list[int]:
        """The seeds every candidate measured in this block plays.

        Shared, because a season is most of what a rating measures and two
        candidates ranked on different ones are barely being compared. One
        unchanged agent through the gate five times, opponents held fixed and
        only the seeds moving, gave fitted ratings from -3.466 to -2.226 -- a
        standard deviation of 0.491, where varying the *opponents* instead
        moved it 0.070. The maps are seven times the draw.

        Rotated, because a set that never moves is one the search can be
        selected against. `SEED_ROTATION` candidates share a block and
        then it is redrawn from the whole range, so no program is measured for
        long on maps its ancestors were selected on -- which is what the
        reserved held-out block used to be for, and why there is no longer
        one.

        Called on the loop thread, where the counter is nobody else's.
        """
        if self.measured % SEED_ROTATION == 0:
            # One sample split in two, not two samples. `random.sample` cannot
            # repeat within a draw, so the sweep's seasons and the champion
            # pairing's extra seasons are disjoint by construction -- and a
            # season played in both blocks would be the same game counted
            # twice in that pairing's rate.
            drawn = self.rng.sample(config.GATE_SEED_RANGE, config.DUEL_SEEDS)
            self.block = drawn[:GATE_SEEDS]
            self.duel = drawn[GATE_SEEDS:]
            LOGGER.info(
                "seasons: a fresh block of %d, and %d more against the "
                "champion, after %d candidates",
                GATE_SEEDS,
                len(self.duel),
                self.measured,
            )
        self.measured += 1
        return self.block

    def snapshot(self) -> Pool:
        """The pool as it stands, copied on the loop for one evaluation to keep."""
        return self.pool.model_copy(deep=True)

    def start(self, stagnant: bool) -> tuple[Path, str]:
        """The program this session begins from, and the name it goes by.

        The name is a pool name or a database id, never a path: it is
        interpolated raw into the session's own message, and it is also the
        pool entry the starting program is not scored against. Until the
        first promotion there is no champion, so every session starts from the
        draw below -- which is why that draw has to be a selection and not a
        shuffle.
        """
        if self.rng.random() < SCRATCH_CHANCE:
            return self.scratch()
        champion = self.state.champion
        if champion is not None and not stagnant:
            return Path(champion.path), champion.name
        candidates = self.database.top(PARENT_POOL)
        # Weighted by rank, not uniform: `PARENT_POOL` says why.
        weights = [PARENT_DECAY**rank for rank in range(len(candidates))]
        program = self.rng.choices(candidates, weights=weights)[0]
        return Path(program.source_path), program.id

    def scratch(self) -> tuple[Path, str]:
        """A session that begins outside the champion's ancestry.

        Every champion descends from `SEED`, a harvested public agent,
        so every round edits one program's sixty-fifth-generation descendant
        and no instruction reaches outside that basin. This is the only
        starting point the campaign has that does not.

        The scratch lineage is parented from its own best rather than from the
        database's, and that is what makes it more than a lottery ticket. A
        program that begins from nothing scores far below a champion, and
        `Database.top` ranks on the win rate -- so without this it would be
        scored once, never drawn again, and the lineage would die in a single
        session however promising it was. Parented from itself it gets a
        ratchet of its own, ordered the way `top` orders, and joins the global
        draw when its record earns a place there rather than being asked to
        earn one immediately.

        Returns:
            The program to start from and the name it goes by: the best
            scratch-descended program, or the blank slate when there is none.
        """
        grown = self.database.descendants(SCRATCH_ID)
        if grown:
            best = max(
                grown,
                key=lambda program: (program.fitness, archive.swept_margin(program)),
            )
            return Path(best.source_path), best.id
        blank = self.database.store(SCRATCH_AGENT, SCRATCH_ID)
        return blank, SCRATCH_ID

    def fail(self, started_from: str, instruction: str, reason: str) -> None:
        """Record an attempt that produced no program, and say why."""
        LOGGER.warning("%s: %s", started_from, reason)
        self.database.record_failure(
            archive.Failure(
                started_from=started_from,
                instruction=instruction,
                reason=reason,
                created=time.time(),
            )
        )

    def finish(self, rounds: int) -> None:
        """Count the session, persist the state, and log its line."""
        self.state.sessions += 1
        self.state.sessions_since_promotion += 1
        resume = state_file(self.paths)
        resume.parent.mkdir(parents=True, exist_ok=True)
        resume.write_text(self.state.model_dump_json(indent=2), encoding="utf-8")
        self.log.log(
            {
                "sessions": self.state.sessions,
                "sessions/rounds": rounds,
                "sessions/since_promotion": self.state.sessions_since_promotion,
            }
        )

    def _lines(self, program_id: str) -> int:
        """How many lines the program is, so its growth is on the record."""
        try:
            source = Path(self.database.get(program_id).source_path)
            return len(source.read_text(encoding="utf-8").splitlines())
        except (KeyError, OSError):
            return 0

    def promotion_record(
        self,
        result: Result,
        promoted: bool,
        baseline: Champion | None,
    ) -> dict[str, float]:
        """Section 10: one program judged, and whether it moved the floor.

        ``baseline`` is the floor as it stood when this was judged, read
        before the promotion: after it, a promoting result would be logged
        beside the score of the champion it replaced -- itself.

        Args:
            result: The measurement it was judged on.
            promoted: Whether it cleared the gate.
            baseline: The floor as it stood when this was judged.
        """
        record: dict[str, float] = {
            "sessions": self.state.sessions,
            "gate/score": result.fitness,
            "gate/promoted": int(promoted),
            "gate/pool": len(result.rates),
            # Nothing has been watching this. The seed is 619 lines and
            # `SEED` says why that matters -- a round is handed the
            # whole program, and a big one spends the call being read rather
            # than improved. champion_69 is 3,220 lines, five times its own
            # ancestor, and the growth arrived a few hundred lines at a time
            # with no measurement to show it.
            "gate/lines": self._lines(result.program_id),
            # How much of this candidate's verdict against the floor rests on
            # games that were actually decided. A run whose candidates keep
            # drawing the floor is a run that has stopped producing new
            # programs, and until this was recorded there was no way to see
            # that from the outside -- the win rates looked like 0.53 either
            # way.
            "gate/decisive": result.decisive.get(baseline.name if baseline else "", 0),
            **{f"gate/rate/{name}": rate for name, rate in result.rates.items()},
            **{f"gate/margin/{n}": m.mean for n, m in result.margins.items()},
        }
        if baseline is not None:
            record["gate/champion"] = baseline.result.fitness
        if promoted:
            # The series to watch. `gate/score` carries every program judged,
            # promoted or not, so the champions are a few dozen points buried
            # among hundreds; this is only ever written when the floor moves.
            record["champion/win_rate"] = result.fitness
            if baseline is not None and baseline.name in result.rates:
                # What it did to the champion it replaced, head to head. The
                # win rate above is against a pool that strengthens with every
                # promotion, so it is not comparable across the campaign; this
                # is always the same comparison -- the new floor against the
                # old one -- and a run of these near 0.5 is a search that has
                # stopped finding anything.
                record["champion/over_previous"] = result.rates[baseline.name]
        return record


def _first(failures: BaseExceptionGroup) -> BaseException:
    """The first leaf of a task group's exception group, unwrapped."""
    failure: BaseException = failures
    while isinstance(failure, BaseExceptionGroup):
        failure = failure.exceptions[0]
    return failure


if __name__ == "__main__":
    main()
