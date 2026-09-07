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
import shutil
import signal
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import wandb
from git import Git, Repo
from pydantic import BaseModel

# `gate` is also the name of a `Campaign` method, so annotations in the class
# body cannot see this module -- hence `Champion` imported by name below.
from kaggriculture.campaign import archive, config, evaluator, gate, prompt, validate
from kaggriculture.campaign.evaluator import Result
from kaggriculture.campaign.gate import Champion
from kaggriculture.campaign.harness import OpponentCrash
from kaggriculture.campaign.mutate import (
    CodexMutator,
    FakeMutator,
    Mutation,
    Mutator,
    validate_model,
)
from kaggriculture.campaign.pool import Pool

LOGGER = logging.getLogger(__name__)

# Threads the loop needs at once: one per session for whichever blocking step
# it is on -- scoring a program, validating one, reading a transcript, and
# never two at once within a session -- plus the one a promotion packages on.
THREADS = config.SESSIONS + 1

# The database id of the program a cold start seeds itself from. Until the
# first promotion there is no champion, so this is the name the first
# sessions are told they are editing and the bar their verdict quotes. It
# lives in `config` because `copycheck` needs the file it names -- the stored
# copy is what the campaign's lineage actually is -- and two modules agreeing
# on a magic string is how they come to disagree on one.
SEED_ID = config.SEED_ID

# What the wandb run records as its configuration: spec section 8's table.
HYPERPARAMETERS = (
    "SESSIONS ROUNDS_PER_SESSION GATE_SEEDS POOL_SIZE "
    "STAGNATION_SESSIONS CODEX_MODEL CODEX_FALLBACK_MODEL"
).split()

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
    if args.dry_run:
        _isolate(config.RUN / "dry-run")
    else:
        # A typo'd model is hundreds of failed sessions discovered one at a
        # time; caught here, before the run opens or a call is ever made.
        validate_model(config.CODEX_MODEL)
        if config.CODEX_FALLBACK_MODEL:
            validate_model(config.CODEX_FALLBACK_MODEL)
    # 1. wandb, named for the model and the code that produced the run.
    log = _open_run(dry_run=args.dry_run)
    mutator: Mutator = (
        FakeMutator(edit=lambda source: source + "\n# dry-run mutation\n")
        if args.dry_run
        else CodexMutator()
    )
    try:
        # 2-4. the event loop, the workers, and the gate they fire.
        run(args.sessions, mutator, args.workers, args.seed_agent, random.Random(), log)
    finally:
        log.finish()


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    """Parse the loop's command line; ``sys.argv[1:]`` when ``argv`` is None."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=10**9)
    # Section 7: every session in flight can be evaluating at once.
    parser.add_argument(
        "--workers", type=int, default=max(1, config.CORE_BUDGET // config.SESSIONS)
    )
    # Whatever this names is copied to `config.SEED_PROGRAM`, and that copy is
    # what `copycheck` exempts, so a run started from a snapshot exempts the
    # snapshot and a run started from the default exempts the default. There
    # is nothing here for the gate to disagree with.
    parser.add_argument("--seed-agent", type=Path, default=config.SEED)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="fake calls, no log, every write under run/campaign/dry-run",
    )
    return parser.parse_args(argv)


def _isolate(root: Path) -> None:
    """Point every file the campaign writes at ``root``, for a dry run.

    A dry run fakes the codex call and disables the log, but the rest of it
    is the campaign: it evaluates, inserts, gates and promotes. Left on the
    real paths it does that to the live database, the live pool and the live
    floor, which is a campaign silently corrupted by someone checking that
    the plumbing works. The pool moves too, so a dry run starts from the
    roster rather than a live pool holding champions it will then retire.

    Args:
        root: The directory the dry run owns.
    """
    for name in ("ARCHIVE", "PROGRAMS", "FLOOR", "CHAMPIONS", "CHAMPION"):
        setattr(config, name, root / getattr(config, name).relative_to(config.RUN))
    config.POOL = root / "pool.json"
    LOGGER.info("dry run: every write goes under %s", root)


def _open_run(dry_run: bool) -> wandb.Run:
    """Open the run this campaign logs to, named for the model and the revision.

    Two axes, because a session is many rounds now: everything a round
    produces is stepped by ``calls`` and everything a session or a gate
    produces by ``sessions``. One axis for both would file five rounds under
    one session number and keep the last.

    Exits on uncommitted changes under ``src/``: a run named by a hash has to
    be that hash.
    """
    # `Git` is bound to the directory and runs there. `Repo.git` is not: in a
    # linked worktree `Repo` resolves to the shared git directory with no
    # working tree of its own, and `git status` fails outright there.
    dirty = Git(config.ROOT).status("--porcelain", "--", "src")
    if dirty:
        raise SystemExit(f"uncommitted changes under src/:\n{dirty}")
    name = f"{config.CODEX_MODEL}-{Repo(config.ROOT).head.commit.hexsha[:7]}"
    log = wandb.init(
        entity=config.WANDB_ENTITY,
        project=config.WANDB_PROJECT,
        id=name,
        name=name,
        resume="allow",
        mode="disabled" if dry_run else "online",
        config={key: getattr(config, key) for key in HYPERPARAMETERS},
    )
    log.define_metric("sessions")
    log.define_metric("calls")
    log.define_metric("sessions/*", step_metric="sessions")
    log.define_metric("gate/*", step_metric="sessions")
    log.define_metric("calls/*", step_metric="calls")
    log.define_metric("database/*", step_metric="calls")
    return log


def state_file() -> Path:
    """Where the state a restart resumes from lives, beside the database."""
    return config.ARCHIVE.with_name("state.json")


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
) -> State:
    """Load what a restart resumes, seed an empty database, drive the workers.

    ``sessions`` is how many this run starts; what is in flight when the last
    is taken is drained. Returns the state as of the last completed session.
    """
    resume = state_file()
    state = (
        State.model_validate_json(resume.read_text(encoding="utf-8"))
        if resume.exists()
        else State()
    )
    # `state.json` is written after every session, `champion.json` the moment
    # a champion is in the pool. A kill in between leaves the first stale, so
    # the second wins: otherwise the gate would restart with no baseline.
    champion = gate.load_champion()
    if champion is not None:
        state.champion = champion
        LOGGER.info("resuming on champion %s from champion.json", champion.name)
    pool = Pool.load(config.POOL) if config.POOL.exists() else Pool.initial()
    # A champion's name resolves through the pool file, so the pool on disk
    # must be current before anything plays a game.
    pool.save(config.POOL)
    database = archive.Database(config.ARCHIVE, config.PROGRAMS)
    if not database.programs:
        seed = evaluator.score(seed_agent, SEED_ID, pool, rng, workers)
        stored = database.store(seed_agent.read_text(encoding="utf-8"), SEED_ID)
        database.add(_program(SEED_ID, stored, "", "seed", "", seed))
        LOGGER.info("seeded from %s at fast fitness %.3f", seed_agent, seed.fitness)
    campaign = Campaign(state, database, pool, mutator, workers, rng, log)
    asyncio.run(campaign.drive(sessions))
    return campaign.state


def _program(
    program_id: str,
    source: Path,
    started_from: str,
    instruction: str,
    model: str,
    result: Result,
    standings: dict[str, float] | None = None,
) -> archive.Program:
    """One database entry, stamped now.

    ``standings`` is the tournament this program's evaluation was part of. It
    is optional because the cold start seeds the database before any pool is
    loaded, and a program with no rating is honestly recorded as having none.
    """
    ranked = sorted(standings or {}, key=lambda name: -(standings or {})[name])
    return archive.Program(
        id=program_id,
        source_path=str(source),
        started_from=started_from,
        instruction=instruction,
        model=model,
        fitness=result.fitness,
        field=result.field,
        rating=None if standings is None else standings[program_id],
        place=0 if standings is None else 1 + ranked.index(program_id),
        rates=result.rates,
        margins=result.margins,
        created=time.time(),
    )


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
    ) -> None:
        self.state = state
        self.database = database
        self.pool = pool
        self.mutator = mutator
        self.workers = workers
        self.rng = rng
        self.log = log
        self.remaining = 0
        # Calls in a row that ran to no verdict. A call that never reached the
        # model is nobody's failure, so it writes nothing and the worker
        # simply starts another -- which, when the cause is the login, the
        # model slug or the provider rather than one call, is a spin at full
        # rate that leaves the database empty while every counter advances.
        # It happened on this campaign's first launch: sixteen sessions in
        # forty-seven seconds over a missing flag.
        self.no_verdict = 0
        # A promotion renumbers the pool, writes the floor and joins the
        # champion; eight sessions promoting at once would race on all three.
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
        for number in (signal.SIGINT, signal.SIGTERM):
            running.add_signal_handler(number, work.cancel)
        try:
            await work
        except asyncio.CancelledError:
            LOGGER.warning("stopped on a signal at %d sessions", self.state.sessions)
        finally:
            for number in (signal.SIGINT, signal.SIGTERM):
                running.remove_signal_handler(number)

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
        a verdict, or after ``ROUNDS_PER_SESSION`` rounds. A round that was
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
            and self.state.sessions_since_promotion >= config.STAGNATION_SESSIONS
        )
        source, name = self.start(stagnant)
        # Not caught: a starting program that raises is a broken floor, and
        # the floor is in the pool, so the next evaluation of anything would
        # raise too. The run stops rather than mutating what cannot play.
        result = await self.measure(source, name)
        # Drawn once, for the whole session: rounds go deeper on one line and
        # sessions go wider, so a session that drew "a completely different
        # algorithm" three times out of five would be three first rounds
        # rather than one line taken further.
        drawn, instruction = self.rng.choice(prompt.INSTRUCTIONS)
        if stagnant:
            note = STAGNATION_NOTE.format(sessions=self.state.sessions_since_promotion)
            instruction = note + instruction
        rounds = 0
        for _ in range(config.ROUNDS_PER_SESSION):
            failures = self.database.failures(name)
            siblings = self.database.children(name)
            standings = gate.standing(
                name, result.rates, self.snapshot(), 2 * config.GATE_SEEDS
            )
            message = prompt.compose(
                name, result, failures, siblings, instruction, standings
            )
            outcome = await self.round(source, name, result, message, drawn)
            rounds += 1
            if outcome is None:
                break
            source, name, result = outcome
            cleared, why = gate.promotion(
                gate.standing(
                    name, result.rates, self.snapshot(), 2 * config.GATE_SEEDS
                ),
                name,
            )
            if cleared:
                LOGGER.info("%s %s: the session is done", name, why)
                break
        self.finish(rounds)

    async def round(
        self, source: Path, name: str, result: Result, message: str, drawn: str
    ) -> tuple[Path, str, Result] | None:
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
            drawn: The name of the drawn instruction, recorded on the program.

        Returns:
            The program the next round continues from, its id and its verdict.
            That is what this round wrote, or what it started from when the
            round was rejected. None when the call never ran to a verdict, and
            the session ends there.
        """
        program_id = f"p{uuid.uuid4().hex[:12]}"
        box = Path(tempfile.mkdtemp(prefix="campaign-round-"))
        child = box / "child.py"
        shutil.copy(source, child)
        # The gate writes a champion read-only so nothing can edit the file
        # the pool plays, and `shutil.copy` carries that mode across. This
        # copy is the one file the call must be able to write.
        child.chmod(0o644)
        mutation = await self.mutator(box, message, program_id)
        kept = await self.keep(mutation, name, drawn, program_id)
        await asyncio.to_thread(shutil.rmtree, box, ignore_errors=True)
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
            "database/top_field": max(p.field for p in self.database.programs),
        }
        rated = [p.rating for p in self.database.programs if p.rating is not None]
        if rated:
            record["database/top_rating"] = max(rated)
        if kept is not None:
            source, program_id, result = kept
            del source
            record["calls/fitness"] = result.fitness
            record["calls/field"] = result.field
            table = gate.standing(
                program_id, result.rates, self.snapshot(), 2 * config.GATE_SEEDS
            )
            record["calls/rating"] = table[program_id]
            record["calls/place"] = 1 + sorted(
                table, key=lambda name: -table[name]
            ).index(program_id)
            record["calls/pool"] = len(result.rates)
        self.log.log(record)
        if mutation.status == "exec_error":
            # No verdict, and no failure on the lineage either: there is
            # nothing to tell a next round, so the worker starts a new
            # session rather than spending the rest of this one's budget.
            return None
        # A rejected round continues from what it started from. Its reason is
        # in the ledger and the next message carries it back, which is worth
        # more than throwing away the rounds that remain.
        return kept if kept is not None else (source, name, result)

    async def keep(
        self, mutation: Mutation, started_from: str, drawn: str, program_id: str
    ) -> tuple[Path, str, Result] | None:
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

        Returns:
            The stored program, its id and its verdict, or None.
        """
        if mutation.status == "exec_error":
            LOGGER.warning("%s: no verdict (%s)", program_id, mutation.reason[:200])
            self.no_verdict += 1
            if self.no_verdict >= config.NO_VERDICT_LIMIT:
                raise SystemExit(
                    f"{self.no_verdict} calls in a row ran to no verdict, the "
                    f"last on {config.CODEX_MODEL} and {config.CODEX_FALLBACK_MODEL}: "
                    f"{mutation.reason[:200]}"
                )
            return None
        self.no_verdict = 0
        if mutation.child is None:
            self.fail(started_from, drawn, f"{mutation.status}: {mutation.reason}")
            return None
        verdict = await asyncio.to_thread(validate.validate, mutation.child)
        if verdict.status != "ok":
            self.fail(started_from, drawn, f"{verdict.status}: {verdict.reason}")
            return None
        stored = self.database.store(
            mutation.child.read_text(encoding="utf-8"), program_id
        )
        try:
            result = await self.measure(stored, program_id)
        except OpponentCrash:
            raise
        except RuntimeError as error:
            self.fail(started_from, drawn, f"gate: {error}")
            return None
        table = gate.standing(
            program_id, result.rates, self.snapshot(), 2 * config.GATE_SEEDS
        )
        self.database.add(
            _program(
                program_id, stored, started_from, drawn, mutation.model, result, table
            )
        )
        LOGGER.info("%s %s gate %.3f", program_id, drawn, result.fitness)
        await self.consider(program_id, result, table)
        return stored, program_id, result

    async def consider(
        self, program_id: str, result: Result, table: dict[str, float]
    ) -> None:
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
            # can promote while another is mid-game. A result that never
            # played the new champion cannot be said to have topped the pool
            # it is being judged against, and promoting it would ship a
            # program that never met the bar. The next round measures against
            # the pool as it now stands.
            missing = set(self.pool.names()) - {program_id} - set(result.rates)
            if missing:
                LOGGER.info(
                    "%s: measured before %s joined the pool; not promoted",
                    program_id,
                    ", ".join(sorted(missing)),
                )
                return
            verdict, why = gate.promotion(table, program_id)
            if verdict:
                LOGGER.info("%s %s", program_id, why)
                # The file work in a thread; the pool it joins on the loop,
                # where the other seven workers are reading it.
                program = self.database.get(program_id)
                champion = await asyncio.to_thread(gate.promote, program, result)
                gate.enroll(champion, self.pool)
                # The pool is the top of the tournament: the champion has just
                # joined it and the weakest opponent makes way, so the field a
                # candidate has to finish above is the strongest one there is.
                dropped = self.pool.trim(
                    {**table, champion.name: table[program_id]}, config.POOL_SIZE
                )
                if dropped:
                    self.pool.save(config.POOL)
                    LOGGER.info("pool: %s made way", ", ".join(dropped))
                # The champion joined the pool with no pairings of its own, and
                # `standing` plays nothing -- so until these are measured it
                # sits in every tournament on one edge, the row of whichever
                # candidate is being judged against it, and beating it drops
                # its rating far enough to make topping the standings easy.
                # Each promotion would buy the next one cheaply. Measured after
                # the trim, so nothing is played for an opponent just dropped.
                measured = await asyncio.to_thread(
                    gate.refresh,
                    self.snapshot(),
                    self.rng.sample(config.GATE_SEED_RANGE, config.GATE_SEEDS),
                    self.workers,
                )
                LOGGER.info(
                    "field: %d new pairing(s) for %s", len(measured), champion.name
                )
                self.state.champion = gate.record(champion)
                self.state.sessions_since_promotion = 0
                artifact = wandb.Artifact(
                    champion.name, "champion", metadata=result.model_dump()
                )
                artifact.add_file(champion.tarball)
                self.log.log_artifact(artifact)
            self.log.log(self.promotion_record(result, verdict, baseline, table))

    async def measure(self, source: Path, program_id: str) -> Result:
        """Play ``source`` against the pool as it stands, off the loop thread.

        This is the campaign's only measurement of a round, and the only one
        a model is ever shown: the model plays nothing, so the seeds, the
        seating and the reading of "won" are all ours.
        """
        return await asyncio.to_thread(
            evaluator.score,
            source,
            program_id,
            self.snapshot(),
            random.Random(self.rng.random()),
            self.workers,
        )

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
        champion = self.state.champion
        if champion is not None and not stagnant:
            return Path(champion.path), champion.name
        candidates = self.database.top(config.PARENT_POOL)
        # Weighted by rank, not uniform: `config.PARENT_POOL` says why.
        weights = [config.PARENT_DECAY**rank for rank in range(len(candidates))]
        program = self.rng.choices(candidates, weights=weights)[0]
        return Path(program.source_path), program.id

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
        resume = state_file()
        resume.parent.mkdir(parents=True, exist_ok=True)
        resume.write_text(self.state.model_dump_json(indent=2), encoding="utf-8")
        self.log.log(
            {
                "sessions": self.state.sessions,
                "sessions/rounds": rounds,
                "sessions/since_promotion": self.state.sessions_since_promotion,
            }
        )

    def promotion_record(
        self,
        result: Result,
        promoted: bool,
        baseline: Champion | None,
        standings: dict[str, float],
    ) -> dict[str, float]:
        """Section 10: one program judged, and whether it moved the floor.

        ``baseline`` is the floor as it stood when this was judged, read
        before the promotion: after it, a promoting result would be logged
        beside the score of the champion it replaced -- itself.

        Args:
            result: The measurement it was judged on.
            promoted: Whether it cleared the gate.
            baseline: The floor as it stood when this was judged.
            standings: The tournament that decided it, so the rating and the
                place go on the record beside the win rates behind them.
        """
        record: dict[str, float] = {
            "sessions": self.state.sessions,
            "gate/score": result.fitness,
            "gate/field": result.field,
            "gate/promoted": int(promoted),
            "gate/rating": standings[result.program_id],
            "gate/place": 1
            + sorted(standings, key=lambda name: -standings[name]).index(
                result.program_id
            ),
            "gate/pool": len(result.rates),
            **{f"gate/rate/{name}": rate for name, rate in result.rates.items()},
            **{f"gate/margin/{n}": m.mean for n, m in result.margins.items()},
        }
        if baseline is not None:
            record["gate/champion"] = baseline.result.fitness
        return record


def _first(failures: BaseExceptionGroup) -> BaseException:
    """The first leaf of a task group's exception group, unwrapped."""
    failure: BaseException = failures
    while isinstance(failure, BaseExceptionGroup):
        failure = failure.exceptions[0]
    return failure


if __name__ == "__main__":
    main()
