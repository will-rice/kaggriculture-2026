"""Eight coding agents, each handed the champion and told to beat it.

Spec section 1, four steps: name the run, start the event loop, spin off
``config.SESSIONS`` workers, and gate what they produce. A worker is section
3 -- build a sandbox from the champion, run a codex session in it, validate,
fast-evaluate and keep the child, and gate whatever enters the top
``DEEP_TOP_K`` -- and eight of them run that against one database, one pool
and one champion. A promotion by any worker changes what the other seven
start from next; that concurrency is the only structure there is.

One event loop on one thread owns the database, the pool, the state and the
wandb run, so none of them needs a lock. Everything that blocks happens
elsewhere and is awaited: the session is a child process in its own process
group, and validation, both evaluators, the promotion and the sandbox
removal go to threads.

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
import time
import uuid
from pathlib import Path

import wandb
from git import Repo
from pydantic import BaseModel

from kaggriculture.campaign import archive, config, evaluator, gate, prompt, validate
from kaggriculture.campaign.evaluator import DeepResult
from kaggriculture.campaign.harness import OpponentCrash
from kaggriculture.campaign.mutate import CodexMutator, FakeMutator, Mutation, Mutator
from kaggriculture.campaign.pool import Pool
from kaggriculture.report import spearman

LOGGER = logging.getLogger(__name__)

# The database id of the program a cold start seeds itself from. Until the
# first promotion there is no champion, so this is the name the first
# sessions are told they are editing and the bar their feedback quotes.
SEED_ID = "seed"

# What the wandb run records as its configuration: spec section 8's table.
HYPERPARAMETERS = (
    "SESSIONS SESSION_LIMIT_SECONDS GAME_LIMIT_SECONDS FAST_SEEDS DEEP_TOP_K "
    "DEEP_CONCURRENCY CHAMPION_WEIGHT POOL_CAP RETIRE_THRESHOLD WEAKNESS_CAP "
    "STAGNATION_SESSIONS CROSS_PROBABILITY CODEX_MODEL CODEX_FALLBACK_MODEL"
).split()

# Prepended to the instruction under stagnation, so PROMPT.md says that this
# session is not editing the champion, and why.
STAGNATION_NOTE = """child.py is not the champion this session. {sessions} \
sessions have passed with no promotion, so child.py is `{name}`, drawn from \
the database's top ten: the champion's line is stuck and this one may not be.

"""


def main(argv: list[str] | None = None) -> None:
    """``campaign loop``: eight agents, each told to beat the champion."""
    args = _arguments(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
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
    # Section 7: every session in flight can be evaluating at once, so one
    # evaluation's share is the core budget split between them.
    parser.add_argument(
        "--workers", type=int, default=max(1, config.CORE_BUDGET // config.SESSIONS)
    )
    parser.add_argument("--seed-agent", type=Path, default=config.SERVED)
    parser.add_argument("--dry-run", action="store_true", help="fake sessions, no log")
    return parser.parse_args(argv)


def _open_run(dry_run: bool) -> wandb.Run:
    """Open the run this campaign logs to, named for the model and the revision.

    Raises:
        SystemExit: ``src/`` has uncommitted changes; a run named by a hash
            has to be that hash.
    """
    repo = Repo(config.ROOT)
    dirty = repo.git.status("--porcelain", "--", "src")
    if dirty:
        raise SystemExit(f"uncommitted changes under src/:\n{dirty}")
    name = f"{config.CODEX_MODEL}-{repo.head.commit.hexsha[:7]}"
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
    log.define_metric("*", step_metric="sessions")
    return log


class State(BaseModel):
    """What survives a restart, written after every session.

    ``champion`` is the floor's deep result on the pool as it stands *after*
    the promotion that put it there, and ``champion_path`` its immutable copy
    under ``config.CHAMPIONS``: the file every session starts from.
    """

    sessions: int = 0
    champion: DeepResult | None = None
    champion_path: str | None = None
    sessions_since_promotion: int = 0


def run(
    sessions: int,
    mutator: Mutator,
    workers: int,
    seed_agent: Path,
    rng: random.Random,
    log: wandb.Run,
) -> State:
    """Load what a restart resumes, seed an empty database, then drive the workers.

    ``sessions`` is how many this run starts; whatever is in flight when the
    last one is taken is drained, so it completes and counts. Returns the
    state as of the last completed session.
    """
    state_file = config.ARCHIVE.with_name("state.json")
    state = (
        State.model_validate_json(state_file.read_text(encoding="utf-8"))
        if state_file.exists()
        else State()
    )
    # `state.json` is written after every session, `champion.json` before
    # `gate.promote` returns. A kill in between leaves the first stale and
    # the second current, so the second wins -- otherwise the gate would
    # restart with no baseline and promote a second time.
    champion = gate.load_champion()
    if champion is not None:
        state.champion, state.champion_path = champion.result, champion.path
        LOGGER.info("resuming on champion %s from champion.json", champion.name)
    pool = Pool.load(config.POOL) if config.POOL.exists() else Pool.initial()
    # A champion's name resolves through the pool file, so the pool on disk
    # must be current before anything plays a game.
    pool.save(config.POOL)
    database = archive.Database(config.ARCHIVE, config.PROGRAMS)
    if not database.programs:
        seed = evaluator.fast(seed_agent, pool, rng, workers)
        stored = database.store(seed_agent.read_text(encoding="utf-8"), SEED_ID)
        database.add(_program(SEED_ID, stored, "", "seed", seed.fitness, seed.rates))
        LOGGER.info("seeded from %s at fast fitness %.3f", seed_agent, seed.fitness)
    campaign = Campaign(state, database, pool, mutator, workers, rng, log)
    asyncio.run(campaign.drive(sessions))
    return campaign.state


def _program(
    program_id: str,
    source: Path,
    started_from: str,
    instruction: str,
    fitness: float,
    rates: dict[str, float],
) -> archive.Program:
    """One database entry, stamped now."""
    return archive.Program(
        id=program_id,
        source_path=str(source),
        started_from=started_from,
        instruction=instruction,
        fitness=fitness,
        rates=rates,
        created=time.time(),
    )


class Campaign:
    """One run: the shared state, and the coroutines that advance it.

    Every attribute here is read and written only from coroutines on the one
    event-loop thread, which is what makes the database, the pool, the state
    and the wandb run safe without a lock.
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
        self.deep = asyncio.Semaphore(config.DEEP_CONCURRENCY)
        # A promotion renumbers the pool, writes the floor and re-scores the
        # champion; two gates doing that at once would race on all three.
        self.promotions = asyncio.Lock()
        # Programs whose deep evaluation is in flight: their result is not in
        # the database yet, so without this the next session gates them again.
        self.gating: set[str] = set()
        self.group = asyncio.TaskGroup()
        self.state_file = config.ARCHIVE.with_name("state.json")

    async def drive(self, sessions: int) -> None:
        """Steps 2-4: the workers and their gates, stopping cleanly on a signal.

        SIGINT and SIGTERM cancel the work, which cancels every task in it: a
        cancelled session kills its own process group, so a restart never
        finds sessions from the run before it still alive.

        Raises:
            OpponentCrash: A pool opponent failed in its own seat, in a
                session or a gate. The task group has cancelled everything
                else by the time it arrives here.
        """
        self.remaining = sessions
        work = asyncio.ensure_future(self.work())
        running = asyncio.get_running_loop()
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
        """One session: the sandbox, the agent that edits it, and what it wrote.

        A session that never ran to a verdict -- the provider refused, or
        codex died -- leaves no failure behind: nothing about the lineage
        caused it and no child it produced could fix it.

        Raises:
            OpponentCrash: A pool opponent failed in its own seat. Not this
                lineage's failure and never its feedback: the pool is broken,
                so it comes out of here and stops the run.
        """
        program_id = f"p{uuid.uuid4().hex[:12]}"
        stagnant = self.state.sessions_since_promotion >= config.STAGNATION_SESSIONS
        source, name, bar, rates = self.start(stagnant)
        drawn, instruction, inspiration = self.instruction()
        if stagnant:
            note = STAGNATION_NOTE.format(
                sessions=self.state.sessions_since_promotion, name=name
            )
            instruction = note + instruction
        box = prompt.build_sandbox(
            program_id,
            source,
            instruction,
            inspiration,
            {"fitness": bar},
            rates,
            self.pool.weights,
            self.pool.weakest(rates) if rates else "",
            [f.reason for f in self.database.failures(name)][-3:],
            name,
        )
        mutation = await self.mutator(box, program_id)
        fitness = None
        if mutation.status == "exec_error":
            LOGGER.warning("%s: no verdict (%s)", program_id, mutation.reason[:200])
        elif mutation.child is None:
            self.fail(name, drawn, f"{mutation.status}: {mutation.reason}")
        else:
            fitness = await self.evaluate(program_id, name, drawn, mutation.child, box)
        self.finish(mutation, fitness)

    async def evaluate(
        self, program_id: str, started_from: str, drawn: str, child: Path, box: Path
    ) -> float | None:
        """Validate, fast-score and add the child; gate what is in the top K.

        Returns its fast fitness, or None where it never got that far.
        """
        verdict = await asyncio.to_thread(validate.validate, child)
        if verdict.status != "ok":
            self.fail(started_from, drawn, f"{verdict.status}: {verdict.reason}")
            return None
        stored = self.database.store(child.read_text(encoding="utf-8"), program_id)
        try:
            result = await asyncio.to_thread(
                evaluator.fast,
                stored,
                self.snapshot(),
                random.Random(self.rng.random()),
                self.workers,
            )
        except OpponentCrash:
            raise
        except RuntimeError as error:
            self.fail(started_from, drawn, f"fast: {error}")
            return None
        entry = _program(
            program_id, stored, started_from, drawn, result.fitness, result.rates
        )
        self.database.add(entry)
        LOGGER.info("%s %s fast %.3f", program_id, drawn, result.fitness)
        await asyncio.to_thread(shutil.rmtree, box, ignore_errors=True)
        for program in self.database.top(config.DEEP_TOP_K):
            if program.deep is None and program.id not in self.gating:
                self.gating.add(program.id)
                self.group.create_task(self.gate(program))
        return result.fitness

    async def gate(self, program: archive.Program) -> None:
        """Section 5: the sealed block, the promotion rule, and what follows a yes.

        A promoted champion is re-scored on the pool its own promotion just
        changed -- it is in that pool now, the weights renormalised and
        weakness pressure moved them -- and that is the number the next
        candidate is compared against. Both halves of the champion land
        together, so no session starts from a floor whose bar came from a
        different exam.
        """
        try:
            async with self.deep:
                result = await self.measure(Path(program.source_path), program.id)
        except OpponentCrash:
            raise
        except RuntimeError as error:
            self.fail(program.started_from, program.instruction, f"deep: {error}")
            return
        finally:
            self.gating.discard(program.id)
        self.database.record_deep(program.id, result)
        async with self.promotions:
            verdict, why = gate.promotion(result, self.state.champion)
            LOGGER.info("%s deep %.4f: %s", program.id, result.score, why)
            if verdict:
                # Packaging and the floor's file writes, off the loop thread.
                champion = await asyncio.to_thread(
                    gate.promote, program, result, self.pool
                )
                artifact = wandb.Artifact(
                    champion.name, "champion", metadata=result.model_dump()
                )
                artifact.add_file(champion.tarball)
                self.log.log_artifact(artifact)
                rescored = await self.measure(Path(champion.path), champion.name)
                self.state.champion = rescored
                self.state.champion_path = champion.path
                self.state.sessions_since_promotion = 0
        self.log.log(self.deep_record(result, verdict))

    async def measure(self, agent: Path, program_id: str) -> DeepResult:
        """One deep evaluation on the sealed exam block, in a thread."""
        return await asyncio.to_thread(
            evaluator.deep, agent, program_id, self.snapshot(), self.workers
        )

    def snapshot(self) -> Pool:
        """The pool as it stands, copied on the loop for one evaluation to keep."""
        return self.pool.model_copy(deep=True)

    def start(self, stagnant: bool) -> tuple[Path, str, float, dict[str, float]]:
        """The program this session edits, the name it goes by, and the bar to beat.

        The name is a pool name or a database id, never a path: it is
        interpolated into the session's own feedback. Until the first
        promotion there is no champion, so the seed generation starts from
        the draw below and is told the seed's database id.
        """
        if self.state.champion is not None and not stagnant:
            # `gate.promote` writes CHAMPIONS/<pool name>.py, so the stem is
            # the name the pool knows this program by.
            path = Path(str(self.state.champion_path))
            return path, path.stem, self.state.champion.score, self.state.champion.rates
        program = self.rng.choice(self.database.top(10))
        return Path(program.source_path), program.id, program.fitness, program.rates

    def instruction(self) -> tuple[str, str, Path | None]:
        """Draw an instruction by name and text, and sometimes what to fold in."""
        top = self.database.top(10)
        if self.rng.random() < config.CROSS_PROBABILITY and len(top) > 1:
            return "cross", prompt.CROSS, Path(self.rng.choice(top).source_path)
        name, text = self.rng.choice(prompt.INSTRUCTIONS)
        return name, text, None

    def fail(self, started_from: str, instruction: str, reason: str) -> None:
        """Record an attempt that produced no program, and say why.

        The ledger explains a session that spent tokens and left nothing, and
        the next prompt on that lineage carries the reason back to it.
        """
        LOGGER.warning("%s: %s", started_from, reason)
        self.database.record_failure(
            archive.Failure(
                started_from=started_from,
                instruction=instruction,
                reason=reason,
                created=time.time(),
            )
        )

    def finish(self, mutation: Mutation, fitness: float | None) -> None:
        """Count the session, persist the state, and log section 10's line."""
        self.state.sessions += 1
        self.state.sessions_since_promotion += 1
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(
            self.state.model_dump_json(indent=2), encoding="utf-8"
        )
        record: dict[str, float] = {
            "sessions": self.state.sessions,
            "sessions/ok": int(mutation.child is not None),
            "sessions/timed_out": int(mutation.status == "timeout"),
            "sessions/fallback": int(mutation.fallback),
            "sessions/input_tokens": mutation.input_tokens,
            "sessions/output_tokens": mutation.output_tokens,
            "sessions/seconds": mutation.seconds,
            "sessions/since_promotion": self.state.sessions_since_promotion,
            "database/programs": len(self.database.programs),
            "database/top": self.database.top(1)[0].fitness,
        }
        if fitness is not None:
            record["fast/fitness"] = fitness
        self.log.log(record)

    def deep_record(self, result: DeepResult, promoted: bool) -> dict[str, float]:
        """Section 10: one deep evaluation, and whether it moved the floor."""
        record: dict[str, float] = {
            "sessions": self.state.sessions,
            "deep/score": result.score,
            "deep/low": result.low,
            "deep/high": result.high,
            "deep/field": result.field,
            "deep/promoted": int(promoted),
            **{f"deep/rate/{name}": rate for name, rate in result.rates.items()},
            **{f"deep/held_out/{n}": rate for n, rate in result.held_out.items()},
        }
        if self.state.champion is not None:
            record["deep/champion"] = self.state.champion.score
        both = [(p.fitness, p.deep.score) for p in self.database.programs if p.deep]
        if len(both) > 2:
            record["deep/rho_fast_deep"] = spearman(
                [f for f, _ in both], [d for _, d in both]
            )
        return record


def _first(failures: BaseExceptionGroup) -> BaseException:
    """The first leaf of a task group's exception group, unwrapped."""
    failure: BaseException = failures
    while isinstance(failure, BaseExceptionGroup):
        failure = failure.exceptions[0]
    return failure


if __name__ == "__main__":
    main()
