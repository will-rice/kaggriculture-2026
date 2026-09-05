"""Algorithm 1, fully asynchronous: nothing waits on anything it does not need.

One event loop on one thread owns the archive, the pool, the state and the
wandb run, so none of them needs a lock. A dispatcher keeps
``CODEX_CONCURRENCY`` codex sessions in flight; each finished session becomes
a job that validates, fast-evaluates and inserts its child, and the schedule
-- migration, the deep-evaluation epoch, the island reset -- runs off the
count of completed calls rather than off a barrier. An epoch is a task like
any other and competes for cores instead of stopping dispatch.

Everything blocking runs somewhere else and is awaited: codex is a child
process in its own process group, and validation, the evaluators and the
promotion commit go to threads. Everything a run remembers is on disk -- the
archive log, the pool, ``champion.json`` and ``state.json`` beside the
archive -- so a killed run resumes where it stopped. Metrics go to wandb, one
run per campaign, keyed by ``calls``.

Every evaluation forks a process pool (spawn, because the harness caps
tasks per child), so every caller of ``run`` must sit behind an
``if __name__ == "__main__":`` guard: ``python -m kaggriculture.campaign.loop``,
the ``campaign loop`` entry point and pytest all do.
"""

import argparse
import asyncio
import contextlib
import datetime
import logging
import random
import shutil
import signal
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Literal

import wandb
from git import Repo
from pydantic import BaseModel

from kaggriculture.campaign import archive as archive_module
from kaggriculture.campaign import (
    config,
    evaluator,
    gate,
    harness,
    mutate,
    prompt,
    validate,
)
from kaggriculture.campaign.evaluator import DeepResult
from kaggriculture.campaign.mutate import Mutation, Mutator
from kaggriculture.campaign.pool import Pool

LOGGER = logging.getLogger(__name__)

# Seconds to idle for once the day's codex quota is spent. Without it the
# dispatcher would spin at no cost but full CPU until midnight.
QUOTA_SLEEP_SECONDS = 60

# What the wandb run records as its configuration.
HYPERPARAMETERS = (
    "ISLANDS",
    "ISLAND_SIZE",
    "MIGRATION_INTERVAL",
    "MIGRANTS",
    "RESET_INTERVAL",
    "UCB_C",
    "CROSS_PROBABILITY",
    "FAST_SEEDS",
    "EPOCH_INTERVAL",
    "DEEP_TOP_K",
    "CHAMPION_WEIGHT",
    "POOL_CAP",
    "RETIRE_THRESHOLD",
    "WEAKNESS_CAP",
    "CODEX_CONCURRENCY",
    "CODEX_MODEL",
    "CODEX_FALLBACK_MODEL",
    "DAILY_CALL_BUDGET",
)

# One island's plan: what kind of mutation, from which parent, and with which
# second program as inspiration when the kind is a crossover.
Plan = tuple[
    Literal["full", "cross"], archive_module.Program, archive_module.Program | None
]


class State(BaseModel):
    """What survives a restart.

    Attributes:
        calls: Mutation calls completed so far, ever. The schedule is keyed
            on this, so a ``state.json`` written by the old barrier loop --
            which had ``iteration`` and no ``calls`` -- resumes at zero and
            only shifts the cadence.
        champion: The deep result of the program currently on the floor,
            re-measured against the pool as it stands at the start of every
            epoch rather than kept as it was at promotion.
        champion_path: The champion's own immutable copy under
            ``config.CHAMPIONS``, so an epoch can re-measure it.
        calls_today: Mutation calls spent on ``day``.
        day: The calendar day ``calls_today`` counts, ISO format.
    """

    calls: int = 0
    champion: DeepResult | None = None
    champion_path: str | None = None
    calls_today: int = 0
    day: str = ""


class Cores:
    """The cores the loop's own evaluations may use between them.

    An ``asyncio.Semaphore`` cannot express "take seven at once", so this is
    a counter behind a condition: a taker waits until its whole share is
    free, and every waiter is woken on a release because the one that fits
    is not necessarily the one that has waited longest.
    """

    def __init__(self, budget: int) -> None:
        """Initializes the counter.

        Args:
            budget: Permits available in total.
        """
        self.budget = budget
        self.free = budget
        self.condition = asyncio.Condition()

    @contextlib.asynccontextmanager
    async def take(self, n: int) -> AsyncIterator[None]:
        """Hold ``n`` permits for the body, waiting until that many are free.

        Args:
            n: Permits to hold.

        Yields:
            None, once the permits are held.

        Raises:
            ValueError: More permits than the whole budget, which no release
                could ever satisfy.
        """
        if n > self.budget:
            raise ValueError(f"{n} cores requested; the budget is {self.budget}")
        async with self.condition:
            await self.condition.wait_for(lambda: self.free >= n)
            self.free -= n
        try:
            yield
        finally:
            async with self.condition:
                self.free += n
                self.condition.notify_all()


def today() -> str:
    """The calendar day the daily call budget counts against, ISO format."""
    return datetime.date.today().isoformat()


class Campaign:
    """One run: the shared state, and the coroutines that advance it.

    Every attribute here is read and written only from coroutines on the one
    event-loop thread, which is what makes the archive, the pool, the state
    and the wandb run safe without a lock. The blocking work they describe
    happens in child processes and threads.
    """

    def __init__(
        self,
        state: State,
        archive: archive_module.Archive,
        pool: Pool,
        mutator: Mutator,
        workers: int,
        concurrency: int,
        rng: random.Random,
        log: wandb.Run,
        commit: bool,
    ) -> None:
        """Initializes the campaign.

        Args:
            state: The state to advance, mutated in place.
            archive: The population.
            pool: The opponents candidates are measured against.
            mutator: What turns a sandbox into a child program.
            workers: Processes each evaluation fans its games over.
            concurrency: Codex sessions in flight at once.
            rng: The loop's generator; every call gets a child of it.
            log: The wandb run metrics go to.
            commit: Whether a promotion records itself in the repository.
        """
        self.state = state
        self.archive = archive
        self.pool = pool
        self.mutator = mutator
        self.workers = workers
        self.rng = rng
        self.log = log
        self.commit = commit
        self.sessions = asyncio.Semaphore(concurrency)
        self.cores = Cores(config.CORE_BUDGET)
        # Every job and every epoch is a child of this group, so the first
        # OpponentCrash out of any of them cancels the rest -- which is what
        # kills the codex process groups still running -- and comes back out
        # of `work`.
        self.group = asyncio.TaskGroup()
        # A job counts against its island from the moment it is planned until
        # its completion step has run, so the dispatcher spreads calls over
        # islands rather than piling them onto whichever finished last.
        self.in_flight = [0] * config.ISLANDS
        self.epoch_task: asyncio.Task[None] | None = None
        self.state_file = config.ARCHIVE.with_name("state.json")

    async def drive(self, calls: int) -> None:
        """Run the campaign, stopping early and cleanly on a signal.

        SIGTERM and SIGINT cancel the work, which cancels every task in it:
        a cancelled codex call kills its own process group, so a restart
        never finds sessions from the run before it still alive. The state
        file is written after every completed call, so stopping here loses
        only what was in flight.

        Args:
            calls: How many mutation calls this run dispatches.
        """
        work = asyncio.ensure_future(self.work(calls))
        running = asyncio.get_running_loop()
        for number in (signal.SIGINT, signal.SIGTERM):
            running.add_signal_handler(number, work.cancel)
        try:
            await work
        except asyncio.CancelledError:
            LOGGER.warning("stopped on a signal at %d calls", self.state.calls)
        finally:
            for number in (signal.SIGINT, signal.SIGTERM):
                running.remove_signal_handler(number)

    async def work(self, calls: int) -> None:
        """Dispatch ``calls`` codex sessions, then drain what is in flight.

        Args:
            calls: How many mutation calls this run dispatches.

        Raises:
            OpponentCrash: A pool opponent failed in its own seat, in a job
                or an epoch. The task group has already cancelled everything
                else by the time it arrives here.
        """
        try:
            async with self.group as group:
                group.create_task(self.dispatch(calls), name="dispatch")
        except BaseExceptionGroup as failures:
            raise _first(failures) from failures

    async def dispatch(self, calls: int) -> None:
        """Keep sessions in flight until ``calls`` of them have been planned.

        The permit is acquired here and released by the job that spends it,
        so a session's slot is refilled the moment codex exits rather than
        when its child finishes being scored.

        Args:
            calls: How many mutation calls to plan.
        """
        for _ in range(calls):
            await self.sessions.acquire()
            while self.spent():
                self.sessions.release()
                LOGGER.warning("daily call budget spent; idling until it resets")
                await asyncio.sleep(QUOTA_SLEEP_SECONDS)
                await self.sessions.acquire()
            island = min(range(config.ISLANDS), key=lambda i: self.in_flight[i])
            self.in_flight[island] += 1
            self.state.calls_today += 1
            self.group.create_task(
                self.job(
                    island,
                    _plan(self.archive, island, self.rng),
                    random.Random(self.rng.random()),
                )
            )

    def spent(self) -> bool:
        """Whether the day's call budget is gone, rolling the day over first."""
        now = today()
        if self.state.day != now:
            self.state.day = now
            self.state.calls_today = 0
        return self.state.calls_today >= config.DAILY_CALL_BUDGET

    async def job(self, island: int, plan: Plan, rng: random.Random) -> None:
        """One call: a codex session, then whatever its child earns.

        Every way this can end short of an insert is recorded on the archive
        as a failure -- a mutation that wrote nothing, a candidate that failed
        validation, a game that raised -- so the ledger explains a call that
        spent tokens and left no program, and the next prompt on that lineage
        carries the reason. The sandbox is removed only on a successful
        insert: on failure its codex log is the evidence.

        The two counters the dispatcher reads are given up in ``finally``
        blocks that cover everything after it handed them over: the session
        permit as soon as codex is gone, and the island's in-flight count
        when the job is. Anything this raises -- an ``OpponentCrash`` on its
        way to stopping the run, a sandbox that could not be built -- would
        otherwise leak a permit the dispatcher never gets back or an island
        that looks busy for the rest of the run.

        Args:
            island: The island this call belongs to.
            plan: What to mutate, from ``_plan``.
            rng: This call's generator.
        """
        kind, parent, inspiration = plan
        program_id = f"i{island}-{int(rng.random() * 1e9):09d}"
        try:
            try:
                failures = [
                    f.reason for f in self.archive.failures() if parent.id in f.parents
                ]
                box = prompt.build_sandbox(
                    program_id,
                    kind,
                    Path(parent.source_path),
                    Path(inspiration.source_path) if inspiration else None,
                    feedback=parent.rates,
                    weakest=self.pool.weakest(parent.rates) if parent.rates else "",
                    weights=self.pool.weights,
                    failures=failures[-3:],
                )
                mutation = await self.mutator(box, program_id)
            finally:
                self.sessions.release()
            parents = [parent.id] + ([inspiration.id] if inspiration else [])
            if mutation.status == "exec_error":
                # The provider refused or codex died, on both models: nothing
                # about the lineage caused it, so it gets no failure line and
                # the island is simply dispatched again.
                LOGGER.warning(
                    "%s: no session ran to a verdict (%s)",
                    program_id,
                    mutation.reason[:200],
                )
                fitness = None
            elif mutation.child is None:
                self.archive.record_failure(
                    island, parents, kind, f"{mutation.status}: {mutation.reason}"
                )
                fitness = None
            else:
                fitness = await self.score(
                    island, parents, kind, program_id, mutation.child, box, rng
                )
            self.finish(mutation, fitness)
        finally:
            self.in_flight[island] -= 1

    async def score(
        self,
        island: int,
        parents: list[str],
        kind: Literal["full", "cross"],
        program_id: str,
        child: Path,
        box: Path,
        rng: random.Random,
    ) -> float | None:
        """Validate, fast-evaluate and insert one child.

        Args:
            island: The island the child belongs to.
            parents: The archive ids it came from.
            kind: "full" or "cross".
            program_id: The child's archive id.
            child: The file the session wrote.
            box: The sandbox, removed once the child is in the archive.
            rng: This call's generator, which draws the fast seeds.

        Returns:
            The child's fast fitness, or None where it never got that far.

        Raises:
            OpponentCrash: A pool opponent failed in its own seat. Not this
                lineage's failure and never its feedback: the pool is broken,
                so it comes out of here and stops the run.
        """
        verdict = await asyncio.to_thread(validate.validate, child)
        if verdict.status != "ok":
            self.archive.record_failure(
                island, parents, kind, f"{verdict.status}: {verdict.reason}"
            )
            return None
        stored = self.archive.store(child.read_text(encoding="utf-8"), program_id)
        try:
            async with self.cores.take(self.workers):
                result = await asyncio.to_thread(
                    evaluator.fast, stored, self.snapshot(), rng, self.workers
                )
        except harness.OpponentCrash:
            raise
        except RuntimeError as error:
            self.archive.record_failure(island, parents, kind, f"fast: {error}")
            return None
        program = archive_module.Program(
            id=program_id,
            island=island,
            source_path=str(stored),
            parents=parents,
            kind=kind,
            fitness_sum=result.fitness,
            n_evals=1,
            status="ok",
            reason="",
            created=time.time(),
            rates=result.rates,
        )
        replaced = self.archive.insert(program)
        LOGGER.info(
            "%s %s fast %.3f (%s)",
            program_id,
            kind,
            result.fitness,
            "dropped" if replaced is program else "inserted",
        )
        await asyncio.to_thread(shutil.rmtree, box, ignore_errors=True)
        return result.fitness

    def finish(self, mutation: Mutation, fitness: float | None) -> None:
        """Count the call, persist the state, log it, and run the schedule.

        Args:
            mutation: What the session cost and whether it wrote a child.
            fitness: The child's fast fitness, or None where it never got
                that far.
        """
        self.state.calls += 1
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(
            self.state.model_dump_json(indent=2), encoding="utf-8"
        )
        self.log.log(self.record(mutation, fitness))
        calls = self.state.calls
        if calls % config.MIGRATION_INTERVAL == 0:
            self.archive.migrate()
        if calls % config.EPOCH_INTERVAL == 0:
            if self.epoch_task is not None and not self.epoch_task.done():
                LOGGER.info(
                    "epoch at %d calls skipped: the last one is still running", calls
                )
            else:
                self.epoch_task = self.group.create_task(self.epoch())
        if calls % config.RESET_INTERVAL == 0 and self.archive.top(1):
            self.archive.reset_worst_island(self.archive.top(1)[0])

    def record(
        self, mutation: Mutation, fitness: float | None
    ) -> dict[str, float | int]:
        """What one call cost and what it put in the archive.

        Args:
            mutation: The session's outcome. A call is ``ok`` when it yielded
                a child, whether or not the session finished.
            fitness: The child's fast fitness, or None where none was
                measured.

        Returns:
            The metrics for ``wandb.Run.log``, keyed by ``calls``.
        """
        record: dict[str, float | int] = {
            "calls": self.state.calls,
            "calls/ok": int(mutation.child is not None),
            "calls/timed_out": int(mutation.status == "timeout"),
            "calls/fallback": int(mutation.fallback),
            "calls/input_tokens": mutation.input_tokens,
            "calls/output_tokens": mutation.output_tokens,
            "calls/seconds": mutation.seconds,
            "calls/today": self.state.calls_today,
            # This call is still counted against its island until its job
            # returns, so it is the one subtracted here.
            "calls/in_flight": sum(self.in_flight) - 1,
            "archive/programs": sum(
                len(self.archive.island(i)) for i in range(config.ISLANDS)
            ),
        }
        if fitness is not None:
            record["fast/fitness"] = fitness
        top = self.archive.top(1)
        if top:
            record["archive/top"] = top[0].mean
        return record

    async def epoch(self) -> None:
        """Deep-evaluate the top K on the exam block; promote if the rule says so.

        The champion is re-measured first, on the pool as it stands now. Its
        stored result was computed against the pool as it was at its own
        promotion -- before it joined that pool, before the weights
        renormalised and before weakness pressure moved them -- so comparing a
        candidate's fresh score against it would be two numbers from two
        different exams. One extra deep evaluation per epoch buys a comparison
        that means something.

        Every measurement is its own coroutine taking its own share of the
        cores, so they overlap with each other and with the jobs the
        dispatcher keeps starting. A promotion changes the pool while fast
        evaluations against the previous pool may be in flight; their fitness
        is inserted as measured, which is the inconsistency the archive's mean
        already spans across epochs, at a finer grain.

        Raises:
            OpponentCrash: A pool opponent failed in its own seat, so the gate
                halts rather than record a verdict it did not compute.
        """
        calls = self.state.calls
        candidates = self.archive.top(config.DEEP_TOP_K)
        agents = [(Path(p.source_path), p.id) for p in candidates]
        if self.state.champion_path is not None:
            agents.insert(0, (Path(self.state.champion_path), "champion"))
        outcomes = await asyncio.gather(
            *(self.measure(agent, pid) for agent, pid in agents),
            return_exceptions=True,
        )
        if self.state.champion_path is not None:
            baseline = outcomes.pop(0)
            if isinstance(baseline, BaseException):
                raise baseline
            self.state.champion = baseline
        measured, results = self.sift(candidates, outcomes)
        rho = (
            _spearman([p.mean for p in measured], [r.score for r in results])
            if len(results) > 2
            else None
        )
        best = max(results, key=lambda r: r.score, default=None)
        record: dict[str, float | int] = {"calls": calls}
        if self.state.champion is not None:
            record["deep/champion"] = self.state.champion.score
        if rho is not None:
            record["deep/rho_fast_deep"] = rho
        promoted = None
        if best is not None:
            record.update(
                {
                    "deep/best": best.score,
                    "deep/best_low": best.low,
                    "deep/best_high": best.high,
                    "deep/best_field": best.field,
                    **{f"deep/rate/{name}": rate for name, rate in best.rates.items()},
                    **{f"deep/held_out/{n}": rate for n, rate in best.held_out.items()},
                }
            )
            ok, why = gate.promotion(best, self.state.champion)
            LOGGER.info(
                "epoch at %d calls: best deep %.4f [%.4f, %.4f] -- %s",
                calls,
                best.score,
                best.low,
                best.high,
                why,
            )
            if ok:
                promoted = await self.promote(
                    next(p for p in measured if p.id == best.program_id), best
                )
        record["deep/promoted"] = int(promoted is not None)
        self.log.log(record)

    def sift(
        self,
        candidates: list[archive_module.Program],
        outcomes: list[DeepResult | BaseException],
    ) -> tuple[list[archive_module.Program], list[DeepResult]]:
        """Split an epoch's measurements into scores and recorded failures.

        A candidate that raised a plain ``RuntimeError`` crashed in its own
        seat, which is its lineage's failure and goes in the ledger. Anything
        else -- a broken opponent, a pool with no field to average over --
        is not a verdict on the candidate, so it comes back out.

        Args:
            candidates: The programs measured, in the order they were sent.
            outcomes: What each measurement returned or raised.

        Returns:
            The programs that scored and their results, in the same order.

        Raises:
            OpponentCrash: A pool opponent failed in its own seat.
        """
        measured: list[archive_module.Program] = []
        results: list[DeepResult] = []
        for program, outcome in zip(candidates, outcomes, strict=True):
            if isinstance(outcome, harness.OpponentCrash):
                raise outcome
            if isinstance(outcome, RuntimeError):
                self.archive.record_failure(
                    program.island, [program.id], program.kind, f"deep: {outcome}"
                )
                continue
            if isinstance(outcome, BaseException):
                raise outcome
            measured.append(program)
            results.append(outcome)
        return measured, results

    async def measure(self, agent: Path, program_id: str) -> DeepResult:
        """One deep evaluation, holding the cores it fans its games over.

        Args:
            agent: The file to measure.
            program_id: The program the measurement belongs to.

        Returns:
            Its deep result on the exam block.
        """
        async with self.cores.take(self.workers):
            return await asyncio.to_thread(
                evaluator.deep, agent, program_id, self.snapshot(), self.workers
            )

    def snapshot(self) -> Pool:
        """The pool as it stands, copied for one evaluation to keep.

        A promotion adds a champion and renormalises the weights, and it
        happens on the loop thread while evaluations are running in others.
        An evaluation reads the pool more than once -- who to play, then how
        to weight what it measured -- so sharing the live object would let a
        promotion land between those reads and leave it weighting a game it
        never played. The copy is taken here, on the loop, so it cannot
        straddle a promotion: the measurement is against the pool it started
        with, and its fitness is inserted as measured.

        Returns:
            A pool of its own for one evaluation.
        """
        return self.pool.model_copy(deep=True)

    async def promote(self, program: archive_module.Program, best: DeepResult) -> str:
        """Put ``program`` on the floor and record it everywhere it belongs.

        The file writes, the pool save and the wandb artifact are
        milliseconds on the loop. The version-control record is two
        subprocesses, so it goes to a thread.

        Args:
            program: The archive entry being promoted.
            best: Its deep evaluation.

        Returns:
            The champion's pool name.
        """
        champion = gate.promote(program, best, self.pool)
        self.state.champion = champion.result
        self.state.champion_path = champion.path
        artifact = wandb.Artifact(
            champion.name, type="champion", metadata=best.model_dump()
        )
        artifact.add_file(champion.path)
        self.log.log_artifact(artifact)
        if self.commit:
            await asyncio.to_thread(gate.commit_floor, champion, program.id)
        return champion.name


def main(argv: list[str] | None = None) -> None:
    """``campaign loop`` / ``campaign dry-run``: run the evolution loop.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` when None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=10**9)
    # Every session in flight runs an evaluation of its own, so the default
    # share is the budget split between them, never the whole of it.
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, config.CORE_BUDGET // config.CODEX_CONCURRENCY),
    )
    parser.add_argument("--concurrency", type=int, default=config.CODEX_CONCURRENCY)
    parser.add_argument("--seed-agent", type=Path, default=gate.SERVED)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="fake mutator: copies the parent with one visible edit",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    mutator: Mutator = (
        mutate.FakeMutator(edit=lambda source: source + "\n# dry-run mutation\n")
        if args.dry_run
        else mutate.CodexMutator()
    )
    # A dry run proves the pipeline; its numbers would only pollute the
    # campaign's run.
    revision = Repo(config.ROOT).head.commit.hexsha[:7]
    log = wandb.init(
        entity=config.WANDB_ENTITY,
        project=config.WANDB_PROJECT,
        id=config.WANDB_RUN_ID,
        name=f"{config.CODEX_MODEL}-{revision}",
        resume="allow",
        mode="disabled" if args.dry_run else "online",
        config={name: getattr(config, name) for name in HYPERPARAMETERS},
    )
    log.define_metric("calls")
    log.define_metric("*", step_metric="calls")
    try:
        run(
            args.calls,
            mutator,
            args.workers,
            args.concurrency,
            args.seed_agent,
            random.Random(),
            log,
        )
    finally:
        log.finish()


def run(
    calls: int,
    mutator: Mutator,
    workers: int,
    concurrency: int,
    seed_agent: Path,
    rng: random.Random,
    log: wandb.Run,
    commit: bool = True,
) -> State:
    """Seed the archive if it is empty, then run ``calls`` mutation calls.

    Args:
        calls: How many mutation calls to dispatch. Whatever is in flight
            when the last one is planned is drained, so it completes and
            counts.
        mutator: What turns a sandbox into a child program.
        workers: Processes each evaluation fans its games over.
        concurrency: Codex sessions in flight at once.
        seed_agent: The program every island starts from, on a cold start.
        rng: The loop's generator; every call gets a child of it.
        log: The wandb run metrics go to; ``wandb.init(mode="disabled")``
            where nothing should be recorded.
        commit: Whether a promotion records itself in the repository. Tests
            pass False so a promotion never runs a version-control command.

    Returns:
        The state as of the last completed call.

    Raises:
        ValueError: ``workers * concurrency`` exceeds ``config.CORE_BUDGET``:
            the mutation share has to fit even in the moment every session in
            flight is evaluating at once.
    """
    if workers * concurrency > config.CORE_BUDGET:
        raise ValueError(
            f"workers * concurrency ({workers} * {concurrency}) exceeds "
            f"CORE_BUDGET ({config.CORE_BUDGET})"
        )
    state_file = config.ARCHIVE.with_name("state.json")
    state = (
        State.model_validate_json(state_file.read_text(encoding="utf-8"))
        if state_file.exists()
        else State()
    )
    # `state.json` is written after every completed call; `champion.json` is
    # written before `gate.promote` returns. A kill in between leaves the
    # first stale and the second current, so the second wins -- otherwise the
    # gate would restart with no baseline and promote a second time.
    champion = gate.load_champion()
    if champion is not None:
        state.champion = champion.result
        state.champion_path = champion.path
        LOGGER.info("resuming on champion %s from champion.json", champion.name)
    pool = Pool.load(config.POOL) if config.POOL.exists() else Pool.initial()
    # A champion's name resolves through the pool file, so the pool on disk
    # must be current before anything plays a game (`gate.promote` keeps it
    # that way from here on).
    pool.save(config.POOL)
    archive = archive_module.Archive(config.ARCHIVE, config.PROGRAMS)
    if not any(archive.island(i) for i in range(config.ISLANDS)):
        fitness = evaluator.fast(seed_agent, pool, rng, workers).fitness
        archive.seed(seed_agent, fitness)
        LOGGER.info(
            "seeded every island from %s at fast fitness %.3f", seed_agent, fitness
        )
    campaign = Campaign(
        state, archive, pool, mutator, workers, concurrency, rng, log, commit
    )
    asyncio.run(campaign.drive(calls))
    return campaign.state


def _first(failures: BaseExceptionGroup) -> BaseException:
    """The first leaf of a task group's exception group.

    A task group reports what its children raised as a tree. The loop's own
    failure mode is a single ``OpponentCrash`` out of one job, and callers --
    the tests and the operator reading a traceback -- want that exception,
    not a wrapper around it. The caller raises it *from* the group, so the
    siblings that failed with it stay reachable as the cause rather than
    being thrown away.

    Args:
        failures: What the task group raised.

    Returns:
        The first exception in it that is not itself a group.
    """
    failure: BaseException = failures
    while isinstance(failure, BaseExceptionGroup):
        failure = failure.exceptions[0]
    return failure


def _plan(archive: archive_module.Archive, island: int, rng: random.Random) -> Plan:
    """Pick this island's parent, and a crossover partner some of the time."""
    parent = archive.ucb_parent(island, rng)
    if rng.random() < config.CROSS_PROBABILITY and len(archive.island(island)) > 1:
        inspiration = archive.ucb_parent(island, rng)
        if inspiration.id != parent.id:
            return "cross", parent, inspiration
    return "full", parent, None


def _spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation between two equal-length sequences, ties unbroken."""

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        for rank, index in enumerate(order):
            out[index] = float(rank)
        return out

    ra, rb = ranks(a), ranks(b)
    n = len(a)
    return 1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb, strict=True)) / (
        n * (n * n - 1)
    )


if __name__ == "__main__":
    main()
