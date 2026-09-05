"""Algorithm 1: mutate, validate, fast-evaluate, insert; migrate, deep-evaluate, reset.

One iteration is one mutation per island, run in parallel; migration, the
deep-evaluation epoch and the island reset happen on their own intervals.
Everything a run remembers is on disk -- the archive log, the pool, the
epoch lines, ``state.json`` beside the archive -- so a killed run resumes
where it stopped.

Every evaluation forks a process pool (spawn, because the harness caps
tasks per child), so every caller of ``run`` must sit behind an
``if __name__ == "__main__":`` guard: ``python -m kaggriculture.campaign.loop``,
the ``campaign loop`` entry point and pytest all do.
"""

import argparse
import datetime
import logging
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from kaggriculture.campaign import archive as archive_module
from kaggriculture.campaign import config, evaluator, gate, mutate, prompt, validate
from kaggriculture.campaign.evaluator import DeepResult
from kaggriculture.campaign.mutate import Mutator
from kaggriculture.campaign.pool import Pool

LOGGER = logging.getLogger(__name__)

# Seconds to idle for once the day's codex quota is spent. Without it the
# loop would spin through iterations at no cost but full CPU until midnight.
QUOTA_SLEEP_SECONDS = 60


class State(BaseModel):
    """What survives a restart.

    Attributes:
        iteration: Iterations completed so far.
        champion: The deep result of the program currently on the floor.
        calls_today: Mutation calls spent on ``day``.
        day: The calendar day ``calls_today`` counts, ISO format.
    """

    iteration: int = 0
    champion: DeepResult | None = None
    calls_today: int = 0
    day: str = ""


def main(argv: list[str] | None = None) -> None:
    """``campaign loop`` / ``campaign dry-run``: run the evolution loop.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` when None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10**9)
    # Every mutation in flight runs an evaluation of its own, so the default
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
    run(
        args.iterations,
        mutator,
        args.workers,
        args.concurrency,
        args.seed_agent,
        random.Random(),
    )


def run(
    iterations: int,
    mutator: Mutator,
    workers: int,
    concurrency: int,
    seed_agent: Path,
    rng: random.Random,
) -> State:
    """Seed the archive if it is empty, then run ``iterations`` iterations.

    Args:
        iterations: How many iterations to run.
        mutator: What turns a sandbox into a child program.
        workers: Processes each evaluation fans its games over.
        concurrency: Mutations in flight at once.
        seed_agent: The program every island starts from, on a cold start.
        rng: The loop's generator; every thread gets a child of it.

    Returns:
        The state as of the last completed iteration.

    Raises:
        ValueError: ``workers * concurrency`` exceeds ``config.CORE_BUDGET``:
            each of the ``concurrency`` mutations in flight runs an
            evaluation that forks ``workers`` processes, so the peak is
            their product and it must fit the budget.
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
    for _ in range(iterations):
        state = iterate(state, archive, pool, mutator, workers, concurrency, rng)
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    return state


def iterate(
    state: State,
    archive: archive_module.Archive,
    pool: Pool,
    mutator: Mutator,
    workers: int,
    concurrency: int,
    rng: random.Random,
) -> State:
    """One iteration: a mutation per island, then whatever the schedule owes.

    Args:
        state: The state to advance.
        archive: The population.
        pool: The opponents candidates are measured against.
        mutator: What turns a sandbox into a child program.
        workers: Processes each evaluation fans its games over.
        concurrency: Mutations in flight at once.
        rng: The loop's generator; every thread gets a child of it.

    Returns:
        The advanced state.
    """
    today = datetime.date.today().isoformat()
    if state.day != today:
        state = state.model_copy(update={"day": today, "calls_today": 0})
    if state.calls_today >= config.DAILY_CALL_BUDGET:
        LOGGER.warning("daily call budget spent; evaluating only")
        time.sleep(QUOTA_SLEEP_SECONDS)
    else:
        # `random.Random` is not thread-safe, so each mutation carries its
        # own generator, drawn from the loop's before any thread starts.
        jobs = [
            (island, _plan(archive, island, rng), random.Random(rng.random()))
            for island in range(config.ISLANDS)
        ]
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            spent = list(
                executor.map(
                    lambda job: _mutate(job, archive, pool, mutator, workers), jobs
                )
            )
        state = state.model_copy(update={"calls_today": state.calls_today + len(spent)})
    iteration = state.iteration + 1
    state = state.model_copy(update={"iteration": iteration})
    if iteration % config.MIGRATION_INTERVAL == 0:
        archive.migrate()
    if iteration % config.EPOCH_INTERVAL == 0:
        state = epoch(state, archive, pool, workers)
    if iteration % config.RESET_INTERVAL == 0 and archive.top(1):
        archive.reset_worst_island(archive.top(1)[0])
    return state


def _plan(
    archive: archive_module.Archive, island: int, rng: random.Random
) -> tuple[
    Literal["full", "cross"], archive_module.Program, archive_module.Program | None
]:
    """Pick this island's parent, and a crossover partner some of the time."""
    parent = archive.ucb_parent(island, rng)
    if rng.random() < config.CROSS_PROBABILITY and len(archive.island(island)) > 1:
        inspiration = archive.ucb_parent(island, rng)
        if inspiration.id != parent.id:
            return "cross", parent, inspiration
    return "full", parent, None


def _mutate(
    job: tuple[
        int,
        tuple[
            Literal["full", "cross"],
            archive_module.Program,
            archive_module.Program | None,
        ],
        random.Random,
    ],
    archive: archive_module.Archive,
    pool: Pool,
    mutator: Mutator,
    workers: int,
) -> None:
    """Mutate, validate, fast-evaluate and insert one child. Runs in a thread.

    Every way this can end short of an insert is recorded on the archive as
    a failure -- a mutation that wrote nothing, a candidate that failed
    validation, a game that raised -- so the ledger explains a call that
    spent tokens and left no program, and the next prompt on that lineage
    carries the reason. The sandbox is removed only on a successful insert:
    on failure its codex log is the evidence.

    Args:
        job: The island, its plan from ``_plan``, and this thread's generator.
        archive: The population.
        pool: The opponents the child is measured against.
        mutator: What turns the sandbox into a child program.
        workers: Processes the fast evaluation fans its games over.
    """
    island, (kind, parent, inspiration), rng = job
    program_id = f"i{island}-{int(rng.random() * 1e9):09d}"
    failures = [f.reason for f in archive.failures() if parent.id in f.parents][-3:]
    box = prompt.build_sandbox(
        program_id,
        kind,
        Path(parent.source_path),
        Path(inspiration.source_path) if inspiration else None,
        feedback=parent.rates,
        weakest=pool.weakest(parent.rates) if parent.rates else "",
        weights=pool.weights,
        failures=failures,
    )
    mutation = mutator(box, program_id)
    mutate.record(mutation)
    parents = [parent.id] + ([inspiration.id] if inspiration else [])
    if mutation.status != "ok" or mutation.child is None:
        archive.record_failure(
            island, parents, kind, f"{mutation.status}: {mutation.reason}"
        )
        return
    verdict = validate.validate(mutation.child)
    if verdict.status != "ok":
        archive.record_failure(
            island, parents, kind, f"{verdict.status}: {verdict.reason}"
        )
        return
    stored = archive.store(mutation.child.read_text(encoding="utf-8"), program_id)
    try:
        result = evaluator.fast(stored, pool, rng, workers)
    except RuntimeError as error:
        archive.record_failure(island, parents, kind, f"fast: {error}")
        return
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
    replaced = archive.insert(program)
    LOGGER.info(
        "%s %s fast %.3f (%s)",
        program_id,
        kind,
        result.fitness,
        "dropped" if replaced is program else "inserted",
    )
    shutil.rmtree(box, ignore_errors=True)


def epoch(
    state: State, archive: archive_module.Archive, pool: Pool, workers: int
) -> State:
    """Deep-evaluate the top K on the exam block; promote if the rule says so.

    Args:
        state: The state this epoch closes.
        archive: The population the candidates come from.
        pool: The opponents that count towards the score; a promotion
            changes it and saves it.
        workers: Processes each deep evaluation fans its games over.

    Returns:
        The state, carrying the new champion if one was promoted.
    """
    measured: list[archive_module.Program] = []
    results: list[DeepResult] = []
    for program in archive.top(config.DEEP_TOP_K):
        try:
            results.append(
                evaluator.deep(Path(program.source_path), program.id, pool, workers)
            )
        except RuntimeError as error:
            archive.record_failure(
                program.island, [program.id], program.kind, f"deep: {error}"
            )
            continue
        measured.append(program)
    rho = (
        _spearman([p.mean for p in measured], [r.score for r in results])
        if len(results) > 2
        else None
    )
    best = max(results, key=lambda r: r.score, default=None)
    promoted = None
    if best is not None:
        ok, why = gate.promotion(best, state.champion)
        LOGGER.info(
            "epoch %d: best deep %.4f [%.4f, %.4f] -- %s",
            state.iteration,
            best.score,
            best.low,
            best.high,
            why,
        )
        if ok:
            program = next(p for p in measured if p.id == best.program_id)
            promoted = gate.promote(program, best, pool)
            state = state.model_copy(update={"champion": best})
    gate.epoch_line(state.iteration, results, promoted, rho)
    return state


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
