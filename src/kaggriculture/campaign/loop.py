"""Algorithm 1: mutate, validate, fast-evaluate, insert; migrate, deep-evaluate, reset.

One iteration is one mutation per island, run in parallel; migration, the
deep-evaluation epoch and the island reset happen on their own intervals.
Everything a run remembers is on disk -- the archive log, the pool,
``champion.json`` and ``state.json`` beside the archive -- so a killed run
resumes where it stopped. Metrics go to wandb, one run per campaign, keyed
by ``iteration`` so a resumed loop continues the same curves.

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

import wandb
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
# loop would spin through iterations at no cost but full CPU until midnight.
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
    "DAILY_CALL_BUDGET",
)


class State(BaseModel):
    """What survives a restart.

    Attributes:
        iteration: Iterations completed so far.
        champion: The deep result of the program currently on the floor,
            re-measured against the pool as it stands at the start of every
            epoch rather than kept as it was at promotion.
        champion_path: The champion's own immutable copy under
            ``config.CHAMPIONS``, so an epoch can re-measure it.
        calls_today: Mutation calls spent on ``day``.
        day: The calendar day ``calls_today`` counts, ISO format.
    """

    iteration: int = 0
    champion: DeepResult | None = None
    champion_path: str | None = None
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
    # A dry run proves the pipeline; its numbers would only pollute the
    # campaign's run.
    log = wandb.init(
        entity=config.WANDB_ENTITY,
        project=config.WANDB_PROJECT,
        id=config.WANDB_RUN_ID,
        resume="allow",
        mode="disabled" if args.dry_run else "online",
        config={name: getattr(config, name) for name in HYPERPARAMETERS},
    )
    log.define_metric("iteration")
    log.define_metric("*", step_metric="iteration")
    try:
        run(
            args.iterations,
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
    iterations: int,
    mutator: Mutator,
    workers: int,
    concurrency: int,
    seed_agent: Path,
    rng: random.Random,
    log: wandb.Run,
    commit: bool = True,
) -> State:
    """Seed the archive if it is empty, then run ``iterations`` iterations.

    Args:
        iterations: How many iterations to run.
        mutator: What turns a sandbox into a child program.
        workers: Processes each evaluation fans its games over.
        concurrency: Mutations in flight at once.
        seed_agent: The program every island starts from, on a cold start.
        rng: The loop's generator; every thread gets a child of it.
        log: The wandb run metrics go to; ``wandb.init(mode="disabled")``
            where nothing should be recorded.
        commit: Whether a promotion records itself in the repository. Tests
            pass False so a promotion never runs a version-control command.

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
    # `state.json` is written only once an iteration finishes; `champion.json`
    # is written before `gate.promote` returns. A kill in between leaves the
    # first stale and the second current, so the second wins -- otherwise the
    # gate would restart with no baseline and promote a second time.
    champion = gate.load_champion()
    if champion is not None:
        state = state.model_copy(
            update={"champion": champion.result, "champion_path": champion.path}
        )
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
    for _ in range(iterations):
        state = iterate(
            state, archive, pool, mutator, workers, concurrency, rng, log, commit
        )
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
    log: wandb.Run,
    commit: bool = True,
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
        log: The wandb run metrics go to.
        commit: Whether a promotion records itself in the repository.

    Returns:
        The advanced state, or the state unchanged when the day's mutation
        quota is spent: an iteration that ran no mutation left the archive
        exactly as it found it, so advancing the counter would walk the
        migrate/epoch/reset cadence over a population nothing touched.
    """
    today = datetime.date.today().isoformat()
    if state.day != today:
        state = state.model_copy(update={"day": today, "calls_today": 0})
    if state.calls_today >= config.DAILY_CALL_BUDGET:
        LOGGER.warning("daily call budget spent; idling until it resets")
        time.sleep(QUOTA_SLEEP_SECONDS)
        return state
    # `random.Random` is not thread-safe, so each mutation carries its own
    # generator, drawn from the loop's before any thread starts. The threads
    # share one `Archive`, which is safe only because there is exactly one job
    # per island per iteration: every `insert` and every `record_failure` from
    # a given thread lands in that thread's own island, so no two threads ever
    # read or replace the same island's worst program.
    jobs = [
        (island, _plan(archive, island, rng), random.Random(rng.random()))
        for island in range(config.ISLANDS)
    ]
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        outcomes = list(
            executor.map(
                lambda job: _mutate(job, archive, pool, mutator, workers), jobs
            )
        )
    state = state.model_copy(update={"calls_today": state.calls_today + len(outcomes)})
    iteration = state.iteration + 1
    state = state.model_copy(update={"iteration": iteration})
    log.log(_iteration_record(iteration, state, outcomes, archive))
    if iteration % config.MIGRATION_INTERVAL == 0:
        archive.migrate()
    if iteration % config.EPOCH_INTERVAL == 0:
        # The mutation executor has joined and nothing else is running, so the
        # epoch gets the whole budget rather than one mutation's share of it.
        state = epoch(state, archive, pool, config.CORE_BUDGET, log, commit)
    if iteration % config.RESET_INTERVAL == 0 and archive.top(1):
        archive.reset_worst_island(archive.top(1)[0])
    return state


def _iteration_record(
    iteration: int,
    state: State,
    outcomes: list[tuple[Mutation, float | None]],
    archive: archive_module.Archive,
) -> dict[str, float | int]:
    """What one iteration cost and what it put in the archive.

    Args:
        iteration: The iteration just completed.
        state: The state as of that iteration.
        outcomes: Each island's mutation and its child's fast fitness, or
            None where no child reached evaluation. A call is ``ok`` when it
            yielded a child, whether or not the session finished.
        archive: The population after the iteration's inserts.

    Returns:
        The metrics for ``wandb.Run.log``, keyed by ``iteration``.
    """
    mutations = [mutation for mutation, _ in outcomes]
    fitnesses = [fitness for _, fitness in outcomes if fitness is not None]
    ok = sum(mutation.child is not None for mutation in mutations)
    record: dict[str, float | int] = {
        "iteration": iteration,
        "calls/ok": ok,
        "calls/failed": len(mutations) - ok,
        "calls/timed_out": sum(m.status == "timeout" for m in mutations),
        "calls/input_tokens": sum(m.input_tokens for m in mutations),
        "calls/output_tokens": sum(m.output_tokens for m in mutations),
        "calls/seconds": sum(m.seconds for m in mutations) / len(mutations),
        "calls/today": state.calls_today,
        "fast/evaluated": len(fitnesses),
        "archive/programs": sum(len(archive.island(i)) for i in range(config.ISLANDS)),
    }
    if fitnesses:
        record["fast/best_child"] = max(fitnesses)
    top = archive.top(1)
    if top:
        record["archive/top"] = top[0].mean
    return record


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
) -> tuple[Mutation, float | None]:
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

    Returns:
        The mutation and the child's fast fitness, or None where the child
        never reached evaluation.
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
    parents = [parent.id] + ([inspiration.id] if inspiration else [])
    if mutation.child is None:
        archive.record_failure(
            island, parents, kind, f"{mutation.status}: {mutation.reason}"
        )
        return mutation, None
    verdict = validate.validate(mutation.child)
    if verdict.status != "ok":
        archive.record_failure(
            island, parents, kind, f"{verdict.status}: {verdict.reason}"
        )
        return mutation, None
    stored = archive.store(mutation.child.read_text(encoding="utf-8"), program_id)
    try:
        result = evaluator.fast(stored, pool, rng, workers)
    except harness.OpponentCrash:
        # A broken opponent is not this lineage's failure and must not become
        # its feedback; the pool is broken, so the loop halts (spec section 8).
        raise
    except RuntimeError as error:
        archive.record_failure(island, parents, kind, f"fast: {error}")
        return mutation, None
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
    return mutation, result.fitness


def epoch(
    state: State,
    archive: archive_module.Archive,
    pool: Pool,
    workers: int,
    log: wandb.Run,
    commit: bool = True,
) -> State:
    """Deep-evaluate the top K on the exam block; promote if the rule says so.

    The champion is re-measured first, on the pool as it stands now. Its
    stored result was computed against the pool as it was at its own
    promotion -- before it joined that pool, before the weights renormalised
    and before weakness pressure moved them -- so comparing a candidate's
    fresh score against it would be two numbers from two different exams. One
    extra deep evaluation per epoch buys a comparison that means something.

    Args:
        state: The state this epoch closes.
        archive: The population the candidates come from.
        pool: The opponents that count towards the score; a promotion
            changes it and saves it.
        workers: Processes each deep evaluation fans its games over.
        log: The wandb run the epoch's scores go to; a promotion also logs
            the champion's file as an artifact named after it.
        commit: Whether a promotion records itself in the repository.

    Returns:
        The state, carrying the champion's fresh score and the new champion
        if one was promoted.
    """
    if state.champion_path is not None:
        baseline = evaluator.deep(Path(state.champion_path), "champion", pool, workers)
        state = state.model_copy(update={"champion": baseline})
    measured: list[archive_module.Program] = []
    results: list[DeepResult] = []
    for program in archive.top(config.DEEP_TOP_K):
        try:
            results.append(
                evaluator.deep(Path(program.source_path), program.id, pool, workers)
            )
        except harness.OpponentCrash:
            # Not the candidate's failure, so not its archive line: the gate
            # halts rather than record a verdict it did not compute.
            raise
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
    record: dict[str, float | int] = {"iteration": state.iteration}
    if state.champion is not None:
        record["deep/champion"] = state.champion.score
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
            champion = gate.promote(program, best, pool, commit=commit)
            promoted = champion.name
            state = state.model_copy(
                update={"champion": champion.result, "champion_path": champion.path}
            )
            artifact = wandb.Artifact(
                champion.name, type="champion", metadata=best.model_dump()
            )
            artifact.add_file(champion.path)
            log.log_artifact(artifact)
    record["deep/promoted"] = int(promoted is not None)
    log.log(record)
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
