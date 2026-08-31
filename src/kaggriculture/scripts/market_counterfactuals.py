"""Turning seasons into counterfactual shards at a rate somebody budgeted.

This command is the only producer of market-residual counterfactual evidence,
and its three modes exist because the expensive thing and the checkable thing
have to be separable. ``--create`` claims a fresh root and writes down the
parameters every later run must match. ``--resume`` reopens a claimed root,
reconciles what is already published, and collects only what is missing.
``--validate-only`` does the reconcile and stops: it never records a season,
never branches an event, and never writes an artifact, so a root can be audited
for the price of reading it.

**The budget is the interface.** A branch is not cheap and the cost is not where
intuition puts it. Each arm restores two fresh controllers, and one fresh load
of the served controller costs about 0.9 s -- almost none of which is compiling
its source (~1.2 ms); the rest is the payload's own module-level work, which a
cache cannot skip without handing two arms the same mutable tables. On top of
that each arm replays the whole season prefix through those controllers and then
plays the tail forward in the batched simulator. So a mid-season event at the
shipped 48-row budget is minutes of work, and ``--max-events`` and
``--max-alternatives`` are not conveniences: they are the only two dials that
decide whether a collection finishes this week. The report writes down what each
phase cost so the next run's dials are set from a measurement.

**Opponents must be restorable, and being loadable is not the same thing.** A
branch leaves the recorded season at turn ``k`` and both seats must go on
deciding on a board that no longer exists. The learner and the opponent are
therefore ``BaselineIdentity`` controllers, reloaded fresh and replayed to turn
``k`` -- ``AgentTranscript.restore`` proves the replay by refusing any action
that differs from the recorded one. Two whole classes of opponent cannot pass
that bar and are refused by name, before any season is played:

* **Route replayers.** ``kaggriculture.searched_route_policy`` and every
  ``search.arena`` route opponent choose their action from ``observation["step"]``
  alone -- their own source says ``_ROUTE[step + 1]``. They restore without
  complaint, because a transcript replay only ever asks them for the turns they
  already played, and then, on the diverged board, they keep emitting actions
  harvested from a game that is not this one. Nothing raises. The season
  finishes. The label is meaningless. That silence is why the refusal is a hard
  raise on the name rather than a check on the outcome.
* **Frontier artifacts.** The verified public frontier addresses its agents by
  path and digest under ``--artifact-root``, not by importable module, so
  ``BaselineIdentity`` cannot name one and ``load_verified_baseline`` cannot
  produce the private per-arm copy a branch needs. Admitting them means
  extending the identity schema every published shard is already stamped with.

The manifest and the pinned frontier generation are still consumed, and they earn
their place twice: they pin the engine and the field this run's identity was
chosen against, and they turn a mistyped ``--opponent`` into a diagnosis instead
of a ``KeyError``.

**Nothing here overwrites evidence.** One nonblocking ``flock`` on the root is
taken before any file is touched, so two collectors cannot interleave. Only a
shard that has both its bytes and its manifest counts as published, and it is
re-read through ``load_shard_strict`` before it is skipped -- a corrupt shard
stops the run rather than shrinking the dataset by one cell. A shard whose
manifest never landed is simply collected again: ``write_shard_atomic``
re-links identical bytes as a no-op and publishes the missing manifest, and
raises if the bytes differ. Orphan temporaries from a killed worker are moved
aside into ``quarantine/`` rather than deleted, because a half-written shard is
the only evidence of how a worker died.
"""

import argparse
import fcntl
import hashlib
import json
import logging
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence, TypeVar

import torch
from tqdm import tqdm

from kaggriculture.learn.market_residual.alternatives import (
    FIXED_FAMILIES,
    AlternativeConfig,
)
from kaggriculture.learn.market_residual.artifacts import (
    MANIFEST_SUFFIX,
    CounterfactualShard,
    ShardIdentity,
    counterfactual_rows,
    load_shard_strict,
    shard_path,
    write_shard_atomic,
)
from kaggriculture.learn.market_residual.counterfactual import (
    ENGINE_VERSION,
    SeasonIdentity,
    canonical_digest,
    record_season,
    season_events,
    snapshot_event,
)
from kaggriculture.market_residual.baseline import (
    SERVED,
    BaselineIdentity,
    load_report,
    reset_load_report,
)
from kaggriculture.market_residual.events import EventConfig
from kaggriculture.market_residual.schema import validate_seed_bank
from kaggriculture.search.frontier import verify_frontier

LOGGER = logging.getLogger(__name__)

# One recorded market event; parameterised only so the spread can be pinned on
# integers instead of on a season that has to be played to be tested.
Event = TypeVar("Event")

COLLECTION_VERSION = 1

PARAMETERS_NAME = "collection-parameters.json"
REPORT_NAME = "collection-report.json"
LOCK_NAME = "collection.lock"
SHARD_DIRECTORY = "shards"
QUARANTINE_DIRECTORY = "quarantine"
TEMPORARY_GLOB = "*.tmp-*"

DEFAULT_MANIFEST = Path("src/kaggriculture/search/frontier_manifest.json")
DEFAULT_ARTIFACT_ROOT = Path("/data/kaggriculture/search/public-frontier")
DEFAULT_FRONTIER_GENERATION = Path("run/hybrid/frontier.json")

DEFAULT_BASELINE = "kaito_v54"

# Every controller a branch may restore: importable, digest-pinned, and deciding
# from the board it is handed. The frozen baseline is the whole registry today,
# so collection is self-play against it; promoting another vendored controller
# into this mapping is the one change that widens the opponent field, and the
# module docstring says what disqualifies a candidate.
RESTORABLE_OPPONENTS: dict[str, BaselineIdentity] = {DEFAULT_BASELINE: SERVED}

# Opponents that load, restore, and then quietly stop playing the game they are
# in. Named here so the refusal cites the reason rather than the absence.
ROUTE_OPPONENTS: tuple[str, ...] = ("route", "searched_route_policy")

# The banks collection may draw from. Every other declared bank is spent on a
# decision -- a gate, a temporal selection, a single-use promotion -- and a run
# that trained on one would be graded on its own training seeds.
COLLECTION_BANKS: tuple[str, ...] = ("counterfactual_train", "counterfactual_temporal")

MAXIMUM_WORKERS = 32

# One anchor, three fixed families, and at least one row from each of the two
# generated families.
MINIMUM_ALTERNATIVES = 6

SHIPPED_BUDGET = AlternativeConfig()

# Half the slowest of three consecutive eight-seed profiles measured on this
# workstation: `--workers 8 --max-events 8 --max-alternatives 16` over seeds
# 860000..860007 in both seats branched 128 events and wrote 994 rows in 1059 s,
# 1066 s and 1068 s -- 435.0, 432.4 and 431.4 events/hour, a 0.8% spread. The
# machine is 64 CPU threads with another project holding a GPU throughout; each
# run reported a one-minute load average of 4.0 to 4.8, almost all of it this
# collection's own eight workers.
#
# The floor is half the minimum rather than just under it because this number is
# a property of a machine as much as of the code. Contention here costs about
# 35%: the same controller load that takes 0.89 s alone took 1.23 s averaged
# over the 2020 loads eight workers made. A run below half of a quiet-machine
# rate is either sharing the box with something serious or has regressed, and
# either way its timings should not be extrapolated into a full collection.
THROUGHPUT_FLOOR_EVENTS_PER_HOUR = 215.0


class CollectionParameterError(RuntimeError):
    """Raised when a root's stored parameters and this run's disagree."""


class CollectionLockError(RuntimeError):
    """Raised when a collection root is already owned by a live collector."""


class UnrestorableOpponentError(RuntimeError):
    """Raised when an opponent cannot go on deciding after a branch."""


class CollectionThroughputError(RuntimeError):
    """Raised when a collecting run is too slow to be worth extrapolating."""


@dataclass(frozen=True)
class CollectionPlan:
    """Every cell this run will produce, and what it was parameterised by."""

    root: Path
    identities: tuple[SeasonIdentity, ...]
    max_events: int
    workers: int
    provenance: dict[str, Any]


@dataclass(frozen=True)
class PhaseTimings:
    """Where one cell's wall time went, by the phase that spent it.

    ``policy_seconds`` is not a sixth slice: it is the part of ``engine`` and
    ``branch`` that went into loading fresh controllers, reported separately
    because it is the cost a budget can actually do something about.
    """

    engine_seconds: float
    feature_seconds: float
    policy_seconds: float
    snapshot_seconds: float
    branch_seconds: float
    fsync_seconds: float


@dataclass(frozen=True)
class CellResult:
    """One published seat-cell: what it produced and what it cost."""

    seed: int
    seat: int
    events: int
    rows: int
    loads: int
    timings: PhaseTimings


def main(argv: Sequence[str] | None = None) -> None:
    """Plan, own, reconcile, and collect one counterfactual root.

    Args:
        argv: The argument vector; the process's own when omitted.

    Raises:
        CollectionThroughputError: If a run that branched events was slower
            than the floor this machine was measured at.
    """
    args = parser().parse_args(None if argv is None else list(argv))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    plan = collection_plan(args)
    started = time.perf_counter()
    with root_lock(plan.root):
        parameters = reconcile_parameters(plan, create=args.create)
        quarantined = quarantine_temporaries(plan.root)
        published, outstanding, reconciled_rows = reconcile_shards(plan)
        LOGGER.info(
            "%d of %d cells published, %d outstanding, %d temporaries quarantined",
            len(published),
            len(plan.identities),
            len(outstanding),
            quarantined,
        )
        results = () if args.validate_only else collect(plan, outstanding)
        wall_seconds = time.perf_counter() - started
        report = collection_report(
            plan=plan,
            parameters=parameters,
            validate_only=args.validate_only,
            published=len(published),
            outstanding=len(outstanding) - len(results),
            reconciled_rows=reconciled_rows,
            quarantined=quarantined,
            results=results,
            wall_seconds=wall_seconds,
        )
        write_json_atomic(plan.root / REPORT_NAME, report)
    LOGGER.info(
        "%d cells collected in %.1f s (%.1f events/hour)",
        len(results),
        wall_seconds,
        report["throughput"]["events_per_hour"],
    )
    check_throughput(report["events"]["branched"], wall_seconds)


def parser() -> argparse.ArgumentParser:
    """Build the command line without reading a file or playing a game.

    Returns:
        The parser; exactly one of ``--create`` and ``--resume`` is required.
    """
    arguments = argparse.ArgumentParser(
        prog="market_counterfactuals", description=__doc__
    )
    arguments.add_argument("root", type=Path, help="collection root to own")
    mode = arguments.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create", action="store_true", help="claim a fresh root")
    mode.add_argument("--resume", action="store_true", help="reopen a claimed root")
    arguments.add_argument(
        "--validate-only",
        action="store_true",
        help="reconcile and report; play nothing",
    )
    arguments.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    arguments.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    arguments.add_argument(
        "--frontier-generation", type=Path, default=DEFAULT_FRONTIER_GENERATION
    )
    arguments.add_argument("--baseline", default=DEFAULT_BASELINE)
    arguments.add_argument("--opponent", default=DEFAULT_BASELINE)
    arguments.add_argument("--seed-bank", default=COLLECTION_BANKS[0])
    arguments.add_argument("--seed-start", type=int, default=860_000)
    arguments.add_argument("--seed-count", type=int, default=8)
    arguments.add_argument("--workers", type=int, default=8)
    arguments.add_argument("--max-events", type=int, default=8)
    arguments.add_argument(
        "--max-alternatives", type=int, default=SHIPPED_BUDGET.max_alternatives
    )
    return arguments


def collection_plan(args: argparse.Namespace) -> CollectionPlan:
    """Return every cell this run owns, refusing anything unplayable first.

    Nothing here opens the engine, so every refusal below happens before the
    first season is recorded.

    Args:
        args: The parsed command line.

    Returns:
        The plan: one season identity per seat-cell, in seed then seat order.

    Raises:
        ValueError: If the worker count, event cap, bank or seeds are outside
            what this command may collect.
    """
    if not 1 <= args.workers <= MAXIMUM_WORKERS:
        raise ValueError(f"workers must be 1..{MAXIMUM_WORKERS}, got {args.workers}")
    if args.max_events < 1:
        raise ValueError(f"max events must be at least 1, got {args.max_events}")
    if args.seed_count < 1:
        raise ValueError(f"seed count must be at least 1, got {args.seed_count}")
    budget = alternative_budget(args.max_alternatives)
    if args.seed_bank not in COLLECTION_BANKS:
        raise ValueError(
            f"{args.seed_bank!r} is not a collection bank; collection draws from "
            f"{list(COLLECTION_BANKS)}"
        )
    seeds = validate_seed_bank(
        args.seed_bank, range(args.seed_start, args.seed_start + args.seed_count)
    )
    provenance = frontier_provenance(
        args.manifest, args.artifact_root, args.frontier_generation
    )
    learner = restorable_identity(args.baseline, provenance)
    opponent = restorable_identity(args.opponent, provenance)
    seats: tuple[Literal[0, 1], ...] = (0, 1)
    return CollectionPlan(
        root=args.root,
        identities=tuple(
            SeasonIdentity.current(
                learner=learner,
                opponent=opponent,
                seed_bank=args.seed_bank,
                seed=seed,
                seat=seat,
                event_config=EventConfig(),
                alternative_config=budget,
            )
            for seed in seeds
            for seat in seats
        ),
        max_events=args.max_events,
        workers=args.workers,
        provenance=provenance,
    )


def alternative_budget(cap: int) -> AlternativeConfig:
    """Return the per-family budget one total alternative cap allows.

    The shipped split -- 36 single-slot rows to 8 ranked combinations -- is kept
    in proportion as the cap shrinks, so lowering the cap buys throughput
    without silently turning collection into a different experiment that only
    ever sees one family.

    Args:
        cap: The most alternatives one event may produce, anchor included.

    Returns:
        The budget, whose families sum to exactly ``cap``.

    Raises:
        ValueError: If the cap leaves no row for some family.
    """
    if cap < MINIMUM_ALTERNATIVES:
        raise ValueError(
            f"{cap} alternatives leaves a family with no row; the minimum is "
            f"{MINIMUM_ALTERNATIVES}"
        )
    available = cap - 1 - len(FIXED_FAMILIES)
    shipped = SHIPPED_BUDGET.max_single + SHIPPED_BUDGET.max_ranked_multi
    ranked = max(1, round(available * SHIPPED_BUDGET.max_ranked_multi / shipped))
    return AlternativeConfig(
        max_single=available - ranked, max_ranked_multi=ranked, max_alternatives=cap
    )


def frontier_provenance(
    manifest: Path, artifact_root: Path, generation: Path
) -> dict[str, Any]:
    """Verify the public frontier and the generation pinned against it.

    Args:
        manifest: The frontier manifest.
        artifact_root: Where its artifacts live.
        generation: A frontier report produced from that manifest.

    Returns:
        The engine, both digests, the generation's leader, and every agent name
        the field contains.

    Raises:
        ValueError: If the generation was not produced under this engine, or
            not against this manifest.
    """
    verified = verify_frontier(manifest, artifact_root)
    if verified.engine != ENGINE_VERSION:
        raise ValueError(
            f"manifest engine {verified.engine}, expected {ENGINE_VERSION}"
        )
    payload = generation.read_bytes()
    report = json.loads(payload)
    if report["engine"] != ENGINE_VERSION:
        raise ValueError(
            f"{generation} was played on engine {report['engine']}, "
            f"expected {ENGINE_VERSION}"
        )
    if report["manifest_sha256"] != verified.manifest_sha256:
        raise ValueError(
            f"{generation} was played against manifest {report['manifest_sha256']}, "
            f"this run verified {verified.manifest_sha256}"
        )
    return {
        "engine": verified.engine,
        "manifest_sha256": verified.manifest_sha256,
        "frontier_generation": str(generation),
        "frontier_generation_sha256": hashlib.sha256(payload).hexdigest(),
        "frontier_name": report["frontier_name"],
        "field": sorted(
            {row["name"] for row in report["rows"]} | set(verified.opponents)
        ),
    }


def restorable_identity(name: str, provenance: dict[str, Any]) -> BaselineIdentity:
    """Return the verified controller one name refers to, or say why not.

    Args:
        name: The controller named on the command line.
        provenance: The verified frontier this run is pinned to.

    Returns:
        The identity a branch can reload fresh and replay.

    Raises:
        UnrestorableOpponentError: If the name is a route replayer, a frontier
            artifact, or unknown.
    """
    if name in RESTORABLE_OPPONENTS:
        return RESTORABLE_OPPONENTS[name]
    if name in ROUTE_OPPONENTS:
        raise UnrestorableOpponentError(
            f"{name!r} chooses its action from the season clock, not from the "
            "board: after a branch it replays a route harvested from a game "
            "that no longer exists, and nothing raises"
        )
    if name in provenance["field"]:
        raise UnrestorableOpponentError(
            f"{name!r} is a frontier agent addressed by path and digest, not by "
            "module, so a branch cannot load the private per-arm copy it needs"
        )
    raise UnrestorableOpponentError(
        f"unknown opponent {name!r}; restorable controllers are "
        f"{sorted(RESTORABLE_OPPONENTS)}"
    )


@contextmanager
def root_lock(root: Path) -> Iterator[None]:
    """Own one collection root exclusively, or refuse immediately.

    Args:
        root: The collection root.

    Yields:
        Nothing; the lock is held for the body.

    Raises:
        CollectionLockError: If another collector holds the root.
    """
    root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(root / LOCK_NAME, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise CollectionLockError(f"{root} is held by another collector") from error
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def reconcile_parameters(plan: CollectionPlan, create: bool) -> dict[str, Any]:
    """Bind a root to one parameterisation, once, and hold it there.

    Args:
        plan: The plan this invocation would collect.
        create: Whether the root is being claimed rather than reopened.

    Returns:
        The stored parameters, fields and digest.

    Raises:
        CollectionParameterError: If a create meets a live root, a resume meets
            an unclaimed one, the stored file does not match its own digest, or
            the stored parameters are not this run's.
    """
    path = plan.root / PARAMETERS_NAME
    wanted = collection_parameters(plan)
    if create:
        if path.exists():
            raise CollectionParameterError(
                f"{plan.root} is already collecting; use --resume"
            )
        write_json_atomic(path, wanted)
        return wanted
    if not path.exists():
        raise CollectionParameterError(f"{plan.root} was never created; use --create")
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored["sha256"] != canonical_digest(stored["fields"]):
        raise CollectionParameterError(f"{path} does not match its own digest")
    if stored["fields"] != wanted["fields"]:
        differing = sorted(
            key
            for key in set(stored["fields"]) | set(wanted["fields"])
            if stored["fields"].get(key) != wanted["fields"].get(key)
        )
        raise CollectionParameterError(
            f"{plan.root} was created with different parameters: {differing}"
        )
    return stored


def collection_parameters(plan: CollectionPlan) -> dict[str, Any]:
    """Return the canonical parameterisation a root is bound to.

    The cell digests are part of it, so a rerun that changed the controller, the
    thresholds, the budget or the engine differs here rather than at the point
    where it would have written a shard nobody could interpret.

    Args:
        plan: The planned run.

    Returns:
        The fields and their digest.
    """
    first = plan.identities[0]
    fields: dict[str, Any] = {
        "version": COLLECTION_VERSION,
        "engine": ENGINE_VERSION,
        "seed_bank": first.seed_bank,
        "seeds": sorted({identity.seed for identity in plan.identities}),
        "max_events": plan.max_events,
        "learner_sha256": first.learner.sha256,
        "opponent_sha256": first.opponent.sha256,
        "event_config": asdict(first.event_config),
        "alternative_config": asdict(first.alternative_config),
        "cells": [identity.sha256 for identity in plan.identities],
        "provenance": plan.provenance,
    }
    return {"fields": fields, "sha256": canonical_digest(fields)}


def quarantine_temporaries(root: Path) -> int:
    """Move every orphan shard temporary aside, keeping it readable.

    Args:
        root: The collection root.

    Returns:
        How many temporaries were moved.
    """
    shards = root / SHARD_DIRECTORY
    shards.mkdir(parents=True, exist_ok=True)
    orphans = sorted(shards.glob(TEMPORARY_GLOB))
    if not orphans:
        return 0
    quarantine = root / QUARANTINE_DIRECTORY
    quarantine.mkdir(parents=True, exist_ok=True)
    for orphan in orphans:
        shutil.move(str(orphan), str(quarantine / orphan.name))
    return len(orphans)


def reconcile_shards(
    plan: CollectionPlan,
) -> tuple[tuple[SeasonIdentity, ...], tuple[SeasonIdentity, ...], int]:
    """Split the plan into what is published and what is still owed.

    Args:
        plan: The planned run.

    Returns:
        The published cells, the outstanding cells, and how many rows the
        published cells already hold.

    Raises:
        CounterfactualIntegrityError: If a manifested shard fails any of the
            loader's checks, which is a corruption this run may not paper over.
    """
    directory = plan.root / SHARD_DIRECTORY
    published: list[SeasonIdentity] = []
    outstanding: list[SeasonIdentity] = []
    rows = 0
    for identity in plan.identities:
        path = shard_path(directory, ShardIdentity.of(identity))
        if path.exists() and path.with_suffix(MANIFEST_SUFFIX).exists():
            rows += len(load_shard_strict(path, identity).rows)
            published.append(identity)
        else:
            outstanding.append(identity)
    return tuple(published), tuple(outstanding), rows


def collect(
    plan: CollectionPlan, outstanding: Sequence[SeasonIdentity]
) -> tuple[CellResult, ...]:
    """Play every outstanding cell and publish its shard.

    Cells run in processes rather than threads. The frozen controller keeps its
    per-episode memory in module-level closures and its replay is pure Python,
    so the work is interpreter-bound and one process per cell is both the only
    way to use more than one core and the isolation that keeps a crashed cell
    from taking the run down with it.

    Args:
        plan: The planned run.
        outstanding: The cells still owed.

    Returns:
        One result per collected cell.
    """
    if not outstanding:
        return ()
    with ProcessPoolExecutor(
        max_workers=plan.workers, initializer=bound_worker
    ) as pool:
        futures = [
            pool.submit(collect_cell, plan.root, identity, plan.max_events)
            for identity in outstanding
        ]
        return tuple(
            future.result()
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="cells", unit="cell"
            )
        )


def bound_worker() -> None:
    """Keep one worker to one core, so the pool's width is the pool's width."""
    torch.set_num_threads(1)


def collect_cell(root: Path, identity: SeasonIdentity, max_events: int) -> CellResult:
    """Record one season, branch its budgeted events, and publish one shard.

    Args:
        root: The collection root.
        identity: The cell to collect.
        max_events: The most events this cell may branch.

    Returns:
        What the cell produced and what each phase of it cost.
    """
    reset_load_report()
    started = time.perf_counter()
    season = record_season(identity)
    engine_seconds = time.perf_counter() - started

    started = time.perf_counter()
    events = selected_events(season_events(season), max_events)
    feature_seconds = time.perf_counter() - started

    rows = []
    snapshot_seconds = 0.0
    branch_seconds = 0.0
    for event in events:
        started = time.perf_counter()
        snapshot = snapshot_event(season, event)
        snapshot_seconds += time.perf_counter() - started
        started = time.perf_counter()
        rows.extend(counterfactual_rows(snapshot))
        branch_seconds += time.perf_counter() - started

    started = time.perf_counter()
    write_shard_atomic(
        root / SHARD_DIRECTORY,
        CounterfactualShard(identity=ShardIdentity.of(identity), rows=tuple(rows)),
    )
    fsync_seconds = time.perf_counter() - started

    loads = load_report()
    return CellResult(
        seed=identity.seed,
        seat=identity.seat,
        events=len(events),
        rows=len(rows),
        loads=loads.count,
        timings=PhaseTimings(
            engine_seconds=engine_seconds,
            feature_seconds=feature_seconds,
            policy_seconds=loads.seconds,
            snapshot_seconds=snapshot_seconds,
            branch_seconds=branch_seconds,
            fsync_seconds=fsync_seconds,
        ),
    )


def selected_events(events: Sequence[Event], max_events: int) -> tuple[Event, ...]:
    """Return at most ``max_events`` events, spread across the season.

    The selection is even rather than the first ``n``: the first events of a
    season are the cheapest to branch and the least like the endgame the residual
    is being trained for, so taking the prefix would buy speed by collecting a
    different distribution.

    Args:
        events: Every event the season opened, in order.
        max_events: The cap.

    Returns:
        The chosen events, in the order they opened.
    """
    if len(events) <= max_events:
        return tuple(events)
    stride = len(events) / max_events
    return tuple(events[int(index * stride)] for index in range(max_events))


def collection_report(
    plan: CollectionPlan,
    parameters: dict[str, Any],
    validate_only: bool,
    published: int,
    outstanding: int,
    reconciled_rows: int,
    quarantined: int,
    results: Sequence[CellResult],
    wall_seconds: float,
) -> dict[str, Any]:
    """Return everything the next run needs to set its own dials.

    Args:
        plan: The planned run.
        parameters: The parameters the root is bound to.
        validate_only: Whether anything was allowed to be played.
        published: How many cells were already published.
        outstanding: How many cells remain owed after this run.
        reconciled_rows: How many rows the published cells hold.
        quarantined: How many orphan temporaries were moved aside.
        results: One entry per cell collected by this run.
        wall_seconds: How long the owned section of this run took.

    Returns:
        The report, JSON-safe.
    """
    branched = sum(result.events for result in results)
    timings = {
        field: sum(getattr(result.timings, field) for result in results)
        for field in PhaseTimings.__dataclass_fields__
    }
    return {
        "version": COLLECTION_VERSION,
        "validate_only": validate_only,
        "parameters_sha256": parameters["sha256"],
        "cells": {
            "planned": len(plan.identities),
            "reconciled": published,
            "outstanding": outstanding,
            "collected": len(results),
        },
        "events": {"branched": branched, "cap_per_cell": plan.max_events},
        "rows": {
            "reconciled": reconciled_rows,
            "collected": sum(result.rows for result in results),
        },
        "quarantined": quarantined,
        "loads": sum(result.loads for result in results),
        "timings": timings,
        "throughput": {
            "wall_seconds": wall_seconds,
            "events_per_hour": events_per_hour(branched, wall_seconds),
            "floor_events_per_hour": THROUGHPUT_FLOOR_EVENTS_PER_HOUR,
        },
        "machine": {
            "workers": plan.workers,
            "cpu_count": os.cpu_count(),
            "load_average": list(os.getloadavg()),
        },
        "cells_collected": [
            {"seed": result.seed, "seat": result.seat, "events": result.events}
            for result in sorted(results, key=lambda cell: (cell.seed, cell.seat))
        ],
    }


def events_per_hour(events: int, wall_seconds: float) -> float:
    """Return the rate one run branched events at.

    Args:
        events: How many events were branched.
        wall_seconds: How long the run took.

    Returns:
        Events per hour.
    """
    return events * 3600.0 / wall_seconds


def check_throughput(events: int, wall_seconds: float) -> None:
    """Refuse to call a run that branched too slowly a usable profile.

    A run that branched nothing -- a validate-only pass, or a fully reconciled
    resume -- is not graded, because there is no rate to measure.

    Args:
        events: How many events were branched.
        wall_seconds: How long the run took.

    Raises:
        CollectionThroughputError: If the measured rate is below the floor.
    """
    if events == 0:
        return
    rate = events_per_hour(events, wall_seconds)
    if rate < THROUGHPUT_FLOOR_EVENTS_PER_HOUR:
        raise CollectionThroughputError(
            f"{rate:.1f} events/hour is below the floor of "
            f"{THROUGHPUT_FLOOR_EVENTS_PER_HOUR:.1f} events/hour; check the "
            "machine's load average before lowering the budget"
        )


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON document so a reader never sees half of it.

    Args:
        path: Where the document belongs.
        payload: The document.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
