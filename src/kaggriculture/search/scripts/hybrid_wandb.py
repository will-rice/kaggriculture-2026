"""Mirror durable hybrid-search generations into a fault-isolated W&B run.

The search process owns ``search-state.json`` and replaces it atomically after
each complete generation.  This sidecar only reads that file.  W&B outages,
sidecar restarts, and cursor writes therefore cannot change or stop evolution.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from kaggriculture.search.evolution import GenerationRecord, SearchState

ENTITY = "will-rice"
PROJECT = "kaggriculture-2026"
CURSOR_SCHEMA_VERSION = 1
LOGGER = logging.getLogger(__name__)


class MetricSink(Protocol):
    """Minimal external-logger boundary used by the pure publisher."""

    def log(self, step: int, metrics: dict[str, float]) -> None:
        """Publish one complete durable generation."""


@dataclass(frozen=True)
class Cursor:
    """Last generation successfully handed to the external logger."""

    search_identity: str
    last_generation: int
    observed_at: float


def search_identity(state: SearchState) -> str:
    """Return the stable initial digest shared by every resumed generation."""
    return state.history[0].parent_digest if state.history else state.integrity_digest


def generation_metrics(
    record: GenerationRecord,
    *,
    completed: int,
    target: int,
) -> dict[str, float]:
    """Flatten one canonical generation into W&B-safe scalar metrics."""
    if completed != record.generation + 1:
        raise ValueError("completed generation does not match its record")
    if target < completed:
        raise ValueError("target generations cannot trail completed generations")
    if not record.elites:
        raise ValueError("generation metrics require at least one elite")
    best = record.elites[0]
    if best.fitness is None:
        raise ValueError("generation elite has no fitness")
    metrics = {
        "search/generation": float(completed),
        "search/progress": completed / target if target else 1.0,
        "search/remaining_generations": float(target - completed),
        "screening/candidates": float(len(record.screened)),
        "screening/eligible": float(
            sum(
                candidate.fitness is not None and candidate.fitness.eligible
                for candidate in record.screened
            )
        ),
        "screening/failures": float(
            sum(len(candidate.failures) for candidate in record.screened)
        ),
        "development/candidates": float(len(record.developed)),
        "development/eligible": float(
            sum(
                candidate.fitness is not None and candidate.fitness.eligible
                for candidate in record.developed
            )
        ),
        "development/failures": float(
            sum(len(candidate.failures) for candidate in record.developed)
        ),
        "best/fitness": best.fitness.value,
        "best/weighted_mean": best.fitness.weighted_mean,
        "best/worst": best.fitness.worst,
        "best/margin": best.paired_normalized_margin,
    }
    for name, rate in sorted(best.matchup_rates.items()):
        metrics[f"best/matchup/{name}"] = float(rate)
    for index, elite in enumerate(record.elites):
        if elite.fitness is None:
            raise ValueError(f"generation elite {index} has no fitness")
        metrics.update(
            {
                f"elite/{index}/fitness": elite.fitness.value,
                f"elite/{index}/weighted_mean": elite.fitness.weighted_mean,
                f"elite/{index}/worst": elite.fitness.worst,
                f"elite/{index}/margin": elite.paired_normalized_margin,
            }
        )
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("generation metrics must all be finite")
    return metrics


def wandb_config(
    state: SearchState,
    state_path: Path,
    revision: str,
) -> dict[str, Any]:
    """Describe the complete immutable search identity in the W&B config."""
    identity = state.identity
    evolution = identity.evolution_config
    return {
        "commit": revision,
        "state_path": str(state_path),
        "search_identity": search_identity(state),
        "engine": identity.engine,
        "manifest_sha256": identity.manifest_sha256,
        "genome_schema_sha256": identity.genome_schema_sha256,
        "target_generations": evolution.generations,
        "population": evolution.population,
        "elites": evolution.elites,
        "mutation_sigma": evolution.mutation_sigma,
        "search_seed": evolution.seed,
        "workers": evolution.workers,
        "strength_weights": dict(identity.strength_weights),
        "league": sorted(name for name, _weight in identity.strength_weights),
        "frontier_seeds": list(identity.frontier_seeds),
        "screening_seeds": list(identity.screening_seeds),
        "development_seeds": list(identity.development_seeds),
        "promotion_seeds": list(identity.promotion_seeds),
        "league_snapshots": [list(item) for item in identity.league_snapshots],
    }


def publish_unseen(
    state_path: Path,
    cursor_path: Path,
    sink: MetricSink,
    *,
    expected_identity: str,
    now: float | None = None,
) -> int:
    """Backfill each unseen complete generation and advance the cursor per row."""
    observed_at = time.time() if now is None else now
    if not math.isfinite(observed_at) or observed_at < 0.0:
        raise ValueError("observation time must be finite and non-negative")
    state = SearchState.load(state_path)
    _validate_cursor_path(state_path, cursor_path, state)
    identity = search_identity(state)
    if identity != expected_identity:
        raise ValueError("durable state differs from the opened search identity")
    cursor = _load_cursor(cursor_path)
    if cursor is not None and cursor.search_identity != identity:
        raise ValueError("W&B cursor search identity differs from durable state")
    last_generation = 0 if cursor is None else cursor.last_generation
    if last_generation > state.generation:
        raise ValueError("W&B cursor is ahead of durable search state")
    records = state.history[last_generation:]
    measured_seconds: float | None = None
    if cursor is not None and len(records) == 1:
        elapsed = observed_at - cursor.observed_at
        if math.isfinite(elapsed) and elapsed >= 0.0:
            measured_seconds = elapsed
    published = 0
    target = state.identity.evolution_config.generations
    for record in records:
        completed = record.generation + 1
        metrics = generation_metrics(record, completed=completed, target=target)
        if measured_seconds is not None:
            metrics["monitor/generation_wall_seconds"] = measured_seconds
            eta_seconds = measured_seconds * (target - completed)
            if math.isfinite(eta_seconds):
                metrics["monitor/eta_seconds"] = eta_seconds
        sink.log(completed, metrics)
        _write_cursor_atomic(
            cursor_path,
            Cursor(
                search_identity=identity,
                last_generation=completed,
                observed_at=observed_at,
            ),
        )
        published += 1
    return published


def _load_cursor(path: Path) -> Cursor | None:
    if not path.exists():
        return None
    payload = json.loads(
        path.read_text(),
        parse_constant=_reject_constant,
        object_pairs_hook=_reject_duplicates,
    )
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "search_identity",
        "last_generation",
        "observed_at",
    }:
        raise ValueError("W&B cursor has an invalid schema")
    if payload["schema_version"] != CURSOR_SCHEMA_VERSION:
        raise ValueError("W&B cursor schema version is unsupported")
    identity = payload["search_identity"]
    generation = payload["last_generation"]
    observed_at = payload["observed_at"]
    if not isinstance(identity, str) or len(identity) != 64:
        raise ValueError("W&B cursor search identity must be sha256")
    if type(generation) is not int or generation < 0:
        raise ValueError("W&B cursor generation must be non-negative")
    if (
        not isinstance(observed_at, (int, float))
        or isinstance(observed_at, bool)
        or not math.isfinite(observed_at)
        or observed_at < 0.0
    ):
        raise ValueError("W&B cursor observation time must be finite")
    return Cursor(identity, generation, float(observed_at))


def _validate_cursor_path(
    state_path: Path, cursor_path: Path, state: SearchState
) -> None:
    """Keep sidecar writes outside authoritative state and snapshot inodes."""
    state_absolute = state_path.absolute()
    cursor_absolute = cursor_path.absolute()
    if (
        cursor_absolute == state_absolute
        or cursor_absolute.resolve() == state_absolute.resolve()
        or (
            cursor_path.exists()
            and state_path.exists()
            and cursor_path.samefile(state_path)
        )
    ):
        raise ValueError("W&B cursor path aliases the search state")
    snapshot_paths = tuple(
        Path(snapshot_path)
        for _name, _source_path, snapshot_path, _sha256 in (
            state.identity.league_snapshots
        )
    )
    for snapshot_path in snapshot_paths:
        root = snapshot_path.parent.absolute()
        if (
            cursor_absolute == root
            or root in cursor_absolute.parents
            or cursor_absolute.resolve() == root.resolve()
            or root.resolve() in cursor_absolute.resolve().parents
            or (
                cursor_path.exists()
                and snapshot_path.exists()
                and cursor_path.samefile(snapshot_path)
            )
        ):
            raise ValueError("W&B cursor path aliases the immutable snapshot league")


def _write_cursor_atomic(path: Path, cursor: Cursor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CURSOR_SCHEMA_VERSION,
        **asdict(cursor),
    }
    source = (
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(source)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _reject_constant(value: str) -> None:
    raise ValueError(f"W&B cursor contains non-finite JSON constant {value}")


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"W&B cursor contains duplicate key {key!r}")
        result[key] = value
    return result


class WandbSink:
    """Lazy W&B adapter kept out of the submission/import path."""

    def __init__(
        self,
        state: SearchState,
        state_path: Path,
        revision: str,
        *,
        entity: str,
        project: str,
        name: str | None,
    ) -> None:
        import wandb

        identity = search_identity(state)
        run_id = f"hybrid-search-{identity[:24]}"
        self._wandb = wandb
        self._entity = entity
        self._project = project
        self._run_id = run_id
        self._name = name or run_id
        self._config = wandb_config(state, state_path, revision)

    def log(self, step: int, metrics: dict[str, float]) -> None:
        """Commit and flush one generation before the cursor acknowledges it."""
        run = self._wandb.init(
            entity=self._entity,
            project=self._project,
            job_type="hybrid-search",
            id=self._run_id,
            resume="allow",
            name=self._name,
            config=self._config,
        )
        try:
            run.define_metric("search/generation")
            run.define_metric("*", step_metric="search/generation")
            run.log(metrics, step=step, commit=True)
        finally:
            run.finish()

    def finish(self) -> None:
        """Finish the adapter; each generation is already independently flushed."""


def parser() -> argparse.ArgumentParser:
    """Build the sidecar CLI without reading state or opening W&B."""
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--state", type=Path, required=True)
    arguments.add_argument("--cursor", type=Path)
    arguments.add_argument("--search-commit", required=True)
    arguments.add_argument("--interval", type=float, default=30.0)
    arguments.add_argument("--entity", default=ENTITY)
    arguments.add_argument("--project", default=PROJECT)
    arguments.add_argument("--name")
    arguments.add_argument("--once", action="store_true")
    return arguments


def main() -> None:
    """Backfill durable history, then follow future atomic state updates."""
    args = parser().parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0.0:
        raise SystemExit("--interval must be finite and positive")
    cursor = args.cursor or args.state.with_name(f"{args.state.name}.wandb-cursor")
    while True:
        try:
            state = SearchState.load(args.state)
            opened_identity = search_identity(state)
            sink = WandbSink(
                state,
                args.state,
                args.search_commit,
                entity=args.entity,
                project=args.project,
                name=args.name,
            )
            break
        except Exception:
            if args.once:
                raise
            LOGGER.exception("hybrid W&B initialization failed; retrying")
            time.sleep(args.interval)
    try:
        while True:
            try:
                publish_unseen(
                    args.state,
                    cursor,
                    sink,
                    expected_identity=opened_identity,
                )
            except Exception:
                if args.once:
                    raise
                LOGGER.exception("hybrid W&B publication failed; retrying")
            if args.once:
                return
            time.sleep(args.interval)
    finally:
        sink.finish()


if __name__ == "__main__":
    main()
