"""Run resumable progressive hybrid evolution at low CPU priority."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from kaggriculture.hybrid.config import HybridConfig
from kaggriculture.search.evolution import (
    EvolutionConfig,
    SearchState,
    evolve,
    write_finalists,
)
from kaggriculture.search.fitness import strength_weights
from kaggriculture.search.frontier import FrontierReport, verify_frontier
from kaggriculture.search.genome import GenomeCodec
from kaggriculture.search.scripts.frontier_round_robin import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_MANIFEST,
    FRONTIER_SEEDS,
)


def parser() -> argparse.ArgumentParser:
    """Build the CPU-search command line without touching external state."""
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    arguments.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    arguments.add_argument("--frontier-report", type=Path, required=True)
    arguments.add_argument("--workers", type=int, default=16)
    arguments.add_argument("--seed", type=int, default=20_260_825)
    arguments.add_argument("--output", type=Path, required=True)
    arguments.add_argument("--finalists", type=Path, required=True)
    arguments.add_argument(
        "--resume",
        type=Path,
        help="explicit state to resume; by default an existing --output resumes",
    )
    return arguments


def seed_configs() -> tuple[HybridConfig, ...]:
    """Return the validated baseline from which mutations begin."""
    return (HybridConfig.default(),)


def main() -> None:
    """Verify frontier identity, resume exactly, evolve, and write finalists."""
    args = parser().parse_args()
    if not 1 <= args.workers <= 16:
        raise SystemExit("--workers must be between 1 and 16 while Toad is running")
    os.nice(10)
    frontier = verify_frontier(args.manifest, args.artifact_root)
    try:
        report = TypeAdapter(FrontierReport).validate_json(
            args.frontier_report.read_text()
        )
    except (OSError, ValidationError, ValueError) as error:
        raise SystemExit(f"invalid frontier report: {error}") from error
    if report.engine != frontier.engine:
        raise SystemExit(
            f"frontier report engine {report.engine} != "
            f"verified engine {frontier.engine}"
        )
    if report.seeds != FRONTIER_SEEDS:
        raise SystemExit("frontier report does not use the fixed 64 frontier seeds")
    if report.failures:
        raise SystemExit("frontier report contains failures and cannot seed search")
    if {row.name for row in report.rows} != set(frontier.opponents):
        raise SystemExit("frontier report rows do not match the verified league")
    resume_path = args.resume
    if resume_path is None and args.output.exists():
        resume_path = args.output
    resume = SearchState.load(resume_path) if resume_path is not None else None
    config = EvolutionConfig(
        workers=args.workers,
        seed=args.seed,
        engine=frontier.engine,
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
    )
    state = evolve(
        codec=GenomeCodec.default(),
        initial_configs=seed_configs(),
        league=frontier.opponents,
        weights=strength_weights(report),
        config=config,
        output=args.output,
        resume=resume,
    )
    write_finalists(state, args.finalists)


if __name__ == "__main__":
    main()
