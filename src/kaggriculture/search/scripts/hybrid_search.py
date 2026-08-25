"""Run resumable progressive hybrid evolution at low CPU priority."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from kaggriculture.hybrid.config import HybridConfig
from kaggriculture.search.evolution import (
    EvolutionConfig,
    SearchState,
    certify_frontier,
    evolve,
    write_finalists,
)
from kaggriculture.search.fitness import strength_weights
from kaggriculture.search.frontier import (
    FrontierMatchup,
    FrontierReport,
    FrontierRow,
    VerifiedFrontier,
    verify_frontier,
)
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


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)


def _validate_frontier_report(
    report: FrontierReport, frontier: VerifiedFrontier
) -> None:
    """Reject reports not exactly derived from the verified all-pairs field."""
    if report.manifest_sha256 != frontier.manifest_sha256:
        raise SystemExit("frontier report manifest sha256 differs from verification")
    expected_sources = {
        artifact.name: artifact.sha256 for artifact in frontier.artifacts
    }
    if dict(report.source_sha256) != expected_sources:
        raise SystemExit("frontier report source sha256 differs from verification")
    if report.engine != frontier.engine:
        raise SystemExit(
            f"frontier report engine {report.engine} != "
            f"verified engine {frontier.engine}"
        )
    if report.seeds != FRONTIER_SEEDS:
        raise SystemExit("frontier report does not use the fixed 64 frontier seeds")
    if report.failures or any(row.failures for row in report.rows):
        raise SystemExit("frontier report contains failures and cannot seed search")
    names = tuple(frontier.opponents)
    if (
        not report.rows
        or {row.name for row in report.rows} != set(names)
        or len(report.rows) != len(names)
    ):
        raise SystemExit("frontier report rows do not match the verified league")
    ordered = tuple(
        sorted(
            report.rows,
            key=lambda row: (
                -row.field_win_points,
                -row.worst_matchup_win_points,
                row.name,
            ),
        )
    )
    if report.rows != ordered or report.frontier_name != ordered[0].name:
        raise SystemExit("frontier report rank order is not canonical")

    by_pair: dict[tuple[str, str], FrontierMatchup] = {}
    for row in report.rows:
        _validate_report_row(row, names, by_pair)


def _validate_report_row(
    row: FrontierRow,
    names: tuple[str, ...],
    by_pair: dict[tuple[str, str], FrontierMatchup],
) -> None:
    opponents = set(names) - {row.name}
    if (
        set(row.matchups) != opponents
        or {result.opponent for result in row.matchup_results} != opponents
        or len(row.matchup_results) != len(opponents)
    ):
        raise SystemExit("frontier report lacks complete all-pairs matchup coverage")
    for result in row.matchup_results:
        _validate_matchup(row, result, by_pair)
        by_pair[(row.name, result.opponent)] = result
    expected_games = len(opponents) * 2 * len(FRONTIER_SEEDS)
    field_points = (
        sum(result.win_points * result.games for result in row.matchup_results)
        / expected_games
        if expected_games
        else 0.0
    )
    worst = min(row.matchups.values(), default=0.0)
    margin = (
        sum(result.paired_margin * result.games for result in row.matchup_results)
        / expected_games
        if expected_games
        else 0.0
    )
    if (
        row.games != expected_games
        or not _close(row.field_win_points, field_points)
        or not _close(row.worst_matchup_win_points, worst)
        or not _close(row.paired_margin, margin)
    ):
        raise SystemExit("frontier report row aggregates are inconsistent")


def _validate_matchup(
    row: FrontierRow,
    result: FrontierMatchup,
    by_pair: dict[tuple[str, str], FrontierMatchup],
) -> None:
    games_per_matchup = 2 * len(FRONTIER_SEEDS)
    if result.games != games_per_matchup:
        raise SystemExit(
            f"frontier report matchup must contain {games_per_matchup} games"
        )
    if result.failures:
        raise SystemExit("frontier report contains failures and cannot seed search")
    if result.wins + result.draws + result.losses != result.games:
        raise SystemExit("frontier report matchup result counts are inconsistent")
    points = (result.wins + 0.5 * result.draws) / result.games
    if not _close(result.win_points, points) or not _close(
        row.matchups[result.opponent], result.win_points
    ):
        raise SystemExit("frontier report matchup win points are inconsistent")
    reverse = by_pair.get((result.opponent, row.name))
    if reverse is not None and (
        reverse.games != result.games
        or reverse.wins != result.losses
        or reverse.draws != result.draws
        or reverse.losses != result.wins
        or not _close(reverse.win_points + result.win_points, 1.0)
        or not _close(reverse.paired_margin, -result.paired_margin)
    ):
        raise SystemExit("frontier report paired matchup rows disagree")


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
    _validate_frontier_report(report, frontier)
    resume_path = args.resume
    if resume_path is None and args.output.exists():
        resume_path = args.output
    resume = SearchState.load(resume_path) if resume_path is not None else None
    config = EvolutionConfig(
        artifact_mode="certified",
        manifest_sha256=frontier.manifest_sha256,
        workers=args.workers,
        seed=args.seed,
        engine=frontier.engine,
    )
    state = evolve(
        codec=GenomeCodec.default(),
        initial_configs=seed_configs(),
        league=frontier.opponents,
        weights=strength_weights(report),
        config=config,
        output=args.output,
        resume=resume,
        certification=certify_frontier(frontier),
    )
    write_finalists(state, args.finalists)


if __name__ == "__main__":
    main()
