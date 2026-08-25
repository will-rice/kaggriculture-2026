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


def _require_finite_nonnegative(value: float, label: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise SystemExit(f"frontier report {label} must be finite and non-negative")


def _require_unit_interval(value: float, label: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise SystemExit(f"frontier report {label} must be between zero and one")


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
    _require_finite_nonnegative(report.runtime_seconds, "runtime_seconds")
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
    unique_pair_runtimes = {
        tuple(sorted((candidate, opponent))): result.runtime_seconds
        for (candidate, opponent), result in by_pair.items()
    }
    if not _close(report.runtime_seconds, sum(unique_pair_runtimes.values())):
        raise SystemExit("frontier report runtime aggregate is inconsistent")


def _validate_report_row(
    row: FrontierRow,
    names: tuple[str, ...],
    by_pair: dict[tuple[str, str], FrontierMatchup],
) -> None:
    _require_finite_nonnegative(row.runtime_seconds, f"{row.name} runtime_seconds")
    _require_unit_interval(row.field_win_points, f"{row.name} field win points")
    _require_unit_interval(
        row.worst_matchup_win_points, f"{row.name} worst matchup win points"
    )
    if not math.isfinite(row.paired_margin):
        raise SystemExit(f"frontier report {row.name} paired margin must be finite")
    for opponent, rate in row.matchups.items():
        _require_unit_interval(rate, f"{row.name} vs {opponent} win points")
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
        or not _close(
            row.runtime_seconds,
            sum(result.runtime_seconds for result in row.matchup_results),
        )
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
    for count, label in (
        (result.wins, "wins"),
        (result.draws, "draws"),
        (result.losses, "losses"),
    ):
        if count < 0:
            raise SystemExit(f"frontier report matchup {label} must be non-negative")
    _require_unit_interval(result.win_points, "matchup win points")
    _require_finite_nonnegative(result.runtime_seconds, "matchup runtime_seconds")
    if not math.isfinite(result.paired_margin):
        raise SystemExit("frontier report matchup paired margin must be finite")
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
        or not _close(reverse.runtime_seconds, result.runtime_seconds)
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
    certification = certify_frontier(frontier)
    codec = GenomeCodec.default()
    state = evolve(
        codec=codec,
        initial_configs=seed_configs(),
        league=frontier.opponents,
        weights=strength_weights(report),
        config=config,
        output=args.output,
        resume=resume,
        certification=certification,
    )
    write_finalists(state, args.finalists, certification=certification, codec=codec)


if __name__ == "__main__":
    main()
