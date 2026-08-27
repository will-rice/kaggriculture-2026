"""Run or read-only validate the certified sequential hybrid Optuna study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import optuna
from pydantic import TypeAdapter, ValidationError

from kaggriculture.search.arena_pool import PersistentArena
from kaggriculture.search.evolution import (
    DEVELOPMENT_SEEDS,
    PROMOTION_SEEDS,
    SCREENING_SEEDS,
    SnapshotLeague,
    snapshot_frontier,
)
from kaggriculture.search.fitness import StrengthWeights, strength_weights
from kaggriculture.search.frontier import (
    FrontierReport,
    VerifiedFrontier,
    verify_frontier,
)
from kaggriculture.search.optuna_protocol import (
    RUNG_3_SEEDS,
    PanelSet,
    derive_panels,
    rung_specs,
)
from kaggriculture.search.optuna_search import (
    SearchInputs,
    SearchRunConfig,
    pilot_gate,
    run_search,
)
from kaggriculture.search.optuna_seeds import warm_start_configs
from kaggriculture.search.optuna_state import (
    StudyIdentity,
    StudyPaths,
    build_identity,
    open_study,
    reconcile_running_trials,
    terminal_counts,
    validate_study_evidence,
    validate_study_paths,
)
from kaggriculture.search.promotion import DETERMINISM_SEEDS
from kaggriculture.search.scripts.frontier_round_robin import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_MANIFEST,
    FRONTIER_SEEDS,
)
from kaggriculture.search.scripts.hybrid_search import _validate_frontier_report


def parser() -> argparse.ArgumentParser:
    """Build the side-effect-free production command line."""
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    arguments.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    arguments.add_argument("--frontier-report", type=Path, required=True)
    arguments.add_argument(
        "--legacy-state", type=Path, default=Path("run/hybrid/search-state.json")
    )
    arguments.add_argument("--root", type=Path, default=Path("run/hybrid/optuna-v2"))
    arguments.add_argument("--workers", type=int, choices=range(1, 33), default=32)
    arguments.add_argument("--stop-after", type=int, choices=range(1, 513), default=512)
    arguments.add_argument("--validate-only", action="store_true")
    arguments.add_argument(
        "--wandb", action=argparse.BooleanOptionalAction, default=True
    )
    arguments.add_argument("--wandb-entity", default="will-rice")
    arguments.add_argument("--wandb-project", default="kaggriculture-2026")
    return arguments


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and range-check CLI arguments without touching files or services."""
    return parser().parse_args(argv)


def run(args: argparse.Namespace) -> None:
    """Preflight every semantic input before validation, telemetry, or workers."""
    try:
        warm_starts = warm_start_configs(args.legacy_state)
    except (OSError, ValueError) as error:
        raise SystemExit(f"invalid legacy state: {error}") from error
    try:
        frontier = verify_frontier(args.manifest, args.artifact_root)
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"invalid frontier: {error}") from error
    report = _load_frontier_report(args.frontier_report)
    _validate_frontier_report(report, frontier)
    panels, weights = _protocol_inputs(report)
    paths = StudyPaths.from_root(args.root)
    _preflight_output_ownership(paths, frontier, args.legacy_state)
    if args.validate_only and (
        not paths.sqlite.is_file() or not paths.identity.is_file()
    ):
        raise SystemExit("validate-only requires an existing identity and SQLite study")
    try:
        snapshot = snapshot_frontier(
            frontier,
            paths.root,
            require_existing=args.validate_only,
        )
        identity = build_identity(
            frontier,
            snapshot,
            report,
            panels,
            weights,
            warm_starts,
            hashlib.sha256(args.legacy_state.read_bytes()).hexdigest(),
        )
        preflight = validate_study_paths(
            paths,
            snapshot,
            args.legacy_state,
            identity,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"invalid study identity or paths: {error}") from error

    if args.validate_only:
        with _readonly_study(paths, identity, snapshot, args.legacy_state) as study:
            validate_study_evidence(study, paths, identity)
            _print_preflight(identity, study, args)
            counts = terminal_counts(study)
            if sum(counts.values()) == 32:
                print(
                    json.dumps(
                        pilot_gate(study, paths).model_dump(mode="json"),
                        allow_nan=False,
                        indent=2,
                        sort_keys=True,
                    )
                )
        return

    try:
        study = open_study(paths, identity, preflight)
        reconcile_running_trials(study)
        validate_study_evidence(study, paths, identity)
    except (OSError, ValueError) as error:
        raise SystemExit(f"invalid existing study: {error}") from error
    _print_preflight(identity, study, args)
    inputs = SearchInputs(
        study=study,
        paths=paths,
        identity=identity,
        league=snapshot.opponents,
        weights=weights,
        rungs=rung_specs(panels),
        warm_starts=warm_starts,
    )
    summary = run_search(
        inputs,
        SearchRunConfig(workers=args.workers, stop_after=args.stop_after),
        callbacks=(),
        arena_factory=PersistentArena,
    )
    print(
        json.dumps(
            summary.model_dump(mode="json"),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )


def _load_frontier_report(path: Path) -> FrontierReport:
    """Read one strict report with a CLI-specific failure boundary."""
    try:
        return TypeAdapter(FrontierReport).validate_json(path.read_bytes())
    except (OSError, ValidationError, ValueError) as error:
        raise SystemExit(f"invalid frontier report: {error}") from error


def _protocol_inputs(report: FrontierReport) -> tuple[PanelSet, StrengthWeights]:
    """Validate protected seed disjointness before deriving protocol facts."""
    protected = (
        FRONTIER_SEEDS,
        SCREENING_SEEDS,
        DEVELOPMENT_SEEDS,
        DETERMINISM_SEEDS,
        PROMOTION_SEEDS,
    )
    if any(set(RUNG_3_SEEDS) & set(bank) for bank in protected):
        raise SystemExit("Optuna search seeds overlap a protected seed bank")
    try:
        panels = derive_panels(report)
        weights = strength_weights(report)
        rung_specs(panels)
    except ValueError as error:
        raise SystemExit(f"invalid Optuna protocol inputs: {error}") from error
    return panels, weights


def _preflight_output_ownership(
    paths: StudyPaths, frontier: VerifiedFrontier, legacy_state: Path
) -> None:
    """Reject obvious output aliases before snapshot creation can write anything."""
    _reject_symlink_components(paths.root)
    _reject_symlink_components(paths.root.with_name(paths.root.name + ".league"))
    root = _resolved(paths.root)
    protected = (
        _resolved(legacy_state),
        *(_resolved(Path(source)) for source in frontier.opponents.values()),
    )
    snapshot_root = _resolved(paths.root.with_name(paths.root.name + ".league"))
    for candidate in (root, snapshot_root):
        for item in protected:
            if _same_or_below(candidate, item) or _same_or_below(item, candidate):
                raise SystemExit(
                    "study output aliases protected source or legacy state"
                )


@contextmanager
def _readonly_study(
    paths: StudyPaths,
    identity: StudyIdentity,
    snapshot: SnapshotLeague,
    legacy_state: Path,
) -> Iterator[optuna.Study]:
    """Validate through a temporary SQLite backup without mutating run bytes."""
    with tempfile.TemporaryDirectory(prefix="kaggriculture-optuna-validate-") as name:
        temporary_paths = StudyPaths.from_root(Path(name) / "study")
        temporary_paths.root.mkdir()
        temporary_paths.identity.write_bytes(paths.identity.read_bytes())
        source = sqlite3.connect(f"file:{paths.sqlite.absolute()}?mode=ro", uri=True)
        destination = sqlite3.connect(temporary_paths.sqlite)
        try:
            source.backup(destination)
        finally:
            destination.close()
            source.close()
        preflight = validate_study_paths(
            temporary_paths,
            snapshot,
            legacy_state,
            identity,
        )
        study = open_study(temporary_paths, identity, preflight)
        yield study


def _print_preflight(
    identity: StudyIdentity, study: optuna.Study, args: argparse.Namespace
) -> None:
    counts = terminal_counts(study)
    print(
        json.dumps(
            {
                "device": "cpu",
                "gpu": False,
                "stop_after": args.stop_after,
                "study_identity_sha256": identity.digest,
                "terminal_counts": counts,
                "wandb_requested": bool(args.wandb),
                "workers": args.workers,
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )


def _resolved(path: Path) -> Path:
    return Path(os.path.realpath(path.absolute()))


def _reject_symlink_components(path: Path) -> None:
    lexical = Path(os.path.abspath(path))  # noqa: PTH100
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        if current.is_symlink():
            raise SystemExit(f"study output cannot traverse symlink: {current}")
        if not current.exists():
            return


def _same_or_below(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def main() -> None:
    """Execute the parsed production command."""
    run(parse_args())


if __name__ == "__main__":
    main()
