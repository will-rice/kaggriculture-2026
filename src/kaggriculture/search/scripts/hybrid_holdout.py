"""Run the exact untouched paired promotion gate for certified finalists."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TypeVar, cast

from pydantic import TypeAdapter, ValidationError

from kaggriculture.hybrid.config import to_runtime
from kaggriculture.search.arena import HybridOpponent
from kaggriculture.search.evolution import (
    FinalistArtifact,
    SnapshotLeague,
    SnapshotSource,
)
from kaggriculture.search.fitness import StrengthWeights, strength_weights
from kaggriculture.search.frontier import (
    FrontierReport,
    VerifiedFrontier,
    verify_frontier,
)
from kaggriculture.search.promotion import (
    DETERMINISM_SEEDS,
    PROMOTION_SEEDS,
    check_determinism,
    evaluate_promotion,
)
from kaggriculture.search.scripts.frontier_round_robin import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_MANIFEST,
)
from kaggriculture.search.scripts.hybrid_search import _validate_frontier_report

_CLAIM_SCHEMA_VERSION = 1
_OUTPUT_SCHEMA_VERSION = 2
_T = TypeVar("_T")


def parser() -> argparse.ArgumentParser:
    """Build the fixed-seed CPU holdout command line."""
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--finalists", type=Path, required=True)
    arguments.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    arguments.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    arguments.add_argument("--frontier-report", type=Path, required=True)
    arguments.add_argument("--workers", type=int, default=16)
    arguments.add_argument("--output", type=Path, required=True)
    return arguments


def main() -> None:
    """Preflight all evidence, then consume each finalist holdout at most once."""
    args = parser().parse_args()
    if not 1 <= args.workers <= 16:
        raise SystemExit("--workers must be between 1 and 16 while Toad is running")
    os.nice(10)
    try:
        frontier = verify_frontier(args.manifest, args.artifact_root)
        report = TypeAdapter(FrontierReport).validate_json(
            args.frontier_report.read_text()
        )
        finalists = FinalistArtifact.load(args.finalists)
        snapshot, weights = _bind_frontier(finalists, frontier, report)
        league = snapshot.opponents
        incumbent = league["boatlee_v14_current"]
        public_frontier = league[report.frontier_name]
        finalist_rows = tuple(
            {
                "index": index,
                "genome": list(finalist.genome),
                "config": finalist.config.model_dump(mode="json"),
            }
            for index, finalist in enumerate(finalists.finalists)
        )
        run_identity = {
            "engine": frontier.engine,
            "manifest_sha256": frontier.manifest_sha256,
            "finalist_integrity_digest": finalists.integrity_digest,
            "seeds": list(PROMOTION_SEEDS),
            "seats": [0, 1],
            "incumbent": "boatlee_v14_current",
            "boatlee": "boatlee_v14_current",
            "frontier": report.frontier_name,
            "league_sha256": {
                artifact.name: artifact.sha256 for artifact in frontier.artifacts
            },
            "strength_weights": weights.as_dict(),
        }

        def deterministic(row: Mapping[str, object]) -> bool:
            index = _row_index(row)
            return check_determinism(
                HybridOpponent(to_runtime(finalists.finalists[index].config)),
                league,
                args.workers,
                seeds=DETERMINISM_SEEDS,
            )

        def evaluate(row: Mapping[str, object]) -> Mapping[str, object]:
            index = _row_index(row)
            finalist = finalists.finalists[index]
            verdict = evaluate_promotion(
                HybridOpponent(to_runtime(finalist.config)),
                incumbent,
                incumbent,
                public_frontier,
                league,
                PROMOTION_SEEDS,
                args.workers,
                weights=weights,
                deterministic=True,
            )
            return {
                **dict(row),
                "passed": verdict.passed,
                "reasons": list(verdict.reasons),
                "verdict": asdict(verdict),
            }

        _execute_single_use(
            output=args.output,
            run_identity=run_identity,
            finalists=finalist_rows,
            protected_paths=(
                args.finalists,
                args.manifest,
                args.frontier_report,
                *(Path(path) for path in frontier.opponents.values()),
            ),
            snapshot_root=Path(snapshot.root),
            deterministic=deterministic,
            evaluate=evaluate,
        )
    except (
        KeyError,
        OSError,
        RuntimeError,
        ValidationError,
        TypeError,
        ValueError,
    ) as error:
        raise SystemExit(f"invalid holdout provenance: {error}") from error


def _bind_frontier(
    finalists: FinalistArtifact,
    frontier: VerifiedFrontier,
    report: FrontierReport,
) -> tuple[SnapshotLeague, StrengthWeights]:
    """Bind authoritative Task 8 state to freshly verified external evidence."""
    _validate_frontier_report(report, frontier)
    identity = finalists.identity
    if (
        identity.manifest_sha256 != frontier.manifest_sha256
        or identity.engine != frontier.engine
    ):
        raise ValueError("finalist manifest or engine differs from verification")
    expected_identity = tuple(
        (artifact.name, "agent_source", artifact.sha256)
        for artifact in frontier.artifacts
    )
    if identity.league_identity != expected_identity:
        raise ValueError("finalist league identity differs from verified sources")
    expected_weights = strength_weights(report)
    actual_weights = StrengthWeights(dict(identity.strength_weights))
    if actual_weights != expected_weights:
        raise ValueError("finalist strength weights differ from frontier report")
    sources = tuple(SnapshotSource(*row) for row in identity.league_snapshots)
    roots = {str(Path(source.snapshot_path).parent) for source in sources}
    if len(roots) != 1:
        raise ValueError("finalist snapshot provenance has multiple roots")
    snapshot = SnapshotLeague(roots.pop(), sources)
    # Loading FinalistArtifact already hash-verified every snapshot. This fresh
    # equality check binds those immutable copies to the newly verified sources.
    if tuple(source.sha256 for source in snapshot.sources) != tuple(
        artifact.sha256 for artifact in frontier.artifacts
    ):
        raise ValueError("finalist snapshots differ from verified sources")
    return snapshot, actual_weights


def _execute_single_use(
    *,
    output: Path,
    run_identity: Mapping[str, object],
    finalists: Sequence[Mapping[str, object]],
    protected_paths: Sequence[Path],
    snapshot_root: Path,
    deterministic: Callable[[Mapping[str, object]], bool],
    evaluate: Callable[[Mapping[str, object]], Mapping[str, object]],
) -> dict[str, object]:
    """Durably claim and finish each exact finalist/seed identity once."""
    rows = tuple(dict(row) for row in finalists)
    if not rows:
        raise ValueError("holdout requires at least one finalist")
    identity = _json_copy(dict(run_identity))
    claims_dir = output.parent / f".{output.name}.claims"
    claims = tuple(_claim_path(claims_dir, identity, row) for row in rows)
    if output.exists() and not output.is_file():
        raise ValueError("holdout output must be a regular file")
    if claims_dir.exists() and not claims_dir.is_dir():
        raise ValueError("holdout claim path must be a directory")
    if any(path.exists() and not path.is_file() for path in claims):
        raise ValueError("holdout claims must be regular files")
    _preflight_paths(
        write_paths=(output, claims_dir, *claims),
        protected_paths=protected_paths,
        snapshot_root=snapshot_root,
    )
    existing_output, existing_claims = _preflight_existing(
        output, claims_dir, claims, identity, rows
    )
    if existing_output is not None and existing_output["status"] == "complete":
        return existing_output
    pending_rows = (
        row for row, claim in zip(rows, existing_claims, strict=True) if claim is None
    )
    if not all(deterministic(row) for row in pending_rows):
        raise RuntimeError(
            "candidate action traces are nondeterministic; holdout was not claimed"
        )

    results: list[dict[str, object]] = []
    for path, row, existing in zip(claims, rows, existing_claims, strict=True):
        claim_identity = _claim_identity(identity, row)
        if existing is not None:
            result = _claim_result(existing)
        else:
            acquired = _exclusive_claim(path, claim_identity)
            if acquired is not None:
                result = acquired
            else:
                result = _json_copy(dict(evaluate(row)))
                _complete_claim(path, claim_identity, result)
        results.append(result)
        status = "complete" if len(results) == len(rows) else "partial"
        payload = _output_payload(identity, rows, results, status)
        _write_atomic(output, _canonical_json(payload))
    return _output_payload(identity, rows, results, "complete")


def _preflight_existing(
    output: Path,
    claims_dir: Path,
    claims: Sequence[Path],
    identity: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object] | None, tuple[dict[str, object] | None, ...]]:
    """Validate all existing aggregate and per-finalist evidence before games."""
    if claims_dir.exists():
        unexpected = set(claims_dir.glob("*.json")) - set(claims)
        if unexpected:
            raise ValueError("claim directory contains conflicting holdout identities")
    aggregate = _load_existing_output(output, identity, rows)
    existing = tuple(
        _load_claim(path, _claim_identity(identity, row))
        for path, row in zip(claims, rows, strict=True)
    )
    if any(
        claim is not None and claim["status"] == "in_progress" for claim in existing
    ):
        raise RuntimeError("a prior holdout claim is in progress; replay is forbidden")
    if aggregate is not None:
        _validate_output_claims(aggregate, existing)
    return aggregate, existing


def _claim_identity(
    run_identity: Mapping[str, object], finalist: Mapping[str, object]
) -> dict[str, object]:
    return {
        "run_identity": _json_copy(dict(run_identity)),
        "finalist": _json_copy(dict(finalist)),
    }


def _claim_path(
    claims_dir: Path,
    run_identity: Mapping[str, object],
    finalist: Mapping[str, object],
) -> Path:
    digest = hashlib.sha256(
        _canonical_json(_claim_identity(run_identity, finalist)).encode()
    ).hexdigest()
    return claims_dir / f"{digest}.json"


def _exclusive_claim(
    path: Path, identity: Mapping[str, object]
) -> dict[str, object] | None:
    """Create and fsync an exclusive in-progress claim before any holdout game."""
    _mkdir_durable(path.parent)
    source = _canonical_json(
        {
            "schema_version": _CLAIM_SCHEMA_VERSION,
            "status": "in_progress",
            "identity": identity,
        }
    )
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raced = _load_claim(path, identity)
        if raced is not None and raced["status"] == "complete":
            return _claim_result(raced)
        raise RuntimeError(
            "holdout claim already exists; replay is forbidden"
        ) from None
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(source)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)
    return None


def _complete_claim(
    path: Path, identity: Mapping[str, object], result: Mapping[str, object]
) -> None:
    payload = {
        "schema_version": _CLAIM_SCHEMA_VERSION,
        "status": "complete",
        "identity": identity,
        "result": _json_copy(dict(result)),
    }
    _write_atomic(path, _canonical_json(payload))


def _load_claim(path: Path, identity: Mapping[str, object]) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = _strict_object(path)
    allowed = {"schema_version", "status", "identity", "result"}
    if set(payload) not in ({"schema_version", "status", "identity"}, allowed):
        raise ValueError(f"malformed holdout claim: {path}")
    if (
        payload["schema_version"] != _CLAIM_SCHEMA_VERSION
        or payload["identity"] != identity
    ):
        raise ValueError(f"conflicting holdout claim identity: {path}")
    if payload["status"] not in {"in_progress", "complete"}:
        raise ValueError(f"malformed holdout claim status: {path}")
    if payload["status"] == "complete" and "result" not in payload:
        raise ValueError(f"completed holdout claim has no result: {path}")
    if payload["status"] == "in_progress" and "result" in payload:
        raise ValueError(f"in-progress holdout claim has a result: {path}")
    return payload


def _claim_result(claim: Mapping[str, object]) -> dict[str, object]:
    result = claim.get("result")
    if type(result) is not dict:
        raise ValueError("completed holdout claim result must be an object")
    return cast(dict[str, object], _json_copy(result))


def _row_index(row: Mapping[str, object]) -> int:
    value = row.get("index")
    if type(value) is not int or value < 0:
        raise ValueError("finalist index must be a non-negative integer")
    return value


def _load_existing_output(
    output: Path,
    identity: Mapping[str, object],
    finalists: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    if not output.exists():
        return None
    payload = _strict_object(output)
    if set(payload) != {
        "schema_version",
        "status",
        "run_identity",
        "finalists",
        "results",
    }:
        raise ValueError("malformed existing holdout output")
    if (
        payload["schema_version"] != _OUTPUT_SCHEMA_VERSION
        or payload["run_identity"] != identity
    ):
        raise ValueError("conflicting existing holdout output identity")
    if payload["finalists"] != list(finalists) or payload["status"] not in {
        "partial",
        "complete",
    }:
        raise ValueError("conflicting existing holdout finalist identity")
    results = payload["results"]
    if type(results) is not list or len(results) > len(finalists):
        raise ValueError("malformed existing holdout results")
    if payload["status"] == "complete" and len(results) != len(finalists):
        raise ValueError("completed holdout output is missing finalist results")
    return payload


def _validate_output_claims(
    output: Mapping[str, object], claims: Sequence[Mapping[str, object] | None]
) -> None:
    """Require aggregate evidence to be exactly backed by durable claim results."""
    raw_results = output["results"]
    if type(raw_results) is not list:
        raise ValueError("malformed existing holdout results")
    results = cast(list[object], raw_results)
    for index, result in enumerate(results):
        claim = claims[index]
        if claim is None or claim["status"] != "complete":
            raise ValueError("holdout output result has no completed durable claim")
        if result != _claim_result(claim):
            raise ValueError("holdout output conflicts with durable claim result")
    if output["status"] == "complete" and any(claim is None for claim in claims):
        raise ValueError("complete holdout output is missing durable claims")


def _output_payload(
    identity: Mapping[str, object],
    finalists: Sequence[Mapping[str, object]],
    results: Sequence[Mapping[str, object]],
    status: str,
) -> dict[str, object]:
    return {
        "schema_version": _OUTPUT_SCHEMA_VERSION,
        "status": status,
        "run_identity": _json_copy(dict(identity)),
        "finalists": [_json_copy(dict(row)) for row in finalists],
        "results": [_json_copy(dict(row)) for row in results],
    }


def _preflight_paths(
    *, write_paths: Sequence[Path], protected_paths: Sequence[Path], snapshot_root: Path
) -> None:
    """Reject canonical, symlink, hardlink, and snapshot aliases before writes."""
    root = snapshot_root.resolve(strict=True)
    protected = tuple(path for path in protected_paths if path.exists())
    for target in write_paths:
        canonical = target.resolve(strict=False)
        if canonical == root or root in canonical.parents:
            raise ValueError(f"holdout output aliases snapshot root: {target}")
        if target.is_symlink():
            raise ValueError(f"holdout output may not be a symlink: {target}")
        for source in protected:
            source_canonical = source.resolve(strict=True)
            if canonical == source_canonical:
                raise ValueError(f"holdout output aliases input artifact: {target}")
            if target.exists() and os.path.samestat(target.stat(), source.stat()):
                raise ValueError(f"holdout output hardlinks input artifact: {target}")


def _strict_object(path: Path) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key is forbidden: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant is forbidden: {value}")
        ),
        object_pairs_hook=reject_duplicates,
    )
    if type(value) is not dict:
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _json_copy(value: _T) -> _T:
    return cast(_T, json.loads(_canonical_json(value)))


def _canonical_json(payload: object) -> str:
    return (
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    )


def _write_atomic(output: Path, source: str) -> None:
    """Replace an output only after complete bytes and directory metadata persist."""
    _mkdir_durable(output.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(source)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
        _fsync_directory(output.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_durable(path: Path) -> None:
    """Create each missing directory and persist its parent entry immediately."""
    missing: list[Path] = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    path.mkdir(parents=True, exist_ok=True)
    for directory in reversed(missing):
        _fsync_directory(directory.parent)


if __name__ == "__main__":
    main()
