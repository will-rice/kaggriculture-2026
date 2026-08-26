"""Run the exact untouched paired promotion gate for certified finalists."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

from pydantic import TypeAdapter, ValidationError

from kaggriculture.hybrid.config import to_runtime
from kaggriculture.search import evolution
from kaggriculture.search.arena import HybridOpponent
from kaggriculture.search.evolution import (
    DEVELOPMENT_SEEDS,
    SCREENING_SEEDS,
    CandidateEvaluation,
    SearchIdentity,
    SnapshotLeague,
    SnapshotSource,
    certify_frontier,
    genome_schema_sha256,
)
from kaggriculture.search.fitness import StrengthWeights, strength_weights
from kaggriculture.search.frontier import (
    FrontierReport,
    VerifiedFrontier,
    verify_frontier,
)
from kaggriculture.search.genome import GenomeCodec
from kaggriculture.search.promotion import PROMOTION_SEEDS, evaluate_promotion
from kaggriculture.search.scripts.frontier_round_robin import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_MANIFEST,
    FRONTIER_SEEDS,
)
from kaggriculture.search.scripts.hybrid_search import _validate_frontier_report


@dataclass(frozen=True)
class FinalistArtifact:
    """Strict certified Task 8 finalist data accepted by the holdout gate."""

    identity: SearchIdentity
    generation: int
    finalists: tuple[CandidateEvaluation, ...]
    integrity_digest: str

    @classmethod
    def from_json(cls, source: str) -> Self:
        """Parse complete provenance without collapsing duplicate rows."""
        value = json.loads(source, parse_constant=evolution._reject_json_constant)
        payload = evolution._dictionary(
            value,
            {
                "schema_version",
                "identity",
                "generation",
                "finalists",
                "integrity_digest",
            },
            "finalist artifact",
        )
        if payload["schema_version"] != evolution.STATE_SCHEMA_VERSION:
            raise ValueError(
                "finalist artifact schema_version differs from search state"
            )
        identity_payload = evolution._dictionary(
            payload["identity"],
            {
                "certified",
                "manifest_sha256",
                "engine",
                "frontier_seeds",
                "screening_seeds",
                "development_seeds",
                "promotion_seeds",
                "genome_schema_sha256",
                "league_identity",
                "league_snapshots",
                "strength_weights",
                "initial_genomes",
                "evolution_config",
            },
            "search identity",
        )
        _reject_duplicate_rows(identity_payload, "league_identity", "league identity")
        _reject_duplicate_rows(identity_payload, "league_snapshots", "league snapshot")
        identity = evolution._identity_from_payload(identity_payload)
        generation = evolution._integer(
            payload["generation"], "finalist generation", minimum=1
        )
        candidates = tuple(
            evolution._candidate_from_payload(item)
            for item in evolution._list(payload["finalists"], "finalists")
        )
        artifact = cls(
            identity=identity,
            generation=generation,
            finalists=candidates,
            integrity_digest=evolution._sha256_string(
                payload["integrity_digest"], "finalist integrity digest"
            ),
        )
        artifact._validate()
        return artifact

    def _validate(self) -> None:
        """Require a complete certified search result and exact fixed seed sets."""
        identity = self.identity
        config = identity.evolution_config
        if not identity.certified or config.artifact_mode != "certified":
            raise ValueError("holdout requires a certified finalist artifact")
        if self.generation != config.generations:
            raise ValueError("finalist generation differs from configured completion")
        if identity.frontier_seeds != FRONTIER_SEEDS:
            raise ValueError("finalist frontier seeds differ from the fixed set")
        if identity.screening_seeds != SCREENING_SEEDS:
            raise ValueError("finalist screening seeds differ from the fixed set")
        if identity.development_seeds != DEVELOPMENT_SEEDS:
            raise ValueError("finalist development seeds differ from the fixed set")
        if identity.promotion_seeds != PROMOTION_SEEDS:
            raise ValueError("finalist promotion seeds differ from the untouched set")
        codec = GenomeCodec.default()
        if identity.genome_schema_sha256 != genome_schema_sha256(codec):
            raise ValueError("finalist genome schema differs from the current codec")
        self._validate_candidates(codec)
        self._validate_league_names()

    def _validate_candidates(self, codec: GenomeCodec) -> None:
        """Validate finalist uniqueness, eligibility, and genome/config identity."""
        identity = self.identity
        config = identity.evolution_config
        if not self.finalists:
            raise ValueError("finalist artifact contains no candidates")
        genomes = tuple(candidate.genome for candidate in self.finalists)
        if len(genomes) != len(set(genomes)):
            raise ValueError("duplicate finalist provenance is not allowed")
        if len(self.finalists) > config.elites:
            raise ValueError("finalist count exceeds the configured elite count")
        weights = StrengthWeights(dict(identity.strength_weights))
        for candidate in self.finalists:
            try:
                evolution._validate_resume_candidate(
                    candidate,
                    codec,
                    weights,
                    stage="development",
                    seeds=DEVELOPMENT_SEEDS,
                )
            except ValueError as error:
                if "genome" in str(error):
                    raise ValueError(
                        "finalist genome/config provenance differs"
                    ) from error
                raise
            if candidate.failures or candidate.fitness is None:
                raise ValueError(
                    "failed or unevaluated candidates cannot enter holdout"
                )
            if not candidate.fitness.eligible:
                raise ValueError("ineligible candidates cannot enter holdout")

    def _validate_league_names(self) -> None:
        """Reject missing, reordered, or duplicate league provenance names."""
        identity = self.identity
        names = tuple(row[0] for row in identity.league_identity)
        snapshot_names = tuple(row[0] for row in identity.league_snapshots)
        weight_names = tuple(name for name, _ in identity.strength_weights)
        if (
            len(names) != len(set(names))
            or len(snapshot_names) != len(set(snapshot_names))
            or names != snapshot_names
            or set(names) != set(weight_names)
        ):
            raise ValueError("league provenance names must be unique and complete")

    def bind_frontier(
        self, frontier: VerifiedFrontier, report: FrontierReport
    ) -> tuple[SnapshotLeague, StrengthWeights]:
        """Reverify exact source, snapshot, report, and weight provenance."""
        _validate_frontier_report(report, frontier)
        identity = self.identity
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
        sources = tuple(
            SnapshotSource(name, source_path, snapshot_path, sha256)
            for name, source_path, snapshot_path, sha256 in identity.league_snapshots
        )
        roots = {str(Path(source.snapshot_path).parent) for source in sources}
        if len(roots) != 1:
            raise ValueError("finalist snapshot provenance has multiple roots")
        snapshot = SnapshotLeague(roots.pop(), sources)
        certify_frontier(frontier, snapshot)
        return snapshot, actual_weights


def _reject_duplicate_rows(identity: dict[str, object], key: str, label: str) -> None:
    rows = evolution._list(identity[key], f"{label} rows")
    names = [
        evolution._string_row(row, 3 if key == "league_identity" else 4, label)[0]
        for row in rows
    ]
    if len(names) != len(set(names)):
        raise ValueError(f"{label} names must be unique")


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
    """Validate all evidence, run each finalist once, and atomically report."""
    args = parser().parse_args()
    if not 1 <= args.workers <= 16:
        raise SystemExit("--workers must be between 1 and 16 while Toad is running")
    os.nice(10)
    frontier = verify_frontier(args.manifest, args.artifact_root)
    try:
        report = TypeAdapter(FrontierReport).validate_json(
            args.frontier_report.read_text()
        )
        finalists = FinalistArtifact.from_json(args.finalists.read_text())
        snapshot, weights = finalists.bind_frontier(frontier, report)
    except (OSError, ValidationError, TypeError, ValueError) as error:
        raise SystemExit(f"invalid holdout provenance: {error}") from error
    league = snapshot.opponents
    try:
        incumbent = league["boatlee_v14_current"]
        boatlee = league["boatlee_v14_current"]
        public_frontier = league[report.frontier_name]
    except KeyError as error:
        raise SystemExit(f"holdout gate opponent is missing: {error}") from error
    results: list[dict[str, object]] = []
    for index, finalist in enumerate(finalists.finalists):
        verdict = evaluate_promotion(
            HybridOpponent(to_runtime(finalist.config)),
            incumbent,
            boatlee,
            public_frontier,
            league,
            PROMOTION_SEEDS,
            args.workers,
            weights=weights,
        )
        results.append(
            {
                "index": index,
                "genome": list(finalist.genome),
                "config": finalist.config.model_dump(mode="json"),
                "passed": verdict.passed,
                "reasons": list(verdict.reasons),
                "verdict": asdict(verdict),
            }
        )
    payload = {
        "schema_version": 1,
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
        "results": results,
    }
    _write_atomic(
        args.output,
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
    )


def _write_atomic(output: Path, source: str) -> None:
    """Replace an output only after complete report bytes reach disk."""
    output.parent.mkdir(parents=True, exist_ok=True)
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
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
