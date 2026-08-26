"""Strict finalist provenance at the untouched holdout CLI boundary."""

from __future__ import annotations

import json
from typing import Any

import pytest

from kaggriculture.hybrid.config import HybridConfig
from kaggriculture.search import evolution
from kaggriculture.search.evolution import (
    DEVELOPMENT_SEEDS,
    FRONTIER_SEEDS,
    PROMOTION_SEEDS,
    SCREENING_SEEDS,
    CandidateEvaluation,
    EvolutionConfig,
    SearchIdentity,
    genome_schema_sha256,
)
from kaggriculture.search.fitness import StrengthWeights, score_fitness
from kaggriculture.search.genome import GenomeCodec
from kaggriculture.search.scripts import hybrid_holdout


def _finalist_payload() -> dict[str, Any]:
    codec = GenomeCodec.default()
    config = HybridConfig.default()
    weights = StrengthWeights({"frontier": 4, "boatlee_v14_current": 3})
    manifest_sha256 = "a" * 64
    source_rows = (
        ("frontier", "agent_source", "b" * 64),
        ("boatlee_v14_current", "agent_source", "c" * 64),
    )
    snapshot_rows = (
        ("frontier", "/sources/frontier.py", "/snapshots/frontier-b.py", "b" * 64),
        (
            "boatlee_v14_current",
            "/sources/boatlee.py",
            "/snapshots/boatlee-c.py",
            "c" * 64,
        ),
    )
    evolution_config = EvolutionConfig(
        artifact_mode="certified",
        manifest_sha256=manifest_sha256,
        population=4,
        elites=1,
        generations=1,
        seed=73,
        workers=1,
        engine="1.32.7",
    )
    identity = SearchIdentity(
        certified=True,
        manifest_sha256=manifest_sha256,
        engine="1.32.7",
        frontier_seeds=FRONTIER_SEEDS,
        screening_seeds=SCREENING_SEEDS,
        development_seeds=DEVELOPMENT_SEEDS,
        promotion_seeds=PROMOTION_SEEDS,
        genome_schema_sha256=genome_schema_sha256(codec),
        league_identity=source_rows,
        league_snapshots=snapshot_rows,
        strength_weights=tuple(weights.values.items()),
        initial_genomes=(codec.encode(config),),
        evolution_config=evolution_config,
    )
    rates = {"frontier": 0.75, "boatlee_v14_current": 0.80}
    candidate = CandidateEvaluation(
        genome=codec.encode(config),
        config=config,
        stage="development",
        seeds=DEVELOPMENT_SEEDS,
        matchup_rates=rates,
        matchup_games=dict.fromkeys(rates, 2 * len(DEVELOPMENT_SEEDS)),
        fitness=score_fitness(rates, weights),
        paired_normalized_margin=0.2,
        failures=(),
    )
    return {
        "schema_version": evolution.STATE_SCHEMA_VERSION,
        "identity": evolution._identity_payload(identity),
        "generation": 1,
        "finalists": [evolution._candidate_payload(candidate)],
        "integrity_digest": "d" * 64,
    }


def test_finalist_loader_accepts_complete_certified_task8_artifact() -> None:
    """The holdout consumes the exact validated public artifact Task 8 writes."""
    artifact = hybrid_holdout.FinalistArtifact.from_json(
        json.dumps(_finalist_payload())
    )

    assert artifact.identity.certified is True
    assert artifact.identity.promotion_seeds == PROMOTION_SEEDS
    assert len(artifact.finalists) == 1


def test_finalist_loader_rejects_duplicate_candidate_provenance() -> None:
    """One finalist cannot masquerade as two independent promotion attempts."""
    payload = _finalist_payload()
    payload["finalists"] = [*payload["finalists"], *payload["finalists"]]

    with pytest.raises(ValueError, match="duplicate finalist"):
        hybrid_holdout.FinalistArtifact.from_json(json.dumps(payload))


def test_finalist_loader_rejects_changed_promotion_seed_provenance() -> None:
    """A finalist artifact cannot choose holdout seeds after search."""
    payload = _finalist_payload()
    identity = payload["identity"]
    assert isinstance(identity, dict)
    identity["promotion_seeds"] = list(PROMOTION_SEEDS[:-1]) + [999_999]

    with pytest.raises(ValueError, match="promotion seeds"):
        hybrid_holdout.FinalistArtifact.from_json(json.dumps(payload))


def test_finalist_loader_rejects_duplicate_weight_provenance() -> None:
    """Duplicate rows cannot be collapsed by a mapping before the gate."""
    payload = _finalist_payload()
    identity = payload["identity"]
    assert isinstance(identity, dict)
    weights = identity["strength_weights"]
    assert isinstance(weights, list)
    weights.append(weights[0])

    with pytest.raises(ValueError, match="unique"):
        hybrid_holdout.FinalistArtifact.from_json(json.dumps(payload))


def test_finalist_loader_rejects_genome_config_drift() -> None:
    """Promotion runs the config selected by the recorded finalist genome."""
    payload = _finalist_payload()
    finalists = payload["finalists"]
    assert isinstance(finalists, list)
    candidate = finalists[0]
    assert isinstance(candidate, dict)
    candidate["genome"] = [0.0] * len(candidate["genome"])

    with pytest.raises(ValueError, match="genome/config"):
        hybrid_holdout.FinalistArtifact.from_json(json.dumps(payload))


def test_holdout_cli_has_no_seed_override_and_requires_output() -> None:
    """The production command cannot substitute inspected seeds or an implicit path."""
    options = {action.dest for action in hybrid_holdout.parser()._actions}

    assert "seeds" not in options
    with pytest.raises(SystemExit):
        hybrid_holdout.parser().parse_args(
            [
                "--finalists",
                "finalists.json",
                "--frontier-report",
                "frontier.json",
            ]
        )
