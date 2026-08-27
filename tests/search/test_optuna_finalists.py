"""Certified Optuna finalist writer and loader contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import optuna
import pytest
from optuna.storages import RDBStorage

from kaggriculture.hybrid.config import HybridConfig
from kaggriculture.search.arena import GameKey, GameResult
from kaggriculture.search.evolution import PROMOTION_SEEDS
from kaggriculture.search.fitness import StrengthWeights, strength_weights
from kaggriculture.search.frontier import (
    FrontierArtifact,
    FrontierMatchup,
    FrontierReport,
    FrontierRow,
    VerifiedFrontier,
)
from kaggriculture.search.optuna_finalists import (
    OptunaFinalistArtifact,
    write_optuna_finalists,
)
from kaggriculture.search.optuna_protocol import (
    RUNG_1_SEEDS,
    RUNG_2_SEEDS,
    RUNG_3_SEEDS,
    RungEvidence,
    RungSpec,
    build_rung_evidence,
    rung_summary_attributes,
    write_rung_evidence_atomic,
)
from kaggriculture.search.optuna_space import SPACE_SHA256, parameters_for_config
from kaggriculture.search.optuna_state import (
    PanelIdentity,
    PrunerIdentity,
    SamplerIdentity,
    SeedBankIdentity,
    StudyIdentity,
    StudyPaths,
    evaluation_semantics,
)
from kaggriculture.search.scripts import hybrid_holdout
from kaggriculture.search.scripts.frontier_round_robin import FRONTIER_SEEDS

NAMES = (
    "frontier",
    "strong_public",
    "boatlee_v14_current",
    "economic_policy",
    "middle_public",
    "public_6",
    "public_7",
    "public_8",
    "public_9",
    "second_weak_public",
    "weak_public",
)
PANELS = (
    ("economic_policy", "weak_public", "second_weak_public"),
    (
        "economic_policy",
        "weak_public",
        "second_weak_public",
        "boatlee_v14_current",
        "frontier",
        "public_6",
    ),
    NAMES,
)
RUNGS = (
    RungSpec(1, 1, PANELS[0], RUNG_1_SEEDS),
    RungSpec(2, 4, PANELS[1], RUNG_2_SEEDS),
    RungSpec(3, 16, PANELS[2], RUNG_3_SEEDS),
)
WEIGHTS = StrengthWeights(
    {name: (4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1)[index] for index, name in enumerate(NAMES)}
)


def _canonical(payload: object) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()


def _frontier_report(
    source_sha256: dict[str, str], manifest_sha256: str = "a" * 64
) -> FrontierReport:
    """Build an internally consistent strongest-first all-pairs report."""
    games = 2 * len(FRONTIER_SEEDS)
    rows: list[FrontierRow] = []
    for index, name in enumerate(NAMES):
        results: list[FrontierMatchup] = []
        for opponent_index, opponent in enumerate(NAMES):
            if name == opponent:
                continue
            won = index < opponent_index
            results.append(
                FrontierMatchup(
                    opponent=opponent,
                    games=games,
                    win_points=1.0 if won else 0.0,
                    wins=games if won else 0,
                    draws=0,
                    losses=0 if won else games,
                    paired_margin=0.25 if won else -0.25,
                    failures=(),
                    runtime_seconds=0.0,
                )
            )
        matchups = {result.opponent: result.win_points for result in results}
        rows.append(
            FrontierRow(
                name=name,
                games=games * (len(NAMES) - 1),
                field_win_points=sum(matchups.values()) / (len(NAMES) - 1),
                worst_matchup_win_points=min(matchups.values()),
                paired_margin=sum(row.paired_margin for row in results)
                / (len(NAMES) - 1),
                failures=(),
                runtime_seconds=0.0,
                matchups=matchups,
                matchup_results=tuple(results),
            )
        )
    return FrontierReport(
        engine="test-engine",
        seeds=FRONTIER_SEEDS,
        frontier_name=NAMES[0],
        failures=(),
        runtime_seconds=0.0,
        rows=tuple(rows),
        manifest_sha256=manifest_sha256,
        source_sha256=source_sha256,
    )


def _identity(root: Path) -> StudyIdentity:
    root.mkdir()
    sources = root.with_name(root.name + ".sources")
    snapshots = root.with_name(root.name + ".league")
    sources.mkdir()
    snapshots.mkdir()
    source_rows: list[tuple[str, str]] = []
    snapshot_rows: list[tuple[str, str, str, str]] = []
    for name in NAMES:
        content = f"# {name}\n".encode()
        digest = hashlib.sha256(content).hexdigest()
        source = sources / f"{name}.py"
        snapshot = snapshots / f"{name}-{digest}.py"
        source.write_bytes(content)
        snapshot.write_bytes(content)
        snapshot.chmod(0o444)
        source_rows.append((name, digest))
        snapshot_rows.append((name, str(source), str(snapshot), digest))
    report = _frontier_report(dict(source_rows))
    assert strength_weights(report) == WEIGHTS
    semantic_sources = evaluation_semantics()
    return StudyIdentity(
        engine="test-engine",
        manifest_sha256="a" * 64,
        frontier_report_sha256=hashlib.sha256(
            json.dumps(
                asdict(report), allow_nan=False, separators=(",", ":"), sort_keys=True
            ).encode()
        ).hexdigest(),
        source_sha256=tuple(sorted(source_rows)),
        league_snapshots=tuple(sorted(snapshot_rows)),
        space_sha256=SPACE_SHA256,
        evaluation_semantics=semantic_sources,
        evaluation_semantics_sha256=hashlib.sha256(
            json.dumps(
                semantic_sources,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        sampler=SamplerIdentity(),
        pruner=PrunerIdentity(),
        seed_banks=SeedBankIdentity(
            rung_1=RUNG_1_SEEDS,
            rung_2=RUNG_2_SEEDS,
            rung_3=RUNG_3_SEEDS,
            protected_promotion_sha256=hashlib.sha256(
                json.dumps(
                    PROMOTION_SEEDS,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
        ),
        panels=PanelIdentity(
            rung_1=PANELS[0],
            rung_2=PANELS[1],
            rung_3=PANELS[2],
        ),
        strength_weights=tuple(WEIGHTS.as_dict().items()),
        objective_schema="win-primary-margin-epsilon-v1",
        warm_start_sha256=(),
        legacy_state_sha256="e" * 64,
    )


def _promotion_context(
    path: Path,
) -> tuple[VerifiedFrontier, FrontierReport]:
    identity = OptunaFinalistArtifact.model_validate_json(
        path.read_bytes()
    ).study_identity
    source_paths = {
        name: source_path
        for name, source_path, _snapshot_path, _digest in identity.league_snapshots
    }
    source_sha256 = dict(identity.source_sha256)
    artifacts = tuple(
        FrontierArtifact(
            name=name,
            provenance="real Optuna normalization fixture",
            relative_path=Path(source_paths[name]).name,
            sha256=source_sha256[name],
        )
        for name in NAMES
    )
    frontier = VerifiedFrontier(
        engine=identity.engine,
        opponents=source_paths,
        artifacts=artifacts,
        manifest_sha256=identity.manifest_sha256,
    )
    return frontier, _frontier_report(source_sha256, identity.manifest_sha256)


def _games(spec: RungSpec, score: int, runtime: float) -> dict[GameKey, GameResult]:
    return {
        GameKey(name, seed, seat): GameResult(
            GameKey(name, seed, seat),
            100 + score,
            100 - score,
            runtime,
        )
        for name in spec.opponents
        for seed in spec.seeds
        for seat in (0, 1)
    }


def _summary(
    root: Path, row: RungEvidence, path: Path, digest: str
) -> dict[str, object]:
    return rung_summary_attributes(row, path, digest, root)


def _build_certified_study(root: Path) -> Path:
    paths = StudyPaths.from_root(root)
    identity = _identity(root)
    paths.identity.write_bytes(_canonical(identity.model_dump(mode="json"))[:-1])
    study = optuna.create_study(
        storage=RDBStorage(f"sqlite:///{paths.sqlite.absolute()}"),
        study_name=identity.study_name,
        direction="maximize",
    )
    study.set_user_attr("study_identity_sha256", identity.digest)
    config = HybridConfig.default()
    parameters = parameters_for_config(config)
    distributions: dict[str, optuna.distributions.BaseDistribution] = {}
    for name, value in parameters.items():
        if type(value) is int:
            distributions[name] = optuna.distributions.IntDistribution(value, value)
        elif type(value) is float:
            distributions[name] = optuna.distributions.FloatDistribution(value, value)
        else:
            distributions[name] = optuna.distributions.CategoricalDistribution((value,))
    for number in range(512):
        complete = number < 4
        last = None
        intermediate: dict[int, float] = {}
        attrs: dict[str, object] = {}
        for spec in RUNGS if complete else RUNGS[:1]:
            score = 4 - number if complete else 1
            runtime = 0.001 * (number + 1)
            last = build_rung_evidence(
                number, config, spec, _games(spec, score, runtime), WEIGHTS
            )
            path, digest = write_rung_evidence_atomic(paths.root, last)
            attrs = _summary(paths.root, last, path, digest)
            assert last.objective is not None
            intermediate[spec.resource_step] = last.objective
        assert last is not None and last.objective is not None
        study.add_trial(
            optuna.trial.create_trial(
                state=(
                    optuna.trial.TrialState.COMPLETE
                    if complete
                    else optuna.trial.TrialState.PRUNED
                ),
                value=last.objective,
                params=parameters,
                distributions=distributions,
                user_attrs=attrs,
                intermediate_values=intermediate,
            )
        )
    artifact = write_optuna_finalists(study, identity, paths, count=4)
    assert artifact.finalists
    return paths.finalists


@pytest.fixture(scope="module")
def certified_finalists(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build one real terminal certification shared by the tamper matrix."""
    return _build_certified_study(tmp_path_factory.mktemp("optuna-finalists") / "study")


def test_writer_selects_only_clean_rung_three_trials_in_canonical_order(
    certified_finalists: Path,
) -> None:
    """Only clean rung-three rows survive the exact canonical ordering."""
    artifact = OptunaFinalistArtifact.load(certified_finalists)
    assert [row.trial_number for row in artifact.finalists] == [0, 1, 2, 3]
    assert all(row.rung == 3 and not row.failures for row in artifact.finalists)
    assert artifact.promotion_seeds_used is False
    assert sum(artifact.trial_counts.values()) == 512


def test_holdout_normalizes_a_real_optuna_certification(
    certified_finalists: Path,
) -> None:
    """The actual Optuna loader supplies configs, snapshots, and measured weights."""
    frontier, report = _promotion_context(certified_finalists)

    promotion = hybrid_holdout.load_promotion_input(
        certified_finalists, frontier, report
    )

    assert len(promotion.configs) == 4
    assert all(config == HybridConfig.default() for config in promotion.configs)
    assert set(promotion.league) == set(NAMES)
    assert all(
        Path(source).parent == promotion.snapshot_root
        for source in promotion.league.values()
    )
    assert promotion.weights == WEIGHTS == strength_weights(report)
    assert tuple(row["config"] for row in promotion.finalist_rows) == tuple(
        config.model_dump(mode="json") for config in promotion.configs
    )


def test_genuine_optuna_tamper_fails_before_claim_or_holdout(
    certified_finalists: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A coherent artifact mutation fails in the real loader before claim creation."""
    copied = tmp_path / "study"
    shutil.copytree(certified_finalists.parent, copied)
    finalists = copied / "finalists.json"
    _rewrite_artifact(finalists, lambda payload: payload["finalists"].pop())
    frontier, report = _promotion_context(certified_finalists)
    monkeypatch.setattr(hybrid_holdout, "verify_frontier", lambda *_: frontier)
    monkeypatch.setattr(
        hybrid_holdout,
        "_load_frontier_report",
        lambda _path: (report, b"real-report"),
    )
    monkeypatch.setattr(
        hybrid_holdout,
        "_execute_single_use",
        lambda **_kwargs: pytest.fail("claim boundary reached after tamper"),
    )
    monkeypatch.setattr(
        hybrid_holdout,
        "check_determinism",
        lambda *_args, **_kwargs: pytest.fail("determinism game reached after tamper"),
    )
    monkeypatch.setattr(
        hybrid_holdout,
        "evaluate_promotion",
        lambda *_args, **_kwargs: pytest.fail("holdout game reached after tamper"),
    )
    args = argparse.Namespace(
        finalists=finalists,
        manifest=tmp_path / "manifest.json",
        artifact_root=tmp_path / "artifacts",
        frontier_report=tmp_path / "frontier.json",
        workers=1,
        output=tmp_path / "promotion.json",
    )

    with pytest.raises(SystemExit, match="invalid holdout provenance"):
        hybrid_holdout.run(args)

    assert not args.output.exists()


def _rewrite_artifact(path: Path, mutate: Callable[[dict[str, Any]], object]) -> None:
    payload = json.loads(path.read_text())
    mutate(payload)
    payload["integrity_sha256"] = hashlib.sha256(
        _canonical(
            {key: value for key, value in payload.items() if key != "integrity_sha256"}
        )
    ).hexdigest()
    path.write_bytes(_canonical(payload))


@pytest.mark.parametrize(
    "tamper",
    (
        "sqlite_digest",
        "identity_digest",
        "config",
        "objective",
        "remove_finalist",
        "duplicate_finalist",
        "reorder_finalists",
        "rung_evidence_digest",
        "source_snapshot",
        "trial_state",
        "seed_count",
    ),
)
def test_loader_rejects_every_certification_tamper(  # noqa: C901
    certified_finalists: Path, tmp_path: Path, tamper: str
) -> None:
    """Every artifact, SQLite, evidence, and snapshot mutation fails closed."""
    copied = tmp_path / "study"
    shutil.copytree(certified_finalists.parent, copied)
    path = copied / "finalists.json"
    restore: tuple[Path, bytes, int] | None = None
    if tamper == "sqlite_digest":
        _rewrite_artifact(
            path,
            lambda payload: payload.__setitem__("sqlite_sha256", "f" * 64),
        )
    elif tamper == "identity_digest":
        _rewrite_artifact(
            path, lambda payload: payload.__setitem__("study_identity_sha256", "f" * 64)
        )
    elif tamper == "config":
        _rewrite_artifact(
            path,
            lambda payload: payload["finalists"][0]["config"].__setitem__(
                "liquidation_start_day", 20
            ),
        )
    elif tamper == "objective":
        _rewrite_artifact(
            path,
            lambda payload: payload["finalists"][0].__setitem__("objective", 0.0),
        )
    elif tamper == "remove_finalist":
        _rewrite_artifact(path, lambda payload: payload["finalists"].pop())
    elif tamper == "duplicate_finalist":
        _rewrite_artifact(
            path,
            lambda payload: payload["finalists"].__setitem__(
                1, payload["finalists"][0]
            ),
        )
    elif tamper == "reorder_finalists":
        _rewrite_artifact(path, lambda payload: payload["finalists"].reverse())
    elif tamper == "rung_evidence_digest":
        _rewrite_artifact(
            path,
            lambda payload: payload["finalists"][0].__setitem__(
                "evidence_sha256", "f" * 64
            ),
        )
    elif tamper == "source_snapshot":
        payload = json.loads(path.read_text())
        snapshot = Path(payload["study_identity"]["league_snapshots"][0][2])
        restore = (snapshot, snapshot.read_bytes(), snapshot.stat().st_mode)
        snapshot.chmod(0o644)
        snapshot.write_text("tampered\n")
    elif tamper == "trial_state":
        sqlite_path = copied / "study.sqlite3"
        connection = sqlite3.connect(sqlite_path)
        try:
            connection.execute("UPDATE trials SET state = 'FAIL' WHERE number = 0")
            connection.commit()
        finally:
            connection.close()
        digest = hashlib.sha256(sqlite_path.read_bytes()).hexdigest()
        _rewrite_artifact(
            path, lambda payload: payload.__setitem__("sqlite_sha256", digest)
        )
    elif tamper == "seed_count":
        payload = json.loads(path.read_text())
        evidence = copied / payload["finalists"][0]["evidence_path"]
        evidence_payload = json.loads(evidence.read_text())
        evidence_payload["seeds"].pop()
        evidence.write_bytes(_canonical(evidence_payload))
        digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
        _rewrite_artifact(
            path,
            lambda artifact: artifact["finalists"][0].__setitem__(
                "evidence_sha256", digest
            ),
        )
    try:
        with pytest.raises(ValueError):
            OptunaFinalistArtifact.load(path)
    finally:
        if restore is not None:
            snapshot, source, mode = restore
            snapshot.write_bytes(source)
            snapshot.chmod(mode)


def test_writer_refuses_less_than_the_full_terminal_budget(tmp_path: Path) -> None:
    """Pilot and diagnostic boundaries can never emit a finalist artifact."""
    paths = StudyPaths.from_root(tmp_path / "study")
    identity = _identity(paths.root)
    paths.identity.write_bytes(_canonical(identity.model_dump(mode="json"))[:-1])
    study = optuna.create_study(
        storage=RDBStorage(f"sqlite:///{paths.sqlite.absolute()}"),
        study_name=identity.study_name,
        direction="maximize",
    )
    study.set_user_attr("study_identity_sha256", identity.digest)
    with pytest.raises(ValueError, match="512 terminal"):
        write_optuna_finalists(study, identity, paths)
    assert not paths.finalists.exists()


def test_writer_refuses_a_noncanonical_finalist_count(tmp_path: Path) -> None:
    """Certification always means the canonical top four, never a caller subset."""
    paths = StudyPaths.from_root(tmp_path / "study")
    identity = _identity(paths.root)
    study = optuna.create_study()

    with pytest.raises(ValueError, match="exactly 4"):
        write_optuna_finalists(study, identity, paths, count=3)

    assert not paths.finalists.exists()
