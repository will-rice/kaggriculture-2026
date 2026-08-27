"""Certified Optuna finalist writer and loader contracts."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from collections.abc import Callable
from pathlib import Path

import optuna
import pytest
from optuna.storages import RDBStorage

from kaggriculture.hybrid.config import HybridConfig
from kaggriculture.search.arena import GameKey, GameResult
from kaggriculture.search.evolution import PROMOTION_SEEDS
from kaggriculture.search.fitness import StrengthWeights
from kaggriculture.search.optuna_finalists import (
    OptunaFinalistArtifact,
    write_optuna_finalists,
)
from kaggriculture.search.optuna_protocol import (
    RUNG_1_SEEDS,
    RUNG_2_SEEDS,
    RUNG_3_SEEDS,
    RungSpec,
    build_rung_evidence,
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
)

NAMES = (
    "economic_policy",
    "weak_public",
    "second_weak_public",
    "boatlee_v14_current",
    "frontier",
    "median",
    "public_6",
    "public_7",
    "public_8",
    "public_9",
    "public_10",
)
RUNGS = (
    RungSpec(1, 1, NAMES[:3], RUNG_1_SEEDS),
    RungSpec(2, 4, NAMES[:6], RUNG_2_SEEDS),
    RungSpec(3, 16, NAMES, RUNG_3_SEEDS),
)
WEIGHTS = StrengthWeights(dict.fromkeys(NAMES, 1))


def _canonical(payload: object) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()


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
    return StudyIdentity(
        engine="test-engine",
        manifest_sha256="a" * 64,
        frontier_report_sha256="b" * 64,
        source_sha256=tuple(sorted(source_rows)),
        league_snapshots=tuple(sorted(snapshot_rows)),
        space_sha256=SPACE_SHA256,
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
            rung_1=NAMES[:3],
            rung_2=NAMES[:6],
            rung_3=NAMES,
        ),
        strength_weights=tuple(WEIGHTS.as_dict().items()),
        objective_schema="win-primary-margin-epsilon-v1",
        warm_start_sha256=(),
        legacy_state_sha256="e" * 64,
    )


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
    root: Path, evidence: object, path: Path, digest: str
) -> dict[str, object]:
    row = evidence
    return {
        "config_sha256": row.config_sha256,
        "rung": row.rung,
        "resource_step": row.resource_step,
        "primary": row.primary,
        "dense_margin": row.dense_margin,
        "objective": row.objective,
        "games": len(row.games),
        "failures": len(row.failures),
        "runtime_seconds": sum(game.runtime_seconds for game in row.games),
        "evidence_path": str(path.relative_to(root)),
        "evidence_sha256": digest,
    }


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


def _rewrite_artifact(
    path: Path, mutate: Callable[[dict[str, object]], object]
) -> None:
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
        "rank",
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
    elif tamper == "rank":
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
