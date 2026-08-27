"""Durable identity, SQLite resume, and evidence contracts for Optuna search."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import optuna
import pytest
from optuna.trial import TrialState

from kaggriculture.hybrid.config import HybridConfig
from kaggriculture.search.arena import GameKey, GameResult
from kaggriculture.search.evolution import SnapshotLeague, SnapshotSource
from kaggriculture.search.fitness import StrengthWeights
from kaggriculture.search.frontier import (
    FrontierArtifact,
    FrontierReport,
    FrontierRow,
    VerifiedFrontier,
)
from kaggriculture.search.optuna_protocol import (
    PanelSet,
    RungEvidence,
    RungSpec,
    build_rung_evidence,
    write_rung_evidence_atomic,
)
from kaggriculture.search.optuna_space import SPACE_SHA256, config_sha256
from kaggriculture.search.optuna_state import (
    StudyEvidenceError,
    StudyIdentity,
    StudyIdentityError,
    StudyPaths,
    build_identity,
    close_and_hash_storage,
    open_study,
    reconcile_running_trials,
    terminal_counts,
    validate_study_evidence,
    validate_study_paths,
)

NAMES = (
    "frontier_policy",
    "strong_policy",
    "boatlee_v14_current",
    "economic_policy",
    "middle_policy",
    "sixth_policy",
    "seventh_policy",
    "eighth_policy",
    "ninth_policy",
    "tenth_policy",
    "weakest_policy",
)
PANELS = PanelSet(
    rung_1=("economic_policy", "weakest_policy", "tenth_policy"),
    rung_2=(
        "economic_policy",
        "weakest_policy",
        "tenth_policy",
        "boatlee_v14_current",
        "frontier_policy",
        "sixth_policy",
    ),
    rung_3=NAMES,
)
WEIGHTS = StrengthWeights({name: index % 4 + 1 for index, name in enumerate(NAMES)})
WARM_STARTS = (HybridConfig.default(),)


def _row(name: str) -> FrontierRow:
    return FrontierRow(name, 0, 0.0, 0.0, 0.0, (), 0.0, {}, ())


@pytest.fixture
def identity_inputs(
    tmp_path: Path,
) -> tuple[VerifiedFrontier, SnapshotLeague, FrontierReport, Path]:
    """Create byte-verified source and snapshot facts for one complete league."""
    source_root = tmp_path / "sources"
    snapshot_root = tmp_path / "snapshot"
    source_root.mkdir()
    snapshot_root.mkdir()
    artifacts: list[FrontierArtifact] = []
    snapshots: list[SnapshotSource] = []
    opponents: dict[str, str] = {}
    source_sha256: dict[str, str] = {}
    for name in NAMES:
        source = source_root / f"{name}.py"
        snapshot = snapshot_root / f"{name}.py"
        content = f"# {name}\n".encode()
        source.write_bytes(content)
        snapshot.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        artifacts.append(
            FrontierArtifact(
                name=name,
                provenance="test fixture",
                relative_path=Path(f"{name}.py"),
                sha256=digest,
            )
        )
        snapshots.append(SnapshotSource(name, str(source), str(snapshot), digest))
        opponents[name] = str(source)
        source_sha256[name] = digest
    frontier = VerifiedFrontier(
        engine="1.32.7",
        opponents=opponents,
        artifacts=tuple(artifacts),
        manifest_sha256="a" * 64,
    )
    report = FrontierReport(
        engine="1.32.7",
        seeds=(1,),
        frontier_name=NAMES[0],
        failures=(),
        runtime_seconds=0.0,
        rows=tuple(_row(name) for name in NAMES),
        manifest_sha256="a" * 64,
        source_sha256=source_sha256,
    )
    legacy_state = tmp_path / "legacy-state.json"
    legacy_state.write_text("{}")
    return (
        frontier,
        SnapshotLeague(str(snapshot_root), tuple(snapshots)),
        report,
        legacy_state,
    )


@pytest.fixture
def identity(
    identity_inputs: tuple[VerifiedFrontier, SnapshotLeague, FrontierReport, Path],
) -> StudyIdentity:
    """Build the canonical immutable semantic identity used by storage tests."""
    frontier, snapshot, report, legacy_state = identity_inputs
    return build_identity(
        frontier,
        snapshot,
        report,
        PANELS,
        WEIGHTS,
        WARM_STARTS,
        hashlib.sha256(legacy_state.read_bytes()).hexdigest(),
    )


def test_identity_binds_every_semantic_input(identity: StudyIdentity) -> None:
    """The canonical study record includes every field that changes game meaning."""
    assert identity.study_name == "hybrid-optuna-v2"
    assert identity.optuna_version == "4.9.0"
    assert identity.maximum_trials == 512
    assert identity.space_sha256 == SPACE_SHA256
    assert identity.sampler.model_dump() == {
        "seed": 20260827,
        "multivariate": True,
        "group": True,
        "n_startup_trials": 16,
    }
    assert identity.pruner.model_dump() == {
        "min_resource": 1,
        "reduction_factor": 4,
        "min_early_stopping_rate": 0,
    }
    assert identity.warm_start_sha256 == tuple(
        config_sha256(item) for item in WARM_STARTS
    )
    assert (
        identity.digest
        == hashlib.sha256(
            json.dumps(
                identity.model_dump(mode="json"),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest()
    )


def test_resume_rejects_semantic_drift_before_sqlite_mutation(
    tmp_path: Path, identity: StudyIdentity
) -> None:
    """A changed semantic identity cannot cause even an incidental SQLite write."""
    paths = StudyPaths.from_root(tmp_path / "study")
    open_study(paths, identity)
    original = paths.sqlite.read_bytes()

    with pytest.raises(StudyIdentityError, match="space_sha256"):
        open_study(paths, identity.model_copy(update={"space_sha256": "f" * 64}))

    assert paths.sqlite.read_bytes() == original


def test_paths_reject_snapshot_or_legacy_state_aliases(
    tmp_path: Path,
    identity_inputs: tuple[VerifiedFrontier, SnapshotLeague, FrontierReport, Path],
) -> None:
    """The new run cannot be rooted in immutable snapshots or legacy evidence."""
    _, snapshot, _, legacy_state = identity_inputs
    with pytest.raises(ValueError, match="protected"):
        validate_study_paths(
            StudyPaths.from_root(Path(snapshot.root) / "new-study"),
            snapshot,
            legacy_state,
        )
    with pytest.raises(ValueError, match="protected"):
        validate_study_paths(StudyPaths.from_root(legacy_state), snapshot, legacy_state)


def test_paths_reject_symlink_and_inode_aliases_before_run_root_mutation(
    tmp_path: Path,
    identity_inputs: tuple[VerifiedFrontier, SnapshotLeague, FrontierReport, Path],
) -> None:
    """Neither a symlink nor a hard-linked fixed child can reach protected bytes."""
    _, snapshot, _, legacy_state = identity_inputs
    alias = tmp_path / "snapshot-alias"
    alias.symlink_to(snapshot.root, target_is_directory=True)
    linked_root = StudyPaths.from_root(alias / "new-study")
    with pytest.raises(ValueError, match="symlink"):
        validate_study_paths(linked_root, snapshot, legacy_state)
    assert not (Path(snapshot.root) / "new-study").exists()

    root = tmp_path / "study"
    root.mkdir()
    paths = StudyPaths.from_root(root)
    paths.identity.hardlink_to(legacy_state)
    with pytest.raises(ValueError, match="aliases protected"):
        validate_study_paths(paths, snapshot, legacy_state)


def _set_summary(
    trial: optuna.trial.Trial,
    evidence_digest: str,
    evidence: RungEvidence,
) -> None:
    trial.set_user_attr("config_sha256", evidence.config_sha256)
    trial.set_user_attr("rung", evidence.rung)
    trial.set_user_attr("resource_step", evidence.resource_step)
    trial.set_user_attr("objective", evidence.objective)
    trial.set_user_attr("evidence_sha256", evidence_digest)


def _frozen_trial_payload(trials: list[optuna.trial.FrozenTrial]) -> list[object]:
    """Keep enough immutable trial history to detect accidental reconciliation edits."""
    return [
        (
            trial.number,
            trial.state,
            trial.value,
            trial.params,
            trial.user_attrs,
            trial.system_attrs,
        )
        for trial in trials
    ]


def _rung_rows(spec: RungSpec) -> dict[GameKey, GameResult]:
    return {
        GameKey(opponent, seed, seat): GameResult(
            GameKey(opponent, seed, seat), 10 + seat, 3, 0.01
        )
        for opponent in spec.opponents
        for seed in spec.seeds
        for seat in (0, 1)
    }


def _complete_trial(
    study: optuna.Study, paths: StudyPaths, *, number: int = 0
) -> RungEvidence:
    trial = study.ask()
    assert trial.number == number
    spec = RungSpec(1, 1, PANELS.rung_1, (860_000, 860_001, 860_002, 860_003))
    evidence = build_rung_evidence(
        number, HybridConfig.default(), spec, _rung_rows(spec), WEIGHTS
    )
    _, digest = write_rung_evidence_atomic(paths.root, evidence)
    _set_summary(trial, digest, evidence)
    assert evidence.objective is not None
    study.tell(trial, evidence.objective)
    return evidence


def test_resume_marks_only_stale_running_trial_failed(
    tmp_path: Path, identity: StudyIdentity
) -> None:
    """Restart reconciliation changes exactly stale RUNNING rows, never history."""
    paths = StudyPaths.from_root(tmp_path / "study")
    study = open_study(paths, identity)
    complete = study.ask()
    study.tell(complete, 0.2)
    pruned = study.ask()
    study.tell(pruned, state=TrialState.PRUNED)
    running = study.ask()
    before = _frozen_trial_payload(study.trials[:2])

    reconciled = reconcile_running_trials(study)

    assert reconciled == (running.number,)
    assert study.trials[2].state is TrialState.FAIL
    assert study.trials[2].system_attrs["interruption_reason"] == "process_restarted"
    assert _frozen_trial_payload(study.trials[:2]) == before
    assert terminal_counts(study) == {"complete": 1, "pruned": 1, "failed": 1}


def test_completed_evidence_must_match_sqlite_user_attributes(
    tmp_path: Path, identity: StudyIdentity
) -> None:
    """SQLite summaries cannot bless a changed canonical evidence record."""
    paths = StudyPaths.from_root(tmp_path / "study")
    study = open_study(paths, identity)
    evidence = _complete_trial(study, paths)
    evidence_file = paths.evidence / f"trial-0-rung-{evidence.rung}.json"
    payload = json.loads(evidence_file.read_text())
    payload["objective"] = 0.99
    evidence_file.write_text(json.dumps(payload))

    with pytest.raises(StudyEvidenceError, match="trial 0 rung 1"):
        validate_study_evidence(study, paths, identity)


def test_validation_rejects_missing_and_orphan_completed_evidence(
    tmp_path: Path, identity: StudyIdentity
) -> None:
    """Every terminal record and every durable rung file must have one counterpart."""
    paths = StudyPaths.from_root(tmp_path / "study")
    study = open_study(paths, identity)
    _complete_trial(study, paths)
    with pytest.raises(StudyEvidenceError, match="rung 3"):
        validate_study_evidence(study, paths, identity)


def test_validation_accepts_complete_cumulative_evidence(
    tmp_path: Path, identity: StudyIdentity
) -> None:
    """Every stored rung may be replayed against the identity before certification."""
    paths = StudyPaths.from_root(tmp_path / "study")
    study = open_study(paths, identity)
    trial = study.ask()
    final: RungEvidence | None = None
    for rung, resource_step, opponents, seeds in (
        (1, 1, PANELS.rung_1, tuple(range(860_000, 860_004))),
        (2, 4, PANELS.rung_2, tuple(range(860_000, 860_016))),
        (3, 16, PANELS.rung_3, tuple(range(860_000, 860_032))),
    ):
        spec = RungSpec(rung, resource_step, opponents, seeds)
        evidence = build_rung_evidence(
            trial.number, HybridConfig.default(), spec, _rung_rows(spec), WEIGHTS
        )
        _, digest = write_rung_evidence_atomic(paths.root, evidence)
        _set_summary(trial, digest, evidence)
        final = evidence
    assert final is not None
    assert final.objective is not None
    study.tell(trial, final.objective)

    validate_study_evidence(study, paths, identity)


def test_close_hash_checkpoints_wal_and_preserves_terminal_states(
    tmp_path: Path, identity: StudyIdentity
) -> None:
    """The final database digest covers a checkpointed database with no sidecars."""
    paths = StudyPaths.from_root(tmp_path / "study")
    study = open_study(paths, identity)
    trial = study.ask()
    study.tell(trial, 0.4)

    digest = close_and_hash_storage(study, paths)

    assert digest == hashlib.sha256(paths.sqlite.read_bytes()).hexdigest()
    assert not paths.sqlite.with_name("study.sqlite3-wal").exists()
    assert not paths.sqlite.with_name("study.sqlite3-shm").exists()
    connection = sqlite3.connect(f"file:{paths.sqlite}?mode=ro", uri=True)
    try:
        assert connection.execute("SELECT state FROM trials").fetchall() == [
            ("COMPLETE",)
        ]
    finally:
        connection.close()
