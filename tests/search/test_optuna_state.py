"""Durable identity, SQLite resume, and evidence contracts for Optuna search."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
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
from kaggriculture.search.optuna_space import (
    SPACE_SHA256,
    config_sha256,
    suggest_config,
)
from kaggriculture.search.optuna_state import (
    StudyEvidenceError,
    StudyIdentity,
    StudyIdentityError,
    StudyPaths,
    StudyPreflight,
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
PreflightFactory = Callable[[StudyPaths], StudyPreflight]


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


@pytest.fixture
def preflight_factory(
    identity: StudyIdentity,
    identity_inputs: tuple[VerifiedFrontier, SnapshotLeague, FrontierReport, Path],
) -> PreflightFactory:
    """Return a capability factory for preflighted study paths in this fixture."""
    _, snapshot, _, legacy_state = identity_inputs

    def prepare(paths: StudyPaths) -> StudyPreflight:
        preflight = validate_study_paths(paths, snapshot, legacy_state, identity)
        assert isinstance(preflight, StudyPreflight)
        return preflight

    return prepare


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
    tmp_path: Path,
    identity: StudyIdentity,
    preflight_factory: PreflightFactory,
    identity_inputs: tuple[VerifiedFrontier, SnapshotLeague, FrontierReport, Path],
) -> None:
    """A changed semantic identity cannot cause even an incidental SQLite write."""
    paths = StudyPaths.from_root(tmp_path / "study")
    preflight = preflight_factory(paths)
    open_study(paths, identity, preflight)
    original = paths.sqlite.read_bytes()

    _, snapshot, _, legacy_state = identity_inputs
    drift = identity.model_copy(update={"space_sha256": "f" * 64})
    drift_preflight = validate_study_paths(paths, snapshot, legacy_state, drift)
    with pytest.raises(StudyIdentityError, match="space_sha256"):
        open_study(
            paths,
            drift,
            drift_preflight,
        )

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


def test_preflight_rejects_parent_traversal_and_every_fixed_output_alias(
    tmp_path: Path,
    identity: StudyIdentity,
    identity_inputs: tuple[VerifiedFrontier, SnapshotLeague, FrontierReport, Path],
) -> None:
    """Lexical traversal and all six fixed outputs are protected before mutation."""
    _, snapshot, _, legacy_state = identity_inputs
    traversal = StudyPaths.from_root(
        tmp_path / "sibling" / ".." / "snapshot" / "new-study"
    )
    with pytest.raises(ValueError, match="protected"):
        validate_study_paths(traversal, snapshot, legacy_state, identity)
    assert not (Path(snapshot.root) / "new-study").exists()

    for field in ("sqlite", "identity", "evidence", "diagnostic", "finalists"):
        root = tmp_path / f"study-{field}"
        root.mkdir()
        paths = StudyPaths.from_root(root)
        getattr(paths, field).hardlink_to(legacy_state)
        with pytest.raises(ValueError, match="aliases protected"):
            validate_study_paths(paths, snapshot, legacy_state, identity)


def test_open_requires_preflight_and_rejects_mutated_external_facts(
    tmp_path: Path,
    identity: StudyIdentity,
    identity_inputs: tuple[VerifiedFrontier, SnapshotLeague, FrontierReport, Path],
) -> None:
    """No root or SQLite file is created without a matching external preflight."""
    _, snapshot, _, legacy_state = identity_inputs
    paths = StudyPaths.from_root(tmp_path / "study")
    with pytest.raises(StudyIdentityError, match="preflight"):
        open_study(paths, identity, None)
    assert not paths.root.exists()

    mutated = Path(snapshot.sources[0].snapshot_path)
    mutated.write_text("changed")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_study_paths(paths, snapshot, legacy_state, identity)
    assert not paths.root.exists()


def test_open_rechecks_preflight_external_bytes_before_mutating_root(
    tmp_path: Path,
    identity: StudyIdentity,
    preflight_factory: PreflightFactory,
    identity_inputs: tuple[VerifiedFrontier, SnapshotLeague, FrontierReport, Path],
) -> None:
    """A capability is not a stale authorization if a snapshot changes afterward."""
    _, snapshot, _, _ = identity_inputs
    paths = StudyPaths.from_root(tmp_path / "study")
    preflight = preflight_factory(paths)
    Path(snapshot.sources[0].source_path).write_text("changed")

    with pytest.raises(ValueError, match="SHA-256"):
        open_study(paths, identity, preflight)

    assert not paths.root.exists()


def test_resume_reconstructs_exact_sampler_and_pruner(
    tmp_path: Path, identity: StudyIdentity, preflight_factory: PreflightFactory
) -> None:
    """Resume never silently replaces identity-bound TPE/halving behavior."""
    paths = StudyPaths.from_root(tmp_path / "study")
    preflight = preflight_factory(paths)
    open_study(paths, identity, preflight)

    resumed = open_study(paths, identity, preflight_factory(paths))

    assert isinstance(resumed.sampler, optuna.samplers.TPESampler)
    assert resumed.sampler._n_startup_trials == identity.sampler.n_startup_trials  # noqa: SLF001
    assert isinstance(resumed.pruner, optuna.pruners.SuccessiveHalvingPruner)
    assert resumed.pruner._min_resource == identity.pruner.min_resource  # noqa: SLF001
    assert resumed.pruner._reduction_factor == identity.pruner.reduction_factor  # noqa: SLF001
    assert (
        resumed.pruner._min_early_stopping_rate
        == identity.pruner.min_early_stopping_rate
    )  # noqa: SLF001


def test_resume_db_identity_mismatch_does_not_mutate_sqlite(
    tmp_path: Path, identity: StudyIdentity, preflight_factory: PreflightFactory
) -> None:
    """The stored DB identity is read through a readonly connection before RDB open."""
    paths = StudyPaths.from_root(tmp_path / "study")
    open_study(paths, identity, preflight_factory(paths))
    connection = sqlite3.connect(paths.sqlite)
    try:
        connection.execute(
            "UPDATE study_user_attributes SET value_json = ? "
            "WHERE key = 'study_identity_sha256'",
            (json.dumps("tampered"),),
        )
        connection.commit()
    finally:
        connection.close()
    original = paths.sqlite.read_bytes()

    with pytest.raises(StudyIdentityError, match="study_identity_sha256"):
        open_study(paths, identity, preflight_factory(paths))

    assert paths.sqlite.read_bytes() == original


def test_runtime_optuna_drift_rejects_before_creating_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    identity: StudyIdentity,
    preflight_factory: PreflightFactory,
) -> None:
    """A dependency upgrade cannot create an identity-labelled incompatible study."""
    paths = StudyPaths.from_root(tmp_path / "study")
    monkeypatch.setattr(optuna, "__version__", "9.9.9")

    with pytest.raises(StudyIdentityError, match="Optuna 9.9.9"):
        open_study(paths, identity, preflight_factory(paths))

    assert not paths.root.exists()


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
    config = suggest_config(trial)
    spec = RungSpec(1, 1, PANELS.rung_1, (860_000, 860_001, 860_002, 860_003))
    evidence = build_rung_evidence(number, config, spec, _rung_rows(spec), WEIGHTS)
    _, digest = write_rung_evidence_atomic(paths.root, evidence)
    _set_summary(trial, digest, evidence)
    assert evidence.objective is not None
    trial.report(evidence.objective, step=evidence.resource_step)
    study.tell(trial, evidence.objective)
    return evidence


def test_resume_marks_only_stale_running_trial_failed(
    tmp_path: Path, identity: StudyIdentity, preflight_factory: PreflightFactory
) -> None:
    """Restart reconciliation changes exactly stale RUNNING rows, never history."""
    paths = StudyPaths.from_root(tmp_path / "study")
    study = open_study(paths, identity, preflight_factory(paths))
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


def test_reconciliation_private_storage_bridge_is_version_guarded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    identity: StudyIdentity,
    preflight_factory: PreflightFactory,
) -> None:
    """The narrow private system-attribute bridge cannot silently span Optuna APIs."""
    paths = StudyPaths.from_root(tmp_path / "study")
    study = open_study(paths, identity, preflight_factory(paths))
    trial = study.ask()
    monkeypatch.setattr(optuna, "__version__", "9.9.9")

    with pytest.raises(RuntimeError, match="private storage bridge"):
        reconcile_running_trials(study)

    assert study.trials[trial.number].state is TrialState.RUNNING


def test_completed_evidence_must_match_sqlite_user_attributes(
    tmp_path: Path, identity: StudyIdentity, preflight_factory: PreflightFactory
) -> None:
    """SQLite summaries cannot bless a changed canonical evidence record."""
    paths = StudyPaths.from_root(tmp_path / "study")
    study = open_study(paths, identity, preflight_factory(paths))
    evidence = _complete_trial(study, paths)
    evidence_file = paths.evidence / f"trial-0-rung-{evidence.rung}.json"
    payload = json.loads(evidence_file.read_text())
    payload["objective"] = 0.99
    evidence_file.write_text(json.dumps(payload))

    with pytest.raises(StudyEvidenceError, match="trial 0 rung 1"):
        validate_study_evidence(study, paths, identity)


def test_validation_rejects_missing_and_orphan_completed_evidence(
    tmp_path: Path, identity: StudyIdentity, preflight_factory: PreflightFactory
) -> None:
    """Every terminal record and every durable rung file must have one counterpart."""
    paths = StudyPaths.from_root(tmp_path / "study")
    study = open_study(paths, identity, preflight_factory(paths))
    _complete_trial(study, paths)
    with pytest.raises(StudyEvidenceError, match="rung 3"):
        validate_study_evidence(study, paths, identity)


@pytest.mark.parametrize("state", (TrialState.PRUNED, TrialState.FAIL))
def test_validation_rejects_missing_pruned_or_arbitrary_failed_evidence(
    tmp_path: Path,
    identity: StudyIdentity,
    preflight_factory: PreflightFactory,
    state: TrialState,
) -> None:
    """Terminal non-complete rows need evidence or a restart/failure cause."""
    paths = StudyPaths.from_root(tmp_path / state.name.lower())
    study = open_study(paths, identity, preflight_factory(paths))
    trial = study.ask()
    suggest_config(trial)
    study.tell(trial, state=state)

    expected = "missing evidence" if state is TrialState.PRUNED else "FAIL lacks"
    with pytest.raises(StudyEvidenceError, match=expected):
        validate_study_evidence(study, paths, identity)


def test_validation_rejects_orphan_and_noncanonical_evidence(
    tmp_path: Path, identity: StudyIdentity, preflight_factory: PreflightFactory
) -> None:
    """Parsable JSON is insufficient unless it is canonical and owned by one trial."""
    paths = StudyPaths.from_root(tmp_path / "study")
    study = open_study(paths, identity, preflight_factory(paths))
    evidence = _complete_trial(study, paths)
    path = paths.evidence / f"trial-0-rung-{evidence.rung}.json"
    path.write_text(json.dumps(json.loads(path.read_text())))
    with pytest.raises(StudyEvidenceError, match="not canonical"):
        validate_study_evidence(study, paths, identity)


def test_validation_rejects_orphan_evidence_and_accepts_execution_failed_trial(
    tmp_path: Path, identity: StudyIdentity, preflight_factory: PreflightFactory
) -> None:
    """Orphans fail, while a durable engine failure is a legitimate FAILED record."""
    paths = StudyPaths.from_root(tmp_path / "orphan")
    study = open_study(paths, identity, preflight_factory(paths))
    orphan_spec = RungSpec(1, 1, PANELS.rung_1, tuple(range(860_000, 860_004)))
    orphan = build_rung_evidence(
        9, HybridConfig.default(), orphan_spec, _rung_rows(orphan_spec), WEIGHTS
    )
    write_rung_evidence_atomic(paths.root, orphan)
    with pytest.raises(StudyEvidenceError, match="orphan"):
        validate_study_evidence(study, paths, identity)

    paths = StudyPaths.from_root(tmp_path / "failed")
    study = open_study(paths, identity, preflight_factory(paths))
    trial = study.ask()
    config = suggest_config(trial)
    spec = RungSpec(1, 1, PANELS.rung_1, tuple(range(860_000, 860_004)))
    rows = _rung_rows(spec)
    key = next(iter(rows))
    rows[key] = GameResult(key, None, None, 0.01, "engine failed")
    failed = build_rung_evidence(trial.number, config, spec, rows, WEIGHTS)
    _, digest = write_rung_evidence_atomic(paths.root, failed)
    _set_summary(trial, digest, failed)
    study.tell(trial, state=TrialState.FAIL)

    validate_study_evidence(study, paths, identity)


def test_validation_rejects_tampered_intermediate_value(
    tmp_path: Path, identity: StudyIdentity, preflight_factory: PreflightFactory
) -> None:
    """Each rung objective must equal its exact Optuna intermediate report."""
    paths = StudyPaths.from_root(tmp_path / "study")
    study = open_study(paths, identity, preflight_factory(paths))
    _complete_trial(study, paths)
    connection = sqlite3.connect(paths.sqlite)
    try:
        connection.execute(
            "UPDATE trial_intermediate_values SET intermediate_value = 0.0"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(StudyEvidenceError, match="intermediate differs"):
        validate_study_evidence(study, paths, identity)


def test_validation_recomputes_config_digest_from_stored_trial_params(
    tmp_path: Path, identity: StudyIdentity, preflight_factory: PreflightFactory
) -> None:
    """Matching evidence and summary hashes cannot replace the actual Optuna params."""
    paths = StudyPaths.from_root(tmp_path / "study")
    study = open_study(paths, identity, preflight_factory(paths))
    evidence = _complete_trial(study, paths)
    path = paths.evidence / f"trial-0-rung-{evidence.rung}.json"
    payload = evidence.model_dump(mode="json")
    payload["config_sha256"] = "f" * 64
    source = (
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()
    path.write_bytes(source)
    digest = hashlib.sha256(source).hexdigest()
    connection = sqlite3.connect(paths.sqlite)
    try:
        connection.execute(
            "UPDATE trial_user_attributes SET value_json = ? "
            "WHERE key = 'config_sha256'",
            (json.dumps("f" * 64),),
        )
        connection.execute(
            "UPDATE trial_user_attributes SET value_json = ? "
            "WHERE key = 'evidence_sha256'",
            (json.dumps(digest),),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(StudyEvidenceError, match="stored parameters"):
        validate_study_evidence(study, paths, identity)


def test_validation_accepts_complete_cumulative_evidence(
    tmp_path: Path, identity: StudyIdentity, preflight_factory: PreflightFactory
) -> None:
    """Every stored rung may be replayed against the identity before certification."""
    paths = StudyPaths.from_root(tmp_path / "study")
    study = open_study(paths, identity, preflight_factory(paths))
    trial = study.ask()
    config = suggest_config(trial)
    final: RungEvidence | None = None
    for rung, resource_step, opponents, seeds in (
        (1, 1, PANELS.rung_1, tuple(range(860_000, 860_004))),
        (2, 4, PANELS.rung_2, tuple(range(860_000, 860_016))),
        (3, 16, PANELS.rung_3, tuple(range(860_000, 860_032))),
    ):
        spec = RungSpec(rung, resource_step, opponents, seeds)
        evidence = build_rung_evidence(
            trial.number, config, spec, _rung_rows(spec), WEIGHTS
        )
        _, digest = write_rung_evidence_atomic(paths.root, evidence)
        _set_summary(trial, digest, evidence)
        assert evidence.objective is not None
        trial.report(evidence.objective, step=resource_step)
        final = evidence
    assert final is not None
    assert final.objective is not None
    study.tell(trial, final.objective)

    validate_study_evidence(study, paths, identity)


def test_close_hash_checkpoints_wal_and_preserves_terminal_states(
    tmp_path: Path, identity: StudyIdentity, preflight_factory: PreflightFactory
) -> None:
    """The final database digest covers a checkpointed database with no sidecars."""
    paths = StudyPaths.from_root(tmp_path / "study")
    study = open_study(paths, identity, preflight_factory(paths))
    trial = study.ask()
    study.tell(trial, 0.4)
    pruned = study.ask()
    study.tell(pruned, state=TrialState.PRUNED)
    failed = study.ask()
    study.tell(failed, state=TrialState.FAIL)

    digest = close_and_hash_storage(study, paths)

    assert digest == hashlib.sha256(paths.sqlite.read_bytes()).hexdigest()
    assert not paths.sqlite.with_name("study.sqlite3-wal").exists()
    assert not paths.sqlite.with_name("study.sqlite3-shm").exists()
    reopened = open_study(paths, identity, preflight_factory(paths))
    assert [trial.state for trial in reopened.trials] == [
        TrialState.COMPLETE,
        TrialState.PRUNED,
        TrialState.FAIL,
    ]
