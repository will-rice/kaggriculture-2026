"""Immutable Optuna study identity and durable SQLite lifecycle helpers."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import sqlite3
import stat
import tempfile
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Literal, Self, cast

import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from optuna.storages import BaseStorage, RDBStorage
from optuna.trial import FrozenTrial, TrialState
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from kaggriculture.hybrid.config import HybridConfig
from kaggriculture.search.evolution import PROMOTION_SEEDS, SnapshotLeague
from kaggriculture.search.fitness import StrengthWeights
from kaggriculture.search.frontier import FrontierReport, VerifiedFrontier
from kaggriculture.search.optuna_protocol import (
    RUNG_1_SEEDS,
    RUNG_2_SEEDS,
    RUNG_3_SEEDS,
    PanelSet,
    RungEvidence,
    evidence_path,
    rung_summary_attributes,
)
from kaggriculture.search.optuna_space import (
    SPACE_SHA256,
    config_sha256,
    parameters_for_config,
    suggest_config,
)
from kaggriculture.search.promotion import DETERMINISM_SEEDS

_IDENTITY_ATTRIBUTE = "study_identity_sha256"
_IDENTITY_TEMPORARY_NAME = re.compile(r"\.identity\.[a-z0-9_]{8}\.tmp")
_PREFLIGHT_CAPABILITY = object()
_EVALUATION_SEMANTIC_PATHS = (
    "src/kaggriculture/action_codec.py",
    "src/kaggriculture/actions.py",
    "src/kaggriculture/constants.py",
    "src/kaggriculture/features.py",
    "src/kaggriculture/observation.py",
    "src/kaggriculture/hybrid/config.py",
    "src/kaggriculture/hybrid/jobs.py",
    "src/kaggriculture/hybrid/market.py",
    "src/kaggriculture/hybrid/opening.py",
    "src/kaggriculture/hybrid/policy.py",
    "src/kaggriculture/hybrid/runtime.py",
    "src/kaggriculture/hybrid/schema.py",
    "src/kaggriculture/search/arena.py",
    "src/kaggriculture/search/arena_pool.py",
    "src/kaggriculture/search/fitness.py",
    "src/kaggriculture/search/optuna_protocol.py",
    "src/kaggriculture/search/optuna_search.py",
    "src/kaggriculture/search/optuna_space.py",
    "src/kaggriculture/search/route.py",
)


class StudyIdentityError(ValueError):
    """Stored and requested semantic identities differ."""


class StudyEvidenceError(ValueError):
    """SQLite trial state and canonical evidence disagree."""


class StudyOwnershipError(RuntimeError):
    """Another process already owns the canonical study root."""


@contextmanager
def study_root_lock(root: Path, *, exclusive: bool) -> Iterator[Path]:
    """Take one nonblocking process lock keyed by the canonical study root."""
    canonical = _canonical_lexical_path(root)
    _reject_symlink_components(canonical)
    parent = canonical.parent
    parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(parent)
    lock_path = parent / f".{canonical.name}.lock"
    if lock_path.is_symlink():
        raise StudyOwnershipError("study lock path cannot be a symlink")
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise StudyOwnershipError("study lock path must be a regular file")
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(descriptor, operation | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise StudyOwnershipError(
                f"study root is already owned: {canonical}"
            ) from error
        yield lock_path
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class StudyPaths:
    """The only permitted durable locations under one Optuna run root."""

    root: Path
    sqlite: Path
    identity: Path
    evidence: Path
    quarantine: Path
    diagnostic: Path
    finalists: Path

    @classmethod
    def from_root(cls, root: Path) -> Self:
        """Derive the complete fixed path layout from a caller-owned root."""
        return cls(
            root=root,
            sqlite=root / "study.sqlite3",
            identity=root / "identity.json",
            evidence=root / "evidence",
            quarantine=root / "quarantine",
            diagnostic=root / "diagnostic.json",
            finalists=root / "finalists.json",
        )


@dataclass(frozen=True, init=False)
class StudyPreflight:
    """An unforgeable in-process proof that external study facts were verified."""

    paths: StudyPaths
    identity_digest: str
    _snapshot: SnapshotLeague
    _legacy_state: Path
    _capability: object

    @classmethod
    def _create(
        cls,
        paths: StudyPaths,
        snapshot: SnapshotLeague,
        legacy_state: Path,
        identity: "StudyIdentity",
    ) -> Self:
        """Create a capability only after all external bytes and paths validate."""
        result = object.__new__(cls)
        object.__setattr__(result, "paths", paths)
        object.__setattr__(result, "identity_digest", identity.digest)
        object.__setattr__(result, "_snapshot", snapshot)
        object.__setattr__(result, "_legacy_state", legacy_state)
        object.__setattr__(result, "_capability", _PREFLIGHT_CAPABILITY)
        return result


class SamplerIdentity(BaseModel):
    """All TPE settings that affect sequential suggestion order."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    seed: Literal[20260827] = 20260827
    multivariate: Literal[True] = True
    group: Literal[True] = True
    n_startup_trials: Literal[16] = 16


class PrunerIdentity(BaseModel):
    """All successive-halving settings that affect promotion decisions."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    min_resource: Literal[1] = 1
    reduction_factor: Literal[4] = 4
    min_early_stopping_rate: Literal[0] = 0


class SeedBankIdentity(BaseModel):
    """The complete tuning banks and protected promotion-bank digest."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    rung_1: tuple[int, ...]
    rung_2: tuple[int, ...]
    rung_3: tuple[int, ...]
    protected_promotion_sha256: str


class PanelIdentity(BaseModel):
    """The exact ordered opponent panels used at each resource rung."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    rung_1: tuple[str, str, str]
    rung_2: tuple[str, str, str, str, str, str]
    rung_3: tuple[str, ...]


class StudyIdentity(BaseModel):
    """Every immutable semantic fact required to interpret a study."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal[2] = 2
    study_name: Literal["hybrid-optuna-v2"] = "hybrid-optuna-v2"
    optuna_version: Literal["4.9.0"] = "4.9.0"
    engine: str
    manifest_sha256: str
    frontier_report_sha256: str
    source_sha256: tuple[tuple[str, str], ...]
    league_snapshots: tuple[tuple[str, str, str, str], ...]
    space_sha256: str
    evaluation_semantics: tuple[tuple[str, str], ...]
    evaluation_semantics_sha256: str
    sampler: SamplerIdentity
    pruner: PrunerIdentity
    seed_banks: SeedBankIdentity
    panels: PanelIdentity
    strength_weights: tuple[tuple[str, int], ...]
    objective_schema: Literal["win-primary-margin-epsilon-v1"]
    warm_start_sha256: tuple[str, ...]
    legacy_state_sha256: str
    maximum_trials: Literal[512] = 512

    @model_validator(mode="after")
    def has_canonical_evaluation_semantics(self) -> Self:
        """Bind a canonical, content-addressed executable source manifest."""
        if self.evaluation_semantics != tuple(sorted(self.evaluation_semantics)):
            raise ValueError("evaluation semantics must be sorted")
        names = tuple(name for name, _digest in self.evaluation_semantics)
        if len(names) != len(set(names)) or any(
            Path(name).is_absolute() or Path(name) == Path() or ".." in Path(name).parts
            for name in names
        ):
            raise ValueError("evaluation semantic paths must be unique relative paths")
        for _name, digest in self.evaluation_semantics:
            _required_sha256(digest, "evaluation semantic source")
        expected = hashlib.sha256(
            _canonical_json(self.evaluation_semantics)
        ).hexdigest()
        if self.evaluation_semantics_sha256 != expected:
            raise ValueError("evaluation semantics SHA-256 differs from its manifest")
        return self

    @property
    def digest(self) -> str:
        """Return the content address of this canonical semantic record."""
        return hashlib.sha256(_canonical_json(self.model_dump(mode="json"))).hexdigest()


def build_identity(
    frontier: VerifiedFrontier,
    snapshot: SnapshotLeague,
    report: FrontierReport,
    panels: PanelSet,
    weights: StrengthWeights,
    warm_starts: Sequence[HybridConfig],
    legacy_state_sha256: str,
) -> StudyIdentity:
    """Bind verified frontier, snapshot, protocol, and starting-policy facts."""
    _validate_identity_inputs(frontier, snapshot, report, panels, weights, warm_starts)
    source_sha256 = tuple(
        sorted((artifact.name, artifact.sha256) for artifact in frontier.artifacts)
    )
    snapshots = tuple(
        sorted(
            (source.name, source.source_path, source.snapshot_path, source.sha256)
            for source in snapshot.sources
        )
    )
    semantic_sources = evaluation_semantics()
    return StudyIdentity(
        engine=frontier.engine,
        manifest_sha256=_required_sha256(frontier.manifest_sha256, "manifest"),
        frontier_report_sha256=hashlib.sha256(
            _canonical_json(asdict(report))
        ).hexdigest(),
        source_sha256=source_sha256,
        league_snapshots=snapshots,
        space_sha256=SPACE_SHA256,
        evaluation_semantics=semantic_sources,
        evaluation_semantics_sha256=hashlib.sha256(
            _canonical_json(semantic_sources)
        ).hexdigest(),
        sampler=SamplerIdentity(),
        pruner=PrunerIdentity(),
        seed_banks=SeedBankIdentity(
            rung_1=RUNG_1_SEEDS,
            rung_2=RUNG_2_SEEDS,
            rung_3=RUNG_3_SEEDS,
            protected_promotion_sha256=hashlib.sha256(
                _canonical_json(PROMOTION_SEEDS)
            ).hexdigest(),
        ),
        panels=PanelIdentity(
            rung_1=cast(tuple[str, str, str], panels.rung_1),
            rung_2=cast(tuple[str, str, str, str, str, str], panels.rung_2),
            rung_3=panels.rung_3,
        ),
        strength_weights=tuple(weights.as_dict().items()),
        objective_schema="win-primary-margin-epsilon-v1",
        warm_start_sha256=tuple(config_sha256(config) for config in warm_starts),
        legacy_state_sha256=_required_sha256(legacy_state_sha256, "legacy state"),
    )


def evaluation_semantics() -> tuple[tuple[str, str], ...]:
    """Hash the reviewed executable source set that defines trial meaning."""
    repository = _canonical_lexical_path(Path(__file__)).parents[3]
    rows: list[tuple[str, str]] = []
    for relative_name in _EVALUATION_SEMANTIC_PATHS:
        source = repository / relative_name
        _reject_symlink_components(source)
        if not source.is_file() or source.is_symlink():
            raise StudyIdentityError(
                f"evaluation semantic source is not a real file: {relative_name}"
            )
        rows.append((relative_name, hashlib.sha256(source.read_bytes()).hexdigest()))
    return tuple(sorted(rows))


def validate_study_paths(
    paths: StudyPaths,
    snapshot: SnapshotLeague,
    legacy_state: Path,
    identity: StudyIdentity | None = None,
) -> StudyPreflight | None:
    """Reject path aliases before a new run root or SQLite file is created."""
    _validate_layout(paths)
    root = _canonical_lexical_path(paths.root)
    snapshot_root = _canonical_lexical_path(Path(snapshot.root))
    legacy = _canonical_lexical_path(legacy_state)
    _validate_protected_roots(root, snapshot_root, legacy)
    protected = [
        snapshot_root,
        legacy,
        *_validate_snapshot_sources(snapshot, snapshot_root),
    ]
    _reject_existing_protected_aliases(paths, protected)
    if identity is None:
        return None
    _validate_external_identity(snapshot, legacy, identity)
    return StudyPreflight._create(paths, snapshot, legacy, identity)


def _validate_protected_roots(root: Path, snapshot_root: Path, legacy: Path) -> None:
    """Ensure direct protected roots are real, disjoint, and free of symlinks."""
    _reject_symlink_components(root)
    _reject_symlink_components(snapshot_root)
    _reject_symlink_components(legacy)
    if not snapshot_root.is_dir() or snapshot_root.is_symlink():
        raise ValueError("snapshot root must be a real directory")
    if not legacy.is_file() or legacy.is_symlink():
        raise ValueError("legacy state must be a real file")
    _reject_protected_path(root, snapshot_root)
    _reject_protected_path(root, legacy)


def _validate_snapshot_sources(
    snapshot: SnapshotLeague, snapshot_root: Path
) -> list[Path]:
    """Verify immutable source bytes before the study root can be made."""
    protected: list[Path] = []
    for source in snapshot.sources:
        original_raw = _absolute(Path(source.source_path))
        copied_raw = _absolute(Path(source.snapshot_path))
        _reject_symlink_components(original_raw)
        _reject_symlink_components(copied_raw)
        original = _canonical_lexical_path(original_raw)
        copied = _canonical_lexical_path(copied_raw)
        _reject_symlink_components(original)
        _reject_symlink_components(copied)
        if not original.is_file() or not copied.is_file():
            raise ValueError(f"snapshot source {source.name} is not a regular file")
        if not _is_descendant(copied, snapshot_root):
            raise ValueError(f"snapshot source {source.name} escapes snapshot root")
        if hashlib.sha256(original.read_bytes()).hexdigest() != source.sha256:
            raise ValueError(
                f"snapshot source {source.name} no longer matches its SHA-256"
            )
        if hashlib.sha256(copied.read_bytes()).hexdigest() != source.sha256:
            raise ValueError(
                f"snapshot copy {source.name} no longer matches its SHA-256"
            )
        protected.extend((original, copied))
    return protected


def _reject_existing_protected_aliases(
    paths: StudyPaths, protected: Sequence[Path]
) -> None:
    """Reject pre-existing fixed children that share protected inodes."""
    for candidate in _study_output_paths(paths):
        absolute = _canonical_lexical_path(candidate)
        if absolute.exists():
            for protected_path in protected:
                if _same_inode(absolute, protected_path):
                    raise ValueError(
                        "study path aliases protected snapshot or legacy state"
                    )


def open_study(
    paths: StudyPaths,
    identity: StudyIdentity,
    preflight: StudyPreflight | None,
) -> optuna.Study:
    """Create or reopen exactly one identity-bound SQLite-backed Optuna study."""
    _validate_layout(paths)
    _validate_runtime_optuna_version(identity)
    _require_preflight(paths, identity, preflight)
    exists = (paths.sqlite.exists(), paths.identity.exists())
    if exists == (False, False):
        _remove_recoverable_identity_temporaries(paths, identity)
        _create_run_root(paths)
        _write_identity(paths.identity, identity)
        return _complete_initial_study(paths, identity)
    if not paths.identity.exists():
        raise StudyIdentityError("study storage and identity file must appear together")
    _validate_stored_identity(paths.identity, identity)
    if not _is_initialization_only_layout(paths):
        if not paths.sqlite.exists():
            raise StudyIdentityError(
                "study storage and identity file must appear together"
            )
        _validate_sqlite_identity_readonly(paths.sqlite, identity)
        return _load_study(paths, identity)
    state = _sqlite_initialization_state(paths.sqlite, identity)
    if state == "complete":
        _validate_sqlite_identity_readonly(paths.sqlite, identity)
        return _load_study(paths, identity)
    return _complete_initial_study(paths, identity)


def _load_study(paths: StudyPaths, identity: StudyIdentity) -> optuna.Study:
    """Open a read-validated identity-bound study with exact sampler semantics."""
    _validate_sqlite_identity_readonly(paths.sqlite, identity)
    study = optuna.load_study(
        storage=RDBStorage(_sqlite_url(paths.sqlite)),
        study_name=identity.study_name,
        sampler=_sampler_for(identity),
        pruner=_pruner_for(identity),
    )
    stored_digest = study.user_attrs.get(_IDENTITY_ATTRIBUTE)
    if stored_digest != identity.digest:
        raise StudyIdentityError("study_identity_sha256 differs from identity.json")
    return study


def _is_initialization_only_layout(paths: StudyPaths) -> bool:
    """Return whether no durable post-initialization artifact has ever appeared."""
    if not paths.root.is_dir() or paths.root.is_symlink():
        return False
    allowed = {paths.identity.name, paths.sqlite.name}
    return all(path.name in allowed for path in paths.root.iterdir())


def _sqlite_initialization_state(
    path: Path, identity: StudyIdentity
) -> Literal["missing", "uninitialized", "missing_study", "missing_attr", "complete"]:
    """Classify only recoverable empty first-open states through read-only SQLite."""
    if not path.exists():
        return "missing"
    if not path.is_file() or path.is_symlink():
        raise StudyIdentityError("SQLite storage is not a real file")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        if "studies" not in tables:
            return "uninitialized"
        return _initialized_sqlite_state(connection, identity)
    except sqlite3.DatabaseError as error:
        raise StudyIdentityError("cannot read stored SQLite study identity") from error
    finally:
        connection.close()


def _initialized_sqlite_state(
    connection: sqlite3.Connection, identity: StudyIdentity
) -> Literal["missing_study", "missing_attr", "complete"]:
    """Classify a schema-complete database without mutating its study rows."""
    studies = connection.execute(
        "SELECT study_id, study_name FROM studies ORDER BY study_id"
    ).fetchall()
    named = [row for row in studies if row[1] == identity.study_name]
    if not named:
        if studies:
            raise StudyIdentityError("SQLite study name differs from identity")
        return "missing_study"
    if len(studies) != 1 or len(named) != 1:
        raise StudyIdentityError("SQLite contains unexpected study rows")
    study_id = named[0][0]
    directions = connection.execute(
        "SELECT direction FROM study_directions WHERE study_id = ?", (study_id,)
    ).fetchall()
    if directions != [("MAXIMIZE",)]:
        raise StudyIdentityError("SQLite study direction differs from MAXIMIZE")
    return _sqlite_identity_attribute_state(connection, study_id, identity)


def _sqlite_identity_attribute_state(
    connection: sqlite3.Connection,
    study_id: int,
    identity: StudyIdentity,
) -> Literal["missing_attr", "complete"]:
    """Accept only an empty crash gap or the exact stored identity digest."""
    attribute = connection.execute(
        "SELECT value_json FROM study_user_attributes WHERE study_id = ? AND key = ?",
        (study_id, _IDENTITY_ATTRIBUTE),
    ).fetchone()
    if attribute is None:
        trial_count = connection.execute(
            "SELECT COUNT(*) FROM trials WHERE study_id = ?", (study_id,)
        ).fetchone()
        other_attributes = connection.execute(
            "SELECT COUNT(*) FROM study_user_attributes WHERE study_id = ?",
            (study_id,),
        ).fetchone()
        if trial_count != (0,) or other_attributes != (0,):
            raise StudyIdentityError(
                "SQLite study identity attribute is missing from nonempty study"
            )
        return "missing_attr"
    try:
        digest = json.loads(attribute[0])
    except json.JSONDecodeError as error:
        raise StudyIdentityError(
            "SQLite study identity attribute is invalid"
        ) from error
    if digest != identity.digest:
        raise StudyIdentityError("study_identity_sha256 differs from identity.json")
    return "complete"


def _complete_initial_study(paths: StudyPaths, identity: StudyIdentity) -> optuna.Study:
    """Idempotently finish the identity-first empty-study creation transaction."""
    if not _is_initialization_only_layout(paths):
        raise StudyIdentityError(
            "incomplete first-open study contains post-initialization artifacts"
        )
    storage = RDBStorage(_sqlite_url(paths.sqlite))
    study = optuna.create_study(
        storage=storage,
        sampler=_sampler_for(identity),
        pruner=_pruner_for(identity),
        study_name=identity.study_name,
        direction="maximize",
        load_if_exists=True,
    )
    if study.trials:
        raise StudyIdentityError("incomplete first-open study already contains trials")
    stored_digest = study.user_attrs.get(_IDENTITY_ATTRIBUTE)
    if stored_digest is None:
        study.set_user_attr(_IDENTITY_ATTRIBUTE, identity.digest)
    elif stored_digest != identity.digest:
        raise StudyIdentityError("study_identity_sha256 differs from identity.json")
    _validate_sqlite_identity_readonly(paths.sqlite, identity)
    return study


def terminal_counts(study: optuna.Study) -> dict[str, int]:
    """Count only terminal trial states under stable lowercase keys."""
    trials = study.get_trials(deepcopy=False)
    return {
        "complete": sum(trial.state is TrialState.COMPLETE for trial in trials),
        "pruned": sum(trial.state is TrialState.PRUNED for trial in trials),
        "failed": sum(trial.state is TrialState.FAIL for trial in trials),
    }


def reconcile_running_trials(
    study: optuna.Study,
    paths: StudyPaths,
    identity: StudyIdentity,
) -> tuple[int, ...]:
    """Recover proven rung commits, quarantine gaps, then fail stale RUNNING rows."""
    _validate_layout(paths)
    _quarantine_abandoned_temporaries(paths)
    trials = {trial.number: trial for trial in study.get_trials(deepcopy=False)}
    evidence_by_trial = _load_evidence_records(paths, trials, allow_running=True)
    running = tuple(
        trial for trial in trials.values() if trial.state is TrialState.RUNNING
    )
    for trial in running:
        _recover_running_trial(
            study,
            trial,
            evidence_by_trial.get(trial.number, {}),
            paths,
            identity,
        )
    for trial in running:
        _set_interruption_reason(study, trial)
        study.tell(trial.number, state=TrialState.FAIL)
    return tuple(trial.number for trial in running)


def _recover_running_trial(
    study: optuna.Study,
    trial: FrozenTrial,
    records: dict[int, tuple[RungEvidence, str]],
    paths: StudyPaths,
    identity: StudyIdentity,
) -> None:
    """Use only canonical evidence plus exact intermediate rows as commit proof."""
    committed, uncommitted = _classify_running_evidence(trial, records, identity)
    _validate_recovery_prefixes(trial, committed, uncommitted)
    for evidence, _digest in uncommitted:
        _quarantine_path(
            evidence_path(paths.root, trial.number, evidence.rung),
            paths,
            "uncommitted",
        )
    if committed:
        evidence, digest = committed[-1]
        path = evidence_path(paths.root, trial.number, evidence.rung)
        for key, value in rung_summary_attributes(
            evidence, path, digest, paths.root
        ).items():
            _set_trial_user_attr(study, trial, key, value)


def _classify_running_evidence(
    trial: FrozenTrial,
    records: dict[int, tuple[RungEvidence, str]],
    identity: StudyIdentity,
) -> tuple[
    list[tuple[RungEvidence, str]],
    list[tuple[RungEvidence, str]],
]:
    """Partition canonical running evidence by exact SQLite report proof."""
    ordered_rungs = tuple(sorted(records))
    if ordered_rungs and ordered_rungs != tuple(range(1, ordered_rungs[-1] + 1)):
        raise StudyEvidenceError(
            f"trial {trial.number} has non-cumulative running evidence"
        )
    stored_config_sha256: str | None = None
    if records:
        stored_config_sha256 = _config_sha256_from_trial_params(trial)
    committed: list[tuple[RungEvidence, str]] = []
    uncommitted: list[tuple[RungEvidence, str]] = []
    missing = object()
    for rung in ordered_rungs:
        evidence, digest = records[rung]
        _validate_evidence_identity(evidence, identity, trial.number)
        if evidence.config_sha256 != stored_config_sha256:
            raise StudyEvidenceError(
                f"trial {trial.number} rung {rung} config differs from "
                "stored parameters"
            )
        actual = trial.intermediate_values.get(evidence.resource_step, missing)
        if actual is not missing and actual != evidence.objective:
            raise StudyEvidenceError(
                f"trial {trial.number} rung {rung} intermediate differs from evidence"
            )
        if evidence.objective is not None and actual == evidence.objective:
            committed.append((evidence, digest))
        else:
            uncommitted.append((evidence, digest))
    return committed, uncommitted


def _validate_recovery_prefixes(
    trial: FrozenTrial,
    committed: list[tuple[RungEvidence, str]],
    uncommitted: list[tuple[RungEvidence, str]],
) -> None:
    """Require reported and trailing unreported rungs to form one exact prefix."""
    committed_steps = {evidence.resource_step for evidence, _digest in committed}
    if set(trial.intermediate_values) != committed_steps:
        raise StudyEvidenceError(
            f"trial {trial.number} intermediate exists without canonical evidence"
        )
    committed_rungs = tuple(evidence.rung for evidence, _digest in committed)
    if committed_rungs != tuple(range(1, len(committed_rungs) + 1)):
        raise StudyEvidenceError(
            f"trial {trial.number} committed evidence is not a cumulative prefix"
        )
    if uncommitted and any(
        evidence.rung <= len(committed_rungs) for evidence, _digest in uncommitted
    ):
        raise StudyEvidenceError(
            f"trial {trial.number} uncommitted evidence precedes a committed rung"
        )


def validate_study_evidence(
    study: optuna.Study, paths: StudyPaths, identity: StudyIdentity
) -> None:
    """Cross-check terminal trial summaries with every canonical evidence file."""
    _validate_layout(paths)
    _validate_stored_identity(paths.identity, identity)
    if study.user_attrs.get(_IDENTITY_ATTRIBUTE) != identity.digest:
        raise StudyEvidenceError(
            "study identity digest differs from requested identity"
        )
    trials = {trial.number: trial for trial in study.get_trials(deepcopy=False)}
    evidence_by_trial = _load_evidence_records(paths, trials)
    for number, trial in trials.items():
        _validate_terminal_trial_evidence(
            number, trial, evidence_by_trial.get(number, {}), identity, paths
        )


def _load_evidence_records(
    paths: StudyPaths,
    trials: dict[int, FrozenTrial],
    *,
    allow_running: bool = False,
) -> dict[int, dict[int, tuple[RungEvidence, str]]]:
    """Load canonical evidence and reject orphan, duplicate, or active-trial rows."""
    loaded: dict[int, dict[int, tuple[RungEvidence, str]]] = {}
    if not paths.evidence.exists():
        return loaded
    if not paths.evidence.is_dir() or paths.evidence.is_symlink():
        raise StudyEvidenceError("evidence path is not a real directory")
    for path in sorted(paths.evidence.iterdir()):
        evidence, digest = _read_evidence(path, paths)
        trial = trials.get(evidence.trial_number)
        if trial is None:
            raise StudyEvidenceError(
                "orphan completed evidence for trial "
                f"{evidence.trial_number} rung {evidence.rung}"
            )
        if trial.state is TrialState.RUNNING and not allow_running:
            raise StudyEvidenceError(
                f"trial {evidence.trial_number} rung {evidence.rung} "
                "has RUNNING SQLite state"
            )
        trial_rows = loaded.setdefault(evidence.trial_number, {})
        if evidence.rung in trial_rows:
            raise StudyEvidenceError(
                "duplicate evidence for trial "
                f"{evidence.trial_number} rung {evidence.rung}"
            )
        trial_rows[evidence.rung] = (evidence, digest)
    return loaded


def _quarantine_abandoned_temporaries(paths: StudyPaths) -> None:
    """Move only atomic-writer temporary names out of canonical evidence."""
    if not paths.evidence.exists():
        return
    if not paths.evidence.is_dir() or paths.evidence.is_symlink():
        raise StudyEvidenceError("evidence path is not a real directory")
    pattern = re.compile(r"\.trial-\d+-rung-[123]\.json\..+\.tmp")
    for path in sorted(paths.evidence.iterdir()):
        if pattern.fullmatch(path.name):
            _quarantine_path(path, paths, "temporary")


def _quarantine_path(path: Path, paths: StudyPaths, reason: str) -> Path:
    """Durably move one untrusted interrupted-write artifact without deleting it."""
    if not path.is_file() or path.is_symlink():
        raise StudyEvidenceError(f"cannot quarantine non-regular entry {path.name}")
    if paths.quarantine.exists():
        if not paths.quarantine.is_dir() or paths.quarantine.is_symlink():
            raise StudyEvidenceError("quarantine path is not a real directory")
    else:
        paths.quarantine.mkdir()
        _fsync_directory(paths.root)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    base = paths.quarantine / f"{path.name}.{reason}.{digest}"
    destination = base
    suffix = 0
    while destination.exists() or destination.is_symlink():
        suffix += 1
        destination = base.with_name(f"{base.name}.{suffix}")
    os.replace(path, destination)  # noqa: PTH105
    _fsync_directory(path.parent)
    _fsync_directory(paths.quarantine)
    return destination


def _validate_terminal_trial_evidence(
    number: int,
    trial: FrozenTrial,
    records: dict[int, tuple[RungEvidence, str]],
    identity: StudyIdentity,
    paths: StudyPaths,
) -> None:
    """Check one terminal trial's cumulative rungs and final SQLite summary."""
    if trial.state is TrialState.RUNNING:
        return
    if not records:
        _validate_empty_terminal_trial(trial, number)
        return
    _validate_present_terminal_trial(number, trial, records, identity, paths)


def _validate_empty_terminal_trial(trial: FrozenTrial, number: int) -> None:
    """Require evidence except for a restart or Optuna-recorded non-game failure."""
    if trial.state in (TrialState.COMPLETE, TrialState.PRUNED):
        raise StudyEvidenceError(f"trial {number} is terminal but has missing evidence")
    if trial.state is TrialState.FAIL:
        _validate_failed_trial_contract(trial, {}, number)


def _validate_present_terminal_trial(
    number: int,
    trial: FrozenTrial,
    records: dict[int, tuple[RungEvidence, str]],
    identity: StudyIdentity,
    paths: StudyPaths,
) -> None:
    """Validate one complete ordered evidence sequence and its final trial state."""
    ordered_rungs = tuple(sorted(records))
    if ordered_rungs != tuple(range(1, ordered_rungs[-1] + 1)):
        raise StudyEvidenceError(f"trial {number} has non-cumulative rung evidence")
    final, digest = records[ordered_rungs[-1]]
    expected_steps = {
        evidence.resource_step
        for evidence, _ in records.values()
        if evidence.objective is not None
    }
    if set(trial.intermediate_values) != expected_steps:
        raise StudyEvidenceError(
            f"trial {number} intermediate step set differs from canonical evidence"
        )
    stored_config_sha256 = _config_sha256_from_trial_params(trial)
    for rung, (evidence, _) in records.items():
        _validate_evidence_identity(evidence, identity, number)
        _validate_trial_rung(trial, evidence)
        if evidence.config_sha256 != stored_config_sha256:
            raise StudyEvidenceError(
                f"trial {number} rung {rung} config differs from stored parameters"
            )
        if evidence.config_sha256 != final.config_sha256:
            raise StudyEvidenceError(
                f"trial {number} rung {rung} config differs from its final evidence"
            )
    _validate_trial_summary(trial, final, digest, paths)
    if trial.state is TrialState.COMPLETE and ordered_rungs != (1, 2, 3):
        raise StudyEvidenceError(
            f"trial {number} is COMPLETE but is missing rung 3 evidence"
        )
    if (
        trial.state in (TrialState.COMPLETE, TrialState.PRUNED)
        and trial.value != final.objective
    ):
        raise StudyEvidenceError(f"trial {number} objective differs from SQLite value")
    if trial.state is TrialState.FAIL:
        _validate_failed_trial_contract(trial, records, number)


def _validate_trial_rung(trial: FrozenTrial, evidence: RungEvidence) -> None:
    """Cross-check each durable rung against its Optuna intermediate value."""
    actual = trial.intermediate_values.get(evidence.resource_step)
    if evidence.objective is None:
        if actual is not None:
            raise StudyEvidenceError(
                f"trial {trial.number} rung {evidence.rung} has unexpected intermediate"
            )
        return
    if actual != evidence.objective:
        raise StudyEvidenceError(
            f"trial {trial.number} rung {evidence.rung} intermediate "
            "differs from evidence"
        )


def _validate_failed_trial_contract(
    trial: FrozenTrial,
    records: dict[int, tuple[RungEvidence, str]],
    number: int,
) -> None:
    """Reject FAILED rows without canonical execution-failure or restart facts."""
    interrupted = trial.system_attrs.get("interruption_reason") == "process_restarted"
    failure = trial.system_attrs.get("fail_reason")
    if not records:
        if not interrupted and not isinstance(failure, str):
            raise StudyEvidenceError(
                f"trial {number} FAIL lacks interruption/failure contract"
            )
        return
    if records[max(records)][0].failures or interrupted:
        return
    raise StudyEvidenceError(f"trial {number} FAIL lacks interruption/failure contract")


@dataclass(frozen=True)
class _StoredParameterTrial:
    """A strict TrialLike adapter used to rebuild config from stored parameters."""

    params: dict[str, int | float | str]

    def suggest_int(self, name: str, low: int, high: int, *, step: int = 1) -> int:
        """Return a stored integer only when it remains in the declared domain."""
        value = self._value(name)
        if type(value) is not int or not low <= value <= high or (value - low) % step:
            raise StudyEvidenceError(
                f"stored trial parameter {name} is not a valid int"
            )
        return value

    def suggest_float(self, name: str, low: float, high: float) -> float:
        """Return a stored float only when it remains in the declared domain."""
        value = self._value(name)
        if type(value) not in (int, float) or not low <= float(value) <= high:
            raise StudyEvidenceError(
                f"stored trial parameter {name} is not a valid float"
            )
        return float(value)

    def suggest_categorical(self, name: str, choices: Sequence[str]) -> str:
        """Return a stored category only when it remains one declared choice."""
        value = self._value(name)
        if type(value) is not str or value not in choices:
            raise StudyEvidenceError(
                f"stored trial parameter {name} is not a valid category"
            )
        return value

    def _value(self, name: str) -> int | float | str:
        if name not in self.params:
            raise StudyEvidenceError(f"stored trial parameter {name} is missing")
        return self.params[name]


def _config_sha256_from_trial_params(trial: FrozenTrial) -> str:
    """Rebuild a validated HybridConfig and require its exact canonical params."""
    params = dict(trial.params)
    config = suggest_config(_StoredParameterTrial(params))
    if parameters_for_config(config) != params:
        raise StudyEvidenceError(
            f"trial {trial.number} stored parameters are not canonical "
            "config parameters"
        )
    return config_sha256(config)


def close_and_hash_storage(study: optuna.Study, paths: StudyPaths) -> str:
    """Checkpoint, dispose, fsync, and hash a terminal SQLite study database."""
    running = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
    if running:
        raise StudyEvidenceError("cannot close storage while trials remain RUNNING")
    if not paths.sqlite.is_file() or paths.sqlite.is_symlink():
        raise StudyEvidenceError("SQLite storage is not a real file")
    connection = sqlite3.connect(paths.sqlite)
    try:
        result = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if result is None or result[0] != 0:
            raise StudyEvidenceError("SQLite WAL checkpoint could not complete")
    finally:
        connection.close()
    storage = _optuna_49_private_storage(study)
    backend = getattr(storage, "_backend", storage)
    if not isinstance(backend, RDBStorage):
        raise StudyEvidenceError("study is not backed by direct RDBStorage")
    backend.engine.dispose()
    _fsync_file(paths.sqlite)
    for sidecar in (
        paths.sqlite.with_name(paths.sqlite.name + "-wal"),
        paths.sqlite.with_name(paths.sqlite.name + "-shm"),
    ):
        if sidecar.exists():
            raise StudyEvidenceError(
                f"SQLite sidecar remains after checkpoint: {sidecar.name}"
            )
    return hashlib.sha256(paths.sqlite.read_bytes()).hexdigest()


def _optuna_49_private_storage(study: optuna.Study) -> BaseStorage:
    """Return Optuna 4.9's storage bridge for APIs Study does not expose.

    Public ``get_trials`` and ``tell`` handle trial state. Optuna exposes no
    public API to set a *trial* system attribute or dispose a study engine, so
    this narrow, version-guarded bridge is limited to those two operations.
    """
    if optuna.__version__ != "4.9.0":
        raise RuntimeError("private storage bridge is pinned to Optuna 4.9.0")
    return study._storage  # noqa: SLF001


def _set_interruption_reason(study: optuna.Study, trial: FrozenTrial) -> None:
    """Set the unavailable trial system attribute through the versioned bridge."""
    storage = _optuna_49_private_storage(study)
    trial_id = storage.get_trial_id_from_study_id_trial_number(
        study._study_id,  # noqa: SLF001
        trial.number,
    )
    storage.set_trial_system_attr(trial_id, "interruption_reason", "process_restarted")


def _set_trial_user_attr(
    study: optuna.Study, trial: FrozenTrial, key: str, value: object
) -> None:
    """Restore a canonical trial summary through the version-pinned bridge."""
    storage = _optuna_49_private_storage(study)
    trial_id = storage.get_trial_id_from_study_id_trial_number(
        study._study_id,  # noqa: SLF001
        trial.number,
    )
    storage.set_trial_user_attr(trial_id, key, value)


def _validate_identity_inputs(
    frontier: VerifiedFrontier,
    snapshot: SnapshotLeague,
    report: FrontierReport,
    panels: PanelSet,
    weights: StrengthWeights,
    warm_starts: Sequence[HybridConfig],
) -> None:
    if not isinstance(frontier, VerifiedFrontier) or not isinstance(
        snapshot, SnapshotLeague
    ):
        raise ValueError("frontier and snapshot must be verified facts")
    if not isinstance(report, FrontierReport) or not isinstance(panels, PanelSet):
        raise ValueError("report and panels must be canonical protocol facts")
    if not isinstance(weights, StrengthWeights):
        raise ValueError("weights must be StrengthWeights")
    _validate_frontier_bindings(frontier, snapshot, report)
    _validate_protocol_bindings(report, panels, weights)
    _validate_warm_starts(warm_starts)
    if set(RUNG_3_SEEDS) & set(DETERMINISM_SEEDS):
        raise ValueError("development seed bank overlaps determinism seeds")


def _validate_frontier_bindings(
    frontier: VerifiedFrontier, snapshot: SnapshotLeague, report: FrontierReport
) -> None:
    """Require frontier, report, and snapshot facts to bind the same sources."""
    if frontier.engine != report.engine:
        raise ValueError("frontier and report engines differ")
    if frontier.manifest_sha256 != report.manifest_sha256:
        raise ValueError("frontier and report manifests differ")
    artifact_hashes = tuple(
        sorted((item.name, item.sha256) for item in frontier.artifacts)
    )
    if artifact_hashes != tuple(sorted(report.source_sha256.items())):
        raise ValueError("frontier artifacts and report source hashes differ")
    if {source.name for source in snapshot.sources} != set(frontier.opponents):
        raise ValueError("snapshot opponents differ from verified frontier")
    snapshot_hashes = tuple(
        sorted((source.name, source.sha256) for source in snapshot.sources)
    )
    if snapshot_hashes != artifact_hashes:
        raise ValueError("snapshot source hashes differ from verified frontier")


def _validate_protocol_bindings(
    report: FrontierReport, panels: PanelSet, weights: StrengthWeights
) -> None:
    """Require ranked panels and strength weights to describe one full league."""
    if tuple(row.name for row in report.rows) != panels.rung_3:
        raise ValueError("rung-three panel differs from frontier report ordering")
    if not set(panels.rung_1) <= set(panels.rung_2) <= set(panels.rung_3):
        raise ValueError("panels must be cumulative")
    if set(weights.as_dict()) != set(panels.rung_3):
        raise ValueError("strength weights differ from full frontier panel")


def _validate_warm_starts(warm_starts: Sequence[HybridConfig]) -> None:
    """Require unique validated start policies before sampler state can exist."""
    if not warm_starts or any(
        not isinstance(config, HybridConfig) for config in warm_starts
    ):
        raise ValueError("warm starts must be complete HybridConfig values")
    hashes = tuple(config_sha256(config) for config in warm_starts)
    if len(hashes) != len(set(hashes)):
        raise ValueError("warm-start configurations must be distinct")


def _validate_external_identity(
    snapshot: SnapshotLeague, legacy_state: Path, identity: StudyIdentity
) -> None:
    """Bind preflighted mutable paths to every matching immutable identity fact."""
    snapshots = tuple(
        sorted(
            (source.name, source.source_path, source.snapshot_path, source.sha256)
            for source in snapshot.sources
        )
    )
    if snapshots != identity.league_snapshots:
        raise StudyIdentityError("snapshot facts differ from study identity")
    if (
        hashlib.sha256(legacy_state.read_bytes()).hexdigest()
        != identity.legacy_state_sha256
    ):
        raise StudyIdentityError("legacy state SHA-256 differs from study identity")
    semantic_sources = evaluation_semantics()
    semantic_digest = hashlib.sha256(_canonical_json(semantic_sources)).hexdigest()
    if (
        semantic_sources != identity.evaluation_semantics
        or semantic_digest != identity.evaluation_semantics_sha256
    ):
        raise StudyIdentityError(
            "executable evaluation semantics differ from study identity"
        )


def _require_preflight(
    paths: StudyPaths,
    identity: StudyIdentity,
    preflight: StudyPreflight | None,
) -> None:
    """Require the capability produced by this module's external-byte preflight."""
    if (
        not isinstance(preflight, StudyPreflight)
        or preflight._capability is not _PREFLIGHT_CAPABILITY
        or preflight.paths != paths
        or preflight.identity_digest != identity.digest
    ):
        raise StudyIdentityError("open_study requires matching validated preflight")
    validate_study_paths(paths, preflight._snapshot, preflight._legacy_state, identity)


def _validate_runtime_optuna_version(identity: StudyIdentity) -> None:
    """Reject a dependency upgrade before it can alter sampler or storage state."""
    if optuna.__version__ != identity.optuna_version:
        raise StudyIdentityError(
            f"Optuna {optuna.__version__} differs from identity "
            f"{identity.optuna_version}"
        )


def _sampler_for(identity: StudyIdentity) -> TPESampler:
    """Reconstruct the exact identity-bound sequential TPE sampler."""
    return TPESampler(
        seed=identity.sampler.seed,
        multivariate=identity.sampler.multivariate,
        group=identity.sampler.group,
        n_startup_trials=identity.sampler.n_startup_trials,
    )


def _pruner_for(identity: StudyIdentity) -> SuccessiveHalvingPruner:
    """Reconstruct the exact identity-bound successive-halving pruner."""
    return SuccessiveHalvingPruner(
        min_resource=identity.pruner.min_resource,
        reduction_factor=identity.pruner.reduction_factor,
        min_early_stopping_rate=identity.pruner.min_early_stopping_rate,
    )


def _validate_sqlite_identity_readonly(paths: Path, identity: StudyIdentity) -> None:
    """Read the stored study identity through SQLite without opening RDBStorage."""
    connection = sqlite3.connect(f"file:{paths}?mode=ro", uri=True)
    try:
        row = connection.execute(
            "SELECT study_id FROM studies WHERE study_name = ?", (identity.study_name,)
        ).fetchone()
        if row is None:
            raise StudyIdentityError("SQLite study name differs from identity")
        attribute = connection.execute(
            "SELECT value_json FROM study_user_attributes "
            "WHERE study_id = ? AND key = ?",
            (row[0], _IDENTITY_ATTRIBUTE),
        ).fetchone()
        direction = connection.execute(
            "SELECT direction FROM study_directions WHERE study_id = ?", (row[0],)
        ).fetchall()
    except sqlite3.DatabaseError as error:
        raise StudyIdentityError("cannot read stored SQLite study identity") from error
    finally:
        connection.close()
    if attribute is None:
        raise StudyIdentityError("SQLite study identity attribute is missing")
    if direction != [("MAXIMIZE",)]:
        raise StudyIdentityError("SQLite study direction differs from MAXIMIZE")
    try:
        digest = json.loads(attribute[0])
    except json.JSONDecodeError as error:
        raise StudyIdentityError(
            "SQLite study identity attribute is invalid"
        ) from error
    if digest != identity.digest:
        raise StudyIdentityError("study_identity_sha256 differs from identity.json")


def _validate_layout(paths: StudyPaths) -> None:
    expected = StudyPaths.from_root(paths.root)
    if paths != expected:
        raise ValueError("study paths must use the canonical fixed layout")
    for path in _study_output_paths(paths):
        _reject_symlink_components(_absolute(path))


def validate_create_root(paths: StudyPaths) -> tuple[Path, ...]:
    """Accept only an absent, empty, or protocol-temporary-only create root."""
    _validate_layout(paths)
    root = _absolute(paths.root)
    if not root.exists():
        return ()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("create root must be a real directory")
    temporaries: list[Path] = []
    expected_source: bytes | None = None
    for entry in sorted(root.iterdir()):
        status = entry.lstat()
        if (
            _IDENTITY_TEMPORARY_NAME.fullmatch(entry.name) is None
            or entry.is_symlink()
            or not stat.S_ISREG(status.st_mode)
            or status.st_nlink != 1
        ):
            raise ValueError("create root contains a non-recoverable entry")
        source = entry.read_bytes()
        try:
            stored = StudyIdentity.model_validate_json(source)
        except ValidationError as error:
            raise ValueError(
                "identity temporary is not a strict StudyIdentity"
            ) from error
        if source != _canonical_json(stored.model_dump(mode="json")):
            raise ValueError("identity temporary is not canonical JSON")
        if expected_source is not None and source != expected_source:
            raise ValueError("identity temporaries disagree")
        expected_source = source
        temporaries.append(entry)
    return tuple(temporaries)


def _study_output_paths(paths: StudyPaths) -> tuple[Path, ...]:
    """Return every fixed output that must remain disjoint from protected bytes."""
    return (
        paths.root,
        paths.sqlite,
        paths.identity,
        paths.evidence,
        paths.quarantine,
        paths.diagnostic,
        paths.finalists,
    )


def _remove_recoverable_identity_temporaries(
    paths: StudyPaths, identity: StudyIdentity
) -> None:
    """Remove only exact durable identity temporaries from a crashed first open."""
    temporaries = validate_create_root(paths)
    expected = _canonical_json(identity.model_dump(mode="json"))
    for temporary in temporaries:
        if temporary.read_bytes() != expected:
            raise StudyIdentityError(
                "identity temporary differs from requested study identity"
            )
    for temporary in temporaries:
        temporary.unlink()
    if temporaries:
        _fsync_directory(_absolute(paths.root))


def _create_run_root(paths: StudyPaths) -> None:
    root = _absolute(paths.root)
    if root.exists():
        if not root.is_dir() or root.is_symlink():
            raise ValueError("study root must be a real directory")
        if any(root.iterdir()):
            raise ValueError("new study root must be empty")
        return
    root.mkdir(parents=True)
    _fsync_directory(root.parent)


def _write_identity(path: Path, identity: StudyIdentity) -> None:
    source = _canonical_json(identity.model_dump(mode="json"))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".identity.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(source)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_stored_identity(path: Path, requested: StudyIdentity) -> None:
    if not path.is_file() or path.is_symlink():
        raise StudyIdentityError("identity.json is not a real file")
    source = path.read_bytes()
    try:
        stored = StudyIdentity.model_validate_json(source)
    except ValidationError as error:
        raise StudyIdentityError(
            "identity.json is not a valid strict StudyIdentity"
        ) from error
    if source != _canonical_json(stored.model_dump(mode="json")):
        raise StudyIdentityError("identity.json is not canonical JSON")
    if stored != requested:
        for field in StudyIdentity.model_fields:
            if getattr(stored, field) != getattr(requested, field):
                raise StudyIdentityError(f"{field} differs from stored study identity")
        raise StudyIdentityError("stored study identity differs")


def _read_evidence(path: Path, paths: StudyPaths) -> tuple[RungEvidence, str]:
    if not path.is_file() or path.is_symlink() or path.suffix != ".json":
        raise StudyEvidenceError(f"unexpected evidence entry {path.name}")
    try:
        source = path.read_bytes()
        evidence = RungEvidence.model_validate_json(source)
    except (ValidationError, ValueError) as error:
        matched = re.fullmatch(r"trial-(\d+)-rung-([123])\.json", path.name)
        if matched is not None:
            raise StudyEvidenceError(
                f"trial {matched.group(1)} rung {matched.group(2)} has invalid evidence"
            ) from error
        raise StudyEvidenceError(f"invalid evidence file {path.name}") from error
    expected = evidence_path(paths.root, evidence.trial_number, evidence.rung)
    if path != expected:
        raise StudyEvidenceError(f"evidence file {path.name} has a non-canonical path")
    if source != _canonical_json(evidence.model_dump(mode="json")) + b"\n":
        raise StudyEvidenceError(f"evidence file {path.name} is not canonical JSON")
    return evidence, hashlib.sha256(source).hexdigest()


def _validate_evidence_identity(
    evidence: RungEvidence, identity: StudyIdentity, number: int
) -> None:
    expected_panels = {
        1: identity.panels.rung_1,
        2: identity.panels.rung_2,
        3: identity.panels.rung_3,
    }
    expected_seeds = {
        1: identity.seed_banks.rung_1,
        2: identity.seed_banks.rung_2,
        3: identity.seed_banks.rung_3,
    }
    if evidence.opponents != expected_panels[evidence.rung]:
        raise StudyEvidenceError(
            f"trial {number} rung {evidence.rung} panel differs from identity"
        )
    if evidence.seeds != expected_seeds[evidence.rung]:
        raise StudyEvidenceError(
            f"trial {number} rung {evidence.rung} seeds differ from identity"
        )
    identity_weights = dict(identity.strength_weights)
    expected_weights = tuple(
        (name, identity_weights[name]) for name in evidence.opponents
    )
    if evidence.strength_weights != expected_weights:
        raise StudyEvidenceError(
            f"trial {number} rung {evidence.rung} weights differ from identity"
        )


def _validate_trial_summary(
    trial: FrozenTrial,
    evidence: RungEvidence,
    digest: str,
    paths: StudyPaths,
) -> None:
    attributes = trial.user_attrs
    path = evidence_path(paths.root, evidence.trial_number, evidence.rung)
    expected = rung_summary_attributes(evidence, path, digest, paths.root)
    for key, value in expected.items():
        if attributes.get(key, object()) != value:
            raise StudyEvidenceError(
                f"trial {trial.number} rung {evidence.rung} {key} "
                "differs from canonical evidence"
            )


def _canonical_json(value: object) -> bytes:
    """Serialize a JSON value in its one allowed content-addressed form."""
    return json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()


def _sqlite_url(path: Path) -> str:
    """Return the local SQLite URL for an already preflighted absolute path."""
    return f"sqlite:///{_absolute(path)}"


def _required_sha256(value: str | None, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} SHA-256 must be lowercase hexadecimal")
    return value


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _canonical_lexical_path(path: Path) -> Path:
    """Normalize ``.``/``..`` without resolving symlinks or requiring existence."""
    return Path(os.path.normpath(str(_absolute(path))))


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if not current.exists() and not current.is_symlink():
            return
        if current.is_symlink():
            raise ValueError(f"study paths cannot traverse symlinks: {current}")


def _reject_protected_path(candidate: Path, protected: Path) -> None:
    if _is_descendant(candidate, protected) or _is_descendant(protected, candidate):
        raise ValueError("study path aliases protected snapshot or legacy state")


def _is_descendant(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return True


def _same_inode(left: Path, right: Path) -> bool:
    try:
        return left.samefile(right)
    except OSError:
        return False


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
