"""Authoritative certified finalists for the terminal hybrid Optuna study."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Self, cast

import optuna
from optuna.storages import RDBStorage
from optuna.trial import FixedTrial, FrozenTrial, TrialState
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from kaggriculture.hybrid.config import HybridConfig
from kaggriculture.search.evolution import PROMOTION_SEEDS
from kaggriculture.search.fitness import StrengthWeights
from kaggriculture.search.optuna_protocol import (
    RUNG_1_SEEDS,
    RUNG_2_SEEDS,
    RUNG_3_SEEDS,
    RungEvidence,
    evidence_path,
    score_evidence,
)
from kaggriculture.search.optuna_space import (
    SPACE_SHA256,
    config_sha256,
    parameters_for_config,
    suggest_config,
)
from kaggriculture.search.optuna_state import (
    StudyEvidenceError,
    StudyIdentity,
    StudyPaths,
    close_and_hash_storage,
    terminal_counts,
    validate_study_evidence,
)

_FINALIST_COUNT = 4


class OptunaFinalist(BaseModel):
    """One fully recomputed clean rung-three promotion candidate."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, allow_inf_nan=False
    )

    trial_number: int = Field(ge=0)
    rung: Literal[3]
    parameters: dict[str, int | float | str]
    config: HybridConfig
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    primary: float = Field(ge=0.0, le=1.0)
    dense_margin: float = Field(ge=-1.0, le=1.0)
    objective: float = Field(ge=0.0, le=1.000001)
    evidence_path: str
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    failures: tuple[str, ...] = ()


class OptunaFinalistArtifact(BaseModel):
    """A closed-SQLite-bound terminal study certification."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, populate_by_name=True
    )

    schema_value: Literal["optuna-finalists-v1"] = Field(
        default="optuna-finalists-v1", alias="schema"
    )
    study_identity: StudyIdentity
    study_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sqlite_path: str
    sqlite_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trial_counts: dict[str, int]
    promotion_seeds_used: Literal[False] = False
    finalists: tuple[OptunaFinalist, ...] = Field(
        min_length=_FINALIST_COUNT, max_length=_FINALIST_COUNT
    )
    integrity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def schema(self) -> Literal["optuna-finalists-v1"]:
        """Expose the serialized schema discriminator under its public name."""
        return self.schema_value

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load and independently revalidate SQLite, evidence, and snapshots."""
        source = _read_regular_file(path, "finalist artifact")
        try:
            payload = _strict_json_object(source)
            artifact = cls.model_validate_json(source)
        except (ValidationError, ValueError) as error:
            raise ValueError(f"invalid Optuna finalist artifact: {error}") from error
        if source != _canonical_line(artifact.model_dump(mode="json", by_alias=True)):
            raise ValueError("Optuna finalist artifact is not canonical JSON")
        expected_integrity = _artifact_integrity(payload)
        if artifact.integrity_sha256 != expected_integrity:
            raise ValueError("finalist artifact integrity digest differs")
        _validate_loaded_artifact(path, artifact)
        return artifact


def write_optuna_finalists(
    study: optuna.Study,
    identity: StudyIdentity,
    paths: StudyPaths,
    count: int = 4,
) -> OptunaFinalistArtifact:
    """Certify exactly one validated 512-terminal-trial Optuna study."""
    if type(count) is not int or count != _FINALIST_COUNT:
        raise ValueError("finalist count must be exactly 4")
    if paths != StudyPaths.from_root(paths.root):
        raise ValueError("study paths must use the canonical fixed layout")
    identity = StudyIdentity.model_validate(identity.model_dump())
    running = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
    if running:
        raise ValueError("finalists require no RUNNING trials")
    counts = terminal_counts(study)
    terminal = sum(counts.values())
    if terminal != identity.maximum_trials or terminal != 512:
        raise ValueError("finalists require exactly 512 terminal trials")
    validate_study_evidence(study, paths, identity)
    _validate_semantic_identity(identity)
    _validate_study_root(paths.root, identity)
    _validate_identity_files(paths, identity)
    _validate_snapshot_sources(identity)
    _require_economic_gate(study, paths)
    finalists = _recomputed_finalists(study, identity, paths)
    sqlite_sha256 = close_and_hash_storage(study, paths)
    payload: dict[str, object] = {
        "schema": "optuna-finalists-v1",
        "study_identity": identity.model_dump(mode="json"),
        "study_identity_sha256": identity.digest,
        "sqlite_path": str(paths.sqlite.relative_to(paths.root)),
        "sqlite_sha256": sqlite_sha256,
        "trial_counts": counts,
        "promotion_seeds_used": False,
        "finalists": [row.model_dump(mode="json") for row in finalists],
    }
    payload["integrity_sha256"] = _artifact_integrity(payload)
    _write_atomic(paths.finalists, _canonical_line(payload))
    return OptunaFinalistArtifact.load(paths.finalists)


def _validate_loaded_artifact(  # noqa: C901
    path: Path, artifact: OptunaFinalistArtifact
) -> None:
    paths = StudyPaths.from_root(path.parent)
    if path != paths.finalists:
        raise ValueError("finalist artifact path is not canonical")
    if artifact.sqlite_path != str(paths.sqlite.relative_to(paths.root)):
        raise ValueError("SQLite path differs from canonical study layout")
    if artifact.study_identity_sha256 != artifact.study_identity.digest:
        raise ValueError("study identity digest differs from embedded identity")
    _validate_semantic_identity(artifact.study_identity)
    _validate_study_root(paths.root, artifact.study_identity)
    _validate_identity_files(paths, artifact.study_identity)
    _validate_snapshot_sources(artifact.study_identity)
    sqlite_source = _read_regular_file(paths.sqlite, "SQLite storage")
    for suffix in ("-wal", "-shm"):
        if paths.sqlite.with_name(paths.sqlite.name + suffix).exists():
            raise ValueError("closed SQLite storage has a sidecar")
    if hashlib.sha256(sqlite_source).hexdigest() != artifact.sqlite_sha256:
        raise ValueError("SQLite digest differs from certified closed storage")
    with tempfile.TemporaryDirectory(prefix="kaggriculture-finalists-load-") as name:
        copy = Path(name) / "study.sqlite3"
        shutil.copyfile(paths.sqlite, copy)
        try:
            study = optuna.load_study(
                storage=RDBStorage(f"sqlite:///{copy.absolute()}"),
                study_name=artifact.study_identity.study_name,
            )
            validate_study_evidence(study, paths, artifact.study_identity)
            _validate_terminal_shape(study, artifact)
            _require_economic_gate(study, paths)
            expected = _recomputed_finalists(study, artifact.study_identity, paths)
        except (KeyError, OSError, RuntimeError, ValidationError, ValueError) as error:
            raise ValueError(f"certified trial state is invalid: {error}") from error
        finally:
            try:
                storage = cast(object, locals().get("study"))
                backend = getattr(getattr(storage, "_storage", None), "_backend", None)
                engine = getattr(backend, "engine", None)
                if engine is not None:
                    engine.dispose()
            except Exception:
                pass
    if artifact.finalists != expected:
        raise ValueError("finalist rank or certified fields differ from recomputation")


def _validate_terminal_shape(
    study: optuna.Study, artifact: OptunaFinalistArtifact
) -> None:
    if study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        raise ValueError("trial state contains RUNNING rows")
    counts = terminal_counts(study)
    if counts != artifact.trial_counts:
        raise ValueError("trial state counts differ from finalist artifact")
    if set(counts) != {"complete", "pruned", "failed"}:
        raise ValueError("trial state count keys are invalid")
    if sum(counts.values()) != artifact.study_identity.maximum_trials:
        raise ValueError("trial state does not contain 512 terminal rows")


def _recomputed_finalists(
    study: optuna.Study,
    identity: StudyIdentity,
    paths: StudyPaths,
) -> tuple[OptunaFinalist, ...]:
    weights = _identity_weights(identity)
    ranked: list[tuple[OptunaFinalist, float]] = []
    for trial in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)):
        if trial.user_attrs.get("rung") != 3:
            continue
        row, runtime = _finalist_from_trial(trial, paths, weights)
        ranked.append((row, runtime))
    ranked.sort(
        key=lambda item: (
            -item[0].primary,
            -item[0].dense_margin,
            item[1],
            item[0].config_sha256,
        )
    )
    if len(ranked) < _FINALIST_COUNT:
        raise ValueError("finalists require at least four clean rung-three trials")
    return tuple(row for row, _ in ranked[:_FINALIST_COUNT])


def _finalist_from_trial(
    trial: FrozenTrial,
    paths: StudyPaths,
    weights: StrengthWeights,
) -> tuple[OptunaFinalist, float]:
    parameters = dict(trial.params)
    try:
        config = suggest_config(FixedTrial(parameters))
    except (KeyError, ValidationError, ValueError) as error:
        raise StudyEvidenceError(
            f"trial {trial.number} config parameters are invalid"
        ) from error
    if parameters_for_config(config) != parameters:
        raise StudyEvidenceError(
            f"trial {trial.number} config parameters are not canonical"
        )
    digest = config_sha256(config)
    relative = trial.user_attrs.get("evidence_path")
    expected_path = evidence_path(paths.root, trial.number, 3)
    if type(relative) is not str or relative != str(
        expected_path.relative_to(paths.root)
    ):
        raise StudyEvidenceError(f"trial {trial.number} evidence path differs")
    source = _read_regular_file(expected_path, "rung evidence")
    if hashlib.sha256(source).hexdigest() != trial.user_attrs.get("evidence_sha256"):
        raise StudyEvidenceError(f"trial {trial.number} rung evidence digest differs")
    try:
        evidence = RungEvidence.model_validate_json(source)
    except ValidationError as error:
        raise StudyEvidenceError(
            f"trial {trial.number} rung evidence is invalid"
        ) from error
    if source != _canonical_line(evidence.model_dump(mode="json")):
        raise StudyEvidenceError(f"trial {trial.number} rung evidence is not canonical")
    if evidence.rung != 3 or evidence.failures:
        raise StudyEvidenceError(
            f"trial {trial.number} is not clean rung-three evidence"
        )
    if evidence.config_sha256 != digest:
        raise StudyEvidenceError(f"trial {trial.number} config differs from evidence")
    score = score_evidence(evidence, weights)
    if trial.value != score.objective:
        raise StudyEvidenceError(
            f"trial {trial.number} objective differs from evidence"
        )
    runtime = sum(game.runtime_seconds for game in evidence.games)
    row = OptunaFinalist(
        trial_number=trial.number,
        rung=3,
        parameters=parameters,
        config=config,
        config_sha256=digest,
        primary=score.primary,
        dense_margin=score.dense_margin,
        objective=score.objective,
        evidence_path=relative,
        evidence_sha256=hashlib.sha256(source).hexdigest(),
        failures=(),
    )
    return row, runtime


def _require_economic_gate(study: optuna.Study, paths: StudyPaths) -> None:
    trials = tuple(
        trial
        for trial in study.get_trials(deepcopy=False)
        if trial.state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)
    )
    first = trials[:128]
    if len(first) != 128:
        raise ValueError("economic gate requires the first 128 terminal trials")
    passed = False
    for trial in first:
        rung = trial.user_attrs.get("rung")
        if type(rung) is not int or rung not in (1, 2, 3):
            continue
        path = evidence_path(paths.root, trial.number, rung)
        if not path.exists():
            continue
        evidence = RungEvidence.model_validate_json(path.read_bytes())
        economic = evidence.matchups.get("economic_policy")
        if not evidence.failures and economic is not None and economic.win_points > 0.0:
            passed = True
            break
    if not passed:
        raise ValueError("128-trial economic gate did not pass")
    if paths.diagnostic.exists():
        raise ValueError("economic diagnostic exists for a finalist-producing study")


def _identity_weights(identity: StudyIdentity) -> StrengthWeights:
    weights = StrengthWeights(dict(identity.strength_weights))
    if tuple(weights.as_dict()) != tuple(name for name, _ in identity.strength_weights):
        raise ValueError("study strength weights are not canonical")
    return weights


def _validate_identity_files(paths: StudyPaths, identity: StudyIdentity) -> None:
    source = _read_regular_file(paths.identity, "study identity")
    if (
        source
        != json.dumps(
            identity.model_dump(mode="json"),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ):
        raise ValueError("study identity bytes differ from embedded identity")


def _validate_snapshot_sources(identity: StudyIdentity) -> None:  # noqa: C901
    source_hashes = dict(identity.source_sha256)
    if len(source_hashes) != len(identity.source_sha256):
        raise ValueError("source snapshot names are duplicated")
    snapshots = identity.league_snapshots
    if identity.source_sha256 != tuple(
        sorted(identity.source_sha256)
    ) or snapshots != tuple(sorted(snapshots)):
        raise ValueError("source snapshot facts are not canonical")
    if not snapshots or tuple(row[0] for row in snapshots) != tuple(source_hashes):
        raise ValueError("source snapshot names differ from study identity")
    if len({row[1] for row in snapshots}) != len(snapshots) or len(
        {row[2] for row in snapshots}
    ) != len(snapshots):
        raise ValueError("source snapshot paths must be unique")
    roots = {Path(row[2]).parent for row in snapshots}
    if len(roots) != 1:
        raise ValueError("source snapshots have multiple roots")
    root = roots.pop()
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("source snapshot root is invalid")
    expected_paths = {Path(row[2]) for row in snapshots}
    if set(root.iterdir()) != expected_paths:
        raise ValueError("source snapshot paths differ from identity")
    for name, source_name, snapshot_name, digest in snapshots:
        if source_hashes.get(name) != digest:
            raise ValueError("source snapshot digest differs from source identity")
        source = Path(source_name)
        snapshot = Path(snapshot_name)
        if not source.is_absolute() or not snapshot.is_absolute():
            raise ValueError("source snapshot paths must be absolute")
        if snapshot.parent != root or digest not in snapshot.name:
            raise ValueError("source snapshot path is not content-addressed")
        _require_digest(source, digest, "source snapshot original")
        _require_digest(snapshot, digest, "source snapshot")
        status = snapshot.lstat()
        if (
            snapshot.is_symlink()
            or not stat.S_ISREG(status.st_mode)
            or status.st_nlink != 1
            or status.st_mode & 0o222
        ):
            raise ValueError("source snapshot must be read-only and unaliased")


def _require_digest(path: Path, digest: str, label: str) -> None:
    source = _read_regular_file(path, label)
    if hashlib.sha256(source).hexdigest() != digest:
        raise ValueError(f"{label} SHA-256 differs from identity")


def _read_regular_file(path: Path, label: str) -> bytes:
    _reject_symlink_components(path.absolute())
    try:
        status = path.lstat()
        source = path.read_bytes()
    except OSError as error:
        raise ValueError(f"{label} is missing or unreadable") from error
    if path.is_symlink() or not stat.S_ISREG(status.st_mode):
        raise ValueError(f"{label} must be a regular file")
    return source


def _validate_semantic_identity(identity: StudyIdentity) -> None:
    """Pin every protocol constant not derived from SQLite evidence itself."""
    if identity.space_sha256 != SPACE_SHA256:
        raise ValueError("parameter-space digest differs from the production space")
    if (
        identity.seed_banks.rung_1 != RUNG_1_SEEDS
        or identity.seed_banks.rung_2 != RUNG_2_SEEDS
        or identity.seed_banks.rung_3 != RUNG_3_SEEDS
    ):
        raise ValueError("search seed count or values differ from the fixed banks")
    promotion_digest = hashlib.sha256(
        json.dumps(
            PROMOTION_SEEDS,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if identity.seed_banks.protected_promotion_sha256 != promotion_digest:
        raise ValueError("protected promotion seed digest differs")


def _reject_symlink_components(path: Path) -> None:
    """Reject a symlink at any existing component, not just the final file."""
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"certified path traverses symlink: {current}")
        if not current.exists():
            return


def _validate_study_root(root: Path, identity: StudyIdentity) -> None:
    """Keep mutable study files outside every source and immutable snapshot."""
    study_root = root.absolute()
    _reject_symlink_components(study_root)
    for _name, source_name, snapshot_name, _digest in identity.league_snapshots:
        for protected in (Path(source_name), Path(snapshot_name)):
            if _same_or_below(study_root, protected) or _same_or_below(
                protected, study_root
            ):
                raise ValueError("study path aliases source snapshot provenance")


def _same_or_below(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _strict_json_object(source: bytes) -> dict[str, object]:
    def reject_duplicates(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key is forbidden: {key}")
            result[key] = value
        return result

    value = json.loads(
        source,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant is forbidden: {token}")
        ),
        object_pairs_hook=reject_duplicates,
    )
    if type(value) is not dict:
        raise ValueError("JSON artifact must be an object")
    return value


def _artifact_integrity(payload: Mapping[str, object]) -> str:
    body = {key: value for key, value in payload.items() if key != "integrity_sha256"}
    return hashlib.sha256(_canonical_line(body)).hexdigest()


def _canonical_line(payload: object) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()


def _write_atomic(path: Path, source: bytes) -> None:
    if path.is_symlink() or path.exists() and not path.is_file():
        raise ValueError("finalist output must be a regular file")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(source)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)  # noqa: PTH105
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
