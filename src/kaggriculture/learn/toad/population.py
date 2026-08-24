"""Durable, verified snapshots for frozen Toad opponents."""

from __future__ import annotations

import hashlib
import io
import os
import pickle
import random
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    NonNegativeInt,
    TypeAdapter,
    field_serializer,
    field_validator,
)

from kaggriculture.learn.model import Policy
from kaggriculture.learn.toad.config import ModelConfig, TeacherSpec

_EVALUATION = TypeAdapter(dict[str, FiniteFloat])
_SNAPSHOT_NAME = re.compile(r"^snapshot-(\d{12,})-([0-9a-f]{12})\.pt$")


class SnapshotIntegrityError(RuntimeError):
    """A snapshot cannot safely be used as the selected frozen opponent."""


class EmptySnapshotPoolError(RuntimeError):
    """A collection requested a frozen opponent before any snapshot existed."""


class TeacherCompatibilityError(RuntimeError):
    """A teacher checkpoint does not satisfy its immutable declared contract."""


TeacherHead = Literal["operation", "quantity", "market", "value"]
_TEACHER_HEAD_PREFIXES: dict[TeacherHead, str] = {
    "operation": "head.",
    "quantity": "quantity_head.",
    "market": "trade_head.",
    "value": "value.",
}
_POLICY_KEY_PREFIXES = (
    "stem.",
    "market.",
    "blocks.",
    *tuple(_TEACHER_HEAD_PREFIXES.values()),
)


@dataclass(frozen=True)
class LoadedTeacher:
    """One verified, frozen teacher and its confirmed effective contract."""

    policy: Policy
    spec: TeacherSpec
    model: ModelConfig
    actual_heads: tuple[TeacherHead, ...]

    def __post_init__(self) -> None:
        """Defend the confirmed canonical head contract at every construction seam."""
        expected = tuple(
            head for head in _TEACHER_HEAD_PREFIXES if getattr(self.spec, head)
        )
        if self.actual_heads != expected:
            raise ValueError(
                "LoadedTeacher actual_heads must exactly match the canonical "
                "declared head contract"
            )
        if self.policy.training or any(
            parameter.requires_grad for parameter in self.policy.parameters()
        ):
            raise ValueError("LoadedTeacher policy must be frozen in evaluation mode")

    def metadata(self) -> dict[str, object]:
        """Return the complete path, digest, topology, and head contract."""
        return {
            "present": True,
            "checkpoint": str(self.spec.checkpoint),
            "sha256": self.spec.sha256,
            "topology": {
                "blocks": self.model.blocks,
                "channels": self.model.channels,
                "kernel_size": self.model.kernel_size,
                "activation": self.model.activation,
                "value_bound": self.model.value_bound,
            },
            "declared_heads": {
                head: getattr(self.spec, head) for head in _TEACHER_HEAD_PREFIXES
            },
            "actual_heads": list(self.actual_heads),
        }


def _teacher_policy_weights(  # noqa: C901
    checkpoint: object,
) -> dict[str, torch.Tensor]:
    """Extract named policy tensors from one accepted checkpoint envelope."""
    if not isinstance(checkpoint, Mapping):
        raise TeacherCompatibilityError("teacher checkpoint must contain a mapping")
    typed_checkpoint = cast(Mapping[object, object], checkpoint)
    if "state_dict" in typed_checkpoint and "learner" in typed_checkpoint:
        raise TeacherCompatibilityError(
            "teacher checkpoint has mixed state_dict and learner layouts"
        )
    envelope = "state_dict" in typed_checkpoint or "learner" in typed_checkpoint
    bare_policy_keys = [
        name
        for name in typed_checkpoint
        if isinstance(name, str) and name.startswith(_POLICY_KEY_PREFIXES)
    ]
    if envelope and bare_policy_keys:
        raise TeacherCompatibilityError(
            "teacher checkpoint has mixed bare and envelope layouts"
        )
    candidate: object
    if "state_dict" in typed_checkpoint:
        state_dict = typed_checkpoint["state_dict"]
        if not isinstance(state_dict, Mapping):
            raise TeacherCompatibilityError(
                "teacher Lightning checkpoint state_dict must be a mapping"
            )
        has_prefixed_policy = any(
            isinstance(name, str) and name.startswith("policy.") for name in state_dict
        )
        has_bare_policy = any(
            isinstance(name, str) and name.startswith(_POLICY_KEY_PREFIXES)
            for name in state_dict
        )
        if has_prefixed_policy and has_bare_policy:
            raise TeacherCompatibilityError(
                "teacher checkpoint has mixed policy-prefixed and bare layouts"
            )
        candidate = {
            name.removeprefix("policy."): value
            for name, value in state_dict.items()
            if isinstance(name, str) and name.startswith("policy.")
        }
        if not candidate:
            raise TeacherCompatibilityError(
                "teacher Lightning checkpoint has no policy.* weights"
            )
    elif "learner" in typed_checkpoint:
        candidate = typed_checkpoint["learner"]
        if not isinstance(candidate, Mapping):
            raise TeacherCompatibilityError(
                "teacher learner checkpoint must be a mapping"
            )
    else:
        candidate = typed_checkpoint
    if not isinstance(candidate, Mapping) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in candidate.items()
    ):
        raise TeacherCompatibilityError(
            "teacher checkpoint policy must contain only named tensors"
        )
    return cast(dict[str, torch.Tensor], dict(candidate))


def resolve_teacher_model(spec: TeacherSpec, student: ModelConfig) -> ModelConfig:
    """Return the explicit bare control topology for one teacher contract."""
    return ModelConfig.control(
        blocks=spec.blocks or student.blocks,
        channels=student.channels,
    ).model_copy(update={"value_bound": student.value_bound})


def load_teacher_from_state(  # noqa: C901
    spec: TeacherSpec,
    model_config: ModelConfig,
    state: Mapping[str, torch.Tensor],
    *,
    sha256: str,
) -> LoadedTeacher:
    """Validate cached bare tensors under one independently confirmed binding."""
    digest = sha256
    if spec.sha256 is not None and digest != spec.sha256:
        raise TeacherCompatibilityError(
            "teacher digest does not match TeacherSpec: "
            f"expected {spec.sha256}, found {digest}"
        )
    resolved_model = resolve_teacher_model(spec, model_config)
    policy = Policy(
        blocks=resolved_model.blocks,
        channels=resolved_model.channels,
        value_bound=resolved_model.value_bound,
        kernel_size=resolved_model.kernel_size,
        activation=resolved_model.activation,
    )
    weights = _teacher_policy_weights(state)
    expected = policy.state_dict()
    expected_keys = set(expected)
    actual_keys = set(weights)
    unexpected = sorted(actual_keys - expected_keys)
    if unexpected:
        raise TeacherCompatibilityError(f"teacher has unexpected keys: {unexpected}")

    actual_heads: list[TeacherHead] = []
    allowed_missing: set[str] = set()
    for head, prefix in _TEACHER_HEAD_PREFIXES.items():
        family = {name for name in expected_keys if name.startswith(prefix)}
        present = family.intersection(actual_keys)
        if present and present != family:
            missing = sorted(family - present)
            raise TeacherCompatibilityError(
                f"teacher has incomplete {prefix.removesuffix('.')} head: {missing}"
            )
        if present == family and getattr(spec, head):
            actual_heads.append(head)
        if getattr(spec, head):
            if present != family:
                raise TeacherCompatibilityError(
                    f"teacher is missing declared {prefix.removesuffix('.')} keys: "
                    f"{sorted(family - present)}"
                )
        else:
            allowed_missing.update(family)

    missing_shared = sorted((expected_keys - actual_keys) - allowed_missing)
    if missing_shared:
        raise TeacherCompatibilityError(
            f"teacher is missing shared or declared keys: {missing_shared}"
        )
    mismatched_shapes = sorted(
        name
        for name, value in weights.items()
        if tuple(value.shape) != tuple(expected[name].shape)
    )
    if mismatched_shapes:
        rendered = ", ".join(
            f"{name}: checkpoint={tuple(weights[name].shape)}, "
            f"expected={tuple(expected[name].shape)}"
            for name in mismatched_shapes
        )
        raise TeacherCompatibilityError(f"teacher tensor shape mismatch: {rendered}")
    policy.load_state_dict(weights, strict=False)
    policy.requires_grad_(False).eval()
    effective_spec = spec.model_copy(
        update={"sha256": digest, "blocks": resolved_model.blocks}
    )
    return LoadedTeacher(
        policy=policy,
        spec=effective_spec,
        model=resolved_model,
        actual_heads=tuple(actual_heads),
    )


def load_teacher(spec: TeacherSpec, model_config: ModelConfig) -> LoadedTeacher:
    """Load exactly one checkpoint and enforce shared and declared-head strictness."""
    try:
        serialized = spec.checkpoint.read_bytes()
    except OSError as error:
        raise TeacherCompatibilityError(
            f"teacher digest verification failed: {spec.checkpoint}"
        ) from error
    digest = hashlib.sha256(serialized).hexdigest()
    if spec.sha256 is not None and digest != spec.sha256:
        raise TeacherCompatibilityError(
            "teacher digest does not match TeacherSpec: "
            f"expected {spec.sha256}, found {digest}"
        )
    try:
        checkpoint = torch.load(
            io.BytesIO(serialized),
            map_location="cpu",
            weights_only=True,
        )
    except (
        OSError,
        RuntimeError,
        ValueError,
        EOFError,
        pickle.UnpicklingError,
    ) as error:
        raise TeacherCompatibilityError(
            f"teacher checkpoint cannot be loaded: {spec.checkpoint}"
        ) from error
    return load_teacher_from_state(
        spec,
        model_config,
        _teacher_policy_weights(checkpoint),
        sha256=digest,
    )


class SnapshotEntry(BaseModel):
    """Immutable provenance for one content-addressed policy state dictionary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: Path
    sha256: str
    structure: str
    environment_steps: NonNegativeInt
    round_id: NonNegativeInt
    run_id: str
    created_at: datetime
    evaluation: Mapping[str, FiniteFloat] = Field(
        default_factory=dict,
        validate_default=True,
    )

    @field_validator("evaluation")
    @classmethod
    def freeze_evaluation(
        cls, evaluation: Mapping[str, FiniteFloat]
    ) -> Mapping[str, FiniteFloat]:
        """Copy evaluation metadata into an immutable defensive mapping."""
        return MappingProxyType(dict(evaluation))

    @field_serializer("evaluation")
    def serialize_evaluation(
        self, evaluation: Mapping[str, FiniteFloat]
    ) -> dict[str, float]:
        """Keep the immutable mapping's durable JSON representation ordinary."""
        return {name: float(value) for name, value in evaluation.items()}


class SnapshotManifest(BaseModel):
    """The atomically published members of one frozen-opponent population."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entries: tuple[SnapshotEntry, ...] = ()

    @classmethod
    def load(cls, path: Path) -> SnapshotManifest:
        """Deserialize one manifest without accepting unrecognised fields."""
        return cls.model_validate_json(path.read_text())


class SnapshotPoolIdentity(BaseModel):
    """Checkpoint-safe identity for reopening one exact verified store view."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    directory: Path
    structure: str
    seed: int
    capacity: int
    manifest: SnapshotManifest


def sha256_file(path: Path) -> str:
    """Return the digest of bytes currently stored at ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as snapshot:
        for chunk in iter(lambda: snapshot.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SnapshotStore:
    """Store immutable snapshots and publish bounded manifests atomically."""

    def __init__(self, directory: Path, *, capacity: int, structure: str) -> None:
        if capacity < 1:
            raise ValueError("snapshot capacity must be positive")
        self.directory = directory.resolve()
        self.capacity = capacity
        self.structure = structure
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.directory / "manifest.json"
        self.manifest = (
            SnapshotManifest.load(self.manifest_path)
            if self.manifest_path.is_file()
            else SnapshotManifest()
        )

    @classmethod
    def open_generation(
        cls,
        directory: Path,
        *,
        capacity: int,
        structure: str,
        manifest: SnapshotManifest,
    ) -> SnapshotStore:
        """Open one immutable checkpointed manifest, even after the head advances."""
        store = cls(directory, capacity=capacity, structure=structure)
        if store.manifest == manifest:
            store._validate_existing_members()
            return store
        generation_path = store._generation_path(manifest)
        if not generation_path.is_file():
            raise SnapshotIntegrityError(
                "checkpointed snapshot manifest generation is not durable"
            )
        try:
            durable = SnapshotManifest.load(generation_path)
        except (OSError, ValueError) as error:
            raise SnapshotIntegrityError(
                "checkpointed snapshot manifest generation is corrupt"
            ) from error
        if durable != manifest:
            raise SnapshotIntegrityError(
                "checkpointed snapshot manifest generation does not match identity"
            )
        store.manifest = manifest
        store._validate_existing_members()
        return store

    def add(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        environment_steps: int,
        round_id: int,
        run_id: str,
        evaluation: Mapping[str, float] | None = None,
    ) -> SnapshotEntry:
        """Write a new immutable member before atomically publishing it.

        The manifest is always replaced before old files are unlinked.  An
        interruption can therefore leave an unreferenced old file, but cannot
        publish a manifest whose entry was already removed from disk.
        """
        validated_evaluation = _EVALUATION.validate_python(evaluation or {})
        self._validate_existing_members()
        if self.manifest_path.is_file():
            # Backfill the generation for a head written before versioned
            # manifests existed, before any newer head can supersede it.
            self._publish_generation(self.manifest)
        temporary_path = self._write_temporary(state_dict)
        digest = sha256_file(temporary_path)
        final_path = self.directory / (
            f"snapshot-{environment_steps:012d}-{digest[:12]}.pt"
        )
        self._publish_snapshot(temporary_path, final_path, digest)
        entry = SnapshotEntry(
            path=final_path,
            sha256=digest,
            structure=self.structure,
            environment_steps=environment_steps,
            round_id=round_id,
            run_id=run_id,
            created_at=datetime.now(timezone.utc),
            evaluation=validated_evaluation,
        )
        overflow = max(0, len(self.manifest.entries) + 1 - self.capacity)
        manifest = SnapshotManifest(entries=(*self.manifest.entries[overflow:], entry))
        self._publish_manifest(manifest)
        self.manifest = manifest
        return entry

    def load(self, entry: SnapshotEntry) -> dict[str, torch.Tensor]:
        """Validate a selected entry before loading its tensor state dictionary."""
        self._validate_entry_path(entry)
        _verify_entry(entry, self.structure)
        return _load_state_dict(entry)

    def _validate_existing_members(self) -> None:
        """Refuse to replace a member that is invalid, missing, or corrupt."""
        verified: set[tuple[Path, str, str]] = set()
        for entry in self.manifest.entries:
            self._validate_entry_path(entry)
            identity = (entry.path, entry.sha256, entry.structure)
            if identity in verified:
                continue
            _verify_entry(entry, self.structure)
            verified.add(identity)

    def _validate_entry_path(self, entry: SnapshotEntry) -> None:
        """Require an entry to name one canonical content-addressed pool file."""
        path = entry.path
        match = _SNAPSHOT_NAME.fullmatch(path.name)
        if (
            not path.is_absolute()
            or path.parent != self.directory
            or path.resolve().parent != self.directory
            or match is None
            or match.group(1) != f"{entry.environment_steps:012d}"
            or match.group(2) != entry.sha256[:12]
            or len(entry.sha256) != 64
            or any(character not in "0123456789abcdef" for character in entry.sha256)
        ):
            raise SnapshotIntegrityError(f"manifest snapshot path is invalid: {path}")

    def _write_temporary(self, state_dict: Mapping[str, torch.Tensor]) -> Path:
        """Serialize and fsync a sibling temporary snapshot before publishing it."""
        with tempfile.NamedTemporaryFile(
            dir=self.directory, suffix=".tmp", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            torch.save(dict(state_dict), temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        return temporary_path

    def _publish_snapshot(
        self, temporary_path: Path, final_path: Path, digest: str
    ) -> None:
        """Rename a complete snapshot, without replacing a corrupt same-name file."""
        if final_path.exists():
            if sha256_file(final_path) != digest:
                temporary_path.unlink(missing_ok=True)
                raise SnapshotIntegrityError(
                    f"content-addressed snapshot path has wrong digest: {final_path}"
                )
            temporary_path.unlink()
            return
        os.replace(temporary_path, final_path)  # noqa: PTH105
        _fsync_directory(self.directory)

    def _publish_manifest(self, manifest: SnapshotManifest) -> None:
        """Durably retain a generation before atomically advancing the head."""
        serialized = manifest.model_dump_json().encode()
        self._publish_generation(manifest, serialized=serialized)
        temporary_path = self.manifest_path.with_suffix(".json.tmp")
        with temporary_path.open("wb") as temporary:
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, self.manifest_path)  # noqa: PTH105
        _fsync_directory(self.directory)

    def _publish_generation(
        self,
        manifest: SnapshotManifest,
        *,
        serialized: bytes | None = None,
    ) -> None:
        """Create or verify one immutable manifest generation durably."""
        if serialized is None:
            serialized = manifest.model_dump_json().encode()
        generation_path = self._generation_path(manifest)
        if generation_path.is_file():
            try:
                if SnapshotManifest.load(generation_path) != manifest:
                    raise SnapshotIntegrityError(
                        "immutable snapshot manifest generation changed"
                    )
            except (OSError, ValueError) as error:
                raise SnapshotIntegrityError(
                    "immutable snapshot manifest generation is corrupt"
                ) from error
        else:
            generation_temporary = generation_path.with_suffix(".json.tmp")
            with generation_temporary.open("wb") as temporary:
                temporary.write(serialized)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(generation_temporary, generation_path)  # noqa: PTH105
            _fsync_directory(self.directory)

    def _generation_path(self, manifest: SnapshotManifest) -> Path:
        """Return the content-addressed immutable path for ``manifest``."""
        digest = hashlib.sha256(manifest.model_dump_json().encode()).hexdigest()
        return self.directory / f"manifest-{digest}.json"


class SnapshotPool:
    """Deterministically select and load already-published frozen opponents."""

    def __init__(
        self,
        manifest: SnapshotManifest,
        *,
        seed: int,
        structure: str | None = None,
        store: SnapshotStore | None = None,
    ) -> None:
        self.manifest = manifest
        self.seed = seed
        self.structure = structure
        self._store = store

    @classmethod
    def empty(cls, *, seed: int = 0, structure: str | None = None) -> SnapshotPool:
        """Return an explicit empty pool for configurations without members yet."""
        return cls(SnapshotManifest(), seed=seed, structure=structure)

    @classmethod
    def from_store(cls, store: SnapshotStore, *, seed: int) -> SnapshotPool:
        """Bind deterministic sampling to a store's structural fingerprint."""
        return cls(
            store.manifest,
            seed=seed,
            structure=store.structure,
            store=store,
        )

    def rebind(self, *, structure: str, seed: int) -> SnapshotPool:
        """Bind selection to resolved config identity, rejecting foreign stores."""
        if not self.manifest.entries and self._store is None:
            return SnapshotPool.empty(seed=seed, structure=structure)
        if self._store is None:
            raise SnapshotIntegrityError(
                "nonempty snapshot pool has no store directory for validation"
            )
        if self.structure != structure or self._store.structure != structure:
            raise SnapshotIntegrityError(
                "snapshot pool structure does not match resolved config: "
                f"expected {structure}, found {self.structure}"
            )
        if self.manifest != self._store.manifest:
            raise SnapshotIntegrityError(
                "snapshot pool manifest does not match its bound store"
            )
        return SnapshotPool.from_store(self._store, seed=seed)

    def identity(self) -> SnapshotPoolIdentity:
        """Return the exact store view required to resume without republishing."""
        if self._store is None:
            raise SnapshotIntegrityError(
                "snapshot pool has no store directory for checkpoint identity"
            )
        if self.manifest != self._store.manifest:
            raise SnapshotIntegrityError(
                "snapshot pool manifest does not match its bound store"
            )
        return SnapshotPoolIdentity(
            directory=self._store.directory,
            structure=self._store.structure,
            seed=self.seed,
            capacity=self._store.capacity,
            manifest=self.manifest,
        )

    @classmethod
    def reopen(
        cls,
        identity: SnapshotPoolIdentity,
        *,
        structure: str,
        seed: int,
        capacity: int,
    ) -> SnapshotPool:
        """Reopen the exact checkpointed manifest without publishing entries."""
        if (
            identity.structure != structure
            or identity.seed != seed
            or identity.capacity != capacity
        ):
            raise SnapshotIntegrityError(
                "checkpointed snapshot pool identity does not match resolved config"
            )
        store = SnapshotStore.open_generation(
            identity.directory,
            capacity=capacity,
            structure=structure,
            manifest=identity.manifest,
        )
        return cls.from_store(store, seed=seed)

    def sample(self, game_id: int) -> SnapshotEntry:
        """Select one entry solely from the configured seed and global game ID."""
        if not self.manifest.entries:
            raise EmptySnapshotPoolError(
                "frozen opponent requested before pool population"
            )
        rng = random.Random((self.seed << 64) ^ game_id)
        return rng.choice(self.manifest.entries)

    def load(self, entry: SnapshotEntry) -> dict[str, torch.Tensor]:
        """Fail closed if the selected entry is incompatible or byte-corrupt."""
        return self.load_many((entry,))[entry.sha256]

    def load_many(
        self, entries: Sequence[SnapshotEntry]
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Verify the full manifest once, then materialize each selected digest once."""
        if self.structure is None:
            raise SnapshotIntegrityError(
                "snapshot pool has no structural fingerprint for validation"
            )
        if self._store is None:
            raise SnapshotIntegrityError(
                "snapshot pool has no store directory for path validation"
            )
        if self.manifest != self._store.manifest:
            raise SnapshotIntegrityError(
                "snapshot pool manifest does not match its bound store"
            )
        self._store._validate_existing_members()
        loaded: dict[str, dict[str, torch.Tensor]] = {}
        selected_by_digest: dict[str, SnapshotEntry] = {}
        for entry in entries:
            if entry not in self._store.manifest.entries:
                raise SnapshotIntegrityError(
                    f"selected snapshot is not in the bound store: {entry.path}"
                )
            # Every distinct path/digest binding was independently verified by
            # ``_validate_existing_members`` above. Tensor deserialization alone
            # is content-addressed, so equal bytes may reuse the first loaded state
            # without collapsing either entry's provenance.
            selected_by_digest.setdefault(entry.sha256, entry)
        for digest, entry in selected_by_digest.items():
            loaded[digest] = _load_state_dict(entry)
        return loaded


def _verify_entry(entry: SnapshotEntry, expected_structure: str) -> None:
    """Verify model compatibility and bytes before unpickling the snapshot."""
    if entry.structure != expected_structure:
        raise SnapshotIntegrityError(
            "snapshot structure mismatch: "
            f"expected {expected_structure}, found {entry.structure}"
        )
    try:
        actual_digest = sha256_file(entry.path)
    except OSError as error:
        raise SnapshotIntegrityError(
            "snapshot digest verification failed for "
            f"{entry.path}: expected {entry.sha256}"
        ) from error
    if actual_digest != entry.sha256:
        raise SnapshotIntegrityError(
            f"snapshot digest mismatch for {entry.path}: expected {entry.sha256}, "
            f"found {actual_digest}"
        )


def _load_state_dict(entry: SnapshotEntry) -> dict[str, torch.Tensor]:
    """Load only tensor weights after the caller verified the file's identity."""
    try:
        loaded = torch.load(entry.path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise SnapshotIntegrityError(
            f"snapshot cannot be loaded: {entry.path}"
        ) from error
    if not isinstance(loaded, dict) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in loaded.items()
    ):
        raise SnapshotIntegrityError(
            f"snapshot is not a tensor state dictionary: {entry.path}"
        )
    return cast(dict[str, torch.Tensor], loaded)


def _fsync_directory(directory: Path) -> None:
    """Persist a sibling rename on filesystems that require directory syncing."""
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
