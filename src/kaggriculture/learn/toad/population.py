"""Durable, verified snapshots for frozen Toad opponents."""

from __future__ import annotations

import hashlib
import os
import random
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, cast

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    NonNegativeInt,
    TypeAdapter,
)

_EVALUATION = TypeAdapter(dict[str, FiniteFloat])
_SNAPSHOT_NAME = re.compile(r"^snapshot-(\d{12})-([0-9a-f]{12})\.pt$")


class SnapshotIntegrityError(RuntimeError):
    """A snapshot cannot safely be used as the selected frozen opponent."""


class EmptySnapshotPoolError(RuntimeError):
    """A collection requested a frozen opponent before any snapshot existed."""


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
    evaluation: dict[str, FiniteFloat] = Field(default_factory=dict)


class SnapshotManifest(BaseModel):
    """The atomically published members of one frozen-opponent population."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entries: tuple[SnapshotEntry, ...] = ()

    @classmethod
    def load(cls, path: Path) -> SnapshotManifest:
        """Deserialize one manifest without accepting unrecognised fields."""
        return cls.model_validate_json(path.read_text())


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
        evicted = self.manifest.entries[:overflow]
        manifest = SnapshotManifest(entries=(*self.manifest.entries[overflow:], entry))
        self._publish_manifest(manifest)
        self.manifest = manifest
        for old_entry in evicted:
            if old_entry.path not in {member.path for member in manifest.entries}:
                old_entry.path.unlink(missing_ok=True)
        if evicted:
            _fsync_directory(self.directory)
        return entry

    def load(self, entry: SnapshotEntry) -> dict[str, torch.Tensor]:
        """Validate a selected entry before loading its tensor state dictionary."""
        self._validate_entry_path(entry)
        _verify_entry(entry, self.structure)
        return _load_state_dict(entry)

    def _validate_existing_members(self) -> None:
        """Refuse to replace a member that is invalid, missing, or corrupt."""
        for entry in self.manifest.entries:
            self._validate_entry_path(entry)
            _verify_entry(entry, self.structure)

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
        """Write and atomically replace the population's sole manifest."""
        temporary_path = self.manifest_path.with_suffix(".json.tmp")
        with temporary_path.open("wb") as temporary:
            temporary.write(manifest.model_dump_json().encode())
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, self.manifest_path)  # noqa: PTH105
        _fsync_directory(self.directory)


class SnapshotPool:
    """Deterministically select and load already-published frozen opponents."""

    def __init__(
        self,
        manifest: SnapshotManifest,
        *,
        seed: int,
        structure: str | None = None,
    ) -> None:
        self.manifest = manifest
        self.seed = seed
        self.structure = structure

    @classmethod
    def empty(cls, *, seed: int = 0, structure: str | None = None) -> SnapshotPool:
        """Return an explicit empty pool for configurations without members yet."""
        return cls(SnapshotManifest(), seed=seed, structure=structure)

    @classmethod
    def from_store(cls, store: SnapshotStore, *, seed: int) -> SnapshotPool:
        """Bind deterministic sampling to a store's structural fingerprint."""
        return cls(store.manifest, seed=seed, structure=store.structure)

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
        if self.structure is None:
            raise SnapshotIntegrityError(
                "snapshot pool has no structural fingerprint for validation"
            )
        _verify_entry(entry, self.structure)
        return _load_state_dict(entry)


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
