"""Counterfactual evidence that can be trusted a month after it was written.

A shard of counterfactual rows is read by a training run that has no way to
re-derive it. Every property the learner assumes about that file has to be
provable from the file: which controllers and which engine produced it, which
event each row belongs to, which action row three actually was, and that the
season behind each row finished. So publication is strict at both ends.

**Write-once by construction.** A shard's address is its cell -- seed, seat, and
the digest of the run identity -- and it is claimed with ``os.link``, which the
kernel refuses if the name already exists. There is no check-then-write window
for a second worker to slip through. Republishing the identical bytes is a
no-op, which is what ``--resume`` needs; publishing *different* bytes to a
claimed address raises, which is what an accidentally re-parameterised rerun
deserves. Nothing here ever overwrites evidence.

**Published means manifested.** Bytes reach the address through a same-directory
temporary file that is flushed, fsynced, linked, and only then followed by a
SHA-256 manifest, itself linked last. A crash before the temp write leaves
nothing; a crash after it leaves an orphan temporary a resume can quarantine; a
crash between the shard link and the manifest link leaves a shard that
``load_shard_strict`` refuses, because a shard without a manifest is not a
published shard. The ordering is what makes "present" and "trustworthy" the same
question.

**Loading is all-or-nothing.** ``load_shard_strict`` parses the whole file,
checks the manifest digest, checks the identity against the one the caller
expected, and checks the row-level invariants -- one anchor per event, no
duplicate alternative, the declared seed, no failed outcome -- before it returns
anything. A partially valid shard yields no rows at all, because a caller that
received four of five events would train on a dataset with a hole in it and
never know.
"""

import hashlib
import os
from pathlib import Path
from typing import Any, Literal, Self, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kaggriculture.learn.market_residual.alternatives import FAMILY_KAITO
from kaggriculture.learn.market_residual.counterfactual import (
    CounterfactualIntegrityError,
    CounterfactualSnapshot,
    SeasonIdentity,
    branch_event,
    branch_rows,
    canonical_digest,
    canonical_json,
)
from kaggriculture.market_residual.schema import validate_seed_bank
from kaggriculture.sim.state import PRODUCT_NAMES

SHARD_VERSION = 1

SHARD_SUFFIX = ".jsonl"
MANIFEST_SUFFIX = ".sha256"


class CounterfactualImmutabilityError(RuntimeError):
    """Raised when a published artifact address is asked to hold new bytes."""


class OutcomeRecord(BaseModel):
    """One branch arm's finished season, as it is written down."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, allow_inf_nan=False
    )

    banks: tuple[int, int]
    win_points: float = Field(ge=0.0, le=1.0)
    margin: float = Field(ge=-1.0, le=1.0)
    terminal_prices: tuple[int, ...]


class ShardIdentity(BaseModel):
    """The run identity a shard was produced under, and its digest.

    The fields are carried as the canonical mapping ``SeasonIdentity`` already
    produces rather than re-declared here, so the two cannot drift apart, and
    the digest is validated against them on every load: editing a field without
    recomputing the digest is refused, and recomputing it changes the address.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    version: int
    fields: dict[str, Any]
    sha256: str

    @classmethod
    def of(cls, identity: SeasonIdentity) -> "ShardIdentity":
        """Return the shard identity for one season identity.

        Args:
            identity: The season identity the rows were produced under.

        Returns:
            Its canonical fields and digest.
        """
        return cls(
            version=SHARD_VERSION,
            fields=identity.canonical(),
            sha256=identity.sha256,
        )

    @model_validator(mode="after")
    def digest_matches_fields(self) -> Self:
        """Keep the digest and the fields it names inseparable.

        Returns:
            This identity, when its digest is the digest of its own fields.

        Raises:
            ValueError: If the two disagree.
        """
        if self.sha256 != canonical_digest(self.fields):
            raise ValueError("shard identity digest does not match its own fields")
        return self


class CounterfactualRow(BaseModel):
    """One alternative at one event, scored against the controller's own play."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, allow_inf_nan=False
    )

    identity_sha256: str
    opponent: str
    seed: int
    seat: Literal[0, 1]
    event_index: int = Field(ge=0)
    event_turn: int = Field(ge=0)
    event_fingerprint: str
    alternatives_sha256: str
    alternative_id: int = Field(ge=0)
    family: str
    buckets: tuple[int, ...]
    event_prices: tuple[int, ...]
    kaito: OutcomeRecord
    alternative: OutcomeRecord
    win_point_delta: float = Field(ge=-1.0, le=1.0)
    margin_delta: float = Field(ge=-2.0, le=2.0)
    failure: str | None

    @model_validator(mode="after")
    def anchor_is_row_zero_and_scores_itself(self) -> Self:
        """Keep the anchor's identity, position, and null delta in agreement.

        Returns:
            This row, when the controller's own play is row zero and nothing
            else claims to be it.

        Raises:
            ValueError: If the family and the position disagree, or the anchor
                is scored against something other than itself.
        """
        anchor = self.family == FAMILY_KAITO
        if anchor != (self.alternative_id == 0):
            raise ValueError(
                f"family {self.family!r} at alternative {self.alternative_id} "
                "is not the anchor's position"
            )
        if anchor and self.kaito != self.alternative:
            raise ValueError("the anchor row must be scored against its own outcome")
        if self.win_point_delta != self.alternative.win_points - self.kaito.win_points:
            raise ValueError("win point delta is not the paired difference")
        if self.margin_delta != self.alternative.margin - self.kaito.margin:
            raise ValueError("margin delta is not the paired difference")
        return self


class CounterfactualShard(BaseModel):
    """One seat of one seed: every scored alternative, under one identity."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    identity: ShardIdentity
    rows: tuple[CounterfactualRow, ...]


def counterfactual_rows(
    snapshot: CounterfactualSnapshot,
) -> tuple[CounterfactualRow, ...]:
    """Branch one event and return every alternative scored against the anchor.

    Args:
        snapshot: The instant to branch.

    Returns:
        One row per alternative, the anchor first, each carrying the identity
        digest of the run that produced it.

    Raises:
        CounterfactualIntegrityError: If the branch does not reproduce the
            reference engine's own season on its anchor arm.
    """
    rows = branch_rows(snapshot)
    outcomes = branch_event(snapshot, rows)
    identity = snapshot.identity
    anchor = OutcomeRecord(
        banks=outcomes[0].banks,
        win_points=outcomes[0].win_points,
        margin=outcomes[0].margin,
        terminal_prices=outcomes[0].terminal_prices,
    )
    event_prices = tuple(
        int(snapshot.state.prices[0, index]) for index in range(len(PRODUCT_NAMES))
    )
    return tuple(
        CounterfactualRow(
            identity_sha256=identity.sha256,
            opponent=identity.opponent.module,
            seed=identity.seed,
            seat=identity.seat,
            event_index=snapshot.index,
            event_turn=snapshot.event.turn,
            event_fingerprint=snapshot.event.fingerprint,
            alternatives_sha256=snapshot.alternatives.sha256,
            alternative_id=index,
            family=row.family,
            buckets=row.buckets,
            event_prices=event_prices,
            kaito=anchor,
            alternative=OutcomeRecord(
                banks=outcome.banks,
                win_points=outcome.win_points,
                margin=outcome.margin,
                terminal_prices=outcome.terminal_prices,
            ),
            win_point_delta=outcome.win_points - anchor.win_points,
            margin_delta=outcome.margin - anchor.margin,
            failure=None,
        )
        for index, (row, outcome) in enumerate(zip(rows, outcomes, strict=True))
    )


def shard_bytes(shard: CounterfactualShard) -> bytes:
    """Return the exact bytes one shard publishes.

    The identity is the first line and every row follows in order, each one
    canonical JSON, so two processes holding the same shard write the same file.

    Args:
        shard: The shard to encode.

    Returns:
        The canonical JSONL encoding, newline-terminated.
    """
    lines = [canonical_json(shard.identity.model_dump(mode="json"))]
    lines.extend(canonical_json(row.model_dump(mode="json")) for row in shard.rows)
    return ("\n".join(lines) + "\n").encode("utf-8")


def shard_path(root: Path, identity: ShardIdentity) -> Path:
    """Return the one address a cell's shard may ever occupy.

    The cell -- seed, seat, and the digest of everything else the run was
    produced under -- is the whole address, so a rerun that changed any of it
    writes somewhere else and a rerun that changed none of it writes here again.

    Args:
        root: The artifact root.
        identity: The run identity the shard was produced under.

    Returns:
        The shard's path; its manifest sits beside it.
    """
    return root / (
        f"seed{identity.fields['seed']}-seat{identity.fields['seat']}"
        f"-{identity.sha256[:16]}{SHARD_SUFFIX}"
    )


def write_shard_atomic(root: Path, shard: CounterfactualShard) -> Path:
    """Publish one shard and its manifest, once, or refuse.

    Args:
        root: The artifact root; created if absent.
        shard: The shard to publish.

    Returns:
        The published path.

    Raises:
        CounterfactualImmutabilityError: If the address already holds different
            bytes.
    """
    root.mkdir(parents=True, exist_ok=True)
    path = shard_path(root, shard.identity)
    payload = shard_bytes(shard)
    digest = hashlib.sha256(payload).hexdigest()
    _claim(path, payload)
    _claim(path.with_suffix(MANIFEST_SUFFIX), f"{digest}\n".encode("utf-8"))
    return path


def _claim(path: Path, payload: bytes) -> None:
    """Bind one address to exactly these bytes, forever.

    Args:
        path: The address to claim.
        payload: The bytes it must hold.

    Raises:
        CounterfactualImmutabilityError: If the address already holds different
            bytes.
    """
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}-{id(payload):x}")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise CounterfactualImmutabilityError(
                f"{path} is published and holds different bytes"
            ) from None
    finally:
        temporary.unlink()
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def load_shard_strict(path: Path, expected: SeasonIdentity) -> CounterfactualShard:
    """Return a published shard, or nothing at all.

    Args:
        path: The published shard's path.
        expected: The run identity the caller is collecting under.

    Returns:
        The shard, once every invariant holds.

    Raises:
        CounterfactualIntegrityError: If the manifest is absent or wrong, the
            identity is not the expected one, the file does not parse, or the
            rows are duplicated, anchorless, out of bank, or failed.
    """
    manifest = path.with_suffix(MANIFEST_SUFFIX)
    if not manifest.exists():
        raise CounterfactualIntegrityError(f"{path} has no published manifest")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if manifest.read_text(encoding="utf-8").strip() != digest:
        raise CounterfactualIntegrityError(f"{path} does not match its manifest")

    lines = payload.decode("utf-8").splitlines()
    if not lines:
        raise CounterfactualIntegrityError(f"{path} is empty")
    wanted = ShardIdentity.of(expected)
    try:
        identity = ShardIdentity.model_validate_json(lines[0])
        rows = tuple(CounterfactualRow.model_validate_json(line) for line in lines[1:])
    except ValueError as error:
        raise CounterfactualIntegrityError(f"{path} does not parse: {error}") from error
    if identity != wanted:
        raise CounterfactualIntegrityError(
            f"{path} was produced under {identity.sha256}, expected {wanted.sha256}"
        )
    _check_rows(path, rows, expected)
    return CounterfactualShard(identity=identity, rows=rows)


def _check_rows(
    path: Path, rows: Sequence[CounterfactualRow], expected: SeasonIdentity
) -> None:
    """Raise unless every row belongs to this cell and every event has an anchor.

    Args:
        path: The shard being loaded, for the message.
        rows: Every parsed row, in file order.
        expected: The run identity the caller is collecting under.

    Raises:
        CounterfactualIntegrityError: If any invariant fails.
    """
    seen: set[tuple[int, int]] = set()
    anchored: set[int] = set()
    events: set[int] = set()
    for row in rows:
        if row.failure is not None:
            raise CounterfactualIntegrityError(
                f"{path} event {row.event_index} alternative {row.alternative_id} "
                f"records failure {row.failure!r}"
            )
        if row.identity_sha256 != expected.sha256:
            raise CounterfactualIntegrityError(
                f"{path} event {row.event_index} carries identity "
                f"{row.identity_sha256}, expected {expected.sha256}"
            )
        if (row.seed, row.seat) != (expected.seed, expected.seat):
            raise CounterfactualIntegrityError(
                f"{path} holds seed {row.seed} seat {row.seat}, expected "
                f"seed {expected.seed} seat {expected.seat}"
            )
        key = (row.event_index, row.alternative_id)
        if key in seen:
            raise CounterfactualIntegrityError(f"{path} repeats row {key}")
        seen.add(key)
        events.add(row.event_index)
        if row.alternative_id == 0:
            anchored.add(row.event_index)
    validate_seed_bank(expected.seed_bank, (expected.seed,))
    missing = sorted(events - anchored)
    if missing:
        raise CounterfactualIntegrityError(
            f"{path} scores events {missing} against no controller anchor"
        )
