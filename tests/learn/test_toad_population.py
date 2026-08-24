"""Contracts for verified immutable frozen-opponent snapshots."""

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from kaggriculture.learn.toad.population import (
    EmptySnapshotPoolError,
    SnapshotEntry,
    SnapshotIntegrityError,
    SnapshotManifest,
    SnapshotPool,
    SnapshotStore,
    sha256_file,
)


def state_dict(value: float = 1.0) -> dict[str, torch.Tensor]:
    """Return a small real torch state dictionary with a distinct payload."""
    return {"weight": torch.tensor([value])}


def populated_pool(path: Path, *, count: int, seed: int) -> SnapshotPool:
    """Create a pool with distinct snapshots through the public store API."""
    store = SnapshotStore(path, capacity=count, structure="model-v1")
    for step in range(count):
        store.add(
            state_dict(float(step)),
            environment_steps=step,
            round_id=step,
            run_id="run",
        )
    return SnapshotPool(store.manifest, seed=seed)


def manifest_entry(
    path: Path, *, digest: str, steps: int = 1, structure: str = "model-v1"
) -> SnapshotEntry:
    """Build metadata for a controlled manifest-path validation case."""
    return SnapshotEntry(
        path=path,
        sha256=digest,
        structure=structure,
        environment_steps=steps,
        round_id=0,
        run_id="run",
        created_at=datetime.now(timezone.utc),
    )


def test_snapshot_store_writes_a_verified_manifest_entry(tmp_path: Path) -> None:
    """A published entry names an intact snapshot and leaves no torn temporary."""
    store = SnapshotStore(tmp_path, capacity=3, structure="abc")

    entry = store.add(state_dict(), environment_steps=100, round_id=4, run_id="run")

    assert entry.path.is_file()
    assert entry.path.name == f"snapshot-000000000100-{entry.sha256[:12]}.pt"
    assert sha256_file(entry.path) == entry.sha256
    assert SnapshotManifest.load(tmp_path / "manifest.json").entries == (entry,)
    assert not list(tmp_path.glob("*.tmp"))


def test_snapshot_store_evicts_the_oldest_only_after_manifest_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unlink failure leaves the old file orphaned, never manifest-referenced."""
    store = SnapshotStore(tmp_path, capacity=1, structure="model-v1")
    first = store.add(state_dict(1.0), environment_steps=1, round_id=1, run_id="run")

    def refuse_unlink(path: Path, *args: object, **kwargs: object) -> None:
        manifest = SnapshotManifest.load(tmp_path / "manifest.json")
        assert first not in manifest.entries
        raise OSError("simulated unlink interruption")

    monkeypatch.setattr(Path, "unlink", refuse_unlink)

    with pytest.raises(OSError, match="simulated unlink interruption"):
        store.add(state_dict(2.0), environment_steps=2, round_id=2, run_id="run")

    manifest = SnapshotManifest.load(tmp_path / "manifest.json")
    assert len(manifest.entries) == 1
    assert manifest.entries[0].environment_steps == 2
    assert first.path.is_file()


@pytest.mark.parametrize(
    "kind",
    ["absolute", "traversal", "nested", "wrong_steps", "wrong_digest"],
)
def test_store_refuses_unconfined_or_misnamed_manifest_members_before_eviction(
    tmp_path: Path, kind: str
) -> None:
    """Malformed metadata cannot replace or unlink a file outside its pool."""
    pool = tmp_path / "pool"
    pool.mkdir()
    payload = b"outside sentinel"
    digest = hashlib.sha256(payload).hexdigest()
    name = f"snapshot-000000000001-{digest[:12]}.pt"
    if kind == "absolute":
        candidate = tmp_path / "outside" / name
    elif kind == "traversal":
        candidate = pool / ".." / "outside" / name
    elif kind == "nested":
        candidate = pool / "nested" / name
    elif kind == "wrong_steps":
        candidate = pool / f"snapshot-000000000002-{digest[:12]}.pt"
    else:
        wrong_prefix = "0" if digest[0] != "0" else "1"
        candidate = pool / f"snapshot-000000000001-{wrong_prefix}{digest[1:12]}.pt"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_bytes(payload)
    entry = manifest_entry(candidate, digest=digest)
    manifest_path = pool / "manifest.json"
    manifest_path.write_text(SnapshotManifest(entries=(entry,)).model_dump_json())
    before_manifest = manifest_path.read_text()

    store = SnapshotStore(pool, capacity=1, structure="model-v1")
    with pytest.raises(SnapshotIntegrityError, match="manifest snapshot path"):
        store.add(state_dict(2.0), environment_steps=2, round_id=2, run_id="run")

    assert candidate.read_bytes() == payload
    assert manifest_path.read_text() == before_manifest


@pytest.mark.parametrize("failure", ["missing", "corrupt", "structure"])
def test_store_add_rejects_invalid_existing_member_without_mutation(
    tmp_path: Path, failure: str
) -> None:
    """A capacity replacement never conceals an invalid manifest member."""
    store = SnapshotStore(tmp_path, capacity=1, structure="model-v1")
    existing = store.add(state_dict(1.0), environment_steps=1, round_id=1, run_id="run")
    if failure == "missing":
        existing.path.unlink()
    elif failure == "corrupt":
        existing.path.write_bytes(b"corrupt")
    else:
        store = SnapshotStore(tmp_path, capacity=1, structure="model-v2")
    before_manifest = (tmp_path / "manifest.json").read_bytes()
    before_files = {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    }

    with pytest.raises(SnapshotIntegrityError):
        store.add(state_dict(2.0), environment_steps=2, round_id=2, run_id="run")

    assert (tmp_path / "manifest.json").read_bytes() == before_manifest
    assert {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    } == before_files


def test_nonempty_finite_evaluation_round_trips_through_manifest(
    tmp_path: Path,
) -> None:
    """Evaluation metadata remains valid JSON after durable manifest publication."""
    store = SnapshotStore(tmp_path, capacity=1, structure="model-v1")
    entry = store.add(
        state_dict(),
        environment_steps=1,
        round_id=0,
        run_id="run",
        evaluation={"win_rate": 0.75},
    )

    assert SnapshotManifest.load(tmp_path / "manifest.json").entries[0].evaluation == {
        "win_rate": 0.75
    }
    assert entry.evaluation == {"win_rate": 0.75}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_evaluation_is_rejected_before_snapshot_mutation(
    tmp_path: Path, value: float
) -> None:
    """Invalid JSON numbers cannot leave an unreloadable snapshot behind."""
    store = SnapshotStore(tmp_path, capacity=1, structure="model-v1")

    with pytest.raises(ValidationError):
        store.add(
            state_dict(),
            environment_steps=1,
            round_id=0,
            run_id="run",
            evaluation={"score": value},
        )

    assert not list(tmp_path.iterdir())


def test_population_sampling_is_deterministic_by_game_id(tmp_path: Path) -> None:
    """A game ID selects the same member independently of call ordering."""
    pool = populated_pool(tmp_path, count=3, seed=17)

    assert pool.sample(game_id=91) == pool.sample(game_id=91)


def test_empty_population_refuses_a_frozen_opponent_request(tmp_path: Path) -> None:
    """No arbitrary fallback policy replaces an empty frozen pool."""
    pool = SnapshotPool(SnapshotManifest(), seed=17)

    with pytest.raises(
        EmptySnapshotPoolError, match="frozen opponent requested before pool population"
    ):
        pool.sample(game_id=91)


def test_corrupt_snapshot_is_never_substituted(tmp_path: Path) -> None:
    """A selected member with a changed digest fails closed before unpickling."""
    store = SnapshotStore(tmp_path, capacity=1, structure="model-v1")
    selected = store.add(state_dict(), environment_steps=1, round_id=0, run_id="run")
    pool = SnapshotPool(store.manifest, seed=17, structure="model-v1")
    selected.path.write_bytes(b"corrupt")

    with pytest.raises(SnapshotIntegrityError, match=selected.sha256):
        pool.load(selected)


def test_structure_mismatch_is_rejected_before_loading(tmp_path: Path) -> None:
    """A valid foreign snapshot cannot cross model-shape compatibility boundaries."""
    store = SnapshotStore(tmp_path, capacity=1, structure="model-v1")
    selected = store.add(state_dict(), environment_steps=1, round_id=0, run_id="run")
    pool = SnapshotPool(store.manifest, seed=17, structure="model-v2")

    with pytest.raises(SnapshotIntegrityError, match="model-v2"):
        pool.load(selected)
