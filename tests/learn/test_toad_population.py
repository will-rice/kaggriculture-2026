"""Contracts for verified immutable frozen-opponent snapshots."""

from pathlib import Path

import pytest
import torch

from kaggriculture.learn.toad.population import (
    EmptySnapshotPoolError,
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
