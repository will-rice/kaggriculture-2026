"""Contracts for verified immutable frozen-opponent snapshots."""

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from kaggriculture.learn.model import Policy
from kaggriculture.learn.toad.config import ModelConfig, TeacherSpec, ToadConfig
from kaggriculture.learn.toad.population import (
    EmptySnapshotPoolError,
    SnapshotEntry,
    SnapshotIntegrityError,
    SnapshotManifest,
    SnapshotPool,
    SnapshotStore,
    TeacherCompatibilityError,
    load_teacher,
    sha256_file,
)


def state_dict(value: float = 1.0) -> dict[str, torch.Tensor]:
    """Return a small real torch state dictionary with a distinct payload."""
    return {"weight": torch.tensor([value])}


def _teacher_state(*, quantity: bool = True) -> dict[str, torch.Tensor]:
    """Return a small real teacher state with an optionally absent head."""
    state = Policy(blocks=1, channels=4, value_bound=1.0).state_dict()
    return {
        name: tensor
        for name, tensor in state.items()
        if quantity or not name.startswith("quantity_head.")
    }


def test_teacher_cannot_claim_a_missing_quantity_head(tmp_path: Path) -> None:
    """A declared head may never retain random constructor parameters."""
    checkpoint = tmp_path / "teacher.pt"
    torch.save(_teacher_state(quantity=False), checkpoint)
    spec = TeacherSpec(checkpoint=checkpoint, blocks=1, quantity=True)

    with pytest.raises(TeacherCompatibilityError, match="quantity_head"):
        load_teacher(spec, ModelConfig.control(blocks=1, channels=4))


def test_teacher_accepts_a_completely_absent_undeclared_quantity_head(
    tmp_path: Path,
) -> None:
    """Historical teachers remain usable only under an explicit false claim."""
    checkpoint = tmp_path / "teacher.pt"
    torch.save(_teacher_state(quantity=False), checkpoint)

    loaded = load_teacher(
        TeacherSpec(checkpoint=checkpoint, blocks=1, quantity=False),
        ModelConfig.control(blocks=1, channels=4),
    )

    assert loaded.spec.quantity is False
    assert loaded.spec.sha256 == sha256_file(checkpoint)
    assert not loaded.policy.training
    assert not any(parameter.requires_grad for parameter in loaded.policy.parameters())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_shared", "stem[.]weight"),
        ("unexpected", "unexpected keys"),
        ("partial_head", "incomplete quantity_head"),
        ("shape", "shape mismatch.*stem[.]weight"),
    ],
)
def test_teacher_rejects_malformed_or_structurally_incompatible_state(
    tmp_path: Path, mutation: str, message: str
) -> None:
    """Only a complete declared topology may become a frozen teacher."""
    state = _teacher_state()
    if mutation == "missing_shared":
        del state["stem.weight"]
    elif mutation == "unexpected":
        state["foreign.weight"] = torch.ones(1)
    elif mutation == "partial_head":
        del state["quantity_head.bias"]
    else:
        state["stem.weight"] = state["stem.weight"][:1]
    checkpoint = tmp_path / "teacher.pt"
    torch.save(state, checkpoint)

    with pytest.raises(TeacherCompatibilityError, match=message):
        load_teacher(
            TeacherSpec(checkpoint=checkpoint, blocks=1, quantity=False),
            ModelConfig.control(blocks=1, channels=4),
        )


def test_teacher_rejects_a_declared_digest_mismatch(tmp_path: Path) -> None:
    """A path can never silently replace the bytes named by its contract."""
    checkpoint = tmp_path / "teacher.pt"
    torch.save(_teacher_state(), checkpoint)

    with pytest.raises(TeacherCompatibilityError, match="digest.*expected"):
        load_teacher(
            TeacherSpec(checkpoint=checkpoint, blocks=1, sha256="0" * 64),
            ModelConfig.control(blocks=1, channels=4),
        )


def test_teacher_rejects_mixed_checkpoint_envelopes(tmp_path: Path) -> None:
    """Neither of two plausible policy containers may win by branch order."""
    state = _teacher_state()
    checkpoint = tmp_path / "teacher.pt"
    torch.save(
        {
            "learner": state,
            "state_dict": {f"policy.{name}": value for name, value in state.items()},
        },
        checkpoint,
    )

    with pytest.raises(TeacherCompatibilityError, match="mixed.*layout"):
        load_teacher(
            TeacherSpec(checkpoint=checkpoint, blocks=1, quantity=True),
            ModelConfig.control(blocks=1, channels=4),
        )


@pytest.mark.parametrize("envelope", ["learner", "state_dict"])
def test_teacher_rejects_bare_weights_mixed_with_an_envelope(
    tmp_path: Path, envelope: str
) -> None:
    """Envelope selection cannot silently discard a second plausible policy."""
    state = _teacher_state()
    nested = (
        state
        if envelope == "learner"
        else {f"policy.{name}": value for name, value in state.items()}
    )
    checkpoint = tmp_path / "teacher.pt"
    torch.save({**state, envelope: nested}, checkpoint)

    with pytest.raises(TeacherCompatibilityError, match="mixed.*layout"):
        load_teacher(
            TeacherSpec(checkpoint=checkpoint, blocks=1, quantity=True),
            ModelConfig.control(blocks=1, channels=4),
        )


@pytest.mark.parametrize("payload", [[], {"learner": []}, {"state_dict": []}])
def test_teacher_rejects_malformed_non_mapping_policy_payloads(
    tmp_path: Path, payload: object
) -> None:
    """Malformed historical containers fail as compatibility errors."""
    checkpoint = tmp_path / "teacher.pt"
    torch.save(payload, checkpoint)

    with pytest.raises(TeacherCompatibilityError, match="mapping"):
        load_teacher(
            TeacherSpec(checkpoint=checkpoint, blocks=1),
            ModelConfig.control(blocks=1, channels=4),
        )


def test_teacher_wraps_a_corrupt_serialized_checkpoint(tmp_path: Path) -> None:
    """Unreadable bytes fail through the stable teacher compatibility API."""
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.write_bytes(b"not a torch checkpoint")

    with pytest.raises(TeacherCompatibilityError, match="cannot be loaded"):
        load_teacher(
            TeacherSpec(checkpoint=checkpoint, blocks=1),
            ModelConfig.control(blocks=1, channels=4),
        )


@pytest.mark.parametrize("layout", ["bare", "learner", "lightning"])
def test_teacher_loads_each_explicit_historical_policy_envelope_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
) -> None:
    """Envelope extraction must not deserialize the teacher a second time."""
    state = _teacher_state()
    checkpoint = tmp_path / "teacher.pt"
    if layout == "bare":
        payload: object = state
    elif layout == "learner":
        payload = {"learner": state, "steps": 1}
    else:
        payload = {
            "state_dict": {
                **{f"policy.{name}": value for name, value in state.items()},
                "teacher_policy.ignored": torch.ones(1),
            }
        }
    torch.save(payload, checkpoint)
    calls: list[tuple[Path, bool | None]] = []
    real_load = torch.load

    def counted_load(path: Path, *, map_location: str, weights_only: bool) -> object:
        calls.append((path, weights_only))
        return real_load(
            path,
            map_location=map_location,
            weights_only=weights_only,
        )

    monkeypatch.setattr(torch, "load", counted_load)

    loaded = load_teacher(
        TeacherSpec(checkpoint=checkpoint, blocks=1, quantity=True),
        ModelConfig.control(blocks=1, channels=4),
    )

    assert loaded.actual_heads == ("operation", "quantity", "market")
    assert calls == [(checkpoint, True)]


def test_initial_population_checkpoint_must_be_readable(tmp_path: Path) -> None:
    """A declared initial member fails config validation before pool mutation."""
    with pytest.raises(ValidationError, match="initial snapshot is not readable"):
        ToadConfig(
            population={
                "selfplay": 0.0,
                "scripted": 0.0,
                "frozen_opponent": 1.0,
                "initial_snapshots": [tmp_path / "missing.pt"],
            }
        )


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


@pytest.mark.parametrize(
    "kind",
    ["absolute", "traversal", "nested", "wrong_steps", "wrong_digest"],
)
def test_pool_load_refuses_malformed_reloaded_manifest_members(
    tmp_path: Path, kind: str
) -> None:
    """Pool sampling cannot turn a reloaded manifest path into a torch load."""
    pool_directory = tmp_path / "pool"
    pool_directory.mkdir()
    source = tmp_path / "source.pt"
    torch.save(state_dict(), source)
    digest = sha256_file(source)
    name = f"snapshot-000000000001-{digest[:12]}.pt"
    if kind == "absolute":
        candidate = tmp_path / "outside" / name
    elif kind == "traversal":
        candidate = pool_directory / ".." / "outside" / name
    elif kind == "nested":
        candidate = pool_directory / "nested" / name
    elif kind == "wrong_steps":
        candidate = pool_directory / f"snapshot-000000000002-{digest[:12]}.pt"
    else:
        wrong_prefix = "0" if digest[0] != "0" else "1"
        candidate = pool_directory / (
            f"snapshot-000000000001-{wrong_prefix}{digest[1:12]}.pt"
        )
    candidate.parent.mkdir(parents=True, exist_ok=True)
    source.replace(candidate)
    entry = manifest_entry(candidate, digest=digest)
    (pool_directory / "manifest.json").write_text(
        SnapshotManifest(entries=(entry,)).model_dump_json()
    )

    reloaded = SnapshotStore(pool_directory, capacity=1, structure="model-v1")
    pool = SnapshotPool.from_store(reloaded, seed=17)
    selected = pool.sample(game_id=91)
    with pytest.raises(SnapshotIntegrityError, match="manifest snapshot path"):
        pool.load(selected)

    assert candidate.is_file()


def test_store_round_trips_and_revalidates_large_environment_step_names(
    tmp_path: Path,
) -> None:
    """Nonnegative steps above twelve digits keep their exact canonical spelling."""
    steps = 1_000_000_000_000
    store = SnapshotStore(tmp_path, capacity=2, structure="model-v1")
    entry = store.add(state_dict(), environment_steps=steps, round_id=0, run_id="run")
    reloaded = SnapshotStore(tmp_path, capacity=2, structure="model-v1")

    assert entry.path.name == f"snapshot-{steps:012d}-{entry.sha256[:12]}.pt"
    loaded = reloaded.load(reloaded.manifest.entries[0])
    assert torch.equal(loaded["weight"], torch.tensor([1.0]))
    reloaded.add(state_dict(2.0), environment_steps=steps + 1, round_id=1, run_id="run")


def test_store_rejects_an_alternate_large_step_filename_spelling(
    tmp_path: Path,
) -> None:
    """Equivalent-looking leading-zero step spellings cannot name pool members."""
    steps = 1_000_000_000_000
    store = SnapshotStore(tmp_path, capacity=1, structure="model-v1")
    entry = store.add(state_dict(), environment_steps=steps, round_id=0, run_id="run")
    alternate = entry.path.with_name(f"snapshot-0{steps:012d}-{entry.sha256[:12]}.pt")
    entry.path.rename(alternate)
    malformed = entry.model_copy(update={"path": alternate})
    (tmp_path / "manifest.json").write_text(
        SnapshotManifest(entries=(malformed,)).model_dump_json()
    )
    reloaded = SnapshotStore(tmp_path, capacity=1, structure="model-v1")

    with pytest.raises(SnapshotIntegrityError, match="manifest snapshot path"):
        reloaded.add(
            state_dict(2.0), environment_steps=steps + 1, round_id=1, run_id="run"
        )


def test_population_sampling_is_deterministic_by_game_id(tmp_path: Path) -> None:
    """A game ID selects the same member independently of call ordering."""
    pool = populated_pool(tmp_path, count=3, seed=17)

    assert pool.sample(game_id=91) == pool.sample(game_id=91)


def test_pool_load_verifies_each_manifest_digest_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One store-bound load checks every member once and materializes the selection."""
    store = SnapshotStore(tmp_path, capacity=2, structure="model-v1")
    first = store.add(state_dict(1.0), environment_steps=1, round_id=0, run_id="run")
    second = store.add(state_dict(2.0), environment_steps=2, round_id=0, run_id="run")
    pool = SnapshotPool.from_store(store, seed=17)
    calls: list[Path] = []
    real_hash = sha256_file

    def counted_hash(path: Path) -> str:
        calls.append(path)
        return real_hash(path)

    monkeypatch.setattr("kaggriculture.learn.toad.population.sha256_file", counted_hash)

    loaded = pool.load(first)

    assert torch.equal(loaded["weight"], torch.tensor([1.0]))
    assert calls.count(first.path) == 1
    assert calls.count(second.path) == 1


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
    pool = SnapshotPool.from_store(store, seed=17)
    selected.path.write_bytes(b"corrupt")

    with pytest.raises(SnapshotIntegrityError, match=selected.sha256):
        pool.load(selected)


def test_structure_mismatch_is_rejected_before_loading(tmp_path: Path) -> None:
    """A valid foreign snapshot cannot cross model-shape compatibility boundaries."""
    store = SnapshotStore(tmp_path, capacity=1, structure="model-v1")
    selected = store.add(state_dict(), environment_steps=1, round_id=0, run_id="run")
    pool = SnapshotPool.from_store(
        SnapshotStore(tmp_path, capacity=1, structure="model-v2"), seed=17
    )

    with pytest.raises(SnapshotIntegrityError, match="model-v2"):
        pool.load(selected)
