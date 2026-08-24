"""Native tensor-simulator collection behind the common learner boundary."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import kaggriculture.learn.toad.data as data_module
from kaggriculture.learn.model import Policy
from kaggriculture.learn.scripts import toad as toad_script
from kaggriculture.learn.toad.config import (
    ModelConfig,
    ToadConfig,
    structural_fingerprint,
)
from kaggriculture.learn.toad.data import (
    BatchKind,
    CollectionAssignment,
    LearnerBatch,
    NativeRoundSource,
)
from kaggriculture.learn.toad.population import SnapshotPool, SnapshotStore, sha256_file


def _native_config(*, model: ModelConfig | None = None) -> ToadConfig:
    """Return one tiny round whose native segments form one optimizer batch."""
    return ToadConfig.model_validate(
        {
            "model": (model or ModelConfig.control(blocks=1, channels=4)).model_dump(),
            "population": {
                "selfplay": 1.0,
                "scripted": 0.0,
                "environments_per_rank": 1,
                "collection_processes": 1,
            },
            "optimizer": {
                "unroll_length": 2,
                "batch_segments": 4,
                "value_passes": 1,
            },
            "runtime": {
                "rollout_backend": "native",
                "rollout_device": "cpu",
            },
        }
    )


def test_native_runtime_configuration_is_typed_and_serializable() -> None:
    """A native request must survive the sole immutable experiment contract."""
    config = _native_config()

    assert config.runtime.rollout_backend == "native"
    assert config.runtime.rollout_device == "cpu"
    assert ToadConfig.model_validate(config.model_dump()) == config


def test_data_module_selects_the_requested_native_source() -> None:
    """Selecting native collection must not silently build the reference source."""
    data = toad_script.build_reference_data_module(_native_config())

    assert isinstance(data.source, NativeRoundSource)


def test_preflight_rejects_unproved_native_compile_without_fallback() -> None:
    """Native actor compilation cannot silently become eager collection."""
    payload = _native_config().model_dump(mode="python")
    payload["runtime"]["compile"]["enabled"] = True
    config = ToadConfig.model_validate(payload)

    with pytest.raises(
        toad_script.RuntimePreflightError,
        match=r"native rollout.*torch\.compile",
    ):
        toad_script.runtime_preflight(config)


def test_preflight_rejects_scripted_cuda_graph_without_fallback() -> None:
    """A host-driven scripted opponent must abort a requested graph mode."""
    payload = _native_config().model_dump(mode="python")
    payload["runtime"]["rollout_device"] = "cuda"
    payload["runtime"]["rollout_cuda_graph"] = True
    payload["population"].update(selfplay=0.0, scripted=1.0)
    config = ToadConfig.model_validate(payload)

    with pytest.raises(
        toad_script.RuntimePreflightError,
        match="scripted CUDA-graph rollout",
    ):
        toad_script.runtime_preflight(config)


def test_native_source_emits_common_batches_with_exact_selfplay_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native collection must not need a reference Trajectory conversion."""
    monkeypatch.setattr(data_module, "EPISODE_STEPS", 5)
    config = _native_config()
    assignment = CollectionAssignment(
        game_id=41,
        seed=307,
        kind=BatchKind.SELFPLAY,
        opponent_id="self",
    )
    source = NativeRoundSource(config, assignments=(assignment,))
    actor = Policy(blocks=1, channels=4)
    source.publish_actor(actor.state_dict(), version=7)

    batches = list(source)

    assert len(batches) == 2
    assert all(isinstance(batch, LearnerBatch) for batch in batches)
    assert [batch.baseline_only for batch in batches] == [False, True]
    assert batches[0].first_of_round
    assert batches[-1].end_of_round
    assert batches[0].collected_steps == 8
    assert batches[0].round_id == 0
    assert batches[0].actor_version == 7
    assert batches[0].game_ids == (41,)
    assert batches[0].seeds == (307,)
    assert batches[0].opponent_ids == ("self",)
    assert batches[0].segment_game_ids == (41, 41, 41, 41)
    assert batches[0].segment_seeds == (307, 307, 307, 307)
    assert batches[0].segment_opponent_ids == ("self",) * 4
    assert batches[0].segment_kinds == (BatchKind.SELFPLAY,) * 4
    assert all(
        tensor.device.type == "cpu"
        for segment in batches[0].segments
        for tensor in segment.values()
    )
    required = {
        "board",
        "scalars",
        "positions",
        "unit_actions",
        "unit_quantities",
        "market_actions",
        "unit_masks",
        "unit_quantity_masks",
        "market_masks",
        "log_probs",
        "shaped",
        "shaped_money",
        "margin",
        "own",
        "sparse",
        "dones",
        "unit_valid",
    }
    assert all(required <= segment.keys() for segment in batches[0].segments)


def test_native_source_preserves_sparse_recurrent_and_belief_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Entry state and labels must reach the learner without dense actor memory."""
    monkeypatch.setattr(data_module, "EPISODE_STEPS", 5)
    model = ModelConfig.control(blocks=1, channels=4).model_copy(
        update={
            "recurrent": True,
            "recurrent_channels": 3,
            "belief": True,
        }
    )
    config = _native_config(model=model)
    assignment = CollectionAssignment(
        game_id=43,
        seed=311,
        kind=BatchKind.SELFPLAY,
        opponent_id="self",
    )
    source = NativeRoundSource(config, assignments=(assignment,))
    from kaggriculture.learn.toad.model import StatefulPolicy

    actor = StatefulPolicy(model).eval()
    source.publish_actor(actor.state_dict(), version=9)

    policy_batch = next(batch for batch in source if not batch.baseline_only)

    assert policy_batch.actor_version == 9
    for segment in policy_batch.segments:
        assert segment["initial_hidden"].shape == (3, 10, 10)
        assert segment["initial_cell"].shape == (3, 10, 10)
        assert segment["initial_belief"].shape == (model.belief_size,)
        assert segment["belief_targets"].shape == (2, model.belief_size)
        assert segment["belief_valid"].dtype is torch.bool
        assert segment["belief_valid"].all()


def test_native_source_collects_every_online_kind_with_verified_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Selfplay, script, frozen, and teacher all retain exact actor provenance."""
    monkeypatch.setattr(data_module, "EPISODE_STEPS", 3)
    actor = Policy(blocks=1, channels=4)
    teacher_path = tmp_path / "teacher.pt"
    torch.save(actor.state_dict(), teacher_path)
    config = ToadConfig.model_validate(
        {
            "model": ModelConfig.control(blocks=1, channels=4).model_dump(),
            "population": {
                "selfplay": 0.25,
                "scripted": 0.25,
                "frozen_opponent": 0.25,
                "teacher_distill": 0.25,
                "scripted_opponent": "economic",
                "teacher": {
                    "checkpoint": teacher_path,
                    "blocks": 1,
                    "operation": True,
                    "quantity": True,
                    "market": True,
                },
                "environments_per_rank": 4,
                "collection_processes": 1,
                "snapshot_at_start": True,
            },
            "optimizer": {
                "unroll_length": 2,
                "batch_segments": 5,
                "value_passes": 0,
            },
            "runtime": {
                "rollout_backend": "native",
                "rollout_device": "cpu",
            },
        }
    )
    store = SnapshotStore(
        tmp_path / "pool",
        capacity=1,
        structure=structural_fingerprint(config),
    )
    frozen = store.add(
        actor.state_dict(), environment_steps=1, round_id=0, run_id="frozen-run"
    )
    pool = SnapshotPool.from_store(store, seed=config.population.population_seed)
    source = NativeRoundSource(config, pool=pool)
    source.publish_actor(actor.state_dict(), version=13)

    batches = list(source)

    assert len(batches) == 1
    batch = batches[0]
    assert batch.first_of_round and batch.end_of_round
    assert batch.actor_version == 13
    assert set(batch.segment_kinds) == {
        BatchKind.SELFPLAY,
        BatchKind.SCRIPTED,
        BatchKind.FROZEN_OPPONENT,
        BatchKind.TEACHER_DISTILL,
    }
    identities = set(
        zip(
            batch.segment_kinds,
            batch.segment_opponent_ids,
            batch.segment_opponent_digests,
            strict=True,
        )
    )
    assert (BatchKind.SELFPLAY, "self", None) in identities
    assert (BatchKind.SCRIPTED, "economic", None) in identities
    assert any(
        kind is BatchKind.FROZEN_OPPONENT
        and opponent.startswith("snapshot:frozen-run:1:")
        and digest == frozen.sha256
        for kind, opponent, digest in identities
    )
    assert any(
        kind is BatchKind.TEACHER_DISTILL
        and opponent.startswith("teacher:teacher.pt:")
        and digest == sha256_file(teacher_path)
        for kind, opponent, digest in identities
    )
    assert batch.collected_steps == 10
    assert sorted(batch.game_ids) == [0, 1, 2, 3]
    assert source.next_game_id == 4


def test_native_source_batches_compatible_assignments_in_one_actor_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two self-play games cost one vectorized forward per simulated turn."""
    monkeypatch.setattr(data_module, "EPISODE_STEPS", 3)
    config = _native_config()
    assignments = tuple(
        CollectionAssignment(
            game_id=game_id,
            seed=401 + game_id,
            kind=BatchKind.SELFPLAY,
            opponent_id="self",
        )
        for game_id in (51, 52)
    )
    source = NativeRoundSource(config, assignments=assignments)

    class CountingPolicy(Policy):
        calls = 0

        def forward(
            self,
            board: torch.Tensor,
            scalars: torch.Tensor,
            positions: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            self.calls += 1
            return super().forward(board, scalars, positions)

    actor = CountingPolicy(blocks=1, channels=4).eval()
    source.publish_actor(actor.state_dict(), version=17)
    monkeypatch.setattr(source, "_native_policy", lambda _model, _state: actor)

    batches = list(source)

    assert actor.calls == 2
    assert batches[0].game_ids == (51, 52)
    assert batches[0].collected_steps == 8
