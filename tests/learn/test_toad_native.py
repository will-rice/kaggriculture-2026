"""Native tensor-simulator collection behind the common learner boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import pytest
import torch

import kaggriculture.learn.rollout as reference_rollout
import kaggriculture.learn.toad.data as data_module
import kaggriculture.sim.day as sim_day
from kaggriculture.learn.encoding import QUANTITIES, UNIT_OPS
from kaggriculture.learn.model import Policy
from kaggriculture.learn.rollout import Trajectory as ReferenceTrajectory
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
    ReferenceRoundSource,
)
from kaggriculture.learn.toad.model import PolicyOutput, PolicyState, StatefulPolicy
from kaggriculture.learn.toad.population import SnapshotPool, SnapshotStore, sha256_file
from kaggriculture.sim.rollout import Trajectory as NativeTrajectory


def _native_config(*, model: ModelConfig | None = None) -> ToadConfig:
    """Return one tiny round whose native segments form one optimizer batch."""
    return ToadConfig.model_validate(
        {
            "model": (model or ModelConfig.control(blocks=1, channels=4)).model_dump(),
            "population": {
                "selfplay": 1.0,
                "scripted": 0.0,
                # Two full self-play environments resolve to 1,436 segments,
                # exactly divisible by the four-segment optimizer contract.
                "environments_per_rank": 2,
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


class _DeterministicLegalPolicy(StatefulPolicy):
    """Keep a real recurrent/value/belief path while forcing a legal PASS tape."""

    def forward(
        self,
        board: torch.Tensor,
        scalars: torch.Tensor,
        positions: torch.Tensor,
        state: PolicyState | None = None,
        dones: torch.Tensor | None = None,
    ) -> PolicyOutput:
        output = super().forward(board, scalars, positions, state=state, dones=dones)
        units = torch.full_like(output.unit_logits, -torch.inf)
        quantities = torch.full_like(output.quantity_logits, -torch.inf)
        markets = torch.full_like(output.market_logits, -torch.inf)
        units[..., UNIT_OPS.index("PASS")] = 0.0
        quantities[..., QUANTITIES.index(1)] = 0.0
        markets[..., QUANTITIES.index(0)] = 0.0
        return PolicyOutput(
            units,
            quantities,
            markets,
            output.values,
            output.belief_logits,
            output.state,
            output.input_state,
        )


class _NativeCollectionLike(Protocol):
    """Narrow structural view used by the reference-metric test oracle."""

    @property
    def trajectories(self) -> tuple[NativeTrajectory, ...]: ...


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
    payload["optimizer"]["batch_segments"] = 2
    config = ToadConfig.model_validate(payload)

    with pytest.raises(
        toad_script.RuntimePreflightError,
        match="scripted CUDA-graph rollout",
    ):
        toad_script.runtime_preflight(config)


def test_preflight_rejects_unverified_native_eager_scripted_identity() -> None:
    """Enabled native script collection must fail before constructing actors."""
    payload = _native_config().model_dump(mode="python")
    payload["population"].update(
        selfplay=0.0,
        scripted=1.0,
        scripted_opponent="starter",
    )
    payload["optimizer"]["batch_segments"] = 2
    config = ToadConfig.model_validate(payload)

    with pytest.raises(
        toad_script.RuntimePreflightError,
        match=r"native scripted rollout.*economic",
    ):
        toad_script.runtime_preflight(config)


def test_preflight_ignores_disabled_native_scripted_identity() -> None:
    """An inert script field does not constrain a self-play-only native run."""
    payload = _native_config().model_dump(mode="python")
    payload["population"]["scripted_opponent"] = "starter"
    config = ToadConfig.model_validate(payload)

    toad_script.runtime_preflight(config)


def test_native_metrics_preserve_reference_outcomes_and_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native collection exposes every accepted legacy round metric directly."""
    monkeypatch.setattr(data_module, "EPISODE_STEPS", 3)
    config = _native_config()
    assignments = (
        CollectionAssignment(
            game_id=31,
            seed=211,
            kind=BatchKind.SELFPLAY,
            opponent_id="self",
        ),
        CollectionAssignment(
            game_id=32,
            seed=223,
            kind=BatchKind.SCRIPTED,
            opponent_id="economic",
        ),
    )
    source = NativeRoundSource(config, assignments=assignments)
    actor = Policy(blocks=1, channels=4)
    source.publish_actor(actor.state_dict(), version=3)

    collected = source._collect_native_round(assignments)
    metrics = source._native_metrics(collected, 0.25)

    legacy = {
        "diag/bank_mean",
        "diag/bank_max",
        "diag/bank_mirror",
        "diag/bank_vs_econ",
        "diag/n_econ_envs",
        "objective/win_rate_vs_econ",
        "diag/mirror_decisive_rate",
        "objective/margin_vs_econ",
        "diag/margin_mean_mirror",
        "diag/mean_sale_price_vs_econ",
        "diag/realisation_vs_econ",
        "diag/sales_vs_econ",
        "diag/units_sold_vs_econ",
        "diag/bought_vs_econ",
        "diag/final_capital_vs_econ",
        "diag/mean_sale_price_mirror",
        "diag/realisation_mirror",
        "diag/sales_mirror",
        "diag/units_sold_mirror",
        "diag/bought_mirror",
        "diag/final_capital_mirror",
        "critic/ev_econ_games",
        "critic/ev_within_turn_econ_games",
        "critic/ev_mirror_games",
        "critic/ev_within_turn_mirror_games",
        "proxy/shaped_mean",
        "proxy/shaped_reward_mean",
        "diag/illegal",
        "diag/gross_purchases",
        "proxy/money_term",
    }
    assert legacy <= metrics.keys()
    assert metrics["diag/n_econ_envs"] == 1
    assert metrics["collection/games/selfplay"] == 1
    assert metrics["collection/return_count_by_opponent/self"] == 2
    assert metrics["collection/games/scripted"] == 1
    assert metrics["collection/return_count_by_opponent/economic"] == 1

    def reference_stream(
        collection: _NativeCollectionLike, environment: int, seat: int
    ) -> ReferenceTrajectory:
        chunks = collection.trajectories
        final = chunks[-1]

        def metric(chunk: NativeTrajectory, name: str) -> torch.Tensor:
            value = getattr(chunk, name)
            assert isinstance(value, torch.Tensor)
            return value

        units = torch.stack(
            [metric(chunk, "units_sold")[environment, seat] for chunk in chunks]
        ).sum()
        proceeds = torch.stack(
            [metric(chunk, "sale_proceeds")[environment, seat] for chunk in chunks]
        ).sum()
        market_value = torch.stack(
            [metric(chunk, "sale_market_value")[environment, seat] for chunk in chunks]
        ).sum()
        mean_sale = float(torch.where(units > 0, proceeds / units, 0.0))
        mean_market = torch.where(units > 0, market_value / units, 0.0)
        return ReferenceTrajectory(
            **{
                name: torch.cat(
                    [getattr(chunk, name)[:, environment, seat] for chunk in chunks]
                )
                for name in (
                    "unit_actions",
                    "unit_quantities",
                    "market_actions",
                    "unit_masks",
                    "unit_quantity_masks",
                    "market_masks",
                    "log_probs",
                    "values",
                    "rewards",
                    "own",
                    "shaped",
                    "shaped_money",
                    "margin",
                    "sparse",
                    "potentials",
                    "dones",
                )
            },
            **{
                name: torch.cat(
                    [getattr(chunk, name)[:-1, environment, seat] for chunk in chunks]
                )
                for name in ("board", "scalars", "positions")
            },
            final_margin=float(metric(final, "final_margin")[environment, seat]),
            final_bank=float(metric(final, "final_bank")[environment, seat]),
            final_capital=float(metric(final, "final_capital")[environment, seat]),
            illegal=int(
                torch.stack(
                    [
                        metric(chunk, "illegal_by_stream")[environment, seat]
                        for chunk in chunks
                    ]
                ).sum()
            ),
            sales=float(
                torch.stack(
                    [metric(chunk, "sales")[environment, seat] for chunk in chunks]
                ).sum()
            ),
            units_sold=float(units),
            mean_sale_price=mean_sale,
            realisation=float(
                torch.where(mean_market > 0, mean_sale / mean_market, 0.0)
            ),
            bought=float(
                torch.stack(
                    [metric(chunk, "bought")[environment, seat] for chunk in chunks]
                ).sum()
            ),
        )

    mirror = [reference_stream(collected[0], 0, seat) for seat in range(2)]
    econ = [reference_stream(collected[1], 0, 0)]
    expected = toad_script._collection_metrics(
        mirror, econ, config.curriculum.reward_field
    )
    for name, value in expected.items():
        if isinstance(value, float) and torch.isnan(torch.tensor(value)):
            assert torch.isnan(torch.tensor(metrics[name]))
        else:
            assert metrics[name] == pytest.approx(value)


@pytest.mark.parametrize("kind", [BatchKind.SELFPLAY, BatchKind.SCRIPTED])
def test_reference_and_native_collectors_emit_identical_stateful_batches(  # noqa: C901
    kind: BatchKind,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fixed legal actions prove collection and adaptation, not only sim.step."""
    monkeypatch.setattr(reference_rollout, "EPISODE_STEPS", 5)
    monkeypatch.setattr(data_module, "EPISODE_STEPS", 5)
    monkeypatch.setattr(sim_day, "EPISODE_STEPS", 5)
    model = ModelConfig.control(blocks=1, channels=4).model_copy(
        update={"recurrent": True, "recurrent_channels": 3, "belief": True}
    )
    payload = _native_config(model=model).model_dump(mode="python")
    payload["optimizer"].update(unroll_length=2, batch_segments=2, value_passes=0)
    native_config = ToadConfig.model_validate(payload)
    payload["runtime"]["rollout_backend"] = "reference"
    reference_config = ToadConfig.model_validate(payload)
    assignment = CollectionAssignment(
        game_id=61,
        seed=277,
        kind=kind,
        opponent_id="self" if kind is BatchKind.SELFPLAY else "economic",
    )
    reference_actor = _DeterministicLegalPolicy(model).eval()
    native_actor = _DeterministicLegalPolicy(model).eval()
    native_actor.load_state_dict(reference_actor.state_dict())
    reference_trajectories: list[ReferenceTrajectory] = []
    scripted_path = tmp_path / "pass_agent.py"
    scripted_path.write_text(
        "def agent(observation, configuration=None):\n"
        "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
    )

    def pass_script(_observation: object) -> dict[str, object]:
        return {"farmer": ["PASS"], "hands": [], "market": []}

    def collect_reference(
        received: CollectionAssignment,
    ) -> tuple[ReferenceTrajectory, ...]:
        opponent: Any = (
            reference_actor
            if received.kind is BatchKind.SELFPLAY
            else str(scripted_path)
        )
        rows = tuple(
            reference_rollout.rollout_many(
                reference_actor,
                opponent,
                (received.seed,),
                state_unroll_length=2,
            )
        )
        reference_trajectories.extend(rows)
        return rows

    reference_source = ReferenceRoundSource(
        reference_config,
        assignments=(assignment,),
        collect_assignment=collect_reference,
    )
    native_source = NativeRoundSource(native_config, assignments=(assignment,))
    reference_source.publish_actor(reference_actor.state_dict(), version=19)
    native_source.publish_actor(native_actor.state_dict(), version=19)
    monkeypatch.setattr(
        native_source,
        "_native_policy",
        lambda _model, _state: native_actor,
    )
    if kind is BatchKind.SCRIPTED:
        monkeypatch.setattr(native_source, "_scripted_opponent", lambda: pass_script)
    native_collections = []
    collect_native = native_source._collect_native_round

    def record_native(
        received: tuple[CollectionAssignment, ...],
    ) -> list[Any]:
        rows = collect_native(received)
        native_collections.extend(rows)
        return rows

    monkeypatch.setattr(native_source, "_collect_native_round", record_native)

    reference_batches = list(reference_source)
    native_batches = list(native_source)

    assert len(native_collections) == 1
    native_collection = native_collections[0]
    assert len(reference_trajectories) == native_collection.recorded_seats
    for seat, reference in enumerate(reference_trajectories):
        chunks = native_collection.trajectories
        for name in ("board", "scalars", "positions"):
            actual = torch.cat([getattr(chunk, name)[:-1, 0, seat] for chunk in chunks])
            torch.testing.assert_close(actual, getattr(reference, name))
        for name in (
            "unit_actions",
            "unit_quantities",
            "market_actions",
            "unit_masks",
            "unit_quantity_masks",
            "market_masks",
            "log_probs",
            "values",
            "rewards",
            "own",
            "shaped",
            "shaped_money",
            "margin",
            "sparse",
            "potentials",
            "dones",
            "belief_targets",
            "belief_valid",
        ):
            actual = torch.cat([getattr(chunk, name)[:, 0, seat] for chunk in chunks])
            torch.testing.assert_close(actual, getattr(reference, name))
        native_own = torch.cat([chunk.own[:, 0, seat] for chunk in chunks])
        native_initial_bank = float(chunks[-1].final_bank[0, seat]) - float(
            native_own.sum()
        )
        reference_initial_bank = reference.final_bank - float(reference.own.sum())
        torch.testing.assert_close(
            native_initial_bank + native_own.cumsum(0),
            reference_initial_bank + reference.own.cumsum(0),
        )
        assert float(chunks[-1].final_bank[0, seat]) == pytest.approx(
            reference.final_bank
        )
        assert float(chunks[-1].final_margin[0, seat]) == pytest.approx(
            reference.final_margin
        )

    reference_policy = [batch for batch in reference_batches if not batch.baseline_only]
    native_policy = [batch for batch in native_batches if not batch.baseline_only]
    assert len(reference_policy) == len(native_policy)
    for reference_batch, native_batch in zip(
        reference_policy, native_policy, strict=True
    ):
        assert native_batch.actor_version == reference_batch.actor_version == 19
        assert native_batch.game_ids == reference_batch.game_ids == (61,)
        assert native_batch.seeds == reference_batch.seeds == (277,)
        assert native_batch.opponent_ids == reference_batch.opponent_ids
        assert native_batch.segment_kinds == reference_batch.segment_kinds
        assert native_batch.segment_game_ids == reference_batch.segment_game_ids
        assert native_batch.segment_seeds == reference_batch.segment_seeds
        assert native_batch.segment_opponent_ids == reference_batch.segment_opponent_ids
        for reference_segment, native_segment in zip(
            reference_batch.segments, native_batch.segments, strict=True
        ):
            assert native_segment.keys() == reference_segment.keys()
            for name in reference_segment:
                torch.testing.assert_close(
                    native_segment[name], reference_segment[name]
                )
    assert (
        native_batches[0].round_metrics.keys()
        == reference_batches[0].round_metrics.keys()
    )
    for name, value in reference_batches[0].round_metrics.items():
        actual = native_batches[0].round_metrics[name]
        if isinstance(value, float) and torch.isnan(torch.tensor(value)):
            assert torch.isnan(torch.tensor(actual))
        elif name != "throughput/collection_seconds":
            assert actual == pytest.approx(value)


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


def test_stochastic_reference_sampling_is_invariant_to_regrouping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each game's action stream is keyed by its seed, not its vector peers."""
    monkeypatch.setattr(reference_rollout, "EPISODE_STEPS", 5)
    torch.manual_seed(991)
    policy = Policy(blocks=1, channels=4).eval()
    seeds = (401, 409)

    grouped = reference_rollout.rollout_many(policy, policy, seeds)
    separate = [
        trajectory
        for seed in seeds
        for trajectory in reference_rollout.rollout_many(policy, policy, (seed,))
    ]

    assert len(grouped) == len(separate) == 4
    for together, alone in zip(grouped, separate, strict=True):
        torch.testing.assert_close(together.unit_actions, alone.unit_actions)
        torch.testing.assert_close(together.unit_quantities, alone.unit_quantities)
        torch.testing.assert_close(together.market_actions, alone.market_actions)
        torch.testing.assert_close(together.log_probs, alone.log_probs)


def test_stochastic_native_sampling_is_invariant_to_regrouping_and_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vectorized native rows retain each game's reference stochastic stream."""
    monkeypatch.setattr(reference_rollout, "EPISODE_STEPS", 5)
    monkeypatch.setattr(data_module, "EPISODE_STEPS", 5)
    monkeypatch.setattr(sim_day, "EPISODE_STEPS", 5)

    class FixedStochasticPolicy(Policy):
        def forward(
            self,
            board: torch.Tensor,
            scalars: torch.Tensor,
            positions: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            units, quantities, markets, values = super().forward(
                board, scalars, positions
            )

            def fixed(logits: torch.Tensor) -> torch.Tensor:
                axis = torch.linspace(
                    -0.7,
                    0.9,
                    logits.shape[-1],
                    dtype=logits.dtype,
                    device=logits.device,
                )
                return axis.expand_as(logits)

            return fixed(units), fixed(quantities), fixed(markets), values

    torch.manual_seed(997)
    policy = FixedStochasticPolicy(blocks=1, channels=4).eval()
    seeds = (421, 431)
    assignments = tuple(
        CollectionAssignment(
            game_id=91 + index,
            seed=seed,
            kind=BatchKind.SELFPLAY,
            opponent_id="self",
        )
        for index, seed in enumerate(seeds)
    )

    grouped_source = NativeRoundSource(_native_config(), assignments=assignments)
    grouped_source.publish_actor(policy.state_dict(), version=0)
    monkeypatch.setattr(grouped_source, "_native_policy", lambda _model, _state: policy)
    grouped = grouped_source._collect_native_round(assignments)[0]
    separate = []
    for assignment in assignments:
        source = NativeRoundSource(_native_config(), assignments=(assignment,))
        source.publish_actor(policy.state_dict(), version=0)
        monkeypatch.setattr(source, "_native_policy", lambda _model, _state: policy)
        separate.append(source._collect_native_round((assignment,))[0])
    reference = reference_rollout.rollout_many(policy, policy, seeds)

    for environment, collection in enumerate(separate):
        for seat in range(2):
            grouped_actions = torch.cat(
                [
                    chunk.unit_actions[:, environment, seat]
                    for chunk in grouped.trajectories
                ]
            )
            separate_actions = torch.cat(
                [chunk.unit_actions[:, 0, seat] for chunk in collection.trajectories]
            )
            separate_quantities = torch.cat(
                [chunk.unit_quantities[:, 0, seat] for chunk in collection.trajectories]
            )
            separate_market = torch.cat(
                [chunk.market_actions[:, 0, seat] for chunk in collection.trajectories]
            )
            grouped_quantities = torch.cat(
                [
                    chunk.unit_quantities[:, environment, seat]
                    for chunk in grouped.trajectories
                ]
            )
            grouped_market = torch.cat(
                [
                    chunk.market_actions[:, environment, seat]
                    for chunk in grouped.trajectories
                ]
            )
            row = environment * 2 + seat
            torch.testing.assert_close(grouped_actions, separate_actions)
            torch.testing.assert_close(grouped_actions, reference[row].unit_actions)
            torch.testing.assert_close(grouped_quantities, separate_quantities)
            torch.testing.assert_close(
                grouped_quantities, reference[row].unit_quantities
            )
            torch.testing.assert_close(grouped_market, separate_market)
            torch.testing.assert_close(grouped_market, reference[row].market_actions)


def test_native_grouped_row_failure_identifies_the_second_game(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A row-addressable vector failure must not be attributed to chunk row zero."""
    assignments = (
        CollectionAssignment(71, 501, BatchKind.SELFPLAY, "self"),
        CollectionAssignment(72, 503, BatchKind.SELFPLAY, "self"),
    )
    source = NativeRoundSource(_native_config(), assignments=assignments)
    actor = Policy(blocks=1, channels=4)
    source.publish_actor(actor.state_dict(), version=0)

    def fail_second(*_args: object, **_kwargs: object) -> object:
        raise data_module.CollectionRowError(
            row=1,
            error_type="ValueError",
            message="second row failed",
        )

    monkeypatch.setattr("kaggriculture.sim.rollout.collect_segment", fail_second)

    with pytest.raises(data_module.CollectionError) as raised:
        source._collect_native_round(assignments)

    assert raised.value.game_ids == (72,)
    assert raised.value.seeds == (503,)
    assert "second row failed" in str(raised.value)


def test_native_group_failure_reports_every_affected_game(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuinely vector-wide failure names the full affected assignment set."""
    assignments = (
        CollectionAssignment(81, 601, BatchKind.SELFPLAY, "self"),
        CollectionAssignment(82, 607, BatchKind.SELFPLAY, "self"),
    )
    source = NativeRoundSource(_native_config(), assignments=assignments)
    actor = Policy(blocks=1, channels=4)
    source.publish_actor(actor.state_dict(), version=0)
    monkeypatch.setattr(
        "kaggriculture.sim.rollout.collect_segment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("vector failed")),
    )

    with pytest.raises(data_module.CollectionError) as raised:
        source._collect_native_round(assignments)

    assert raised.value.game_ids == (81, 82)
    assert raised.value.seeds == (601, 607)


def test_native_second_row_failure_survives_distributed_error_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every DDP peer receives the second grouped game's exact provenance."""
    assignments = (
        CollectionAssignment(101, 701, BatchKind.SELFPLAY, "self"),
        CollectionAssignment(102, 709, BatchKind.SELFPLAY, "self"),
    )
    source = NativeRoundSource(_native_config(), assignments=assignments)
    source.configure_distributed(rank=0, world_size=2)
    actor = Policy(blocks=1, channels=4)
    source.publish_actor(actor.state_dict(), version=0)

    def fail_second(*_args: object, **_kwargs: object) -> object:
        raise data_module.CollectionRowError(1, "ValueError", "rank row sentinel")

    monkeypatch.setattr("kaggriculture.sim.rollout.collect_segment", fail_second)
    monkeypatch.setattr(
        data_module,
        "_all_gather_objects",
        lambda local, world_size, **_kwargs: (local,) * world_size,
    )

    with pytest.raises(data_module.DistributedCollectionError) as raised:
        list(source)

    assert "game_ids=(102,)" in str(raised.value)
    assert "seeds=(709,)" in str(raised.value)
    assert "game_ids=(101,)" not in str(raised.value)


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
    build_policy = source._native_policy
    policy_calls: list[int] = []

    def recording_policy(
        model: ModelConfig, state: dict[str, torch.Tensor]
    ) -> torch.nn.Module:
        policy = build_policy(model, state)
        index = len(policy_calls)
        policy_calls.append(0)

        def counted(
            _module: torch.nn.Module,
            _inputs: tuple[object, ...],
            _output: object,
        ) -> None:
            policy_calls[index] += 1

        policy.register_forward_hook(counted)
        return policy

    monkeypatch.setattr(source, "_native_policy", recording_policy)

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
    # The learner plus both immutable neural bindings were separately built,
    # and every one reached its own action heads during collection.
    assert len(policy_calls) == 3
    assert all(calls > 0 for calls in policy_calls)


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
