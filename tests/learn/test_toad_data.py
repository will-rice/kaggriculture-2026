"""Typed optimizer-boundary batches for the native Toad collector."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace

import pytest
import torch

from kaggriculture.learn.encoding import (
    MARKET_SLOTS,
    MAX_UNITS,
    QUANTITIES,
    SCALARS,
    TILE_PLANES,
    UNIT_OPS,
)
from kaggriculture.learn.rollout import Trajectory
from kaggriculture.learn.scripts import toad
from kaggriculture.learn.toad.config import ToadConfig
from kaggriculture.learn.toad.data import (
    BatchKind,
    CollectionAssignment,
    CollectionError,
    ReferenceRoundSource,
    ReferenceWorkerInput,
    RoundBatchExpander,
    RoundMeta,
    ToadDataModule,
    segments,
)
from kaggriculture.learn.toad.model import StatefulPolicy


def _trajectory(turns: int) -> Trajectory:
    """Return a complete, cheap recorded episode with a terminal final row."""
    slots = len(MARKET_SLOTS) + 2
    dones = torch.zeros(turns, dtype=torch.bool)
    dones[-1] = True
    rewards = torch.ones(turns)
    return Trajectory(
        board=torch.zeros(turns, TILE_PLANES, 10, 10),
        scalars=torch.zeros(turns, SCALARS),
        positions=torch.zeros(turns, MAX_UNITS, dtype=torch.int64),
        unit_actions=torch.zeros(turns, MAX_UNITS, dtype=torch.int64),
        unit_quantities=torch.zeros(turns, MAX_UNITS, dtype=torch.int64),
        market_actions=torch.zeros(turns, slots, dtype=torch.int64),
        unit_masks=torch.ones(turns, MAX_UNITS, len(UNIT_OPS), dtype=torch.bool),
        unit_quantity_masks=torch.ones(
            turns, MAX_UNITS, len(QUANTITIES), dtype=torch.bool
        ),
        market_masks=torch.ones(turns, slots, len(QUANTITIES), dtype=torch.bool),
        log_probs=torch.zeros(turns),
        values=torch.zeros(turns),
        rewards=rewards,
        own=rewards,
        shaped=rewards,
        shaped_money=rewards,
        margin=rewards,
        sparse=rewards,
        potentials=torch.zeros(turns, 1),
        dones=dones,
        final_margin=1.0,
        final_bank=1.0,
        final_capital=1.0,
        illegal=0,
        sales=0.0,
        units_sold=0.0,
        mean_sale_price=0.0,
        realisation=0.0,
        bought=0.0,
    )


def _recurrent_trajectory(turns: int = 32, layers: int = 2) -> Trajectory:
    """Return a trajectory with distinct autograd-connected state per row."""
    base = _trajectory(turns)
    source = torch.arange((turns + 1) * layers * 3 * 10 * 10, dtype=torch.float32).view(
        turns + 1, layers, 3, 10, 10
    )
    source.requires_grad_()
    belief = torch.arange((turns + 1) * 9, dtype=torch.float32).view(turns + 1, 9)
    belief.requires_grad_()
    return replace(
        base,
        hidden=source * 2,
        cell=source.square(),
        prior_belief=belief.sigmoid(),
    )


def test_segments_store_only_detached_state_before_their_first_observation() -> None:
    """Start 16 must replay state 16, never actor state 15 or all later states."""
    trajectory = _recurrent_trajectory()
    assert trajectory.hidden is not None
    assert trajectory.cell is not None
    assert trajectory.prior_belief is not None

    batches = segments(trajectory, unroll_length=16)

    assert torch.equal(batches[0]["initial_hidden"], trajectory.hidden[0])
    assert torch.equal(batches[1]["initial_hidden"], trajectory.hidden[16])
    assert torch.equal(batches[1]["initial_cell"], trajectory.cell[16])
    assert torch.equal(batches[1]["initial_belief"], trajectory.prior_belief[16])
    assert "hidden" not in batches[1]
    assert "cell" not in batches[1]
    assert "prior_belief" not in batches[1]
    assert not batches[0]["initial_hidden"].requires_grad
    assert not batches[0]["initial_cell"].requires_grad
    assert not batches[0]["initial_belief"].requires_grad


def test_recurrent_reference_worker_receives_the_resolved_model_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker reconstruction must retain recurrent layers and exact stateful keys."""
    config = ToadConfig.model_validate(
        {
            "model": {
                "blocks": 1,
                "channels": 16,
                "recurrent": True,
                "recurrent_channels": 3,
                "recurrent_layers": 2,
            }
        }
    )
    policy = StatefulPolicy(config.model)
    assignment = CollectionAssignment(4, 12, "self", BatchKind.SELFPLAY)
    seen: list[tuple[object, dict[str, torch.Tensor]]] = []

    def fake_play(work: ReferenceWorkerInput) -> list[Trajectory]:
        seen.append((work.model, work.actor_state))
        return []

    monkeypatch.setattr(toad, "_play_reference", fake_play)
    source = ReferenceRoundSource(config, assignments=(assignment,))
    source.publish_actor(policy.state_dict(), version=3)

    source.collect_assignment(assignment)

    assert seen[0][0] == config.model
    assert tuple(seen[0][1]) == tuple(policy.state_dict())


def test_recurrent_reference_worker_rejects_bare_control_weights() -> None:
    """Enabled actor components may never survive as random worker parameters."""
    config = ToadConfig.model_validate(
        {
            "model": {
                "blocks": 1,
                "channels": 16,
                "recurrent": True,
                "recurrent_channels": 3,
            }
        }
    )
    bare = toad.Policy(blocks=1, channels=16, value_bound=toad.VALUE_BOUND)
    work = ReferenceWorkerInput(
        actor_state=dict(bare.state_dict()),
        seeds=[0],
        model=config.model,
        versus=None,
        money_weight=0.01,
    )

    with pytest.raises(RuntimeError, match="Missing key.*control"):
        toad._play_reference(work)


def test_recurrent_reference_worker_builds_and_loads_the_stateful_actor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real worker entrypoint must consume every resolved model dimension."""
    config = ToadConfig.model_validate(
        {
            "model": {
                "blocks": 1,
                "channels": 16,
                "recurrent": True,
                "recurrent_channels": 3,
                "recurrent_layers": 2,
            }
        }
    )
    expected = StatefulPolicy(config.model)
    seen: list[StatefulPolicy] = []

    def fake_rollout(
        actor: object, opponent: object, seeds: Sequence[int]
    ) -> list[Trajectory]:
        assert isinstance(actor, StatefulPolicy)
        assert opponent is actor
        assert list(seeds) == [19]
        seen.append(actor)
        return []

    monkeypatch.setattr(toad, "rollout_many", fake_rollout)
    work = ReferenceWorkerInput(
        actor_state=dict(expected.state_dict()),
        seeds=[19],
        model=config.model,
        versus=None,
        money_weight=0.01,
    )

    assert toad._play_reference(work) == []
    assert seen[0].config == config.model
    for name, tensor in expected.state_dict().items():
        assert torch.equal(seen[0].state_dict()[name], tensor)


@pytest.mark.parametrize(
    "model",
    [
        {"transformer": True, "transformer_blocks": 1},
        {"local_patch": True, "local_patch_blocks": 1},
        {"interaction_value": True},
    ],
)
def test_optional_reference_worker_builds_exact_stateful_actor(
    model: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Workers must construct and strictly load optional stateful weights."""
    config = ToadConfig.model_validate(
        {"model": {"blocks": 1, "channels": 16, **model}}
    )
    expected = StatefulPolicy(config.model)
    seen: list[StatefulPolicy] = []

    def fake_rollout(
        actor: object, opponent: object, seeds: Sequence[int]
    ) -> list[Trajectory]:
        assert isinstance(actor, StatefulPolicy)
        assert opponent is actor
        seen.append(actor)
        return []

    monkeypatch.setattr(toad, "rollout_many", fake_rollout)
    work = ReferenceWorkerInput(
        actor_state=dict(expected.state_dict()),
        seeds=[23],
        model=config.model,
        versus=None,
        money_weight=0.01,
    )

    assert toad._play_reference(work) == []
    for name, tensor in expected.state_dict().items():
        assert torch.equal(seen[0].state_dict()[name], tensor)


def test_round_expansion_preserves_policy_and_value_pass_counts() -> None:
    """Dropping a fresh group or counting a replay as collection is a bug."""
    trajectories = [_trajectory(turns=32) for _ in range(2)]
    expander = RoundBatchExpander(batch_segments=4, value_passes=2, seed=7)

    batches = list(expander.expand(trajectories, RoundMeta.control(round_id=3)))

    policy = [batch for batch in batches if not batch.baseline_only]
    replay = [batch for batch in batches if batch.baseline_only]
    assert len(policy) == 1
    assert len(replay) == 2
    assert all(len(batch.segments) == 4 for batch in batches)
    assert batches[0].first_of_round
    assert batches[-1].end_of_round
    assert sum(batch.collected_steps for batch in batches) == 64
    assert all(batch.collected_steps == 0 for batch in batches[1:])


def test_collection_error_names_the_failed_game() -> None:
    """A collection failure must retain the assignment that actually failed."""
    assignment = CollectionAssignment(
        game_id=41,
        seed=99,
        opponent_id="economic",
        kind=BatchKind.SCRIPTED,
    )

    def fail(_: CollectionAssignment) -> Sequence[Trajectory]:
        raise RuntimeError("boom")

    source = ReferenceRoundSource(
        ToadConfig.control(), assignments=(assignment,), collect_assignment=fail
    )

    with pytest.raises(CollectionError, match="game_id=41 seed=99 opponent=economic"):
        next(iter(source))


def test_source_aggregates_assignments_into_one_logical_round() -> None:
    """Per-game boundaries would advance the scheduler and actor too often."""
    assignments = (
        CollectionAssignment(7, 17, "self", BatchKind.SELFPLAY),
        CollectionAssignment(8, 18, "self", BatchKind.SELFPLAY),
    )
    source = ReferenceRoundSource(
        ToadConfig.control(),
        assignments=assignments,
        collect_assignment=lambda _: (_trajectory(64),),
    )

    batches = list(source)

    assert {batch.round_id for batch in batches} == {0}
    assert sum(batch.end_of_round for batch in batches) == 1
    assert batches[-1].end_of_round
    assert batches[0].game_ids == (7, 8)
    assert batches[0].opponent_ids == ("self", "self")
    assert sum(batch.collected_steps for batch in batches) == 128


def test_default_round_honors_typed_population_quotas_and_one_clock() -> None:
    """Mixed opponent kinds remain one collection and scheduler boundary."""
    base = ToadConfig.control()
    config = base.model_copy(
        update={
            "population": base.population.model_copy(
                update={
                    "selfplay": 0.75,
                    "scripted": 0.25,
                    "environments_per_rank": 4,
                }
            )
        }
    )
    seen: list[CollectionAssignment] = []

    def collect(assignment: CollectionAssignment) -> Sequence[Trajectory]:
        seen.append(assignment)
        return (_trajectory(64),)

    source = ReferenceRoundSource(config, collect_assignment=collect)

    batches = list(source)

    assert [assignment.kind for assignment in seen].count(BatchKind.SCRIPTED) == 1
    assert [assignment.kind for assignment in seen].count(BatchKind.SELFPLAY) == 3
    assert [assignment.opponent_id for assignment in seen] == [
        "economic",
        "self",
        "self",
        "self",
    ]
    assert {batch.round_id for batch in batches} == {0}
    assert sum(batch.first_of_round for batch in batches) == 1
    assert sum(batch.end_of_round for batch in batches) == 1
    assert batches[0].first_of_round
    assert batches[-1].end_of_round
    assert batches[0].collected_steps == 256
    assert all(batch.collected_steps == 0 for batch in batches[1:])
    scripted = [batch for batch in batches if batch.kind is BatchKind.SCRIPTED]
    selfplay = [batch for batch in batches if batch.kind is BatchKind.SELFPLAY]
    assert scripted[0].game_ids == (0,)
    assert scripted[0].opponent_ids == ("economic",)
    assert selfplay[0].game_ids == (1, 2, 3)
    assert selfplay[0].opponent_ids == ("self", "self", "self")

    next_round = list(source)
    assert {batch.round_id for batch in next_round} == {1}
    assert next_round[0].game_ids == (5, 6, 7)


def test_default_collection_uses_one_typed_round_pool_and_fans_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Collection-process width applies once to every assignment in a round."""
    base = ToadConfig.control()
    config = base.model_copy(
        update={
            "population": base.population.model_copy(
                update={
                    "selfplay": 0.5,
                    "scripted": 0.5,
                    "environments_per_rank": 4,
                    "collection_processes": 3,
                }
            ),
            "curriculum": base.curriculum.model_copy(update={"money_weight": 0.01}),
        }
    )
    pool_widths: list[int] = []
    work: list[ReferenceWorkerInput] = []

    class FakePool:
        def __init__(self, max_workers: int) -> None:
            pool_widths.append(max_workers)

        def __enter__(self) -> "FakePool":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def map(
            self,
            function: Callable[[ReferenceWorkerInput], object],
            assignments: Iterable[ReferenceWorkerInput],
        ) -> Iterable[object]:
            assigned = list(assignments)
            work.extend(assigned)
            return map(function, assigned)

    monkeypatch.setattr("kaggriculture.learn.toad.data.ProcessPoolExecutor", FakePool)
    monkeypatch.setattr(toad, "_play_reference", lambda _: [_trajectory(64)])
    source = ReferenceRoundSource(config)
    source.publish_actor({"weight": torch.ones(1)}, version=7)

    list(source)

    assert pool_widths == [3]
    assert len(work) == 4
    assert all(item.actor_state == source.actor_state for item in work)
    assert [item.versus for item in work] == [
        toad.OPPONENT,
        toad.OPPONENT,
        None,
        None,
    ]
    assert all(item.money_weight == 0.01 for item in work)


def test_data_module_detaches_actor_before_publication() -> None:
    """Collector publication must never retain a gradient-bearing learner tensor."""
    source = ReferenceRoundSource(
        ToadConfig.control(), assignments=(), collect_assignment=lambda _: ()
    )
    module = ToadDataModule(ToadConfig.control(), source)
    weight = torch.ones(2, requires_grad=True)

    module.publish_actor({"weight": weight}, version=4)

    assert source.actor_version == 4
    assert source.actor_state["weight"].device.type == "cpu"
    assert not source.actor_state["weight"].requires_grad
    with torch.no_grad():
        weight.zero_()
    assert source.actor_state["weight"].tolist() == [1.0, 1.0]
    assert source.actor_state["weight"].data_ptr() != weight.data_ptr()
    loader = module.train_dataloader()
    assert loader.batch_size is None
    assert loader.num_workers == 0


def test_reference_worker_uses_the_typed_money_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker must not fall back to an inherited or process-local reward default."""
    seen: list[float] = []
    policy = toad.Policy(blocks=1, channels=16, value_bound=toad.VALUE_BOUND)

    def fake_rollout(*args: object, **kwargs: object) -> list[Trajectory]:
        seen.append(toad.money_weight())
        return []

    monkeypatch.setattr(toad, "rollout_many", fake_rollout)

    toad._play((policy.state_dict(), [0], 1, 16, None, 0.01))

    assert seen == [0.01]
