"""Privileged opponent-belief supervision without policy-input leakage."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from typing import cast

import pytest
import torch

import kaggriculture.learn.rollout as rollout_module
from kaggriculture.constants import SHED_CAPACITY
from kaggriculture.learn import toad_loss
from kaggriculture.learn.encoding import (
    CARRIED_SCALE,
    CROP_NAMES,
    PRODUCT_NAMES,
    SEED_SCALE,
    SHED_NAMES,
    BeliefTarget,
    encode_private_belief_target,
)
from kaggriculture.learn.scripts import toad
from kaggriculture.learn.toad.config import ModelConfig, ToadConfig
from kaggriculture.learn.toad.data import (
    BatchKind,
    LearnerBatch,
    ReferenceWorkerInput,
    segments,
)
from kaggriculture.learn.toad.lightning import ToadLightningModule, compute_loss
from kaggriculture.learn.toad.model import PolicyOutput, PolicyState, StatefulPolicy
from tests.learn.test_toad_data import _trajectory
from tests.learn.test_toad_model import recurrent_inputs


def test_belief_target_uses_fixed_family_order_and_normalization() -> None:
    """Wrong family order, missing inventory aggregation, or scaling must fail."""
    observation = {
        "private": {
            "shed": {name: index + 1 for index, name in enumerate(SHED_NAMES)},
            "seeds": {name: 2 * index + 3 for index, name in enumerate(CROP_NAMES)},
            "inventories": [
                {PRODUCT_NAMES[0]: 2, PRODUCT_NAMES[-1]: 5},
                {PRODUCT_NAMES[0]: 4, PRODUCT_NAMES[-1]: 7},
            ],
        }
    }

    target = encode_private_belief_target(observation)

    assert target.shed.tolist() == pytest.approx(
        [(index + 1) / SHED_CAPACITY for index in range(len(SHED_NAMES))]
    )
    assert target.seeds.tolist() == pytest.approx(
        [(2 * index + 3) / SEED_SCALE for index in range(len(CROP_NAMES))]
    )
    expected_carried = [0.0] * len(PRODUCT_NAMES)
    expected_carried[0] = 6 / CARRIED_SCALE
    expected_carried[-1] = 12 / CARRIED_SCALE
    assert target.carried.tolist() == pytest.approx(expected_carried)
    assert target.tensor.dtype is torch.float32
    assert target.tensor.shape == (
        len(SHED_NAMES) + len(CROP_NAMES) + len(PRODUCT_NAMES),
    )


def test_opponent_private_label_cannot_change_policy_inputs() -> None:
    """Collector target changes may not alter the policy tensors it forwards."""
    from kaggle_environments import make

    from kaggriculture.constants import ENVIRONMENT

    environment = make(ENVIRONMENT, configuration={"episodeSteps": 3, "seed": 8})
    environment.reset(2)
    own = environment.state[0].observation
    opponent_a = environment.state[1].observation
    opponent_b = deepcopy(opponent_a)
    product = PRODUCT_NAMES[0]
    opponent_b["private"]["shed"][product] += 11

    policy = StatefulPolicy(
        ModelConfig.control(blocks=1, channels=16).model_copy(update={"belief": True})
    ).eval()
    first, _ = rollout_module._decide(
        policy,
        [(own, 0)],
        torch.Generator().manual_seed(2),
        belief_observations=[opponent_a],
    )
    second, _ = rollout_module._decide(
        policy,
        [(own, 0)],
        torch.Generator().manual_seed(2),
        belief_observations=[opponent_b],
    )

    assert torch.equal(first[0].board, second[0].board)
    assert torch.equal(first[0].scalars, second[0].scalars)
    assert torch.equal(first[0].positions, second[0].positions)
    assert first[0].belief_target is not None
    assert second[0].belief_target is not None
    assert not torch.equal(first[0].belief_target, second[0].belief_target)


@pytest.mark.parametrize("recurrent", [False, True])
def test_nonbelief_rollout_and_segments_have_no_privileged_labels(
    recurrent: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control and recurrent-only collection must retain the Task 3 schema."""
    monkeypatch.setattr(rollout_module, "EPISODE_STEPS", 3)
    if recurrent:
        policy: toad.Policy | StatefulPolicy = StatefulPolicy(
            ModelConfig.control(blocks=1, channels=16).model_copy(
                update={"recurrent": True, "recurrent_channels": 4}
            )
        ).eval()
    else:
        policy = toad.Policy(blocks=1, channels=16).eval()

    trajectories = rollout_module.rollout_many(policy, policy, (43,))

    for trajectory in trajectories:
        assert trajectory.belief_targets is None
        assert trajectory.belief_valid is None
        segment = segments(trajectory, 1)[0]
        assert "belief_targets" not in segment
        assert "belief_valid" not in segment


def test_belief_only_rollout_records_zero_sized_absent_recurrent_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Belief feedback must not allocate full ConvLSTM maps when it is absent."""
    monkeypatch.setattr(rollout_module, "EPISODE_STEPS", 3)
    policy = StatefulPolicy(
        ModelConfig.control(blocks=1, channels=16).model_copy(
            update={"belief": True, "belief_feedback": True}
        )
    ).eval()

    trajectory = rollout_module.rollout_many(policy, "starter", (47,))[0]

    assert trajectory.hidden is not None
    assert trajectory.cell is not None
    assert trajectory.hidden.shape[0] == trajectory.dones.shape[0] + 1
    assert trajectory.hidden.numel() == 0
    assert trajectory.cell.numel() == 0
    segment = segments(trajectory, 1)[0]
    assert segment["initial_hidden"].numel() == 0
    assert segment["initial_cell"].numel() == 0


def test_belief_only_rejects_nonempty_absent_recurrent_state() -> None:
    """Malformed segment entry state must not reintroduce hidden map storage."""
    policy = StatefulPolicy(
        ModelConfig.control(blocks=1, channels=16).model_copy(
            update={"belief": True, "belief_feedback": True}
        )
    )
    board, scalars, positions = recurrent_inputs(time=2, batch=1)
    state = policy.initial_state(1, like=board)
    assert state is not None
    malformed = replace(
        state,
        hidden=board.new_zeros(1, 1, 1, 1),
        cell=board.new_zeros(1, 1, 1, 1),
    )

    with pytest.raises(ValueError, match="zero-sized"):
        policy(board, scalars, positions, state=malformed)


def test_segments_preserve_acted_belief_targets_and_boolean_validity() -> None:
    """Dropping or bootstrap-shifting privileged labels must fail this test."""
    trajectory = _trajectory(6)
    width = len(SHED_NAMES) + len(CROP_NAMES) + len(PRODUCT_NAMES)
    targets = torch.arange(6 * width, dtype=torch.float32).view(6, width)
    valid = torch.tensor([True, False, True, True, False, True])

    result = segments(
        replace(trajectory, belief_targets=targets, belief_valid=valid),
        unroll_length=3,
    )

    assert torch.equal(result[0]["belief_targets"], targets[:3])
    assert torch.equal(result[1]["belief_targets"], targets[3:])
    assert torch.equal(result[0]["belief_valid"], valid[:3])
    assert result[0]["belief_valid"].dtype is torch.bool


@pytest.mark.parametrize("recurrent", [False, True])
def test_belief_only_and_recurrent_belief_shapes_and_gradients(
    recurrent: bool,
) -> None:
    """Either supported topology must emit T rows and train the belief head."""
    config = ModelConfig.control(blocks=1, channels=16).model_copy(
        update={
            "belief": True,
            "belief_feedback": True,
            "recurrent": recurrent,
            "recurrent_channels": 4,
        }
    )
    policy = StatefulPolicy(config)
    board, scalars, positions = recurrent_inputs(time=4, batch=2)

    output = policy(
        board,
        scalars,
        positions,
        dones=torch.zeros(4, 2, dtype=torch.bool),
    )

    assert output.belief_logits is not None
    assert output.belief_logits.shape == (4, 2, config.belief_size)
    assert output.state is not None
    assert torch.equal(output.state.prior_belief, output.belief_logits[-1])
    (output.belief_logits.square().mean() + output.unit_logits[-1].mean()).backward()
    assert policy.belief_head.weight.grad is not None
    assert torch.isfinite(policy.belief_head.weight.grad).all()
    assert policy.feedback.weight.grad is not None


@pytest.mark.parametrize("recurrent", [False, True])
def test_feedback_state_resets_once_before_a_mid_sequence_observation(
    recurrent: bool,
) -> None:
    """A terminal must clear hidden, cell, and prior prediction on one clock."""
    torch.manual_seed(19)
    config = ModelConfig.control(blocks=1, channels=16).model_copy(
        update={
            "belief": True,
            "belief_feedback": True,
            "recurrent": recurrent,
            "recurrent_channels": 4,
        }
    )
    policy = StatefulPolicy(config)
    board, scalars, positions = recurrent_inputs(time=3, batch=1)

    full = policy(
        board,
        scalars,
        positions,
        dones=torch.tensor([[False], [True], [False]]),
    )
    fresh = policy(
        board[1:],
        scalars[1:],
        positions[1:],
        dones=torch.zeros(2, 1, dtype=torch.bool),
    )

    assert full.belief_logits is not None
    assert fresh.belief_logits is not None
    assert torch.allclose(full.belief_logits[1:], fresh.belief_logits, atol=1e-6)
    assert torch.allclose(full.unit_logits[1:], fresh.unit_logits, atol=1e-6)


def _belief_batch(
    valid: torch.Tensor | None,
) -> tuple[ToadConfig, StatefulPolicy, LearnerBatch]:
    config = ToadConfig.model_validate(
        {
            "model": {
                "blocks": 1,
                "channels": 16,
                "belief": True,
                "belief_loss_weight": 0.25,
            },
            "optimizer": {"value_warmup_batches": 0},
        }
    )
    policy = StatefulPolicy(config.model)
    torch.nn.init.zeros_(policy.belief_head.weight)
    torch.nn.init.zeros_(policy.belief_head.bias)
    base = _trajectory(2)
    state = policy.initial_state(1, like=base.board)
    assert state is not None
    trajectory = replace(
        base,
        hidden=torch.stack([state.hidden[0]] * 3),
        cell=torch.stack([state.cell[0]] * 3),
        prior_belief=torch.stack([state.prior_belief[0]] * 3),
        belief_targets=(
            torch.ones(2, config.model.belief_size) if valid is not None else None
        ),
        belief_valid=valid,
    )
    batch = LearnerBatch(
        segments=(segments(trajectory, 2)[0],),
        kind=BatchKind.SELFPLAY,
        baseline_only=False,
        first_of_round=True,
        end_of_round=True,
        collected_steps=2,
        round_id=0,
        actor_version=0,
        game_ids=(0,),
        opponent_ids=("self",),
    )
    return config, policy, batch


def _zero_rl_losses(**kwargs: object) -> toad_loss.Losses:
    values = cast(torch.Tensor, kwargs["values"])
    zero = values.sum() * 0
    return toad_loss.Losses(zero, zero, zero, zero, zero, zero)


def test_masked_belief_loss_is_weighted_and_reports_family_errors() -> None:
    """Invalid rows must not dilute or contribute to normalized supervision."""
    config, policy, batch = _belief_batch(torch.tensor([True, False]))

    report = compute_loss(policy, batch, config, _losses=_zero_rl_losses)

    assert report.terms["belief"].item() == pytest.approx(0.5)
    assert report.total.item() == pytest.approx(0.125)
    assert report.terms["belief_shed"].item() == pytest.approx(0.5)
    assert report.terms["belief_seeds"].item() == pytest.approx(0.5)
    assert report.terms["belief_carried"].item() == pytest.approx(0.5)
    assert report.terms["belief_valid"].item() == 1


def test_zero_valid_belief_rows_produce_finite_exact_zero() -> None:
    """A collection with no valid labels must never divide by zero."""
    config, policy, batch = _belief_batch(torch.tensor([False, False]))
    invalid_segment = dict(batch.segments[0])
    invalid_segment["belief_targets"] = torch.full_like(
        invalid_segment["belief_targets"], float("nan")
    )
    batch = replace(batch, segments=(invalid_segment,))

    report = compute_loss(policy, batch, config, _losses=_zero_rl_losses)

    assert torch.isfinite(report.total)
    assert report.total.item() == 0.0
    assert report.terms["belief"].item() == 0.0
    assert report.terms["belief_valid"].item() == 0.0


def test_active_belief_training_rejects_missing_targets() -> None:
    """Optional trajectory defaults cannot silently disable an active loss."""
    config, policy, batch = _belief_batch(None)

    with pytest.raises(ValueError, match="belief learner batches require"):
        compute_loss(policy, batch, config, _losses=_zero_rl_losses)


def test_belief_only_learner_replays_nonzero_segment_entry_feedback() -> None:
    """Belief-only unrolls may not restart prior prediction from zero."""
    config, source, batch = _belief_batch(torch.tensor([True, True]))
    config = config.model_copy(
        update={"model": config.model.model_copy(update={"belief_feedback": True})}
    )
    segment = dict(batch.segments[0])
    expected = torch.full_like(segment["initial_belief"], 0.75)
    segment["initial_belief"] = expected
    captured: list[PolicyState | None] = []

    class SpyPolicy(StatefulPolicy):
        def forward(
            self,
            board: torch.Tensor,
            scalars: torch.Tensor,
            positions: torch.Tensor,
            state: PolicyState | None = None,
            dones: torch.Tensor | None = None,
        ) -> PolicyOutput:
            captured.append(state)
            return super().forward(board, scalars, positions, state, dones)

    policy = SpyPolicy(config.model)
    policy.control.load_state_dict(source.control.state_dict())

    compute_loss(
        policy,
        replace(batch, segments=(segment,)),
        config,
        _losses=_zero_rl_losses,
    )

    assert captured[0] is not None
    assert torch.equal(captured[0].prior_belief[0], expected)


@pytest.mark.parametrize("scripted", [True, False])
def test_rollout_captures_opposing_seat_private_targets_for_every_recorded_seat(
    scripted: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Using the learner's private state or omitting self-play seat 1 must fail."""
    real_encoder = encode_private_belief_target
    captured: list[tuple[int, torch.Tensor]] = []

    def capture(observation: Mapping[str, object]) -> BeliefTarget:
        target = real_encoder(observation)
        captured.append((cast(int, observation["player"]), target.tensor))
        return target

    monkeypatch.setattr(rollout_module, "EPISODE_STEPS", 3)
    monkeypatch.setattr(rollout_module, "encode_private_belief_target", capture)
    config = ModelConfig.control(blocks=1, channels=16).model_copy(
        update={"belief": True}
    )
    policy = StatefulPolicy(config).eval()

    trajectories = rollout_module.rollout_many(
        policy,
        "starter" if scripted else policy,
        (41,),
    )

    expected_seats = [1, 1] if scripted else [1, 0, 1, 0]
    assert [seat for seat, _ in captured] == expected_seats
    assert len(trajectories) == (1 if scripted else 2)
    for index, trajectory in enumerate(trajectories):
        assert trajectory.belief_targets is not None
        assert trajectory.belief_valid is not None
        expected = captured[index :: len(trajectories)]
        assert torch.equal(
            trajectory.belief_targets,
            torch.stack([target for _, target in expected]),
        )
        assert torch.equal(
            trajectory.belief_valid,
            torch.ones(2, dtype=torch.bool),
        )


@pytest.mark.parametrize("recurrent", [False, True])
def test_actor_and_learner_feedback_outputs_match_across_terminal(
    recurrent: bool,
) -> None:
    """Actor detachment may not change feedback values or reset timing."""
    torch.manual_seed(31)
    config = ModelConfig.control(blocks=1, channels=16).model_copy(
        update={
            "belief": True,
            "belief_feedback": True,
            "recurrent": recurrent,
            "recurrent_channels": 4,
        }
    )
    policy = StatefulPolicy(config).eval()
    board, scalars, positions = recurrent_inputs(time=4, batch=2)
    dones = torch.tensor(
        [[False, False], [False, False], [True, False], [False, False]]
    )

    learner = policy(board, scalars, positions, dones=dones)
    states = [None, None]
    actor_units: list[torch.Tensor] = []
    actor_beliefs: list[torch.Tensor] = []
    for step in range(4):
        actor, _used_states = rollout_module._actor_forward(
            policy,
            board[step],
            scalars[step],
            positions[step],
            states,
            dones[step],
        )
        assert actor.belief_logits is not None
        actor_units.append(actor.unit_logits)
        actor_beliefs.append(actor.belief_logits)
        states = rollout_module._unbatch_policy_state(actor.state)

    assert learner.belief_logits is not None
    assert torch.allclose(torch.stack(actor_units), learner.unit_logits, atol=1e-6)
    assert torch.allclose(torch.stack(actor_beliefs), learner.belief_logits, atol=1e-6)


def test_belief_only_lightning_and_worker_use_strict_stateful_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Belief-only production paths may not fall back to a bare policy."""
    config = ToadConfig.model_validate(
        {"model": {"blocks": 1, "channels": 16, "belief": True}}
    )
    module = ToadLightningModule(config)
    assert isinstance(module.policy, StatefulPolicy)
    seen: list[object] = []

    def capture(actor: object, opponent: object, seeds: object) -> list[object]:
        seen.append(actor)
        return []

    monkeypatch.setattr(toad, "rollout_many", capture)
    work = ReferenceWorkerInput(
        actor_state=module.policy.state_dict(),
        seeds=[1],
        model=config.model,
        versus="starter",
        money_weight=0.0,
    )
    assert toad._play_reference(work) == []
    assert isinstance(seen[0], StatefulPolicy)

    bare = toad.Policy(blocks=1, channels=16).state_dict()
    with pytest.raises(RuntimeError, match="Missing key.*control"):
        toad._play_reference(replace(work, actor_state=bare))
