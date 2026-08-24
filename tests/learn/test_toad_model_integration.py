"""End-to-end gates for the fully enabled recurrent Toad model."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import lightning
import torch
from torch.utils.data import DataLoader, Dataset

from kaggriculture.constants import BOARD_SIZE
from kaggriculture.learn.encoding import (
    MARKET_SLOTS,
    MAX_UNITS,
    QUANTITIES,
    SCALARS,
    TILE_PLANES,
    UNIT_OPS,
)
from kaggriculture.learn.ppo import joint_log_prob
from kaggriculture.learn.rollout import Trajectory
from kaggriculture.learn.toad.config import ToadConfig, structural_fingerprint
from kaggriculture.learn.toad.data import BatchKind, LearnerBatch, segments
from kaggriculture.learn.toad.lightning import ToadLightningModule, compute_loss
from kaggriculture.learn.toad.model import PolicyState, StatefulPolicy


@dataclass(frozen=True)
class _SyntheticTrajectory:
    """One deterministic trajectory plus actor-clock boundary states."""

    trajectory: Trajectory
    terminal_input: PolicyState
    terminal_output: PolicyState
    next_input: PolicyState


class _BatchDataset(Dataset[LearnerBatch]):
    """Finite typed batches for Lightning's real automatic-optimization path."""

    def __init__(self, *batches: LearnerBatch) -> None:
        self.batches = batches

    def __getitem__(self, index: int) -> LearnerBatch:
        return self.batches[index]

    def __len__(self) -> int:
        return len(self.batches)


def _all_feature_config(
    *,
    output_dir: Path | None = None,
    teacher_checkpoint: Path | None = None,
    value_warmup_batches: int = 0,
) -> ToadConfig:
    """Return the smallest production-valid topology with every feature on."""
    population: dict[str, object] = {
        "selfplay": 1.0,
        "scripted": 0.0,
        "environments_per_rank": 1,
        "collection_processes": 1,
        "actor_sync_every_rounds": 1,
    }
    runtime: dict[str, object] = {
        "seed": 73,
        "deterministic": True,
        "checkpoint_every_environment_steps": 32,
        "total_environment_steps": 96,
    }
    payload: dict[str, object] = {
        "model": {
            "blocks": 1,
            "channels": 4,
            "recurrent": True,
            "recurrent_channels": 2,
            "recurrent_layers": 1,
            "belief": True,
            "belief_loss_weight": 0.25,
            "belief_feedback": True,
            "transformer": True,
            "transformer_blocks": 1,
            "transformer_heads": 1,
            "transformer_mlp_ratio": 2,
            "local_patch": True,
            "local_patch_size": 3,
            "local_patch_blocks": 1,
            "interaction_value": True,
        },
        "population": population,
        "optimizer": {
            "lr": 0.001,
            "unroll_length": 16,
            "batch_segments": 1,
            "value_warmup_batches": value_warmup_batches,
            "value_passes": 0,
        },
        "runtime": runtime,
    }
    if output_dir is not None:
        runtime["output_dir"] = output_dir
    if teacher_checkpoint is not None:
        population.update(
            teacher={"checkpoint": teacher_checkpoint, "blocks": 1},
        )
    return ToadConfig.model_validate(payload)


def _initial_policy_state(
    config: ToadConfig, *, seed: int = 202_608_24
) -> dict[str, torch.Tensor]:
    """Create one reproducible fully enabled actor snapshot."""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        policy = StatefulPolicy(config.model)
    return {
        name: tensor.detach().clone() for name, tensor in policy.state_dict().items()
    }


def _synthetic_trajectory(
    config: ToadConfig,
    policy_state: Mapping[str, torch.Tensor],
    *,
    game_id: int = 0,
) -> _SyntheticTrajectory:
    """Collect 32 all-head-active turns with an episode boundary after turn 15."""
    turns = 32
    slots = len(MARKET_SLOTS) + 2
    board_values = torch.arange(
        turns * TILE_PLANES * BOARD_SIZE * BOARD_SIZE, dtype=torch.float32
    )
    board = torch.sin(board_values.mul(0.013).add(float(game_id))).view(
        turns, TILE_PLANES, BOARD_SIZE, BOARD_SIZE
    )
    scalar_values = torch.arange(turns * SCALARS, dtype=torch.float32)
    scalars = torch.cos(scalar_values.mul(0.031).add(float(game_id))).view(
        turns, SCALARS
    )
    time = torch.arange(turns)[:, None]
    unit = torch.arange(MAX_UNITS)[None, :]
    positions = (time * 7 + unit * 11 + game_id).remainder(BOARD_SIZE * BOARD_SIZE)
    unit_actions = 20 + (time + unit).remainder(len(UNIT_OPS) - 20)
    unit_quantities = (time * 3 + unit).remainder(len(QUANTITIES))
    market_slot = torch.arange(slots)[None, :]
    market_actions = (time + market_slot * 2).remainder(len(QUANTITIES))
    unit_masks = torch.ones(turns, MAX_UNITS, len(UNIT_OPS), dtype=torch.bool)
    quantity_masks = torch.ones(turns, MAX_UNITS, len(QUANTITIES), dtype=torch.bool)
    market_masks = torch.ones(turns, slots, len(QUANTITIES), dtype=torch.bool)
    dones = torch.zeros(turns, dtype=torch.bool)
    dones[15] = True
    dones[31] = True
    rewards = torch.sin(torch.arange(turns, dtype=torch.float32).mul(0.37))
    rewards[15] += 1.0
    rewards[31] -= 0.75
    belief_targets = (
        torch.arange(turns * config.model.belief_size, dtype=torch.float32)
        .view(turns, config.model.belief_size)
        .remainder(17)
        .div(16)
    )

    actor = StatefulPolicy(config.model).eval()
    actor.load_state_dict(policy_state, strict=True)
    state: PolicyState | None = None
    input_states: list[PolicyState] = []
    output_states: list[PolicyState] = []
    log_probs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    with torch.no_grad():
        for step in range(turns):
            pre_observation_done = torch.tensor(
                [[bool(dones[step - 1])] if step else [False]], dtype=torch.bool
            )
            output = actor(
                board[step : step + 1, None],
                scalars[step : step + 1, None],
                positions[step : step + 1, None],
                state=state,
                dones=pre_observation_done,
            )
            assert output.input_state is not None
            assert output.state is not None
            input_states.append(output.input_state.detach())
            output_states.append(output.state.detach())
            units = torch.log_softmax(output.unit_logits[0], dim=-1)
            quantities = torch.log_softmax(output.quantity_logits[0], dim=-1)
            market = torch.log_softmax(output.market_logits[0], dim=-1)
            log_probs.append(
                joint_log_prob(
                    units,
                    quantities,
                    market,
                    unit_actions[step : step + 1],
                    unit_quantities[step : step + 1],
                    market_actions[step : step + 1],
                )[0]
            )
            values.append(output.values[0, 0])
            state = output.state
    assert state is not None

    def unbatch(item: PolicyState) -> PolicyState:
        return PolicyState(item.hidden[0], item.cell[0], item.prior_belief[0])

    recorded_inputs = [unbatch(item) for item in input_states]
    trailing = unbatch(state)
    trajectory = Trajectory(
        board=board,
        scalars=scalars,
        positions=positions,
        unit_actions=unit_actions,
        unit_quantities=unit_quantities,
        market_actions=market_actions,
        unit_masks=unit_masks,
        unit_quantity_masks=quantity_masks,
        market_masks=market_masks,
        log_probs=torch.stack(log_probs),
        values=torch.stack(values),
        rewards=rewards,
        own=rewards,
        shaped=rewards,
        shaped_money=rewards,
        margin=rewards,
        sparse=rewards,
        potentials=torch.zeros(turns, 1),
        dones=dones,
        final_margin=float(rewards.sum()),
        final_bank=1.0,
        final_capital=1.0,
        illegal=0,
        sales=1.0,
        units_sold=1.0,
        mean_sale_price=1.0,
        realisation=1.0,
        bought=1.0,
        hidden=torch.stack(
            [item.hidden for item in recorded_inputs] + [trailing.hidden]
        ),
        cell=torch.stack([item.cell for item in recorded_inputs] + [trailing.cell]),
        prior_belief=torch.stack(
            [item.prior_belief for item in recorded_inputs] + [trailing.prior_belief]
        ),
        belief_targets=belief_targets,
        belief_valid=torch.ones(turns, dtype=torch.bool),
    )
    return _SyntheticTrajectory(
        trajectory=trajectory,
        terminal_input=input_states[15],
        terminal_output=output_states[15],
        next_input=input_states[16],
    )


def _training_batches(trajectory: Trajectory) -> tuple[LearnerBatch, LearnerBatch]:
    """Return one optimizer batch per 16-turn segment in a single round."""
    first, second = segments(trajectory, 16)
    return (
        LearnerBatch(
            segments=(first,),
            kind=BatchKind.SELFPLAY,
            baseline_only=False,
            first_of_round=True,
            end_of_round=False,
            collected_steps=32,
            round_id=0,
            actor_version=0,
            game_ids=(0,),
            opponent_ids=("self",),
        ),
        LearnerBatch(
            segments=(second,),
            kind=BatchKind.SELFPLAY,
            baseline_only=False,
            first_of_round=False,
            end_of_round=True,
            collected_steps=0,
            round_id=0,
            actor_version=0,
            game_ids=(0,),
            opponent_ids=("self",),
        ),
    )


def _component_parameters(
    policy: StatefulPolicy,
) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
    """Group parameters by each independently required all-feature consumer."""
    predicates: dict[str, Callable[[str], bool]] = {
        "trunk_stem": lambda name: name.startswith("control.stem."),
        "trunk_scalar": lambda name: name.startswith("control.market."),
        "trunk_residual": lambda name: name.startswith("control.blocks."),
        "market_head": lambda name: name.startswith("control.trade_head."),
        "recurrent_gates": lambda name: name.startswith("recurrent.cells."),
        "recurrent_merge": lambda name: name.startswith("merge."),
        "belief_feedback": lambda name: name.startswith("feedback."),
        "belief_head": lambda name: name.startswith("belief_head."),
        "transformer_position": lambda name: name == "transformer.position",
        "transformer_attention_norm": lambda name: name.startswith(
            "transformer.blocks.0.attention_norm."
        ),
        "transformer_attention": lambda name: name.startswith(
            "transformer.blocks.0.attention."
        ),
        "transformer_mlp_norm": lambda name: name.startswith(
            "transformer.blocks.0.mlp_norm."
        ),
        "transformer_mlp": lambda name: name.startswith("transformer.blocks.0.mlp."),
        "interaction_query": lambda name: name == "interaction_value.value_token",
        "interaction_query_norm": lambda name: name.startswith(
            "interaction_value.query_norm."
        ),
        "interaction_context_norm": lambda name: name.startswith(
            "interaction_value.context_norm."
        ),
        "interaction_global": lambda name: name.startswith(
            "interaction_value.global_projection."
        ),
        "interaction_attention": lambda name: name.startswith(
            "interaction_value.attention."
        ),
        "interaction_mlp": lambda name: name.startswith("interaction_value.mlp."),
        "interaction_mlp_norm": lambda name: name.startswith(
            "interaction_value.mlp_norm."
        ),
        "interaction_value": lambda name: name.startswith(
            "interaction_value.value_projection."
        ),
        "local_preprocess": lambda name: name.startswith("local_head.preprocess."),
        "local_blocks": lambda name: name.startswith("local_head.blocks."),
        "local_operation": lambda name: name.startswith("local_head.operation."),
        "local_quantity": lambda name: name.startswith("local_head.quantity."),
    }
    named = list(policy.named_parameters())
    return {
        component: [(name, parameter) for name, parameter in named if predicate(name)]
        for component, predicate in predicates.items()
    }


def test_two_segment_training_resets_and_updates_every_component(
    tmp_path: Path,
) -> None:
    """A real all-feature loss and two Adam steps must train every consumer."""
    config = _all_feature_config()
    initial_state = _initial_policy_state(config)
    synthetic = _synthetic_trajectory(config, initial_state)
    batches = _training_batches(synthetic.trajectory)
    module = ToadLightningModule(config)
    assert isinstance(module.policy, StatefulPolicy)
    module.policy.load_state_dict(initial_state, strict=True)

    assert synthetic.trajectory.dones.tolist() == [False] * 15 + [True] + [
        False
    ] * 15 + [True]
    assert torch.count_nonzero(synthetic.terminal_input.hidden)
    assert torch.count_nonzero(synthetic.terminal_input.cell)
    assert torch.count_nonzero(synthetic.terminal_input.prior_belief)
    assert torch.count_nonzero(synthetic.terminal_output.hidden)
    assert torch.count_nonzero(synthetic.terminal_output.cell)
    assert torch.count_nonzero(synthetic.terminal_output.prior_belief)
    assert not torch.count_nonzero(synthetic.next_input.hidden)
    assert not torch.count_nonzero(synthetic.next_input.cell)
    assert not torch.count_nonzero(synthetic.next_input.prior_belief)
    second = batches[1].segments[0]
    assert torch.equal(second["initial_hidden"], synthetic.next_input.hidden[0])
    assert torch.equal(second["initial_cell"], synthetic.next_input.cell[0])
    assert torch.equal(second["initial_belief"], synthetic.next_input.prior_belief[0])
    assert second["belief_valid"].all()
    assert second["unit_masks"].sum(dim=-1).min() > 1
    assert second["unit_quantity_masks"].sum(dim=-1).min() > 1
    assert second["market_masks"].sum(dim=-1).min() > 1

    learner_entries: list[PolicyState | None] = []
    entry_hook = module.policy.register_forward_hook(
        lambda _module, _args, output: learner_entries.append(output.input_state)
    )
    second_report = compute_loss(module.policy, batches[1], config)
    entry_hook.remove()
    learner_entry = learner_entries[0]
    assert learner_entry is not None
    assert torch.equal(learner_entry.hidden, synthetic.next_input.hidden)
    assert torch.equal(learner_entry.cell, synthetic.next_input.cell)
    assert torch.equal(learner_entry.prior_belief, synthetic.next_input.prior_belief)
    reports = [compute_loss(module.policy, batches[0], config), second_report]
    assert all(torch.isfinite(report.total) for report in reports)
    assert all(report.terms["belief_valid"].item() == 16 for report in reports)
    torch.stack([report.total for report in reports]).sum().backward()
    components = _component_parameters(module.policy)
    # These legacy readouts are structurally superseded by the enabled local
    # operation/quantity heads and the interaction value head, respectively.
    superseded = (
        "control.head.",
        "control.quantity_head.",
        "control.value.",
    )
    active_parameters = {
        name: parameter
        for name, parameter in module.policy.named_parameters()
        if not name.startswith(superseded)
    }
    covered = {
        name for parameters in components.values() for name, _parameter in parameters
    }
    assert covered == active_parameters.keys()
    initial_zero_gradients: list[str] = []
    for component, parameters in components.items():
        assert parameters, component
        gradients = [parameter.grad for _, parameter in parameters]
        assert all(gradient is not None for gradient in gradients), component
        present_gradients = [gradient for gradient in gradients if gradient is not None]
        assert all(torch.isfinite(gradient).all() for gradient in present_gradients), (
            component
        )
        initial_zero_gradients.extend(
            name
            for (name, _parameter), gradient in zip(
                parameters, present_gradients, strict=True
            )
            if not torch.count_nonzero(gradient)
        )
    # The query token is initialized to zero, so the first LayerNorm sees an
    # identically zero normalized input and its scale has no gradient until the
    # first Adam step moves that token. The second automatic step must train it.
    assert initial_zero_gradients == ["interaction_value.query_norm.weight"]

    before = {
        name: parameter.detach().clone()
        for name, parameter in module.policy.named_parameters()
    }
    module.zero_grad(set_to_none=True)
    trainer = lightning.Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        max_steps=2,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
        gradient_clip_val=config.optimizer.clip_grad_norm,
        gradient_clip_algorithm="norm",
        deterministic=True,
    )
    trainer.fit(
        module,
        train_dataloaders=DataLoader(
            _BatchDataset(*batches), batch_size=None, num_workers=0
        ),
    )

    assert trainer.global_step == 2
    assert module.environment_steps == 32
    assert module.collection_round == 1
    for name, parameter in active_parameters.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad), name
        assert not torch.equal(before[name], parameter.detach()), name
        assert torch.isfinite(parameter).all(), name


def test_every_optional_model_field_round_trips_and_changes_the_fingerprint() -> None:
    """No accepted architecture setting may disappear from resume structure."""
    config = _all_feature_config()
    serialized = config.model_dump(mode="json")
    model_payload = serialized["model"]
    assert isinstance(model_payload, dict)
    alternatives: dict[str, object] = {
        "recurrent": False,
        "recurrent_channels": 3,
        "recurrent_kernel_size": 5,
        "recurrent_layers": 2,
        "transformer": False,
        "transformer_blocks": 2,
        "transformer_heads": 2,
        "transformer_mlp_ratio": 3,
        "local_patch": False,
        "local_patch_size": 5,
        "local_patch_blocks": 2,
        "belief": False,
        "belief_size": config.model.belief_size + 1,
        "belief_loss_weight": 0.5,
        "belief_feedback": False,
        "interaction_value": False,
    }

    assert alternatives.keys() <= model_payload.keys()
    assert ToadConfig.model_validate(serialized) == config
    baseline = structural_fingerprint(config)
    for field, value in alternatives.items():
        changed = config.model_copy(
            update={"model": config.model.model_copy(update={field: value})}
        )
        assert structural_fingerprint(changed) != baseline, field
