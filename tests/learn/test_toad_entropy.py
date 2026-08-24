"""Head-wise entropy accounting for the native Toad learner."""

import math
from dataclasses import replace
from typing import cast

import pytest
import torch
from pydantic import ValidationError

from kaggriculture.constants import EPISODE_STEPS
from kaggriculture.learn.encoding import IGNORE, TRANSFER_OPS
from kaggriculture.learn.model import Policy
from kaggriculture.learn.scripts import toad
from kaggriculture.learn.toad.config import (
    EntropyControllerConfig,
    EntropyControllersConfig,
    ToadConfig,
)
from kaggriculture.learn.toad.data import LearnerBatch
from kaggriculture.learn.toad.lightning import (
    EntropyControllerState,
    HeadEntropy,
    ToadLightningModule,
    compute_head_entropy,
    compute_loss,
    update_entropy_controller,
)
from tests.learn.test_toad_control_fixture import (
    control_fixture_batch,
    control_fixture_config,
    load_control_fixture,
)


def _controller_config(
    **updates: float,
) -> EntropyControllerConfig:
    """Return a valid controller with visible, hand-checkable changes."""
    values = {
        "initial_target": 1.0,
        "target_change_per_step": 0.0,
        "target_floor": 0.0,
        "initial_multiplier": 0.1,
        "multiplier_change_per_step": 0.01,
        "minimum": 0.01,
        "maximum": 1.0,
        **updates,
    }
    return EntropyControllerConfig.model_validate(values)


def _controllers(
    controller: EntropyControllerConfig | None = None,
) -> EntropyControllersConfig:
    """Use one immutable value for all heads in config-validation tests."""
    selected = controller or _controller_config(multiplier_change_per_step=1e-6)
    return EntropyControllersConfig(
        operation=selected,
        quantity=selected,
        market=selected,
    )


def test_controller_reduces_multiplier_above_target() -> None:
    """An over-entropic policy must weaken its entropy reward."""
    state = EntropyControllerState(target=1.0, multiplier=0.1, last_steps=10)

    updated = update_entropy_controller(
        state,
        observed=1.2,
        steps=20,
        config=_controller_config(),
    )

    assert updated.target == 1.0
    assert updated.multiplier == pytest.approx(0.09)
    assert updated.last_steps == 20


def test_controller_increases_multiplier_below_target() -> None:
    """An under-entropic policy must strengthen its entropy reward."""
    state = EntropyControllerState(target=1.0, multiplier=0.1, last_steps=10)

    updated = update_entropy_controller(
        state,
        observed=0.8,
        steps=20,
        config=_controller_config(),
    )

    assert updated.multiplier == pytest.approx(0.11)


def test_controller_equality_advances_steps_without_changing_multiplier() -> None:
    """Exact target equality is neutral rather than taking the low branch."""
    state = EntropyControllerState(target=1.0, multiplier=0.1, last_steps=10)

    updated = update_entropy_controller(
        state,
        observed=1.0,
        steps=20,
        config=_controller_config(),
    )

    assert updated == EntropyControllerState(
        target=1.0,
        multiplier=state.multiplier,
        last_steps=20,
    )


def test_controller_decays_target_to_its_floor_before_comparison() -> None:
    """The current completed-round target, not the stale target, sets direction."""
    state = EntropyControllerState(target=0.7, multiplier=0.1, last_steps=10)
    config = _controller_config(
        target_change_per_step=-0.02,
        target_floor=0.6,
    )

    updated = update_entropy_controller(
        state,
        observed=0.65,
        steps=20,
        config=config,
    )

    assert updated.target == pytest.approx(0.6)
    assert updated.multiplier == pytest.approx(0.09)


@pytest.mark.parametrize(
    ("observed", "initial", "expected"),
    [(0.0, 0.95, 1.0), (2.0, 0.011, 0.01)],
)
def test_controller_clamps_multiplier_to_configured_bounds(
    observed: float,
    initial: float,
    expected: float,
) -> None:
    """A large valid round delta cannot move alpha outside its safe range."""
    state = EntropyControllerState(target=1.0, multiplier=initial, last_steps=0)

    updated = update_entropy_controller(
        state,
        observed=observed,
        steps=10,
        config=_controller_config(),
    )

    assert updated.multiplier == pytest.approx(expected)


@pytest.mark.parametrize(("last_steps", "steps"), [(-1, 0), (10, 9), (0, -1)])
def test_controller_rejects_negative_or_backward_step_deltas(
    last_steps: int,
    steps: int,
) -> None:
    """Resume corruption cannot silently invert target/controller time."""
    state = EntropyControllerState(target=1.0, multiplier=0.1, last_steps=last_steps)

    with pytest.raises(ValueError, match="steps"):
        update_entropy_controller(
            state,
            observed=1.0,
            steps=steps,
            config=_controller_config(),
        )


@pytest.mark.parametrize("observed", [math.nan, math.inf, -math.inf])
def test_controller_rejects_nonfinite_observations(observed: float) -> None:
    """A nonfinite diagnostic cannot poison checkpointed controller state."""
    state = EntropyControllerState(target=1.0, multiplier=0.1, last_steps=0)

    with pytest.raises(ValueError, match="finite"):
        update_entropy_controller(
            state,
            observed=observed,
            steps=1,
            config=_controller_config(),
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"minimum": 0.2},
        {"maximum": 0.05},
        {"initial_target": math.nan},
        {"target_change_per_step": math.inf},
    ],
)
def test_controller_config_rejects_invalid_bounds_and_nonfinite_numbers(
    updates: dict[str, float],
) -> None:
    """Invalid controller arithmetic fails during config construction."""
    with pytest.raises(ValidationError):
        _controller_config(**updates)


def test_controller_config_is_frozen_and_forbids_extra_fields() -> None:
    """Checkpoint semantics cannot drift through mutation or ignored keys."""
    config = _controller_config()

    with pytest.raises(ValidationError, match="frozen"):
        config.maximum = 2.0  # ty: ignore[invalid-assignment]
    with pytest.raises(ValidationError, match="extra_forbidden"):
        EntropyControllerConfig.model_validate({**config.model_dump(), "unknown": 1.0})


def test_controller_speed_must_be_safe_for_the_largest_actual_round() -> None:
    """The decrease factor stays positive for two self-play seats per game."""
    environments = 2
    max_steps_per_round = 2 * environments * (EPISODE_STEPS - 1)
    unsafe = _controller_config(multiplier_change_per_step=1.0 / max_steps_per_round)

    with pytest.raises(ValidationError, match="largest.*round|round.*less than 1"):
        ToadConfig.model_validate(
            {
                "population": {"environments_per_rank": environments},
                "optimizer": {"entropy": _controllers(unsafe).model_dump()},
            }
        )


def test_existing_config_defaults_to_the_fixed_entropy_path() -> None:
    """Historical payloads remain valid and do not opt into adaptive entropy."""
    config = ToadConfig.model_validate({})

    assert not config.optimizer.adaptive_entropy
    assert isinstance(config.optimizer.entropy.operation, EntropyControllerConfig)


def _adaptive_fixture() -> tuple[
    Policy,
    LearnerBatch,
    ToadConfig,
    dict[str, EntropyControllerState],
]:
    """Return the real control batch with three distinct controller weights."""
    fixture = load_control_fixture()
    policy = Policy(blocks=1, channels=16, value_bound=toad.VALUE_BOUND)
    policy.load_state_dict(cast(dict[str, torch.Tensor], fixture["initial_model"]))
    base = control_fixture_config()
    config = base.model_copy(
        update={
            "optimizer": base.optimizer.model_copy(
                update={
                    "adaptive_entropy": True,
                    "entropy": _controllers(),
                }
            )
        }
    )
    states = {
        "operation": EntropyControllerState(1.0, 0.1, 0),
        "quantity": EntropyControllerState(1.0, 0.2, 0),
        "market": EntropyControllerState(1.0, 0.3, 0),
    }
    return policy, control_fixture_batch(fixture), config, states


def _adaptive_module_config(*, warmup: int = 0) -> ToadConfig:
    """Return controller settings whose first observed round changes state."""
    base = control_fixture_config()
    controller = _controller_config(
        initial_target=10.0,
        target_change_per_step=-0.001,
        target_floor=0.5,
        initial_multiplier=0.1,
        multiplier_change_per_step=1e-6,
    )
    return base.model_copy(
        update={
            "optimizer": base.optimizer.model_copy(
                update={
                    "adaptive_entropy": True,
                    "entropy": _controllers(controller),
                    "value_warmup_batches": warmup,
                }
            )
        }
    )


def test_adaptive_loss_uses_independent_head_means_without_fixed_double_count() -> None:
    """Each alpha weights only its valid-action mean and replaces fixed entropy."""
    policy, batch, config, states = _adaptive_fixture()
    fixed = compute_loss(policy, batch, control_fixture_config())

    adaptive = compute_loss(
        policy,
        batch,
        config,
        entropy_state=states,
    )

    weighted = []
    for name, multiplier in (
        ("operation", 0.1),
        ("quantity", 0.2),
        ("market", 0.3),
    ):
        raw = adaptive.terms[f"entropy/{name}_raw"]
        term = adaptive.terms[f"entropy/{name}_weighted"]
        torch.testing.assert_close(term, raw * multiplier)
        weighted.append(term)
    torch.testing.assert_close(adaptive.terms["entropy"], sum(weighted))
    torch.testing.assert_close(
        adaptive.total,
        fixed.total - fixed.terms["entropy"] + adaptive.terms["entropy"],
    )


def test_adaptive_entropy_is_not_added_to_baseline_only_total() -> None:
    """Warmup observes entropy but cannot train the policy through its reward."""
    policy, batch, config, states = _adaptive_fixture()

    adaptive = compute_loss(
        policy,
        batch,
        config,
        entropy_state=states,
        baseline_only=True,
    )

    assert adaptive.terms["entropy"].item() < 0
    torch.testing.assert_close(adaptive.total, adaptive.terms["baseline"])


def test_adaptive_entropy_loss_terms_stay_fp32_under_autocast() -> None:
    """Controller multipliers cannot pull differentiable entropy back to BF16."""
    policy, batch, config, states = _adaptive_fixture()

    with torch.autocast("cpu", dtype=torch.bfloat16):
        adaptive = compute_loss(
            policy,
            batch,
            config,
            entropy_state=states,
        )

    for name in ("operation", "quantity", "market"):
        assert adaptive.terms[f"entropy/{name}_raw"].dtype == torch.float32
        assert adaptive.terms[f"entropy/{name}_weighted"].dtype == torch.float32
    assert adaptive.terms["entropy"].dtype == torch.float32


def test_training_step_updates_once_after_value_replay_from_fresh_entropy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay closes the round but neither contributes entropy nor updates twice."""
    fixture = load_control_fixture()
    module = ToadLightningModule(_adaptive_module_config())
    module.policy.load_state_dict(
        cast(dict[str, torch.Tensor], fixture["initial_model"])
    )
    records: list[dict[str, object]] = []
    monkeypatch.setattr(module, "log_dict", lambda record: records.append(dict(record)))
    batch = control_fixture_batch(fixture)
    fresh = replace(batch, end_of_round=False)
    replay = replace(
        batch,
        baseline_only=True,
        first_of_round=False,
        collected_steps=0,
    )
    initial = dict(module.entropy_state)
    expected = compute_loss(
        module.policy,
        fresh,
        module.config,
        entropy_state=initial,
    ).entropy

    module.training_step(fresh, 0)

    assert module.entropy_state == initial
    assert len(module._round_fresh_entropy) == 1
    for _name, stat in module._round_fresh_entropy[0].items():
        assert stat.sum.dtype == torch.float32
        assert stat.valid.dtype == torch.int64
        assert not stat.sum.requires_grad

    module.training_step(replay, 1)

    assert len(records) == 1
    assert not module._round_fresh_entropy
    for name, stat in expected.items():
        assert module.entropy_state[name].last_steps == batch.collected_steps
        torch.testing.assert_close(
            cast(torch.Tensor, records[0][f"entropy/{name}_sum"]),
            stat.sum,
        )
        torch.testing.assert_close(
            cast(torch.Tensor, records[0][f"entropy/{name}_valid"]),
            stat.valid.float(),
        )


def test_warmup_still_observes_each_fresh_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Policy-loss suppression cannot suppress controller observations."""
    fixture = load_control_fixture()
    module = ToadLightningModule(_adaptive_module_config(warmup=1))
    module.policy.load_state_dict(
        cast(dict[str, torch.Tensor], fixture["initial_model"])
    )
    records: list[dict[str, object]] = []
    monkeypatch.setattr(module, "log_dict", lambda record: records.append(dict(record)))
    batch = control_fixture_batch(fixture)

    total = module.training_step(batch, 0)

    assert module.warmup_remaining == 0
    assert all(
        state.last_steps == batch.collected_steps
        for state in module.entropy_state.values()
    )
    assert all(
        cast(torch.Tensor, records[0][f"entropy/{name}_valid"]).item() > 0
        for name in ("operation", "quantity", "market")
    )
    expected = compute_loss(
        module.policy,
        batch,
        module.config,
        entropy_state={
            name: EntropyControllerState(
                target=config.initial_target,
                multiplier=config.initial_multiplier,
                last_steps=0,
            )
            for name, config in module.config.optimizer.entropy.items()
        },
        baseline_only=True,
    )
    torch.testing.assert_close(total, expected.total)


def test_completed_round_controller_applies_to_the_next_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The batch that supplies an observation still uses the prior multiplier."""
    fixture = load_control_fixture()
    module = ToadLightningModule(_adaptive_module_config())
    module.policy.load_state_dict(
        cast(dict[str, torch.Tensor], fixture["initial_model"])
    )
    monkeypatch.setattr(module, "log_dict", lambda _record: None)
    batch = control_fixture_batch(fixture)

    first = module.training_step(batch, 0)
    next_states = dict(module.entropy_state)
    assert all(state.multiplier > 0.1 for state in next_states.values())
    second_batch = replace(batch, round_id=1)
    expected_second = compute_loss(
        module.policy,
        second_batch,
        module.config,
        entropy_state=next_states,
    )

    second = module.training_step(second_batch, 1)

    torch.testing.assert_close(second, expected_second.total)
    assert not torch.equal(first, second)


def test_zero_valid_quantity_advances_target_without_changing_multiplier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No transfer decisions produce no fabricated zero-entropy observation."""
    fixture = load_control_fixture()
    module = ToadLightningModule(_adaptive_module_config())
    module.policy.load_state_dict(
        cast(dict[str, torch.Tensor], fixture["initial_model"])
    )
    records: list[dict[str, object]] = []
    monkeypatch.setattr(module, "log_dict", lambda record: records.append(dict(record)))
    batch = control_fixture_batch(fixture)
    no_transfers = replace(
        batch,
        segments=tuple(
            {
                **segment,
                "unit_actions": torch.where(
                    segment["unit_valid"],
                    torch.zeros_like(segment["unit_actions"]),
                    segment["unit_actions"],
                ),
            }
            for segment in batch.segments
        ),
    )
    before = module.entropy_state["quantity"]

    module.training_step(no_transfers, 0)

    after = module.entropy_state["quantity"]
    assert after.last_steps == batch.collected_steps
    assert after.target < before.target
    assert after.multiplier == before.multiplier
    assert cast(torch.Tensor, records[0]["entropy/quantity_valid"]).item() == 0
    assert torch.isnan(cast(torch.Tensor, records[0]["entropy/quantity_mean"]))
    assert torch.isnan(cast(torch.Tensor, records[0]["entropy/quantity_observed"]))
    assert records[0]["entropy/quantity_target"] == after.target
    assert records[0]["entropy/quantity_multiplier"] == after.multiplier


def _log_probs(shape: tuple[int, ...]) -> torch.Tensor:
    """Return differentiable, uniform two-option log probabilities."""
    return torch.full(shape, -math.log(2), dtype=torch.float32, requires_grad=True)


def _entropy(
    *,
    unit_actions: torch.Tensor,
    unit_valid: torch.Tensor | None = None,
    unit_mask: torch.Tensor | None = None,
    quantity_mask: torch.Tensor | None = None,
    market_mask: torch.Tensor | None = None,
) -> tuple[HeadEntropy, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    leading = unit_actions.shape
    units = _log_probs((*leading, 2))
    quantities = _log_probs((*leading, 2))
    market = _log_probs((*leading[:-1], 3, 2))
    return (
        compute_head_entropy(
            unit_log_probs=units,
            quantity_log_probs=quantities,
            market_log_probs=market,
            unit_masks=(
                torch.ones_like(units, dtype=torch.bool)
                if unit_mask is None
                else unit_mask
            ),
            quantity_masks=(
                torch.ones_like(quantities, dtype=torch.bool)
                if quantity_mask is None
                else quantity_mask
            ),
            market_masks=(
                torch.ones_like(market, dtype=torch.bool)
                if market_mask is None
                else market_mask
            ),
            unit_actions=unit_actions,
            unit_valid=(unit_actions != IGNORE if unit_valid is None else unit_valid),
        ),
        (units, quantities, market),
    )


def test_quantity_entropy_counts_only_transfer_decisions() -> None:
    """A quantity bucket ignored by every selected operation has no gradient."""
    entropy, (_units, quantities, _market) = _entropy(
        unit_actions=torch.tensor([[0, 1], [2, IGNORE]])
    )

    assert entropy.quantity.valid.item() == 0
    assert entropy.quantity.sum.item() == 0.0
    entropy.quantity.sum.backward()
    assert quantities.grad is not None
    assert not quantities.grad.any()


def test_padded_unit_slots_do_not_count_toward_operation_or_quantity_entropy() -> None:
    """Placeholder logits and a placeholder transfer label are not decisions."""
    transfer = TRANSFER_OPS.index(True)
    entropy, (units, quantities, _market) = _entropy(
        unit_actions=torch.tensor([[0, IGNORE], [0, IGNORE]])
    )
    padded_actions = torch.tensor([[0, transfer], [0, transfer]])
    padded_entropy, padded_logits = _entropy(unit_actions=padded_actions)

    assert entropy.operation.valid.item() == 2
    assert entropy.quantity.valid.item() == 0
    torch.testing.assert_close(entropy.operation.sum, torch.tensor(2 * math.log(2)))
    assert units.grad is None
    entropy.operation.sum.backward()
    assert units.grad is not None
    assert not units.grad[:, 1].any()
    assert quantities.grad is None
    # This is the counterfactual that would be wrong if IGNORE were treated as
    # a transfer; a real transfer slot, unlike the padded one above, counts.
    assert padded_entropy.quantity.valid.item() == 2
    assert padded_logits[1].grad is None


def test_explicit_unit_valid_ignores_a_mutated_padded_transfer_action() -> None:
    """A mutable action label cannot turn an absent slot into a decision."""
    transfer = TRANSFER_OPS.index(True)
    unit_valid = torch.tensor([[True, False]])
    baseline, _ = _entropy(
        unit_actions=torch.tensor([[0, IGNORE]]), unit_valid=unit_valid
    )
    changed, (units, quantities, _market) = _entropy(
        unit_actions=torch.tensor([[0, transfer]]), unit_valid=unit_valid
    )

    for before, after in zip(baseline.items(), changed.items(), strict=True):
        assert before[1].valid.item() == after[1].valid.item()
        torch.testing.assert_close(before[1].sum, after[1].sum)
    (changed.operation.sum + changed.quantity.sum).backward()
    assert units.grad is not None
    assert quantities.grad is not None
    assert not units.grad[:, 1].any()
    assert not quantities.grad[:, 1].any()


def test_real_unit_with_ignore_action_remains_an_operation_entropy_decision() -> None:
    """Only unit_valid denotes padding; IGNORE is not an entropy sentinel."""
    entropy, _logits = _entropy(
        unit_actions=torch.tensor([[IGNORE]]),
        unit_valid=torch.tensor([[True]]),
    )

    assert entropy.operation.valid.item() == 1
    assert entropy.quantity.valid.item() == 0


def test_all_padding_and_empty_market_have_exact_safe_zero_statistics() -> None:
    """Empty reductions neither leak NaNs nor manufacture valid decisions."""
    unit_actions = torch.full((2, 3), IGNORE, dtype=torch.int64)
    units = _log_probs((2, 3, 2))
    quantities = _log_probs((2, 3, 2))
    market = _log_probs((2, 0, 2))
    entropy = compute_head_entropy(
        unit_log_probs=units,
        quantity_log_probs=quantities,
        market_log_probs=market,
        unit_masks=torch.zeros_like(units, dtype=torch.bool),
        quantity_masks=torch.zeros_like(quantities, dtype=torch.bool),
        market_masks=torch.zeros_like(market, dtype=torch.bool),
        unit_actions=unit_actions,
        unit_valid=torch.zeros_like(unit_actions, dtype=torch.bool),
    )

    for stat in (entropy.operation, entropy.quantity, entropy.market):
        assert stat.valid.item() == 0
        assert stat.sum.item() == 0.0
        assert torch.isfinite(stat.sum)


def test_market_entropy_counts_each_real_market_slot_once() -> None:
    """Market has no unit-style padding: every present slot is a decision."""
    entropy, _logits = _entropy(unit_actions=torch.tensor([[0, 0], [0, 0]]))

    assert entropy.market.valid.item() == 6
    torch.testing.assert_close(entropy.market.sum, torch.tensor(6 * math.log(2)))


def test_masked_illegal_logits_are_irrelevant_to_entropy_and_gradients() -> None:
    """Changing forbidden logits must not change a valid row's entropy."""
    unit_actions = torch.tensor([[0]])
    units = torch.tensor([[[0.0, 123.0]]], dtype=torch.float32, requires_grad=True)
    quantities = _log_probs((1, 1, 2))
    market = _log_probs((1, 1, 2))
    entropy = compute_head_entropy(
        unit_log_probs=units,
        quantity_log_probs=quantities,
        market_log_probs=market,
        unit_masks=torch.tensor([[[True, False]]]),
        quantity_masks=torch.ones_like(quantities, dtype=torch.bool),
        market_masks=torch.ones_like(market, dtype=torch.bool),
        unit_actions=unit_actions,
        unit_valid=torch.ones_like(unit_actions, dtype=torch.bool),
    )

    assert entropy.operation.sum.item() == 0.0
    entropy.operation.sum.backward()
    assert units.grad is not None
    assert units.grad[..., 1].item() == 0.0


def test_entropy_flattens_batch_and_time_dimensions_in_fp32() -> None:
    """All leading batch dimensions count, and reductions stay FP32 in autocast."""
    transfer = TRANSFER_OPS.index(True)
    actions = torch.tensor([[[transfer, 0]], [[transfer, IGNORE]]])
    units = _log_probs((2, 1, 2, 2)).to(dtype=torch.bfloat16).detach().requires_grad_()
    quantities = (
        _log_probs((2, 1, 2, 2)).to(dtype=torch.bfloat16).detach().requires_grad_()
    )
    market = _log_probs((2, 1, 3, 2)).to(dtype=torch.bfloat16).detach().requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        entropy = compute_head_entropy(
            unit_log_probs=units,
            quantity_log_probs=quantities,
            market_log_probs=market,
            unit_masks=torch.ones_like(units, dtype=torch.bool),
            quantity_masks=torch.ones_like(quantities, dtype=torch.bool),
            market_masks=torch.ones_like(market, dtype=torch.bool),
            unit_actions=actions,
            unit_valid=actions != IGNORE,
        )

    assert entropy.operation.valid.item() == 3
    assert entropy.quantity.valid.item() == 2
    assert entropy.market.valid.item() == 6
    assert entropy.operation.sum.dtype == torch.float32
    assert entropy.quantity.sum.dtype == torch.float32
    assert entropy.market.sum.dtype == torch.float32
