"""Integration contracts for the dependency-light hybrid policy."""
# ruff: noqa: D103

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import replace
from typing import Any
from unittest.mock import Mock

import pytest
from kaggle_environments import make

from kaggriculture.action_codec import (
    HIRE_SLOT,
    LAND_SLOT,
    MARKET_SLOTS,
    QUANTITIES,
    UNIT_OPS,
    safe_pass_action,
)
from kaggriculture.constants import EPISODE_STEPS
from kaggriculture.features import EncodedObservation, encode_observation
from kaggriculture.hybrid import policy
from tests.feature_golden_generator import rich_observation


def _assert_action_uses_encoded_masks(
    encoded: EncodedObservation, action: dict[str, Any]
) -> None:
    operations = [action["farmer"], *action["hands"]]
    assert len(operations) == encoded.units
    for unit, operation in enumerate(operations):
        verb = operation[0]
        name = (
            f"{verb}:{operation[1]}" if verb in {"PLANT", "PICKUP", "PLACE"} else verb
        )
        op_index = UNIT_OPS.index(name)
        assert encoded.unit_mask[unit][op_index]
        if verb in {"PICKUP", "PLACE"}:
            quantity_index = QUANTITIES.index(operation[2])
            assert encoded.quantity_mask[unit][quantity_index]

    hire_count = sum(order[0] == "HIRE" for order in action["market"])
    for order in action["market"]:
        if order[0] == "HIRE":
            continue
        if order[0] == "BUY_LAND":
            slot, bucket = LAND_SLOT, 1
        else:
            slot = MARKET_SLOTS.index((order[0], order[1]))
            bucket = QUANTITIES.index(order[2])
        assert encoded.market_mask[slot][bucket]
    if hire_count:
        hire_bucket = QUANTITIES.index(hire_count)
        assert encoded.market_mask[HIRE_SLOT][hire_bucket]


def _assert_meaningful_activity(
    *,
    non_pass_turns: int,
    market_turns: int,
    changed_farms: int,
    reward_differences: int,
) -> None:
    assert non_pass_turns > 0, "candidate emitted no non-PASS unit action"
    assert market_turns > 0, "candidate emitted no market action"
    assert changed_farms > 0, "candidate never changed its live farm state"
    assert reward_differences > 0, "candidate was economically identical to pass"


def test_integrated_policy_encodes_raw_observation_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = rich_observation()
    real_encode = policy.encode_observation
    wrapped = Mock(side_effect=real_encode)
    monkeypatch.setattr(policy, "encode_observation", wrapped)

    action = policy.agent(observation)

    assert wrapped.call_count == 1
    assert set(action) == {"farmer", "hands", "market"}


def test_agent_is_deterministic_and_does_not_mutate_the_observation() -> None:
    observation = rich_observation()
    before = deepcopy(observation)

    first = policy.agent(observation)
    second = policy.agent(observation)

    assert first == second
    assert observation == before


@pytest.mark.parametrize("exception", [KeyError, TypeError, ValueError, IndexError])
def test_outer_failure_boundary_returns_exact_safe_pass_for_encoding_errors(
    monkeypatch: pytest.MonkeyPatch,
    exception: type[Exception],
) -> None:
    monkeypatch.setattr(
        policy,
        "encode_observation",
        Mock(side_effect=exception("new schema")),
    )

    assert policy.agent({"unexpected": "schema"}) == safe_pass_action()


@pytest.mark.parametrize("exception", [KeyError, TypeError, ValueError, IndexError])
def test_outer_failure_boundary_does_not_hide_downstream_errors(
    monkeypatch: pytest.MonkeyPatch,
    exception: type[Exception],
) -> None:
    monkeypatch.setattr(
        policy,
        "decode_selected",
        Mock(side_effect=exception("broken invariant")),
    )

    with pytest.raises(exception, match="broken invariant"):
        policy.agent(rich_observation())


def test_repeated_hires_are_checked_as_one_quantity_bucket() -> None:
    observation = rich_observation()
    encoded = encode_observation(observation, 0)
    rows = [list(row) for row in encoded.market_mask]
    rows[HIRE_SLOT][1] = False
    rows[HIRE_SLOT][3] = True
    encoded = replace(encoded, market_mask=tuple(tuple(row) for row in rows))
    action = {
        "farmer": ["PASS"],
        "hands": [["PASS"]] * (encoded.units - 1),
        "market": [["HIRE"], ["HIRE"], ["HIRE"]],
    }

    _assert_action_uses_encoded_masks(encoded, action)


def test_live_policy_multi_hire_turn_uses_the_aggregate_mask_bucket() -> None:
    observation = rich_observation()
    encoded = encode_observation(observation, 0)
    action = policy.agent(observation)
    hire_count = sum(order == ["HIRE"] for order in action["market"])

    assert hire_count == 3
    assert encoded.market_mask[HIRE_SLOT][QUANTITIES.index(hire_count)]
    _assert_action_uses_encoded_masks(encoded, action)


def test_generated_legal_observations_emit_only_mask_enabled_categories() -> None:
    rng = random.Random(7_031)
    environment = make(
        "kaggriculture",
        configuration={"episodeSteps": 96, "seed": 7_031},
        debug=True,
    )
    environment.reset(2)
    non_pass_turns = 0
    market_turns = 0
    for _ in range(48):
        observation = deepcopy(environment.state[0].observation)
        action = policy.agent(observation)
        encoded = encode_observation(observation, int(observation.get("player", 0)))
        _assert_action_uses_encoded_masks(encoded, action)
        operations = [action["farmer"], *action["hands"]]
        non_pass_turns += any(operation[0] != "PASS" for operation in operations)
        market_turns += bool(action["market"])
        opponent = {
            "farmer": [rng.choice(("PASS", "NORTH", "SOUTH", "EAST", "WEST"))],
            "hands": [],
            "market": [],
        }
        environment.step([action, opponent])
        if environment.done:
            break
    assert non_pass_turns > 0
    assert market_turns > 0


def test_activity_gate_rejects_an_inert_agent() -> None:
    with pytest.raises(AssertionError, match="no non-PASS"):
        _assert_meaningful_activity(
            non_pass_turns=0,
            market_turns=0,
            changed_farms=0,
            reward_differences=0,
        )


@pytest.mark.slow
def test_fixed_twenty_seed_reference_episodes_have_no_error_or_invalid() -> None:
    non_pass_turns = 0
    market_turns = 0
    changed_farms = 0
    reward_differences = 0
    for seed in range(20):
        environment = make(
            "kaggriculture",
            configuration={"episodeSteps": EPISODE_STEPS, "seed": seed},
        )
        environment.run([policy.agent, "pass"])
        statuses = [str(state.status) for state in environment.steps[-1]]
        assert all(
            "ERROR" not in status and "INVALID" not in status for status in statuses
        )
        assert statuses == ["DONE", "DONE"]
        initial_money = environment.steps[0][0].observation.farms[0].money
        final = environment.steps[-1]
        changed_farms += final[0].observation.farms[0].money != initial_money
        reward_differences += final[0].reward != final[1].reward
        for step in environment.steps[1:]:
            action = step[0].action
            operations = [action["farmer"], *action["hands"]]
            non_pass_turns += any(op[0] != "PASS" for op in operations)
            market_turns += bool(action["market"])
    _assert_meaningful_activity(
        non_pass_turns=non_pass_turns,
        market_turns=market_turns,
        changed_farms=changed_farms,
        reward_differences=reward_differences,
    )
