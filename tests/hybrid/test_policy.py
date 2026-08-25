"""Integration contracts for the dependency-light hybrid policy."""
# ruff: noqa: D103

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any, Mapping
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
from kaggriculture.features import encode_observation
from kaggriculture.hybrid import policy
from tests.feature_golden_generator import rich_observation


def _assert_action_uses_encoded_masks(
    observation: Mapping[str, Any], action: dict[str, Any]
) -> None:
    seat = int(observation.get("player", 0))
    encoded = encode_observation(observation, seat)
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

    for order in action["market"]:
        if order[0] == "HIRE":
            slot, bucket = HIRE_SLOT, 1
        elif order[0] == "BUY_LAND":
            slot, bucket = LAND_SLOT, 1
        else:
            slot = MARKET_SLOTS.index((order[0], order[1]))
            bucket = QUANTITIES.index(order[2])
        assert encoded.market_mask[slot][bucket]


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


def test_outer_failure_boundary_returns_exact_safe_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(policy, "encode_observation", Mock(side_effect=KeyError("new")))

    assert policy.agent({"unexpected": "schema"}) == safe_pass_action()


def test_outer_failure_boundary_does_not_hide_development_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        policy.HybridPolicy,
        "decide",
        Mock(side_effect=RuntimeError("broken invariant")),
    )

    with pytest.raises(RuntimeError, match="broken invariant"):
        policy.agent(rich_observation())


def test_generated_legal_observations_emit_only_mask_enabled_categories() -> None:
    rng = random.Random(7_031)
    environment = make(
        "kaggriculture",
        configuration={"episodeSteps": 96, "seed": 7_031},
        debug=True,
    )
    environment.reset(2)
    for _ in range(48):
        observation = deepcopy(environment.state[0].observation)
        action = policy.agent(observation)
        _assert_action_uses_encoded_masks(observation, action)
        opponent = {
            "farmer": [rng.choice(("PASS", "NORTH", "SOUTH", "EAST", "WEST"))],
            "hands": [],
            "market": [],
        }
        environment.step([action, opponent])
        if environment.done:
            break


@pytest.mark.slow
def test_fixed_twenty_seed_reference_episodes_have_no_error_or_invalid() -> None:
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
