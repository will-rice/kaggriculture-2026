"""The Rust engine against the installed reference, one action tape each.

Every step of every episode is compared on every observable field of both
seats, plus the done flag and the final rewards. A divergence names the
episode, the step and the first differing path.
"""
# ruff: noqa: D103

import json
import random
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from kaggle_environments import make

from tests.rust.policies import biased_random, husbandry

VARIANT_CONFIGURATION = {
    "episodeSteps": 200,
    "turnsPerDay": 8,
    "shedCapacity": 12,
    "startingMoney": 20000,
    "weedSpawnChance": 0.2,
    "townShopUnlockInterval": 1,
    "townShopSellInterval": 1,
    "townCenterSellInterval": 3,
    "farmHandCostMult": 3,
    "maxMarketOrdersPerTurn": 3,
    "marketParams": {
        "MELON": {"base": 300, "above_func": "linear", "above_target": 2.0},
        "WHEAT": {"below_func": "hinge", "T": 50},
        "EGG": {"above_func": "log10"},
        "GOLD": {"base": 1},
    },
}


def _expected(environment: Any, seat: int) -> dict[str, Any]:  # noqa: ANN401
    observation = dict(environment.state[seat].observation)
    observation.pop("remainingOverageTime", None)
    return json.loads(json.dumps(observation))


def _compare(expected: object, actual: object, path: str) -> None:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        left: dict[Any, Any] = dict(expected)
        right: dict[Any, Any] = dict(actual)
        ordered = ".inventories[" in path
        if (ordered and list(left) != list(right)) or (
            not ordered and set(left) != set(right)
        ):
            raise AssertionError(
                f"{path} keys differ: {list(left)!r} != {list(right)!r}"
            )
        for key in left:
            _compare(left[key], right[key], f"{path}.{key}")
        return
    if (
        isinstance(expected, Sequence)
        and isinstance(actual, Sequence)
        and not isinstance(expected, str)
        and not isinstance(actual, str)
    ):
        if len(expected) != len(actual):
            raise AssertionError(
                f"{path} length differs: {len(expected)} != {len(actual)}"
            )
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _compare(left, right, f"{path}[{index}]")
        return
    if expected != actual or type(expected) is not type(actual):
        raise AssertionError(f"{path} differs: {expected!r} != {actual!r}")


def _record_reference(
    configuration: dict[str, Any],
    policy: Any,  # noqa: ANN401
    turns: int,
) -> tuple[list[list[Any]], list[dict[str, Any]]]:
    """Play the reference and return the tape plus the per-step expectations."""
    environment = make("kaggriculture", configuration=dict(configuration), debug=True)
    environment.reset(2)
    tape: list[list[Any]] = []
    expected = [_snapshot(environment)]
    for _ in range(turns):
        if environment.done:
            break
        actions = [policy(environment.state[seat].observation) for seat in range(2)]
        tape.append(actions)
        environment.step(actions)
        expected.append(_snapshot(environment))
    return tape, expected


def _snapshot(environment: Any) -> dict[str, Any]:  # noqa: ANN401
    return {
        "observations": [_expected(environment, seat) for seat in range(2)],
        "done": all(str(agent.status) == "DONE" for agent in environment.state),
        "rewards": [
            0.0 if agent.reward is None else float(agent.reward)
            for agent in environment.state
        ],
        "seed": environment.info["seed"],
    }


def _run_rust(
    binary: Path, configuration: dict[str, Any], tape: list[list[Any]]
) -> list[Any]:
    payload = json.dumps({"configuration": configuration, "actions": tape})
    completed = subprocess.run(
        [str(binary), "run"],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    )
    return [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]


def _assert_tape_identical(
    binary: Path,
    configuration: dict[str, Any],
    tape: list[list[Any]],
    expected: list[Any],
) -> None:
    actual = _run_rust(binary, configuration, tape)
    assert len(actual) == len(expected), (
        f"{len(actual)} rust states for {len(expected)} expected"
    )
    for step, (want, got) in enumerate(zip(expected, actual, strict=True)):
        assert got["seed"] == want["seed"], f"step {step}: seed differs"
        for seat in range(2):
            observation = dict(got["observations"][seat])
            if "step" not in want["observations"][seat]:
                observation.pop("step")
            _compare(
                want["observations"][seat], observation, f"step {step} seat[{seat}]"
            )
        assert got["done"] == want["done"], f"step {step}: done differs"
        if want["done"]:
            assert got["rewards"] == want["rewards"], f"step {step}: rewards differ"
        else:
            assert got["rewards"] == [], f"step {step}: rewards before done"


def test_biased_random_campaign_has_zero_divergences(
    engine_binary: Path, request: pytest.FixtureRequest
) -> None:
    episodes = int(request.config.getoption("--rust-episodes"))
    turns = int(request.config.getoption("--rust-turns"))
    for episode in range(episodes):
        seed = 173 + 18 * episode
        rng = random.Random(20260904 + episode)
        configuration = {"seed": seed}
        tape, expected = _record_reference(
            configuration, lambda obs, rng=rng: biased_random(rng, obs), turns
        )
        _assert_tape_identical(engine_binary, configuration, tape, expected)


def test_scripted_husbandry_matches_every_step(engine_binary: Path) -> None:
    configuration = {"seed": 991}
    tape, expected = _record_reference(configuration, husbandry, 719)
    # The script must actually reach the animal rules, or it proves nothing.
    final = expected[-1]["observations"][0]
    farm = final["farms"][0]
    assert any(
        isinstance(tile, dict) and tile.get("animal") == "GOOSE"
        for row in farm["tiles"]
        for tile in row
    )
    assert final["farms"][0]["money"] > 3000
    _assert_tape_identical(engine_binary, configuration, tape, expected)


def test_variant_configuration_matches_every_step(engine_binary: Path) -> None:
    rng = random.Random(7)
    configuration = {"seed": 55, **VARIANT_CONFIGURATION}
    tape, expected = _record_reference(
        configuration, lambda obs: biased_random(rng, obs), 200
    )
    assert expected[-1]["done"]
    assert "params" in expected[-1]["observations"][0]["market"]
    _assert_tape_identical(engine_binary, configuration, tape, expected)


def test_random_seed_is_drawn_when_absent(engine_binary: Path) -> None:
    states = _run_rust(engine_binary, {}, [[None, None]])
    assert 0 <= states[0]["seed"] < 2**31
    assert states[0]["observations"][0]["day"] == 0
    assert states[1]["observations"][0]["hour"] == 1
