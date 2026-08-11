"""Shared differential assertions for the simulator suite."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from kaggriculture.sim.state import SimState, unpack


def pytest_addoption(parser: Any) -> None:  # noqa: ANN401
    """Register scalable differential-campaign controls."""
    parser.addoption("--sim-episodes", type=int, default=4)
    parser.addoption("--sim-turns", type=int, default=48)


def _reference_projection(reference: Any, seat: int) -> dict[str, Any]:  # noqa: ANN401
    agents = reference.state if hasattr(reference, "state") else reference
    observation = dict(agents[seat].observation)
    observation.pop("remainingOverageTime", None)
    return observation


def _canonical(value: object, path: str = "") -> object:
    if isinstance(value, Mapping):
        if ".inventories[" in path:
            return [
                [key, _canonical(item, f"{path}.{key}")] for key, item in value.items()
            ]
        return {key: _canonical(value[key], f"{path}.{key}") for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _canonical(item, f"{path}[{index}]") for index, item in enumerate(value)
        ]
    return value


def _digest(value: object) -> bytes:
    encoded = json.dumps(
        _canonical(value), separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).digest()


def _compare(expected: object, actual: object, path: str) -> None:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        expected_keys = list(expected)
        actual_keys = list(actual)
        ordered = ".inventories[" in path
        if (ordered and expected_keys != actual_keys) or (
            not ordered and set(expected_keys) != set(actual_keys)
        ):
            raise AssertionError(
                f"{path} keys differ: {expected_keys!r} != {actual_keys!r}"
            )
        for key in expected_keys:
            _compare(expected.get(key), actual.get(key), f"{path}.{key}")
        return
    if (
        isinstance(expected, Sequence)
        and isinstance(actual, Sequence)
        and not isinstance(expected, (str, bytes))
        and not isinstance(actual, (str, bytes))
    ):
        if len(expected) != len(actual):
            raise AssertionError(
                f"{path} length differs: {len(expected)} != {len(actual)}"
            )
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _compare(left, right, f"{path}[{index}]")
        return
    if expected != actual:
        raise AssertionError(f"{path} differs: {expected!r} != {actual!r}")


def assert_identical(reference: Any, batched: SimState, batch: int) -> None:  # noqa: ANN401
    """Assert every observable field matches, localizing the first mismatch."""
    for seat in range(2):
        expected = _reference_projection(reference, seat)
        actual = unpack(batched, batch, seat)
        if _digest(expected) != _digest(actual):
            _compare(expected, actual, f"seat[{seat}]")
    agents = reference.state if hasattr(reference, "state") else reference
    expected_done = all(str(agent.status) == "DONE" for agent in agents)
    if expected_done != bool(batched.done[batch]):
        raise AssertionError(
            f"done differs: {expected_done!r} != {bool(batched.done[batch])!r}"
        )
    for seat, agent in enumerate(agents):
        expected_reward = 0.0 if agent.reward is None else float(agent.reward)
        actual_reward = float(batched.reward[batch, seat])
        if expected_reward != actual_reward:
            raise AssertionError(
                f"reward[{seat}] differs: {expected_reward!r} != {actual_reward!r}"
            )
