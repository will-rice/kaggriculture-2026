"""Frozen control for Toad's pre-Lightning optimizer boundary."""

from pathlib import Path
from typing import cast

import pytest
import torch

from kaggriculture.learn.model import Policy
from kaggriculture.learn.scripts import toad

CONTROL_FIXTURE = Path(__file__).parent / "fixtures" / "toad_control_batch.pt"


def load_control_fixture() -> dict[str, object]:
    """Return the checked-in, one-step control batch without regenerating it."""
    return torch.load(CONTROL_FIXTURE, map_location="cpu", weights_only=False)


def _assert_state_equal(actual: object, expected: object) -> None:
    """Assert exact equality for nested optimizer state, including tensors."""
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            _assert_state_equal(actual[key], value)
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_state_equal(actual_item, expected_item)
    else:
        assert actual == expected


def test_control_fixture_pins_one_optimizer_step() -> None:
    """A reordered or numerically changed optimizer boundary must diverge."""
    fixture = load_control_fixture()
    initial_model = cast(dict[str, torch.Tensor], fixture["initial_model"])
    initial_optimizer = cast(dict[str, object], fixture["initial_optimizer"])
    segments = cast(list[dict[str, torch.Tensor]], fixture["segments"])
    updated_model = cast(dict[str, torch.Tensor], fixture["updated_model"])
    updated_optimizer = cast(dict[str, object], fixture["updated_optimizer"])
    policy = Policy(blocks=1, channels=16, value_bound=toad.VALUE_BOUND)
    policy.load_state_dict(initial_model)
    optimizer = toad._optimizer(policy, cast(float, fixture["lr"]))
    optimizer.load_state_dict(initial_optimizer)

    terms = toad._step(
        policy,
        optimizer,
        segments,
        "cpu",
        "shaped_money",
        entropy_cost=cast(float, fixture["entropy_cost"]),
        discounting=cast(float, fixture["gamma"]),
        lmb=cast(float, fixture["lmb"]),
    )

    assert terms == pytest.approx(fixture["terms"], rel=1e-6, abs=1e-7)
    for name, value in policy.state_dict().items():
        assert torch.equal(value, updated_model[name])
    _assert_state_equal(optimizer.state_dict(), updated_optimizer)


def test_control_fixture_preserves_tensor_layout_and_fp32() -> None:
    """A differently segmented or lower-precision control batch is not comparable."""
    fixture = load_control_fixture()
    segments = cast(list[dict[str, torch.Tensor]], fixture["segments"])
    initial_model = cast(dict[str, torch.Tensor], fixture["initial_model"])

    assert len(segments) == toad.BATCH_SEGMENTS
    assert tuple(segments[0]) == cast(tuple[str, ...], fixture["segment_keys"])
    assert all(
        value.dtype == torch.float32
        for segment in segments
        for value in segment.values()
        if value.is_floating_point()
    )
    assert all(value.dtype == torch.float32 for value in initial_model.values())
    assert tuple(initial_model) == tuple(
        Policy(blocks=1, channels=16, value_bound=toad.VALUE_BOUND).state_dict()
    )


def test_control_fixture_pins_round_clocks() -> None:
    """Value replays must not advance collection or scheduler clocks."""
    fixture = load_control_fixture()
    assert fixture["policy_batches"] == toad._batches_per_update(0.5)
    assert fixture["collected_steps_with_value_passes"] == fixture["collected_steps"]
    assert fixture["scheduler_steps"] == fixture["collection_rounds"]
