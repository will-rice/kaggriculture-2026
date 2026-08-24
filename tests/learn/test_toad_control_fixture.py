"""Frozen control for Toad's pre-Lightning optimizer boundary."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pytest
import torch

from kaggriculture.learn.model import Policy
from kaggriculture.learn.rollout import Trajectory
from kaggriculture.learn.scripts import toad
from kaggriculture.learn.toad.config import ToadConfig
from kaggriculture.learn.toad.data import BatchKind, LearnerBatch
from kaggriculture.learn.toad.lightning import compute_loss

CONTROL_FIXTURE = Path(__file__).parent / "fixtures" / "toad_control_batch.pt"
CONTROL_THREADS = 32


def load_control_fixture() -> dict[str, object]:
    """Return the checked-in, one-step control batch without regenerating it."""
    return torch.load(CONTROL_FIXTURE, map_location="cpu", weights_only=False)


@contextmanager
def control_fixture_threads() -> Iterator[None]:
    """Use and then restore the CPU thread count that produced exact tensors."""
    previous = torch.get_num_threads()
    torch.set_num_threads(CONTROL_THREADS)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _assert_state_equal(actual: object, expected: object) -> None:
    """Assert exact equality for nested optimizer state, including tensors."""
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        actual_dict = cast(dict[object, object], actual)
        assert actual_dict.keys() == expected.keys()
        for key, value in expected.items():
            _assert_state_equal(actual_dict[key], value)
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_state_equal(actual_item, expected_item)
    else:
        assert actual == expected


def control_fixture_config() -> ToadConfig:
    """Return a frozen control config matching the fixture's small topology."""
    control = ToadConfig.control()
    return control.model_copy(
        update={
            "model": control.model.model_copy(update={"blocks": 1, "channels": 16}),
            "optimizer": control.optimizer.model_copy(
                update={"value_warmup_batches": 0, "value_passes": 1}
            ),
        }
    )


def control_fixture_batch(fixture: dict[str, object]) -> LearnerBatch:
    """Wrap the checked-in segments in the immutable native batch contract."""
    return LearnerBatch(
        segments=tuple(cast(list[dict[str, torch.Tensor]], fixture["segments"])),
        kind=BatchKind.SELFPLAY,
        baseline_only=False,
        first_of_round=True,
        end_of_round=True,
        collected_steps=cast(int, fixture["collected_steps"]),
        round_id=0,
        actor_version=0,
        game_ids=(0,),
        opponent_ids=("self",),
    )


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

    with control_fixture_threads():
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


def test_control_fixture_pins_pure_loss_without_optimizer_side_effects() -> None:
    """The adapter must return fixture tensors without changing policy state."""
    fixture = load_control_fixture()
    policy = Policy(blocks=1, channels=16, value_bound=toad.VALUE_BOUND)
    policy.load_state_dict(cast(dict[str, torch.Tensor], fixture["initial_model"]))
    initial = {name: value.clone() for name, value in policy.state_dict().items()}

    report = compute_loss(
        policy,
        control_fixture_batch(fixture),
        control_fixture_config(),
    )

    actual = {name: value.item() for name, value in report.terms.items()}
    assert actual == pytest.approx(fixture["terms"], rel=1e-6, abs=1e-7)
    assert all(parameter.grad is None for parameter in policy.parameters())
    for name, value in policy.state_dict().items():
        assert torch.equal(value, initial[name])


def test_control_fixture_preserves_tensor_layout_and_fp32() -> None:
    """A differently segmented or lower-precision control batch is not comparable."""
    fixture = load_control_fixture()
    segments = cast(list[dict[str, torch.Tensor]], fixture["segments"])
    initial_model = cast(dict[str, torch.Tensor], fixture["initial_model"])
    trajectory = cast(Trajectory, fixture["trajectory"])
    unroll_length = cast(int, fixture["unroll_length"])
    regenerated = toad._segments(trajectory)

    assert len(segments) == toad.BATCH_SEGMENTS
    assert len(segments) == cast(int, fixture["segment_count"])
    assert unroll_length == toad.UNROLL_LENGTH
    assert int(trajectory.dones.shape[0]) == cast(int, fixture["trajectory_turns"])
    assert int(trajectory.dones.shape[0]) == toad.BATCH_SEGMENTS * unroll_length
    assert tuple(segments[0]) == cast(tuple[str, ...], fixture["segment_keys"])
    assert len(regenerated) == len(segments)
    for captured, current in zip(segments, regenerated, strict=True):
        assert captured.keys() == current.keys()
        for name in toad.ACTED_FIELDS:
            assert captured[name].shape[0] == unroll_length
            assert torch.equal(captured[name], current[name])
        for name in toad.OBSERVED_FIELDS:
            assert captured[name].shape[0] == unroll_length + 1
            assert torch.equal(captured[name], current[name])
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
    trajectory = cast(Trajectory, fixture["trajectory"])
    policy = Policy(blocks=1, channels=16, value_bound=toad.VALUE_BOUND)
    policy.load_state_dict(cast(dict[str, torch.Tensor], fixture["initial_model"]))
    optimizer = toad._optimizer(policy, cast(float, fixture["lr"]))
    optimizer.load_state_dict(cast(dict[str, object], fixture["initial_optimizer"]))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, toad._decay(cast(float, fixture["econ_fraction"]))
    )
    collected_steps = sum(int(item.shaped.shape[0]) for item in [trajectory])
    torch.manual_seed(cast(int, fixture["round_seed"]))
    terms, consumed = toad._update(
        policy,
        optimizer,
        [trajectory],
        "cpu",
        "shaped_money",
        value_passes=cast(int, fixture["value_passes"]),
    )
    collected_steps_with_value_passes = sum(
        int(item.shaped.shape[0]) for item in [trajectory]
    )
    schedule.step()

    assert terms == pytest.approx(fixture["round_terms"], rel=1e-6, abs=1e-7)
    assert consumed == fixture["consumed_policy_batches"]
    assert consumed == len(toad._segments(trajectory)) // toad.BATCH_SEGMENTS
    assert fixture["policy_batches"] == toad._batches_per_update(
        cast(float, fixture["econ_fraction"])
    )
    assert collected_steps == fixture["collected_steps"]
    assert (
        collected_steps_with_value_passes
        == fixture["collected_steps_with_value_passes"]
    )
    assert collected_steps_with_value_passes == collected_steps
    assert schedule.state_dict() == fixture["scheduler_state"]
    assert schedule.get_last_lr() == pytest.approx(fixture["scheduler_last_lr"])
    assert schedule.last_epoch == fixture["scheduler_steps"]
    assert fixture["scheduler_steps"] == fixture["collection_rounds"]
