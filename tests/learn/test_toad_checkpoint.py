"""Round-trip test for phase-1 checkpointing.

Attempt one at this run carried no checkpointing and lost 2.3 hours of training
to an external kill. The property that matters is not merely that weights
reload, but that the learning-rate schedule continues from where it stopped: a
schedule silently restarted from step zero restores the initial rate and changes
the recipe for the remainder of the run without anything looking wrong.
"""

import itertools
import pathlib

import pytest
import torch

from kaggriculture.learn.encoding import SCALARS, TILE_PLANES
from kaggriculture.learn.model import Policy
from kaggriculture.learn.scripts import toad_phase1
from kaggriculture.learn.toad_loss import (
    ADAM_EPS,
    LEARNING_RATE,
    MIN_LR_MOD,
    TOTAL_STEPS,
)

# The arm's own mix, so the schedule under test is the one that runs.
ECON_FRACTION = 0.5


def _fresh() -> tuple[
    Policy, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler
]:
    """Return a policy, optimizer and schedule as `main` builds them."""
    policy = Policy(blocks=1, channels=16, value_bound=toad_phase1.VALUE_BOUND)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, toad_phase1._decay(ECON_FRACTION)
    )
    return policy, optimizer, schedule


@pytest.mark.parametrize(("econ_fraction", "seats"), [(0.0, 48), (0.5, 36), (0.25, 42)])
def test_the_schedule_reaches_its_floor_no_earlier_than_the_budget(
    econ_fraction: float, seats: int
) -> None:
    """The decay must span the budget, not floor a quarter of the way short.

    Their linear schedule reaches ``min_lr_mod`` at 99% of ``total_steps``. An
    arm that mixes opponents records fewer seats per round than the environment
    count suggests -- 36 rather than 48 at ``--econ-fraction 0.5`` -- so a
    schedule keyed on the naive count floors at 74% of the budget and the last
    quarter of the run trains at 1e-6. The seat counts are literals and the run
    is played forward a round at a time, so this fails on the real quantity
    rather than agreeing with the helper it checks.
    """
    decay = toad_phase1._decay(econ_fraction)
    steps, update = 0, 0
    while steps < TOTAL_STEPS:
        steps += seats * toad_phase1.TURNS
        update += 1
    floor = next(step for step in itertools.count() if decay(step) == MIN_LR_MOD)

    assert decay(0) == 1.0
    assert floor / update > 0.98


def test_resuming_continues_the_schedule_rather_than_restarting_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The LR after a resume must equal the uninterrupted run's LR.

    Broken by restarting the schedule, which is the natural way to get this
    wrong and is invisible except in this number.
    """
    monkeypatch.setattr(toad_phase1, "RUNS", tmp_path)

    # An uninterrupted run: 40 schedule steps.
    _, _, uninterrupted = _fresh()
    for _ in range(40):
        uninterrupted.step()
    expected = uninterrupted.get_last_lr()[0]

    # A run that stops at 25 and resumes.
    policy, optimizer, schedule = _fresh()
    for _ in range(25):
        schedule.step()
    path = toad_phase1._checkpoint(
        policy, optimizer, schedule, steps=863_000, update=25, prefix="phase1"
    )
    assert path.is_file()

    restored_policy, restored_optimizer, restored_schedule = _fresh()
    steps, update = toad_phase1._restore(
        path, restored_policy, restored_optimizer, restored_schedule, "cpu"
    )
    assert (steps, update) == (863_000, 25)
    for _ in range(15):
        restored_schedule.step()

    assert restored_schedule.get_last_lr()[0] == expected


def test_the_checkpoint_carries_weights_and_adam_moments(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Weights and optimizer state must both survive, not just weights.

    Adam's moments are as much of the training state as the parameters; dropping
    them restarts the optimizer cold and loses the run's accumulated scaling.
    """
    monkeypatch.setattr(toad_phase1, "RUNS", tmp_path)
    policy, optimizer, schedule = _fresh()
    # Take a real step so the moments are populated.
    policy(
        torch.randn(2, TILE_PLANES, 10, 10),
        torch.randn(2, SCALARS),
        torch.zeros(2, 20, dtype=torch.int64),
    )[2].sum().backward()
    optimizer.step()

    path = toad_phase1._checkpoint(
        policy, optimizer, schedule, steps=1, update=25, prefix="phase1"
    )
    restored_policy, restored_optimizer, restored_schedule = _fresh()
    toad_phase1._restore(
        path, restored_policy, restored_optimizer, restored_schedule, "cpu"
    )

    for before, after in zip(
        policy.state_dict().values(), restored_policy.state_dict().values(), strict=True
    ):
        assert torch.equal(before, after)
    assert restored_optimizer.state_dict()["state"], "Adam moments were not restored"


def test_the_checkpoint_write_is_atomic(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No temporary file may survive, or a kill mid-write leaves a torn checkpoint."""
    monkeypatch.setattr(toad_phase1, "RUNS", tmp_path)
    policy, optimizer, schedule = _fresh()
    toad_phase1._checkpoint(
        policy, optimizer, schedule, steps=1, update=50, prefix="phase1b"
    )
    assert list(tmp_path.glob("*.pt"))
    assert not list(tmp_path.glob("*.tmp"))
