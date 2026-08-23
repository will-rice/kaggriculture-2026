"""Tests for the self-play loop's run-control guards and its episode metrics."""

import pytest
import torch

from kaggriculture.constants import STARTING_MONEY
from kaggriculture.learn.ppo import PpoConfig
from kaggriculture.learn.progress import POTENTIAL_COMPONENTS
from kaggriculture.learn.rollout import Trajectory
from kaggriculture.learn.scripts.selfplay import (
    COLLAPSE_FRACTION,
    COLLAPSE_PATIENCE,
    FARMING_PATIENCE,
    played,
    refuse_collapse,
    refuse_farming,
)


def test_a_collapsing_run_is_stopped() -> None:
    """The measured collapse: 17,675 to 9 over five iterations, everything else fine.

    Every other signal stayed plausible while it happened -- finite losses, an
    illegal count of zero, a rising KL that reads like exploration -- so this
    number is the only one that says the update is destroying its own start.
    """
    with pytest.raises(ValueError, match="destroying what it started from"):
        refuse_collapse([17675.0, 10869.0, 1557.0, 45.0, 33.0, 9.0])


def test_a_run_that_dips_and_recovers_is_not_stopped() -> None:
    """The clone arm dipped 8,087 to 5,940 and went on to 18,233.

    A guard that fired on that would have stopped the only arm that worked, so
    the threshold has to sit outside ordinary variance rather than at the first
    bad iteration.
    """
    refuse_collapse([1411.0, 3480.0, 6659.0, 8087.0, 5940.0, 13752.0])


def test_a_run_from_noise_has_nothing_to_collapse_from() -> None:
    """A fresh start opens at a bank of zero and must not trip the guard.

    The guard protects a competent start. It is not a demand for progress, and
    the fresh arm of the paired comparison banked exactly zero for 35 straight
    iterations while legitimately exploring.
    """
    refuse_collapse([0.0] * (COLLAPSE_PATIENCE + 3))


def test_a_trivial_opening_bank_is_not_something_to_protect() -> None:
    """A fall from 100 coins to nothing is noise, not a policy being destroyed.

    The bar is ``STARTING_MONEY``: a farm ending below the 3,000 it opened with
    has not preserved its own capital, so there is nothing there worth stopping
    a run over. Without a floor the guard fires on any opening above zero, which
    would have stopped the fresh arm the moment it banked a single coin and then
    stopped banking -- exactly the exploration it was supposed to be doing.
    """
    refuse_collapse([100.0, *([0.0] * COLLAPSE_PATIENCE)])


def test_a_bank_that_kept_its_capital_is_protected() -> None:
    """Just above the floor, the guard is live -- the floor is not a way out."""
    with pytest.raises(ValueError, match="destroying what it started from"):
        refuse_collapse([float(STARTING_MONEY), *([0.0] * COLLAPSE_PATIENCE)])


def test_one_bad_iteration_is_not_a_collapse() -> None:
    """Patience is what separates a collapse from a noisy batch."""
    below = COLLAPSE_FRACTION * 10000.0 - 1.0
    refuse_collapse([10000.0, *([below] * (COLLAPSE_PATIENCE - 1))])


# One window's worth of each series, so a test can say "then" by concatenating.
WINDOW = [1.0] * FARMING_PATIENCE


def test_a_pipeline_grown_instead_of_converted_is_stopped() -> None:
    """The named failure of the shaped reward: standing stock, a flat bank.

    A potential-based term makes this unprofitable rather than impossible, and
    this project has already produced two agents that maximised a shaped term
    without banking -- one that suppressed the opponent, one that hired on
    nearly every turn. Both looked healthy in every metric except the one nobody
    was charting next to the shaped one.
    """
    banked = [*[20000.0] * FARMING_PATIENCE, *[20100.0] * FARMING_PATIENCE]
    pipeline = [*[4000.0] * FARMING_PATIENCE, *[9000.0] * FARMING_PATIENCE]

    with pytest.raises(ValueError, match="instead of converting it"):
        refuse_farming(banked, pipeline)


def test_a_pipeline_that_is_converted_is_the_point_and_is_not_stopped() -> None:
    """A growing pipeline with a growing bank is the shaping doing its job.

    This is the assertion that stops the guard from being a ban on shaping. The
    pipeline more than doubles here -- further than the failing case above --
    and the run continues, because the bank went with it.
    """
    banked = [*[20000.0] * FARMING_PATIENCE, *[45000.0] * FARMING_PATIENCE]
    pipeline = [*[4000.0] * FARMING_PATIENCE, *[12000.0] * FARMING_PATIENCE]

    refuse_farming(banked, pipeline)


def test_a_flat_bank_without_a_growing_pipeline_is_merely_a_plateau() -> None:
    """The plateau this whole change exists to break must not trip the guard.

    Measured: 65 iterations at a bank of about 21,000 and nothing moving. That
    is the state the shaped reward is being introduced *from*, so a guard that
    fired on it would stop every run before it started.
    """
    refuse_farming(
        [21000.0] * (2 * FARMING_PATIENCE), [4000.0] * (2 * FARMING_PATIENCE)
    )


def test_a_short_run_is_never_judged_for_farming() -> None:
    """Two windows or nothing; a pipeline growing in the first minutes is the point.

    Without the length bar the guard fires on the opening iterations of every
    shaped run, when the pipeline is climbing off zero by construction and the
    bank has not had a season to respond.
    """
    banked = [*WINDOW, *[1.0] * (FARMING_PATIENCE - 1)]
    pipeline = [*WINDOW, *[1000.0] * (FARMING_PATIENCE - 1)]

    refuse_farming(banked, pipeline)


def test_a_run_with_no_pipeline_to_grow_is_not_judged_for_farming() -> None:
    """A ratio against zero is not a measurement, and a fresh start opens there."""
    refuse_farming([0.0] * (2 * FARMING_PATIENCE), [0.0] * (2 * FARMING_PATIENCE))


def test_the_farming_guard_is_offered_the_pipeline_and_not_the_bank() -> None:
    """The one metric the guard reads has to exclude the bank, and now must.

    ``refuse_farming`` fires when stock piles up while the bank stays flat.
    Since the potential became net worth its total *contains* the bank, so a
    guard handed ``potential/total`` would watch a number that rises precisely
    when the run is going well -- and the failure it exists for, a pipeline
    growing against a flat bank, would no longer be expressible in it. Hence
    ``potential/pipeline``, which is what the loop passes.

    Each component is given a different value so that a total, a pipeline and a
    single column cannot be confused for one another; a potential of all ones
    would let any two of the three swap.
    """
    turns = 4
    money = POTENTIAL_COMPONENTS.index("money")
    components = torch.arange(1, len(POTENTIAL_COMPONENTS) + 1, dtype=torch.float32)
    trajectory = _trajectory(components.tile((turns, 1)))

    measured = played([trajectory], PpoConfig())

    assert measured["potential/money"] == pytest.approx(float(components[money]))
    assert measured["potential/total"] == pytest.approx(float(components.sum()))
    assert measured["potential/pipeline"] == pytest.approx(
        float(components.sum()) - float(components[money])
    )
    assert measured["potential/pipeline"] < measured["potential/total"]


def _trajectory(potentials: torch.Tensor) -> Trajectory:
    """Return a trajectory carrying these potentials and nothing else of interest.

    ``played`` reads five of the twenty-odd fields, so the rest are single-element
    placeholders rather than correctly shaped tensors: a metrics test that had to
    be updated whenever the board's plane count changed would be a test of the
    encoding, which is tested where the encoding is.
    """
    turns = potentials.shape[0]
    empty = torch.zeros(1)
    return Trajectory(
        board=empty,
        scalars=empty,
        positions=empty,
        unit_actions=empty,
        unit_quantities=empty,
        market_actions=empty,
        unit_masks=empty,
        unit_quantity_masks=empty,
        market_masks=empty,
        log_probs=empty,
        values=empty,
        rewards=torch.zeros(turns),
        own=torch.zeros(turns),
        shaped=torch.zeros(turns),
        shaped_money=torch.zeros(turns),
        margin=torch.zeros(turns),
        sparse=torch.zeros(turns),
        potentials=potentials,
        dones=torch.zeros(turns, dtype=torch.bool),
        final_margin=0.0,
        final_bank=float(STARTING_MONEY),
        final_capital=0.0,
        illegal=0,
        sales=0.0,
        units_sold=0.0,
        mean_sale_price=0.0,
        realisation=0.0,
        bought=0.0,
    )
