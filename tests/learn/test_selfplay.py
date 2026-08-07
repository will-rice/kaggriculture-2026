"""Tests for the self-play loop's run-control guards."""

import pytest

from kaggriculture.constants import STARTING_MONEY
from kaggriculture.learn.scripts.selfplay import (
    COLLAPSE_FRACTION,
    COLLAPSE_PATIENCE,
    refuse_collapse,
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
