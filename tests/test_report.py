"""Tests for win-rate reporting and its confidence intervals."""

from kaggriculture.report import spearman, wilson_interval


def test_wilson_interval_is_not_degenerate_at_zero_wins() -> None:
    """Losing every game is the case we most need honest error bars for."""
    low, high = wilson_interval(wins=0, games=16)

    assert low == 0.0
    assert 0.0 < high < 0.3


def test_wilson_interval_narrows_as_games_accumulate() -> None:
    """More seeded games must buy a tighter interval, or the gate is meaningless."""
    _, few = wilson_interval(wins=5, games=10)
    _, many = wilson_interval(wins=50, games=100)

    assert many < few


def test_a_hundred_even_games_meet_the_phase_one_gate() -> None:
    """The gate asks for a half-width under 0.1, which sizes the sweep at ~100."""
    low, high = wilson_interval(wins=50, games=100)

    assert (high - low) / 2 < 0.1


def test_spearman_reads_an_agreeing_and_a_reversed_ranking() -> None:
    """The fast/deep correlation the campaign logs, at both of its extremes."""
    assert spearman([1.0, 2.0, 3.0], [0.1, 0.2, 0.3]) == 1.0
    assert spearman([1.0, 2.0, 3.0], [0.3, 0.2, 0.1]) == -1.0


def test_spearman_ranks_rather_than_values() -> None:
    """A monotone but wildly nonlinear relation is still a perfect ranking."""
    assert spearman([1.0, 2.0, 3.0, 4.0], [0.01, 0.02, 0.03, 900.0]) == 1.0
