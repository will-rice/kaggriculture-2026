"""Tests for win-rate reporting and its confidence intervals."""

from kaggriculture.report import Standing, standings, wilson_interval
from kaggriculture.result import Result


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


def test_standings_separate_wins_losses_ties_and_errors() -> None:
    """An errored episode is not a loss; counting it as one hides broken agents."""
    results = [
        Result(task_id="meta-build:1", agent_id="a", score=1.0),
        Result(task_id="meta-build:2", agent_id="a", score=0.0),
        Result(task_id="meta-build:3", agent_id="a", score=0.5),
        Result(task_id="meta-build:4", agent_id="a", score=0.0, error="boom"),
    ]

    (standing,) = standings(results)

    assert isinstance(standing, Standing)
    assert standing.opponent == "meta-build"
    assert (standing.wins, standing.losses, standing.ties) == (1, 1, 1)
    assert standing.errors == 1
    assert standing.games == 3


def test_standings_average_the_banks_of_scored_games() -> None:
    """Bank is no longer the acceptance metric, but it is still the diagnostic."""
    results = [
        Result(task_id="tape:1", agent_id="a", score=0.0, scores=[40000.0, 170000.0]),
        Result(task_id="tape:2", agent_id="a", score=0.0, scores=[60000.0, 150000.0]),
        Result(task_id="tape:3", agent_id="a", score=0.0, error="boom"),
    ]

    (standing,) = standings(results)

    assert standing.bank == 50000.0
    assert standing.opponent_bank == 160000.0
