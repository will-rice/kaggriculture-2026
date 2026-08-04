"""Tests for win-rate reporting and its confidence intervals."""

import pytest

from kaggriculture.report import (
    Standing,
    format_league,
    format_standing,
    league_win_rate,
    standings,
    wilson_interval,
)
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


def standing(opponent: str = "tape", win_rate: float = 0.4) -> Standing:
    """Return a standing with every field populated, for formatting tests."""
    return Standing(
        opponent=opponent,
        games=100,
        wins=40,
        losses=60,
        ties=0,
        errors=0,
        win_rate=win_rate,
        low=0.31,
        high=0.50,
        bank=118672.0,
        opponent_bank=140585.0,
    )


def test_half_width_is_half_the_interval() -> None:
    """The Phase 1 gate is stated in terms of this quantity and nothing tested it."""
    assert standing().half_width == pytest.approx((0.50 - 0.31) / 2)


def test_a_formatted_standing_carries_every_number_a_reader_needs() -> None:
    """This line is the entire output of an evaluation; it had no coverage at all."""
    line = format_standing(standing())

    assert "tape" in line
    assert "0.400" in line
    assert "[0.310, 0.500]" in line
    assert "40W 60L 0T" in line
    assert "118672" in line and "140585" in line


def test_errors_appear_in_the_formatted_line_only_when_they_happen() -> None:
    """A silent error count would hide a broken agent behind a plausible win rate."""
    clean = format_standing(standing())
    broken = format_standing(standing().model_copy(update={"errors": 3}))

    assert "E)" not in clean
    assert "3E" in broken


def test_league_win_rate_weights_each_opponent_equally() -> None:
    """The acceptance rule is stated against this number, so it must be computable."""
    table = [standing("a", 0.0), standing("b", 1.0), standing("c", 1.0)]

    assert league_win_rate(table) == pytest.approx(2 / 3)


def test_league_win_rate_of_nothing_is_zero_rather_than_an_error() -> None:
    """An evaluation whose episodes all errored still has to report something."""
    assert league_win_rate([]) == 0.0


def test_the_league_line_reports_how_much_evidence_it_rests_on() -> None:
    """A rate without a game count invites the mistake this whole module prevents."""
    line = format_league([standing("a", 0.0), standing("b", 1.0)])

    assert "0.500" in line
    assert "2 opponents" in line
    assert "200 games" in line
