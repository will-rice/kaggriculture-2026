"""Fixed all-opponent fitness and frontier-strength contracts."""

import math

import pytest

from kaggriculture.search.fitness import (
    StrengthWeights,
    score_fitness,
    strength_weights,
)
from kaggriculture.search.frontier import FrontierReport, FrontierRow


def _row(name: str) -> FrontierRow:
    return FrontierRow(
        name=name,
        games=0,
        field_win_points=0.0,
        worst_matchup_win_points=0.0,
        paired_margin=0.0,
        failures=(),
        runtime_seconds=0.0,
        matchups={},
        matchup_results=(),
    )


def test_fitness_is_seventy_percent_weighted_mean_and_thirty_percent_worst() -> None:
    """Changing either objective coefficient must change this literal result."""
    rates = {"frontier": 0.40, "strong": 0.60, "weak": 1.00}
    weights = StrengthWeights({"frontier": 4, "strong": 3, "weak": 1})

    result = score_fitness(rates, weights)

    mean = (4 * 0.40 + 3 * 0.60 + 1 * 1.00) / 8
    assert result.value == pytest.approx(0.70 * mean + 0.30 * 0.40)
    assert result.weighted_mean == pytest.approx(mean)
    assert result.worst == pytest.approx(0.40)
    assert result.eligible is True


def test_illegal_or_failed_candidate_is_ineligible() -> None:
    """A failure cannot be compensated for by a perfect matchup result."""
    result = score_fitness(
        {"frontier": 1.0}, StrengthWeights({"frontier": 4}), failures=1
    )

    assert result.eligible is False
    assert result.value == -math.inf


def test_fitness_requires_the_exact_weighted_league() -> None:
    """Missing or surplus matchups must not silently change the objective."""
    weights = StrengthWeights({"frontier": 4, "strong": 3})

    with pytest.raises(ValueError, match="matchup names"):
        score_fitness({"frontier": 0.5}, weights)
    with pytest.raises(ValueError, match="matchup names"):
        score_fitness({"frontier": 0.5, "strong": 0.5, "extra": 1.0}, weights)


def test_strength_weights_follow_ranked_quartiles_and_protect_boatlee_v14() -> None:
    """The measured order fixes 4/3/2/1 tiers and both v14 gates stay strong."""
    names = (
        "frontier",
        "second",
        "third",
        "boatlee_v14_public",
        "middle",
        "sixth",
        "seventh",
        "eighth",
        "boatlee_v14_current",
        "tenth",
        "weakest",
    )
    report = FrontierReport(
        engine="1.32.7",
        seeds=(1,),
        frontier_name="frontier",
        failures=(),
        runtime_seconds=0.0,
        rows=tuple(_row(name) for name in names),
    )

    result = strength_weights(report)

    assert result.as_dict() == {
        "frontier": 4,
        "second": 4,
        "third": 4,
        "boatlee_v14_public": 3,
        "middle": 3,
        "sixth": 3,
        "seventh": 2,
        "eighth": 2,
        "boatlee_v14_current": 3,
        "tenth": 1,
        "weakest": 1,
    }
