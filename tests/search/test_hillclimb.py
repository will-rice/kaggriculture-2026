"""The paired comparison the hill-climb's accept/reject decision rests on."""

import json
from pathlib import Path

import pytest

from kaggriculture.search.scripts.hillclimb import load_league, paired_difference


def test_identical_outcomes_produce_a_zero_mean_difference_and_are_rejected() -> None:
    """A candidate identical to the incumbent must never be accepted.

    This is the property the whole acceptance rule (``mean_difference > 2 *
    standard_error``) rests on: with no real difference between the two
    routes, the paired difference on every game is exactly zero, so the mean
    is exactly zero and the standard error is exactly zero too. Zero is not
    greater than zero, so the candidate is rejected -- as it must be, since it
    is not actually a different route.
    """
    outcomes = [1.0, 0.5, 0.0, 1.0, 0.5, 0.5, 0.0, 1.0]

    mean_difference, standard_error = paired_difference(outcomes, outcomes)

    assert mean_difference == 0.0
    assert standard_error == 0.0
    assert not (mean_difference > 2 * standard_error)


def test_a_consistent_advantage_produces_a_positive_mean_and_is_accepted() -> None:
    """A candidate that wins every paired game must clear the threshold."""
    candidate_outcomes = [1.0] * 8
    incumbent_outcomes = [0.0] * 8

    mean_difference, standard_error = paired_difference(
        candidate_outcomes, incumbent_outcomes
    )

    assert mean_difference == 1.0
    assert standard_error == 0.0
    assert mean_difference > 2 * standard_error


def test_standard_error_is_taken_from_the_observed_spread_not_an_assumed_variance() -> (
    None
):
    """The old formula assumed variance 0.25; the new one measures it.

    Constructing a paired-difference sample whose true variance is not 0.25
    and checking the standard error against the textbook sample formula
    directly is what pins that ``paired_difference`` reads its spread from the
    data rather than from a hard-coded constant.
    """
    candidate_outcomes = [1.0, 1.0, 1.0, 0.0]
    incumbent_outcomes = [0.0, 0.0, 0.0, 0.0]
    diffs = [1.0, 1.0, 1.0, 0.0]
    mean = sum(diffs) / len(diffs)
    variance = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
    expected_se = variance**0.5 / len(diffs) ** 0.5

    mean_difference, standard_error = paired_difference(
        candidate_outcomes, incumbent_outcomes
    )

    assert mean_difference == pytest.approx(mean)
    assert standard_error == pytest.approx(expected_se)
    assert standard_error != pytest.approx(2 * (0.25 / len(diffs)) ** 0.5)


def test_load_league_accepts_routes_and_agent_paths(tmp_path: Path) -> None:
    """A league directory must yield both ``.json`` routes and ``.py`` agents.

    Globbing only ``*.json`` makes policy opponents unreachable from this
    script even though ``arena.Opponent`` already supports them -- a league of
    tapes alone cannot contain an opponent that reacts to the candidate.
    """
    route = [{"farmer": ["PASS"], "hands": [], "market": []}] * 720
    (tmp_path / "tape.json").write_text(json.dumps(route))
    (tmp_path / "policy.py").write_text("# not executed by this test\n")

    league = load_league(tmp_path)

    assert set(league) == {"tape", "policy"}
    assert isinstance(league["tape"], list)
    assert league["policy"] == str(tmp_path / "policy.py")
