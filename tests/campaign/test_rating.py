"""The Bradley-Terry fit and its prior."""

import pytest

from kaggriculture.campaign.rating import ratings, standings


def test_a_stronger_agent_rates_above_a_weaker_one() -> None:
    """The whole point, on the simplest field that has one."""
    table = standings(
        [("a", "b", 0.9, 100), ("b", "c", 0.9, 100), ("a", "c", 0.99, 100)]
    )

    assert table["a"] > table["b"] > table["c"]


def test_beating_a_strong_opponent_counts_for_more_than_beating_a_weak_one() -> None:
    """Where a tournament and a mean win rate part company.

    Both `strong_wins` and `weak_wins` take half their games. One takes them
    off the field's best and the other off its worst, and a mean cannot tell
    them apart while a rating must.
    """
    table = standings(
        [
            ("best", "worst", 1.0, 100),
            ("strong_wins", "best", 0.5, 100),
            ("strong_wins", "worst", 1.0, 100),
            ("weak_wins", "best", 0.0, 100),
            ("weak_wins", "worst", 0.5, 100),
            ("strong_wins", "weak_wins", 0.5, 100),
        ]
    )

    assert table["strong_wins"] > table["weak_wins"]


def test_one_bad_matchup_does_not_sink_an_otherwise_dominant_agent() -> None:
    """The case the absolute rule this replaced turned away.

    Measured on the pool of 2026-09-06, the second-strongest published agent
    wins 78.6% of everything and loses one matchup at 0.062. A minimum over
    opponents calls that a failure; a tournament calls it second.
    """
    table = standings(
        [
            ("nemesis", "dominant", 0.94, 100),
            ("nemesis", "a", 0.0, 100),
            ("nemesis", "b", 0.0, 100),
            ("dominant", "a", 0.9, 100),
            ("dominant", "b", 0.9, 100),
            ("a", "b", 0.5, 100),
        ]
    )

    assert min(table, key=lambda n: table[n]) != "dominant"
    assert table["dominant"] > table["a"] and table["dominant"] > table["b"]


def test_an_agent_that_has_won_nothing_still_has_a_number() -> None:
    """A winless agent is minus infinity without the prior.

    264 of this campaign's first 266 programs won nothing at all, so that is
    not a corner case here: a gate cannot compare infinities and a search
    cannot climb one.
    """
    table = standings(
        [
            ("winner", "loser", 1.0, 64),
            ("winner", "other", 1.0, 64),
            ("loser", "other", 0.5, 64),
        ]
    )

    assert table["loser"] > float("-inf")
    assert table["winner"] > table["loser"]


def test_a_pairing_given_from_one_side_only_is_not_a_tournament() -> None:
    """Half a result would be fitted as though it were whole."""
    with pytest.raises(ValueError, match="no mirror"):
        ratings({"a": {"b": 0.9}, "b": {}}, {"a": {"b": 10.0}, "b": {}})
