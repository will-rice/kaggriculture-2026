"""The tournament: the fit, the prior, and the pairings kept between gates."""

import subprocess
from pathlib import Path

import pytest

from kaggriculture.campaign import roster
from kaggriculture.campaign.rating import Field, ratings, standings


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


def test_kept_pairings_survive_a_round_trip(tmp_path: Path) -> None:
    """What is kept is what is read back, or a gate re-measures for nothing."""
    field = Field(games=64)
    field.record("a", "b", 0.75)
    path = tmp_path / "field.json"

    field.save(path)

    back = Field.load(path)
    assert back.rates == field.rates and back.games == 64
    assert back.results(["a", "b"]) == [("a", "b", 0.75, 64)]


def test_an_absent_file_is_an_empty_field_not_a_crash(tmp_path: Path) -> None:
    """The first gate of a campaign has nothing kept yet."""
    field = Field.load(tmp_path / "nothing.json")

    assert field.rates == {} and field.missing(["a", "b"]) == [("a", "b")]


def test_only_a_new_member_s_pairings_are_missing() -> None:
    """The point of keeping them: a champion joining costs its own row only.

    Three opponents already played each other, so a fourth joining leaves
    three pairings to measure rather than the six a fresh tournament would.
    """
    field = Field(games=64)
    for one, two in (("a", "b"), ("a", "c"), ("b", "c")):
        field.record(one, two, 0.5)

    absent = field.missing(["a", "b", "c", "champion_1"])

    assert absent == [("a", "champion_1"), ("b", "champion_1"), ("c", "champion_1")]


def test_an_opponent_that_left_the_pool_is_not_in_the_tournament() -> None:
    """Its games stay on the record; it just is not asked about.

    A pool that trims an opponent should not pay to re-measure it if that
    opponent ever comes back, and should not be rated against it meanwhile.
    """
    field = Field(games=64)
    for one, two in (("a", "b"), ("a", "gone"), ("b", "gone")):
        field.record(one, two, 0.5)

    assert field.results(["a", "b"]) == [("a", "b", 0.5, 64)]
    assert field.missing(["a", "b"]) == []
    assert "gone" in field.rates


@pytest.mark.local_data
def test_no_opponent_draws_on_randomness() -> None:
    """The assumption every kept pairing rests on.

    A pairing is kept because it is a constant: fixed files, seeded games. An
    opponent calling `random` unseeded would make a kept rate a stale sample
    instead, and the gate would rate candidates against numbers that were
    never true twice. Checked here rather than assumed, because opponents
    arrive by being harvested off the ladder and nobody reads 300KB of them.
    """
    roots = {path.parent for path in roster.TRAINING.values()}
    found = subprocess.run(
        [
            "grep",
            "-rlE",
            r"(^|[^A-Za-z_.])random\.[a-z]",
            "--include=*.py",
            *[str(root) for root in sorted(roots)],
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert found.stdout == "", f"opponents calling random: {found.stdout}"
