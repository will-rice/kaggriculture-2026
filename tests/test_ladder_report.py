"""The ladder page's prose, where it names days.

Every number on that page is queried at render time. The sentences around them
were written once, on the argument that a direction outlasts a value. On
2026-09-09 a direction reversed and the sentence did not: the page claimed the
strong agents ran poor "through the first ten days" and crossed over "around
day twelve", while the table beneath it showed them ahead on cash from day zero
to day eight and behind only at day ten and day fourteen.
"""

import sqlite3

from kaggriculture.campaign import dataset
from kaggriculture.scripts.report import TEAM, _bank_story, _ours


def test_the_claim_names_the_days_the_data_names() -> None:
    """The exact shape that went stale: a late, contiguous trailing run."""
    story = _bank_story(list(range(9, 17)), 16)

    assert "From day 9 to day 16" in story
    assert "crosses over after day 16" in story
    # The sentence it replaced, which the same data contradicts.
    assert "first ten days" not in story
    assert "day twelve" not in story


def test_a_single_trailing_day_is_not_described_as_a_range() -> None:
    """A range template would say from day 14 to day 14, which nobody writes."""
    story = _bank_story([14], 14)

    assert "On day 14 alone" in story
    assert "From day" not in story


def test_a_broken_run_is_listed_rather_than_smoothed_into_a_range() -> None:
    """Trailing on days 10 and 14 but not between is not "day 10 to day 14".

    This is the corpus's actual shape when read at the table's marks rather
    than at every day, and stating it as a range would assert six days of
    trailing that were never measured.
    """
    story = _bank_story([10, 14], 14)

    assert "On days 10 and 14" in story
    assert "From day" not in story


def test_never_trailing_is_a_sentence_and_not_a_broken_range() -> None:
    """The top may simply lead throughout; the page still has to read."""
    story = _bank_story([], None)

    assert "never behind on cash" in story
    assert "day None" not in story
    assert "<em>" not in story


def corpus(games: int, won: int, day: str = "2026-09-08") -> sqlite3.Connection:
    """A dataset holding only what `_ours` reads: our episodes and the teams."""
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        "CREATE TABLE episodes (episode TEXT, played TEXT, team_0 TEXT, "
        "team_1 TEXT, winner INT);"
        "CREATE TABLE teams (team TEXT, rating REAL, games INT);"
    )
    connection.executemany(
        "INSERT INTO episodes VALUES (?, ?, ?, ?, ?)",
        [(str(n), day, TEAM, "someone", 0 if n < won else 1) for n in range(games)],
    )
    connection.commit()
    return connection


def test_we_are_shown_below_the_threshold_with_what_is_missing() -> None:
    """The interesting part is watching the measurement fill.

    Our ladder games enter the public corpus like everyone else's, so the
    campaign can be rated in the same fit as the field -- the one reading of
    our own progress that does not move when the field does. Until it has
    `dataset.LEAST` games it is not rated at all, and hiding it until then
    hides the only axis that answers the ladder's noise.
    """
    short = dataset.LEAST - 9
    facts = _ours(corpus(short, won=3), "2026-08-30", ladder=[])

    assert "not yet rated" in facts["ours_standing"]
    assert f"{short} games" in facts["ours_standing"]
    assert "9 short" in facts["ours_standing"]
    assert facts["ours_games"] == str(short)
    assert "2026-09-08" in facts["ours_days"]


def test_once_rated_we_are_placed_against_the_field_in_elo() -> None:
    """A rating is only meaningful beside the field it was fitted with.

    So the standing carries the place as well as the number, and the number
    in Elo points as well as log-odds -- 173.7 to the log-odd -- because a bare
    log-odds figure is unreadable as a strength.
    """
    ladder = [
        ("HowardLeeTW", 2.33, 0, 0, 0),
        (TEAM, 0.5, 0, 0, 0),
        ("kwa", -0.3, 0, 0, 0),
    ]

    facts = _ours(corpus(dataset.LEAST, won=dataset.LEAST // 2), "2026-08-30", ladder)

    assert "+0.500" in facts["ours_standing"]
    assert "+87 Elo" in facts["ours_standing"]
    assert "2 of 3" in facts["ours_standing"]
    assert "not yet rated" not in facts["ours_standing"]


def test_no_games_in_the_window_is_a_row_and_not_an_empty_table() -> None:
    """A fresh window, or a fetch that has not caught up, still has to render."""
    facts = _ours(corpus(0, won=0), "2026-08-30", ladder=[])

    assert "no games in the window yet" in facts["ours_days"]
    assert facts["ours_rate"] == "\u2014"
