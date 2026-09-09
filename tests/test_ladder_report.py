"""The ladder page's prose, where it names days.

Every number on that page is queried at render time. The sentences around them
were written once, on the argument that a direction outlasts a value. On
2026-09-09 a direction reversed and the sentence did not: the page claimed the
strong agents ran poor "through the first ten days" and crossed over "around
day twelve", while the table beneath it showed them ahead on cash from day zero
to day eight and behind only at day ten and day fourteen.
"""

from kaggriculture.scripts.report import _bank_story


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
