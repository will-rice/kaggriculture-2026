"""What the loader reports about the games this lineage really lost.

`refresh` itself is not here: it is a Kaggle listing, thirty downloads and a
parse, and the parse is `dataset._rows`, which `test_dataset` covers against
real replays. What is here is the summary the loop logs hourly, because a figure
that is quietly wrong is worse than one that is missing -- it is read against
promotions for the rest of the campaign.
"""

import pytest

from kaggriculture.campaign import games, losses

# One day of one game, as `dataset._rows` writes it: the episode, the seat, the
# day, the team name, then the measures. Only the bank is read here, and it is
# the first of them.
COLUMNS = len(games.written("days")) - 1


def day(episode: str, seat: int, number: int, team: str, bank: float) -> list[object]:
    """One row of `days`, with every measure but the bank left at zero."""
    rest = COLUMNS - 5
    return [episode, seat, number, team, bank, *([0] * rest)]


def hold(database: str, gaps: dict[int, float]) -> None:
    """One two-seat game whose bank difference is ``gaps`` on those days."""
    batch = games.Batch("live")
    for number, gap in gaps.items():
        batch.add("days", day("e1", 0, number, "ours", 10_000 + gap))
        batch.add("days", day("e1", 1, number, "opponent", 10_000))
    batch.send(database)


def test_the_gap_is_reported_at_the_days_it_is_read_at(
    scratch: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Day 10, the close, and the stretch between -- each from its own days.

    Three figures out of one query, which is three chances to read the wrong
    rows into the wrong name. A single game whose gap differs on every day
    catches a summary that averages the season into all three.
    """
    monkeypatch.setattr(games, "DATABASE", scratch)
    hold(scratch, {10: 500.0, 20: -100.0, 29: -900.0})

    where = losses.deficit()

    assert where["losses/games"] == 1
    assert where["losses/gap_day10"] == 500
    assert where["losses/gap_final"] == -900
    # Days 20 to 29 is the mean over that stretch, so it is neither endpoint.
    assert where["losses/gap_days20_29"] == -500


def test_an_empty_record_reports_nothing_rather_than_nan(
    scratch: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A database with no losses in it yet has nothing to say about them.

    An average over no rows is `nan`, and the server returns that as a row like
    any other. Read as "did it answer", a campaign whose live partition is
    simply still empty logs four `nan`s an hour, and a `nan` in wandb is a gap
    in the series that looks like a run that stopped.
    """
    monkeypatch.setattr(games, "DATABASE", scratch)

    assert losses.deficit() == {}


def test_a_loss_that_was_ahead_early_reports_a_positive_day10(
    scratch: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sign has to survive, because it is the whole of the finding.

    These are games this program lost, so every instinct and every aggregate
    tends negative; the reason to look at all is that day 10 came back *ahead*.
    A summary that clamped or flipped it would report the opposite of what the
    season did and read as perfectly reasonable.
    """
    monkeypatch.setattr(games, "DATABASE", scratch)
    hold(scratch, {10: 87.0, 29: -804.0})

    where = losses.deficit()

    assert where["losses/gap_day10"] > 0
    assert where["losses/gap_final"] < 0
