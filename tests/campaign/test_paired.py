"""Both sides of one game, measured at the same moment, from the same tape.

The walk is the part that can be quietly wrong: an index off by one moves
every quantity a day, and a claim measured against it still comes back with a
percentage and a support count that read exactly like evidence. Two versions
of this measurement had no day 29 at all and said nothing about it.
"""

import json
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from kaggriculture.campaign import paired, strategy, tapes

# What a recorded season holds: thirty days of twenty-four hours, and no step
# after the last one. The last day therefore closes on the final step, which
# no action follows -- which is the whole reason `sides` needs its last line.
STEPS = 720
# Seat 0 pulls ahead and stays there, so the winner is never in doubt and a
# test can talk about "the winner" without restating why.
AHEAD = 10.0
BEHIND = 1.0


def _farm(money: float, planted: int, hands: int) -> dict:
    """One side's public farm at a moment: money, growing tiles, and hands."""
    return {
        "money": money,
        "tiles": [[{"kind": "PLANT", "yield_units": 0} for _ in range(planted)]],
        "hands": [{} for _ in range(hands)],
        "unlocked_quadrants": [0],
    }


def _game(
    state: Callable[[int, int], tuple[float, int, int]],
    orders: Callable[[int, int], Sequence[str]] = lambda index, seat: (),
) -> tapes.Episode:
    """A whole recorded season built from two functions of step and seat.

    Args:
        state: What each side held at each step, as ``(money, planted,
            hands)``.
        orders: The market orders each side submitted at each step.

    Returns:
        An episode shaped like the archive's, in the one respect that
        matters: the action at step ``t`` sits beside the state it produced,
        not the state it was chosen from.
    """
    steps = []
    for index in range(STEPS):
        farms = [_farm(*state(index, seat)) for seat in (0, 1)]
        steps.append(
            [
                {
                    "observation": {
                        "farms": farms,
                        "private": {"shed": {}, "seeds": {}},
                    },
                    "action": {"market": [[name] for name in orders(index, seat)]},
                    # A status the engine itself assigns: anything else means a
                    # seat crashed and the recorded action is not what was
                    # applied, and such a game is filtered out before it is
                    # read.
                    "status": "DONE",
                }
                for seat in (0, 1)
            ]
        )
    return tapes.Episode(seed=1, engine_version="1.32.7", steps=steps)


def _climbing(index: int, seat: int) -> tuple[float, int, int]:
    """Seat 0's bank is ten times the step; seat 1 never moves off one."""
    return (AHEAD * index if seat == 0 else BEHIND, 0, 0)


def test_a_day_closes_on_the_state_its_last_decision_was_read_from() -> None:
    """The action at step ``t`` was chosen looking at step ``t-1``.

    Measured off by one, "the winner plants on day two" becomes a claim about
    a state the player had not seen yet -- and an earlier version of this walk
    produced exactly that, a confident rule about harvesting bare ground.
    """
    game = paired.sides(_game(_climbing))

    assert game is not None
    # Day zero's last decision was made looking at step 23. Step 24 is what
    # that decision produced, and belongs to day one.
    assert game[0]["winner"]["bank"] == AHEAD * 23
    assert game[1]["winner"]["bank"] == AHEAD * 47


def test_the_last_day_is_measured_from_the_final_state() -> None:
    """No action follows the final step, so the day-by-day walk cannot reach it.

    The last day is where a claim about liquidating, or about still holding
    growing tiles at the close, has to be decided. Two earlier versions of
    this measurement silently had no day 29 and reported nothing missing:
    every claim about it came back with zero support, which reads like a
    quantity nobody asked about rather than a day nobody measured.
    """

    # Seat 0 is behind all season and takes it on the very last step.
    def late(index: int, seat: int) -> tuple[float, int, int]:
        if seat == 1:
            return (500.0, 0, 0)
        return (900.0 if index == STEPS - 1 else 100.0, 0, 0)

    game = paired.sides(_game(late))

    assert game is not None
    assert set(game) == set(range(paired.DAYS))
    assert game[paired.DAYS - 1]["winner"]["bank"] == 900.0


def test_a_draw_has_no_winning_side_to_learn_from() -> None:
    """Every claim is about what the winner did, so a draw is not evidence.

    Scoring it either way would put half the corpus behind whichever side the
    tie-break happened to pick.
    """
    assert paired.sides(_game(lambda index, seat: (100.0, 0, 0))) is None


def test_orders_are_counted_through_the_days_last_decision() -> None:
    """An order is an action, so it is counted up to and including the close.

    The state a day closes on is read at step ``t-1``, but the order submitted
    at step ``t`` *is* that day's last decision -- so the two are deliberately
    not cut at the same index, and a claim about hiring covers the whole day
    rather than all but its final hour.
    """
    game = paired.sides(
        _game(_climbing, orders=lambda index, seat: ("SELL",) if seat == 0 else ())
    )

    assert game is not None
    assert game[0]["winner"]["sells"] == 24
    assert game[1]["winner"]["sells"] == 48
    assert game[0]["loser"]["sells"] == 0


def test_a_quantity_no_claim_asks_about_is_still_measured() -> None:
    """One walk serves every claim, which is what lets the store grow.

    Measuring per claim would make the daily job cost one pass per claim; the
    walk is what costs, and a claim is a lookup once a game has been read.
    """
    game = paired.sides(_game(lambda index, seat: (AHEAD * index + seat, 4, 2)))

    assert game is not None
    assert set(game[0]["winner"]) == strategy.QUANTITIES
    assert game[0]["winner"]["planted"] == 4
    assert game[0]["winner"]["hands"] == 2


def _archive(path: Path, episodes: list[tapes.Episode]) -> Path:
    """Write ``episodes`` as a daily archive, shaped as the ladder's are."""
    with zipfile.ZipFile(path, "w") as opened:
        for number, episode in enumerate(episodes):
            opened.writestr(
                f"{number}.json",
                json.dumps(
                    {
                        "info": {"seed": episode.seed},
                        "module_version": episode.engine_version,
                        "steps": episode.steps,
                    }
                ),
            )
    return path


def test_one_walk_of_an_archive_answers_every_claim_at_once(tmp_path: Path) -> None:
    """The walk is what costs; a claim is a lookup once a game has been read.

    Measured one claim at a time, a store of fifty would cost fifty passes
    over the corpus and the daily job would grow with the store rather than
    with the archive.
    """
    written = _archive(
        tmp_path / "kaggriculture-episodes-2026-09-01.zip",
        [_game(_climbing), _game(_climbing)],
    )
    forms = {
        "richer": strategy.Form(quantity="bank", day=29, winner_leads=True),
        "poorer": strategy.Form(quantity="bank", day=29, winner_leads=False),
        "quiet": strategy.Form(quantity="hires", day=29, winner_leads=True),
    }

    counted = paired.tally(written, forms)

    assert counted["richer"] == (2, 2)
    assert counted["poorer"] == (0, 2)
    # Neither side hired, so both games are silent on it rather than counted
    # as agreement -- which is how a claim about a quantity nobody moved would
    # otherwise reach 100%.
    assert counted["quiet"] == (0, 0)


def test_the_whole_corpus_is_read_and_the_counts_add_up(tmp_path: Path) -> None:
    """A day's archive is the unit of work, and the totals are their sum.

    Divided over processes, so what has to hold is that no game is counted
    twice and none is dropped: the support is every game in every archive.
    """
    corpus = [
        _archive(
            tmp_path / "kaggriculture-episodes-2026-09-01.zip", [_game(_climbing)]
        ),
        _archive(
            tmp_path / "kaggriculture-episodes-2026-09-02.zip",
            [_game(_climbing), _game(_climbing)],
        ),
    ]
    store = strategy.Strategies(tmp_path / "strategies.jsonl")
    claim = store.propose(
        strategy.Form(quantity="bank", day=paired.DAYS - 1, winner_leads=True),
        "the winner ends the season with more money",
        [],
    )

    measured = paired.measure(store, corpus, workers=2)

    assert measured[claim.id] == (3, 1.0)
    assert store.get(claim.id).support == 3


@pytest.mark.local_data
def test_the_winner_ends_richer_in_every_recorded_game(tmp_path: Path) -> None:
    """The one claim that cannot be false, checked against real tapes.

    The winner is decided on the final bank and day 29 is measured from the
    final state, so this must hold in every recorded game -- which makes it a
    check on the walk rather than on the ladder. Anything short of every game
    means the day the claim is measured on is not the day the winner was read
    from.
    """
    del tmp_path
    form = strategy.Form(quantity="bank", day=paired.DAYS - 1, winner_leads=True)

    for index in range(12):
        game = paired.sides(tapes.qualifying(index))
        if game is None:
            continue
        day = game[paired.DAYS - 1]
        assert form.holds(day["winner"]["bank"], day["loser"]["bank"]) is True
