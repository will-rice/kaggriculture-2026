"""Days and games built by hand, and a database to put them in."""

import urllib.error

import pytest

from kaggriculture.campaign import dataset, games, harness


def measured(**held: float) -> dict[str, float]:
    """A full measures dict: what is named, and zero for everything else."""
    return dict.fromkeys(dataset.COLUMNS, 0.0) | held


def day(number: int, ours: float, theirs: float, **held: float) -> harness.Day:
    """One day, with something in every field and the banks where asked."""
    return harness.Day(
        day=number,
        ours_bank=ours,
        theirs_bank=theirs,
        ours=measured(bank=ours, **held),
        theirs=measured(bank=theirs),
        ours_plants={"WHEAT": 4},
        theirs_plants={"MELON": 2},
        ours_animals={},
        theirs_animals={"COW": 1},
        ours_weeds=0,
        theirs_weeds=3,
        ours_seeds={"WHEAT": 5},
        ours_shed={"WHEAT": 12},
        theirs_shed={"EGG": 3},
        ours_hands=2,
        theirs_hands=1,
        prices={"WHEAT": 25},
    )


def game(days: list[harness.Day], seat: int = 0, opponent: str = "v54") -> harness.Game:
    """One played game carrying those days."""
    final = days[-1]
    return harness.Game(
        opponent=opponent,
        seed=101,
        seat=seat,
        ours=final.ours_bank,
        theirs=final.theirs_bank,
        worst_step_seconds=0.0,
        days=days,
    )


def running() -> bool:
    """Whether ClickHouse is answering on `GAMES_URL`."""
    try:
        games.query("SELECT 1")
    except (urllib.error.URLError, OSError, RuntimeError):
        return False
    return True


live = pytest.mark.skipif(not running(), reason="no ClickHouse on GAMES_URL")
