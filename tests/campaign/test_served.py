"""The floor: legal, and it banks more than the 3000 it started with."""

from pathlib import Path

import pytest

from kaggriculture.campaign import harness, validate

SERVED = Path("src/kaggriculture/served/main.py")


def test_the_skeleton_validates() -> None:
    """The skeleton passes every static and dynamic check in `validate`."""
    assert validate.validate(SERVED).status == "ok"


@pytest.mark.local_data
def test_the_skeleton_plants_something() -> None:
    """It plays a full port game and a short reference run without error."""
    games = harness.play(SERVED, ["v54"], [1], workers=2)
    assert all(game.ours >= 0 for game in games)
    report = harness.check(SERVED, steps=60)
    assert report.error is None


def test_the_skeleton_banks_a_profit() -> None:
    """A full reference game leaves more in the bank than the starting 3000."""
    report = harness.check(SERVED)
    assert report.error is None
    assert report.bank > 3000
