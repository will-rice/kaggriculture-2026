"""The roster is names in, paths out; nothing else resolves."""

from pathlib import Path

import pytest

from kaggriculture.campaign import roster


def test_every_roster_entry_exists_on_disk() -> None:
    """A name the harness accepts must resolve to a file that is there."""
    for name in roster.names():
        assert roster.path(name).exists(), name


def test_training_and_held_out_do_not_overlap() -> None:
    """The held-out set is only held out while nothing trains against it."""
    assert not set(roster.TRAINING) & set(roster.HELD_OUT)


def test_an_unknown_name_is_an_error_not_a_path() -> None:
    """A traversal string is a missing key, never a filesystem lookup."""
    with pytest.raises(KeyError):
        roster.path("../../../etc/passwd")


def test_no_roster_path_is_ever_relative_or_inside_the_repo() -> None:
    """Opponents live outside the repo, so a sandbox cannot read them from it."""
    for name in roster.names():
        assert roster.path(name).is_absolute()
        assert not str(roster.path(name)).startswith(str(Path.cwd()))
