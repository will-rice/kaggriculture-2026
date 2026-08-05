"""Tests for the dataset build entry point."""

from pathlib import Path

from kaggriculture.learn.scripts.build import clear_shards, write_dataset


def test_writing_a_dataset_clears_the_previous_build_first(tmp_path: Path) -> None:
    """Clearing must be part of writing, not a step a caller can forget.

    Testing ``clear_shards`` alone proves the function works and says nothing
    about whether the build path calls it -- which is the failure that actually
    happens. Writing an empty dataset exercises the whole path.
    """
    for name in ("train.npz", "train-001.npz", "holdout.npz"):
        (tmp_path / name).touch()

    write_dataset(tmp_path, [], [], stride=4)

    assert list(tmp_path.glob("*.npz")) == []


def test_a_rebuild_removes_the_previous_build_s_shards(tmp_path: Path) -> None:
    """A shorter rebuild would otherwise train on the tail of the longer one.

    The builder writes ``train.npz``, ``train-001.npz`` and onwards until it
    runs out of rows, and never deletes. Raise the rating floor or lengthen the
    stride and the next build writes fewer shards, leaving the old tail behind
    under names that are indistinguishable from real ones. Those files carry the
    same shapes, so they concatenate without complaint and training silently
    mixes two datasets -- including rows the new selection deliberately excluded.
    """
    for name in ("train.npz", "train-001.npz", "train-002.npz", "holdout.npz"):
        (tmp_path / name).touch()
    keep = tmp_path / "notes.txt"
    keep.touch()

    clear_shards(tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == ["notes.txt"]


def test_clearing_an_empty_directory_is_not_an_error(tmp_path: Path) -> None:
    """The first build on a fresh machine has nothing to clear."""
    clear_shards(tmp_path)

    assert list(tmp_path.iterdir()) == []
