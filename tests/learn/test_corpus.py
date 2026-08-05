"""Tests for selecting demonstrations from the replay corpus, against real archives."""

import pytest

from kaggriculture.learn.corpus import CORPUS

ARCHIVE = CORPUS / "kaggriculture-episodes-2026-08-03.zip"

pytestmark = pytest.mark.skipif(
    not ARCHIVE.exists(), reason="replay corpus not present on this machine"
)


def test_the_manifest_carries_a_rating_for_every_episode() -> None:
    """Ratings are what make selection possible without decoding 21GB."""
    from kaggriculture.learn.corpus import read_manifest

    rows = read_manifest(ARCHIVE)

    assert len(rows) == 787
    assert all(row.episode_id > 0 for row in rows)
    assert all(row.min_score <= row.avg_score for row in rows)
    assert max(row.avg_score for row in rows) > 2000


def test_selection_excludes_the_recorded_tape_by_rating() -> None:
    """The tape rates about 1720; a floor above that removes it and its clones.

    This is what replaces a per-team cap. Three-quarters of the field replays
    one recording, and every copy of it rates where the original does.
    """
    from kaggriculture.learn.corpus import select

    chosen = select([ARCHIVE], min_rating=2500.0, per_archive=1000)

    assert chosen
    assert all(sample.rating >= 2500.0 for sample in chosen)


def test_both_seats_of_a_strong_episode_are_taken() -> None:
    """min_score gates on the weaker seat, so passing it means both played well."""
    from kaggriculture.learn.corpus import select

    chosen = select([ARCHIVE], min_rating=2500.0, per_archive=1000)
    seats = {(sample.name, sample.seat) for sample in chosen}
    names = {name for name, _ in seats}

    assert len(seats) == 2 * len(names)


def test_the_per_archive_cap_bounds_the_dataset() -> None:
    """Five archives of unbounded episodes would not fit in memory as tensors."""
    from kaggriculture.learn.corpus import select

    chosen = select([ARCHIVE], min_rating=0.0, per_archive=10)

    assert len(chosen) == 20


def test_the_holdout_split_shares_no_episode_with_training() -> None:
    """An episode in both halves makes validation accuracy a memorisation score."""
    from kaggriculture.learn.corpus import select, split

    train, held = split(select([ARCHIVE], min_rating=0.0, per_archive=50), holdout=0.2)

    assert {sample.name for sample in train} & {sample.name for sample in held} == set()
    assert held
