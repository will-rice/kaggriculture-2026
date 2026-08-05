"""Tests for indexing the replay corpus, run against the real archives."""

import zipfile

import pytest

from kaggriculture.learn.corpus import CORPUS, EpisodeRecord, index_archive

ARCHIVE = CORPUS / "kaggriculture-episodes-2026-08-03.zip"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not ARCHIVE.exists(), reason="replay corpus not present on this machine"
    ),
]


def test_index_reads_every_episode_in_an_archive() -> None:
    """Every JSON member must appear in the index; a dropped episode is silent."""
    with zipfile.ZipFile(ARCHIVE) as bundle:
        expected = sum(1 for name in bundle.namelist() if name.endswith(".json"))

    records = index_archive(ARCHIVE)

    assert len(records) == expected == 787
    assert all(isinstance(record, EpisodeRecord) for record in records)


def test_each_record_carries_the_fields_selection_needs() -> None:
    """Rewards decide quality and team names decide diversity; both must be real."""
    record = index_archive(ARCHIVE)[0]

    assert record.episode_id > 0
    assert len(record.rewards) == 2
    assert all(reward > 0 for reward in record.rewards)
    assert len(record.teams) == 2
    assert all(team for team in record.teams)


def test_indexing_does_not_decode_the_steps() -> None:
    """Decoding 27MB of steps per episode would make indexing take hours.

    The guard is memory: holding one episode's steps is ~200MB of Python
    objects, so an index that decoded them could not hold 787 of them at once.
    This asserts the record carries no step data at all.
    """
    record = index_archive(ARCHIVE)[0]

    assert not hasattr(record, "steps")
    assert set(record.model_dump()) == {
        "archive",
        "name",
        "episode_id",
        "rewards",
        "teams",
    }
