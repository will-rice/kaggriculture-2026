"""Tests for the daily replay fetcher, which runs unattended from cron."""

from pathlib import Path

from kaggriculture.scripts import fetch_episodes


def test_archive_path_matches_the_downloaded_name(tmp_path: Path) -> None:
    """Skipping already-downloaded days depends on predicting the archive name."""
    slug = "kaggriculture-episodes-2026-08-02"

    assert fetch_episodes.archive_path(tmp_path, slug).name == f"{slug}.zip"


def test_already_downloaded_days_are_skipped(tmp_path: Path) -> None:
    """A repeat run must not re-download half a gigabyte per day."""
    present = "kaggriculture-episodes-2026-08-02"
    fetch_episodes.archive_path(tmp_path, present).touch()
    published = [present, "kaggriculture-episodes-2026-08-03"]

    missing = [s for s in published if not fetch_episodes.archive_path(tmp_path, s).exists()]

    assert missing == ["kaggriculture-episodes-2026-08-03"]
