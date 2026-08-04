"""Tests for the daily replay fetcher, which runs unattended from cron."""

import os
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_episodes.py"


def load_script() -> ModuleType:
    """Import the cron script by path, since it lives outside the package."""
    module = ModuleType("fetch_episodes")
    SourceFileLoader("fetch_episodes", str(SCRIPT)).exec_module(module)
    return module


def test_script_is_executable_with_a_uv_shebang() -> None:
    """Cron invokes the file directly, so the shebang and mode bit carry the job."""
    assert os.access(SCRIPT, os.X_OK)
    assert SCRIPT.read_text().startswith("#!/usr/bin/env -S uv run --script")


def test_archive_path_matches_the_downloaded_name(tmp_path: Path) -> None:
    """Skipping already-downloaded days depends on predicting the archive name."""
    slug = "kaggriculture-episodes-2026-08-02"

    assert load_script().archive_path(tmp_path, slug).name == f"{slug}.zip"


def test_already_downloaded_days_are_skipped(tmp_path: Path) -> None:
    """A repeat run must not re-download half a gigabyte per day."""
    archive_path = load_script().archive_path
    present = "kaggriculture-episodes-2026-08-02"
    archive_path(tmp_path, present).touch()
    published = [present, "kaggriculture-episodes-2026-08-03"]

    missing = [slug for slug in published if not archive_path(tmp_path, slug).exists()]

    assert missing == ["kaggriculture-episodes-2026-08-03"]
