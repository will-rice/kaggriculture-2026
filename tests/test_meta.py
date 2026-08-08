"""Tests for the meta report.

The report exists because the caches had no reader, and the specific miss it is
meant to prevent is a newer version of an agent we vendor sitting in the cache
unnoticed for a day. So the central test builds exactly that situation -- a
vendored kernel plus a later one by the same author -- and asserts the report
names it. A report that merely runs would not have helped.

The caches are built here rather than mocked: they are two small SQLite tables,
and a fake in front of them would test the fake.
"""

import sqlite3
from pathlib import Path

import pytest

from kaggriculture.scripts import meta

COMPETITION = "kaggriculture"

KERNELS = (
    # ref, title, author, votes, last_run
    (
        "kaitofukami/v21-memory",
        "v21.1 Conditional Memory",
        "kaitofukami",
        88,
        "2026-08-06T04:28",
    ),
    (
        "kaitofukami/v23-sparse",
        "v23 Sparse Closed Loop",
        "kaitofukami",
        7,
        "2026-08-07T09:03",
    ),
    ("someoneelse/baseline", "A Baseline", "someoneelse", 400, "2026-08-08T10:00"),
)


@pytest.fixture
def caches(tmp_path: Path) -> tuple[Path, Path]:
    """Return a populated kernel cache and discussion cache."""
    kernels = tmp_path / "kernels.db"
    with sqlite3.connect(kernels) as connection:
        connection.execute(
            "CREATE TABLE kernels (competition_id TEXT, kernel_ref TEXT, title TEXT, "
            "author TEXT, total_votes INTEGER, last_run_time TEXT, ingested_at TEXT)"
        )
        connection.executemany(
            "INSERT INTO kernels VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (COMPETITION, ref, title, author, votes, run, "2026-08-08T18:00+00:00")
                for ref, title, author, votes, run in KERNELS
            ],
        )

    discussions = tmp_path / "discussions.db"
    with sqlite3.connect(discussions) as connection:
        connection.execute(
            "CREATE TABLE discussions (competition_id TEXT, discussion_id INTEGER, "
            "title TEXT, votes INTEGER, created_at TEXT)"
        )
        connection.execute(
            "INSERT INTO discussions VALUES (?, ?, ?, ?, ?)",
            (COMPETITION, 1, "Balance Changes", 25, "2026-08-06T00:00"),
        )
    return kernels, discussions


def vendoring(directory: Path, name: str, ref: str) -> None:
    """Write a stand-in vendored agent carrying its provenance URL."""
    directory.mkdir(exist_ok=True)
    (directory / f"{name}.py").write_text(
        f'"""Vendored.\n\nhttps://www.kaggle.com/code/{ref}\n"""\n'
    )


def test_it_names_a_newer_kernel_by_an_author_we_vendor(
    caches: tuple[Path, Path], tmp_path: Path
) -> None:
    """The exact miss this module was written to prevent."""
    package = tmp_path / "package"
    vendoring(package, "kaito_policy", "kaitofukami/v21-memory")

    text = "\n".join(meta.report(*caches, package))

    assert "kaitofukami/v23-sparse" in text
    assert "1 unvendored kernel(s) newer than ours" in text


def test_it_says_we_are_level_once_the_newer_kernel_is_vendored(
    caches: tuple[Path, Path], tmp_path: Path
) -> None:
    """Vendoring the successor must clear the warning, or it is just noise."""
    package = tmp_path / "package"
    vendoring(package, "kaito_policy", "kaitofukami/v21-memory")
    vendoring(package, "kaito_v23_policy", "kaitofukami/v23-sparse")

    text = "\n".join(meta.report(*caches, package))

    assert "level with every author we vendor" in text
    assert "newer than ours" not in text


def test_a_newer_kernel_by_an_author_we_do_not_vendor_is_not_a_successor(
    caches: tuple[Path, Path], tmp_path: Path
) -> None:
    """`someoneelse/baseline` is the newest kernel cached, but it succeeds nothing."""
    package = tmp_path / "package"
    vendoring(package, "kaito_policy", "kaitofukami/v21-memory")

    lines = meta.report(*caches, package)
    successors = lines[: lines.index("newest kernels by run date")]

    assert "someoneelse/baseline" not in "\n".join(successors)


def test_it_ranks_by_run_date_and_by_votes(
    caches: tuple[Path, Path], tmp_path: Path
) -> None:
    """The two orderings must actually differ, or one of them is decoration."""
    package = tmp_path / "package"
    vendoring(package, "kaito_policy", "kaitofukami/v21-memory")
    lines = meta.report(*caches, package)

    newest = lines[lines.index("newest kernels by run date") + 1]
    voted = lines[lines.index("highest voted kernels") + 1]

    assert "someoneelse/baseline" in newest
    assert "someoneelse/baseline" in voted
    assert (
        "kaitofukami/v23-sparse" in lines[lines.index("newest kernels by run date") + 2]
    )
    assert "kaitofukami/v21-memory" in lines[lines.index("highest voted kernels") + 2]


def test_it_reports_discussions(caches: tuple[Path, Path], tmp_path: Path) -> None:
    """The discussion cache is the other half, and it carried the balance change."""
    package = tmp_path / "package"
    vendoring(package, "kaito_policy", "kaitofukami/v21-memory")

    assert "Balance Changes" in "\n".join(meta.report(*caches, package))


def test_it_refuses_to_report_nothing_when_the_cache_is_missing(tmp_path: Path) -> None:
    """A broken ingest must not read as an empty field."""
    with pytest.raises(FileNotFoundError):
        meta.report(tmp_path / "gone.db", tmp_path / "also-gone.db", tmp_path)


def test_it_opens_the_cache_read_only(
    caches: tuple[Path, Path], tmp_path: Path
) -> None:
    """A report that can write to the cache is one bug away from corrupting it."""
    kernels, _ = caches
    with meta.connect(kernels) as connection:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("DELETE FROM kernels")


def test_the_real_package_declares_the_kernels_we_actually_vendor() -> None:
    """Provenance is discovered from the source tree, so it must be discoverable."""
    found = {entry.ref for entry in meta.read_vendored(meta.PACKAGE, [])}

    assert "kaitofukami/177-180-fresh-top-30-v21-1-conditional-memory" in found
    assert "kaitofukami/23-23-strict-future-v23-sparse-closed-loop" in found
