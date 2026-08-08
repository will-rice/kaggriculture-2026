"""What the public field is doing, read out of the caches we already keep.

The nightly pass ingests every public kernel and discussion into ``data/*.db``
and, until this module existed, nothing ever read them. That gap cost real
ground: the author whose agent we vendor published two further versions on
2026-08-06 and 2026-08-07, both sitting in the cache, and we went on submitting
the 2026-08-05 one because no step in the pass ever looked. This is the step
that looks.

It is a report and nothing else -- it opens the two databases read-only, prints,
and exits. No network, no submission logic, no writes. Everything it knows was
put there by ``kernel_ingest.py`` and ``discussion_ingest.py``.

The successor section is the part worth getting right, so it is derived rather
than configured. Every vendored agent carries the Kaggle URL it was decoded
from in its module docstring; this scans ``src/kaggriculture`` for those URLs
and asks the cache what else that author has published since. Adding a fourth
vendored kernel therefore extends the report by writing the file, with no list
here to forget to update -- which is the failure mode that produced the gap in
the first place.

Staleness is printed beside every section because a cache is only evidence
about the moment it was taken, and a survey quoted a day after it was ingested
has been read here as current before.
"""

import argparse
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from kaggriculture.constants import ENVIRONMENT

LOGGER = logging.getLogger(__name__)

KERNELS_DB = Path("data/kernels.db")
DISCUSSIONS_DB = Path("data/discussions.db")
PACKAGE = Path("src/kaggriculture")

# The provenance line every vendored kernel carries in its module docstring.
KERNEL_URL = re.compile(r"https://www\.kaggle\.com/code/([\w-]+)/([\w.-]+)")

LISTED = 10


@dataclass(frozen=True)
class Kernel:
    """One row of the kernel cache."""

    ref: str
    title: str
    author: str
    votes: int
    last_run: str
    ingested_at: str


@dataclass(frozen=True)
class Vendored:
    """A kernel this repository ships a decoded copy of.

    Attributes:
        module: The file under ``src/kaggriculture`` holding the copy.
        ref: The ``owner/slug`` it was decoded from.
        last_run: When the cache last saw that kernel run, or ``""`` if the
            cache has never seen it -- which is itself worth printing, because
            it means we vendor something the ingest is not tracking.
    """

    module: str
    ref: str
    last_run: str


def main() -> None:
    """Print the meta report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernels", type=Path, default=KERNELS_DB, help="the kernel cache"
    )
    parser.add_argument(
        "--discussions", type=Path, default=DISCUSSIONS_DB, help="the discussion cache"
    )
    parser.add_argument(
        "--package", type=Path, default=PACKAGE, help="where vendored agents live"
    )
    parser.add_argument(
        "--limit", type=int, default=LISTED, help="rows per ranked section"
    )
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    for line in report(
        arguments.kernels, arguments.discussions, arguments.package, arguments.limit
    ):
        LOGGER.info("%s", line)


def report(
    kernels_db: Path, discussions_db: Path, package: Path, limit: int = LISTED
) -> list[str]:
    """Return the whole report as lines.

    Returning lines rather than printing them is what makes this testable: the
    test asserts on what the report says, not on captured stdout.

    Args:
        kernels_db: The kernel cache.
        discussions_db: The discussion cache.
        package: Directory to scan for vendored agents.
        limit: Rows per ranked section.

    Returns:
        The report, one line per entry.
    """
    kernels = read_kernels(kernels_db)
    vendored = read_vendored(package, kernels)

    lines = [f"meta report for {ENVIRONMENT}", ""]
    lines += section(
        f"newer than what we vendor ({len(vendored)} vendored kernel(s))",
        successors(kernels, vendored),
    )
    lines += section("newest kernels by run date", ranked(kernels, "run", limit))
    lines += section("highest voted kernels", ranked(kernels, "votes", limit))
    lines += section("most voted discussions", discussions(discussions_db, limit))
    lines += staleness(kernels, kernels_db, discussions_db)
    return lines


def read_kernels(database: Path) -> list[Kernel]:
    """Return every cached kernel for this competition, newest run first.

    Args:
        database: The kernel cache.

    Returns:
        The rows, or an empty list if the cache has no table yet.

    Raises:
        FileNotFoundError: If the cache does not exist. A missing cache is a
            broken ingest, not an empty field, and silently reporting nothing
            would recreate exactly the blind spot this module closes.
    """
    if not database.exists():
        raise FileNotFoundError(f"{database}: no kernel cache; has the ingest run?")
    with connect(database) as connection:
        rows = connection.execute(
            "SELECT kernel_ref, title, author, total_votes, "
            "COALESCE(last_run_time, ''), ingested_at "
            "FROM kernels WHERE competition_id = ? "
            "ORDER BY last_run_time DESC",
            (ENVIRONMENT,),
        ).fetchall()
    return [Kernel(*row) for row in rows]


def read_vendored(package: Path, kernels: list[Kernel]) -> list[Vendored]:
    """Return the kernels this repository ships a decoded copy of.

    Args:
        package: Directory to scan.
        kernels: The cache, used to date each vendored kernel.

    Returns:
        One entry per vendored kernel, by module name.
    """
    dated = {kernel.ref: kernel.last_run for kernel in kernels}
    found = []
    for module in sorted(package.glob("*.py")):
        match = KERNEL_URL.search(module.read_text())
        if match is None:
            continue
        ref = f"{match.group(1)}/{match.group(2)}"
        found.append(Vendored(module.stem, ref, dated.get(ref, "")))
    return found


def successors(kernels: list[Kernel], vendored: list[Vendored]) -> list[str]:
    """Return cached kernels by a vendored author that postdate what we ship.

    Newness is the cache's ``last_run_time`` rather than anything parsed out of
    a title. Authors number their agents however they like -- "v21.1", "v22",
    "23/23 Strict-Future" -- and a report that tried to order those strings
    would be wrong the first time someone skipped a number.

    Args:
        kernels: The cache.
        vendored: What we ship.

    Returns:
        Lines, newest first, or a single line saying we are current.
    """
    if not vendored:
        return ["no vendored kernels found -- nothing to compare against"]

    ours = {entry.ref for entry in vendored}
    newest = {
        author: max(
            (
                entry.last_run
                for entry in vendored
                if entry.ref.startswith(f"{author}/")
            ),
            default="",
        )
        for author in {entry.ref.split("/")[0] for entry in vendored}
    }
    lines = [
        f"  we vendor {entry.ref} in {entry.module}.py"
        + (
            f" (last run {entry.last_run[:16]})"
            if entry.last_run
            else " (NOT IN CACHE)"
        )
        for entry in vendored
    ]
    ahead = [
        kernel
        for kernel in kernels
        if kernel.ref not in ours
        and kernel.author in newest
        and kernel.last_run > newest[kernel.author]
    ]
    if not ahead:
        return lines + ["  -> we are level with every author we vendor"]
    return (
        lines
        + [
            f"  -> {kernel.ref}  {kernel.votes:>4} votes  run {kernel.last_run[:16]}"
            for kernel in ahead
        ]
        + [f"  -> {len(ahead)} unvendored kernel(s) newer than ours"]
    )


def ranked(kernels: list[Kernel], by: str, limit: int) -> list[str]:
    """Return the top ``limit`` kernels ordered by run date or by votes.

    Args:
        kernels: The cache, already ordered newest run first.
        by: Either ``"run"`` or ``"votes"``.
        limit: How many to return.

    Returns:
        Formatted lines.

    Raises:
        ValueError: If ``by`` is not a supported ordering.
    """
    if by == "run":
        ordered = kernels
    elif by == "votes":
        ordered = sorted(kernels, key=lambda kernel: kernel.votes, reverse=True)
    else:
        raise ValueError(f"cannot rank kernels by {by!r}")
    return [
        f"  {kernel.votes:>4} votes  run {kernel.last_run[:16]}  {kernel.ref}"
        for kernel in ordered[:limit]
    ]


def discussions(database: Path, limit: int) -> list[str]:
    """Return the most-voted cached discussions.

    Args:
        database: The discussion cache.
        limit: How many to return.

    Returns:
        Formatted lines.

    Raises:
        FileNotFoundError: If the cache does not exist.
    """
    if not database.exists():
        raise FileNotFoundError(f"{database}: no discussion cache; has the ingest run?")
    with connect(database) as connection:
        rows = connection.execute(
            "SELECT votes, COALESCE(created_at, ''), title "
            "FROM discussions WHERE competition_id = ? "
            "ORDER BY votes DESC LIMIT ?",
            (ENVIRONMENT, limit),
        ).fetchall()
    return [
        f"  {votes:>4} votes  {created[:10]:10s}  {title}"
        for votes, created, title in rows
    ]


def staleness(
    kernels: list[Kernel], kernels_db: Path, discussions_db: Path
) -> list[str]:
    """Return how old the caches are, in hours.

    Args:
        kernels: The cache rows, which carry their own ingest timestamps.
        kernels_db: Path, for naming it in the output.
        discussions_db: Path, for naming it in the output.

    Returns:
        Two lines, one per cache.
    """
    ingested = max((kernel.ingested_at for kernel in kernels), default="")
    return [
        "cache freshness",
        f"  {kernels_db}: {age(ingested)} ({len(kernels)} kernels)",
        f"  {discussions_db}: file modified {age(stamp(discussions_db))}",
        "",
    ]


def age(when: str) -> str:
    """Return a human phrase for how long ago an ISO timestamp was."""
    if not when:
        return "never ingested"
    moment = datetime.fromisoformat(when)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    hours = (datetime.now(timezone.utc) - moment).total_seconds() / 3600
    return f"{hours:.1f}h old"


def stamp(path: Path) -> str:
    """Return a file's modification time as an ISO timestamp, or ``""``."""
    if not path.exists():
        return ""
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def connect(database: Path) -> sqlite3.Connection:
    """Open a cache read-only, so a report can never modify what it reads."""
    return sqlite3.connect(f"file:{database}?mode=ro", uri=True)


def section(heading: str, lines: list[str]) -> list[str]:
    """Return a titled block, with a blank line after it."""
    return [heading, *(lines or ["  (nothing cached)"]), ""]


if __name__ == "__main__":
    main()
