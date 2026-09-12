"""Running somebody else's program: where it runs, and where it may write.

Two things, and both are small.

`workers` starts one task per process, because these run policies and a policy
holds state between turns: a second game in the same interpreter would start
from whatever the first left behind.

`sandboxed` runs one call inside a directory of its own. Playing a program
executes it, and an evolved candidate or a kernel harvested an hour ago may
write files -- relative ones, so the working directory is the lever. It is
scoped: entered for the call and restored after it, however the call ends.

Scoped is the whole of it. A chdir that outlives its call relocates everything
the process does next, including every worker it spawns, and the campaign died
on `FileNotFoundError: /tmp/campaign-check-56ludna7` on 2026-09-11 for exactly
that -- a check moved the loop's own directory, the games running at the time
inherited it, and the check then deleted it from under them. Held to one call,
there is nothing to inherit and nothing to outlive.

Every path the campaign itself uses is absolute from `config.ROOT`;
`test_pools.py` reads the source and fails on a relative one, because a
relative path resolved inside a sandbox is a file written somewhere that is
about to be removed.
"""

import os
import tempfile
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TypeVar

# Written out rather than as PEP 695's `def sandboxed[T]`, which the repository
# runs on but pre-commit's `check-ast` hook parses under 3.11 and rejects.
T = TypeVar("T")


def sandboxed(call: Callable[[], T]) -> T:
    """Run ``call`` in a directory of its own, and put the old one back.

    Args:
        call: What to run. Takes nothing, so the caller binds its arguments.

    Returns:
        Whatever ``call`` returns.
    """
    home = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="campaign-") as scratch:
        try:
            os.chdir(scratch)
            return call()
        finally:
            os.chdir(home)


def workers(count: int) -> ProcessPoolExecutor:
    """A pool that runs one task per process.

    Args:
        count: Worker processes.

    Returns:
        The pool, to be used as a context manager.
    """
    return ProcessPoolExecutor(max_workers=count, max_tasks_per_child=1)
