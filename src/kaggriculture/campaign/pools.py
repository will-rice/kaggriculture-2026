"""The only place the campaign starts a process that runs somebody's program.

Playing a program executes it. An evolved candidate and a kernel harvested
from the competition an hour ago are both somebody else's code, and both may
write files -- so neither may run where the caller lives. The lever for that
is the working directory, and `os.chdir` moves the whole interpreter rather
than one task.

Which makes *where* it is called the entire question, and getting it wrong
does not look wrong. A chdir on the loop's own process silently relocates
every worker it spawns next, because a spawned child inherits the cwd of
whoever spawned it. The campaign died on
`FileNotFoundError: /tmp/campaign-check-56ludna7` on 2026-09-11, an hour after
a check was given a scratch directory in the obvious place: games started
while that check ran began inside its scratch tree, and the check then removed
it from under them.

So the chdir happens inside the worker, around one task, and nothing else in
the campaign constructs a process pool. That is enforced rather than
remembered -- `test_pools.py` reads the source and fails on any other
construction -- because isolation is exactly the kind of thing a new call site
forgets, and forgetting it produces a bug that only appears when two things
happen at once. Every task submitted through `workers` is wrapped here, so
there is no argument to pass and none to leave out.
"""

import functools
import os
import tempfile
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import TypeVar

# Written out rather than as PEP 695's `def sandboxed[T]`, which the repository
# runs on but pre-commit's `check-ast` hook parses under 3.11 and rejects.
T = TypeVar("T")


def sandboxed(call: Callable[..., T], *args: object, **kwargs: object) -> T:
    """Run one task in a directory of its own, and take the directory away.

    The unit is the task, not the process. A worker made by `workers` runs one
    task and exits, so the two amount to the same thing today -- but only the
    first is true whatever `max_tasks_per_child` is set to later, and a second
    task landing in whatever the first left behind is the failure this exists
    to prevent.

    The directory is a `TemporaryDirectory` rather than an `mkdtemp` and an
    `atexit` hook. Both remove the tree; only one of them says when. `__exit__`
    runs when the task returns, on the exception path as well, where an
    `atexit` waits on the interpreter and never runs at all if the worker is
    killed -- and a killed worker is routine here, because the pacer cancels
    codex calls and the loop tears its pools down with them.

    The cwd is put back for the same reason: this function does not know
    whether it is the only thing this process will ever run.

    Args:
        call: The task. Any picklable callable.
        *args: Its positional arguments.
        **kwargs: Its keyword arguments.

    Returns:
        Whatever ``call`` returns.
    """
    home = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="campaign-worker-") as scratch:
        try:
            os.chdir(scratch)
            return call(*args, **kwargs)
        finally:
            os.chdir(home)


class Sandbox(ProcessPoolExecutor):
    """A pool that gives every task a scratch directory, whoever submits it.

    `submit` is the single funnel: `Executor.map` is built on it, so wrapping
    here covers both and there is no second path a caller could reach the
    workers by.
    """

    def submit(
        self, fn: Callable[..., T], /, *args: object, **kwargs: object
    ) -> Future[T]:
        """Submit ``fn`` to run inside a directory of its own."""
        # Bound to a name so the task's return type survives the wrapping:
        # `sandboxed` is generic, and passed straight through it reaches the
        # base `submit` as an unsolved variable rather than as this task's.
        wrapped: Callable[..., T] = functools.partial(sandboxed, fn)
        return super().submit(wrapped, *args, **kwargs)


def workers(count: int) -> Sandbox:
    """A pool whose tasks each own a scratch directory, one task per process.

    One task per child because these run programs: a policy holds state
    between turns, and a second game in the same interpreter would start from
    whatever the first left behind.

    Args:
        count: Worker processes.

    Returns:
        The pool, to be used as a context manager.
    """
    return Sandbox(max_workers=count, max_tasks_per_child=1)
