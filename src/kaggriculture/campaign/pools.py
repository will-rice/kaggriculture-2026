"""Running somebody else's program: where it runs, and where it may write.

Playing a program executes it. An evolved candidate and a kernel harvested an
hour ago are both somebody else's code, and both may write files -- relative
ones, so the working directory is the lever.

`os.chdir` moves the whole interpreter, and that is the whole difficulty. A
process that spawns workers must never do it: a spawned child inherits the cwd
of whoever spawned it, so a chdir on the loop's own process relocates every
game started while it is in effect, and removing the scratch tree afterwards
leaves those games standing in a directory that is gone. They die on
`FileNotFoundError: [Errno 2]` with no filename -- `Path.cwd()` failing, which
reads like nothing at all.

That has killed this campaign twice: on 2026-09-11, when a check was given a
scratch directory in the obvious place, and on 2026-09-12, when the wrapper
below was briefly replaced by a function each call site called for itself.
Scoping the chdir to a single call does not fix it. It bounds how long the
hazard lasts, not whether a worker spawned during it inherits a doomed cwd.

So it is not a thing call sites do. `workers` returns a pool that wraps every
task it is handed, the chdir happens in the child around that one task, and
nothing else in the campaign constructs a process pool -- `test_pools.py`
reads the source and fails on any other construction. A funnel rather than a
convention, because the second attempt at this proved the convention: five
task functions were silently left unwrapped when the wrapper came out, and the
one that mattered took three hours to surface.

`isolated` is the door for a caller that holds no pool and cannot chdir --
the loop, and anything on one of its threads. It forks, so the result crosses
a process boundary and must be picklable; a caller that wanted a live object
gets a name instead.

Every path the campaign itself uses is absolute from `config.ROOT`;
`test_pools.py` reads the source and fails on a relative one, because a
relative path resolved inside a sandbox is a file written somewhere that is
about to be removed.
"""

import os
import tempfile
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import TypeVar, cast

# Written out rather than as PEP 695's `def isolated[T]`, which the repository
# runs on but pre-commit's `check-ast` hook parses under 3.11 and rejects.
T = TypeVar("T")


class Sandbox(ProcessPoolExecutor):
    """A pool that gives every task a directory of its own, whoever submits it.

    `submit` is the single funnel: `Executor.map` is built on it, so wrapping
    here covers both and there is no second path a caller could reach a worker
    through. That matters more than it looks -- a task function that runs
    somebody's program is added by whoever needs one, and the isolation is
    exactly the part they will not think about.
    """

    def submit(  # type: ignore[override]
        self, fn: Callable[..., T], /, *args: object, **kwargs: object
    ) -> "Future[T]":
        """Submit ``fn`` to run in a scratch directory in its worker.

        Args:
            fn: The task. Any picklable callable.
            *args: Its positional arguments.
            **kwargs: Its keyword arguments.

        Returns:
            The future for the task's result.
        """
        # `_sandboxed` returns whatever it is given returns, so this future
        # carries `fn`'s result -- which the checker cannot see, because the
        # two functions declare their own type variables and nothing relates
        # them. The cast states the relation the wrapper is built on.
        return cast("Future[T]", super().submit(_sandboxed, fn, *args, **kwargs))


def workers(count: int) -> Sandbox:
    """A pool that runs one task per process, each in a directory of its own.

    One task per process because these run policies, and a policy holds state
    between turns: a second game in the same interpreter would start from
    whatever the first left behind.

    Args:
        count: Worker processes.

    Returns:
        The pool, to be used as a context manager.
    """
    return Sandbox(max_workers=count, max_tasks_per_child=1)


def isolated(call: Callable[..., T], *args: object) -> T:
    """Run ``call`` in a child process, in a directory of its own.

    For a caller that spawns workers of its own and so cannot chdir: the loop,
    the harvest on one of its threads, the validator. The child does the
    moving, and it has nothing to relocate.

    Args:
        call: A picklable callable -- a module-level function, not a lambda
            or a closure.
        *args: Its arguments, picklable on the same terms.

    Returns:
        Whatever ``call`` returns, which must survive pickling back.
    """
    with workers(1) as pool:
        return pool.submit(call, *args).result()


def _sandboxed(call: Callable[..., T], *args: object, **kwargs: object) -> T:
    """Run one task in a directory of its own, and put the old one back.

    Module-level so `spawn` can import it, and private because the only
    correct place to call it is the child `Sandbox.submit` puts it in.

    The directory is a `TemporaryDirectory` rather than an `mkdtemp` and an
    `atexit` hook. Both remove the tree; only one of them says when. `__exit__`
    runs when the task returns, on the exception path as well, where an
    `atexit` waits on the interpreter and never runs at all if the worker is
    killed -- and a killed worker is routine here, because the pacer cancels
    codex calls and the loop tears its pools down with them.

    The cwd is put back because this function does not know whether it is the
    only thing this process will ever run.
    """
    home = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="campaign-worker-") as scratch:
        try:
            os.chdir(scratch)
            return call(*args, **kwargs)
        finally:
            os.chdir(home)
