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

So the directory is given to a worker at birth, by `workers`, and nothing else
in the campaign constructs a process pool. That is enforced rather than
remembered -- `test_pools.py` reads the source and fails on any other
construction -- because the initializer is exactly the kind of argument a new
call site forgets, and forgetting it produces a bug that only appears when two
things happen at once.
"""

import atexit
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor


def isolated() -> None:
    """Give this worker a directory of its own, before it runs anything.

    Once, at the start of the process, rather than around each task: a worker
    made by `workers` runs one task and exits, so there is nothing to restore
    and the directory goes when the process does. Registered for removal as
    well, because a worker killed mid-game would otherwise leave it -- 548 of
    those accumulated in /tmp once already, from a different hole.
    """
    scratch = tempfile.mkdtemp(prefix="campaign-worker-")
    atexit.register(shutil.rmtree, scratch, ignore_errors=True)
    os.chdir(scratch)


def workers(count: int) -> ProcessPoolExecutor:
    """A pool whose workers each own a scratch directory and run one task.

    One task per child because these run programs: a policy holds state
    between turns, and a second game in the same interpreter would start from
    whatever the first left behind.

    Args:
        count: Worker processes.

    Returns:
        The pool, to be used as a context manager.
    """
    return ProcessPoolExecutor(
        max_workers=count, max_tasks_per_child=1, initializer=isolated
    )
