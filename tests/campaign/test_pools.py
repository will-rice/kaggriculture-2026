"""Where somebody else's program may run, and where it may write."""

import ast
import os
from pathlib import Path

import pytest

from kaggriculture.campaign import config, pools

# Where a process pool may be constructed, and why each is allowed.
#
# `pools` is the factory itself. `dataset` is the one pool in the campaign
# that runs none of anybody else's code -- it reads episode archives into the
# games database -- so it neither needs a scratch directory nor wants one
# task per child, which would cost it a process per archive.
#
# Anything else appearing here is the question this file exists to ask: does
# it run a program? If it does, it belongs in `pools.workers`, whose children
# take one task each, and its task belongs in `pools.sandboxed`, which gives
# that task a directory of its own. Isolation is exactly the thing a new call
# site forgets, and forgetting it produces a failure that appears only when
# two things run at once.
ALLOWED = {"pools.py", "dataset.py"}


def campaign_sources() -> list[Path]:
    """Every module in the campaign package."""
    return sorted(Path(config.__file__).parent.glob("*.py"))


def test_only_the_factory_starts_a_worker_process() -> None:
    """A pool built anywhere else is a worker with nowhere safe to write.

    On 2026-09-11 a check was given a scratch directory by moving the caller's
    working directory, which is what it looks like you should do. `os.chdir`
    moves the whole interpreter, a spawned child inherits the cwd of whoever
    spawned it, and so every game the loop started during that check began
    inside a tree the check then deleted. The campaign died on
    `FileNotFoundError`, an hour after the change that was meant to protect it.
    """
    offenders = []
    for source in campaign_sources():
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            named = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if named == "ProcessPoolExecutor" and source.name not in ALLOWED:
                offenders.append(f"{source.name}:{node.lineno}")

    assert not offenders, (
        f"{', '.join(offenders)} builds its own process pool. A worker that "
        "runs a program needs a directory of its own, and `pools.workers` is "
        "where that is decided once rather than remembered at every call site."
    )


def test_a_task_runs_in_a_directory_that_is_gone_when_it_returns() -> None:
    """The scratch tree's life is the task's, and the cwd comes back with it.

    It used to be `mkdtemp` plus an `atexit` hook at process birth. Both
    remove the tree; only one of them says when. A worker killed mid-game --
    routine here, because the pacer cancels codex calls and the loop tears its
    pools down with them -- never reaches its `atexit`, and 164 scratch
    directories from the same shape were sitting in /tmp when this was
    written.
    """
    origin = Path.cwd()
    seen: list[Path] = []

    def note_where() -> str:
        """A task that writes a file and reports where it stood to do it."""
        seen.append(Path.cwd())
        Path("scribble.txt").write_text("x", encoding="utf-8")
        return "done"

    assert pools.sandboxed(note_where) == "done"

    assert seen[0] != origin
    assert seen[0].name.startswith("campaign-")
    assert not seen[0].exists(), "the directory outlived the task"
    assert Path.cwd() == origin, "the task left the process somewhere else"
    assert not (origin / "scribble.txt").exists()


def test_a_task_that_raises_still_gives_its_directory_back() -> None:
    """The exception path is the one a `finally` is written for.

    A candidate raising mid-game is an ordinary outcome here -- the harness
    reports it as a failed evaluation -- so the path that matters most is the
    one where the task does not return normally.
    """
    origin = Path.cwd()
    seen: list[Path] = []

    def explode() -> None:
        """A task that fails the way a broken candidate does."""
        seen.append(Path.cwd())
        raise ValueError("this candidate is broken")

    with pytest.raises(ValueError, match="broken"):
        pools.sandboxed(explode)

    assert Path.cwd() == origin
    assert not seen[0].exists(), "a failed task kept its directory"


def where_and_who(_: int) -> tuple[str, int]:
    """A worker task shaped like the real ones: sandboxed, then reporting.

    `harness._one` and `harness._checked` are this, with a game in place of
    the `Path.cwd()`. The sandbox is the task's own call, not something the
    pool does to it, because the pool cannot know which of its tasks runs
    somebody else's program.
    """
    return pools.sandboxed(lambda: (str(Path.cwd()), os.getpid()))


def test_every_task_gets_a_fresh_process_in_a_directory_of_its_own() -> None:
    """The two guarantees a worker running a program needs, read off real ones.

    A policy holds state between turns, so a second game in the same
    interpreter would start from whatever the first left behind -- hence one
    task per child, which is what `workers` is for. And each of those children
    must land somewhere harmless, because playing a program executes it --
    which is what the task's own `sandboxed` call is for.
    """
    with pools.workers(2) as pool:
        results = list(pool.map(where_and_who, range(4)))

    directories = [where for where, _ in results]
    processes = [who for _, who in results]
    assert len(set(processes)) == len(results), "a worker was reused"
    assert len(set(directories)) == len(results), "two tasks shared a directory"
    assert all(Path(where).name.startswith("campaign-") for where in directories)
    # And none of them is where the test is standing.
    assert str(Path.cwd()) not in directories


def test_no_campaign_path_depends_on_where_the_process_stands() -> None:
    """A module-level path must not be relative, now that workers move.

    `kernel_watch.SEEN` and `WORK` were `Path("run/kernel-watch/...")`, which
    is correct for exactly as long as nothing changes the working directory.
    Things do: every worker that runs a program is given one of its own, and
    the harvest runs in the same process as the loop that starts them. A
    relative constant then resolves somewhere else without raising -- the
    harvest writes its record of which kernels it has already seen into a
    scratch tree about to be deleted, and re-checks every kernel forever.
    """
    relative = []
    for source in campaign_sources():
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            call = node.value
            if not (
                isinstance(call, ast.Call) and getattr(call.func, "id", "") == "Path"
            ):
                continue
            first = call.args[0] if call.args else None
            if isinstance(first, ast.Constant) and not str(first.value).startswith("/"):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                relative.append(f"{source.name}:{node.lineno} {names}")

    assert not relative, (
        f"{', '.join(relative)} is relative to the working directory, which "
        "the campaign moves. Anchor it to `config.ROOT`."
    )
