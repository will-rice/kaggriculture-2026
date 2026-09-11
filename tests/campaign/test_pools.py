"""Nothing starts a process that runs a program except `pools.workers`."""

import ast
import os
from pathlib import Path

from kaggriculture.campaign import config, pools

# Where a process pool may be constructed, and why each is allowed.
#
# `pools` is the factory itself. `dataset` is the one pool in the campaign
# that runs none of anybody else's code -- it reads episode archives into a
# sqlite database -- so it neither needs a scratch directory nor wants one
# task per child, which would cost it a process per archive.
#
# Anything else appearing here is the question this file exists to ask: does
# it run a program? If it does, it belongs in `pools.workers`, because the
# initializer that gives a worker its own directory is exactly the argument a
# new call site forgets, and forgetting it produces a failure that appears
# only when two things run at once.
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


def test_a_worker_is_given_a_directory_of_its_own() -> None:
    """The initializer moves this process, and only this one."""
    origin = Path.cwd()
    try:
        pools.isolated()
        moved = Path.cwd()
    finally:
        os.chdir(origin)

    assert moved != origin
    assert moved.name.startswith("campaign-worker-")


def where_and_who(_: int) -> tuple[str, int]:
    """This worker's directory and process, for the test below."""
    return str(Path.cwd()), os.getpid()


def test_every_task_gets_a_fresh_process_in_a_directory_of_its_own() -> None:
    """The two guarantees the factory exists for, read off real workers.

    A policy holds state between turns, so a second game in the same
    interpreter would start from whatever the first left behind -- hence one
    task per child. And each of those children must land somewhere harmless,
    because playing a program executes it.
    """
    with pools.workers(2) as pool:
        results = list(pool.map(where_and_who, range(4)))

    directories = [where for where, _ in results]
    processes = [who for _, who in results]
    assert len(set(processes)) == len(results), "a worker was reused"
    assert len(set(directories)) == len(results), "two workers shared a directory"
    assert all(Path(where).name.startswith("campaign-worker-") for where in directories)
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
