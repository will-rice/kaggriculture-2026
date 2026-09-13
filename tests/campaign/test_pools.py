"""Where somebody else's program may run, and where it may write."""

import ast
import os
from pathlib import Path

from kaggriculture.campaign import config, kernel_watch, pools

# Where a process pool may be constructed, and why each is allowed.
#
# `pools` is the factory itself. `dataset` is the one pool in the campaign
# that runs none of anybody else's code -- it reads episode archives into the
# games database -- so it neither needs a scratch directory nor wants one
# task per child, which would cost it a process per archive.
#
# Anything else appearing here is the question this file exists to ask: does
# it run a program? If it does, it belongs in `pools.workers`, which gives
# every task a directory of its own. Isolation is exactly the thing a new call
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


def test_nothing_outside_pools_moves_the_working_directory() -> None:
    """`os.chdir` in a process that spawns workers is how this campaign dies.

    Twice now. The second time, on 2026-09-12, the wrapper in `pools` was
    replaced by a function each call site called for itself, and two of those
    call sites -- the harvest's kernel loader and its gate -- run on a thread
    of the loop's own process. Every game started during one of those windows
    inherited the scratch directory, and the harvest then removed it: the
    campaign died three hours later on `FileNotFoundError: [Errno 2]` with no
    filename, which is `Path.cwd()` failing in a worker standing nowhere.

    The chdir lives in exactly one function, called in exactly one place: the
    child a `Sandbox` submits into.
    """
    offenders = []
    for source in campaign_sources():
        if source.name == "pools.py":
            continue
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "chdir":
                offenders.append(f"{source.name}:{node.lineno}")

    assert not offenders, (
        f"{', '.join(offenders)} moves the working directory. It is the whole "
        "interpreter's, so a worker spawned while it is moved inherits it and "
        "outlives it. Submit through `pools.workers`, or `pools.isolated`."
    )


def where_it_stood(marker: str) -> tuple[str, bool]:
    """A task that writes a file and reports where it stood to do it."""
    Path(marker).write_text("x", encoding="utf-8")
    return str(Path.cwd()), Path(marker).exists()


def test_a_task_is_given_a_directory_it_did_not_ask_for() -> None:
    """The pool wraps the task, so a task function cannot forget to be wrapped.

    This is the guarantee that was lost on 2026-09-12 and the reason it is a
    pool rather than a call: five task functions -- two in `arena`, the game,
    the reference replay and the check -- were silently left unwrapped when
    the wrapper came out, because nothing about writing a task function says
    it needs one.
    """
    origin = Path.cwd()

    with pools.workers(1) as pool:
        where, wrote = pool.submit(where_it_stood, "scribble.txt").result()

    assert Path(where) != origin, "the task ran where the caller stands"
    assert Path(where).name.startswith("campaign-worker-")
    assert wrote, "the task could not write in its own directory"
    assert not Path(where).exists(), "the directory outlived the task"
    assert not (origin / "scribble.txt").exists()
    assert Path.cwd() == origin


def who_and_where(_: int) -> tuple[str, int]:
    """This worker's directory and process, for the test below."""
    return str(Path.cwd()), os.getpid()


def test_map_is_wrapped_too_and_every_task_gets_a_fresh_process() -> None:
    """`Executor.map` is built on `submit`, so covering `submit` covers both.

    `harness.play` and `arena.outcomes` both reach their workers through
    `map`, so a wrapper that only caught `submit` would leave every scored
    game unprotected -- which is most of the programs the campaign runs.

    One task per process besides: a policy holds state between turns, so a
    second game in the same interpreter would start from whatever the first
    left behind.
    """
    with pools.workers(2) as pool:
        results = list(pool.map(who_and_where, range(4)))

    directories = [where for where, _ in results]
    processes = [who for _, who in results]
    assert len(set(processes)) == len(results), "a worker was reused"
    assert len(set(directories)) == len(results), "two tasks shared a directory"
    assert all(Path(where).name.startswith("campaign-worker-") for where in directories)
    assert str(Path.cwd()) not in directories


def test_the_harvest_can_load_a_kernel_without_moving_the_loop() -> None:
    """The call that killed the campaign on 2026-09-12, from where it was made.

    `loadable` runs on a thread of the loop's own process, and the loop is
    spawning game workers the whole time. It has to reach a scratch directory
    without taking the process with it, which is what `pools.isolated` is for:
    the child moves, and the child has nothing to relocate.

    What this checks is that the call still works through a process boundary:
    `isolated` pickles its arguments and its result, so returning the loader's
    callable -- which is what this used to do -- stops working here, loudly,
    rather than in the harvest at three in the morning.

    It deliberately does not claim to catch the regression itself. The broken
    version restored the cwd in a `finally`, so a caller looking afterwards
    sees exactly what it sees here; the hazard was the window in between, and
    a worker spawned inside it. Forbidding the chdir is
    `test_nothing_outside_pools_moves_the_working_directory`, which reads the
    source, and that is the one that goes red.
    """
    origin = Path.cwd()

    assert kernel_watch.loadable(
        "def agent(observation, configuration):\n    return {}\n"
    )
    assert not kernel_watch.loadable("this is not python(")

    assert Path.cwd() == origin, "loading a kernel moved the loop's own process"
    assert origin.exists()


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
