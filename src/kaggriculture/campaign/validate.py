"""Everything that rejects a candidate before a game is scored.

Cheap checks first, in the order they were each found necessary: a syntax
error costs nothing to find; a missing entrypoint costs one `ast` pass; a
copy check costs a handful of set intersections against a cached corpus,
and runs before the imports and shadowed-entrypoint checks so a wholesale
copy of an opponent (see roster "v56", whose real submission imports its
own modules and names its real entrypoint something other than `agent` on
purpose) reports as a copy, not as an unrelated contract or import
complaint about code that was never ours to bound. The imports check
itself has two layers: `ast.Import`/`ast.ImportFrom` nodes, and calls --
`__import__`, `eval`, `exec`, `compile`, `open`, `input`, and anything
rooted at `importlib` -- that reach a module or a resource without ever
naming it in an import statement. The shadowed-entrypoint check has two
layers too: a syntactic pass over top-level definitions, and a dynamic one
that loads the module the way Kaggle's runner does, because a decorator or
a conditional definition can rebind the last name in ways no AST walk
sees. Only once every one of those clears does a candidate earn ~20s of
reference-engine time in `harness.check`.

That dynamic half -- the load and the reference-engine run -- happens in a
child process with a wall-clock cap. Loading a module runs its top-level
code, so a candidate with an unbounded loop outside any function would
otherwise wedge the validating thread, and through it the island it was
mutating, for the life of the run. The child also works from a scratch
directory, because loading a candidate is running it.
"""

import ast
import multiprocessing
import os
import queue
import tempfile
import time
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from kaggriculture.campaign import copycheck, harness

# One file ships, so this is the whole surface a program may name. The
# `kaggriculture` package and `ctypes` were once here, for the engine library
# the archive used to carry; the archive is now `main.py` and the licence, so
# a candidate importing either would validate here and die on the ladder --
# the one failure this system cannot afford.
#
# `base64` and `zlib` are here because one file must carry everything: an
# agent with a learned table -- and the seed the campaign starts from is one
# -- has nowhere to put it but a compressed literal in its own source. Both
# are pure computation over bytes: neither opens a file, reaches the network,
# or loads a module. Nor can what they return become code, because
# `_forbidden_call` blocks `eval`, `exec` and `compile` outright, so a
# decoded blob stays data no matter what it decodes to.
ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "math",
        "statistics",
        "itertools",
        "collections",
        "functools",
        "operator",
        "random",
        "heapq",
        "bisect",
        "dataclasses",
        "typing",
        "enum",
        "copy",
        "json",
        "base64",
        "zlib",
        "pathlib",
        "time",
        "kaggle_environments",
    }
)


class Verdict(BaseModel):
    """The result of `validate`: the first failing check, or ok.

    Attributes:
        status: Which check failed, or "ok" if none did.
        reason: Why, in a sentence that never names a filesystem path --
            a copied opponent is named by its roster key, never its file.
        worst_step_seconds: The slowest single call to the agent, from
            `harness.check`; 0.0 for a verdict reached before that check ran.
    """

    status: Literal[
        "ok", "syntax", "contract", "imports", "copy", "crashed", "too_slow"
    ]
    reason: str
    worst_step_seconds: float = 0.0


_DENIED_CALLS: frozenset[str] = frozenset(
    {"__import__", "eval", "exec", "compile", "open", "input"}
)


def _has_top_level_agent(tree: ast.Module) -> bool:
    """Whether a top-level function literally named ``agent`` exists."""
    return any(
        isinstance(node, ast.FunctionDef) and node.name == "agent" for node in tree.body
    )


def _top_level_callable_name(node: ast.stmt) -> str | None:
    """The name a top-level statement binds, if the value it binds is callable.

    A `def` or `class` always binds a callable; an assignment binds one only
    when its value is a `lambda` -- an ordinary value (a dict, a number)
    does not shadow anything Kaggle's loader would call instead.
    """
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Lambda):
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        return names[-1] if names else None
    if (
        isinstance(node, ast.AnnAssign)
        and isinstance(node.value, ast.Lambda)
        and isinstance(node.target, ast.Name)
    ):
        return node.target.id
    return None


def _last_top_level_callable_name(tree: ast.Module) -> str | None:
    """The name of the last top-level callable Kaggle's runner would call.

    ``get_last_callable`` takes the last callable defined in the file, not
    the one named ``agent`` -- a helper defined after ``agent`` (a function,
    a class, or a lambda bound to a name) would run instead of it, silently.
    This is the syntactic half of that rule; `validate` also checks it
    dynamically, against the actual module namespace, right before
    `harness.check` -- a decorator or a conditional definition can rebind
    the last name without looking like it from the syntax alone.
    """
    names = [
        name
        for node in tree.body
        if (name := _top_level_callable_name(node)) is not None
    ]
    return names[-1] if names else None


def _forbidden_import(tree: ast.Module) -> str | None:
    """The first imported top-level module name not in `ALLOWED_IMPORTS`."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            top = name.split(".", 1)[0]
            if top not in ALLOWED_IMPORTS:
                return name
    return None


def _forbidden_call(tree: ast.Module) -> str | None:
    """The first call that reaches a module without ever naming it in an import.

    `ast.Import`/`ast.ImportFrom` are not the only door to a module:
    `__import__` (and `eval`/`exec`/`compile`, which can build and run an
    `__import__` call from a string) reach any module by name at runtime,
    bypassing `ALLOWED_IMPORTS` entirely. `open` and `input` are blocked the
    same way, as the same kind of ambient access a sandboxed agent must not
    have. `importlib` is already unimportable, but a call chain rooted at
    that name is rejected explicitly too, for a reason that names it rather
    than relying on the import check having caught it first.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in _DENIED_CALLS:
            return func.id
        if isinstance(func, ast.Attribute):
            root = func.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id == "importlib":
                return f"importlib.{func.attr}"
    return None


def _shadowed_entrypoint(tree: ast.Module) -> Verdict | None:
    """Whether something other than `agent` is the last top-level definition.

    The syntactic half of the rule. The dynamic half -- against the loaded
    module's actual namespace, where a decorator or a conditional definition
    can rebind the last name in ways no AST walk sees -- is `_dynamic`, which
    runs in the child process because it has to load the file to look.
    """
    last_name = _last_top_level_callable_name(tree)
    if last_name != "agent":
        return Verdict(
            status="contract",
            reason=f"agent is shadowed by {last_name!r}, the last top-level definition",
        )
    return None


def _dynamic(agent: Path, steps: int) -> Verdict:
    """Load the candidate as Kaggle does, then play it on the reference engine.

    Everything here executes the candidate's own code, which is why the caller
    runs it in a child process. A load failure is exactly what
    `harness.check` would report as `crashed` a moment later, so it is
    reported the same way.

    Args:
        agent: The candidate's `main.py`, already absolute.
        steps: How many turns `harness.check` plays before stopping.

    Returns:
        The failing `Verdict`, or `status="ok"`.
    """
    try:
        last_callable = harness.load_agent(agent)
    except Exception as error:  # noqa: BLE001 - a candidate's own failure
        return Verdict(status="crashed", reason=f"{type(error).__name__}: {error}")
    loaded_name = getattr(last_callable, "__name__", "<anonymous>")
    if loaded_name != "agent":
        return Verdict(
            status="contract",
            reason=f"agent is shadowed by {loaded_name!r}, the last callable",
        )
    report = harness.check(agent, steps=steps)
    if not report.loaded or report.error is not None:
        return Verdict(
            status="crashed",
            reason=report.error or "did not load",
            worst_step_seconds=report.worst_step_seconds,
        )
    if report.worst_step_seconds > harness.LATENCY_BUDGET:
        return Verdict(
            status="too_slow",
            reason=f"worst step {report.worst_step_seconds:.3f}s",
            worst_step_seconds=report.worst_step_seconds,
        )
    return Verdict(status="ok", reason="", worst_step_seconds=report.worst_step_seconds)


def _dynamic_child(agent: str, steps: int, results: "Queue[dict]") -> None:
    """Run `_dynamic` in a scratch directory and put its verdict on ``results``.

    Module-level so ``spawn`` can import it. The scratch directory is never
    left, because the process ends with this call.

    Args:
        agent: The candidate's `main.py` as an absolute path string.
        steps: How many turns `harness.check` plays before stopping.
        results: Where the verdict goes, as a plain dict.
    """
    with tempfile.TemporaryDirectory(prefix="campaign-check-") as scratch:
        os.chdir(scratch)
        results.put(_dynamic(Path(agent), steps).model_dump())


# Seconds a candidate may take to import before it is called at all. Spawning
# the child and importing the module are not the agent's per-call work, so
# Kaggle's `actTimeout` does not cover them and neither should the deadline
# below. Sixty seconds is sixty times the heaviest import in the opponent
# roster: `v56` unpacks its tables in 0.95s and `pilkwang_economic` in 0.74s,
# both far larger than anything the campaign has evolved.
LOAD_ALLOWANCE = 60.0


def _dynamic_verdict(agent: Path, steps: int) -> Verdict:
    """`_dynamic` in a child process, waited on for as long as it takes.

    The child process is isolation first: a candidate is evolved source that
    may write files or take the interpreter down with it, and neither should
    reach the thread validating it.

    The wait is bounded, but not by a budget of ours. Kaggle gives an agent
    one second per call, so a program Kaggle would accept finishes ``steps``
    calls inside ``steps * ACT_TIMEOUT`` -- plus ``LOAD_ALLOWANCE`` to import,
    which is not per-call work -- however slow it is; anything still running
    past that is not slow, it is a program the ladder would have killed.

    The distinction matters because the campaign keeps no clocks of its own
    any more: a codex call runs until it is done, and codex calls do return.
    Evolved source does not have to. A module-level ``while True``
    parses, imports `agent` as the last callable, passes every static check,
    and then loads forever -- and without this the thread validating it, and
    the worker behind that thread, are gone for the life of the campaign.

    Args:
        agent: The candidate's `main.py`.
        steps: How many turns `harness.check` plays before stopping.

    Returns:
        The child's verdict; `too_slow` if it outran what Kaggle itself
        allows; `crashed` if the child exited without reporting one.
    """
    context = multiprocessing.get_context("spawn")
    results: "Queue[dict]" = context.Queue()
    process = context.Process(
        target=_dynamic_child, args=(str(agent.resolve()), steps, results)
    )
    process.start()
    allowed = LOAD_ALLOWANCE + steps * harness.ACT_TIMEOUT
    deadline = time.monotonic() + allowed
    while True:
        # Poll in short slices rather than blocking outright, so a child that
        # dies without reporting -- a module-level ``SystemExit``, a native
        # crash, a failed spawn -- is seen at once instead of waiting on a
        # queue nothing will ever write to.
        try:
            payload = results.get(timeout=1.0)
            break
        except queue.Empty:
            if process.exitcode is not None:
                return Verdict(
                    status="crashed",
                    reason=(
                        f"check process exited with code {process.exitcode} "
                        "before reporting"
                    ),
                )
            if time.monotonic() >= deadline:
                process.kill()
                process.join()
                return Verdict(
                    status="too_slow",
                    reason=(
                        f"still running after {allowed:.0f}s: {steps} calls "
                        f"at Kaggle's own {harness.ACT_TIMEOUT}s each, and "
                        f"{LOAD_ALLOWANCE:.0f}s to load"
                    ),
                )
    process.join()
    return Verdict.model_validate(payload)


def validate(agent: Path, steps: int = 720, seed: Path | None = None) -> Verdict:
    """Run every check, cheapest first, and return the first one that fails.

    Args:
        agent: The candidate's `main.py`.
        steps: How many turns `harness.check` plays before stopping.
        seed: The program this campaign was seeded from, whose lineage
            the copy check exempts. None exempts nothing.

    Returns:
        The first failing `Verdict`, or `status="ok"`.
    """
    source = agent.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return Verdict(status="syntax", reason=str(error))

    if not _has_top_level_agent(tree):
        return Verdict(status="contract", reason="no top-level function named agent")

    # The copy check runs before the imports scan: a wholesale copy of an
    # opponent (see roster "v56", whose real submission imports modules of
    # its own, like ``base64`` and a private ``v49``) must report as a copy,
    # not as an incidental import the whitelist happens not to name -- the
    # whitelist exists to bound what our own candidates may do, and a copy
    # is not one of ours to bound.
    offender, score = copycheck.against_opponents(source, seed)
    if score >= copycheck.THRESHOLD:
        name = offender.split(":", 1)[0]
        return Verdict(status="copy", reason=f"{score:.3f} similar to {name}")

    forbidden_import = _forbidden_import(tree)
    if forbidden_import is not None:
        return Verdict(
            status="imports", reason=f"import of {forbidden_import!r} is not allowed"
        )

    forbidden_call = _forbidden_call(tree)
    if forbidden_call is not None:
        return Verdict(
            status="imports", reason=f"call to {forbidden_call!r} is not allowed"
        )

    # Checked after the copy check: a copied opponent's own wrapper-name
    # games -- see roster "v56", whose last callable is not named ``agent``
    # on purpose -- would otherwise report the wrong reason for the same
    # rejected file.
    shadowed = _shadowed_entrypoint(tree)
    if shadowed is not None:
        return shadowed

    return _dynamic_verdict(agent, steps)
