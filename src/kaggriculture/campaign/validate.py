"""Everything that rejects a candidate before a game is scored.

Cheap checks first, in the order they were each found necessary: a syntax
error costs nothing to find; a missing entrypoint costs one `ast` pass; a
copy check costs a handful of set intersections against a cached corpus,
and runs before the imports and shadowed-entrypoint checks so a wholesale
copy of an opponent (see roster "v56", whose real submission imports its
own modules and names its real entrypoint something other than `agent` on
purpose) reports as a copy, not as an unrelated contract or import
complaint about code that was never ours to bound -- and only then does a
candidate earn ~20s of reference-engine time in `harness.check`.
"""

import ast
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from kaggriculture.campaign import copycheck, harness

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
        "ctypes",
        "pathlib",
        "time",
        "kaggriculture",
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


def _has_top_level_agent(tree: ast.Module) -> bool:
    """Whether a top-level function literally named ``agent`` exists."""
    return any(
        isinstance(node, ast.FunctionDef) and node.name == "agent" for node in tree.body
    )


def _last_top_level_callable_name(tree: ast.Module) -> str | None:
    """The name of the last top-level function Kaggle's runner would call.

    ``get_last_callable`` takes the last callable defined in the file, not
    the one named ``agent`` -- a helper defined after ``agent`` would run
    instead of it, silently.
    """
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    return functions[-1].name if functions else None


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


def validate(agent: Path, steps: int = 720) -> Verdict:
    """Run every check, cheapest first, and return the first one that fails.

    Args:
        agent: The candidate's `main.py`.
        steps: How many turns `harness.check` plays before stopping.

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
    offender, score = copycheck.against_opponents(source)
    if score >= copycheck.THRESHOLD:
        name = offender.split(":", 1)[0]
        return Verdict(status="copy", reason=f"{score:.3f} similar to {name}")

    forbidden = _forbidden_import(tree)
    if forbidden is not None:
        return Verdict(
            status="imports", reason=f"import of {forbidden!r} is not allowed"
        )

    # A cheap AST check, but it must run after the copy check: a copied
    # opponent's own wrapper-name games -- see roster "v56", whose last
    # top-level function is not named ``agent`` on purpose -- would
    # otherwise report the wrong reason for the same rejected file.
    last_name = _last_top_level_callable_name(tree)
    if last_name != "agent":
        return Verdict(
            status="contract",
            reason=f"agent is shadowed by {last_name!r}, the last top-level function",
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
