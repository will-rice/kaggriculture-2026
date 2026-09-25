"""One pass over every module: what it says it is for, what it exports, who imports it.

The consolidation question is which of these are the same thing under two names,
and that is not answerable from the modules touched today. This reads the lot.
"""

import ast
import collections
import re
import sys
from pathlib import Path

ROOT = Path(
    "/home/will/projects/kaggriculture-2026/.claude/worktrees/campaign/src/kaggriculture"
)
# Comments that record a decision reversed or a path retired. Each is a seam
# where two designs met; the survivor is documented, the loser often is too.
REVERSALS = re.compile(
    r"until 2026-|used to |is gone|are gone|no longer|there was a second|"
    r"was never |never rendered|nothing reads|never read|dead |retired",
    re.IGNORECASE,
)


def main() -> None:
    """Print a table per module, then the import graph, then who is unreferenced."""
    modules = sorted(
        path
        for path in ROOT.rglob("*.py")
        if path.name != "__init__.py" and "served" not in path.parts
    )
    imported_by: dict[str, set[str]] = collections.defaultdict(set)
    exports: dict[str, list[str]] = {}
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        name = ".".join(path.relative_to(ROOT).with_suffix("").parts)
        doc = (ast.get_docstring(tree) or "").split("\n\n")[0].replace("\n", " ")
        public = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and not node.name.startswith("_")
        ]
        exports[name] = public
        lines = path.read_text(encoding="utf-8").splitlines()
        flips = [i + 1 for i, line in enumerate(lines) if REVERSALS.search(line)]
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and "kaggriculture" in node.module
            ):
                for alias in node.names:
                    imported_by[
                        f"{node.module.split('kaggriculture.')[-1]}.{alias.name}"
                    ].add(name)
                    imported_by[node.module.split("kaggriculture.")[-1]].add(name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if "kaggriculture" in alias.name:
                        imported_by[alias.name.split("kaggriculture.")[-1]].add(name)
        print(
            f"### {name}  ({len(lines)} lines, {len(public)} public, {len(flips)} reversal marks)"
        )
        print(f"    {doc[:230]}")
        print(
            f"    exports: {', '.join(public[:14])}{' ...' if len(public) > 14 else ''}"
        )
        if flips:
            print(
                f"    reversals at lines: {flips[:12]}{' ...' if len(flips) > 12 else ''}"
            )

    print("\n### who imports whom (campaign modules)")
    for name in sorted(exports):
        users = sorted(u for u in imported_by.get(name, ()) if u != name)
        print(
            f"  {name:<28} <- {', '.join(users) if users else '(nothing in the package)'}"
        )

    sys.stdout.flush()


if __name__ == "__main__":
    main()
