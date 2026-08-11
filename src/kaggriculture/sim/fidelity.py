"""Reference-engine identity and branch-manifest helpers."""

import ast
import hashlib
import importlib.metadata
import importlib.util
import inspect
import textwrap
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

REFERENCE_FUNCTIONS = (
    "_apply_unit_action",
    "_commit_unit",
    "_parse_order",
    "_do_hire",
    "_do_buy_land",
    "_daily_refresh_plants",
    "_daily_refresh_animals",
    "_decay_plants",
    "_spawn_weeds",
    "_drop_inventories_to_shed",
    "_town_consume",
    "_end_of_day",
)


@dataclass(frozen=True)
class ReferenceIdentity:
    """Installed ground-truth package identity."""

    version: str
    source_sha256: str
    configuration_sha256: str


def reference_directory() -> Path:
    """Return the directory containing the installed reference engine."""
    spec = importlib.util.find_spec(
        "kaggle_environments.envs.kaggriculture.kaggriculture"
    )
    if spec is None or spec.origin is None:
        raise RuntimeError("installed Kaggriculture reference engine was not found")
    return Path(spec.origin).parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_identity() -> ReferenceIdentity:
    """Hash the installed wheel files used as simulator ground truth."""
    root = reference_directory()
    return ReferenceIdentity(
        version=importlib.metadata.version("kaggle-environments"),
        source_sha256=_sha256(root / "kaggriculture.py"),
        configuration_sha256=_sha256(root / "kaggriculture.json"),
    )


def reference_branches() -> tuple[str, ...]:
    """Parse stable names for every conditional in the named reference functions."""
    path = reference_directory() / "kaggriculture.py"
    source = path.read_text()
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    branches: list[str] = []
    for function_name in REFERENCE_FUNCTIONS:
        function = functions[function_name]
        for node in ast.walk(function):
            if isinstance(node, ast.If):
                branches.append(
                    f"{function_name}:{node.lineno}:{ast.unparse(node.test)}"
                )
    return tuple(sorted(branches))


class _BranchTransformer(ast.NodeTransformer):
    def __init__(self, function_name: str, first_line: int) -> None:
        self.function_name = function_name
        self.first_line = first_line

    def visit_If(self, node: ast.If) -> ast.AST:  # noqa: N802
        """Wrap one condition while retaining its original source identity."""
        self.generic_visit(node)
        branch = (
            f"{self.function_name}:{self.first_line + node.lineno - 1}:"
            f"{ast.unparse(node.test)}"
        )
        node.test = ast.Call(
            func=ast.Name(id="_sim_branch", ctx=ast.Load()),
            args=[ast.Constant(branch), node.test],
            keywords=[],
        )
        return node


def _reference_module() -> ModuleType:
    from kaggle_environments.envs.kaggriculture import (  # noqa: PLC0415
        kaggriculture,
    )

    return kaggriculture


@contextmanager
def track_reference_branches() -> Iterator[dict[str, dict[str, int]]]:
    """Count true and false outcomes of every named reference conditional."""
    module = _reference_module()
    counts = {branch: {"true": 0, "false": 0} for branch in reference_branches()}

    def record(branch: str, value: Any) -> Any:  # noqa: ANN401
        counts[branch]["true" if bool(value) else "false"] += 1
        return value

    originals = {name: getattr(module, name) for name in REFERENCE_FUNCTIONS}
    try:
        for name, function in originals.items():
            lines, first_line = inspect.getsourcelines(function)
            tree = ast.parse(textwrap.dedent("".join(lines)))
            tree = _BranchTransformer(name, first_line).visit(tree)
            ast.fix_missing_locations(tree)
            namespace = dict(function.__globals__)
            namespace["_sim_branch"] = record
            exec(
                compile(tree, inspect.getsourcefile(function) or "<reference>", "exec"),
                namespace,
            )
            setattr(module, name, namespace[name])
        yield counts
    finally:
        for name, function in originals.items():
            setattr(module, name, function)
