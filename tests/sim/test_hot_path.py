"""CUDA-capturable hot-path invariants."""
# ruff: noqa: D103

import ast
import inspect

from kaggriculture.sim import day, engine, market, units


def test_step_path_contains_no_host_synchronization_calls() -> None:
    forbidden = {"item", "nonzero", "tolist", "cpu", "numpy"}
    for module in (engine, units, market, day):
        tree = ast.parse(inspect.getsource(module))
        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert not calls & forbidden, (module.__name__, calls & forbidden)


def test_step_path_does_not_import_the_reference_interpreter() -> None:
    source = inspect.getsource(engine)

    assert "kaggle_environments.envs.kaggriculture" not in source
    assert "interpreter" not in source
