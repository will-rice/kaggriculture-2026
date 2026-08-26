"""Frozen, authoring-free hybrid runtime modules."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from kaggriculture.hybrid.config import HybridConfig, to_runtime
from kaggriculture.search.scripts import freeze_hybrid
from kaggriculture.search.scripts.freeze_hybrid import freeze_runtime


def test_freeze_runtime_round_trips_only_to_the_caller_path(tmp_path: Path) -> None:
    """Freezing writes one requested module and no repository winner default."""
    output = tmp_path / "candidate_runtime.py"

    actual = freeze_runtime(HybridConfig.default(), output)

    assert actual == output
    assert freeze_hybrid.load_runtime_module(output).RUNTIME == to_runtime(
        HybridConfig.default()
    )
    assert tuple(path.name for path in tmp_path.iterdir()) == (output.name,)
    assert "pydantic" not in output.read_text()
    assert not Path("src/kaggriculture/hybrid/winner.py").exists()


def test_frozen_module_imports_without_pydantic_or_torch(tmp_path: Path) -> None:
    """The generated runtime is safe to import on the packaged turn path."""
    output = freeze_runtime(HybridConfig.default(), tmp_path / "runtime.py")
    source = (
        "import runpy, sys; "
        f"runpy.run_path({str(output)!r}); "
        "assert 'pydantic' not in sys.modules; "
        "assert 'torch' not in sys.modules"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src"

    result = subprocess.run(
        [sys.executable, "-I", "-c", source],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr


def test_freeze_runtime_preserves_existing_output_when_validation_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A bad generated round trip cannot replace the last valid candidate."""
    output = tmp_path / "runtime.py"
    output.write_text("old evidence\n")

    def reject(path: Path) -> object:
        raise RuntimeError(f"bad generated module {path.name}")

    monkeypatch.setattr(freeze_hybrid, "load_runtime_module", reject)

    with pytest.raises(RuntimeError, match="bad generated module"):
        freeze_runtime(HybridConfig.default(), output)

    assert output.read_text() == "old evidence\n"
    assert not list(tmp_path.glob("*.tmp"))


def test_freeze_cli_requires_a_caller_provided_output() -> None:
    """The CLI has no implicit path capable of creating hybrid/winner.py."""
    with pytest.raises(SystemExit):
        freeze_hybrid.parser().parse_args(["--config", "candidate.json"])
