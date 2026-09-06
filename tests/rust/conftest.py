"""Build the Rust engine once per session and expose it as a fixture."""

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
CRATE = ROOT / "rust"


def pytest_addoption(parser: Any) -> None:  # noqa: ANN401
    """Register the campaign size controls."""
    parser.addoption("--rust-episodes", type=int, default=2)
    parser.addoption("--rust-turns", type=int, default=719)


@pytest.fixture(scope="session")
def engine_binary() -> Path:
    """Path to a freshly built debug binary of the Rust engine."""
    if shutil.which("cargo") is None:
        pytest.skip("cargo is not installed")
    subprocess.run(
        ["cargo", "build", "--quiet", "--manifest-path", str(CRATE / "Cargo.toml")],
        check=True,
        cwd=CRATE,
    )
    binary = CRATE / "target" / "debug" / "kaggriculture-engine"
    if not binary.exists():
        raise RuntimeError(f"cargo build did not produce {binary}")
    return binary
