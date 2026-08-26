"""Freeze a validated hybrid config into a dependency-free runtime module."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from types import ModuleType

from kaggriculture.hybrid.config import HybridConfig, to_runtime


def parser() -> argparse.ArgumentParser:
    """Build a CLI with no implicit repository output path."""
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--config", type=Path, required=True)
    arguments.add_argument("--output", type=Path, required=True)
    return arguments


def load_runtime_module(path: Path) -> ModuleType:
    """Execute a generated module without writing import-cache files."""
    module = ModuleType("_frozen_hybrid_runtime")
    source = path.read_text()
    exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102
    return module


def freeze_runtime(config: HybridConfig, output: Path) -> Path:
    """Atomically write and validate a runtime at the caller's exact path."""
    if output.name in {"", ".", ".."} or output.exists() and not output.is_file():
        raise ValueError("output must name a runtime module file")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output parent does not exist: {output.parent}")
    runtime = to_runtime(config)
    source = (
        '"""Generated validated hybrid runtime configuration."""\n\n'
        "from kaggriculture.hybrid.runtime import RuntimeConfig\n\n"
        f"RUNTIME = RuntimeConfig.from_payload({runtime.to_payload()!r})\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(source)
            stream.flush()
            os.fsync(stream.fileno())
        loaded = load_runtime_module(temporary)
        if getattr(loaded, "RUNTIME", None) != runtime:
            raise RuntimeError("frozen runtime round trip changed the configuration")
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output


def main() -> None:
    """Validate authoring JSON and freeze it only to ``--output``."""
    args = parser().parse_args()
    config = HybridConfig.model_validate_json(args.config.read_text())
    freeze_runtime(config, args.output)


if __name__ == "__main__":
    main()
