"""Shared helpers for probes against the installed Kaggriculture engine."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as engine


ROOT = Path(__file__).resolve().parents[1]
PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def assert_exact_engine() -> dict[str, str]:
    """Prove that probes use 1.32.7 and the supplied ground-truth source."""
    version = importlib.metadata.version("kaggle-environments")
    assert version == "1.32.7", version
    supplied = (ROOT / "engine" / "kaggriculture.py").read_bytes()
    installed = Path(engine.__file__).read_bytes()
    supplied_sha256 = hashlib.sha256(supplied).hexdigest()
    installed_sha256 = hashlib.sha256(installed).hexdigest()
    assert supplied_sha256 == installed_sha256
    return {
        "kaggle_environments": version,
        "engine_sha256": supplied_sha256,
    }


def new_env(seed: int = 1, **configuration: Any):
    config = {"episodeSteps": 720, "seed": seed, **configuration}
    return make("kaggriculture", configuration=config, debug=True)


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))
