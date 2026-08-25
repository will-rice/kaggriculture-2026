"""Contracts for authoring and runtime hybrid configurations."""
# ruff: noqa: D103

import json
import subprocess
import sys
from types import MappingProxyType
from typing import Any

import pytest
from pydantic import ValidationError

from kaggriculture.hybrid.config import HybridConfig, to_runtime
from kaggriculture.hybrid.runtime import RuntimeConfig


def import_in_fresh_process(module: str) -> set[str]:
    """Return modules loaded by importing ``module`` in a fresh interpreter."""
    output = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import importlib, json, sys; "
            f"importlib.import_module({module!r}); "
            "print(json.dumps(sorted(sys.modules)))",
        ],
        text=True,
    )
    return set(json.loads(output))


def test_hybrid_config_is_frozen_finite_and_forbids_extra_fields() -> None:
    """Reject mutation, non-finite authoring input, and undeclared controls."""
    with pytest.raises(ValidationError):
        HybridConfig.model_validate({"cash_reserve": float("nan")})
    with pytest.raises(ValidationError):
        HybridConfig.model_validate({"unknown_knob": 3})

    config = HybridConfig.default()
    with pytest.raises(ValidationError):
        setattr(config, "liquidation_start_day", 26)  # noqa: B010

    payload = config.model_dump(mode="python")
    payload["opening"]["phases"][0]["target_hands"] = "4"
    with pytest.raises(ValidationError):
        HybridConfig.model_validate(payload)


def test_hybrid_config_rejects_invalid_phase_schemas_and_order() -> None:
    """Reject milestones that a runtime cannot unambiguously execute."""
    payload = HybridConfig.default().model_dump(mode="python")
    phase = payload["opening"]["phases"][0]
    phase["primary_crop"] = "NOT_A_CROP"
    with pytest.raises(ValidationError):
        HybridConfig.model_validate(payload)

    payload = HybridConfig.default().model_dump(mode="python")
    first = payload["opening"]["phases"][0]
    payload["opening"]["phases"] = (first, {**first, "start_day": 0})
    with pytest.raises(ValidationError, match="unique increasing"):
        HybridConfig.model_validate(payload)

    payload = HybridConfig.default().model_dump(mode="python")
    payload["opening"]["phases"][0]["crop_targets"] = (0,)
    with pytest.raises(ValidationError, match="fixed crop schema"):
        HybridConfig.model_validate(payload)


def test_pydantic_to_runtime_round_trip_is_exact_and_dependency_free() -> None:
    """Freeze validated data into a submission-safe, lossless runtime value."""
    config = HybridConfig.default()

    runtime = to_runtime(config)

    assert HybridConfig.from_runtime(runtime) == config
    assert RuntimeConfig.from_payload(runtime.to_payload()) == runtime
    assert RuntimeConfig.from_payload(MappingProxyType(runtime.to_payload())) == runtime
    imported = import_in_fresh_process("kaggriculture.hybrid.runtime")
    assert "pydantic" not in imported
    assert "torch" not in imported


def test_runtime_payload_rejects_wrong_keys_and_containers() -> None:
    """Reject generated payloads that do not exactly mirror runtime dataclasses."""
    payload: dict[str, Any] = to_runtime(HybridConfig.default()).to_payload()

    with pytest.raises(ValueError, match="keys"):
        RuntimeConfig.from_payload({**payload, "extra": True})

    malformed = {**payload, "phases": tuple(payload["phases"])}
    with pytest.raises(TypeError, match="phases"):
        RuntimeConfig.from_payload(malformed)

    malformed = {**payload, "jobs": {"recovery": 1.0}}
    with pytest.raises(ValueError, match="job"):
        RuntimeConfig.from_payload(malformed)
