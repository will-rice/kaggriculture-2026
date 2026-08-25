"""Contracts for authoring and runtime hybrid configurations."""
# ruff: noqa: D103

import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from itertools import product
from types import MappingProxyType
from typing import Any

import pytest
from pydantic import ValidationError

from kaggriculture.hybrid.config import HybridConfig, to_runtime
from kaggriculture.hybrid.runtime import RuntimeConfig
from kaggriculture.hybrid.schema import PHASE_START_DAYS
from kaggriculture.search.genome import GenomeCodec

PhasePayload = dict[str, object]
PhaseSchedule = Callable[[tuple[PhasePayload, ...]], tuple[PhasePayload, ...]]


def phase_payloads(*, starts: tuple[int, int, int]) -> tuple[dict[str, Any], ...]:
    """Return the baseline three phases with independently chosen start days."""
    phases = HybridConfig.default().model_dump(mode="python")["opening"]["phases"]
    return tuple(
        {**phase, "start_day": start}
        for phase, start in zip(phases, starts, strict=True)
    )


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
    phases = payload["opening"]["phases"]
    payload["opening"]["phases"] = (
        phases[0],
        {**phases[1], "start_day": 9},
        phases[2],
    )
    with pytest.raises(ValidationError, match="fixed position domain"):
        HybridConfig.model_validate(payload)

    payload = HybridConfig.default().model_dump(mode="python")
    payload["opening"]["phases"] = (
        {**payload["opening"]["phases"][0], "start_day": 1},
        *payload["opening"]["phases"][1:],
    )
    with pytest.raises(ValidationError, match="fixed position domain"):
        HybridConfig.model_validate(payload)

    payload = HybridConfig.default().model_dump(mode="python")
    payload["opening"]["phases"][0]["crop_targets"] = (0,)
    with pytest.raises(ValidationError, match="fixed crop schema"):
        HybridConfig.model_validate(payload)


@pytest.mark.parametrize(
    "phases",
    (
        lambda baseline: baseline[:1],
        lambda baseline: baseline[:2],
        lambda baseline: (*baseline, {**baseline[-1], "start_day": 29}),
        lambda baseline: (
            baseline[0],
            {**baseline[1], "start_day": 9},
            baseline[2],
        ),
    ),
)
def test_hybrid_config_rejects_schedules_outside_the_fixed_genome_schema(
    phases: PhaseSchedule,
) -> None:
    """Reject schedules that the fixed codec previously could not round-trip."""
    payload = HybridConfig.default().model_dump(mode="python")
    payload["opening"]["phases"] = phases(payload["opening"]["phases"])

    with pytest.raises(ValidationError):
        HybridConfig.model_validate(payload)


def test_every_authoring_phase_day_in_the_fixed_schema_round_trips() -> None:
    """Every accepted three-position day combination is genome encodable."""
    codec = GenomeCodec.default()
    domains = tuple((days[0], days[-1]) for days in PHASE_START_DAYS)
    template = HybridConfig.default().model_dump(mode="python")

    for starts in product(*domains):
        config = HybridConfig.model_validate(
            {
                **template,
                "opening": {
                    "phases": phase_payloads(starts=(starts[0], starts[1], starts[2]))
                },
            }
        )

        assert codec.decode(codec.encode(config)) == config


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


def test_runtime_construction_rejects_invalid_first_and_unordered_phases() -> None:
    """Keep direct dependency-free construction inside the fixed phase schema."""
    runtime = to_runtime(HybridConfig.default())

    with pytest.raises(ValueError, match="phase start_day"):
        replace(runtime.phases[0], start_day=1)
    with pytest.raises(ValueError, match="phase 0"):
        RuntimeConfig(
            phases=(runtime.phases[1], runtime.phases[0], runtime.phases[2]),
            jobs=runtime.jobs,
            market=runtime.market,
            liquidation_start_day=runtime.liquidation_start_day,
        )


def test_runtime_payload_rejects_unordered_phase_schedules() -> None:
    """Apply fixed position day domains again at the generated-payload boundary."""
    payload: dict[str, Any] = to_runtime(HybridConfig.default()).to_payload()
    phases = payload["phases"]
    assert type(phases) is list
    phases[0], phases[1] = phases[1], phases[0]

    with pytest.raises(ValueError, match="phase 0"):
        RuntimeConfig.from_payload(payload)
