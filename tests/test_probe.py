"""Tests for the Phase 0 sandbox probe."""

import logging

import pytest

from kaggriculture import probe
from kaggriculture.policy import probe_sandbox


def test_probe_reports_the_facts_phase_zero_asks_for(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The probe names the interpreter and the machine it found itself on."""
    with caplog.at_level(logging.INFO, logger="kaggriculture.probe"):
        probe.report()

    messages = [record.getMessage() for record in caplog.records]
    assert any("PROBE python" in message for message in messages)
    assert any("PROBE cpus" in message for message in messages)


def test_a_missing_module_is_a_finding_rather_than_a_crash(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Whether torch imports at all is the question, so a failure must be logged."""
    logger = probe.probe_logger()

    with caplog.at_level(logging.INFO, logger="kaggriculture.probe"):
        module = probe.import_timed(logger, "a_module_the_sandbox_does_not_have")

    assert module is None
    assert any("unavailable" in record.getMessage() for record in caplog.records)


def test_trunk_sizes_match_the_solutions_they_are_named_after() -> None:
    """The point is timing the real shapes, so the parameter counts must land."""
    sizes = {
        label: probe.trunk_parameters(blocks, channels)
        for label, blocks, channels in probe.TRUNKS
    }

    assert sizes["toad-brigade-lux-s1-20M"] == pytest.approx(20e6, rel=0.15)
    assert sizes["lux-s3-second-place-300M"] == pytest.approx(300e6, rel=0.05)


def test_a_broken_probe_cannot_cost_the_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It rides on a live ladder submission, so it may fail but never raise."""

    def explode() -> None:
        raise RuntimeError("the sandbox did something unexpected")

    monkeypatch.setattr(probe, "report", explode)

    probe_sandbox()
