"""Tripwires for an unreviewed reference-engine change."""
# ruff: noqa: D103

import hashlib
import json
from pathlib import Path

from kaggriculture.sim.fidelity import reference_branches, reference_identity

MANIFEST = Path(__file__).with_name("reference-1.32.7.json")


def test_installed_reference_matches_reviewed_files() -> None:
    expected = json.loads(MANIFEST.read_text())
    actual = reference_identity()

    assert actual.version == expected["version"]
    assert actual.source_sha256 == expected["source_sha256"]
    assert actual.configuration_sha256 == expected["configuration_sha256"]


def test_reference_branch_manifest_is_current() -> None:
    expected = json.loads(MANIFEST.read_text())
    branches = reference_branches()
    digest = hashlib.sha256("\n".join(branches).encode()).hexdigest()

    assert len(branches) == expected["branch_count"]
    assert digest == expected["branch_manifest_sha256"]
