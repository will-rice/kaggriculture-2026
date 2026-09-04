"""Prove a packaged tarball loads and plays inside the actual Kaggle image."""

import os
import shutil
from pathlib import Path

import pytest

from kaggriculture.campaign import harness, kaggle_image

pytestmark = pytest.mark.slow

PASS_AGENT = (
    "def agent(observation, configuration=None):\n"
    "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
)


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available")
def test_the_packaged_skeleton_loads_and_plays_in_the_kaggle_image(
    tmp_path: Path,
) -> None:
    """A tarball built by ``harness.package`` loads and plays on the Kaggle image."""
    agent = tmp_path / "main.py"
    agent.write_text(PASS_AGENT, encoding="utf-8")
    tarball = harness.package(agent, tmp_path / "submission.tar.gz")
    output = kaggle_image.load_test(tarball)
    assert "ENGINE_OK 1.32.7" in output and "EPISODE_OK" in output
    # The container must never leave root-owned files behind in a caller's
    # tmp_path -- that would make it undeletable outside a privileged shell.
    owned_by_us = {path: path.stat().st_uid for path in tmp_path.rglob("*")}
    assert all(uid == os.getuid() for uid in owned_by_us.values()), owned_by_us
