"""Compile the engine bridge into the library the wrapper loads.

The command is the one the port's own kernel used, with our bridge in place
of its policy.
"""

import logging
import subprocess
from pathlib import Path

from kaggriculture.campaign import config

LOGGER = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent


def build() -> Path:
    """Compile ``bridge.cpp`` and return the library path; skip if up to date."""
    sources = [HERE / "bridge.cpp", HERE / "sim.hpp", HERE / "pyrandom.hpp"]
    target = config.ENGINE_LIBRARY
    if target.exists() and target.stat().st_mtime >= max(
        s.stat().st_mtime for s in sources
    ):
        return target
    command = [
        "g++",
        "-O3",
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-I",
        str(HERE),
        "-o",
        str(target),
        str(HERE / "bridge.cpp"),
    ]
    LOGGER.info("building %s", target.name)
    subprocess.run(command, check=True)
    return target
