"""Compile the engine bridge into the library the wrapper loads.

The command is the one the port's own kernel used, with our bridge in place
of its policy.
"""

import logging
import os
import subprocess
from pathlib import Path

from kaggriculture.campaign import config

LOGGER = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent


def build() -> Path:
    """Compile ``bridge.cpp`` and return the library path; skip if up to date.

    Every evaluation forks a pool of workers and each one loads the library,
    so several processes can reach this function at once on a cold tree.
    ``g++`` writes its output incrementally, so compiling straight to the
    final path would let one worker ``dlopen`` a half-written file. Each
    compile goes to a private ``<target>.<pid>.tmp`` and lands with
    ``Path.replace``, which is ``rename(2)`` and so atomic within a
    filesystem: a concurrent reader sees either the old library or the whole
    new one.

    Returns:
        The compiled library's path.
    """
    sources = [HERE / "bridge.cpp", HERE / "sim.hpp", HERE / "pyrandom.hpp"]
    target = config.ENGINE_LIBRARY
    if target.exists() and target.stat().st_mtime >= max(
        s.stat().st_mtime for s in sources
    ):
        return target
    scratch = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    command = [
        "g++",
        "-O3",
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-I",
        str(HERE),
        "-o",
        str(scratch),
        str(HERE / "bridge.cpp"),
    ]
    LOGGER.info("building %s", target.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    scratch.replace(target)
    return target
