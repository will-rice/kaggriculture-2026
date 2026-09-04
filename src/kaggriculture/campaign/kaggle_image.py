"""Prove a tarball loads on the Kaggle image before a slot is spent on it.

The image is the notebook image already pulled on this box; the runner's
glibc and architecture are the same, which is what the ``.so`` depends on.
The image's own ``kaggle_environments`` may be a different version, or the
image may lack network access to install ours -- either way the script below
falls back to this repo's vendored copy without ever reaching the network.
"""

import subprocess
from importlib.metadata import version
from pathlib import Path

from kaggriculture.campaign import config

IMAGE = "gcr.io/kaggle-gpu-images/python:latest"
VENV_KAGGLE_ENVIRONMENTS = (
    config.ROOT
    / ".venv"
    / "lib"
    / "python3.11"
    / "site-packages"
    / "kaggle_environments"
)
# The venv this repo runs on, queried once here rather than trusted from
# inside the container: the fallback re-import below cannot see its own
# dist-info (only the package directory is mounted), so its version has to
# travel in as a literal, not be rediscovered at runtime.
KAGGLE_ENVIRONMENTS_VERSION = version("kaggle-environments")

SCRIPT = """
import ctypes
import sys
import tarfile

tarfile.open('/work/{tarball_name}').extractall('/work/agent')
lib = ctypes.CDLL('/work/agent/kaggriculture_engine.so')
lib.kag_engine_version.restype = ctypes.c_char_p
print('ENGINE_OK', lib.kag_engine_version().decode())

import kaggle_environments
if kaggle_environments.__version__ == '{ke_version}':
    print('KE', kaggle_environments.__version__, 'system')
else:
    for name in list(sys.modules):
        if name == 'kaggle_environments' or name.startswith('kaggle_environments.'):
            del sys.modules[name]
    sys.path.insert(0, '/work')
    import kaggle_environments
    print('KE', '{ke_version}', 'fallback')

from kaggle_environments import make
env = make('kaggriculture', configuration={{'episodeSteps': 720}})
env.run(['/work/agent/main.py', '/work/agent/main.py'])
print('EPISODE_OK', env.steps[-1][0].reward, env.steps[-1][1].reward)
"""


def load_test(tarball: Path) -> str:
    """Run the load script against ``tarball`` inside the Kaggle image.

    Args:
        tarball: A tarball written by ``harness.package``.

    Returns:
        The container's stdout.

    Raises:
        RuntimeError: The container exited nonzero; carries the stderr tail.
    """
    script = SCRIPT.format(
        tarball_name=tarball.name, ke_version=KAGGLE_ENVIRONMENTS_VERSION
    )
    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{tarball.resolve().parent}:/work",
        "-v",
        f"{VENV_KAGGLE_ENVIRONMENTS}:/work/kaggle_environments:ro",
        IMAGE,
        "python",
        "-c",
        script,
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:])
    return result.stdout
