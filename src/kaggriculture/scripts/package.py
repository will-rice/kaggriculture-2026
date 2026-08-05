"""Build the submission tarball.

Kaggle unpacks the archive into ``/kaggle_simulations/agent`` and imports
``main.py`` from its root, so the archive holds ``main.py`` beside a flat copy of
the ``kaggriculture`` package.
"""

import argparse
import logging
import shutil
import tarfile
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "src" / "kaggriculture"
ENTRYPOINT = REPO_ROOT / "main.py"
SUBMISSION = REPO_ROOT / "submission.tar.gz"
EXCLUDED = shutil.ignore_patterns("__pycache__", "scripts", "learn")


def main() -> None:
    """Build the submission archive from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SUBMISSION, help="archive path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    archive = build(args.output)
    logging.getLogger(__name__).info(
        "wrote %s (%.1f KiB)", archive, archive.stat().st_size / 1024
    )


def build(output: Path = SUBMISSION) -> Path:
    """Write the submission archive and return its path.

    The packaging scripts themselves are left out: they import ``argparse`` and
    the Kaggle client, neither of which the agent needs at play time.
    """
    with tempfile.TemporaryDirectory() as staging:
        root = Path(staging)
        shutil.copytree(PACKAGE_ROOT, root / PACKAGE_ROOT.name, ignore=EXCLUDED)
        shutil.copy(ENTRYPOINT, root / ENTRYPOINT.name)
        with tarfile.open(output, "w:gz") as archive:
            for path in sorted(root.iterdir()):
                archive.add(path, arcname=path.name)
    return output


if __name__ == "__main__":
    main()
