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

from kaggriculture.learn import CHECKPOINT

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "src" / "kaggriculture"
ENTRYPOINT = REPO_ROOT / "main.py"
SUBMISSION = REPO_ROOT / "submission.tar.gz"

# ``copytree`` applies these at every directory level, so one "scripts" entry
# drops both ``kaggriculture/scripts`` and ``kaggriculture/learn/scripts``.
#
# ``learn`` itself is no longer excluded: the trained policy is the agent now,
# and ``learn/play.py`` imports ``learn/model.py``, ``learn/encoding.py`` and
# the checkpoint beside them. ``corpus.py`` and ``dataset.py`` are named here
# instead, one file at a time, because they are training-only and pull in tqdm
# and a ``/data`` path the sandbox does not have.
#
# ``*.pt`` is excluded so that ``build`` is the single thing that decides the
# checkpoint ships. Left to ``copytree``, the weights would be included exactly
# when a training run happened to have left them in the source tree, and absent
# without complaint when it had not.
EXCLUDED = shutil.ignore_patterns(
    "__pycache__", "scripts", "corpus.py", "dataset.py", "*.pt"
)


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

    The checkpoint is copied in explicitly rather than left to ``copytree``,
    which would pick it up only when it happened to be sitting in the source
    tree. It is gitignored and written by a training run, so a fresh checkout
    has no such file, and the archive would ship a policy of random weights
    that plays a full episode and loses without ever raising.

    Args:
        output: Where to write the archive.

    Returns:
        ``output``, unchanged.

    Raises:
        FileNotFoundError: If no checkpoint has been trained yet.
    """
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"no checkpoint at {CHECKPOINT} — run "
            "`uv run python -m kaggriculture.learn.scripts.train` first"
        )
    with tempfile.TemporaryDirectory() as staging:
        root = Path(staging)
        package = root / PACKAGE_ROOT.name
        shutil.copytree(PACKAGE_ROOT, package, ignore=EXCLUDED)
        shutil.copy(CHECKPOINT, package / CHECKPOINT.relative_to(PACKAGE_ROOT))
        shutil.copy(ENTRYPOINT, root / ENTRYPOINT.name)
        with tarfile.open(output, "w:gz") as archive:
            for path in sorted(root.iterdir()):
                archive.add(path, arcname=path.name)
    return output


if __name__ == "__main__":
    main()
