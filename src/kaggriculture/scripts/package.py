"""Build the submission tarball.

There is one packager, and it is ``harness.package``: Kaggle unpacks the
archive into ``/kaggle_simulations/agent`` and imports ``main.py`` from its
root, so the archive holds the served agent as ``main.py`` beside the engine
library, the four plumbing modules, and the attribution the library carries.

This module is the command-line face of that one function. Two packagers is
how the shipping path came to have no engine in it: the artefact
``kaggle_image.load_test`` proves and the artefact ``uv run submit`` uploads
have to be the same bytes, built by the same code.
"""

import argparse
import logging
from pathlib import Path

from kaggriculture.campaign import config, harness, validate

REPO_ROOT = config.ROOT
ENTRYPOINT = config.SERVED
SUBMISSION = REPO_ROOT / "submission.tar.gz"


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


def build(output: Path = SUBMISSION, *, entrypoint: Path = ENTRYPOINT) -> Path:
    """Write the submission archive and return its path.

    Args:
        output: Where to write the archive.
        entrypoint: Self-contained ``main.py`` to ship at the archive root.
            The default is the served agent, which is what the gate writes on
            every promotion.

    Returns:
        ``output``, unchanged.
    """
    _refuse_a_shadowed_entrypoint(entrypoint)
    return harness.package(entrypoint, output)


def _refuse_a_shadowed_entrypoint(entrypoint: Path) -> None:
    """Raise unless the entrypoint passes the candidate gate.

    ``kaggle_environments`` plays whatever callable is defined last, so
    anything appended below the agent is served instead of the agent, and the
    episode dies on turn zero with no useful diagnostic -- after the upload
    has spent a submission slot and displaced an agent from the scored pair.

    Checked here rather than only in a test because this file is written by
    automation: the gate overwrites it on every promotion, and a subagent
    decoding a public kernel already clobbered it once by executing a
    notebook cell. The check is `validate.validate`, the same gate every
    candidate passes, and it executes the file only inside that gate's child
    process and scratch directory -- never in the caller's.

    Args:
        entrypoint: The ``main.py`` that will ship, checked as it sits.

    Raises:
        RuntimeError: If the entrypoint fails any check, the shadowed-agent
            check included.
    """
    verdict = validate.validate(entrypoint)
    if verdict.status != "ok":
        raise RuntimeError(
            f"{entrypoint} must not ship: {verdict.status}: {verdict.reason}"
        )
