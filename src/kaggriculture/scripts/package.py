"""Build the submission tarball.

There is one packager, and it is ``harness.package``: Kaggle unpacks the
archive into ``/kaggle_simulations/agent`` and imports ``main.py`` from its
root, so the archive holds one self-contained file as ``main.py``, beside the
repository licence and nothing else.

This module is the command-line face of that one function; a second packager
is how a shipping path comes to disagree with the one the campaign measured.
What ships is the campaign floor -- the file the gate writes on every
promotion -- not the cold-start seed committed under ``src/``.
"""

import argparse
import logging
from pathlib import Path

from kaggriculture.campaign import config, harness, validate

REPO_ROOT = config.ROOT
ENTRYPOINT = config.LIVE.floor / "main.py"
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
            The default is the campaign floor, which is what the gate writes
            on every promotion.

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
    automation: the gate overwrites the floor on every promotion, and a
    subagent decoding a public kernel already clobbered an entrypoint once by
    executing a notebook cell. The check is `validate.validate`, the same gate
    every candidate passes, and it executes the file only inside that gate's
    child process and scratch directory -- never in the caller's.

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
