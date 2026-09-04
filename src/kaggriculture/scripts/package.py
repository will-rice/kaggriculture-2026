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

# ``scripts`` and ``campaign`` are offline tooling -- packaging, submission and
# the codex-driven search -- that has no place in a 4 MB agent archive.
EXCLUDED = shutil.ignore_patterns("__pycache__", "scripts", "campaign")


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

    The packaging and campaign packages are left out: they import ``argparse``,
    the Kaggle client and codex tooling, none of which the served agent needs
    at play time.

    Args:
        output: Where to write the archive.
        entrypoint: Self-contained root ``main.py`` to stage. The default is
            the repository's served entrypoint.

    Returns:
        ``output``, unchanged.
    """
    with tempfile.TemporaryDirectory() as staging:
        root = Path(staging)
        package = root / PACKAGE_ROOT.name
        shutil.copytree(PACKAGE_ROOT, package, ignore=EXCLUDED)
        staged_entrypoint = root / "main.py"
        shutil.copy(entrypoint, staged_entrypoint)
        _refuse_a_shadowed_entrypoint(staged_entrypoint)
        with tarfile.open(output, "w:gz") as archive:
            for path in sorted(root.iterdir()):
                archive.add(path, arcname=path.name)
    return output


def _refuse_a_shadowed_entrypoint(entrypoint: Path) -> None:
    """Raise unless the last callable in the staged entrypoint is its agent.

    ``kaggle_environments`` plays whatever callable is defined last, so anything
    appended below the agent import is served instead of the agent, and the
    episode dies on turn zero with no useful diagnostic -- after the upload has
    spent a submission slot and displaced an agent from the scored pair.

    Checked here rather than only in a test because this file is edited by
    automation: a subagent decoding a public kernel already clobbered it once by
    executing a notebook cell. A test reports the damage; a build that refuses
    means a broken archive cannot exist to be uploaded.

    Args:
        entrypoint: The staged copy of ``main.py``, checked as it will ship
            rather than as it sits in the repository.

    Raises:
        RuntimeError: If the entrypoint binds no ``agent``, or if some other
            callable is defined after it.
    """
    source = entrypoint.read_text()
    namespace: dict[str, object] = {}
    exec(compile(source, str(entrypoint), "exec"), namespace)  # noqa: S102
    agent = namespace.get("agent")
    if agent is None:
        raise RuntimeError(f"{entrypoint} binds no `agent`; nothing would play")
    served = next(
        (value for value in reversed(tuple(namespace.values())) if callable(value)),
        None,
    )
    if served is not agent:
        raise RuntimeError(
            f"{entrypoint} defines {getattr(served, '__name__', served)!r} after "
            "its agent, so the runner would play that instead. Move it above the "
            "agent import or into the package."
        )


if __name__ == "__main__":
    main()
