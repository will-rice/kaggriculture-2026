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

from kaggle_environments.agent import get_last_callable

from kaggriculture.routes import STORE

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
# ``learn`` goes entirely: ``main.py`` serves the route agent, which imports
# none of it, and shipping it carried a 39 MB checkpoint the archive never
# loaded -- measured, an archive built with it was 41.7 MB and `torch` was
# absent from ``sys.modules`` after a full episode. Excluding the package also
# removes any path by which a later edit could import torch into the agent and
# spend 10.7 s of the 60 s overage pool.
#
# ``*.pt`` and the prototype store are excluded so that ``build`` is the single
# thing that decides they ship. Left to ``copytree``, each would be included
# exactly when a harvest run happened to have left it in the source tree, and
# absent without complaint when it had not.
#
# ``search`` is offline hill-climbing tooling: it plays hundreds of games
# against a league to find a better route, work that has no place in a 4 MB
# agent archive regardless of what it imports today. It happens to import no
# torch right now, but that is not the reason it is excluded -- a later edit
# could add one, and this entry means that edit cannot also reintroduce it
# into the submission.
EXCLUDED = shutil.ignore_patterns(
    "__pycache__", "scripts", "learn", "search", "*.pt", STORE.name
)

# The two build artifacts the archive cannot be assembled without, each mapped
# to the module that produces it. Both are gitignored, so a fresh checkout has
# neither, and both are loaded from beside the package at play time -- an
# archive missing one is not a degraded agent but a broken one.
REQUIRED = {STORE: "kaggriculture.routes.scripts.harvest"}


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

    ``REQUIRED`` is copied in explicitly rather than left to ``copytree``,
    which would pick each file up only when it happened to be sitting in the
    source tree. Both are gitignored build artifacts, so a fresh checkout has
    neither, and the failure is silent in the direction that matters: without
    the checkpoint the archive ships a policy of random weights that plays a
    full episode and loses without ever raising, and without the store the
    route agent raises ``FileNotFoundError`` on turn zero, in the sandbox,
    where nobody sees it until the leaderboard reads zero.

    Args:
        output: Where to write the archive.

    Returns:
        ``output``, unchanged.

    Raises:
        FileNotFoundError: If a required build artifact has not been produced.
    """
    for artifact, producer in REQUIRED.items():
        if not artifact.is_file():
            raise FileNotFoundError(
                f"no {artifact.name} at {artifact} — run "
                f"`uv run python -m {producer}` first"
            )
    with tempfile.TemporaryDirectory() as staging:
        root = Path(staging)
        package = root / PACKAGE_ROOT.name
        shutil.copytree(PACKAGE_ROOT, package, ignore=EXCLUDED)
        for artifact in REQUIRED:
            shutil.copy(artifact, package / artifact.relative_to(PACKAGE_ROOT))
        shutil.copy(ENTRYPOINT, root / ENTRYPOINT.name)
        _refuse_a_shadowed_entrypoint(root / ENTRYPOINT.name)
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
    served = get_last_callable(source, path=str(entrypoint))
    if served is not agent:
        raise RuntimeError(
            f"{entrypoint} defines {getattr(served, '__name__', served)!r} after "
            "its agent, so the runner would play that instead. Move it above the "
            "agent import or into the package."
        )


if __name__ == "__main__":
    main()
