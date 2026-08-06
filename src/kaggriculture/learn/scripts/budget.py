"""Measure a full episode of the packaged agent against the sandbox's clock.

The competition gives an agent one second a turn (``actTimeout``) and a
60-second pool (``remainingOverageTime``) to cover every turn that overruns it.
The torch import is paid inside turn zero, because ``kaggle_environments``
compiles and execs an agent file lazily on its first call, and it comes out of
that pool. So the question this answers is not whether the model is fast -- it
is whether a whole season of it fits in what is left. If it does not, a torch
policy is not submittable at all, whatever it scores locally.

Two things make the number honest rather than flattering:

* The episode runs from the **built archive**, extracted to a temporary
  directory that goes on ``sys.path`` ahead of everything else, in a fresh
  interpreter. Measuring the repo would measure a tree with the training
  scripts, ``/data`` and wandb in it, which is not what plays.
* The agent gets ``play.THREADS`` cores, which is what the sandbox has. This
  workstation has 64, and letting the model take them flatters the turn cost by
  an order of magnitude.

One thing it cannot make honest: this is still a workstation. The Phase 0
sandbox probe measured the torch import at **10.7 s** there against ~1 s here,
and ~20-26 GMAC/s at two threads against this machine's ~90. Read the import
figure below as a lower bound and add the sandbox's own 10.7 s to the overage
when deciding whether an episode fits; the per-turn figures scale by roughly
the same ratio.

The overage arithmetic mirrors ``kaggle_environments``' own, in ``core.py`` and
``agent.py``: a turn consumes ``max(0, duration - actTimeout)`` from the pool,
and the agent is disqualified the moment a turn's overrun exceeds what remains.
"""

import argparse
import json
import logging
import os
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

LOGGER = logging.getLogger(__name__)

# One episode against the environment's own baseline. A seeded head-to-head is
# enough: the forward pass costs the same on every board, and this measures a
# clock, not a score.
OPPONENT = "starter"
SEED = 42

# The competition's own numbers, restated here because this script has to do
# the engine's arithmetic to say what is left. `kaggriculture.json` sets
# `actTimeout` to 1 and `remainingOverageTime` to 60.
ACT_TIMEOUT = 1.0
OVERAGE_POOL = 60.0


def main() -> None:
    """Measure the packaged agent, or -- given a root -- be the measured child.

    Called without arguments this builds the archive, unpacks it and re-runs
    itself against the unpacked copy in a fresh interpreter. Called with that
    root it is the child, and does the measuring. One file rather than two
    because the two halves are one measurement, and a child that drifted from
    its parent would report a budget for code nobody ships.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", nargs="?", type=Path, help="unpacked archive to play from"
    )
    args = parser.parse_args()
    if args.root is None:
        report(measure_packaged_agent())
    else:
        sys.stdout.write(json.dumps(play(args.root)))


def measure_packaged_agent() -> dict[str, Any]:
    """Build the archive, unpack it, and play one episode out of it.

    Returns:
        The child's timings, as ``play`` returns them.
    """
    from kaggriculture.scripts.package import build

    with tempfile.TemporaryDirectory() as staging:
        root = Path(staging)
        archive = build(root / "submission.tar.gz")
        unpacked = root / "agent"
        with tarfile.open(archive) as tar:
            tar.extractall(unpacked, filter="data")
        LOGGER.info(
            "playing from %s (%.1f MiB)", archive, archive.stat().st_size / 2**20
        )
        return json.loads(child_output(unpacked))


def child_output(unpacked: Path) -> str:
    """Return the timings a fresh interpreter reports for the unpacked agent.

    ``PYTHONPATH`` precedes the ``.pth`` that puts this repo's ``src`` on the
    path, so ``kaggriculture`` resolves inside the archive. The child asserts
    that rather than trusting it.

    Args:
        unpacked: The extracted archive root.

    Returns:
        The child's stdout, a JSON object.
    """
    environment = dict(os.environ, PYTHONPATH=str(unpacked))
    finished = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(unpacked)],
        capture_output=True,
        text=True,
        check=True,
        env=environment,
    )
    sys.stderr.write(finished.stderr)
    return finished.stdout


def play(unpacked: Path) -> dict[str, Any]:
    """Play one episode from ``unpacked`` and return what each turn cost.

    The agent is imported on the first turn rather than at module scope,
    because that is when the sandbox pays for it: ``kaggle_environments``
    compiles and execs an agent file the first time it is asked to act, so the
    import lands inside turn zero's duration and comes out of the overage pool.

    Args:
        unpacked: The extracted archive root, which must already be on the
            import path.

    Returns:
        The import cost and one duration per turn.

    Raises:
        RuntimeError: If ``kaggriculture`` resolved to somewhere other than the
            unpacked archive, which would measure the repo instead.
    """
    from kaggle_environments import make

    import kaggriculture
    from kaggriculture.constants import ENVIRONMENT, EPISODE_STEPS

    source = Path(kaggriculture.__file__).resolve().parent
    if source != unpacked.resolve() / "kaggriculture":
        raise RuntimeError(f"imported {source}, not the unpacked archive at {unpacked}")

    durations: list[float] = []
    imported = 0.0
    played: Optional[Callable[[Mapping[str, Any]], dict[str, Any]]] = None

    def timed(observation: Mapping[str, Any]) -> dict[str, Any]:
        """Play one turn, importing the agent on the first, and time it."""
        nonlocal imported, played
        start = time.perf_counter()
        if played is None:
            from kaggriculture.learn.play import agent

            imported = time.perf_counter() - start
            played = agent
        action = played(observation)
        durations.append(time.perf_counter() - start)
        return action

    environment = make(
        ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS, "seed": SEED}
    )
    environment.run([timed, OPPONENT])
    return {"import": imported, "durations": durations}


def report(timings: dict[str, Any]) -> None:
    """Log the budget, and whether the episode fits inside the pool.

    Args:
        timings: What ``play`` returned, as parsed from the child.
    """
    durations = timings["durations"]
    overages = [max(0.0, duration - ACT_TIMEOUT) for duration in durations]
    consumed = sum(overages)
    LOGGER.info("torch import        %7.3f s (inside turn 0)", timings["import"])
    LOGGER.info("turn 0              %7.3f s", durations[0])
    tail = sorted(durations[1:])
    LOGGER.info(
        "turns 1+            mean %6.4f s, median %6.4f s, p99 %6.4f s, max %6.4f s",
        statistics.fmean(tail),
        statistics.median(tail),
        tail[int(0.99 * len(tail))],
        tail[-1],
    )
    LOGGER.info(
        "turns over %.0fs        %d of %d",
        ACT_TIMEOUT,
        sum(over > 0 for over in overages),
        len(durations),
    )
    LOGGER.info(
        "overage consumed    %7.3f s of %.0f s, %.3f s left",
        consumed,
        OVERAGE_POOL,
        OVERAGE_POOL - consumed,
    )
    LOGGER.info(
        "verdict             %s",
        "fits" if consumed < OVERAGE_POOL else "DOES NOT FIT - not submittable",
    )
    LOGGER.info(
        "wall clock          %7.1f s over %d turns", sum(durations), len(durations)
    )


if __name__ == "__main__":
    main()
