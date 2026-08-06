"""Measure a full episode of the packaged agent against the sandbox's clock.

The competition gives an agent one second a turn (``actTimeout``) and a
60-second pool (``remainingOverageTime``) to cover every turn that overruns it.
Whatever an agent pays to start up -- importing torch, decoding a 3.6 MiB route
store -- is paid inside turn zero, because ``kaggle_environments`` compiles and
execs an agent file lazily on its first call, and it comes out of that pool. So
the question this answers is not whether one turn is fast: it is whether a
whole season plus the startup fits in what is left. If it does not, the agent
is not submittable at all, whatever it scores locally.

Two things make the number honest rather than flattering:

* The episode runs from the **built archive**, extracted to a temporary
  directory that goes on ``sys.path`` ahead of everything else, in a fresh
  interpreter. Measuring the repo would measure a tree with the training
  scripts, ``/data`` and wandb in it, which is not what plays.
* By default it plays the archive's own ``main.py``, loaded the way the runner
  loads it, so the thing measured is the thing submitted rather than a module
  somebody believed ``main.py`` served. ``--module`` measures a candidate the
  entrypoint has not been repointed at yet.

One thing it cannot make honest: this is still a workstation. The Phase 0
sandbox probe measured the torch import at **10.7 s** there against ~1 s here,
and ~20-26 GMAC/s at two threads against this machine's ~90. Read a torch
agent's import figure as a lower bound and add the sandbox's own 10.7 s to the
overage; the per-turn figures scale by roughly the same ratio. A pure-Python
agent has no such multiplier to apply and no thread count to set -- it takes
one core whatever the sandbox has -- but its startup is disk and CPU on a
machine slower than this one, so read that as a lower bound too.

The overage arithmetic mirrors ``kaggle_environments``' own, in ``core.py`` and
``agent.py``: a turn consumes ``max(0, duration - actTimeout)`` from the pool,
and the agent is disqualified the moment a turn's overrun exceeds what remains.
"""

import argparse
import importlib
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
    parser.add_argument(
        "--module",
        default=None,
        help="dotted module whose `agent` to measure, instead of main.py",
    )
    args = parser.parse_args()
    if args.root is None:
        report(measure_packaged_agent(args.module))
    else:
        sys.stdout.write(json.dumps(play(args.root, args.module)))


def measure_packaged_agent(module: Optional[str] = None) -> dict[str, Any]:
    """Build the archive, unpack it, and play one episode out of it.

    Args:
        module: Dotted module whose ``agent`` to measure, or ``None`` for the
            archive's ``main.py``.

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
            "playing %s from %s (%.1f MiB)",
            module or "main.py",
            archive,
            archive.stat().st_size / 2**20,
        )
        return json.loads(child_output(unpacked, module))


def child_output(unpacked: Path, module: Optional[str]) -> str:
    """Return the timings a fresh interpreter reports for the unpacked agent.

    ``PYTHONPATH`` precedes the ``.pth`` that puts this repo's ``src`` on the
    path, so ``kaggriculture`` resolves inside the archive. The child asserts
    that rather than trusting it.

    Args:
        unpacked: The extracted archive root.
        module: Dotted module to measure, or ``None`` for ``main.py``.

    Returns:
        The child's stdout, a JSON object.
    """
    environment = dict(os.environ, PYTHONPATH=str(unpacked))
    finished = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(unpacked)]
        + ([] if module is None else ["--module", module]),
        capture_output=True,
        text=True,
        check=True,
        env=environment,
    )
    sys.stderr.write(finished.stderr)
    return finished.stdout


def load_agent(unpacked: Path, module: Optional[str]) -> Callable[..., dict[str, Any]]:
    """Return the agent callable to time, loaded the way the runner loads one.

    With no ``module`` this reads the archive's ``main.py`` and takes the last
    callable defined in it, which is exactly what ``kaggle_environments`` does
    with a submitted entrypoint -- so the measurement follows the entrypoint
    wherever it is pointed instead of naming a module somebody believed it
    served. A ``module`` names a candidate directly, for measuring an agent
    before deciding whether to repoint ``main.py`` at it.

    Args:
        unpacked: The extracted archive root.
        module: Dotted module whose ``agent`` to load, or ``None``.

    Returns:
        The agent callable.
    """
    if module is not None:
        return importlib.import_module(module).agent
    from kaggle_environments.agent import get_last_callable

    entrypoint = unpacked / "main.py"
    return get_last_callable(entrypoint.read_text(), path=str(entrypoint))


def play(unpacked: Path, module: Optional[str] = None) -> dict[str, Any]:
    """Play one episode from ``unpacked`` and return what each turn cost.

    The agent is imported on the first turn rather than at module scope,
    because that is when the sandbox pays for it: ``kaggle_environments``
    compiles and execs an agent file the first time it is asked to act, so the
    import lands inside turn zero's duration and comes out of the overage pool.

    Args:
        unpacked: The extracted archive root, which must already be on the
            import path.
        module: Dotted module whose ``agent`` to play, or ``None`` for the
            archive's ``main.py``.

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
    played: Optional[Callable[..., dict[str, Any]]] = None

    def timed(observation: Mapping[str, Any]) -> dict[str, Any]:
        """Play one turn, importing the agent on the first, and time it."""
        nonlocal imported, played
        start = time.perf_counter()
        if played is None:
            played = load_agent(unpacked, module)
            imported = time.perf_counter() - start
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

    The import line and turn zero are reported separately because an agent can
    pay its startup in either: ``learn.play`` pays it at import, in torch, and
    ``routes.play`` pays it inside its first call, decoding the store behind an
    ``lru_cache``. Turn zero covers both, so it is the figure the pool is
    charged.

    Args:
        timings: What ``play`` returned, as parsed from the child.
    """
    durations = timings["durations"]
    overages = [max(0.0, duration - ACT_TIMEOUT) for duration in durations]
    consumed = sum(overages)
    LOGGER.info("agent import        %7.3f s (inside turn 0)", timings["import"])
    LOGGER.info("turn 0              %7.3f s (import plus any lazy load)", durations[0])
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
