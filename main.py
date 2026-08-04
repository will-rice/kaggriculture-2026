"""Competition entrypoint: Kaggle loads this file and calls ``agent``.

The loader ``exec``s this file with an empty globals dict and takes the *last*
callable defined in it, so ``agent`` must stay the final import — and nothing
here may reference ``__file__``, which is not defined under that exec.

The submission tarball places this file next to the ``kaggriculture`` package,
and the loader puts that directory on ``sys.path`` while this module executes,
so the import below resolves both locally and on the competition runner.

The agent served here is the vendored economic policy, not our own heuristic.
Measured over 20 seeded games it beat the heuristic 20-0 at 148k to 51k, and it
prices from the live market curve and the opponent's visible supply rather than
from the base table. The heuristic is frozen at ``baselines/heuristic_v2.py``
and remains a league opponent.
"""

from kaggriculture.economic_policy import agent

__all__ = ["agent"]
