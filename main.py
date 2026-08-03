"""Competition entrypoint: Kaggle loads this file and calls ``agent``.

The loader ``exec``s this file with an empty globals dict and takes the *last*
callable defined in it, so ``agent`` must stay the final import — and nothing
here may reference ``__file__``, which is not defined under that exec.

The submission tarball places this file next to the ``kaggriculture`` package,
and the loader puts that directory on ``sys.path`` while this module executes,
so the import below resolves both locally and on the competition runner.
"""

from kaggriculture.policy import agent

__all__ = ["agent"]
