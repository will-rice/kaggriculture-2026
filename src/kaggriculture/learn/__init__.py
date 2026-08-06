"""Training-time code, plus the one inference module the submission ships.

The competition sandbox has two cores, 6.8 GB and no network, and importing
torch there costs 10.7 seconds of a 60-second overage pool. `package.py` used
to exclude this whole package from the submission archive for that reason.

That stopped being true once the policy grew a market head. A cloned agent with
no market verbs cannot gain a coin — the engine credits money in exactly one
place, a completed SELL — so it banks its opening 3,000 and loses to anything
that trades. Playing the checkpoint is therefore the only way this project's
learned policy can score at all, and `learn/play.py`, `learn/model.py`,
`learn/encoding.py` and the checkpoint beside this file now ship. What still
never ships is `learn/scripts/`, `corpus.py` and `dataset.py`: they import
wandb, tqdm and a `/data` path, none of which exist in the sandbox.

The 10.7 seconds are still spent, on the first turn, out of the overage pool.
`scripts/budget.py` measures whether a full 720-turn episode fits inside
what is left.

``CHECKPOINT`` lives here rather than in `play.py` or `train.py` so that the
module that writes the weights, the module that reads them and the module that
packages them all name the same path, and `package.py` can ask where the
weights are without importing torch to find out.
"""

from pathlib import Path

# Beside this file, so one path serves both the local league and the unpacked
# archive: the agent imports `kaggriculture.learn.play` in each case, and
# `__file__` resolves to whichever tree it was imported from. A path under
# `/data` would exist only on the workstation, and the submitted agent would
# load nothing and play untrained with no error to say so.
CHECKPOINT = Path(__file__).parent / "policy.pt"
