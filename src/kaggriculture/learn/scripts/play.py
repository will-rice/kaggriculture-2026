"""Wrap a trained checkpoint as an agent the harness can evaluate.

Not a submission entrypoint. This imports torch, which the competition sandbox
pays 10.7 seconds to load, and lives under ``learn`` which ``package.py`` never
copies into the archive. It exists so a checkpoint can be measured against the
same frozen league every other agent in this project is measured against.

``CHECKPOINT`` is defined here rather than in the training script, and the
training script imports it from this module, so there is exactly one statement
of where the weights live. Two constants would let training write somewhere the
agent does not read, and the failure would look like an untrained model rather
than a missing file.

The whole file is arranged so ``agent`` is the last callable defined:
``kaggle_environments`` execs an agent path and takes the last callable in the
resulting namespace, so anything defined after ``agent`` would be played
instead of it.
"""

import functools
from typing import Any, Mapping

import torch

from kaggriculture.learn.encoding import (
    decode_units,
    encode_board,
    encode_positions,
    encode_scalars,
    unit_count,
)
from kaggriculture.learn.model import Policy
from kaggriculture.learn.scripts.build import SHARDS

CHECKPOINT = SHARDS / "policy.pt"

# The sandbox has two cores. Timing at 64 flatters it by an order of magnitude,
# and a league evaluation fans episodes out one per core, so a policy that
# helped itself to every thread would measure a machine the agent never gets.
THREADS = 2


@functools.lru_cache(maxsize=1)
def model() -> Policy:
    """Return the cloned policy, loaded once per process.

    Cached rather than reloaded per turn: the weights are ~40 MB and a 720-turn
    episode would otherwise spend the whole budget on ``torch.load``. The cache
    is a public, clearable one rather than a module global so a test can point
    ``CHECKPOINT`` at its own file and drop what a previous test loaded.

    Returns:
        The policy in eval mode, on the CPU.
    """
    torch.set_num_threads(THREADS)
    policy = Policy()
    policy.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    return policy.eval()


def agent(raw_obs: Mapping[str, Any]) -> dict[str, Any]:
    """Return this turn's action from the cloned policy.

    Every tensor handed to the model comes from the same observation, through
    the same encoders the dataset was built with, so play and training cannot
    disagree about what a row means. The unit count comes from
    ``unit_count`` -- the observation, not the action, is what the engine walks
    -- so the decoded action names exactly the units standing on the farm.

    The action carries no market orders: ``UNIT_OPS`` has none to decode. The
    cloned agent farms and never trades, which is a property of the action
    space rather than of the weights.

    Args:
        raw_obs: One turn's observation, as the environment hands it over.

    Returns:
        The action dict the environment consumes.
    """
    seat = int(raw_obs["player"])
    with torch.no_grad():
        logits = model()(
            encode_board(raw_obs, seat),
            encode_scalars(raw_obs, seat),
            encode_positions(raw_obs, seat),
        )
    return decode_units(logits, unit_count(raw_obs, seat))
