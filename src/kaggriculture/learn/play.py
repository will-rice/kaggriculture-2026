"""Wrap a trained checkpoint as an agent, for the league and for the archive.

This is the one module under ``learn`` that ships. It imports torch, which the
competition sandbox pays 10.7 seconds of its 60-second overage pool to load on
the first turn, so ``scripts/budget.py`` measures a full episode against
that pool rather than assuming it fits.

It deliberately imports nothing from ``learn.scripts``, ``learn.corpus`` or
``learn.dataset``: those reach for wandb, tqdm and a ``/data`` path, none of
which exist in the sandbox, and an import that reaches the network forfeits the
episode on turn zero. ``CHECKPOINT`` comes from ``kaggriculture.learn`` so that
training, play and packaging share one statement of where the weights live;
two constants would let training write somewhere the agent does not read, and
the failure would look like an untrained model rather than a missing file.

The whole file is arranged so ``agent`` is the last callable defined:
``kaggle_environments`` execs an agent path and takes the last callable in the
resulting namespace, so anything defined after ``agent`` would be played
instead of it.
"""

import functools
from typing import Any, Mapping

import torch

from kaggriculture.learn import CHECKPOINT
from kaggriculture.learn.encoding import (
    decode_market,
    decode_units,
    encode_board,
    encode_positions,
    encode_scalars,
    unit_count,
)
from kaggriculture.learn.model import Policy

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

    ``CHECKPOINT`` was behaviour-cloned before the value head existed, so it
    carries every trunk and head key but no ``value.*``. Loading it strict
    would refuse to load at all, so this loads non-strict instead -- but
    ``strict=False`` alone proves nothing: it would load just as "successfully"
    if a trunk key had also gone missing, or if a key had silently drifted
    name, leaving that part of the network randomly initialised while every
    other check keeps passing. So the missing keys are checked explicitly:
    every one of them must belong to the value head, and any gap outside it
    -- a missing trunk key, an unexpected one -- raises rather than playing
    weights that loaded silently wrong.

    Returns:
        The policy in eval mode, on the CPU.

    Raises:
        ValueError: If the checkpoint is missing or renaming anything besides
            the value head.
    """
    torch.set_num_threads(THREADS)
    policy = Policy()
    result = policy.load_state_dict(
        torch.load(CHECKPOINT, map_location="cpu"), strict=False
    )
    value_keys = {f"value.{name}" for name, _ in policy.value.named_parameters()}
    if not set(result.missing_keys) <= value_keys or result.unexpected_keys:
        raise ValueError(
            f"{CHECKPOINT} does not match the current trunk: missing "
            f"{result.missing_keys}, unexpected {result.unexpected_keys}"
        )
    return policy.eval()


def agent(raw_obs: Mapping[str, Any]) -> dict[str, Any]:
    """Return this turn's action from the cloned policy.

    Every tensor handed to the model comes from the same observation, through
    the same encoders the dataset was built with, so play and training cannot
    disagree about what a row means. The unit count comes from ``unit_count``
    -- the observation, not the action, is what the engine walks -- so the
    decoded action names exactly the units standing on the farm.

    Both heads are decoded. Decoding only the first is not a smaller version of
    this agent, it is a different one that cannot score: the engine increases a
    farm's money in exactly one place, crediting a completed ``SELL``, so an
    action with an empty market list banks the opening 3,000 and nothing more,
    whatever the unit head predicts. That was measured over 400 episodes before
    the market head existed.

    Args:
        raw_obs: One turn's observation, as the environment hands it over.

    Returns:
        The action dict the environment consumes, carrying one op per unit and
        this turn's market orders.
    """
    seat = int(raw_obs["player"])
    with torch.no_grad():
        unit_logits, market_logits, _value = model()(
            encode_board(raw_obs, seat),
            encode_scalars(raw_obs, seat),
            encode_positions(raw_obs, seat),
        )
    action = decode_units(unit_logits, unit_count(raw_obs, seat))
    action["market"] = decode_market(market_logits)
    return action
