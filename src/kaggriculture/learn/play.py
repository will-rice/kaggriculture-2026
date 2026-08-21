"""Wrap a trained checkpoint as an agent, for the league and for the archive.

This is the one module under ``learn`` that ships. It imports torch, which the
competition sandbox pays 10.7 seconds of its 60-second overage pool to load on
the first turn, so ``scripts/budget.py`` measures a full episode against
that pool rather than assuming it fits.

It deliberately imports nothing from ``learn.scripts``, ``learn.corpus`` or
``learn.dataset``: those reach for wandb, tqdm and a ``/data`` path, none of
which exist in the sandbox, and an import that reaches the network forfeits the
episode on turn zero. ``learn.mask`` is safe to import and is imported here:
it is pure torch over ``kaggriculture.constants``, ``kaggriculture.observation``
and ``learn.encoding``, all of which this module already loads, so it adds no
dependency the sandbox does not already have.

``CHECKPOINT`` comes from ``kaggriculture.learn`` so that training, play and
packaging share one statement of where the weights live; two constants would
let training write somewhere the agent does not read, and the failure would
look like an untrained model rather than a missing file.

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
from kaggriculture.learn.mask import market_mask, unit_mask
from kaggriculture.learn.model import Policy, load_policy_weights

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

    ``load_policy_weights`` does the loading; see it for what non-strict means
    here. It is not a promise that ``CHECKPOINT`` loads -- the quantity-lane
    widening resized ``trade_head``, an existing key, and ``CHECKPOINT`` on
    disk predates that widening, so this currently raises out of
    ``load_state_dict`` before ``load_policy_weights``'s own check ever runs.
    Retraining the clone is Phase 2's job; until then this loader's contract
    is correctness, not availability.

    Returns:
        The policy in eval mode, on the CPU.

    Raises:
        ValueError: If the checkpoint is missing or renaming anything besides
            the quantity head.
        RuntimeError: If a key present in both the checkpoint and the module
            has a shape ``load_state_dict`` cannot reconcile -- the case
            ``CHECKPOINT`` currently hits.
    """
    torch.set_num_threads(THREADS)
    policy = Policy()
    load_policy_weights(policy, torch.load(CHECKPOINT, map_location="cpu"))
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

    **Both heads are decoded under the same legality masks the policy was
    trained and gated under.** This path used to argmax the raw logits, which
    made the deployed agent a different agent than the measured one: it chose
    by a rule the training distribution never used, and it could name ops the
    engine silently discards -- ``_apply_unit_action`` returns without a word
    and the unit has spent its turn. Nothing raises, nothing logs, and the only
    symptom is a season that banks less than the gate said it would. The masks
    are built here, from this turn's observation, exactly as ``rollout`` builds
    them, so play and training cannot disagree about which ops exist.

    Selection stays an ``argmax``, where ``rollout`` samples: deployment is
    deterministic, so a league result is reproducible from the seed alone.

    The masks cost ~1.1 ms a turn, measured, against a one-second turn budget --
    see ``scripts/budget.py`` for the whole-episode figure.

    Args:
        raw_obs: One turn's observation, as the environment hands it over.

    Returns:
        The action dict the environment consumes, carrying one op per unit and
        this turn's market orders.
    """
    seat = int(raw_obs["player"])
    with torch.no_grad():
        # The quantity head is not decoded from yet -- PICKUP/PLACE still
        # carry no explicit quantity, as before this head existed.
        unit_logits, _unit_quantity_logits, market_logits, _value = model()(
            encode_board(raw_obs, seat),
            encode_scalars(raw_obs, seat),
            encode_positions(raw_obs, seat),
        )
    action = decode_units(
        unit_logits, unit_count(raw_obs, seat), unit_mask(raw_obs, seat)
    )
    action["market"] = decode_market(market_logits, market_mask(raw_obs, seat))
    return action
