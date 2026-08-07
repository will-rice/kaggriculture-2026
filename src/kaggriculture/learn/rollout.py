"""Play one episode with the policy in the loop and keep what PPO needs from it.

Everything here exists so that the distribution the update differentiates is the
distribution that actually acted. Four things have to line up for that, and
three of them fail silently if they do not.

**The mask is applied before the softmax, never after.** ``_sample`` sets the
masked logits to ``-inf`` and then takes ``log_softmax``, so the renormalisation
happens once, over the legal moves alone, and the number stored in
``Trajectory.log_probs`` is a log-probability of that distribution. Sampling
from the unmasked softmax and rejecting, or zeroing probabilities after
normalising, both produce actions that look identical in a replay and a stored
log-prob that belongs to some other distribution -- so PPO's importance ratio
``exp(new - old)`` is a ratio between two different measures and is wrong by a
factor of the mask's surviving probability mass, which varies turn to turn. No
gradient check catches that; the run simply learns more slowly than it should.

**The mask travels with the action.** The update re-applies a mask to recompute
``new``, and if it recomputes one from a re-encoded observation instead of
reading the one that acted, any drift between the two -- a mask function
changed between collecting and training, an off-by-one in slot order -- makes
``old`` and ``new`` incomparable while both remain finite. So the masks are
stored, at the shape of the logits they gate, and ``Trajectory.illegal`` audits
the *stored* pair rather than the live one: it re-derives whether each stored
action index is permitted by its stored mask row. A rollout that sampled
correctly but stored the wrong mask reports illegal actions even though the
episode was played legally, which is exactly the failure that needs a voice.

**The action is built by the same decoders the dataset uses.** ``decode_units``
and ``decode_market`` take logits and argmax them, so the sampled indices are
handed back as one-hot rows rather than being turned into engine ops by a second
copy of ``_op``'s spelling rules. ``PICKUP`` needs its item and its quantity,
``PLANT`` must be arity 2, ``BUY_LAND`` is a flag and ``HIRE`` repeats by count:
a private re-implementation of any of that drifts from ``encoding.py`` the
moment either side changes, and the engine's response to a malformed op is to
do nothing at all.

**The reward is the per-turn change in (our bank - their bank).** It is the win
condition rather than a proxy for it -- ``interpreter`` ends the episode by
setting each seat's reward to its own ``money``, and ``MatchTask.evaluate``
compares the two -- and because it is a difference of a difference it telescopes:
the 719 rewards sum to the terminal margin exactly, and both farms open on the
same ``startingMoney``, so no constant leaks in. That is what makes it dense
without being shaped. Our own bank change is *not* the same signal: an episode
where we bank 6,000 against an opponent's 20,000 is a loss, and a policy trained
on our bank alone would rate it a strong one.

The episode is driven a step at a time through ``Environment.step`` rather than
handed to ``Environment.run``. ``run`` reaches an agent only as a callable it
times and sandboxes -- stdout captured, exceptions converted into the action,
``DeadlineExceeded`` substituted when the overage pool runs dry -- and every one
of those turns a bug in the policy into a short episode instead of a traceback.
Stepping directly also puts both seats' observations in hand at the moment of
the decision, which is what the bank differential is read from.

``env.steps`` is not read for that differential, and cannot be: ``step`` appends
``self.state`` without copying it and the interpreter mutates ``farms`` in
place, so every recorded step aliases the same farm dicts and the whole history
reads as the final position. The two banks are therefore snapshotted as floats
on the turn they are seen.
"""

import logging
from dataclasses import dataclass
from inspect import signature
from typing import Any, Callable, Mapping

import torch
from kaggle_environments import make
from kaggle_environments.agent import build_agent
from kaggle_environments.core import Environment

from kaggriculture.constants import ENVIRONMENT, EPISODE_STEPS
from kaggriculture.learn.encoding import (
    IGNORE,
    decode_market,
    decode_units,
    encode_board,
    encode_positions,
    encode_scalars,
    unit_count,
)
from kaggriculture.learn.mask import market_mask, unit_mask
from kaggriculture.learn.model import Policy

LOGGER = logging.getLogger(__name__)

# The learner always sits in seat 0 and the opponent in seat 1. Which seat that
# is does not reach the network: `encode_board` and `encode_scalars` take the
# seat and put that farm's planes first, so a policy trained from seat 0 reads
# seat 1 identically. What the seat does decide is whose `private` mapping is
# legible -- the engine hands each agent only its own -- which is why the mask
# and the encoders are called with this constant rather than with whatever seat
# an observation happens to carry.
LEARNER, OPPONENT = 0, 1


@dataclass(frozen=True)
class Trajectory:
    """One episode's worth of what PPO reads, batched along time.

    Every tensor's first dimension is the turn, and every turn is a state the
    policy actually acted from, so an index selects one decision across all of
    them. ``rewards[t]`` is what followed the action in ``unit_actions[t]`` and
    ``market_actions[t]``, taken from ``board[t]``.

    Attributes:
        board: ``(turns, TILE_PLANES, BOARD, BOARD)`` planes.
        scalars: ``(turns, SCALARS)`` market and phase features.
        positions: ``(turns, MAX_UNITS)`` flattened tile indices per unit.
        unit_actions: ``(turns, MAX_UNITS)`` sampled op indices, ``IGNORE``
            in the slots no unit was standing in. That padding is the update's
            only statement of which slots were real, and it is the same
            sentinel ``encode_units`` writes, so the behaviour-cloning loss and
            the PPO loss mask the same slots the same way.
        market_actions: ``(turns, len(MARKET_SLOTS) + 2)`` sampled quantity
            buckets. No slot is ever padding here -- bucket 0 is "trade
            nothing", a decision the engine acts on by emitting no order.
        unit_masks: ``(turns, MAX_UNITS, len(UNIT_OPS))`` bool, the mask the
            unit logits were gated by.
        market_masks: ``(turns, len(MARKET_SLOTS) + 2, len(QUANTITIES))`` bool.
        log_probs: ``(turns,)`` joint log-probability of the whole turn --
            every real unit's op plus every market slot's bucket -- under the
            masked distribution it was sampled from. Joint rather than
            per-head because PPO's ratio is over the action the episode took,
            and a turn's action is all of it at once.
        values: ``(turns,)`` the value head's estimate at that state.
        rewards: ``(turns,)`` change in (our bank - their bank) across the
            turn. Sums to ``final_margin``.
        dones: ``(turns,)`` bool, True on the last turn alone. A rollout is
            one whole episode, so GAE bootstraps from nothing at the end.
        final_margin: Our terminal bank minus theirs, read from the engine's
            own end-of-episode rewards. Positive is a win.
        illegal: How many stored action indices their stored mask forbids.
            Zero unless the sampling or the storing is broken; see the module
            docstring.
    """

    board: torch.Tensor
    scalars: torch.Tensor
    positions: torch.Tensor
    unit_actions: torch.Tensor
    market_actions: torch.Tensor
    unit_masks: torch.Tensor
    market_masks: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    final_margin: float
    illegal: int


@dataclass(frozen=True)
class Turn:
    """One decision: the tensors it was made from, and the action it became.

    Internal to this module. It exists so that the learner and a policy
    opponent go through one code path -- the opponent reads ``action`` and
    drops the rest -- rather than two implementations of masked sampling that
    could disagree about which distribution a policy plays from.
    """

    action: dict[str, Any]
    board: torch.Tensor
    scalars: torch.Tensor
    positions: torch.Tensor
    units: torch.Tensor
    market: torch.Tensor
    unit_mask: torch.Tensor
    market_mask: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor


def rollout(policy: Policy, opponent: Policy | str, seed: int) -> Trajectory:
    """Play one episode from seat 0 and return everything the update needs.

    Args:
        policy: The network being trained. Played by sampling, not by argmax:
            PPO's ratio is only defined against a stochastic behaviour policy,
            and a greedy rollout explores nothing.
        opponent: Either another ``Policy`` -- self-play, sampled the same way
            from its own masked logits -- or an agent spec the environment can
            name: a built-in like ``"starter"`` or a path to a Python file.
        seed: The episode seed, which fixes the weed spawns and the town's
            shop unlock order. The action sampling is seeded from it too, so a
            rollout is reproducible given the weights.

    Returns:
        The episode's ``Trajectory``, 719 turns long -- ``episodeSteps`` is
        720 and the engine marks the season done on the last one, so there are
        720 recorded states and 719 decisions between them.
    """
    generator = torch.Generator().manual_seed(seed)
    environment = make(
        ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS, "seed": seed}
    )
    environment.reset(2)
    act = _opponent_actor(opponent, environment, generator)

    turns: list[Turn] = []
    margins: list[float] = []
    while not environment.done:
        observation = environment.state[LEARNER].observation
        margins.append(_margin(observation))
        turn = _decide(policy, observation, LEARNER, generator)
        turns.append(turn)
        environment.step([turn.action, act(environment.state[OPPONENT].observation)])
    margins.append(_margin(environment.state[LEARNER].observation))

    LOGGER.info("rollout of %d turns, margin %.0f", len(turns), margins[-1])
    return _trajectory(turns, margins)


def _margin(observation: Mapping[str, Any]) -> float:
    """Return our bank minus theirs, as a number rather than a live reference.

    Read now and kept, because ``Environment.step`` appends the state object
    itself and the interpreter mutates the farms in place: the recorded history
    aliases one set of dicts, so asking ``env.steps[t]`` for turn ``t``'s money
    after the fact returns the terminal money for every ``t``.

    Both farms are public, so this is knowable from either seat's observation.

    Args:
        observation: One turn's observation.

    Returns:
        ``farms[0]["money"] - farms[1]["money"]``.
    """
    farms = observation["farms"]
    return float(farms[LEARNER]["money"]) - float(farms[OPPONENT]["money"])


def _opponent_actor(
    opponent: Policy | str, environment: Environment, generator: torch.Generator
) -> Callable[[Mapping[str, Any]], Any]:
    """Return the function that turns seat 1's observation into its action.

    A ``Policy`` opponent goes through ``_decide``, so self-play has both seats
    sampling from masked logits by the same code. Anything else is a spec the
    environment knows how to build: a built-in name, or a path whose last
    callable is the agent.

    ``build_agent`` is used rather than ``kaggle_environments.agent.Agent``
    because ``Agent.act`` converts an exception into the returned action and
    lets ``Environment.step`` mark the seat ERRORed. During an evaluation that
    is the right behaviour; during training it turns a broken opponent into a
    quietly short episode, and a short episode is a trajectory that trains on
    less of the season without saying so. Calling the built agent directly
    lets the traceback out.

    The built agent's arity is not fixed: a built-in takes the observation
    alone and ``build_agent``'s file wrapper takes the configuration too, so
    the argument list is trimmed to what the callable accepts, the way
    ``Agent.act`` trims it. ``inspect.signature`` rather than ``Agent.act``'s
    own ``__code__.co_argcount``, which is an attribute of plain functions and
    not of callables in general.

    Args:
        opponent: A policy, or a spec the environment can build.
        environment: The episode, for its agent registry and configuration.
        generator: The sampling stream, shared with the learner so one seed
            reproduces the whole episode.

    Returns:
        A callable from seat 1's observation to seat 1's action.
    """
    if isinstance(opponent, Policy):
        return lambda observation: (
            _decide(opponent, observation, OPPONENT, generator).action
        )

    agent, _parallelizable = build_agent(opponent, environment.agents, environment.name)
    arguments = len(signature(agent).parameters)
    return lambda observation: agent(
        *[observation, environment.configuration][:arguments]
    )


def _decide(
    policy: Policy,
    observation: Mapping[str, Any],
    seat: int,
    generator: torch.Generator,
) -> Turn:
    """Sample one turn's action from the masked heads.

    The observation is encoded once and every consumer -- both heads, both
    masks, the value estimate -- reads that one encoding, so the state stored
    in the trajectory is provably the state the action was chosen from.

    Ops for slots past the crew are sampled anyway, because a row of the unit
    head exists for every slot and ``masked_fill`` would leave an all-``-inf``
    row for the padded ones if the mask did not keep ``PASS`` alive there. They
    are then overwritten with ``IGNORE`` and contribute nothing to the joint
    log-probability: the engine never reads them -- ``decode_units`` emits ops
    only for the units on the board -- so an update that scored them would be
    fitting a choice that was never played.

    Args:
        policy: The network to sample from.
        observation: That seat's own observation, whose ``private`` mapping is
            the only one legible to it.
        seat: Which seat is acting.
        generator: The sampling stream.

    Returns:
        The ``Turn``.
    """
    board = encode_board(observation, seat)
    scalars = encode_scalars(observation, seat)
    positions = encode_positions(observation, seat)
    units = unit_mask(observation, seat)
    trades = market_mask(observation, seat)

    with torch.no_grad():
        unit_logits, market_logits, value = policy(board, scalars, positions)
    chosen_units, unit_log = _sample(unit_logits, units, generator)
    chosen_market, market_log = _sample(market_logits, trades, generator)

    count = unit_count(observation, seat)
    action = decode_units(_one_hot(chosen_units, unit_logits.shape[-1]), count)
    action["market"] = decode_market(_one_hot(chosen_market, market_logits.shape[-1]))

    return Turn(
        action=action,
        board=board,
        scalars=scalars,
        positions=positions,
        units=torch.cat(
            [chosen_units[:, :count], torch.full_like(chosen_units[:, count:], IGNORE)],
            dim=1,
        ),
        market=chosen_market,
        unit_mask=units,
        market_mask=trades,
        log_prob=unit_log[:, :count].sum(dim=1) + market_log.sum(dim=1),
        value=value,
    )


def _sample(
    logits: torch.Tensor, mask: torch.Tensor, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one index per row and its log-probability, sampled under the mask.

    The mask is applied to the logits and the softmax is taken afterwards, so
    the returned log-probability is the log-probability of the distribution the
    index was drawn from. Normalising first and zeroing after would leave a
    sample that is legal and a log-probability that is too small by
    ``log`` of the legal mass, which is a different number every turn.

    ``masked_fill`` with ``-inf`` is safe here only because no row of either
    mask is ever entirely False -- ``unit_mask`` keeps ``PASS`` on every slot
    and ``market_mask`` keeps bucket 0 on every slot. An all-``-inf`` row
    softmaxes to NaN and ``multinomial`` then draws from nothing.

    Args:
        logits: ``(1, slots, options)`` head output.
        mask: ``(1, slots, options)`` bool, True where the option is legal.
        generator: The sampling stream.

    Returns:
        ``(1, slots)`` int64 indices and ``(1, slots)`` float log-probabilities.
    """
    log_probs = torch.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)
    rows = log_probs.flatten(0, -2)
    chosen = torch.multinomial(rows.exp(), 1, generator=generator)
    return (
        chosen.reshape(logits.shape[:-1]),
        rows.gather(1, chosen).reshape(logits.shape[:-1]),
    )


def _one_hot(chosen: torch.Tensor, options: int) -> torch.Tensor:
    """Return sampled indices as logits the encoding module's decoders can read.

    ``decode_units`` and ``decode_market`` argmax their input, so a one-hot row
    decodes to exactly the index that was sampled. Handing the sample back
    through them, rather than spelling the engine ops out here, is what keeps
    the arity of a ``PICKUP``, the item on a ``PLANT`` and the repetition of a
    ``HIRE`` defined in one place.

    Args:
        chosen: ``(1, slots)`` int64 indices.
        options: How wide the head is.

    Returns:
        A ``(1, slots, options)`` float tensor.
    """
    return torch.nn.functional.one_hot(chosen, options).float()


def _trajectory(turns: list[Turn], margins: list[float]) -> Trajectory:
    """Stack the episode's turns into the tensors the update reads.

    ``margins`` is one longer than ``turns``: it holds the bank differential
    before every decision and once more after the last one, so the differences
    are exactly one per decision and telescope to the terminal margin.

    Args:
        turns: Every decision, in order.
        margins: The bank differential at each state, including the terminal.

    Returns:
        The ``Trajectory``.
    """
    dones = torch.zeros(len(turns), dtype=torch.bool)
    dones[-1] = True
    unit_actions = torch.cat([turn.units for turn in turns])
    market_actions = torch.cat([turn.market for turn in turns])
    unit_masks = torch.cat([turn.unit_mask for turn in turns])
    market_masks = torch.cat([turn.market_mask for turn in turns])
    return Trajectory(
        board=torch.cat([turn.board for turn in turns]),
        scalars=torch.cat([turn.scalars for turn in turns]),
        positions=torch.cat([turn.positions for turn in turns]),
        unit_actions=unit_actions,
        market_actions=market_actions,
        unit_masks=unit_masks,
        market_masks=market_masks,
        log_probs=torch.cat([turn.log_prob for turn in turns]),
        values=torch.cat([turn.value for turn in turns]),
        rewards=torch.tensor(
            [
                after - before
                for before, after in zip(margins[:-1], margins[1:], strict=True)
            ],
            dtype=torch.float32,
        ),
        dones=dones,
        final_margin=margins[-1],
        illegal=(
            _illegal(unit_actions, unit_masks) + _illegal(market_actions, market_masks)
        ),
    )


def _illegal(actions: torch.Tensor, masks: torch.Tensor) -> int:
    """Return how many stored actions their stored mask forbids.

    Audits the pair the update will actually use, not the pair the sampler
    used, so it discriminates two different bugs with one number: sampling from
    an unnormalised or unmasked distribution, and storing a mask that does not
    belong to the action beside it.

    ``IGNORE`` marks a slot no unit stood in. Those slots hold no decision, so
    they are excluded rather than clamped into a legal-looking index -- and
    excluding them is why the market actions, which are never ``IGNORE``, can
    go through the same function.

    Args:
        actions: ``(turns, slots)`` int64 chosen indices, possibly ``IGNORE``.
        masks: ``(turns, slots, options)`` bool.

    Returns:
        The count of real slots whose chosen option is masked out.
    """
    real = actions != IGNORE
    permitted = masks.gather(2, actions.clamp(min=0)[:, :, None]).squeeze(-1)
    return int((real & ~permitted).sum().item())
