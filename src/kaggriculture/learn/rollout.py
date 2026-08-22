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
and ``decode_market`` take logits and a mask and argmax them under it, so the
sampled indices are handed back as one-hot rows -- alongside the very masks
they were sampled under -- rather than being turned into engine ops by a second
copy of ``_op``'s spelling rules. The mask is redundant on this path and passed
anyway: an index sampled under a mask is legal under it, so the masked argmax
of its one-hot row is that same index. It is required by the signature because
the *deployment* path, ``learn.play``, has no sampler to make it redundant, and
an optional mask there is what let deployment argmax raw logits and quietly
play a different agent than the one measured here.

``PICKUP`` needs its item and its quantity, ``PLANT`` must be arity 2,
``BUY_LAND`` is a flag and ``HIRE`` repeats by count: a private
re-implementation of any of that drifts from ``encoding.py`` the moment either
side changes, and the engine's response to a malformed op is to do nothing at
all.

**Three series are recorded, and this module does not choose between them.**
``rewards`` is the per-turn change in (our bank - their bank), ``own`` is the
per-turn change in our own bank alone, and ``potentials`` is what the farm was
holding on its way to the bank at each state, by component. All three are stored
on every trajectory; how they are mixed is a set of weights in ``PpoConfig`` and
a schedule in the training loop, because it is a property of *where a run is*
rather than of what an episode was.

``potentials`` is stored as the potentials themselves rather than as their
differences, and that is deliberate. The shaped reward is
``gamma * P(s') - P(s)``, and ``gamma`` belongs to the update, not to the
rollout: a trajectory collected today has to remain usable if the discount
changes, and a rollout that baked one in would hand PPO a shaping term that no
longer matches the objective it is discounting against.

Unlike the two money series, the potentials are recorded for the acting turns
only and the terminal state is *not* appended. It would be the natural thing to
append -- the other two series read their terminal value here, and a difference
needs a state on each side of it -- and it is the one thing that must not
happen. ``P`` at the state a trajectory stops in is the only endpoint of the
telescoped shaping the agent can choose, so a non-zero one pays for ending the
season holding stock: at ``gamma = 0.9995`` over 719 turns, a unit held from
turn 700 to the horizon keeps 99% of its value, and refusing to sell into a market
both farms have pushed below base would be shaped-optimal. So the terminal is a
hard zero, and it lives in ``progress.progress_reward`` -- which takes the
acting rows and appends the zero itself, so this module has nowhere to put a
terminal row even by accident.

The differential is the win condition rather than a proxy for it --
``interpreter`` ends the episode by setting each seat's reward to its own
``money``, and ``MatchTask.evaluate`` compares the two -- and because it is a
difference of a difference it telescopes: the 719 values sum to the terminal
margin exactly, and both farms open on the same ``startingMoney``, so no
constant leaks in. It is the right final objective and the wrong starting one.

It is the wrong starting one because a purely relative reward has a fixed point
that self-play falls straight into, and this one did. Two copies of a policy
that banks nothing bankrupt each other identically, so the differential is zero
on every turn of every episode and no gradient distinguishes any action from any
other. Measured here over 30 iterations and about 2,000 seasons: mean bank 0,
best episode 4 coins of a possible 200,000, advantage -0.0008 +- 0.0128, value
loss 0.0002 -- a critic that had correctly learned that everything is zero.

``own`` is what a run trains on until that stops being true. It telescopes to
``final_bank - STARTING_MONEY``, so its total over a season is just what the
farm made, and it is deliberately the thinnest possible shaping: it pays for
banking coins, which is the objective itself measured absolutely rather than
relatively, and it encodes no opinion about *how* to farm. Every single-box
winner this project reads from did some version of this and then switched --
Toad Brigade shaped for 20M steps, FLG for 65M before moving to a zero-sum
differential of the same construction as ours, and Frog Parade moved to sparse
win/loss "as soon as training was running stably".

Our own bank cannot replace the differential, which is why both are kept rather
than one being chosen here: an episode where we bank 6,000 against an
opponent's 20,000 is a loss, and a policy trained on ``own`` alone would rate it
a strong one.

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

**Episodes are driven in lockstep so that one forward serves all of them.**
``rollout_many`` is the primitive and ``rollout`` is the one-seed case of it.
Measured in Task 3, 78% of an episode's cost was the network: one forward per
environment per turn, at batch 1, of a 10M-parameter trunk. Stepping ``N``
environments together turns that into one forward of batch ``N`` -- the same
arithmetic, issued once -- which is what makes a GPU worth having here and what
moves the bottleneck onto the encoders, where it belongs. Nothing about *what*
is recorded changes: every row of the batch is encoded, masked, sampled and
stored by the same code that served a single environment, so the mask that
gated a logit is still the mask stored beside the action it produced.

**Both seats are recorded when, and only when, the opponent is the learner
itself.** A self-play episode samples 719 decisions on each side and throwing
one side away halves the data per unit of wall clock for nothing. But that is
free only while both seats are the *same weights*: PPO's ratio is
``exp(new - old)`` against the behaviour policy that acted, and a frozen pool
checkpoint's log-probabilities are not the learner's. So the rule is identity --
``opponent is policy`` -- rather than a flag a caller can set wrongly, and an
episode against anything else records seat 0 alone.
"""

import logging
from dataclasses import dataclass, field
from inspect import signature
from typing import Any, Callable, Mapping, Sequence

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
    transfer_slots,
    unit_count,
)
from kaggriculture.learn.mask import market_mask, unit_mask, unit_quantity_mask
from kaggriculture.learn.model import Policy
from kaggriculture.learn.progress import potential
from kaggriculture.learn.sales import buy_units, sale_metrics
from kaggriculture.learn.toad_reward import (
    Counts,
    counts,
    money_weight,
    rank,
    shaped,
)
from kaggriculture.learn.toad_reward import (
    margin as margin_reward,
)
from kaggriculture.learn.toad_reward import (
    sparse as sparse_reward,
)

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
        unit_quantities: ``(turns, MAX_UNITS)`` the bucket each unit's quantity
            head sampled, into ``QUANTITIES``. Every slot carries one, padded
            slots included, because the head emits a row for every slot and a
            masked-out row would softmax to NaN; the padding is stated once, on
            ``unit_actions``, and a slot spends its bucket only where the op
            beside it is a ``PICKUP`` or a ``PLACE``.
        market_actions: ``(turns, len(MARKET_SLOTS) + 2)`` sampled quantity
            buckets. No slot is ever padding here -- bucket 0 is "trade
            nothing", a decision the engine acts on by emitting no order.
        unit_masks: ``(turns, MAX_UNITS, len(UNIT_OPS))`` bool, the mask the
            unit logits were gated by.
        unit_quantity_masks: ``(turns, MAX_UNITS, len(QUANTITIES))`` bool, the
            mask the quantity logits were gated by.
        market_masks: ``(turns, len(MARKET_SLOTS) + 2, len(QUANTITIES))`` bool.
        log_probs: ``(turns,)`` joint log-probability of the whole turn --
            every real unit's op, every transferring unit's bucket, and every
            market slot's bucket -- under the masked distribution it was
            sampled from. Joint rather than per-head because PPO's ratio is
            over the action the episode took, and a turn's action is all of it
            at once.
        values: ``(turns,)`` the value head's estimate at that state.
        rewards: ``(turns,)`` change in (our bank - their bank) across the
            turn. Sums to ``final_margin`` exactly.
        shaped: ``(turns,)`` Toad's phase-1 shaped reward for this seat with
            the score constituent ablated -- their five Lux-specific weighted
            per-turn deltas plus the 10x terminal result, all through their /500
            normaliser. This is the ``--no-money`` control, not the faithful
            component set; see ``shaped_money``. Recorded alongside ``rewards``
            and ``own`` rather than instead of them, for the same reason those
            two coexist: which one a gradient sees is the learner's choice, and
            a trajectory collected once is then usable under any of them.
        margin: ``(turns,)`` the margin reward -- the win condition itself,
            decomposed onto the turns that produced it, plus a small signed
            own-bank term and Toad's terminal rank. See
            ``toad_reward.margin``. Recorded beside the others for the same
            reason they coexist: one episode, every reward, so an arm is a
            choice of field rather than a re-run.
        shaped_money: ``(turns,)`` the faithful component set: the same reward
            with the signed per-turn coin delta at ``MONEY_WEIGHT`` in the slot
            their ``city`` weight occupies. In Lux ``city`` *is* the score, so
            omitting our equivalent deletes their largest component rather than
            reproducing it -- see ``toad_reward.MONEY_WEIGHT``. Carried beside
            ``shaped`` so the ablation reads the same episode twice rather than
            playing it twice.
        sparse: ``(turns,)`` phases 2-5's ``GameResultReward`` -- +1 terminal
            for a win, -1 for a loss, 0 for a draw, zero on every other turn.
            Not a decomposition of the win condition the way ``margin`` is:
            nothing rides on the turns that produced it. See
            ``toad_reward.sparse``.
        own: ``(turns,)`` change in our own bank across the turn. Sums to
            ``final_bank - STARTING_MONEY`` exactly. Recorded alongside
            ``rewards`` rather than instead of it: which one a gradient sees is
            ``PpoConfig.differential``, and a trajectory collected under one
            setting is usable under the other because both were kept.
        potentials: ``(turns, len(POTENTIAL_COMPONENTS))`` coin value of
            everything the farm held on its way to the bank, one row per turn
            the policy acted from. Undifferenced, because the ``gamma`` in the
            shaped reward belongs to the update; and with **no terminal row**,
            because a non-zero potential at the state a season stops in pays
            the agent to end it holding stock. See this module's docstring and
            ``progress.progress_reward``.
        dones: ``(turns,)`` bool, True on the last turn alone. A rollout is
            one whole episode, so GAE bootstraps from nothing at the end.
        final_margin: This seat's terminal bank minus the other's. Positive is
            a win.
        final_bank: This seat's terminal bank on its own. Not derivable from
            ``final_margin``, and the loop logs both: a margin says who won and
            a bank says whether either farm produced anything, which is the
            number that separates "learned to outproduce" from "learned to
            neutralise".
        illegal: How many stored action indices their stored mask forbids.
            Zero unless the sampling or the storing is broken; see the module
            docstring.
        final_capital: Producing animals owned at the horizon -- the count the
            margin reward's absolute term is a difference of. Logged because
            "margin improves while capital and bank both collapse" is the
            pre-registered read that says that term is too weak, and it cannot
            be read off the reward total without it.
        sales: Completed clears over the episode, from ``sales.sale_metrics``.
        units_sold: Units those clears moved. The corpus says winners take
            +3.3% on price at FLAT volume, so this is the control on
            ``mean_sale_price`` rather than a target of its own.
        mean_sale_price: Coins per unit actually realised.
        realisation: Realised price over the market book, 1.0 being par.
        bought: Units purchases credited to the shed, from ``sales.buy_units``.
    """

    board: torch.Tensor
    scalars: torch.Tensor
    positions: torch.Tensor
    unit_actions: torch.Tensor
    unit_quantities: torch.Tensor
    market_actions: torch.Tensor
    unit_masks: torch.Tensor
    unit_quantity_masks: torch.Tensor
    market_masks: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    own: torch.Tensor
    shaped: torch.Tensor
    shaped_money: torch.Tensor
    margin: torch.Tensor
    sparse: torch.Tensor
    potentials: torch.Tensor
    dones: torch.Tensor
    final_margin: float
    final_bank: float
    final_capital: float
    illegal: int
    sales: float
    units_sold: float
    mean_sale_price: float
    realisation: float
    bought: float


@dataclass
class Stream:
    """One recorded seat of one environment, accumulating its own episode.

    Internal to this module. A lockstep group holds one of these per recorded
    seat, so a self-play episode holds two -- same environment, different seat,
    different sign of the same reward -- and an episode against a named agent
    holds one.

    Attributes:
        environment: Which environment of the group this reads.
        seat: Which seat it plays.
        turns: Every decision it made, in order.
        margins: Its bank differential before each decision.
        banks: Its own bank before each decision, read on the same turns as
            ``margins`` and never derived from it -- the differential loses
            which of the two farms the coins are on, which is the whole reason
            the reward needs both.
        counts: Toad's per-turn component counts for this seat, read on the
            same turns as ``margins``. See ``toad_reward.counts``.
        snapshots: The three things ``sales`` reads, copied out of each
            observation as it is seen. See ``_snapshot``.
        potentials: What its farm held on its way to the bank before each
            decision, by component, read on the same turns as the other two.
    """

    environment: int
    seat: int
    turns: list["Turn"] = field(default_factory=list)
    margins: list[float] = field(default_factory=list)
    banks: list[float] = field(default_factory=list)
    counts: list[Counts] = field(default_factory=list)
    snapshots: list[dict[str, Any]] = field(default_factory=list)
    potentials: list[list[float]] = field(default_factory=list)


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
    quantities: torch.Tensor
    market: torch.Tensor
    unit_mask: torch.Tensor
    quantity_mask: torch.Tensor
    market_mask: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor


def rollout(policy: Policy, opponent: Policy | str, seed: int) -> Trajectory:
    """Play one episode from seat 0 and return everything the update needs.

    The one-seed case of ``rollout_many``, and seat 0's trajectory out of it.
    There is no second implementation: a single episode is a lockstep group of
    one.

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
    return rollout_many(policy, opponent, (seed,))[0]


def rollout_many(
    policy: Policy, opponent: Policy | str, seeds: Sequence[int]
) -> list[Trajectory]:
    """Play a group of episodes in lockstep, one forward per turn for all of them.

    Every environment takes its turn at the same time, so the observations of
    the whole group are encoded into one batch and the trunk is evaluated once
    rather than ``len(seeds)`` times. That is the whole reason this function
    exists: Task 3 measured 78% of an episode's wall clock inside the network,
    at batch 1.

    Seat 1 is recorded as its own trajectory when ``opponent is policy`` and
    not otherwise. The reward is a differential, so seat 1's is the negation of
    seat 0's, and both are on-policy only while both seats carry the same
    weights. See the module docstring.

    Args:
        policy: The network being trained.
        opponent: Another ``Policy``, or a spec the environment can build.
        seeds: One seed per environment in the group. Each seeds its own
            episode; the first also seeds the group's shared sampling stream.

    Returns:
        The group's trajectories, environment-major and seat-minor, so seat 0
        of the first environment is first.

    Raises:
        ValueError: If the group's episodes do not all end on the same turn,
            which would leave the recorded streams ragged.
    """
    generator = torch.Generator().manual_seed(int(seeds[0]))
    environments = [
        make(ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS, "seed": seed})
        for seed in seeds
    ]
    for environment in environments:
        environment.reset(2)

    mirror = opponent is policy
    streams = [
        Stream(environment=index, seat=seat)
        for index in range(len(environments))
        for seat in ((LEARNER, OPPONENT) if mirror else (LEARNER,))
    ]
    actors = (
        ()
        if isinstance(opponent, Policy)
        else [_opponent_actor(opponent, environment) for environment in environments]
    )

    while not environments[0].done:
        for stream in streams:
            seen = _observation(environments, stream.environment, stream.seat)
            stream.margins.append(_margin(seen))
            stream.banks.append(_bank(seen))
            stream.counts.append(counts(seen, stream.seat))
            stream.snapshots.append(_snapshot(seen))
            stream.potentials.append(potential(seen))
        turns = _decide(
            policy,
            [
                (
                    _observation(environments, stream.environment, stream.seat),
                    stream.seat,
                )
                for stream in streams
            ],
            generator,
        )
        actions: list[list[Any]] = [[None, None] for _ in environments]
        for stream, turn in zip(streams, turns, strict=True):
            stream.turns.append(turn)
            actions[stream.environment][stream.seat] = turn.action
        for index, action in enumerate(
            _opponent_actions(policy, opponent, actors, environments, generator)
        ):
            actions[index][OPPONENT] = action
        for environment, pair in zip(environments, actions, strict=True):
            environment.step(pair)

    if not all(environment.done for environment in environments):
        raise ValueError(
            "the group's episodes ended on different turns, so their recorded "
            "streams are ragged"
        )

    LOGGER.info("rolled out %d episodes into %d trajectories", len(seeds), len(streams))
    return [_trajectory(stream, environments[stream.environment]) for stream in streams]


def _observation(
    environments: list[Environment], index: int, seat: int
) -> Mapping[str, Any]:
    """Return one seat's own observation, whose ``private`` mapping it alone sees."""
    return environments[index].state[seat].observation


def _agent_observation(environment: Environment, seat: int) -> Mapping[str, Any]:
    """Return the observation the engine would hand an agent standing in this seat.

    ``Environment.__get_state`` deletes every property the specification marks
    ``shared`` from the schema of every seat past the first, and only
    ``Environment.run`` puts them back -- it calls each agent with
    ``__get_shared_state(position)``, which refills the shared properties from
    seat 0. Driving the episode through ``Environment.step`` skips that, so
    ``state[1].observation`` is what the engine *stores*, not what an agent
    sees.

    For this game the difference is one key. The interpreter writes ``farms``,
    ``market``, ``town``, ``day`` and ``hour`` onto both seats itself, so they
    survive; ``step`` it does not, and seat 1's stored observation has no
    ``step`` at all. That is enough to destroy an agent that indexes a recorded
    route by it -- the vendored kaito agent banks 201,485 from either seat under
    ``run`` and 0 from seat 1 without this, silently, by replaying turn 0 for
    the whole season.

    Our own encoders are unaffected and do not go through here: they derive the
    step from ``day`` and ``hour`` precisely because seat 1 has no ``step``, and
    the masks read ``day``.

    The refill is driven off the specification rather than off a literal
    ``"step"``, so a shared property added to the game reaches opponents
    without this function being edited. It is shallow: the engine deep-copies,
    but the stored observation is already handed to opponents unowned today,
    and a per-turn deep copy of two farms of a hundred tiles would cost more
    than the encoders it is protecting.

    Args:
        environment: The episode.
        seat: Which seat's observation to build.

    Returns:
        That seat's observation with the shared properties refilled.
    """
    observation = dict(environment.state[seat].observation)
    for name, prop in environment.specification["observation"].items():
        if prop.get("shared"):
            observation[name] = environment.state[0].observation[name]
    return observation


def _opponent_actions(
    policy: Policy,
    opponent: Policy | str,
    actors: Sequence[Callable[[Mapping[str, Any]], Any]],
    environments: list[Environment],
    generator: torch.Generator,
) -> list[Any]:
    """Return seat 1's action for every environment in the group.

    Three cases. A mirror opponent's seat-1 action came out of the learner's own
    batched forward -- the same weights, sampled in the same call -- so there is
    nothing left to do and this returns nothing. A *distinct* ``Policy`` -- a
    frozen checkpoint from the pool -- gets its own batched forward, so the pool
    costs one extra forward for the group rather than one per environment.
    Anything else is a named agent, and goes through its own environment's actor.

    Args:
        policy: The learner, to recognise the mirror case by identity.
        opponent: What seat 1 is.
        actors: One built agent per environment, empty for a ``Policy``.
        environments: The group.
        generator: The sampling stream.

    Returns:
        One action per environment, or an empty list when seat 1 was already
        decided by the learner's own forward.
    """
    if opponent is policy:
        return []
    if isinstance(opponent, Policy):
        return [
            turn.action
            for turn in _decide(
                opponent,
                [
                    (_observation(environments, index, OPPONENT), OPPONENT)
                    for index in range(len(environments))
                ],
                generator,
            )
        ]
    return [
        act(_agent_observation(environments[index], OPPONENT))
        for index, act in enumerate(actors)
    ]


def _margin(observation: Mapping[str, Any]) -> float:
    """Return this seat's bank minus the other's, as a number not a reference.

    Read now and kept, because ``Environment.step`` appends the state object
    itself and the interpreter mutates the farms in place: the recorded history
    aliases one set of dicts, so asking ``env.steps[t]`` for turn ``t``'s money
    after the fact returns the terminal money for every ``t``.

    The seat is read from the observation rather than passed, because the
    engine stamps each seat's own index into the observation it hands that seat
    and a differential taken for the wrong seat is a sign error that telescopes
    just as neatly to the wrong answer. Both farms are public, so the other
    seat's bank is legible from here.

    Args:
        observation: One seat's own observation.

    Returns:
        ``farms[player]["money"] - farms[1 - player]["money"]``.
    """
    seat = int(observation["player"])
    farms = observation["farms"]
    return float(farms[seat]["money"]) - float(farms[1 - seat]["money"])


def _bank(observation: Mapping[str, Any]) -> float:
    """Return this seat's own bank, which the margin cannot recover."""
    return float(observation["farms"][int(observation["player"])]["money"])


def _opponent_actor(
    opponent: str, environment: Environment
) -> Callable[[Mapping[str, Any]], Any]:
    """Return the function that turns seat 1's observation into its action.

    One actor per environment, never shared: ``build_agent`` execs the spec into
    a fresh namespace, and the vendored heuristics keep per-episode state at
    module level, so a single actor driving a group of environments would carry
    one episode's memory into another's decisions.

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
        opponent: A spec the environment can build.
        environment: The episode, for its agent registry and configuration.

    Returns:
        A callable from seat 1's observation to seat 1's action.
    """
    agent, _parallelizable = build_agent(opponent, environment.agents, environment.name)
    arguments = len(signature(agent).parameters)
    return lambda observation: agent(
        *[observation, environment.configuration][:arguments]
    )


def _decide(
    policy: Policy,
    requests: Sequence[tuple[Mapping[str, Any], int]],
    generator: torch.Generator,
) -> list[Turn]:
    """Sample one turn's action for every ``(observation, seat)`` in the batch.

    Each row is encoded once and every consumer -- both heads, both masks, the
    value estimate -- reads that one encoding, so the state stored in the
    trajectory is provably the state the action was chosen from. The rows are
    then stacked and the trunk runs once: batching changes which arithmetic is
    issued together and nothing about which state produced which action, since
    every tensor a row contributes is sliced back out at the row's own index.

    The forward runs on whatever device the policy is on and the logits come
    straight back to the CPU. Sampling, masking and storage stay on the CPU
    deliberately: the masks are built there by pure Python, the trajectory is
    consumed there, and one ``torch.Generator`` then seeds the whole run rather
    than one per device.

    Ops for slots past the crew are sampled anyway, because a row of the unit
    head exists for every slot and ``masked_fill`` would leave an all-``-inf``
    row for the padded ones if the mask did not keep ``PASS`` alive there. They
    are then overwritten with ``IGNORE`` and contribute nothing to the joint
    log-probability: the engine never reads them -- ``decode_units`` emits ops
    only for the units on the board -- so an update that scored them would be
    fitting a choice that was never played.

    The quantity head is sampled the same way and scored more narrowly still.
    Every slot draws a bucket, but only the slots whose sampled op is a
    ``PICKUP`` or a ``PLACE`` spend one, so only those enter the log-probability
    -- the engine reads ``action[2]`` for those two verbs and no others, and a
    bucket it never read is not part of the action this turn took.

    Args:
        policy: The network to sample from.
        requests: One ``(observation, seat)`` per row. The observation must be
            that seat's own, whose ``private`` mapping is the only one legible
            to it.
        generator: The sampling stream.

    Returns:
        One ``Turn`` per request, in the order the requests were given.
    """
    board = torch.cat([encode_board(*request) for request in requests])
    scalars = torch.cat([encode_scalars(*request) for request in requests])
    positions = torch.cat([encode_positions(*request) for request in requests])
    units = torch.cat([unit_mask(*request) for request in requests])
    counts = torch.cat([unit_quantity_mask(*request) for request in requests])
    trades = torch.cat([market_mask(*request) for request in requests])

    device = next(policy.parameters()).device
    with torch.no_grad():
        unit_logits, quantity_logits, market_logits, value = policy(
            board.to(device), scalars.to(device), positions.to(device)
        )
    unit_logits, quantity_logits, market_logits, value = (
        unit_logits.cpu(),
        quantity_logits.cpu(),
        market_logits.cpu(),
        value.cpu(),
    )
    chosen_units, unit_log = _sample(unit_logits, units, generator)
    chosen_quantities, quantity_log = _sample(quantity_logits, counts, generator)
    chosen_market, market_log = _sample(market_logits, trades, generator)
    # Which slots spent the bucket they drew. Read off the sampled op, because
    # the bucket's own value cannot say: an unspent 1 and a genuine transfer of
    # one item are the same number, and only one of them acted.
    transferred = transfer_slots(chosen_units)

    turns: list[Turn] = []
    for row, request in enumerate(requests):
        count = unit_count(*request)
        rows = slice(row, row + 1)
        action = decode_units(
            _one_hot(chosen_units[rows], unit_logits.shape[-1]),
            _one_hot(chosen_quantities[rows], quantity_logits.shape[-1]),
            count,
            units[rows],
            counts[rows],
        )
        action["market"] = decode_market(
            _one_hot(chosen_market[rows], market_logits.shape[-1]),
            trades[rows],
        )
        turns.append(
            Turn(
                action=action,
                board=board[rows],
                scalars=scalars[rows],
                positions=positions[rows],
                units=torch.cat(
                    [
                        chosen_units[rows, :count],
                        torch.full_like(chosen_units[rows, count:], IGNORE),
                    ],
                    dim=1,
                ),
                quantities=chosen_quantities[rows],
                market=chosen_market[rows],
                unit_mask=units[rows],
                quantity_mask=counts[rows],
                market_mask=trades[rows],
                log_prob=(
                    unit_log[rows, :count].sum(dim=1)
                    + quantity_log[rows, :count]
                    .masked_fill(~transferred[rows, :count], 0.0)
                    .sum(dim=1)
                    + market_log[rows].sum(dim=1)
                ),
                value=value[rows],
            )
        )
    return turns


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

    ``decode_units`` and ``decode_market`` argmax their input under the mask,
    and the index was sampled under that same mask, so its one-hot row survives
    the ``masked_fill`` at 1.0 while every rival is either 0.0 or ``-inf`` and
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


def _trajectory(stream: Stream, environment: Environment) -> Trajectory:
    """Stack one recorded seat's turns into the tensors the update reads.

    The stream's ``margins``, ``banks`` and ``potentials`` all end one short of
    the terminal state, because the loop appends before each decision and there
    is no decision after the last one. The two money series read their terminal
    value here, from the environment the stream played, which makes each of them
    exactly one longer than ``turns`` so the differences are one per decision and
    each telescopes -- the differential to the final margin, and our own bank to
    the terminal bank less the ``STARTING_MONEY`` both farms open on.

    The potentials do *not* get that treatment and are stored one row per turn,
    undifferenced and terminal-free. The season's score is each seat's ``money``
    and nothing else, so whatever is still in the shed at the horizon is worth
    zero -- and a potential that said otherwise would be paying for the one
    endpoint the agent controls. See the module docstring.

    Args:
        stream: One seat's decisions and the two money series it saw.
        environment: The episode it played, for its terminal position.

    Returns:
        The ``Trajectory``.
    """
    turns = stream.turns
    terminal = environment.state[stream.seat].observation
    margins = [*stream.margins, _margin(terminal)]
    banks = [*stream.banks, _bank(terminal)]
    series = [*stream.counts, counts(terminal, stream.seat)]
    snapshots = [*stream.snapshots, _snapshot(terminal)]
    won = rank(series[-1].money, _other(terminal, stream.seat))
    sold = sale_metrics(snapshots, stream.seat)
    dones = torch.zeros(len(turns), dtype=torch.bool)
    dones[-1] = True
    unit_actions = torch.cat([turn.units for turn in turns])
    unit_quantities = torch.cat([turn.quantities for turn in turns])
    market_actions = torch.cat([turn.market for turn in turns])
    unit_masks = torch.cat([turn.unit_mask for turn in turns])
    unit_quantity_masks = torch.cat([turn.quantity_mask for turn in turns])
    market_masks = torch.cat([turn.market_mask for turn in turns])
    return Trajectory(
        board=torch.cat([turn.board for turn in turns]),
        scalars=torch.cat([turn.scalars for turn in turns]),
        positions=torch.cat([turn.positions for turn in turns]),
        unit_actions=unit_actions,
        unit_quantities=unit_quantities,
        market_actions=market_actions,
        unit_masks=unit_masks,
        unit_quantity_masks=unit_quantity_masks,
        market_masks=market_masks,
        log_probs=torch.cat([turn.log_prob for turn in turns]),
        values=torch.cat([turn.value for turn in turns]),
        rewards=_differences(margins),
        own=_differences(banks),
        shaped=shaped(series, won),
        # Phase-1b's reward, computed from the same episode so the ablation is
        # baseline vs baseline+money with everything else -- seeds, trajectories,
        # actions -- held identical. Recording both costs one pass over a list
        # of counts and removes the only other way to run the comparison, which
        # is to play the season twice and hope it was deterministic.
        shaped_money=shaped(series, won, money_weight=money_weight()),
        margin=margin_reward(series, won),
        sparse=sparse_reward(series, won),
        potentials=torch.tensor(stream.potentials, dtype=torch.float32),
        dones=dones,
        final_margin=margins[-1],
        final_bank=_bank(terminal),
        final_capital=float(series[-1].capital),
        illegal=(
            _illegal(unit_actions, unit_masks)
            + _illegal(unit_quantities, unit_quantity_masks)
            + _illegal(market_actions, market_masks)
        ),
        sales=sold["sales"],
        units_sold=sold["units"],
        mean_sale_price=sold["mean_sale_price"],
        realisation=sold["price_realisation"],
        bought=buy_units(snapshots, stream.seat),
    )


def _snapshot(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the three things ``sales`` reads out of one observation.

    This exists for MEMORY, and specifically not for aliasing. I assumed the
    aliasing and then measured it: ``Environment.state[seat].observation`` is a
    fresh object with a fresh ``farms`` list on every step, and retaining the
    live observations instead of these projections gives byte-identical sale
    metrics over a real 719-turn episode. The alias warning in this module's
    docstring is about ``env.steps``, which is a different structure, and it
    does not reach this path.

    What does reach this path is size. A round is 24 environments of 719 turns
    with both seats recorded, and a full observation carries a 10x10 grid of
    tile dicts; retaining them would hold roughly two orders of magnitude more
    per trajectory than the 15.4 MB one already costs, in every one of the
    twelve worker processes at once. So this takes exactly the fields
    ``sale_metrics`` and ``buy_units`` index -- both seats' money, our shed and
    the market book -- into a fresh dict shaped the way they expect.

    It is a projection of an observation and not a substitute for one; nothing
    else may be read from it, which is why it is built here beside its only two
    consumers rather than exposed.

    Args:
        observation: One seat's observation, as seen at decision time.

    Returns:
        A standalone mapping with the same shape at the keys those two functions
        reach through.
    """
    return {
        "farms": [{"money": float(farm["money"])} for farm in observation["farms"]],
        "private": {"shed": dict(observation["private"]["shed"])},
        "market": {"prices": dict(observation["market"]["prices"])},
    }


def _other(observation: Mapping[str, Any], seat: int) -> float:
    """Return the opposing seat's terminal bank."""
    return float(observation["farms"][1 - seat]["money"])


def _differences(series: list[float]) -> torch.Tensor:
    """Return the per-turn change in a money series that is one longer than the turns.

    ``strict=True`` on the zip, because the ragged truncation it forbids is
    exactly how an off-by-one here would look: a series the same length as the
    turns would silently yield one reward fewer than there were decisions, and
    every reward in the episode would be attributed to the turn after the one
    that earned it.

    Args:
        series: The quantity at each state, terminal state included.

    Returns:
        ``(turns,)`` float32 differences, which sum to ``series[-1] -
        series[0]``.
    """
    return torch.tensor(
        [after - before for before, after in zip(series[:-1], series[1:], strict=True)],
        dtype=torch.float32,
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
