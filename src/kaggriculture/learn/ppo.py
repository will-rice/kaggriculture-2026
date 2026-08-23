"""The PPO update: GAE, a clipped joint ratio, entropy, and a decaying teacher KL.

Four things in here are load-bearing, and three of them fail without raising.

**The ratio is computed under the mask the rollout stored, never a fresh one.**
``exp(new - old)`` is an importance weight only if ``new`` and ``old`` are
log-probabilities of distributions over the same support. Recomputing a mask
from the observation would agree with the stored one on almost every turn and
disagree silently on the turns where any state-dependent legality shifted --
a shop unlocking, a shed emptying, a mask function edited between collecting
and training -- and the resulting number would still be finite, still be
positive, and still be clipped to something plausible. The run would just learn
slowly and wrongly, and nothing would say so. So the update reads
``Trajectory.unit_masks`` and ``Trajectory.market_masks`` and applies them to
the fresh logits. It is structurally incapable of doing anything else: a
``Trajectory`` carries no observation, and ``mask.py`` is not imported here.

**The ratio is joint over the whole turn, because that is what acted.** A turn's
action is every real unit's op, the count each transferring unit asked for, and
all ``len(MARKET_SLOTS) + 2`` market buckets, sampled together; ``rollout``
stores one summed log-probability per turn for exactly that reason. All three
heads therefore contribute to the policy loss through one ratio rather than
through three -- there is no stored per-head ``old`` to form a per-head ratio
against, and inventing one by splitting the recomputed ``new`` against a
fabricated ``old`` would be a ratio between a distribution and itself. The cost
is variance: ``exp`` of a difference of two ~25-term sums moves further per step
than a single-factor ratio would, which is what the clip is there to bound.

**The three heads are padded differently, and the difference is not cosmetic.**
The unit head has a row for every one of ``MAX_UNITS`` slots and the crew is
usually four, so most rows hold no decision; ``rollout`` marks them ``IGNORE``,
the same sentinel ``encode_units`` writes and ``train.unit_loss`` masks on. They
are excluded here from the joint log-probability, from the entropy bonus and
from the teacher KL -- included, they would let the gradient shape a
distribution over ops for hands the engine never asked about, and would make
the entropy term a function of how many slots happen to be empty. The quantity
head is narrower still: it is excluded on every slot whose sampled op was not a
``PICKUP`` or a ``PLACE``, padded or not, because those are the only two ops
whose ``action[2]`` the engine reads and a bucket it never read is not part of
what acted. The market head excludes nothing: all 21 slots are real decisions on
every turn and bucket 0 means "trade nothing", which the corpus chooses on about
a third of rows.

**GAE is taken inside an episode and never across one.** ``update`` takes
trajectories rather than a pre-flattened block precisely so that the boundaries
cannot be lost on the way in: ``flatten`` runs ``advantages`` once per
trajectory and concatenates afterwards, so no advantage can bootstrap the start
of one season off the end of another. Each episode's bootstrap is a hard zero,
which is right because a ``Trajectory`` is a whole episode and the season ends
there.

The teacher penalty exists to survive the first updates on a competent policy,
not to pin the learner to the clone -- and this clone is weak: 90.5% holdout
accuracy and it still banks nothing, hiring on nearly every turn. So
``kl_weight`` decays linearly to exactly zero rather than exponentially toward
it, and after ``KL_STEPS`` the teacher stops being a constraint entirely. The KL
itself keeps being measured after that, because it is the diagnostic that says
how far the policy has travelled from where it started.

``GAMMA`` is 0.9995. The season is 719 turns and the payoff is terminal, so the
discount has to reach the end of it: 0.9995 gives an effective horizon around
2,000 turns and leaves 0.70 of a terminal payoff standing at turn 0, which is
what Toad's 0.999 left standing across their 360-turn game. Their digits over
our length would leave 0.49, and a conventional 0.99 would have a horizon of 100
and blind the policy to five-sixths of the season.

``REWARD_SCALE`` is the one number here that was changed because a measurement
demanded it rather than because a paper suggested it; the constant's comment
carries the figures.

The reward the update climbs is assembled in ``reward_of`` and is *money plus
progress*: a convex blend of the two recorded money series, plus
``gamma * P(s') - P(s)`` over ``progress``'s potential. The money half is the
objective and the progress half is the curriculum, and the two are combined
differently on purpose -- see that function. What the progress half buys is
credit assignment: without it, a coin banked on day 12 has to be attributed to a
seed bought on day 7, across 120 turns of a 719-turn season, and the measured
consequence of asking for that was a policy that never reached a state where
``SELL`` was legal.
"""

import logging
from dataclasses import dataclass
from typing import Sequence

import torch

from kaggriculture.constants import STARTING_MONEY
from kaggriculture.learn.encoding import IGNORE, transfer_slots
from kaggriculture.learn.model import Policy
from kaggriculture.learn.progress import progress_reward
from kaggriculture.learn.rollout import Trajectory

LOGGER = logging.getLogger(__name__)

# The reward is a change in coins, and coins are large: measured over three
# episodes of an untrained policy against `starter`, the per-turn reward has a
# standard deviation around 112 and the value target reaches 2,900. Regressed
# raw, the value term is `0.5 * 13,000` against a policy loss of about 0.001 --
# seven orders of magnitude -- so `max_grad_norm` renormalises a gradient that
# is essentially all value head and the policy stops moving. Measured: with no
# scaling the policy loss after four epochs was -0.0011 and the total loss
# 6,545.
#
# Scaled in units of the opening bank rather than by a fitted constant, because
# `STARTING_MONEY` is an engine fact and 1/112 is a property of one untrained
# policy against one opponent. One unit of reward is then "outproduced them by
# a starting farm's worth", which is also the scale the ladder's margins live
# on.
#
# Only the reward is scaled. `Trajectory.values` is the value head's own
# output, which is already in whatever units it is being trained to predict, so
# scaling it too would divide the prediction by 3,000 a second time.
REWARD_SCALE = 1.0 / STARTING_MONEY

# How much of the reward is the opponent-relative half. 1.0 is the differential
# alone -- the win condition, and the right *final* objective; 0.0 is the change
# in our own bank alone, the same objective measured absolutely.
#
# It defaults to the final objective and the training loop schedules it, because
# a purely relative reward has no gradient before a policy can bank anything.
# Measured on this game: 30 iterations of mirrored self-play on the differential
# alone gave a mean bank of 0, a best episode of 4 coins out of a possible
# 200,000, and an advantage of -0.0008 +- 0.0128. Two copies of a bankrupt
# policy bankrupt each other identically, so the differential is zero on every
# turn and the critic correctly learns that everything is zero.
#
# Every single-box winner this project has a primary source for shaped first and
# switched after: Toad Brigade shaped for 20M steps, FLG for 65M before moving
# to a zero-sum differential of the same construction as ours, and Frog Parade
# moved to sparse win/loss "as soon as training was running stably". The spec
# for this phase said not to shape, on the grounds that shaping would encode our
# own beliefs about good play. That confuses the objective with the curriculum,
# and `own` encodes no belief about how to farm in any case -- it pays for
# banking coins, which is what winning is made of.
DIFFERENTIAL = 1.0

# How much of `progress.progress_reward` is added on top of the money reward.
#
# Added rather than blended in, unlike `differential`. The two money series are
# alternative measurements of the same quantity and mixing them is a choice
# about which one is the objective; the progress term is not an alternative
# objective at all. It is `gamma * P(s') - P(s)` for a potential `P` over the
# farm's own state, and Ng, Harada and Russell prove that adding exactly that
# form leaves the optimal policy unchanged -- so it changes what the gradient
# sees on the way without changing where it is going, which is the entire
# reason to have it.
#
# 1.0, and there is no relative weight to tune, because `P` is denominated in
# coins on the same scale as the bank. One coin of shed produce is worth one
# coin. A weight other than 1 would be an assertion that a coin of standing
# wheat is worth some other number of banked coins, and there is nothing to base
# that on.
#
# The total it contributes to an episode is a constant, and since `P` became net
# worth it is no longer a small one: the series telescopes to `-P(s_0)` under
# the discount, which is the 3,000 both seats open on, but the *undiscounted*
# sum is that plus the `(1 - gamma)` carrying cost on everything owned -- 19,086
# coins over one measured `economic_policy` season, of which 111,312 arrives on
# the last turn alone. It is dense where the money reward is sparse, it does not
# move the optimum, and it does put one reward two orders of magnitude larger
# than its neighbours at the horizon. `progress.progress_reward` carries the
# arithmetic and the measurement.
PROGRESS = 1.0

# Toad Brigade's 0.999, rescaled to our episode. Their game is 360 turns and
# ours is 719 decisions, so the same digits retain 0.488 of a terminal payoff at
# turn 0 where theirs retained 0.698. `0.999 ** (360 / 719)` restores the
# retention; see `toad_loss.DISCOUNTING`, which carries the derivation and is
# the value the Toad runner trains under. Kept equal to it on purpose: the two
# loops read the same episodes and the same potential, and a discount that
# differed between them would make their measurements incomparable.
GAMMA = 0.9995
LAM = 0.95
CLIP = 0.2
EPOCHS = 4
MINIBATCH = 512
VALUE_WEIGHT = 0.5
ENTROPY_WEIGHT = 0.01
MAX_GRAD_NORM = 0.5

# The penalty opens at 1.0 -- the same order as the policy loss, so it is a real
# constraint rather than a decoration -- and reaches exactly zero after
# `KL_STEPS` updates. Linear rather than exponential because "decays to zero"
# should mean zero: an exponential is still tethering the learner to the clone
# at every step of the run, by an amount too small to see and too large to be
# nothing.
#
# 100, not the 4,000 this was written with. `step` here counts *updates*, not
# minibatches -- `update` reads it once per call -- and 4,000 was picked before
# any run length existed. Task 5 measured the loop at roughly 55 iterations per
# hour, so 4,000 updates is three days: the weight would still be 0.92 at the
# end of a six-hour run, and a penalty that never decays is not a decaying
# penalty, it is a constraint on the whole run. 100 reaches zero about a third
# of the way in, which is what "survive the first updates and then let go"
# means at this budget. Any run that changes length has to revisit this number.
KL_INITIAL = 1.0
KL_STEPS = 100

# The progress term is held at full weight for `PROGRESS_HOLD` updates and then
# decays linearly to exactly zero at `PROGRESS_STEPS`, which is where
# `selfplay.SHAPING_ITERATIONS` switches the money reward from our own bank to
# the win condition. So the curriculum is one ramp rather than two: dense
# progress and absolute money first, then the terminal objective on its own.
#
# Both competitors this project has a primary source for did this. FLG shaped
# move-to-resource, clear-rubble and mine for ~65M steps and then switched to
# their zero-sum differential; Toad Brigade shaped for the first 20M steps and
# then distilled onto the sparse reward. Neither kept the shaping to the end.
#
# Unlike the teacher penalty, nothing depends on this decay: a potential-based
# term does not move the optimum, so it can be left on forever without changing
# what is being optimised. It decays anyway for the one effect it does have --
# the `(1 - gamma)` carrying cost, which since the potential became net worth is
# charged on the bank as well as on held stock, so it is a per-turn drag
# proportional to everything the seat owns rather than the small preference for
# selling early it was when only the pipeline was in the potential. Exactly
# cancelled under the discounted objective, and something we would rather not
# carry through a whole run regardless -- and because a reward that is still
# being shaped at the end of a run
# has not been shown to work without the shaping.
#
# At ~55 iterations an hour a workstation day ends inside the hold, so a run of
# this budget measures the shaped phase and nothing else. Its report says so.
PROGRESS_INITIAL = 1.0
PROGRESS_HOLD = 100
PROGRESS_STEPS = 200


@dataclass(frozen=True)
class PpoConfig:
    """The update's hyperparameters, all of them.

    Frozen and defaulted from this module's constants so that a run's numbers
    are readable from the repository at a commit rather than from whatever the
    shell passed. Nothing here is a flag.

    Attributes:
        differential: How much of the reward is opponent-relative. 1.0 is the
            win condition alone, 0.0 the change in our own bank alone; the
            training loop schedules it, and the constant's comment says why a
            run cannot start at 1.0.
        progress: How much of the potential-based progress reward is added on
            top of the money reward. Added, not blended: it is a shaping term
            in Ng, Harada and Russell's form and does not compete with the
            objective. The training loop schedules it too.
        reward_scale: What one coin of bank differential is worth to the value
            head. Coins are large and the value regression would otherwise
            drown every other term; see the constant's comment.
        gamma: Discount. Close to 1 on purpose; see the module docstring.
        lam: GAE's trace decay.
        clip: PPO's ratio clip, the bound on how far one batch can move.
        epochs: Passes over the batch per update.
        minibatch: Rows per gradient step.
        value_weight: Weight on the value regression.
        entropy_weight: Weight on the entropy bonus, subtracted from the loss.
        max_grad_norm: Global gradient norm clip.
        kl_initial: The teacher penalty at step 0.
        kl_steps: How many updates until the teacher penalty is zero.
    """

    differential: float = DIFFERENTIAL
    progress: float = PROGRESS
    reward_scale: float = REWARD_SCALE
    gamma: float = GAMMA
    lam: float = LAM
    clip: float = CLIP
    epochs: int = EPOCHS
    minibatch: int = MINIBATCH
    value_weight: float = VALUE_WEIGHT
    entropy_weight: float = ENTROPY_WEIGHT
    max_grad_norm: float = MAX_GRAD_NORM
    kl_initial: float = KL_INITIAL
    kl_steps: int = KL_STEPS


@dataclass(frozen=True)
class Rows:
    """A batch of trajectories as independent rows, with the targets precomputed.

    Every field's first dimension is one decision, drawn from some episode in
    the batch; which episode it came from is deliberately no longer recoverable,
    because by this point every quantity that needed the episode's ordering --
    the advantage and the value target -- has already been computed inside it.

    Attributes:
        board: ``(rows, TILE_PLANES, BOARD, BOARD)`` planes.
        scalars: ``(rows, SCALARS)`` market and phase features.
        positions: ``(rows, MAX_UNITS)`` flattened tile indices per unit.
        unit_actions: ``(rows, MAX_UNITS)`` op indices, ``IGNORE`` where no unit
            stood.
        unit_quantities: ``(rows, MAX_UNITS)`` the bucket each unit's quantity
            head sampled. Never ``IGNORE``: padding is stated once, on
            ``unit_actions``, and a slot whose op is not a transfer spends no
            bucket whether or not a unit stood there.
        market_actions: ``(rows, len(MARKET_SLOTS) + 2)`` quantity buckets.
        unit_masks: ``(rows, MAX_UNITS, len(UNIT_OPS))`` bool, the mask that
            gated the logits when the action was sampled.
        unit_quantity_masks: ``(rows, MAX_UNITS, len(QUANTITIES))`` bool,
            likewise for the quantity head.
        market_masks: ``(rows, len(MARKET_SLOTS) + 2, len(QUANTITIES))`` bool,
            likewise.
        log_probs: ``(rows,)`` the behaviour policy's joint log-probability of
            the whole turn -- PPO's ``old``.
        advantages: ``(rows,)`` GAE, normalised across the batch.
        returns: ``(rows,)`` the value target, ``advantages + values`` before
            normalisation. Not normalised: it is a regression target on the
            bank differential's own scale.
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
    advantages: torch.Tensor
    returns: torch.Tensor


def advantages(
    rewards: torch.Tensor, values: torch.Tensor, gamma: float, lam: float
) -> torch.Tensor:
    """Return generalised advantage estimates for one episode.

    ``values`` is one longer than ``rewards`` and its last entry is the
    bootstrap: ``V(s_T)``, the value of the state the last action led to. That
    is required rather than optional so the caller cannot quietly hand over a
    same-length pair and get an advantage computed against a value that belongs
    to the wrong state -- an off-by-one that shifts every advantage in the
    episode by one turn and never raises.

    Recursed backwards rather than expanded as a discounted cumulative sum. The
    closed form needs a division by ``(gamma * lam) ** t``, which over 719 turns
    reaches 10^16 and throws away most of float32's mantissa; the loop costs
    microseconds against a 10M-parameter forward pass.

    Args:
        rewards: ``(turns,)`` per-turn reward.
        values: ``(turns + 1,)`` value estimates, the last being the bootstrap.
        gamma: Discount.
        lam: Trace decay. At ``lam=1`` this is the Monte Carlo advantage; at
            ``lam=0`` it is the one-step TD error.

    Returns:
        ``(turns,)`` advantages.

    Raises:
        ValueError: If ``values`` is not exactly one longer than ``rewards``.
    """
    if values.shape != (rewards.shape[0] + 1,):
        raise ValueError(
            f"values must be one longer than rewards and hold the bootstrap: got "
            f"{tuple(values.shape)} against {tuple(rewards.shape)}"
        )
    deltas = rewards + gamma * values[1:] - values[:-1]
    estimates = torch.zeros_like(rewards)
    running = torch.zeros((), dtype=rewards.dtype)
    for turn in range(len(rewards) - 1, -1, -1):
        running = deltas[turn] + gamma * lam * running
        estimates[turn] = running
    return estimates


def policy_loss(
    ratio: torch.Tensor, advantage: torch.Tensor, clip: float
) -> torch.Tensor:
    """Return PPO's clipped surrogate loss.

    The minimum of the clipped and unclipped objectives is what makes the clip
    a bound rather than a suggestion: taking the clipped term alone would leave
    the gradient free to push a ratio that is already past the boundary further
    out, because the clip is flat there and the unclipped term is not.

    Args:
        ratio: ``(rows,)`` ``exp(new_log_prob - old_log_prob)``.
        advantage: ``(rows,)`` normalised advantages.
        clip: The half-width of the trust region on the ratio.

    Returns:
        A scalar loss, negated so that gradient descent increases the
        objective.
    """
    return -torch.min(
        ratio * advantage, ratio.clamp(1.0 - clip, 1.0 + clip) * advantage
    ).mean()


def kl_weight(step: int, initial: float = KL_INITIAL, steps: int = KL_STEPS) -> float:
    """Return the teacher penalty's weight at this update.

    Decays linearly from ``initial`` to exactly zero over ``steps`` updates and
    stays there. The penalty exists so the first updates do not wreck a
    competent policy; it is not a statement that the clone is worth staying
    near, and this one banks nothing in play despite 90.5% holdout accuracy.

    Args:
        step: Which update this is, counted from zero across the whole run.
        initial: The weight at step 0.
        steps: How many updates until the weight is zero.

    Returns:
        The weight, never negative.
    """
    return max(0.0, initial * (1.0 - step / steps))


def progress_weight(
    step: int,
    initial: float = PROGRESS_INITIAL,
    hold: int = PROGRESS_HOLD,
    steps: int = PROGRESS_STEPS,
) -> float:
    """Return the progress term's weight at this update.

    Flat at ``initial`` through ``hold`` updates, then linear to exactly zero at
    ``steps``, then zero. The hold is the difference between this and the
    teacher penalty's schedule and it is the point of the thing: the teacher
    penalty is a trust region that should start relaxing immediately, while the
    progress term is a curriculum that has to stay at full strength long enough
    for the chain to be learned before it is taken away. A term that starts
    decaying on update 1 is weakest exactly when the policy still cannot bank.

    Args:
        step: Which update this is, counted from zero across the whole run.
        initial: The weight through the hold.
        hold: How many updates the weight stays at ``initial``.
        steps: Which update the weight reaches zero on.

    Returns:
        The weight, never negative and never above ``initial``.

    Raises:
        ValueError: If the ramp has no width, which would make the schedule a
            cliff and divide by zero saying so.
    """
    if steps <= hold:
        raise ValueError(
            f"the progress term needs a ramp to decay across: hold {hold} is not "
            f"before steps {steps}"
        )
    return initial * max(0.0, min(1.0, (steps - step) / (steps - hold)))


def flatten(batch: Sequence[Trajectory], config: PpoConfig) -> Rows:
    """Turn episodes into rows, taking GAE inside each one before concatenating.

    The order is what matters: advantages are computed per trajectory, so the
    recursion never runs across a boundary and no season's opening advantage
    bootstraps off another season's close. Each episode's bootstrap value is a
    hard zero because a ``Trajectory`` is a whole episode and ``dones[-1]`` is
    True -- there is no state after the last turn to value.

    Advantages are normalised across the whole batch afterwards, which is where
    it belongs: normalising per episode would rescale each season by its own
    spread and quietly tell the learner that a flat episode's small wins were as
    large as a volatile one's.

    The value target is not normalised -- it is a regression target and moving
    its zero every batch would chase the value head around. ``reward_scale`` is
    what keeps it in range instead, and it is applied here, to the reward
    alone: ``Trajectory.values`` is the value head's own output and is already
    in whatever units it is being trained to predict, so scaling it as well
    would divide the prediction by ``STARTING_MONEY`` a second time and leave a
    residue of ``V`` in every target.

    Args:
        batch: The episodes to learn from.
        config: The update's hyperparameters, for ``reward_scale``, ``gamma``
            and ``lam``.

    Returns:
        The batch as ``Rows``.
    """
    estimates = [
        advantages(
            reward_of(trajectory, config) * config.reward_scale,
            torch.cat([trajectory.values, torch.zeros(1)]),
            config.gamma,
            config.lam,
        )
        for trajectory in batch
    ]
    returns = torch.cat(
        [
            estimate + trajectory.values
            for estimate, trajectory in zip(estimates, batch, strict=True)
        ]
    )
    stacked = torch.cat(estimates)
    LOGGER.info(
        "%d episodes, %d rows, advantage %.4f +- %.4f",
        len(batch),
        len(stacked),
        float(stacked.mean()),
        float(stacked.std()),
    )
    return Rows(
        board=torch.cat([trajectory.board for trajectory in batch]),
        scalars=torch.cat([trajectory.scalars for trajectory in batch]),
        positions=torch.cat([trajectory.positions for trajectory in batch]),
        unit_actions=torch.cat([trajectory.unit_actions for trajectory in batch]),
        unit_quantities=torch.cat([trajectory.unit_quantities for trajectory in batch]),
        market_actions=torch.cat([trajectory.market_actions for trajectory in batch]),
        unit_masks=torch.cat([trajectory.unit_masks for trajectory in batch]),
        unit_quantity_masks=torch.cat(
            [trajectory.unit_quantity_masks for trajectory in batch]
        ),
        market_masks=torch.cat([trajectory.market_masks for trajectory in batch]),
        log_probs=torch.cat([trajectory.log_probs for trajectory in batch]),
        advantages=(stacked - stacked.mean()) / (stacked.std() + 1e-8),
        returns=returns,
    )


def reward_of(trajectory: Trajectory, config: PpoConfig) -> torch.Tensor:
    """Return the reward this update is climbing: money, mixed, plus progress.

    The two *money* series are blended convexly rather than summed, so that the
    reward's scale does not move when the mix does: both are in coins, and
    ``REWARD_SCALE`` was fitted against one series in coins. A sum at weight 1.0
    on each would double the value target halfway through a run and hand
    ``max_grad_norm`` the same problem the reward scale was introduced to fix.

    The progress term is *added* to that blend, because it is not a competing
    measurement of the objective -- it is ``gamma * P(s') - P(s)``, which by Ng,
    Harada and Russell leaves the optimal policy unchanged whatever the money
    reward beneath it is. Blending it in would make the money reward smaller
    when shaping was strong, which is the opposite of what shaping is for. It
    telescopes to a constant and so cannot move the objective; it does move the
    *scale*, because the potential is net worth and the horizon hands all of it
    back on one turn. See the ``PROGRESS`` comment for what that is worth in
    coins.

    Args:
        trajectory: The episode, carrying all three recorded series.
        config: The update's hyperparameters. ``differential`` mixes the money
            series, ``progress`` weights the shaped one, and ``gamma`` is the
            discount the shaped one has to be built with -- the same ``gamma``
            the advantage is taken under, which is why it is read from here
            rather than passed separately.

    Returns:
        ``(turns,)`` per-turn reward, in coins.
    """
    money = (
        config.differential * trajectory.rewards
        + (1.0 - config.differential) * trajectory.own
    )
    return money + config.progress * progress_reward(
        trajectory.potentials, config.gamma
    )


def update(
    policy: Policy,
    teacher: Policy,
    optimiser: torch.optim.Optimizer,
    batch: Sequence[Trajectory],
    step: int,
    config: PpoConfig,
) -> dict[str, float]:
    """Run one PPO update over a batch of episodes and return what it measured.

    The optimiser is passed in rather than built here because Adam's moments
    have to survive between updates; an optimiser constructed per call would
    reset them every time and turn a tuned learning rate into a much larger
    effective one. ``step`` is passed in for the same reason ``config`` is not
    mutated: the teacher penalty depends on where the run is, and ``config``
    holds only what does not move.

    Args:
        policy: The network being trained, updated in place.
        teacher: The behaviour-cloned checkpoint the KL penalty pulls toward.
            Read under ``no_grad`` and never stepped. ``Policy`` carries no
            normalisation layers, so its train/eval mode does not change what
            it returns and is not touched here.
        optimiser: The optimiser over ``policy``'s parameters, carrying its
            state across updates.
        batch: The episodes to learn from.
        step: Which update this is, counted across the run. Only the teacher
            penalty reads it.
        config: The hyperparameters.

    Returns:
        Means over the update's minibatches: ``loss``, ``loss/policy``,
        ``loss/value``, ``entropy``, ``kl``, ``kl/weight``, ``ratio`` and
        ``clipped`` -- the fraction of rows the clip was binding on, which is
        the number that says whether the trust region is doing anything.
    """
    rows = flatten(batch, config)
    device = next(policy.parameters()).device
    weight = kl_weight(step, config.kl_initial, config.kl_steps)
    totals = {
        "loss": 0.0,
        "loss/policy": 0.0,
        "loss/value": 0.0,
        "entropy": 0.0,
        "kl": 0.0,
        "ratio": 0.0,
        "clipped": 0.0,
    }
    taken = 0

    for _epoch in range(config.epochs):
        order = torch.randperm(len(rows.log_probs))
        for start in range(0, len(order), config.minibatch):
            index = order[start : start + config.minibatch]
            board = rows.board[index].to(device)
            scalars = rows.scalars[index].to(device)
            positions = rows.positions[index].to(device)
            unit_actions = rows.unit_actions[index].to(device)
            unit_quantity_actions = rows.unit_quantities[index].to(device)
            market_actions = rows.market_actions[index].to(device)
            unit_masks = rows.unit_masks[index].to(device)
            unit_quantity_masks = rows.unit_quantity_masks[index].to(device)
            market_masks = rows.market_masks[index].to(device)

            unit_logits, unit_quantity_logits, market_logits, values = policy(
                board, scalars, positions
            )
            units = torch.log_softmax(
                unit_logits.masked_fill(~unit_masks, -torch.inf), dim=-1
            )
            quantities = torch.log_softmax(
                unit_quantity_logits.masked_fill(~unit_quantity_masks, -torch.inf),
                dim=-1,
            )
            market = torch.log_softmax(
                market_logits.masked_fill(~market_masks, -torch.inf), dim=-1
            )
            real = unit_actions != IGNORE
            transferred = transfer_slots(unit_actions)
            log_probs = joint_log_prob(
                units,
                quantities,
                market,
                unit_actions,
                unit_quantity_actions,
                market_actions,
            )
            ratio = torch.exp(log_probs - rows.log_probs[index].to(device))

            advantage = rows.advantages[index].to(device)
            surrogate = policy_loss(ratio, advantage, config.clip)
            value = torch.nn.functional.mse_loss(values, rows.returns[index].to(device))
            # The quantity head's entropy is averaged over the slots that
            # actually spent a bucket, the same condition its log-probability
            # is under. Rewarding entropy on a slot whose op ignored the
            # bucket would pay the head to spread mass over a decision the
            # engine never read. The denominator is a count of those slots and
            # a minibatch can hold none of them, in which case the term is a
            # sum of nothing over one, which is the zero contribution an empty
            # mean should make.
            entropy = (
                entropy_of(units, unit_masks)[real].mean()
                + entropy_of(quantities, unit_quantity_masks)
                .masked_fill(~transferred, 0.0)
                .sum()
                / transferred.sum().clamp(min=1)
                + entropy_of(market, market_masks).mean()
            )

            with torch.no_grad():
                # The teacher KL covers the op and market heads only. This
                # loop's teacher is `selfplay.initialise`'s deep copy of the
                # policy's own starting weights, so on a fresh or a
                # behaviour-cloned start its quantity head is the learner's
                # random initialisation -- anchoring the head to that is worse
                # than not anchoring it. `toad` loads its teacher from a
                # state dict, can therefore ask whether the head was ever
                # trained, and includes the term when the answer is yes.
                teacher_units, _teacher_quantity, teacher_market, _value = teacher(
                    board, scalars, positions
                )
            divergence = (
                divergence_of(
                    units,
                    torch.log_softmax(
                        teacher_units.masked_fill(~unit_masks, -torch.inf), dim=-1
                    ),
                    unit_masks,
                )[real].mean()
                + divergence_of(
                    market,
                    torch.log_softmax(
                        teacher_market.masked_fill(~market_masks, -torch.inf), dim=-1
                    ),
                    market_masks,
                ).mean()
            )

            loss = (
                surrogate
                + config.value_weight * value
                - config.entropy_weight * entropy
                + weight * divergence
            )
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimiser.step()

            measured = {
                "loss": loss,
                "loss/policy": surrogate,
                "loss/value": value,
                "entropy": entropy,
                "kl": divergence,
                "ratio": ratio.mean(),
                "clipped": ((ratio - 1.0).abs() > config.clip).float().mean(),
            }
            for key, tensor in measured.items():
                totals[key] += float(tensor.detach())
            taken += 1

    return {
        "kl/weight": weight,
        **{key: total / taken for key, total in totals.items()},
    }


def joint_log_prob(
    units: torch.Tensor,
    unit_quantities: torch.Tensor,
    market: torch.Tensor,
    unit_actions: torch.Tensor,
    unit_quantity_actions: torch.Tensor,
    market_actions: torch.Tensor,
) -> torch.Tensor:
    """Return the log-probability of each whole turn, summed over its decisions.

    Padded unit slots are dropped with ``masked_fill`` on the gathered value
    rather than by multiplying through a float mask. A padded slot's index is
    ``IGNORE``, which has to be clamped into range before ``gather`` will take
    it, and the option it clamps onto can be one the stored mask forbids -- so
    the gathered number is sometimes ``-inf``, and ``-inf * 0`` is NaN in the
    forward pass and NaN in the backward one. ``masked_fill`` replaces the value
    outright and zeroes the gradient there, so neither ever sees it.

    This mirrors ``train.unit_loss``: the same sentinel, the same slots, on both
    sides of the behaviour-cloning-to-PPO handover. The market head drops
    nothing, because none of its slots is padding.

    **The quantity term is conditional on the op that was sampled beside it.**
    Every unit gets a bucket, because the head emits one per slot, but the
    engine reads ``action[2]`` only for ``PICKUP`` and ``PLACE``
    (``TRANSFER_OPS``). A bucket that was never executed is not part of the
    action the episode took, and scoring it puts a variable the environment
    ignored into the importance ratio -- where it produces a gradient, changes
    no reward, and raises nothing. The condition comes from the sampled *op*
    via ``transfer_slots`` and never from the bucket's value: an unspent bucket
    and a genuine transfer of one item are the same number.

    Args:
        units: ``(rows, MAX_UNITS, len(UNIT_OPS))`` masked log-probabilities.
        unit_quantities: ``(rows, MAX_UNITS, len(QUANTITIES))`` likewise, from
            the per-unit quantity head.
        market: ``(rows, len(MARKET_SLOTS) + 2, len(QUANTITIES))`` likewise.
        unit_actions: ``(rows, MAX_UNITS)`` op indices, ``IGNORE`` where padded.
        unit_quantity_actions: ``(rows, MAX_UNITS)`` bucket indices, one per
            slot and never ``IGNORE``. Padding is stated once, on
            ``unit_actions``, and a padded slot is not a transfer, so its
            bucket drops out through the same condition the unspent ones do.
        market_actions: ``(rows, len(MARKET_SLOTS) + 2)`` bucket indices.

    Returns:
        ``(rows,)`` joint log-probabilities.
    """
    chosen = units.gather(2, unit_actions.clamp(min=0)[:, :, None]).squeeze(-1)
    transferred = unit_quantities.gather(2, unit_quantity_actions[:, :, None]).squeeze(
        -1
    )
    return (
        chosen.masked_fill(unit_actions == IGNORE, 0.0).sum(dim=1)
        + transferred.masked_fill(~transfer_slots(unit_actions), 0.0).sum(dim=1)
        + market.gather(2, market_actions[:, :, None]).squeeze(-1).sum(dim=1)
    )


def entropy_of(log_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return the entropy of each masked row.

    Only the legal options are summed. An illegal option's log-probability is
    ``-inf`` and its probability is exactly zero, and ``0 * -inf`` is NaN rather
    than the zero the limit gives -- so the log term is filled before it is
    multiplied, which also keeps the gradient at those entries at zero instead
    of NaN.

    Args:
        log_probs: ``(rows, slots, options)`` masked log-probabilities.
        mask: ``(rows, slots, options)`` bool, True where the option is legal.

    Returns:
        ``(rows, slots)`` entropies, in nats.
    """
    return -(log_probs.exp() * log_probs.masked_fill(~mask, 0.0)).sum(dim=-1)


def divergence_of(
    log_probs: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Return ``KL(policy || teacher)`` for each masked row.

    This direction, not the other: penalising ``KL(policy || teacher)`` punishes
    the policy for putting mass where the teacher put none, which is what keeps
    the first updates from wandering off a competent behaviour. The reverse
    would punish it for failing to cover everything the teacher does, which is
    an instruction to stay average.

    Both distributions are gated by the *same* stored mask, so the two agree on
    which options exist and the sum has no term where one side is ``-inf`` and
    the other is not. Masking the teacher separately would make the divergence
    infinite on any turn the two supports disagreed.

    Args:
        log_probs: ``(rows, slots, options)`` the policy's masked log-probs.
        teacher: ``(rows, slots, options)`` the teacher's, under the same mask.
        mask: ``(rows, slots, options)`` bool, True where the option is legal.

    Returns:
        ``(rows, slots)`` divergences, in nats.
    """
    return (log_probs.exp() * (log_probs - teacher).masked_fill(~mask, 0.0)).sum(dim=-1)
