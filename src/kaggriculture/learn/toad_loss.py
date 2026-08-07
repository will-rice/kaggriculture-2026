"""Toad Brigade's IMPALA loss, over our trajectories.

Their learner is IMPALA with V-trace plus a UPGO policy gradient, and the
baseline regresses on TD(lambda) rather than on either of them. PPO has no UPGO
analogue, so swapping in PPO drops a component of the method rather than
simplifying it; this module exists so we run their loss and not something
shaped like it.

Everything here is a transcription of ``toad/monobeast.py`` lines 395-425, and
the three return computations come from ``toad/core/{vtrace,upgo,td_lambda}.py``
unmodified. Their line numbers are quoted beside each step so the two can be
diffed by eye. What this module does *not* reproduce is monobeast's
multiprocessing actor/learner split -- see ``Lag`` below for what replaces it and
why the difference does not change the loss.

Two details of theirs are easy to get backwards and are called out where they
happen: ``combine_policy_entropy`` returns *negative* entropy, so the entropy
term is added rather than subtracted (monobeast.py:117, :412); and
``flags.baseline_cost`` is never read by their loss at all (only
``teacher_baseline_cost`` is), so the baseline enters unweighted. Their configs
all set ``baseline_cost: 1.``, which is why the dead flag is invisible there.
"""

import dataclasses

import torch

from kaggriculture.learn.toad.core import td_lambda, upgo, vtrace

# Phase 1 of their five, from conf/conv_phase1_shaped_reward.yaml. Phase 1 is
# the pure baseline: it is the only phase with `teacher_kl_cost: 0.` (:66), so
# it is genuinely teacher-free self-play from random initialisation, and none of
# the teacher machinery in monobeast is needed to run it.
DISCOUNTING = 0.999
LMB = 0.8
ENTROPY_COST = 0.001
LEARNING_RATE = 1e-4
ADAM_EPS = 0.0003
MIN_LR_MOD = 0.01
TOTAL_STEPS = int(2e7)
UNROLL_LENGTH = 16
REDUCTION = "sum"
# monobeast.py:502-505 clips the global gradient norm before every optimizer
# step. The value is NOT in the five phase YAMLs -- it is in the run config saved
# beside each of their trained checkpoints (e.g.
# internal_testing/hall_of_fame/11-09_21-32-04_59822400/lux_ai/rl_agent/config.yaml:53),
# where every one of the eight reads `clip_grads: 10.0`. Leaving it out is not a
# small omission: with `reduction: sum` and a joint log-probability over ~41
# decisions a turn, the run diverged to a loss of 6e20 within two updates.
CLIP_GRADS = 10.0


@dataclasses.dataclass(frozen=True)
class Losses:
    """The four terms and their sum, each already scaled by its coefficient.

    Attributes:
        vtrace_pg: Policy gradient against V-trace advantages.
        upgo_pg: Policy gradient against UPGO advantages, themselves scaled by
            the clipped V-trace importance weights.
        baseline: Smooth L1 of the value head against TD(lambda) targets.
        entropy: The entropy term, already multiplied by ``ENTROPY_COST``.
            Negative entropy, so a smaller total loss means a broader policy.
        total: What to call ``.backward()`` on.
    """

    vtrace_pg: torch.Tensor
    upgo_pg: torch.Tensor
    baseline: torch.Tensor
    entropy: torch.Tensor
    total: torch.Tensor


def reduce(losses: torch.Tensor, reduction: str = REDUCTION) -> torch.Tensor:
    """Sum or mean a loss tensor.

    Transcribed from monobeast.py:143-149. Their configs all use ``sum``, which
    makes the loss scale with the batch shape; that is why ``UNROLL_LENGTH`` is
    reproduced rather than chosen.

    Args:
        losses: Any tensor of per-element losses.
        reduction: Either ``"sum"`` or ``"mean"``.

    Returns:
        The reduced scalar.

    Raises:
        ValueError: If ``reduction`` is neither ``"sum"`` nor ``"mean"``.
    """
    if reduction == "mean":
        return losses.mean()
    if reduction == "sum":
        return losses.sum()
    raise ValueError(f"Reduction must be one of 'sum' or 'mean', was: {reduction}")


def policy_gradient(
    action_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    reduction: str = REDUCTION,
) -> torch.Tensor:
    """Return the policy gradient loss for one set of advantages.

    Transcribed from monobeast.py:157-163. The advantages are detached, so this
    is the REINFORCE term and the only gradient path is through
    ``action_log_probs``.

    Args:
        action_log_probs: ``(turns, batch)`` log-probability of the action the
            episode took, under the *current* parameters.
        advantages: ``(turns, batch)`` advantage estimates.
        reduction: Passed to ``reduce``.

    Returns:
        The reduced scalar loss.
    """
    cross_entropy = -action_log_probs.view_as(advantages)
    return reduce(cross_entropy * advantages.detach(), reduction)


def baseline_loss(
    values: torch.Tensor, targets: torch.Tensor, reduction: str = REDUCTION
) -> torch.Tensor:
    """Return the smooth L1 value loss against detached targets.

    Transcribed from monobeast.py:152-154. Note that their ``flags.baseline_cost``
    never reaches this call site, so the term enters the total unweighted.

    Args:
        values: ``(turns, batch)`` value head output.
        targets: ``(turns, batch)`` value targets, detached here.
        reduction: Passed to ``reduce``.

    Returns:
        The reduced scalar loss.
    """
    per_element = torch.nn.functional.smooth_l1_loss(
        values, targets.detach(), reduction="none"
    )
    return reduce(per_element, reduction=reduction)


def losses(
    behaviour_log_probs: torch.Tensor,
    learner_log_probs: torch.Tensor,
    negative_entropy: torch.Tensor,
    values: torch.Tensor,
    bootstrap_value: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    discounting: float = DISCOUNTING,
    lmb: float = LMB,
    entropy_cost: float = ENTROPY_COST,
    reduction: str = REDUCTION,
) -> Losses:
    """Return Toad's four loss terms for one batch of unrolled segments.

    Follows monobeast.py:355-425 step for step. Three separate return
    computations run over the same batch, and they are not interchangeable: the
    policy gradient uses V-trace, the baseline regresses on TD(lambda), and UPGO
    supplies a second policy gradient whose target bootstraps from the *better*
    of the value and the lambda-return, so it only pushes toward trajectories
    that beat expectation.

    All three are ``@torch.no_grad()`` internally, so the only gradient paths out
    of here are ``learner_log_probs``, ``values`` and ``negative_entropy``.

    Args:
        behaviour_log_probs: ``(turns, batch)`` joint log-probability of each
            turn's whole action under the policy that actually played it. Joint
            across every unit and market slot, matching how ``rollout`` records
            it and how their ``combined_behavior_action_log_probs`` sums across
            action spaces.
        learner_log_probs: ``(turns, batch)`` the same actions re-scored under
            current parameters. Equal to ``behaviour_log_probs`` only if the
            actor has not lagged, in which case V-trace correctly degenerates.
        negative_entropy: ``(turns, batch)`` sum of ``p * log p`` over each
            head, i.e. *negative* entropy, matching monobeast.py:117.
        values: ``(turns, batch)`` value head output at each state.
        bootstrap_value: ``(batch,)`` value of the state after the last one.
        rewards: ``(turns, batch)`` reward following each action.
        dones: ``(turns, batch)`` bool, True where the episode ended.
        discounting: Gamma. Their 0.999 across all five phases.
        lmb: Lambda for both TD(lambda) and UPGO. Their 0.8 for phases 1-4.
        entropy_cost: Coefficient on the entropy term.
        reduction: Passed to ``reduce``.

    Returns:
        The four terms and their sum.
    """
    # monobeast.py:355-356. A done step gets zero discount, which is what stops
    # a return leaking across an episode boundary inside a segment.
    discounts = (~dones).float() * discounting

    vtrace_returns = vtrace.from_action_log_probs(
        behavior_action_log_probs=behaviour_log_probs,
        target_action_log_probs=learner_log_probs,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
    )
    td_lambda_returns = td_lambda.td_lambda(
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        discounts=discounts,
        lmb=lmb,
    )
    upgo_returns = upgo.upgo(
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        discounts=discounts,
        lmb=lmb,
    )

    vtrace_pg = policy_gradient(
        learner_log_probs, vtrace_returns.pg_advantages, reduction
    )
    # monobeast.py:386-394. UPGO's advantages carry the clipped V-trace
    # importance weight, so a stale actor damps the UPGO term the same way it
    # damps the V-trace one.
    upgo_clipped_importance = torch.minimum(
        vtrace_returns.log_rhos.exp(), torch.ones_like(vtrace_returns.log_rhos)
    ).detach()
    upgo_pg = policy_gradient(
        learner_log_probs, upgo_clipped_importance * upgo_returns.advantages, reduction
    )
    baseline = baseline_loss(values, td_lambda_returns.vs, reduction)
    # monobeast.py:412. Added, not subtracted: `negative_entropy` is already
    # sum p*log p, so minimising the total broadens the policy.
    entropy = entropy_cost * reduce(negative_entropy, reduction)

    return Losses(
        vtrace_pg=vtrace_pg,
        upgo_pg=upgo_pg,
        baseline=baseline,
        entropy=entropy,
        total=vtrace_pg + upgo_pg + baseline + entropy,
    )
