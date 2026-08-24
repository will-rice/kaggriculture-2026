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
# THEIR DIGITS ARE 0.999 AND OUR SEASON IS TWICE THEIR GAME, so transcribing the
# number does not transcribe what it meant. A discount says how much of a payoff
# at the horizon is still visible from the first turn, and that quantity is
# `gamma ** turns`, not `gamma`. Lux runs MAX_DAYS = 360 turns, where 0.999
# leaves 0.999 ** 359 = 0.698 of the terminal result standing at turn 0. Our
# season is 719 decisions, where the same digits leave 0.999 ** 718 = 0.488 --
# the terminal objective damped 30% harder than the one they tuned, in a game
# whose entire payoff is terminal and whose shaped reward carries that payoff at
# 10x from turn one.
#
# Preserving their retention over our length is one equation:
#
#   gamma ** 719 = 0.999 ** 360   =>   gamma = 0.999 ** (360 / 719) = 0.9994992
#
# which is the same statement as holding the effective horizon at a fixed
# multiple of the episode: 1 / (1 - 0.999) = 1,000 turns is 2.78 of their games,
# and 1 / (1 - 0.9995) = 2,000 turns is 2.78 of ours. Rounded to 0.9995 in the
# style of their own constants, the retention is 0.698 against their 0.698.
#
# Rejected alternative: keep 0.999, which the grounding note calls low-risk on
# the grounding that the half-life (693 turns) is about our episode length. That
# reasoning prices the *median* turn, not the horizon, and the horizon is where
# this game's whole score is decided.
DISCOUNTING = 0.9995
LMB = 0.8
ENTROPY_COST = 0.001
LEARNING_RATE = 1e-4
ADAM_EPS = 0.0003
MIN_LR_MOD = 0.01
# DECLARED DEVIATION, and the only variable in this arm. Their phase 1 is
# `total_steps: 2e7`, which is what every arm this project has run has used --
# and phase 1 is one of five, ~9e7 across the published cascade. Ten arms have
# now failed at a tenth to a fifteenth of the budget the recipe was won on, so
# "we never trained long enough" is untested rather than refuted. 5x their
# phase 1 settles it. The LR schedule is keyed on this, so lengthening the
# budget stretches the decay rather than leaving it floored at 2e7.
TOTAL_STEPS = int(1e8)
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
# conv_phase2_game_result.yaml:78. Their phase-2 transition trains the value head
# alone for this many batches before the policy gradient fires, built for exactly
# the situation arm C hit: a competent policy meeting a reward its critic has not
# learned yet. Random init has nothing to protect and skips it; clone init does not.
VALUE_WARMUP_BATCHES = 4000
# conv_phase2_game_result.yaml:61. Their phase-2 value -- the first phase that
# continues from a competent policy. teacher_baseline_cost is absent there and
# inherits conv_config.yaml's 0.0, so only the KL term is active. Toad never
# continues from a trained policy without one; clone init with kl=0 is a
# configuration their recipe does not contain.
TEACHER_KL_COST = 0.005


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
        teacher: The teacher-KL term, already scaled. Zero when no teacher.
        total: What to call ``.backward()`` on.
    """

    vtrace_pg: torch.Tensor
    upgo_pg: torch.Tensor
    baseline: torch.Tensor
    entropy: torch.Tensor
    teacher: torch.Tensor
    total: torch.Tensor
    intermediates: dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)


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
    baseline_only: bool = False,
    teacher_kl: torch.Tensor | None = None,
    teacher_kl_cost: float = TEACHER_KL_COST,
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
        discounting: Gamma. Their 0.999 over a 360-turn game, rescaled to
            0.9995 so that our 719-turn season retains the same fraction of a
            terminal payoff at turn 0; see ``DISCOUNTING``.
        lmb: Lambda for both TD(lambda) and UPGO. Their 0.8 for phases 1-4.
        entropy_cost: Coefficient on the entropy term.
        reduction: Passed to ``reduce``.
        baseline_only: Train the value head alone, monobeast.py:416-418. The
            policy gradient and entropy terms are excluded from the total so a
            warm-started policy is not dragged around by advantages from an
            untrained critic. Note that their version still backpropagates the
            baseline loss through the shared trunk, so warmup is not perfectly
            policy-neutral even in their code; that is what they shipped and won
            with, and it is reproduced rather than corrected.
        teacher_kl: ``(turns, batch)`` per-step KL(teacher || learner), or None
            for the teacher-free phases. monobeast.py:129-140.
        teacher_kl_cost: Coefficient on that KL. Their cascade runs 0. in phase
            1, 0.005 in phase 2 and 0.001 from phase 3, and this project has
            bracketed the window: 0.005 holds and 0.001 cliffs in five updates
            when the reward is not yet self-sufficient.

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
    importance_ratios = vtrace_returns.log_rhos.exp()
    upgo_clipped_importance = torch.minimum(
        importance_ratios, torch.ones_like(vtrace_returns.log_rhos)
    ).detach()
    upgo_pg = policy_gradient(
        learner_log_probs, upgo_clipped_importance * upgo_returns.advantages, reduction
    )
    baseline = baseline_loss(values, td_lambda_returns.vs, reduction)
    # monobeast.py:412. Added, not subtracted: `negative_entropy` is already
    # sum p*log p, so minimising the total broadens the policy.
    entropy = entropy_cost * reduce(negative_entropy, reduction)
    # monobeast.py:400-403. Their KL is F.kl_div(learner_log_probs, teacher_probs),
    # i.e. KL(teacher || learner) -- the forward direction, which penalises the
    # learner for putting low mass where the teacher puts high mass. Our own
    # ppo.divergence_of is the reverse direction and is deliberately not reused.
    teacher = (
        torch.zeros_like(baseline)
        if teacher_kl is None
        else teacher_kl_cost * reduce(teacher_kl, reduction)
    )

    return Losses(
        vtrace_pg=vtrace_pg,
        upgo_pg=upgo_pg,
        baseline=baseline,
        entropy=entropy,
        teacher=teacher,
        total=baseline
        if baseline_only
        else vtrace_pg + upgo_pg + baseline + entropy + teacher,
        intermediates={
            "importance_ratios": importance_ratios,
            "vtrace/log_rhos": vtrace_returns.log_rhos,
            "vtrace/values": vtrace_returns.vs,
            "vtrace/pg_advantages": vtrace_returns.pg_advantages,
            "vtrace/behaviour_log_probs": vtrace_returns.behavior_action_log_probs,
            "vtrace/learner_log_probs": vtrace_returns.target_action_log_probs,
            "upgo/clipped_importance": upgo_clipped_importance,
            "upgo/values": upgo_returns.vs,
            "upgo/advantages": upgo_returns.advantages,
            "td_lambda/value_targets": td_lambda_returns.vs,
            "td_lambda/advantages": td_lambda_returns.advantages,
        },
    )
