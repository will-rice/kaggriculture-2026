"""Measuring the critic against the return it is supposed to predict.

The ``baseline`` loss term is the value head's smooth-L1 distance from its own
bootstrapped TD(lambda) target. It is a measure of **self-consistency**, not of
accuracy, and the two come apart completely: the 2026-08-09 measurement found
checkpoints scoring 0.998 explained variance against their own target while
scoring **-12.5** against the actual discounted return, i.e. predicting the
return twelvefold worse than a constant would. Ten arms read the falling
``baseline`` term as "the critic is learning" and none of them measured whether
it was true.

So the number this module computes is logged every update rather than
reconstructed afterwards. ``explained_variance`` against ``monte_carlo`` is the
diagnostic that would have caught the defect in an hour instead of after ten
arms, and it is the one a future run has to look at before calling a falling
value loss progress.

The return comes from the learner's own vendored ``td_lambda`` at ``lmb=1.0``,
where the ``(1 - lmb)`` factor deletes the only term values enter through. That
makes it the pure Monte Carlo return-to-go and makes it impossible for the
diagnostic's return to disagree with the learner's target about the discount,
the ordering or the episode boundary.
"""

import torch

from kaggriculture.learn.toad.core import td_lambda
from kaggriculture.learn.toad_loss import DISCOUNTING


def monte_carlo(
    rewards: torch.Tensor, dones: torch.Tensor, discounting: float = DISCOUNTING
) -> torch.Tensor:
    """Return the discounted return-to-go over a whole episode.

    Args:
        rewards: ``(turns,)`` per-turn reward.
        dones: ``(turns,)`` bool, True on the last turn.
        discounting: Gamma.

    Returns:
        ``(turns,)`` discounted return-to-go.
    """
    return td_lambda.td_lambda(
        rewards=rewards,
        values=torch.zeros_like(rewards),
        bootstrap_value=torch.zeros(()),
        discounts=(~dones).float() * discounting,
        lmb=1.0,
    ).vs


def explained_variance(returns: torch.Tensor, values: torch.Tensor) -> float:
    """Return ``1 - Var(returns - values) / Var(returns)``.

    Zero is the score of a critic that has learned the mean and nothing else, so
    a negative reading means the critic is a worse predictor than a constant.

    Args:
        returns: Whatever the critic is being scored against.
        values: The critic's estimate at the same states, same shape.

    Returns:
        The explained variance, or NaN when the returns carry no variance.
    """
    total = float(returns.var(unbiased=False))
    if total <= 0.0:
        return float("nan")
    return 1.0 - float((returns - values).var(unbiased=False)) / total


def critic_scores(
    values: list[torch.Tensor],
    rewards: list[torch.Tensor],
    dones: list[torch.Tensor],
) -> dict[str, float]:
    """Score one population's critic against its real discounted returns.

    Two numbers, and they are not interchangeable. ``ev_vs_return`` pools every
    recorded state, and most of the return's pooled variance is the turn index:
    a critic that has learned only the calendar scores ~0.98 on it.
    ``ev_vs_return_within_turn`` centres both series at each turn index first, so
    the deterministic profile of the return-to-go in ``t`` leaves numerator and
    denominator alike, and what is left is whether the critic tells two episodes
    apart at the same point in the season. A critic that explains everything
    except what the policy did is exactly as useless for the policy gradient as
    one that explains nothing, and only the second number can see the difference.

    Args:
        values: One ``(turns,)`` tensor of critic estimates per episode.
        rewards: The matching per-turn reward series the learner reads.
        dones: The matching terminal flags.

    Returns:
        Both explained variances, NaN for an empty or ragged population -- a
        zero would read as "the critic explains nothing" rather than as the
        absence of a measurement.
    """
    if not values or len({tuple(v.shape) for v in values}) != 1:
        return {"ev_vs_return": float("nan"), "ev_vs_return_within_turn": float("nan")}
    value = torch.stack(values)
    returns = torch.stack(
        [monte_carlo(reward, done) for reward, done in zip(rewards, dones, strict=True)]
    )
    return {
        "ev_vs_return": explained_variance(returns.flatten(), value.flatten()),
        "ev_vs_return_within_turn": explained_variance(
            (returns - returns.mean(dim=0)).flatten(),
            (value - value.mean(dim=0)).flatten(),
        ),
    }
