"""Tests for Toad's IMPALA loss.

These guard the three things a PPO substitution would silently get wrong: that
UPGO is a distinct return and not a second copy of TD(lambda), that the entropy
term is signed so it broadens rather than sharpens the policy, and that the
off-policy correction actually engages when the actor lags.
"""

import dataclasses

import torch

from kaggriculture.learn import toad_loss
from kaggriculture.learn.toad.core import td_lambda, upgo

TURNS, BATCH = 8, 3


@dataclasses.dataclass(frozen=True)
class Batch:
    """A small synthetic segment with no episode boundary inside it."""

    rewards: torch.Tensor
    values: torch.Tensor
    bootstrap_value: torch.Tensor
    dones: torch.Tensor


def _batch(seed: int = 0) -> Batch:
    """Return a reproducible synthetic batch."""
    generator = torch.Generator().manual_seed(seed)
    return Batch(
        rewards=torch.randn(TURNS, BATCH, generator=generator),
        values=torch.randn(TURNS, BATCH, generator=generator),
        bootstrap_value=torch.randn(BATCH, generator=generator),
        dones=torch.zeros(TURNS, BATCH, dtype=torch.bool),
    )


def _losses(
    batch: Batch,
    behaviour: torch.Tensor,
    learner: torch.Tensor,
    negative_entropy: torch.Tensor,
    dones: torch.Tensor | None = None,
) -> toad_loss.Losses:
    """Call the loss with one batch, spelled out rather than unpacked."""
    return toad_loss.losses(
        behaviour_log_probs=behaviour,
        learner_log_probs=learner,
        negative_entropy=negative_entropy,
        values=batch.values,
        bootstrap_value=batch.bootstrap_value,
        rewards=batch.rewards,
        dones=batch.dones if dones is None else dones,
    )


def _returns(batch: Batch, which: str) -> torch.Tensor:
    """Return UPGO or TD(lambda) value targets for a batch."""
    discounts = (~batch.dones).float() * toad_loss.DISCOUNTING
    function = upgo.upgo if which == "upgo" else td_lambda.td_lambda
    return function(
        rewards=batch.rewards,
        values=batch.values,
        bootstrap_value=batch.bootstrap_value,
        discounts=discounts,
        lmb=toad_loss.LMB,
    ).vs


def test_upgo_targets_differ_from_td_lambda() -> None:
    """UPGO's max() must make it a different return from TD(lambda).

    If these ever agree, the UPGO policy gradient has collapsed into a second
    copy of the baseline target and the component Toad's method depends on is
    gone, which is exactly the failure a PPO port makes quietly.
    """
    batch = _batch()
    assert not torch.allclose(_returns(batch, "upgo"), _returns(batch, "td"))


def test_upgo_target_is_never_below_td_lambda() -> None:
    """UPGO bootstraps from the better of the value and the lambda-return.

    That is the whole point of it: it only pulls toward better-than-expected
    outcomes, so its target dominates TD(lambda) pointwise.
    """
    batch = _batch(seed=1)
    assert (_returns(batch, "upgo") >= _returns(batch, "td") - 1e-6).all()


def test_on_policy_importance_weights_are_inert() -> None:
    """With no actor lag the correction must be well defined, not merely small."""
    batch = _batch(seed=2)
    log_probs = torch.full((TURNS, BATCH), -1.5)
    result = _losses(batch, log_probs, log_probs.clone(), torch.zeros(TURNS, BATCH))
    assert torch.isfinite(result.total)


def test_a_lagging_actor_damps_the_gradient() -> None:
    """An action the learner now thinks less likely must count for less.

    V-trace clips the importance ratio at 1, so a learner that has moved away
    from the behaviour policy gets a strictly damped policy gradient. Without
    this the off-policy correction is decorative.
    """
    batch = _batch(seed=3)
    behaviour = torch.full((TURNS, BATCH), -1.0)
    entropy = torch.zeros(TURNS, BATCH)
    on_policy = _losses(batch, behaviour, behaviour.clone(), entropy)
    lagged = _losses(batch, behaviour, torch.full((TURNS, BATCH), -4.0), entropy)
    assert lagged.upgo_pg.abs() < on_policy.upgo_pg.abs()


def test_entropy_term_rewards_a_broader_policy() -> None:
    """A broader policy must lower the loss, or the bonus is inverted.

    ``combine_policy_entropy`` returns sum p*log p, which is negative entropy,
    and monobeast *adds* it. Getting this sign wrong trains the policy to
    collapse onto one action, which looks like fast early progress.
    """
    batch = _batch(seed=4)
    log_probs = torch.full((TURNS, BATCH), -1.0)
    # Uniform over two actions: sum p*log p = log(0.5). Nearly deterministic
    # policies approach 0 from below, so they must score a *higher* loss.
    broad = _losses(
        batch,
        log_probs,
        log_probs.clone(),
        torch.full((TURNS, BATCH), torch.tensor(0.5).log().item()),
    )
    sharp = _losses(
        batch, log_probs, log_probs.clone(), torch.full((TURNS, BATCH), -0.01)
    )
    assert broad.entropy < sharp.entropy


def test_total_is_the_sum_of_its_parts() -> None:
    """The reported terms must add up to what backward() is called on."""
    batch = _batch(seed=5)
    log_probs = torch.full((TURNS, BATCH), -1.0)
    result = _losses(
        batch, log_probs, log_probs.clone(), torch.full((TURNS, BATCH), -0.3)
    )
    assert torch.allclose(
        result.total,
        result.vtrace_pg + result.upgo_pg + result.baseline + result.entropy,
    )


def test_a_done_step_stops_the_return_leaking_across_it() -> None:
    """Discount must go to zero on a done step, isolating episodes in a segment."""
    batch = _batch(seed=6)
    log_probs = torch.full((TURNS, BATCH), -1.0)
    entropy = torch.zeros(TURNS, BATCH)
    without = _losses(batch, log_probs, log_probs.clone(), entropy)
    boundary = batch.dones.clone()
    boundary[TURNS // 2] = True
    with_boundary = _losses(
        batch, log_probs, log_probs.clone(), entropy, dones=boundary
    )
    assert not torch.allclose(without.baseline, with_boundary.baseline)


def test_the_phase_one_constants_are_the_published_ones() -> None:
    """Pin the phase-1 hyperparameters to their literal published values.

    Stated as bare literals rather than by reference, so that a drifting
    constant fails here instead of silently agreeing with itself. Read from
    conf/conv_phase1_shaped_reward.yaml.
    """
    assert toad_loss.DISCOUNTING == 0.999
    assert toad_loss.LMB == 0.8
    assert toad_loss.ENTROPY_COST == 0.001
    assert toad_loss.LEARNING_RATE == 1e-4
    assert toad_loss.ADAM_EPS == 0.0003
    assert toad_loss.TOTAL_STEPS == 20_000_000
    assert toad_loss.UNROLL_LENGTH == 16
    assert toad_loss.REDUCTION == "sum"
