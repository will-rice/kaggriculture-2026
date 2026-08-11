"""Tests for the PPO update.

Nothing here plays an episode. A rollout costs 17 seconds and every property
this file guards -- the advantage recursion, the clip, the decay, which mask the
ratio is taken under, which slots the padding excludes -- is a property of the
arithmetic and is visible on six synthetic turns. ``tests/learn/test_rollout.py``
is where the real environment is exercised.

The synthetic trajectories are built the way ``rollout`` builds real ones: a
random mask that always leaves option 0 legal, an action drawn from what that
mask permits, and a log-probability read off the masked log-softmax of the same
policy that will later be asked to reproduce it. That last step is what makes
``ratio == 1`` a meaningful assertion rather than a tautology -- the stored
number came from a distribution, not from a constant.

The masks are random rather than realistic on purpose. A realistic mask is one
the update could plausibly have rederived from the board; a random one is not,
so a ratio that comes out at 1 against it can only have been computed under the
mask that was stored.
"""

import copy
import dataclasses

import pytest
import torch

from kaggriculture.constants import STARTING_MONEY
from kaggriculture.learn.encoding import (
    BOARD,
    IGNORE,
    MARKET_SLOTS,
    MAX_UNITS,
    QUANTITIES,
    SCALARS,
    TILE_PLANES,
    UNIT_OPS,
)
from kaggriculture.learn.model import Policy
from kaggriculture.learn.ppo import (
    KL_STEPS,
    PROGRESS_HOLD,
    PROGRESS_STEPS,
    PpoConfig,
    advantages,
    flatten,
    joint_log_prob,
    kl_weight,
    policy_loss,
    progress_weight,
    reward_of,
    update,
)
from kaggriculture.learn.progress import POTENTIAL_COMPONENTS, progress_reward
from kaggriculture.learn.rollout import Trajectory

# The crew a synthetic turn is staffed with. Fewer than MAX_UNITS on purpose --
# most slots of most real turns hold nobody, which is the whole reason the
# padding has to be excluded -- and more than one, so "the padded slots" and
# "every slot but the first" are different sets and a test cannot pass by
# confusing them.
CREW = 3

# One update, one minibatch, so every metric returned is measured on the
# policy exactly as it was handed in rather than after some earlier minibatch
# has already moved it.
ONE_STEP = PpoConfig(epochs=1, minibatch=64)


def test_advantages_sum_toward_the_return() -> None:
    """GAE with lambda=1 and gamma=1 is the Monte Carlo advantage."""
    rewards = torch.ones(5)
    values = torch.zeros(6)

    computed = advantages(rewards, values, gamma=1.0, lam=1.0)

    assert computed[0] == pytest.approx(5.0)


def test_the_clip_bounds_the_step() -> None:
    """PPO's whole safety property: one batch cannot move the policy far."""
    ratio = torch.tensor([10.0])
    advantage = torch.tensor([1.0])

    loss = policy_loss(ratio, advantage, clip=0.2)

    assert loss == pytest.approx(-1.2)


def test_the_clip_does_not_shelter_a_bad_action() -> None:
    """The clip is a bound on optimism, not an amnesty on a mistake.

    The test above is satisfied by the clipped term alone, because with a
    positive advantage the minimum *is* the clipped one. Flip the advantage and
    the two part company: an action the new policy has made ten times more
    likely and that turned out to be bad must be pulled back by its full
    unclipped weight, and a loss built from ``clamp`` without the ``min`` would
    cap the correction at 1.2 and leave the policy holding it.
    """
    loss = policy_loss(torch.tensor([10.0]), torch.tensor([-1.0]), clip=0.2)

    assert loss == pytest.approx(10.0)


def test_the_teacher_penalty_falls_to_zero() -> None:
    """It exists to survive the first updates, not to pin us to a weak clone.

    Written against ``KL_STEPS`` rather than against literal step counts,
    because the number is set from a measured run length and has already moved
    once: it was 4,000, chosen before any run length existed, and Task 5's
    throughput made that three days of updates on a six-hour run. A test
    holding 1,000 and 10,000 passed at 4,000 and asserted nothing at 100.

    ``== 0.0`` exactly, not ``approx``. That is the whole difference between
    this schedule and an exponential one, which would satisfy every inequality
    here and every tolerance and still be tethering the learner to a clone that
    banks nothing on the last update of the run.
    """
    assert (
        kl_weight(step=0)
        > kl_weight(step=KL_STEPS // 4)
        > kl_weight(step=KL_STEPS // 2)
    )
    assert kl_weight(step=KL_STEPS) == 0.0
    assert kl_weight(step=KL_STEPS * 10) == 0.0


def test_the_differential_weight_chooses_which_reward_is_climbed() -> None:
    """The curriculum knob has to actually reach the advantage, or it is a comment.

    ``differential`` is what a run schedules from 0.0 to 1.0, and the failure it
    guards against is silent in every other metric: a ``flatten`` that ignored
    the weight would produce finite advantages, a finite value loss and a
    plausible policy loss on every setting, and the whole shaped phase would be
    training on the differential it was introduced to avoid.

    So the two endpoints are computed and required to differ from each other and
    each to equal the series it names. The synthetic trajectory's two series are
    independent draws, so a blend that returned either one unconditionally, or
    their sum, fails.

    Taken at ``progress=0.0`` so that this test says what it is named for and
    nothing else; the progress term has its own tests below.
    """
    policy = _policy(seed=0)
    trajectory = _trajectory(policy, turns=6, seed=3)

    relative = reward_of(trajectory, PpoConfig(differential=1.0, progress=0.0))
    absolute = reward_of(trajectory, PpoConfig(differential=0.0, progress=0.0))
    half = reward_of(trajectory, PpoConfig(differential=0.5, progress=0.0))

    assert torch.equal(relative, trajectory.rewards)
    assert torch.equal(absolute, trajectory.own)
    assert not torch.allclose(relative, absolute)
    assert torch.allclose(half, (trajectory.rewards + trajectory.own) / 2)


def test_the_progress_term_reaches_the_reward_and_its_weight_removes_it() -> None:
    """The shaping knob has to reach the advantage, or the whole change is inert.

    The failure this catches is the one that costs a run rather than a test: a
    ``reward_of`` that computed the progress term and dropped it, or a
    ``flatten`` that read the money series alone, would leave every logged
    number finite and plausible -- the potentials would still be recorded, the
    components would still be charted, and the gradient would still be the
    sparse one the plateau was measured on.

    Asserted in both directions. At weight 0 the reward is exactly the money
    reward, so the term is genuinely removable and the schedule that decays it
    means something; at weight 1 it is the money reward plus exactly
    ``progress_reward``, so the term is added rather than blended -- a blend
    would shrink the money half and is the other plausible implementation.
    """
    policy = _policy(seed=0)
    trajectory = _trajectory(policy, turns=6, seed=3)
    off = PpoConfig(differential=0.0, progress=0.0)
    on = PpoConfig(differential=0.0, progress=1.0)

    assert torch.equal(reward_of(trajectory, off), trajectory.own)
    assert torch.allclose(
        reward_of(trajectory, on),
        trajectory.own + progress_reward(trajectory.potentials, on.gamma),
    )
    assert not torch.allclose(reward_of(trajectory, on), reward_of(trajectory, off))


def test_the_progress_term_cannot_be_farmed_by_any_cycle() -> None:
    """A potential pays for where you end up, never for how many times you got there.

    This is the property the whole design rests on, and it is the one the brief
    names as the failure mode: an agent that maximises the shaped term without
    banking. Under ``gamma * P(s') - P(s)`` that is not a thing an agent can
    choose to do, because the shaped return over a sequence of states depends
    only on its endpoints -- so a policy that plants and digs a tile a hundred
    times collects exactly what a policy that never touched it collects.

    Checked by construction rather than by assertion about a run: a season is
    built that returns to its opening state, and a second one that visits ten
    times as many intermediate states on the way to the same place. At
    ``gamma == 1`` both return exactly zero.

    The ``gamma < 1`` half is the other half of the property and is why the
    checks are separate. The real discount is 0.999, and under it the *longer*
    cycle pays strictly less than the shorter one: holding stock costs
    ``(1 - gamma)`` of its value every turn it is held, so churn is not merely
    unprofitable but charged rent. An implementation that dropped the ``gamma``
    -- the defect this project already documented in its own differential
    reward -- passes the first check and fails this one.
    """
    opening = torch.full((1, len(POTENTIAL_COMPONENTS)), 50.0)
    short = torch.cat(
        [opening, torch.full((4, len(POTENTIAL_COMPONENTS)), 900.0), opening]
    )
    long = torch.cat(
        [opening, torch.full((40, len(POTENTIAL_COMPONENTS)), 900.0), opening]
    )

    # Both episodes stop in the state they opened in, so the shaped return is
    # the constant the telescope leaves behind -- the terminal zero less the
    # opening potential -- and it is the *same* constant for both, however many
    # states the long one visited on the way.
    settled = -float(opening.sum())

    assert float(progress_reward(short, gamma=1.0).sum()) == pytest.approx(settled)
    assert float(progress_reward(long, gamma=1.0).sum()) == pytest.approx(settled)
    assert float(progress_reward(long, gamma=0.999).sum()) < float(
        progress_reward(short, gamma=0.999).sum()
    )
    assert float(progress_reward(short, gamma=0.999).sum()) < settled


def test_a_season_that_ends_holding_stock_does_not_out_score_one_that_sold_it() -> None:
    """The reward must not pay for hoarding, and this is where it nearly did.

    The shaped return over an episode telescopes to
    ``gamma**N * P(s_N) - P(s_0)``. The opening term is a constant and harmless.
    The *closing* one is a function of the state the agent chose to stop in, so
    Grzes (AAMAS 2017, Eq. 3) requires the potential at a trajectory's stopping
    state to be zero -- otherwise, in his words, "this term can modify the
    policy".

    Un-zeroed it modifies it in exactly the direction this project keeps
    failing in. A unit held to the horizon returns ``gamma ** (N - t)`` of its
    base price, which at 0.999 over a 719-turn season is 90% from turn 619 and
    98% from turn 700 -- so once both farms have pushed the market below base
    by selling into it, *not selling* becomes the shaped-optimal play for the
    last quarter of every episode. The shed fills, the bank stays flat, and
    every chart looks like the shaping is working.

    Asserted behaviourally rather than by inspecting the terminal row, which
    would only restate the implementation. Two episodes, identical up to the
    turn where a full shed is either sold at a 10% discount to base or held to
    the horizon. Selling has to win. With the terminal potential carried it
    loses 90 to 100, which is the whole failure in two numbers.
    """
    full = [25.0] * len(POTENTIAL_COMPONENTS)
    empty = [0.0] * len(POTENTIAL_COMPONENTS)
    held = torch.tensor([empty, full, full, full])
    sold = torch.tensor([empty, full, empty, empty])
    # Sold on turn 1 for 10% under base, which is the sort of quote a market
    # both farms are selling into gives back. The hoarding incentive has to be
    # worth less than that discount, or the agent is right to sit on the shed.
    takings = torch.tensor([0.0, 0.9 * sum(full), 0.0, 0.0])

    for gamma in (1.0, 0.999):
        selling = _discounted(takings + progress_reward(sold, gamma), gamma)
        hoarding = _discounted(progress_reward(held, gamma), gamma)

        assert selling > hoarding


def _discounted(rewards: torch.Tensor, gamma: float) -> float:
    """Return the discounted sum of a reward series, which is what PPO climbs."""
    return float((rewards * gamma ** torch.arange(len(rewards))).sum())


def test_the_progress_term_refuses_a_series_that_is_not_turn_major() -> None:
    """One row per acting turn, and no terminal row -- an extra row is off by one.

    The terminal zero belongs to ``progress_reward`` and is appended there, so a
    caller handing over a state-major array with the terminal already in it is
    both one turn too long and reintroducing the hoarding incentive. Raising
    here names the array that is wrong; letting it through would produce one
    reward too many and fail somewhere else entirely.
    """
    with pytest.raises(ValueError, match="no terminal row"):
        progress_reward(torch.zeros(6), gamma=1.0)
    with pytest.raises(ValueError, match="no terminal row"):
        progress_reward(torch.zeros(6, len(POTENTIAL_COMPONENTS) + 1), gamma=1.0)


def test_the_progress_weight_holds_then_falls_to_zero() -> None:
    """A curriculum term has to be at full strength while the chain is unlearned.

    The difference from ``kl_weight``, and the reason it is not the same
    function: the teacher penalty starts relaxing on update 1, while this one
    must not. A schedule that began decaying immediately would be weakest
    exactly where the policy still cannot bank, which is the condition the term
    exists for.

    ``== 0.0`` exactly at the end, for the same reason the teacher penalty's
    test demands it: a term that is still shaping on the last update of the run
    has not been shown to work without the shaping.
    """
    assert progress_weight(step=0) == progress_weight(step=PROGRESS_HOLD) == 1.0
    assert 1.0 > progress_weight(step=(PROGRESS_HOLD + PROGRESS_STEPS) // 2) > 0.0
    assert progress_weight(step=PROGRESS_STEPS) == 0.0
    assert progress_weight(step=PROGRESS_STEPS * 10) == 0.0


def test_the_progress_weight_refuses_a_schedule_with_no_ramp() -> None:
    """``hold >= steps`` is a cliff, not a decay, and would divide by zero saying so."""
    with pytest.raises(ValueError, match="ramp"):
        progress_weight(step=0, hold=200, steps=200)


def test_the_shaped_reward_survives_a_zero_sum_batch() -> None:
    """Two seats of one mirrored episode cancel on the differential, not on ``own``.

    This is the configuration the run actually collects in: one policy plays
    both seats, so the batch holds a trajectory and its exact negation. On
    ``differential=1.0`` those two advantage series are equal and opposite and
    the batch's mean advantage is zero by construction -- which is harmless when
    the rewards are large and fatal when they are not, because a bankrupt mirror
    makes every term zero rather than merely balanced.

    On ``differential=0.0`` the two seats carry their own banks, which are not
    negations of each other, so the spread survives. The assertion is on the
    spread rather than the mean: a zero mean is what advantage normalisation
    wants, and it is the *scale* that says whether there is anything to learn.
    """
    policy = _policy(seed=0)
    ours = _trajectory(policy, turns=6, seed=4)
    theirs = dataclasses.replace(ours, rewards=-ours.rewards)

    # Progress off throughout, so this test measures the cancellation it is
    # named for. The progress term is not zero-sum -- both seats build their own
    # pipeline -- so leaving it on would keep the spread alive for a reason that
    # has nothing to do with `differential`, which is what is under test.
    relative = flatten([ours, theirs], PpoConfig(differential=1.0, progress=0.0))
    absolute = flatten([ours, theirs], PpoConfig(differential=0.0, progress=0.0))

    zero_sum = PpoConfig(differential=1.0, progress=0.0)
    absolute_only = PpoConfig(differential=0.0, progress=0.0)
    assert float(
        reward_of(ours, zero_sum).sum() + reward_of(theirs, zero_sum).sum()
    ) == pytest.approx(0.0)
    assert (
        float(
            reward_of(ours, absolute_only).sum()
            + reward_of(theirs, absolute_only).sum()
        )
        != 0.0
    )
    assert relative.returns.std() > 0.0
    assert absolute.returns.std() > 0.0


def test_the_ratio_is_taken_under_the_stored_mask() -> None:
    """``exp(new - old)`` is an importance weight only over one support.

    The masks below are random, so no rule could rederive them from the board:
    if the update gated the fresh logits with anything other than the mask
    stored beside the action -- a mask recomputed from the observation, a mask
    of all-True, the teacher's own -- ``new`` would be a log-probability of a
    different distribution than ``old``, the ratio would not be 1 for a policy
    nobody has touched, and nothing in the loss would raise.

    The second half is what stops the first half being satisfied by an update
    that ignores the stored log-probability altogether: shift ``old`` by a
    known amount and the ratio has to move by exactly ``exp`` of it.
    """
    policy = _policy(seed=0)
    trajectory = _trajectory(policy, turns=6, seed=1)

    metrics = _metrics(policy, trajectory)
    shifted = dataclasses.replace(trajectory, log_probs=trajectory.log_probs + 0.5)
    moved = _metrics(policy, shifted)

    assert float(trajectory.unit_masks.float().mean()) < 0.7
    assert float(trajectory.market_masks.float().mean()) < 0.7
    assert metrics["ratio"] == pytest.approx(1.0, abs=1e-4)
    assert moved["ratio"] == pytest.approx(float(torch.tensor(-0.5).exp()), abs=1e-4)


def test_padded_unit_slots_contribute_nothing() -> None:
    """A padded slot holds no decision, and three terms must all agree on that.

    ``rollout`` marks the slots no unit stood in with ``IGNORE``, the same
    sentinel ``encode_units`` writes and ``train.unit_loss`` masks on. The
    stored mask on those slots is still a real tensor, so the way to ask
    whether they are being scored is to change it: narrow every padded slot down
    to PASS alone and see whether the policy loss, the entropy bonus or the
    teacher KL notices. None of them may.

    The same narrowing applied to the slots that *do* hold a unit moves all
    three, which is what keeps this from passing on an update that ignores the
    masks entirely -- and it is the assertion the earlier padding test in this
    project was missing, when it asserted PyTorch's default ``ignore_index``
    rather than our own code and passed with the argument deleted.

    Every metric is checked finite first. The masks here forbid roughly half of
    every row, so any term that sums ``0 * -inf`` over the illegal options --
    the entropy bonus and the teacher KL both would, written the obvious way --
    returns NaN, and two NaNs compare unequal, which would turn the comparisons
    below into a test that passes on a broken update.
    """
    policy = _policy(seed=0)
    trajectory = _trajectory(policy, turns=6, seed=2)
    padded = trajectory.unit_actions == IGNORE

    baseline = _metrics(policy, trajectory)
    ignored = _metrics(policy, _narrowed(trajectory, padded))
    noticed = _metrics(policy, _narrowed(trajectory, ~padded))

    assert torch.isfinite(torch.tensor(list(baseline.values()))).all()
    for key in ("loss/policy", "entropy", "kl"):
        assert ignored[key] == pytest.approx(baseline[key], abs=1e-5)
        assert noticed[key] != pytest.approx(baseline[key], abs=1e-3)


def test_a_padded_slot_survives_a_mask_that_forbids_its_clamped_option() -> None:
    """``IGNORE`` clamps onto option 0, and option 0 is not legal by decree.

    ``mask.unit_mask`` keeps PASS alive on padded slots today, so the gathered
    log-probability there is finite today, and nothing enforces that beyond one
    function's current behaviour. The moment a padded row forbids option 0 the
    gather returns ``-inf``, and excluding it by multiplying through a float
    mask -- the obvious way to write this -- makes ``0 * -inf`` NaN in the
    forward pass and NaN in *every* gradient in the batch on the way back. One
    update destroys the weights, and it looks like a loss that went to NaN on a
    day nobody touched the loss.
    """
    logits = torch.randn(1, 2, len(UNIT_OPS), requires_grad=True)
    mask = torch.ones(1, 2, len(UNIT_OPS), dtype=torch.bool)
    mask[0, 1, 0] = False
    units = torch.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)
    market = torch.log_softmax(torch.zeros(1, 1, len(QUANTITIES)), dim=-1)

    joint = joint_log_prob(
        units, market, torch.tensor([[0, IGNORE]]), torch.tensor([[0]])
    )
    gradient = torch.autograd.grad(joint.sum(), logits)[0]

    assert torch.isfinite(joint).all()
    assert torch.isfinite(gradient).all()


def test_gae_never_runs_across_an_episode_boundary() -> None:
    """One season's opening advantage must not bootstrap off another's close.

    With ``gamma`` and ``lam`` at 1 the recursion telescopes, so the value
    target on an episode's first turn is exactly that episode's total reward.
    An update that concatenated the batch and then took GAE once over the whole
    block would put both episodes' rewards there instead, and the number would
    still be finite, still be the right shape, and still train.
    """
    policy = _policy(seed=0)
    first = _trajectory(policy, turns=4, seed=3)
    second = _trajectory(policy, turns=4, seed=4)
    # Progress off, so the target is the reward-to-go of the series this
    # test names. With it on the target is the reward-to-go of the money
    # reward *plus* the shaped one, which is correct and is tested
    # elsewhere, and would make this assertion about two things at once.
    config = PpoConfig(gamma=1.0, lam=1.0, progress=0.0)

    rows = flatten([first, second], config)

    assert float(second.rewards.sum()) != pytest.approx(0.0, abs=1e-3)
    assert rows.returns[0] == pytest.approx(
        float(first.rewards.sum()) * config.reward_scale, abs=1e-4
    )
    assert rows.returns[4] == pytest.approx(
        float(second.rewards.sum()) * config.reward_scale, abs=1e-4
    )


def test_the_value_target_is_the_scaled_reward_to_go() -> None:
    """What the value head is asked to predict, and in what units.

    A bank differential is thousands of coins, and regressed raw the value term
    is seven orders of magnitude above the policy loss -- measured on a real
    episode, ``0.5 * 13,089`` against ``-0.0011`` -- so ``max_grad_norm``
    renormalises a gradient that is essentially all value head and the policy
    stops moving. ``reward_scale`` puts the target in units of the opening bank
    instead.

    At ``gamma`` and ``lam`` of 1 the stored values cancel out of the target
    exactly, leaving the scaled reward-to-go and nothing else -- which is what
    makes this an assertion about the scaling rather than about the network.
    It fails if the scale is not applied, if it is applied twice, and if it is
    applied to ``Trajectory.values`` as well: those are the value head's own
    outputs, already in whatever units it is being asked to predict, and
    scaling them would leave a residue of ``V`` behind in the target.
    """
    trajectory = _trajectory(_policy(seed=0), turns=6, seed=5)
    # Progress off, so the target is the reward-to-go of the series this
    # test names. With it on the target is the reward-to-go of the money
    # reward *plus* the shaped one, which is correct and is tested
    # elsewhere, and would make this assertion about two things at once.
    config = PpoConfig(gamma=1.0, lam=1.0, progress=0.0)

    rows = flatten([trajectory], config)

    to_go = torch.flip(torch.cumsum(torch.flip(trajectory.rewards, [0]), 0), [0])

    assert float(trajectory.values.abs().min()) > 1e-3
    assert torch.allclose(rows.returns, to_go * config.reward_scale, atol=1e-4)


def test_a_bootstrap_is_required_rather_than_assumed() -> None:
    """A same-length pair shifts every advantage by a turn and never raises."""
    with pytest.raises(ValueError, match="one longer"):
        advantages(torch.ones(5), torch.zeros(5), gamma=1.0, lam=1.0)


def _policy(seed: int) -> Policy:
    """Return a small seeded policy.

    One block at 32 channels rather than the shipped 8x256. ``ppo`` reads every
    shape it needs from ``encoding``'s constants and nothing from ``model``'s
    ``BLOCKS`` or ``CHANNELS``, so this exercises the identical code path;
    ``tests/learn/test_model.py`` is where the shipped trunk is pinned.
    """
    torch.manual_seed(seed)
    return Policy(blocks=1, channels=32).eval()


def _trajectory(policy: Policy, turns: int, seed: int) -> Trajectory:
    """Return a synthetic episode, stored the way ``rollout`` stores a real one.

    The masks are random with option 0 forced legal -- ``UNIT_OPS[0]`` is
    ``PASS`` and ``QUANTITIES[0]`` is "trade nothing", the two the engine never
    refuses -- so no row is ever entirely False, which is the condition
    ``-inf`` masking needs to avoid softmaxing a row to NaN.

    The action is drawn from what the mask permits and the stored log-prob is
    read off the same masked distribution, so ``illegal`` would be zero and the
    ratio recomputed against these tensors is 1 for an unchanged policy.

    Args:
        policy: The behaviour policy, whose log-probs and values are stored.
        turns: How many decisions the episode holds.
        seed: Fixes the boards, masks and actions.

    Returns:
        The ``Trajectory``.
    """
    torch.manual_seed(seed)
    board = torch.randn(turns, TILE_PLANES, BOARD, BOARD)
    scalars = torch.randn(turns, SCALARS)
    positions = torch.randint(0, BOARD * BOARD, (turns, MAX_UNITS))
    unit_masks = _mask(turns, MAX_UNITS, len(UNIT_OPS))
    market_masks = _mask(turns, len(MARKET_SLOTS) + 2, len(QUANTITIES))

    with torch.no_grad():
        unit_logits, market_logits, values = policy(board, scalars, positions)
    units = torch.log_softmax(unit_logits.masked_fill(~unit_masks, -torch.inf), dim=-1)
    market = torch.log_softmax(
        market_logits.masked_fill(~market_masks, -torch.inf), dim=-1
    )
    unit_actions = torch.cat(
        [
            _draw(unit_masks[:, :CREW]),
            torch.full((turns, MAX_UNITS - CREW), IGNORE, dtype=torch.int64),
        ],
        dim=1,
    )
    market_actions = _draw(market_masks)

    dones = torch.zeros(turns, dtype=torch.bool)
    dones[-1] = True
    rewards = torch.randn(turns) * 100.0
    # A second, independent series. Not a scaling of `rewards`: `reward_of`
    # blends the two, and a synthetic trajectory whose halves were proportional
    # could not tell a blend that ignores one of them from one that does not.
    own = torch.randn(turns) * 100.0
    # A third, independent series: the potential at each state the policy acted
    # from, with no terminal row -- `progress_reward` supplies the terminal zero
    # itself, and there is nowhere here to put a different one. Positive because
    # a potential is a coin value of stock on hand and cannot be negative, so a
    # sign error in `progress_reward` has somewhere to show.
    potentials = torch.rand(turns, len(POTENTIAL_COMPONENTS)) * 100.0
    return Trajectory(
        board=board,
        scalars=scalars,
        positions=positions,
        unit_actions=unit_actions,
        market_actions=market_actions,
        unit_masks=unit_masks,
        market_masks=market_masks,
        log_probs=joint_log_prob(units, market, unit_actions, market_actions),
        values=values,
        rewards=rewards,
        own=own,
        # A third independent series, for the same reason `own` is independent
        # of `rewards`: Toad's shaped reward is on its own scale entirely, and a
        # synthetic trajectory whose series were proportional could not tell a
        # learner reading the wrong one from a learner reading the right one.
        shaped=torch.randn(turns) * 0.01,
        shaped_money=torch.randn(turns) * 0.01,
        margin=torch.randn(turns) * 0.01,
        potentials=potentials,
        dones=dones,
        final_margin=float(rewards.sum()),
        final_bank=STARTING_MONEY + float(rewards.sum()),
        final_capital=0.0,
        illegal=0,
        sales=0.0,
        units_sold=0.0,
        mean_sale_price=0.0,
        realisation=0.0,
        bought=0.0,
    )


def _mask(turns: int, slots: int, options: int) -> torch.Tensor:
    """Return a random legality mask whose every row keeps option 0."""
    mask = torch.rand(turns, slots, options) < 0.5
    mask[:, :, 0] = True
    return mask


def _draw(mask: torch.Tensor) -> torch.Tensor:
    """Return one index per slot, drawn uniformly from the options it permits."""
    rows = mask.flatten(0, -2).float()
    return torch.multinomial(rows, 1).reshape(mask.shape[:-1])


def _narrowed(trajectory: Trajectory, slots: torch.Tensor) -> Trajectory:
    """Return the episode with the named unit slots narrowed to PASS and the action.

    The chosen option is kept legal so that narrowing a slot that holds a real
    unit changes the shape of its distribution without ever making the action
    it took impossible -- an ``-inf`` log-probability there would drive the
    ratio to zero and the comparison would be measuring the wrong thing.

    Args:
        trajectory: The episode to narrow.
        slots: ``(turns, MAX_UNITS)`` bool, True where the mask is to be cut.

    Returns:
        A copy with ``unit_masks`` replaced.
    """
    narrow = torch.zeros_like(trajectory.unit_masks)
    narrow[:, :, 0] = True
    narrow.scatter_(2, trajectory.unit_actions.clamp(min=0)[:, :, None], True)
    return dataclasses.replace(
        trajectory,
        unit_masks=torch.where(slots[:, :, None], narrow, trajectory.unit_masks),
    )


def _metrics(policy: Policy, trajectory: Trajectory) -> dict[str, float]:
    """Return one update's metrics, leaving the caller's policy untouched.

    ``update`` steps the optimiser, so the policy is copied first: every test
    above compares two updates that must both start from identical weights.

    The teacher is a *differently* seeded network. A teacher that is a copy of
    the learner has a divergence of exactly zero from it, and every assertion
    about the KL term would then be satisfied by an update that never computed
    one -- which is how the first draft of this file passed while measuring
    nothing.
    """
    learner = copy.deepcopy(policy)
    return update(
        learner,
        _policy(seed=9),
        torch.optim.AdamW(learner.parameters(), lr=3e-4),
        [trajectory],
        step=0,
        config=ONE_STEP,
    )
