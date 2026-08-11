"""Explained variance of the critic, and where a seat's return variance comes from.

Ten arms have failed the same way: the policy-gradient term wanders around zero
while the value loss falls. The standing hypothesis is that a seat's return is
dominated by exogenous shared-market variance, that the critic therefore explains
nearly all of it, and that almost no advantage is left for the policy gradient.
This measures that rather than arguing it.

**Four numbers, and they are not interchangeable.**

* ``ev_pooled`` -- ``1 - Var(G - V) / Var(G)`` over every recorded state of every
  episode, against the *full-episode* discounted return-to-go at Toad's
  ``DISCOUNTING``. This is the textbook diagnostic and it is the headline.
* ``ev_within_step`` -- the same quantity after subtracting, at each turn index,
  the cross-episode mean of both ``G`` and ``V``. It exists because ``G_t`` has a
  large *deterministic* profile in ``t``: as the horizon approaches, the
  return-to-go collapses onto the terminal component. A critic that has learned
  only the clock scores near 1.0 on ``ev_pooled`` while discriminating no two
  episodes from each other. Only ``ev_within_step`` says whether the critic
  explains the *draw*, which is what the hypothesis is about.
* ``ev_start`` -- ``1 - Var(G_0 - V_0) / Var(G_0)`` across episodes, i.e. does the
  critic call the season from the opening position.
* ``ev_td_segment`` -- against the learner's own TD(lambda) targets, in the exact
  16-step segments ``toad_phase1._step`` builds, bootstrapped from the value of
  the segment's own last state. This is what the ``baseline`` loss term measures
  and it is reported for continuity with the training logs, but it is partly
  circular by construction: the target is built out of the values being scored.
  It is not evidence about the world; ``ev_pooled`` and ``ev_within_step`` are.

**The returns come from the vendored code, not from a second implementation.**
``td_lambda.td_lambda`` at ``lmb=1.0`` *is* the discounted Monte Carlo
return-to-go -- the ``(1 - lmb)`` factor deletes the only place values enter --
so the same function the learner regresses against computes the ground truth
here, and there is no chance of the diagnostic's return disagreeing with the
learner's by an off-by-one or a discount.

**The behaviour policy is the learner.** A checkpoint scored here plays and is
evaluated as the same weights, so the V-trace importance ratios are exactly 1 and
``pg_advantages`` reduce to ``r_t + gamma * vs_{t+1} - V_t``. In training the
actor lags the learner by up to ``SYNC_EVERY`` updates. That is a real difference
and it makes the advantages measured here an *upper* bound on the training ones:
clipping at ``rho = 1`` can only shrink them.

**The variance decomposition is the law of total variance, and the engine makes
it exact.** ``Var(G) = E_seed[Var_action(G | seed)] + Var_seed(E_action[G | seed])``.
The second term is the part no policy could have moved on this population: the
seed fixes the weed spawns and the town's shop unlock order, and
``economic_policy`` is a deterministic function of state, so conditioning on the
seed and holding our own action distribution fixed leaves only what our sampling
did. ``rollout_many`` is handed the *same* seed repeated, which gives one market
draw and ``REPEATS`` independent action streams out of a single lockstep group --
the group shares one ``torch.Generator`` and ``multinomial`` draws each row
independently, so the environments are identical and the play is not.

The confound is stated here because it cannot be measured away: the controllable
share is the variance *this policy's entropy actually explores*, not the variance
an optimal policy could induce. A near-deterministic policy reads as having no
control even where control exists. The number is a lower bound on controllability
and an upper bound on nothing.
"""

import dataclasses
import json
import logging
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from kaggriculture.learn.critic import explained_variance, monte_carlo
from kaggriculture.learn.model import Policy
from kaggriculture.learn.rollout import rollout_many
from kaggriculture.learn.scripts.toad_phase1 import BATCH_SEGMENTS, OPPONENT
from kaggriculture.learn.toad.core import td_lambda, vtrace
from kaggriculture.learn.toad_loss import DISCOUNTING, LMB, UNROLL_LENGTH

LOGGER = logging.getLogger(__name__)

RUNS = Path("/data/kaggriculture/toad")
SELFPLAY = Path("/data/kaggriculture/selfplay")
OUT = Path("docs/research/2026-08-09-critic-explained-variance.json")

# Disjoint from every seed any arm trained on. The arms walk
# ``range(update * 24, (update + 1) * 24)`` and the longest ran to update 707, so
# training consumed seeds below ~17,000. Scoring a critic on seeds it was fitted
# on would confuse "explains the game" with "remembers the draw".
SEEDS = tuple(range(900_000, 900_016))
# Seeds for the variance decomposition, and how many independent action streams
# each one is played under. 6 x 8 = 48 episodes.
SPLIT_SEEDS = tuple(range(910_000, 910_006))
REPEATS = 8
# The live phase1p run holds 12 workers and the box has 64 cores at a load
# average around 38. Eight is what is left over without starving it.
WORKERS = 8
THREADS = 1
# ``1 / STARTING_MONEY``, the scale the pre-Toad PPO lineage's value head was
# trained on (``ppo.REWARD_SCALE``). Explained variance is invariant to a common
# scaling of returns and values, but the *residual* is not, so the control's
# return is put on its own head's scale rather than left in coins.
PPO_REWARD_SCALE = 1.0 / 3000.0


@dataclasses.dataclass(frozen=True)
class Arm:
    """One checkpoint to score, and everything needed to score it on its own terms.

    Attributes:
        name: How the record is keyed.
        checkpoint: The ``.pt`` to load.
        field: Which recorded reward series this checkpoint's critic was trained
            against. Scoring a critic on a reward it never saw measures nothing.
        update: The optimizer round it was written at, so a record can be placed
            on the training axis. Zero for checkpoints outside a Toad arm.
        value_bound: ``1.0`` for every Toad arm, whose value head is squashed onto
            ``[-1, +1]`` the way ``BaselineLayer`` does. ``None`` for the pre-Toad
            PPO lineage, which was trained with an unbounded head -- constructing
            it bounded would push its output through a sigmoid it never saw and
            score a value function that never existed.
        scale: Multiplier on the reward series, to put the return on the scale the
            value head was regressed against.
        mirror: Play the checkpoint against itself instead of against
            ``economic_policy``.
    """

    name: str
    checkpoint: Path
    field: str
    update: int = 0
    value_bound: float | None = 1.0
    scale: float = 1.0
    mirror: bool = False


# The margin arm across its whole 507 updates, the live arm, the c3 lineage the
# other two descend from, and the control. The control is the pre-Toad shaped-PPO
# checkpoint that reached a real-opponent bank of 21,662 on the old engine -- the
# only policy this project has trained that is not stuck -- and it is here to give
# the headline a way to be wrong: if its critic also scores ~1.0, high explained
# variance is not what separates a working policy from a failing one.
ARMS = (
    *(
        Arm(
            name=f"margin_{update:06d}",
            checkpoint=RUNS / f"toad-phase1m-margin_{update:06d}.pt",
            field="margin",
            update=update,
        )
        for update in (225, 300, 400, 500, 600, 700)
    ),
    *(
        Arm(
            name=f"phase1p_{update:06d}",
            checkpoint=RUNS / f"phase1p_{update:06d}.pt",
            field="shaped",
            update=update,
        )
        for update in (225, 300)
    ),
    *(
        Arm(
            name=f"c3_{update:06d}",
            checkpoint=RUNS / f"toad-phase1c3-clone-teacher_{update:06d}.pt",
            field="shaped",
            update=update,
        )
        for update in (25, 100, 200, 300)
    ),
    # Two compositions, because the PPO lineage's reward was a scheduled blend of
    # the two money series and neither can be assumed. `rewards` is the bank
    # differential and `own` is our own bank; the run's progress-shaping term is
    # not reconstructible on this branch and is omitted -- a declared
    # approximation, and a small one, since a potential-based term telescopes to
    # its terminal value alone.
    Arm(
        name="control_ppo_differential",
        checkpoint=SELFPLAY / "resumed-3cc6068-1786139297" / "policy.pt",
        field="rewards",
        value_bound=None,
        scale=PPO_REWARD_SCALE,
    ),
    Arm(
        name="control_ppo_own",
        checkpoint=SELFPLAY / "resumed-3cc6068-1786139297" / "policy.pt",
        field="own",
        value_bound=None,
        scale=PPO_REWARD_SCALE,
    ),
    # Mirror readings, so the headline cannot be an artefact of the scripted
    # opponent's population alone. Half of every training round was mirror.
    Arm(
        name="margin_000700_mirror",
        checkpoint=RUNS / "toad-phase1m-margin_000700.pt",
        field="margin",
        update=700,
        mirror=True,
    ),
    Arm(
        name="c3_000300_mirror",
        checkpoint=RUNS / "toad-phase1c3-clone-teacher_000300.pt",
        field="shaped",
        update=300,
        mirror=True,
    ),
)

# Which arms get the controllable-versus-exogenous split. The margin arm at the
# end of its budget is the one the hypothesis is about; the control is there for
# the same reason it is in ARMS -- a policy that plays should move more of its own
# return than one that does not.
SPLIT_ARMS = ("margin_000700", "control_ppo_own")


def main() -> None:
    """Score every arm and write one record per arm, plus the variance split."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    started = time.monotonic()
    records: dict[str, object] = {
        "measured": time.strftime("%Y-%m-%d"),
        "discounting": DISCOUNTING,
        "lmb": LMB,
        "seeds": list(SEEDS),
        "split_seeds": list(SPLIT_SEEDS),
        "repeats": REPEATS,
    }
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        arms = {}
        for arm in ARMS:
            episodes = play(pool, arm, SEEDS)
            arms[arm.name] = {
                "checkpoint": str(arm.checkpoint),
                "field": arm.field,
                "update": arm.update,
                "opponent": "mirror" if arm.mirror else OPPONENT,
                "episodes": len(episodes),
                **diagnostics(episodes),
            }
            LOGGER.info("%s: %s", arm.name, json.dumps(arms[arm.name]))
        records["arms"] = arms

        splits = {}
        for name in SPLIT_ARMS:
            arm = next(candidate for candidate in ARMS if candidate.name == name)
            splits[name] = variance_split(pool, arm)
            LOGGER.info("split %s: %s", name, json.dumps(splits[name]))
        records["variance_split"] = splits

    records["hours"] = round((time.monotonic() - started) / 3600.0, 4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(records, indent=2) + "\n")
    LOGGER.info("wrote %s", OUT)


def play(pool: ProcessPoolExecutor, arm: Arm, seeds: Sequence[int]) -> list["Episode"]:
    """Play ``seeds`` with one checkpoint and return the per-episode series.

    The whole ``Trajectory`` is reduced to its value and reward series inside the
    worker. An episode's tensors are ~15 MB and only two 719-long vectors are
    read here, so returning trajectories would move a quarter of a gigabyte
    between processes to throw almost all of it away.

    Args:
        pool: The worker pool.
        arm: Which checkpoint, reward series and opponent.
        seeds: One seed per episode.

    Returns:
        One ``Episode`` per recorded seat, mirror episodes contributing two.
    """
    chunks = [list(seeds[index::WORKERS]) for index in range(WORKERS)]
    work = [(arm, chunk) for chunk in chunks if chunk]
    return [episode for batch in pool.map(_worker, work) for episode in batch]


@dataclasses.dataclass(frozen=True)
class Episode:
    """One recorded seat's value and reward series, and how the season ended.

    Attributes:
        seed: The episode seed.
        values: ``(turns,)`` the critic's estimate at every acted state.
        rewards: ``(turns,)`` the reward series this arm's critic was trained on,
            already on the value head's scale.
        dones: ``(turns,)`` bool, True on the last turn alone.
        log_probs: ``(turns,)`` joint log-probability of the turn's whole action.
        final_bank: This seat's terminal bank.
        final_margin: Its terminal bank less the other seat's.
    """

    seed: int
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    log_probs: torch.Tensor
    final_bank: float
    final_margin: float


def _worker(work: tuple[Arm, list[int]]) -> list[Episode]:
    """Play one worker's share. Runs in a subprocess.

    Torch is pinned to one thread for the reason ``toad_phase1._play`` pins it:
    every worker otherwise claims the whole machine, and there is a training run
    on this box that must not be starved.
    """
    torch.set_num_threads(THREADS)
    arm, seeds = work
    policy = load(arm)
    with torch.no_grad():
        trajectories = rollout_many(policy, policy if arm.mirror else OPPONENT, seeds)
    recorded = [seed for seed in seeds for _ in range(2 if arm.mirror else 1)]
    return [
        Episode(
            seed=seed,
            values=trajectory.values.clone(),
            rewards=getattr(trajectory, arm.field).clone() * arm.scale,
            dones=trajectory.dones.clone(),
            log_probs=trajectory.log_probs.clone(),
            final_bank=trajectory.final_bank,
            final_margin=trajectory.final_margin,
        )
        for seed, trajectory in zip(recorded, trajectories, strict=True)
    ]


def load(arm: Arm) -> Policy:
    """Return the checkpoint's policy in eval mode, at the width it was saved with.

    Both checkpoint layouts are accepted -- the Toad arms wrap the weights beside
    the optimizer and counters, the older self-play runs saved a bare state dict
    -- and the shape is read off the weights rather than assumed.

    ``value_bound`` comes from the arm rather than from a constant, because it is
    not cosmetic here: the Toad arms squash the value head onto ``[-1, +1]`` and
    the pre-Toad lineage does not, so constructing either one the other way scores
    a value function that was never trained.

    Args:
        arm: Which checkpoint and which value-head shape.

    Returns:
        The policy, ready to play.
    """
    state = torch.load(arm.checkpoint, map_location="cpu", weights_only=False)
    weights = state["learner"] if "learner" in state else state
    channels = int(weights["stem.weight"].shape[0])
    blocks = 1 + max(int(k.split(".")[1]) for k in weights if k.startswith("blocks."))
    policy = Policy(blocks=blocks, channels=channels, value_bound=arm.value_bound)
    policy.load_state_dict(weights)
    return policy.eval()


def diagnostics(episodes: Sequence[Episode]) -> dict[str, float]:
    """Return every explained-variance and advantage statistic for one arm.

    Args:
        episodes: The arm's played episodes, all the same length.

    Returns:
        The record's numeric body.
    """
    values = torch.stack([episode.values for episode in episodes])
    returns = torch.stack(
        [monte_carlo(episode.rewards, episode.dones) for episode in episodes]
    )
    banks = torch.tensor([episode.final_bank for episode in episodes])
    margins = torch.tensor([episode.final_margin for episode in episodes])
    # Two reference critics, scored by the same estimator as the learned one so
    # the comparison is like for like. `flat` knows the mean return and nothing
    # else -- its explained variance is 0 by construction and its advantage is
    # what the policy gradient sees with no critic worth the name. `clock` knows
    # the cross-episode mean return at every turn index and nothing about the
    # draw, which is the strongest critic that has learned only the calendar.
    flat = torch.full_like(values, float(returns.mean()))
    clock = returns.mean(dim=0).tile(returns.shape[0], 1)
    segments = segment_terms(episodes)
    return {
        "ev_pooled": explained_variance(returns.flatten(), values.flatten()),
        # Both series centred at each turn index, so the deterministic profile of
        # the return-to-go in `t` is removed from numerator and denominator alike.
        "ev_within_step": explained_variance(
            (returns - returns.mean(dim=0)).flatten(),
            (values - values.mean(dim=0)).flatten(),
        ),
        "ev_start": explained_variance(returns[:, 0], values[:, 0]),
        "ev_pooled_clock_critic": explained_variance(
            returns.flatten(), clock.flatten()
        ),
        "ev_td_segment": segments["ev_td_segment"],
        "return_mean": float(returns[:, 0].mean()),
        "return_std_across_episodes": float(returns[:, 0].std()),
        "value_start_mean": float(values[:, 0].mean()),
        "value_start_std_across_episodes": float(values[:, 0].std()),
        "residual_std_within_step": float((returns - values).std(dim=0).mean()),
        "reward_abs_mean": float(
            torch.stack([episode.rewards.abs().mean() for episode in episodes]).mean()
        ),
        "advantage_abs_mean": segments["advantage_abs_mean"],
        "advantage_rms": segments["advantage_rms"],
        "advantage_rms_flat_critic": segment_terms(episodes, flat)["advantage_rms"],
        "advantage_rms_clock_critic": segment_terms(episodes, clock)["advantage_rms"],
        "log_prob_mean": float(
            torch.stack([episode.log_probs.mean() for episode in episodes]).mean()
        ),
        "bank_mean": float(banks.mean()),
        "bank_std": float(banks.std()),
        "margin_mean": float(margins.mean()),
        # The corpus reads +0.98 between a seat's final bank and its opponent's.
        # Measured again here, on our own rollouts and this engine, because the
        # whole hypothesis rests on it.
        "bank_vs_opponent_correlation": correlation(banks, banks - margins),
    }


def segment_terms(
    episodes: Sequence[Episode], substitute: torch.Tensor | None = None
) -> dict[str, float]:
    """Return the learner's own segment-shaped value target and advantage statistics.

    The episodes are chopped and batched the way the arms in ``ARMS`` were
    trained -- ``UNROLL_LENGTH`` turns per segment tiled forward from turn 0,
    ``BATCH_SEGMENTS`` segments per batch, bootstrapping from the value of the
    batch's own last row -- because the value loss and the advantage both scale
    with that shape, and a segment of a different length is a different number.

    That is deliberately **no longer** what ``toad_phase1._segments`` and
    ``_step`` do. Both defects this reconstruction faithfully reproduces -- the
    ragged tail that dropped the season's only ``done``, and the bootstrap taken
    from inside the segment rather than from the state after it -- were fixed on
    2026-08-11, and fixing them took a fresh critic's explained variance against
    the real return from -14.4 to +0.94. This function is kept as it is on
    purpose: every checkpoint in ``ARMS`` was trained under the old target, and
    scoring them under the new one would report a value loss none of them was
    ever optimising. It is a historical instrument, not a mirror of the runner.

    The behaviour and target log-probabilities are the same tensor here: a
    checkpoint is scored as the policy that played, so the importance ratios are
    exactly 1 and V-trace degenerates the way it is supposed to.

    Args:
        episodes: The arm's played episodes.
        substitute: ``(episodes, turns)`` values to score in place of the
            checkpoint's own, for the reference critics. The estimator, the
            segmentation and the batching are then identical and the critic is
            the only thing that differs, which is the whole point of running it.

    Returns:
        The segment-level explained variance and advantage magnitudes.
    """
    stack = (
        [episode.values for episode in episodes]
        if substitute is None
        else list(substitute)
    )
    columns = [
        (
            value[start : start + UNROLL_LENGTH],
            episode.rewards[start : start + UNROLL_LENGTH],
            episode.dones[start : start + UNROLL_LENGTH],
            episode.log_probs[start : start + UNROLL_LENGTH],
        )
        for episode, value in zip(episodes, stack, strict=True)
        for start in range(
            0, int(episode.dones.shape[0]) - UNROLL_LENGTH + 1, UNROLL_LENGTH
        )
    ]
    targets, values, advantages = [], [], []
    for start in range(0, len(columns) - BATCH_SEGMENTS + 1, BATCH_SEGMENTS):
        batch = columns[start : start + BATCH_SEGMENTS]
        value = torch.stack([column[0] for column in batch], dim=1)
        reward = torch.stack([column[1] for column in batch], dim=1)
        done = torch.stack([column[2] for column in batch], dim=1)
        log_prob = torch.stack([column[3] for column in batch], dim=1)
        discounts = (~done).float() * DISCOUNTING
        bootstrap = value[-1]
        target = td_lambda.td_lambda(
            rewards=reward,
            values=value,
            bootstrap_value=bootstrap,
            discounts=discounts,
            lmb=LMB,
        ).vs
        pg = vtrace.from_action_log_probs(
            behavior_action_log_probs=log_prob,
            target_action_log_probs=log_prob,
            discounts=discounts,
            rewards=reward,
            values=value,
            bootstrap_value=bootstrap,
        ).pg_advantages
        targets.append(target)
        values.append(value)
        advantages.append(pg)
    target = torch.cat(targets)
    value = torch.cat(values)
    advantage = torch.cat(advantages)
    return {
        "ev_td_segment": explained_variance(target.flatten(), value.flatten()),
        "advantage_abs_mean": float(advantage.abs().mean()),
        "advantage_rms": float(advantage.pow(2).mean().sqrt()),
    }


def variance_split(pool: ProcessPoolExecutor, arm: Arm) -> dict[str, float]:
    """Split the return's variance into what our actions moved and what the draw did.

    ``Var(G) = E_seed[Var_action(G | seed)] + Var_seed(E_action[G | seed])``,
    estimated by playing each of ``SPLIT_SEEDS`` under ``REPEATS`` independent
    action streams. One lockstep group per seed holds that seed repeated, so every
    environment in the group is the same market and only the sampling differs.

    Args:
        pool: The worker pool.
        arm: Which checkpoint and reward series.

    Returns:
        The two variance components and the controllable share, for the episode
        return and for the final bank and margin in coins.
    """
    groups = list(pool.map(_worker, [(arm, [seed] * REPEATS) for seed in SPLIT_SEEDS]))
    split = {
        "return": [
            [float(monte_carlo(episode.rewards, episode.dones)[0]) for episode in group]
            for group in groups
        ],
        "bank": [[episode.final_bank for episode in group] for group in groups],
        "margin": [[episode.final_margin for episode in group] for group in groups],
    }
    record: dict[str, float] = {}
    for name, rows in split.items():
        table = torch.tensor(rows)
        within = float(table.var(dim=1, unbiased=True).mean())
        between = float(table.mean(dim=1).var(unbiased=True))
        record[f"{name}_within_seed_variance"] = within
        record[f"{name}_between_seed_variance"] = between
        record[f"{name}_controllable_share"] = (
            within / (within + between) if within + between > 0.0 else float("nan")
        )
        record[f"{name}_mean"] = float(table.mean())
    return record


def correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return Pearson's r between two 1-D tensors."""
    return float(torch.corrcoef(torch.stack([left, right]))[0, 1])


if __name__ == "__main__":
    main()
