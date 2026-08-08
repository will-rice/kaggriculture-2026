"""Toad Brigade's phase 1: shaped self-play from random initialisation.

Phase 1 of their five, and the only one that stands alone: it is the sole phase
with ``teacher_kl_cost: 0.``, so it is genuinely teacher-free self-play from
scratch and none of monobeast's teacher machinery is needed to run it. That makes
it the pure baseline the user asked for -- one faithful run, measured, before
anything is improved.

Every hyperparameter comes from ``toad/conf/conv_phase1_shaped_reward.yaml`` and
is named in ``toad_loss``. The loss is ``toad_loss.losses``, which is their
V-trace + UPGO + TD(lambda) composition over the vendored return functions. The
reward is ``toad_reward.shaped``, their five weighted deltas with the 10x
terminal result riding inside from turn one, through their /500 normaliser.

Deviations from monobeast, all deliberate and all recorded in the task report:

* **Synchronous actor/learner** instead of their multiprocessing queues (D7).
  The loss is unchanged, but V-trace only does work when the actor's weights lag
  the learner's, so the actor is an explicit frozen copy resynced every
  ``SYNC_EVERY`` updates rather than whatever lag the queues happened to give
  them. With no lag the importance ratios would be identically 1, V-trace would
  degenerate to TD silently, and the UPGO clipping would never bind.
* **The market head is ours** (D8). Their actor is per-tile and has no analogue
  for a non-spatial trading decision; ours pools the trunk. The unit head keeps
  their structure -- a per-tile readout gathered at each unit's own square.
* **719 decisions against their 360** (D2), at their gamma of 0.999.
* **Nothing rewards selling well** (D1), because Toad had no such component.
  This is the reproduction's central open question and is left standing so the
  run measures it.
"""

import argparse
import copy
import json
import logging
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
import wandb
from lightning import seed_everything

from kaggriculture.learn.model import Policy
from kaggriculture.learn.ppo import entropy_of, joint_log_prob
from kaggriculture.learn.rollout import Trajectory, rollout_many
from kaggriculture.learn.toad_loss import (
    ADAM_EPS,
    CLIP_GRADS,
    LEARNING_RATE,
    MIN_LR_MOD,
    TOTAL_STEPS,
    UNROLL_LENGTH,
    losses,
)

LOGGER = logging.getLogger(__name__)

SEED = 0
# conv_phase1_shaped_reward.yaml:36. Four unrolls per learner batch, so one
# optimizer step sees 16 x 4 = 64 transitions. Collecting a whole round and
# stepping once on all of it would be a different algorithm -- and, measured, it
# also OOMs: 2,112 segments is a 34k-row forward.
BATCH_SEGMENTS = 4
# conv_phase1_shaped_reward.yaml: n_blocks 8, hidden_dim 128. Our current PPO
# network is 8 blocks at 256 channels; the width is theirs, not ours.
BLOCKS = 8
CHANNELS = 128
# toad/nns/models.py:157-160 and :181. StatefulMultiReward's spec is
# +/-1/MAX_DAYS and its `only_once` is False, so BaselineLayer expands the range
# by MAX_DAYS: the value head is confined to [-1, +1]. Ours was unbounded, and
# that alone diverged the first run.
VALUE_BOUND = 1.0
# Which reward the learner trains on. "shaped" is Toad's five components exactly
# and is the baseline. "shaped_money" is phase-1b: the same five plus our one
# added component paying for coins banked, deviation D1 made explicit. Both are
# recorded on every trajectory, so switching this constant is the entire diff
# between the two arms and the ablation shares seeds, episodes and actions.
REWARD_FIELD = "shaped"
# Episodes per collection round. Their n_actor_envs is 16 across 2 actors; ours
# is one synchronous group, and both seats of a self-play episode are recorded.
ENVIRONMENTS = 24
# The engine is single-threaded Python and is the bottleneck -- measured, the
# rollout is env-bound rather than network-bound, so the GPU does not fix it.
# Collection is spread over processes the way `selfplay.collect` does it.
WORKERS = 12
# Torch threads per worker. One, so WORKERS processes do not each claim the
# whole machine; see _play.
THREADS = 1
# How many updates the actor's weights lag the learner's. See D7 above.
SYNC_EVERY = 4
RUNS = Path("/data/kaggriculture/toad")
# Every 25 updates is ~13 minutes of work at the measured 3.97M steps/hour.
# Attempt one had none, and an external kill at update 253 cost 2.3 hours.
CHECKPOINT_EVERY = 25
WANDB_ENTITY = "will-rice"
WANDB_PROJECT = "kaggriculture-2026"


def main() -> None:
    """Run phase 1 to ``TOTAL_STEPS``, checkpointing and logging as it goes."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="checkpoint to continue from, restoring weights, optimizer, "
        "schedule and counters",
    )
    arguments = parser.parse_args()

    seed_everything(SEED, workers=True)
    RUNS.mkdir(parents=True, exist_ok=True)

    device = _device()
    learner = Policy(blocks=BLOCKS, channels=CHANNELS, value_bound=VALUE_BOUND).to(
        device
    )
    optimizer = torch.optim.Adam(learner.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, _decay)

    steps, update = 0, 0
    if arguments.resume is not None:
        steps, update = _restore(arguments.resume, learner, optimizer, schedule, device)
        LOGGER.info("resumed from %s at update %d, step %d", arguments.resume, update, steps)

    log = RUNS / f"phase1_{int(time.time())}.jsonl"
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=log.stem,
        config={
            "blocks": BLOCKS,
            "channels": CHANNELS,
            "value_bound": VALUE_BOUND,
            "reward_field": REWARD_FIELD,
            "environments": ENVIRONMENTS,
            "batch_segments": BATCH_SEGMENTS,
            "unroll_length": UNROLL_LENGTH,
            "sync_every": SYNC_EVERY,
            "lr": LEARNING_RATE,
            "adam_eps": ADAM_EPS,
            "clip_grads": CLIP_GRADS,
            "total_steps": TOTAL_STEPS,
        },
    )
    parameters = sum(p.numel() for p in learner.parameters())
    LOGGER.info(
        "phase 1: %d params, device %s, log %s, wandb %s",
        parameters,
        device,
        log,
        run.url,
    )

    actor = copy.deepcopy(learner).eval()
    started = time.monotonic()
    pool = ProcessPoolExecutor(max_workers=WORKERS)
    while steps < TOTAL_STEPS:
        seeds = tuple(range(update * ENVIRONMENTS, (update + 1) * ENVIRONMENTS))
        weights = {key: value.cpu() for key, value in actor.state_dict().items()}
        batch = _collect(pool, weights, seeds)
        steps += sum(int(t.shaped.shape[0]) for t in batch)
        terms = _update(learner, optimizer, batch, device)
        schedule.step()
        update += 1
        if update % SYNC_EVERY == 0:
            actor.load_state_dict(learner.state_dict())
        if update % CHECKPOINT_EVERY == 0:
            _checkpoint(learner, optimizer, schedule, steps, update)

        banks = [t.final_bank for t in batch]
        record = {
            "update": update,
            "steps": steps,
            "hours": round((time.monotonic() - started) / 3600.0, 4),
            "bank_mean": sum(banks) / len(banks),
            "bank_max": max(banks),
            "shaped_mean": float(torch.stack([t.shaped.sum() for t in batch]).mean()),
            "illegal": sum(t.illegal for t in batch),
            "lr": schedule.get_last_lr()[0],
            **terms,
        }
        with log.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        wandb.log(record, step=steps)
        LOGGER.info(
            "update %d steps %d bank %.1f (max %.1f) shaped %.4f total_loss %.3f",
            update,
            steps,
            record["bank_mean"],
            record["bank_max"],
            record["shaped_mean"],
            record["total"],
        )
    wandb.finish()


def _decay(step: int) -> float:
    """Return the LR multiplier at ``step``, floored at their ``min_lr_mod``.

    A module-level function rather than a lambda so that resuming restores the
    same schedule object: a lambda cannot be pickled into a checkpoint, and a
    schedule silently rebuilt from step zero would restore the initial learning
    rate and quietly change the recipe partway through a run.
    """
    return max(1.0 - step / max(_updates(), 1), MIN_LR_MOD)


def _checkpoint(
    learner: Policy,
    optimizer: torch.optim.Optimizer,
    schedule: torch.optim.lr_scheduler.LRScheduler,
    steps: int,
    update: int,
) -> Path:
    """Write everything needed to continue the run, atomically.

    Toad's monobeast checkpoints continuously; ours did not, and an external
    kill at update 253 cost 2.3 hours of training because the weights lived only
    in the process. Written to a temporary name and renamed, so a kill during
    the write leaves the previous checkpoint intact rather than a truncated one.

    Args:
        learner: The network being trained.
        optimizer: Its optimizer, whose Adam moments matter as much as the weights.
        schedule: The LR schedule, so the recipe continues rather than restarts.
        steps: Environment steps consumed so far.
        update: Optimizer rounds so far.

    Returns:
        The path written.
    """
    path = RUNS / f"phase1_{update:06d}.pt"
    temporary = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "learner": learner.state_dict(),
            "optimizer": optimizer.state_dict(),
            "schedule": schedule.state_dict(),
            "steps": steps,
            "update": update,
        },
        temporary,
    )
    temporary.rename(path)
    LOGGER.info("checkpoint %s", path)
    return path


def _restore(
    path: Path,
    learner: Policy,
    optimizer: torch.optim.Optimizer,
    schedule: torch.optim.lr_scheduler.LRScheduler,
    device: str,
) -> tuple[int, int]:
    """Load a checkpoint in place and return its ``(steps, update)``.

    Restores the schedule's own state rather than fast-forwarding it, because
    recomputing the position by stepping it ``update`` times is exactly where an
    off-by-one would hide, and a schedule one step out changes the learning rate
    for the rest of the run.

    Args:
        path: The checkpoint.
        learner: Network to load into.
        optimizer: Optimizer to load into.
        schedule: Schedule to load into.
        device: Where to map the tensors.

    Returns:
        The ``(steps, update)`` the checkpoint was written at.
    """
    state = torch.load(path, map_location=device, weights_only=False)
    learner.load_state_dict(state["learner"])
    optimizer.load_state_dict(state["optimizer"])
    schedule.load_state_dict(state["schedule"])
    return int(state["steps"]), int(state["update"])


def _device() -> str:
    """Return the CUDA device with the most free memory, or CPU.

    The machine's GPUs are shared with other runs, so a hardcoded ``cuda:0``
    lands on whichever card someone else already filled -- measured, that is how
    the first attempt died.
    """
    if not torch.cuda.is_available():
        return "cpu"
    free = [torch.cuda.mem_get_info(index)[0] for index in range(torch.cuda.device_count())]
    return f"cuda:{free.index(max(free))}"


def _collect(
    pool: ProcessPoolExecutor,
    state: dict[str, torch.Tensor],
    seeds: Sequence[int],
) -> list[Trajectory]:
    """Play ``seeds`` across worker processes and return every trajectory.

    Both seats of each episode are recorded, because the actor plays itself and
    the two seats therefore carry identical weights -- the condition
    ``rollout_many`` requires before it will record seat 1.

    Args:
        pool: The process pool to spread episodes over.
        state: The actor's weights, on CPU so they pickle to the workers.
        seeds: One seed per episode.

    Returns:
        Every recorded trajectory, two per seed.
    """
    chunks = [list(seeds[index::WORKERS]) for index in range(WORKERS)]
    batches = pool.map(_play, [(state, chunk) for chunk in chunks if chunk])
    return [trajectory for batch in batches for trajectory in batch]


def _play(work: tuple[dict[str, torch.Tensor], list[int]]) -> list[Trajectory]:
    """Play one worker's share of a round. Runs in a subprocess.

    Torch is pinned to one thread here for the same reason ``selfplay`` pins it
    (:509): every worker otherwise defaults to a pool the width of the machine,
    and WORKERS of those oversubscribe it badly enough to be slower than a
    single process. Measured at 64 cores, leaving this out drove load average
    past 230 and the round did not finish.
    """
    torch.set_num_threads(THREADS)
    state, seeds = work
    actor = Policy(blocks=BLOCKS, channels=CHANNELS, value_bound=VALUE_BOUND)
    actor.load_state_dict(state)
    actor.eval()
    with torch.no_grad():
        return rollout_many(actor, actor, seeds)


def _updates() -> int:
    """Return roughly how many updates the run will take, for the LR schedule."""
    return TOTAL_STEPS // (ENVIRONMENTS * 719 * 2)


def _update(
    learner: Policy,
    optimizer: torch.optim.Optimizer,
    batch: list[Trajectory],
    device: str,
) -> dict[str, float]:
    """Take one optimizer step per ``BATCH_SEGMENTS`` unrolls and return the means.

    Their learner consumes batches of four 16-step unrolls and steps once per
    batch, continuously fed by the actors. Collecting a whole round and stepping
    once on all of it would be a different algorithm with a different effective
    learning rate, so the round is chopped into their batch shape instead.

    Args:
        learner: The network being trained, updated in place.
        optimizer: Its optimizer.
        batch: The trajectories collected this round.
        device: Where to run the learner.

    Returns:
        The four loss terms and their total, averaged over the round's steps.
    """
    segments = [s for trajectory in batch for s in _segments(trajectory)]
    totals: dict[str, float] = {}
    steps = 0
    for start in range(0, len(segments) - BATCH_SEGMENTS + 1, BATCH_SEGMENTS):
        terms = _step(
            learner, optimizer, segments[start : start + BATCH_SEGMENTS], device
        )
        for key, value in terms.items():
            totals[key] = totals.get(key, 0.0) + value
        steps += 1
    return {key: value / max(steps, 1) for key, value in totals.items()}


def _step(
    learner: Policy,
    optimizer: torch.optim.Optimizer,
    segments: list[dict[str, torch.Tensor]],
    device: str,
) -> dict[str, float]:
    """Take one gradient step on one batch of unrolls.

    Each unroll is re-scored under the learner's current weights, which is what
    makes the importance ratios meaningful: the batch was played by the actor,
    and the actor lags.

    Args:
        learner: The network being trained, updated in place.
        optimizer: Its optimizer.
        segments: ``BATCH_SEGMENTS`` unrolls of ``UNROLL_LENGTH`` turns.
        device: Where to run the learner.

    Returns:
        The four loss terms and their total, as floats.
    """

    def stacked(name: str) -> torch.Tensor:
        return torch.stack([s[name] for s in segments], dim=1).to(device)

    board = stacked("board")
    scalars = stacked("scalars")
    positions = stacked("positions")
    unit_actions = stacked("unit_actions")
    market_actions = stacked("market_actions")
    unit_masks = stacked("unit_masks")
    market_masks = stacked("market_masks")
    behaviour = stacked("log_probs")
    rewards = stacked(REWARD_FIELD)
    dones = stacked("dones")

    turns, width = behaviour.shape
    unit_logits, market_logits, values = learner(
        board.flatten(0, 1), scalars.flatten(0, 1), positions.flatten(0, 1)
    )
    flat_unit_masks = unit_masks.flatten(0, 1)
    flat_market_masks = market_masks.flatten(0, 1)
    units = torch.log_softmax(
        unit_logits.masked_fill(~flat_unit_masks, -torch.inf), dim=-1
    )
    market = torch.log_softmax(
        market_logits.masked_fill(~flat_market_masks, -torch.inf), dim=-1
    )
    learner_log_probs = joint_log_prob(
        units, market, unit_actions.flatten(0, 1), market_actions.flatten(0, 1)
    ).view(turns, width)
    # `entropy_of` returns positive entropy; Toad's `combine_policy_entropy`
    # returns sum p*log p, which is its negation. Feeding the wrong sign trains
    # the policy to collapse onto one action, which looks like fast progress.
    negative_entropy = -(
        entropy_of(units, flat_unit_masks).sum(dim=-1)
        + entropy_of(market, flat_market_masks).sum(dim=-1)
    ).view(turns, width)

    values = values.view(turns, width)
    terms = losses(
        behaviour_log_probs=behaviour,
        learner_log_probs=learner_log_probs,
        negative_entropy=negative_entropy,
        values=values,
        # The unroll is a slice out of the middle of an episode, so it
        # bootstraps from the value the learner assigns its own last state.
        bootstrap_value=values[-1].detach(),
        rewards=rewards,
        dones=dones,
    )
    optimizer.zero_grad(set_to_none=True)
    terms.total.backward()
    # monobeast.py:502-505, with their saved runs' clip_grads of 10.0. Without
    # it this diverges to a loss of 6e20 inside two updates.
    torch.nn.utils.clip_grad_norm_(learner.parameters(), CLIP_GRADS)
    optimizer.step()
    return {
        "vtrace_pg": terms.vtrace_pg.item(),
        "upgo_pg": terms.upgo_pg.item(),
        "baseline": terms.baseline.item(),
        "entropy": terms.entropy.item(),
        "total": terms.total.item(),
    }


def _segments(trajectory: Trajectory) -> list[dict[str, torch.Tensor]]:
    """Chop one episode into ``UNROLL_LENGTH`` segments, dropping the ragged tail.

    Their unroll_length is 16 and their reduction is ``sum``, so the loss scales
    with the segment shape; keeping both is part of keeping their coefficients
    meaningful.

    Args:
        trajectory: One recorded seat's episode.

    Returns:
        One dict of stacked tensors per whole segment.
    """
    turns = int(trajectory.dones.shape[0])
    fields = (
        "board",
        "scalars",
        "positions",
        "unit_actions",
        "market_actions",
        "unit_masks",
        "market_masks",
        "log_probs",
        REWARD_FIELD,
        "dones",
    )
    return [
        {
            name: getattr(trajectory, name)[start : start + UNROLL_LENGTH]
            for name in fields
        }
        for start in range(0, turns - UNROLL_LENGTH + 1, UNROLL_LENGTH)
    ]


if __name__ == "__main__":
    main()
