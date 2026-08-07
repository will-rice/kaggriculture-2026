"""Self-play PPO: a pool collects episodes, the learner updates, the gate is honest.

The shape of the loop is set by one measurement. Task 3 found 78% of an episode
inside the network at batch 1, and this task's re-measurement (below) found that
batching the forward across environments moves the bottleneck onto the pure
Python encoders and masks, where a process pool is the only lever left. So an
iteration is: twelve worker processes each drive four environments in lockstep
on one GPU, the learner takes one PPO update over everything they return, and
its weights go back out through a file.

    configuration                                   trajectories/hour
    one environment, batch 1, CPU (Task 3)                       205
    + batched inference, 32 environments on a GPU              1,275
    + both seats recorded, 16 environments mirrored            1,567
    + twelve worker processes                                 13,449
    - the PPO update, which the collection rate does not pay    3,983

The last row is the one that decides the run length, and it is the row a
collection benchmark does not show: the update costs 0.548 s per trajectory
against collection's 0.27 s, so 58% of this loop is gradient descent and no
amount of further collection throughput would matter. Six hours therefore buys
about 24,000 trajectories -- 17M decisions over roughly 330 updates -- which is
a twentieth of the spec's 300M-step reference and the honest size of a day's
work on one workstation.

**The teacher is whatever the run started from, and the penalty is on only when
that start was the clone.** ``update`` wants a teacher whether or not there is
anything to imitate, and a KL toward a *randomly initialised* network is noise
with a gradient. So a fresh run passes its own initial weights as the teacher
and ``kl_initial=0``, which leaves the logged KL meaning "how far has this
policy travelled from where it began" -- the diagnostic ``ppo.py`` says it is --
without the penalty ever acting.

**The pool holds this policy's own past, never the gate.** The vendored kaito
agent is held out: an agent that trained against it would have been shown the
answer, and the only rung that means anything is the one measured against
something the run never saw. Snapshots go in every ten iterations, the oldest
falls out past eight, and half the workers mirror the live policy instead --
which is also where the free seat-1 data comes from, since only a mirror episode
has both seats on-policy.

**A nonzero illegal count stops the run.** It means the mask that gated a logit
and the index that was stored beside it disagree, and every update after that
point is spent fitting a distribution nobody sampled. There is nothing to be
learned by continuing and a whole budget to be wasted, so the loop raises.
"""

import argparse
import copy
import dataclasses
import functools
import logging
import multiprocessing
import random
import statistics
import time
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from lightning import seed_everything

import wandb
from kaggriculture.learn.model import Policy
from kaggriculture.learn.play import model as cloned
from kaggriculture.learn.ppo import KL_INITIAL, PpoConfig, update
from kaggriculture.learn.rollout import Trajectory, rollout_many
from kaggriculture.report import wilson_interval
from kaggriculture.scripts.tracking import ENTITY, PROJECT, commit

LOGGER = logging.getLogger(__name__)

SEED = 0
LEARNING_RATE = 3e-4

# The workstation's second card. Chosen rather than "cuda" because the first
# card is a 3090 that other work runs on, and a pool of a dozen processes each
# opening a CUDA context is not a polite thing to do to a job already using it.
DEVICE = "cuda:0"

# One thread per worker, and this is not advisory. A 10x10 board gives torch
# nothing to parallelise, and the default thread count spends the episode in
# synchronisation: measured here at over 400 s for a single episode against
# 17.5 s at one thread on the CPU path, and still 3.11 s against 2.30 s per
# trajectory once the trunk had moved to the GPU and only the encoders' small
# tensor ops were left on the CPU. Twelve of these run at once.
THREADS = 1

# Twelve processes of four environments. Measured: 96 trajectories in 25.7 s,
# transfer included. Twenty-four processes of two collect 5% faster and open
# twice as many CUDA contexts (~600 MB each) on a card another job is already
# holding 19 GB of, which is not worth 5%.
WORKERS = 12
ENVIRONMENTS = 4

# How many workers mirror the live policy rather than playing an opponent from
# the pool. A mirror episode yields two on-policy trajectories and a pool
# episode one, so this trades data rate against opponent diversity -- and a
# mirror episode of a policy that banks nothing is worth *less* than half a pool
# episode, because both seats bankrupt identically and the differential is zero
# on every turn. OpenAI Five sampled 80% latest-self; this run starts at 0.25
# because the pool's fixed agents are the only thing in it that can bank.
MIRROR_SHARE = 0.25
POOL_SIZE = 8
SNAPSHOT_EVERY = 10

# The pool the workers sample from, kaito excluded. The learner's own past
# snapshots are not a pool on their own: a policy that banks nothing plays a
# copy of itself that banks nothing, the differential is zero on every turn of
# every episode, and PPO is handed an advantage of -0.0008 +- 0.0128 to climb.
# Measured over 30 iterations and about 2,000 seasons, with a best episode of 4
# coins out of a possible 200,000.
#
# So the pool opens with two agents that do bank -- the hand-written economic
# policy at 169,166 and the single richest harvested route at 186,163, both
# measured here -- and the learner's snapshots join them as they are written.
# Against a producing opponent the differential is large and signed, and every
# coin we bank moves it.
#
# Kaito is not in this list and must not be. It is the gate, and an agent that
# trained against it would have been shown the answer.
FIXED_OPPONENTS = (
    "src/kaggriculture/economic_policy.py",
    "baselines/best_route.py",
)

# Iterations spent on the own-bank reward before the differential is switched
# on. Every single-box winner with a primary source shaped first and switched
# after -- Toad Brigade at 20M steps, FLG at 65M, Frog Parade "as soon as
# training was running stably" -- and at this loop's ~12,000 decisions per
# trajectory-hour, 20M steps is more than a workstation day. This budget cannot
# reach their phase-1 lengths, so the split is stated rather than tuned: two
# thirds shaped, one third on the win condition, and the report says which
# phase the run ended in.
SHAPING_ITERATIONS = 200

# The win-rate curve. Eight seeds is far too few to gate on -- the gate is a
# separate evaluation over 128 -- but enough to see a curve move, and it costs
# 45 s against an iteration's 65.
EVALUATE_EVERY = 25
EVALUATION_SEEDS = 8
EVALUATION_OPPONENTS = ("src/kaggriculture/economic_policy.py", "starter")

# Far above any seed collection will reach -- a six-hour run strides through
# about 16,000 -- so the curve is never measured on a season the policy has
# just been fitted on.
EVALUATION_BASE = 1_000_000_000

RUNS = Path("/data/kaggriculture/selfplay")


@dataclass(frozen=True)
class Match:
    """One worker's assignment for one iteration.

    Attributes:
        weights: The learner's current weights, written once per iteration and
            read by every worker. A file rather than a pickled state dict
            because twelve workers reading one 41 MB file out of the page cache
            costs less than twelve copies of it down a pipe.
        opponent: A snapshot checkpoint to load, an agent spec to build, or
            ``None`` to mirror the live policy. ``None`` is the only case that
            yields both seats: only then are seat 1's log-probabilities the
            learner's own. The three cases are told apart by type rather than
            by a flag, so a caller cannot name a snapshot and get a mirror.
        seeds: One episode seed per environment, unique across the whole run so
            no iteration replays another's season.
    """

    weights: Path
    opponent: Path | str | None
    seeds: tuple[int, ...]


def main() -> None:
    """Run self-play PPO for a wall-clock budget and save the result."""
    arguments = parse()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    seed_everything(SEED, workers=True)
    revision = commit()

    directory = RUNS / f"{arguments.initialisation}-{revision}-{int(time.time())}"
    directory.mkdir(parents=True)
    policy, teacher = initialise(arguments.initialisation)
    config = PpoConfig(
        kl_initial=KL_INITIAL if arguments.initialisation == "clone" else 0.0
    )
    optimiser = torch.optim.AdamW(policy.parameters(), lr=LEARNING_RATE)
    chooser = random.Random(SEED)
    shaping = dataclasses.replace(config, differential=0.0)
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        job_type="self-play",
        name=f"rl-{arguments.initialisation}-{arguments.hours:g}h-{revision}",
        config={
            "initialisation": arguments.initialisation,
            "hours": arguments.hours,
            "seed": SEED,
            "learning_rate": LEARNING_RATE,
            "workers": WORKERS,
            "environments": ENVIRONMENTS,
            "mirror_share": MIRROR_SHARE,
            "pool_size": POOL_SIZE,
            "snapshot_every": SNAPSHOT_EVERY,
            "shaping_iterations": SHAPING_ITERATIONS,
            "fixed_opponents": list(FIXED_OPPONENTS),
            "commit": revision,
            "directory": str(directory),
            **{f"ppo/{field}": value for field, value in vars(config).items()},
        },
    )

    weights = directory / "learner.pt"
    pool: list[Path] = []
    opened = time.perf_counter()
    deadline = opened + arguments.hours * 3600
    iteration = 0

    # Spawned, not forked. The learner is already on the GPU by this point, and
    # a forked child inherits a CUDA context it cannot use.
    with ProcessPoolExecutor(
        max_workers=WORKERS, mp_context=multiprocessing.get_context("spawn")
    ) as workers:
        while time.perf_counter() < deadline:
            started = time.perf_counter()
            torch.save(policy.state_dict(), weights)
            batch = collect(workers, assignments(weights, pool, iteration, chooser))
            collected = time.perf_counter() - started

            refuse_illegal(batch, iteration)
            phase = shaping if iteration < SHAPING_ITERATIONS else config
            metrics = update(policy, teacher, optimiser, batch, iteration, phase)
            measured = {
                **metrics,
                **played(batch),
                "iteration": iteration,
                "pool": len(pool),
                "differential": phase.differential,
                "seconds/collect": collected,
                "seconds/iteration": time.perf_counter() - started,
                "hours": (time.perf_counter() - opened) / 3600,
            }
            if iteration % EVALUATE_EVERY == 0:
                measured.update(evaluate(policy, iteration))
            run.log(measured)
            LOGGER.info(
                "iteration %d | diff %.1f | %d trajectories | bank %.0f | "
                "margin %.0f | entropy %.3f | kl %.4f | value %.4f | %.1f s",
                iteration,
                phase.differential,
                measured["trajectories"],
                measured["bank/mean"],
                measured["margin/mean"],
                measured["entropy"],
                measured["kl"],
                measured["loss/value"],
                measured["seconds/iteration"],
            )

            if iteration % SNAPSHOT_EVERY == 0:
                snapshot_into(pool, policy, directory, iteration)
            iteration += 1

    final = directory / "policy.pt"
    torch.save(policy.state_dict(), final)
    LOGGER.info("%d iterations, saved %s", iteration, final)
    run.finish()


def parse() -> argparse.Namespace:
    """Return the two things a run is allowed to vary.

    Everything else is a constant in this module, because a run whose
    hyperparameters came from the shell is one whose numbers cannot be read back
    out of the repository at a commit. These two are arguments because the task
    is to compare the two initialisations at an equal budget, and a comparison
    whose arms differ by an edit is not a paired one.

    Returns:
        ``initialisation`` and ``hours``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "initialisation",
        choices=("fresh", "clone"),
        help="start from random weights, or from the behaviour-cloned checkpoint",
    )
    parser.add_argument("hours", type=float, help="wall-clock budget for collection")
    return parser.parse_args()


def initialise(initialisation: str) -> tuple[Policy, Policy]:
    """Return the policy to train and the teacher the KL is measured against.

    The teacher is always the starting point, so the logged KL always means the
    same thing: how far the policy has travelled from where it began. Whether
    that distance is *penalised* is a separate decision, and it lives in
    ``PpoConfig.kl_initial`` -- 1.0 from the clone, whose competence is worth
    not wrecking in the first updates, and 0.0 from noise, which is worth
    nothing to stay near.

    The clone is loaded through ``learn.play.model``, which is the one place
    that checks the checkpoint against the current trunk: it predates the value
    head, so it must load non-strict, and ``strict=False`` on its own would load
    just as quietly with a trunk key missing. Deep-copied twice because that
    function caches and returns one object, and training the policy would
    otherwise train the teacher.

    Args:
        initialisation: ``"fresh"`` or ``"clone"``.

    Returns:
        The policy in train mode and the teacher in eval mode, both on
        ``DEVICE``.
    """
    start = Policy() if initialisation == "fresh" else cloned()
    policy = copy.deepcopy(start).to(DEVICE).train()
    teacher = copy.deepcopy(start).to(DEVICE).eval()
    LOGGER.info(
        "%s start, %d parameters, on %s",
        initialisation,
        sum(parameter.numel() for parameter in policy.parameters()),
        DEVICE,
    )
    return policy, teacher


def assignments(
    weights: Path, pool: list[Path], iteration: int, chooser: random.Random
) -> list[Match]:
    """Return one match per worker, with seeds no other iteration will replay.

    Seeds are laid out as a single stride across the run rather than drawn at
    random, so two workers cannot collide inside an iteration and no iteration
    can revisit a season the value function has already been fitted on. The
    episode seed fixes the weed spawns and the town's unlock order, so a
    repeated seed is a repeated environment, not merely a repeated sample.

    Args:
        weights: The learner's current weights.
        pool: Snapshot checkpoints, oldest first. Empty until the first one is
            written; ``FIXED_OPPONENTS`` are sampled from either way, which is
            what stops the opening iterations being mirror-only.
        iteration: Which iteration this is, for the seed stride.
        chooser: The run's opponent-sampling stream.

    Returns:
        ``WORKERS`` matches.
    """
    base = iteration * WORKERS * ENVIRONMENTS
    return [
        Match(
            weights=weights,
            opponent=(
                None
                if chooser.random() < MIRROR_SHARE
                else chooser.choice([*FIXED_OPPONENTS, *pool])
            ),
            seeds=tuple(
                base + worker * ENVIRONMENTS + index for index in range(ENVIRONMENTS)
            ),
        )
        for worker in range(WORKERS)
    ]


def collect(workers: Executor, matches: Sequence[Match]) -> list[Trajectory]:
    """Play every match in parallel and return one flat batch of trajectories.

    The trajectories come back through ``torch.multiprocessing``'s tensor
    reducer, which passes shared memory rather than copying: 96 trajectories are
    1.5 GB and cost 0.5 s to return, against 25 s to play.

    Args:
        workers: The process pool.
        matches: One assignment per worker.

    Returns:
        Every trajectory the matches produced, mirrors contributing two per
        environment and snapshot matches one.
    """
    return [trajectory for group in workers.map(play, matches) for trajectory in group]


def play(match: Match) -> list[Trajectory]:
    """Play one worker's group of environments in lockstep. Runs in the worker."""
    policy = learner()
    policy.load_state_dict(torch.load(match.weights, map_location=DEVICE))
    opponent = (
        policy
        if match.opponent is None
        else snapshot(match.opponent)
        if isinstance(match.opponent, Path)
        else match.opponent
    )
    return rollout_many(policy, opponent, match.seeds)


@functools.lru_cache(maxsize=1)
def learner() -> Policy:
    """Return this worker's one policy object, built once and refilled per iteration.

    One object rather than one per iteration: the weights change every
    iteration but the shapes do not, so ``load_state_dict`` into a resident
    network avoids re-allocating 41 MB of GPU tensors 330 times per worker.

    This is also where the worker's thread count is set, because it is the
    first thing every task touches. See ``THREADS``.

    Returns:
        An uninitialised policy on ``DEVICE``, in eval mode.
    """
    torch.set_num_threads(THREADS)
    return Policy().to(DEVICE).eval()


@functools.lru_cache(maxsize=POOL_SIZE)
def snapshot(path: Path) -> Policy:
    """Return a pool checkpoint, loaded once per worker.

    Cached on the path, which is safe here in a way caching the learner's would
    not be: a snapshot file is written once and never rewritten, while
    ``learner.pt`` is a different network every iteration under the same name.

    Args:
        path: The snapshot to load.

    Returns:
        The frozen policy on ``DEVICE``, in eval mode.
    """
    policy = Policy()
    policy.load_state_dict(torch.load(path, map_location="cpu"))
    return policy.to(DEVICE).eval()


def refuse_illegal(batch: Sequence[Trajectory], iteration: int) -> None:
    """Stop the run if any sampled action was forbidden by its own stored mask.

    Not a warning and not a filter. A nonzero count means ``mask.py`` and the
    sampler disagree about what was legal, so the log-probability PPO divides by
    belongs to a distribution that was never sampled from -- and the update
    would still run, still produce finite losses, and still spend the whole
    budget. The count is over the *stored* pair, so it also catches a mask
    stored beside the wrong action.

    Args:
        batch: The iteration's trajectories.
        iteration: Which iteration, for the message.

    Raises:
        ValueError: If any trajectory reports an illegal action.
    """
    illegal = sum(trajectory.illegal for trajectory in batch)
    if illegal:
        raise ValueError(
            f"iteration {iteration} sampled {illegal} actions their own stored "
            f"mask forbids across {len(batch)} trajectories -- the mask and the "
            f"sampler disagree, and training through it would waste the run"
        )


def played(batch: Sequence[Trajectory]) -> dict[str, float]:
    """Return what the batch's episodes did, as distinct from what the loss did.

    Bank and margin are both reported because they answer different questions
    and self-play makes the margin nearly useless on its own: a mirror episode's
    two trajectories carry margins of exactly opposite sign, so their mean is
    zero however well or badly either side played. The bank is the number that
    says whether anything was produced at all, and it is where a collapse into
    mutual neutralisation would show.

    Args:
        batch: The iteration's trajectories.

    Returns:
        ``trajectories``, ``decisions``, ``bank/mean``, ``bank/max``,
        ``margin/mean`` and ``illegal``.
    """
    return {
        "trajectories": len(batch),
        "decisions": sum(len(trajectory.rewards) for trajectory in batch),
        "bank/mean": statistics.fmean(t.final_bank for t in batch),
        "bank/max": max(t.final_bank for t in batch),
        "margin/mean": statistics.fmean(t.final_margin for t in batch),
        "illegal": sum(trajectory.illegal for trajectory in batch),
    }


def evaluate(policy: Policy, iteration: int) -> dict[str, float]:
    """Return a win rate and a bank against each fixed opponent.

    Played in the parent on the same device, in one lockstep group per
    opponent, because eight episodes are two thirds of a worker's iteration and
    routing them through the pool would mean shipping the weights out again for
    them. Sampled rather than argmaxed, so this measures the policy that is
    being trained rather than a greedy relative of it.

    Eight seeds is a curve, not a gate. The interval is logged alongside so
    that nobody reads a jump between two iterations as a result: at eight games
    a 0.5 win rate spans roughly [0.22, 0.78].

    Args:
        policy: The learner. Left in whatever mode it was in -- ``Policy``
            carries no normalisation layers, so train and eval return the same
            numbers.
        iteration: Which iteration, to offset the evaluation seeds away from
            the seeds training is collecting on.

    Returns:
        ``win/<name>``, ``win_low/<name>``, ``win_high/<name>`` and
        ``bank/<name>`` per opponent.
    """
    metrics: dict[str, float] = {}
    for opponent in EVALUATION_OPPONENTS:
        name = Path(opponent).stem
        seeds = tuple(
            EVALUATION_BASE + iteration * EVALUATION_SEEDS + index
            for index in range(EVALUATION_SEEDS)
        )
        trajectories = rollout_many(policy, opponent, seeds)
        wins = sum(trajectory.final_margin > 0 for trajectory in trajectories)
        low, high = wilson_interval(wins, len(trajectories))
        metrics[f"win/{name}"] = wins / len(trajectories)
        metrics[f"win_low/{name}"] = low
        metrics[f"win_high/{name}"] = high
        metrics[f"bank/{name}"] = statistics.fmean(
            trajectory.final_bank for trajectory in trajectories
        )
    return metrics


def snapshot_into(
    pool: list[Path], policy: Policy, directory: Path, iteration: int
) -> None:
    """Write the policy into the opponent pool, dropping the oldest past the cap.

    A bounded pool of recent selves rather than every checkpoint ever written:
    the point is to stop the learner from cycling against one opponent, and a
    pool weighted toward the distant past would spend the budget beating a
    policy it has already beaten.

    Args:
        pool: The pool, oldest first. Mutated.
        policy: The learner to snapshot.
        directory: Where the run's files live.
        iteration: Which iteration, for the filename.
    """
    path = directory / f"snapshot-{iteration:05d}.pt"
    torch.save(
        {name: tensor.cpu() for name, tensor in policy.state_dict().items()}, path
    )
    pool.append(path)
    if len(pool) > POOL_SIZE:
        pool.pop(0).unlink()


if __name__ == "__main__":
    main()
