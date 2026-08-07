"""The four-rung gate: what a trained policy beats, and where it stops.

Four opponents in a fixed order of strength, and the run reports the rung it
reached rather than the best number it found. Each rung means something
different:

1. **The behaviour-cloned checkpoint.** Did reinforcement learning add anything
   to the imitation it started from? The clone banks nothing in play at 90.5%
   holdout accuracy, so this is the lowest bar in the project and losing it
   would say the run destroyed a competent-looking network.
2. **``economic_policy``.** The hand-written agent this project shipped before
   route memory. Beating it means the learned policy is worth more than the
   heuristics someone sat down and wrote.
3. **The fixed best route.** One harvested route replayed open-loop. Beating it
   means the policy is worth more than the best single season in the corpus.
4. **Vendored kaito.** The published kernel this repository currently submits,
   and the only rung whose meaning is "we exceeded what was available to copy".
   It is held out of the training pool for exactly that reason: an agent that
   trained against it would have been shown the answer.

**The policy is played the way it was trained** -- masked, and sampled rather
than argmaxed -- through ``rollout_many``. That is a deliberate choice over
routing it through ``learn.play``, which argmaxes *unmasked* logits and would
therefore be measuring a different agent than the one the run produced, on a
distribution it was never optimised for. It also means ``Trajectory.illegal``
is available for every episode, which is what makes the illegal-action rate a
number this script reports rather than one it assumes.

Win rate carries a Wilson interval because the decision this feeds is "did we
clear this rung", and at 128 games a rate of 0.55 and a rate of 0.62 are the
same measurement. Bank is reported beside it because the ladder scores wins but
a policy that wins on a margin of nothing has not learned to farm.
"""

import argparse
import functools
import json
import logging
import multiprocessing
import statistics
import subprocess
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from kaggriculture.learn import CHECKPOINT
from kaggriculture.learn.model import Policy
from kaggriculture.learn.rollout import Trajectory, rollout_many
from kaggriculture.report import wilson_interval

LOGGER = logging.getLogger(__name__)

DEVICE = "cuda:0"
THREADS = 1
WORKERS = 8
GAMES = 128

# Seeds no training run has collected on. Self-play strides from zero and the
# in-training curve from a billion; this is a third disjoint block, so the gate
# is measured on seasons the policy has never been fitted to.
SEED_BASE = 2_000_000_000

# In order of what beating them would mean. ``CHECKPOINT`` is named as a path
# and loaded as a network rather than played through ``learn.play``: both sides
# of that rung are then masked and sampled the same way, which is the only
# comparison that isolates what the reinforcement learning did from what
# masking at play time does.
# The pass compares a gate's opponent against the module our last submission
# served, so the rung names have to map back to what `main.py` imports.
OPPONENT_MODULES = {
    "economic_policy": "kaggriculture.economic_policy",
    "kaito": "kaggriculture.kaito_policy",
    "best_route": "kaggriculture.routes.play",
    "bc_clone": "kaggriculture.learn.play",
}

RUNGS: tuple[tuple[str, Path | str], ...] = (
    ("bc_clone", CHECKPOINT),
    ("economic_policy", "src/kaggriculture/economic_policy.py"),
    ("best_route", "baselines/best_route.py"),
    ("kaito", "src/kaggriculture/kaito_policy.py"),
)


@dataclass(frozen=True)
class Rung:
    """One opponent's result over ``GAMES`` seeded episodes.

    Attributes:
        opponent: Which rung.
        games: Episodes played.
        wins: Episodes whose final margin was positive. A tie is not a win --
            the engine's margin is a coin difference and exact ties happen
            when neither farm ever banks, which is a failure, not a draw.
        win_rate: ``wins / games``.
        low: Wilson lower bound at 95%.
        high: Wilson upper bound at 95%.
        bank: Our mean terminal bank.
        opponent_bank: Theirs.
        illegal: Sampled actions their own stored mask forbade, summed. Zero,
            or the masking is wrong and every other number here is noise.
        decisions: How many actions that count is out of.
    """

    opponent: str
    games: int
    wins: int
    win_rate: float
    low: float
    high: float
    bank: float
    opponent_bank: float
    illegal: int
    decisions: int


@dataclass(frozen=True)
class Assignment:
    """One worker's share of one rung."""

    weights: Path
    opponent: Path | str
    seeds: tuple[int, ...]


def main() -> None:
    """Play every rung and report each one, in order, without stopping early."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path, help="the policy checkpoint to gate")
    parser.add_argument(
        "--record",
        type=Path,
        help="write the top rung reached here, for the autonomous pass to read",
    )
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    measured = []
    with ProcessPoolExecutor(
        max_workers=WORKERS, mp_context=multiprocessing.get_context("spawn")
    ) as workers:
        for name, opponent in RUNGS:
            rung = measure(workers, arguments.weights, name, opponent)
            measured.append(rung)
            LOGGER.info("%s", describe(rung))

    if arguments.record:
        record(measured, arguments.record)


def record(measured: list[Rung], destination: Path) -> None:
    """Write the hardest rung beaten, stamped with the revision it measured.

    The autonomous pass reads this rather than re-gating: 128 seeded episodes
    take minutes, and a cron tick that re-measured every half hour would spend
    the box re-deriving a number that has not changed.

    The revision is the point of the stamp. A gate result is evidence about the
    code it ran on and nothing else, so the pass refuses a verdict whose
    revision does not match the tree it would build from -- which is what makes
    this a gate rather than a rubber stamp.

    The *last* rung beaten is recorded, not the best win rate: the rungs are
    ordered by what beating them would mean, and beating a weak one after
    losing a strong one is not progress.

    Args:
        measured: Every rung played, in the order of `RUNGS`.
        destination: Where to write the verdict.
    """
    revision = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    reached = [rung for rung in measured if rung.low > 0.5]
    top = reached[-1] if reached else measured[0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "revision": revision,
                "opponent": OPPONENT_MODULES.get(top.opponent, top.opponent),
                "games": top.games,
                "win_rate": top.win_rate,
                "low": top.low,
                "high": top.high,
                "bank": top.bank,
                "opponent_bank": top.opponent_bank,
                "illegal": top.illegal,
                "rungs_beaten": [rung.opponent for rung in reached],
            },
            indent=2,
        )
    )
    LOGGER.info("recorded %s: top rung %s", destination, top.opponent)


def measure(workers: Executor, weights: Path, name: str, opponent: Path | str) -> Rung:
    """Play ``GAMES`` episodes against one opponent and summarise them.

    The seeds are split across workers in contiguous blocks so that every rung
    plays the *same* seeds -- the weed spawns and the town's unlock order are
    fixed by the seed, so comparing two opponents on different seeds compares
    two different games as well as two different opponents.

    Args:
        workers: The process pool.
        weights: The policy under test.
        name: What to call this rung.
        opponent: A checkpoint to load as a network, or an agent spec.

    Returns:
        The rung's result.
    """
    share = GAMES // WORKERS
    assignments = [
        Assignment(
            weights=weights,
            opponent=opponent,
            seeds=tuple(SEED_BASE + worker * share + index for index in range(share)),
        )
        for worker in range(WORKERS)
    ]
    played = [
        trajectory for group in workers.map(play, assignments) for trajectory in group
    ]
    return summarise(name, played)


def play(assignment: Assignment) -> list[Trajectory]:
    """Play one worker's share. Runs in the worker."""
    policy = under_test(assignment.weights)
    opponent = (
        frozen(assignment.opponent)
        if isinstance(assignment.opponent, Path)
        else assignment.opponent
    )
    return rollout_many(policy, opponent, assignment.seeds)


@functools.lru_cache(maxsize=1)
def under_test(weights: Path) -> Policy:
    """Return the policy being gated, loaded once per worker.

    Loaded strictly. This is a checkpoint the self-play run wrote from the
    current ``Policy``, so every key must be there; a non-strict load here
    would silently gate a network with a randomly initialised head.

    Args:
        weights: The checkpoint.

    Returns:
        The policy on ``DEVICE``, in eval mode.
    """
    torch.set_num_threads(THREADS)
    policy = Policy()
    policy.load_state_dict(torch.load(weights, map_location="cpu"))
    return policy.to(DEVICE).eval()


@functools.lru_cache(maxsize=1)
def frozen(weights: Path) -> Policy:
    """Return a network opponent, loaded once per worker.

    Non-strict, unlike ``under_test``, and gated the same way ``learn.play``
    gates it: the behaviour-cloned checkpoint predates the value head, so it
    carries every trunk and head key and no ``value.*``. Any gap outside the
    value head raises rather than playing weights that loaded quietly wrong.

    Args:
        weights: The checkpoint.

    Returns:
        The policy on ``DEVICE``, in eval mode.

    Raises:
        ValueError: If anything but the value head is missing or unexpected.
    """
    torch.set_num_threads(THREADS)
    policy = Policy()
    result = policy.load_state_dict(
        torch.load(weights, map_location="cpu"), strict=False
    )
    allowed = {f"value.{name}" for name, _ in policy.value.named_parameters()}
    if not set(result.missing_keys) <= allowed or result.unexpected_keys:
        raise ValueError(
            f"{weights} does not match the current trunk: missing "
            f"{result.missing_keys}, unexpected {result.unexpected_keys}"
        )
    return policy.to(DEVICE).eval()


def summarise(name: str, played: Sequence[Trajectory]) -> Rung:
    """Return one rung's numbers from its episodes.

    Args:
        name: What to call this rung.
        played: Every episode against it.

    Returns:
        The ``Rung``.
    """
    wins = sum(trajectory.final_margin > 0 for trajectory in played)
    low, high = wilson_interval(wins, len(played))
    return Rung(
        opponent=name,
        games=len(played),
        wins=wins,
        win_rate=wins / len(played),
        low=low,
        high=high,
        bank=statistics.fmean(trajectory.final_bank for trajectory in played),
        opponent_bank=statistics.fmean(
            trajectory.final_bank - trajectory.final_margin for trajectory in played
        ),
        illegal=sum(trajectory.illegal for trajectory in played),
        decisions=sum(len(trajectory.rewards) for trajectory in played),
    )


def describe(rung: Rung) -> str:
    """Return one aligned line, laid out like ``report.format_standing``."""
    return (
        f"vs {rung.opponent:<16s} "
        f"{rung.win_rate:.3f} [{rung.low:.3f}, {rung.high:.3f}]  "
        f"({rung.wins}W of {rung.games})  "
        f"bank {rung.bank:9.0f} vs {rung.opponent_bank:9.0f}  "
        f"illegal {rung.illegal} of {rung.decisions}"
    )


if __name__ == "__main__":
    main()
