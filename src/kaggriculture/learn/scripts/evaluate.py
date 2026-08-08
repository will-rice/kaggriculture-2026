"""Play training checkpoints against a real opponent, outside the training loop.

Self-play bank says whether a policy beats itself, which is exactly the number a
private equilibrium inflates. This watches the checkpoints the runs already
write and plays each one against ``economic_policy`` -- a scripted agent that
banks ~155k and cannot drift -- so the question "is it getting better" is asked
against something outside the run.

It is a separate process on purpose. It never imports from a live run, never
writes where a run writes, and touches no training script; the only coupling is
that it reads ``.pt`` files after they appear.

The deliverable is the eval curve over training time. If self-play bank climbs
while eval bank does not, the run is inflating against itself and the opponent
pool moves up the queue.
"""

import argparse
import json
import logging
import os
import random
import time
import zipfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import wandb

from kaggriculture.learn import corpus
from kaggriculture.learn.model import Policy
from kaggriculture.learn.rollout import Trajectory, rollout_many
from kaggriculture.learn.scripts.gate import SEED_BASE

LOGGER = logging.getLogger(__name__)

RUNS = Path("/data/kaggriculture/toad")
# The scripted, state-conditioned opponent. It banks ~155k and cannot drift, so
# a rising score against it is a real improvement rather than a co-adaptation.
OPPONENT = "src/kaggriculture/economic_policy.py"
# Sixteen rather than the gate's 128: this runs continuously beside two training
# runs, and the eval curve's shape matters more than any single point's width.
# Seeded from the gate's base so these episodes are drawn from the same stream
# the gate will eventually use, and the numbers stay comparable.
GAMES = 16
# Episodes sampled once to build the reference bank distribution.
CORPUS_EPISODES = 200
# Fixed, so the reference distribution is the same array on every machine and
# every rebuild -- a percentile that moves when the cache is deleted is not a
# measurement.
CORPUS_SEED = 0
CORPUS_CACHE = RUNS / "corpus_banks.npy"
# Established earlier from the corpus manifests; our own distribution has to
# land near it or one of the two numbers is wrong and neither should be trusted.
CORPUS_MEDIAN = 125_773.0
POLL_SECONDS = 60


def main() -> None:
    """Watch for checkpoints and evaluate the newest one, forever."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", help="checkpoint stem to watch, e.g. phase1c3")
    parser.add_argument("--name", default=None, help="wandb run name")
    arguments = parser.parse_args()

    # Never compete with the training runs for CPU.
    os.nice(15)
    torch.set_num_threads(2)

    banks = reference_banks()
    LOGGER.info(
        "corpus reference: %d episodes, median %.0f (expected ~%.0f)",
        banks.size,
        float(np.median(banks)),
        CORPUS_MEDIAN,
    )

    log = RUNS / f"eval_{arguments.prefix}_{int(time.time())}.jsonl"
    run = wandb.init(
        entity="will-rice",
        project="kaggriculture-2026",
        name=arguments.name or f"toad-eval-{arguments.prefix}",
        config={"opponent": OPPONENT, "games": GAMES, "watching": arguments.prefix},
    )
    LOGGER.info("eval log %s, wandb %s", log, run.url)

    seen: set[Path] = set()
    while True:
        pending = sorted(
            (p for p in RUNS.glob(f"{arguments.prefix}_*.pt") if p not in seen),
            key=_update_of,
        )
        if not pending:
            time.sleep(POLL_SECONDS)
            continue
        # Newest first: a stale eval is worth less than a fresh one, so when the
        # evaluator falls behind it skips rather than queues. Older checkpoints
        # are backfilled only once the newest has been measured.
        checkpoint = pending[-1]
        seen.add(checkpoint)
        try:
            record = evaluate(checkpoint, banks)
        except Exception:
            LOGGER.exception("evaluation of %s failed", checkpoint)
            continue
        with log.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        wandb.log(record, step=record["update"])
        LOGGER.info(
            "update %d: ours %.0f vs %s %.0f, win rate %.2f, percentile %.1f",
            record["update"],
            record["eval_bank_mean"],
            "economic_policy",
            record["opponent_bank_mean"],
            record["win_rate"],
            record["eval_percentile"],
        )


def evaluate(checkpoint: Path, banks: np.ndarray) -> dict[str, float]:
    """Play one checkpoint against the scripted opponent and score it.

    Args:
        checkpoint: The ``.pt`` to load.
        banks: The corpus reference distribution.

    Returns:
        One record for the eval jsonl.
    """
    started = time.monotonic()
    policy = _load(checkpoint)
    seeds = tuple(SEED_BASE + index for index in range(GAMES))
    played: Sequence[Trajectory] = rollout_many(policy, OPPONENT, seeds)

    ours = [t.final_bank for t in played]
    # `final_margin` is our bank minus theirs, so their bank recovers exactly.
    theirs = [t.final_bank - t.final_margin for t in played]
    wins = sum(1.0 for t in played if t.final_margin > 0)
    mean = float(np.mean(ours))
    return {
        "update": _update_of(checkpoint),
        "checkpoint": checkpoint.name,
        "eval_bank_mean": mean,
        "eval_bank_max": float(np.max(ours)),
        "opponent_bank_mean": float(np.mean(theirs)),
        "win_rate": wins / len(played),
        "eval_percentile": percentile(mean, banks),
        "games": len(played),
        "seconds": round(time.monotonic() - started, 1),
    }


def percentile(bank: float, banks: np.ndarray) -> float:
    """Return where ``bank`` falls in the corpus distribution, as a percentage."""
    return float((banks < bank).mean() * 100.0)


def reference_banks() -> np.ndarray:
    """Return the corpus per-seat terminal bank distribution, sampled once.

    Cached, because it costs a few hundred episode decodes. The manifests carry
    ratings but not banked coins, so unlike the rest of the corpus work this one
    has to open the episodes.

    Returns:
        A sorted array of per-seat terminal banks.
    """
    if CORPUS_CACHE.is_file():
        return np.load(CORPUS_CACHE)

    archives = sorted(corpus.CORPUS.glob("*.zip"))
    if not archives:
        raise FileNotFoundError(f"no corpus archives under {corpus.CORPUS}")
    per_archive = max(1, CORPUS_EPISODES // len(archives))
    banks: list[float] = []
    for archive in archives:
        with zipfile.ZipFile(archive) as bundle:
            available = [n for n in bundle.namelist() if n.endswith(".json")]
            # Sampled rather than taken from the head of the listing. The
            # archives are ordered, so the first N episodes are not a random
            # draw -- taking them read 113,006 against an expected 125,773.
            names = random.Random(CORPUS_SEED).sample(
                available, min(per_archive, len(available))
            )
            for name in names:
                with bundle.open(name) as member:
                    episode = json.load(member)
                banks.extend(_terminal_banks(episode))
    array = np.sort(np.array(banks, dtype=float))
    CORPUS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(CORPUS_CACHE, array)
    return array


def _terminal_banks(episode: dict) -> list[float]:
    """Return both seats' banked coins at the end of one recorded episode."""
    farms = episode["steps"][-1][0]["observation"]["farms"]
    return [float(farm["money"]) for farm in farms]


def _load(checkpoint: Path) -> Policy:
    """Return the policy inside a training checkpoint, in eval mode.

    The training checkpoints wrap the weights beside the optimizer and counters,
    so they are not the bare state dict ``gate.under_test`` expects; the width is
    read off the weights rather than assumed, because the arms differ in it.

    Args:
        checkpoint: The ``.pt`` to load.

    Returns:
        The policy, ready to play.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = state["learner"] if "learner" in state else state
    channels = int(weights["stem.weight"].shape[0])
    blocks = 1 + max(int(k.split(".")[1]) for k in weights if k.startswith("blocks."))
    policy = Policy(blocks=blocks, channels=channels, value_bound=1.0)
    policy.load_state_dict(weights)
    return policy.eval()


def _update_of(checkpoint: Path) -> int:
    """Return the update number encoded in a checkpoint's name."""
    return int(checkpoint.stem.rsplit("_", 1)[-1])


if __name__ == "__main__":
    main()
