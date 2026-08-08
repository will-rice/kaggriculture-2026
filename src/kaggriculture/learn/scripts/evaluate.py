"""Play training checkpoints against a real opponent, outside the training loop.

Self-play bank says whether a policy beats itself, which is exactly the number a
private equilibrium inflates. This watches the checkpoints the runs already
write and plays each one against ``economic_policy`` -- a scripted agent that
banks ~155k and cannot drift -- so the question "is it getting better" is asked
against something outside the run.

It is a separate process on purpose. It never imports from a live run, never
writes where a run writes, and touches no training script; the only coupling is
that it reads ``.pt`` files after they appear.

One process watches every arm. A per-arm process was one wandb run per arm and
one nice'd python per arm for the same 16 episodes; this takes a list of
checkpoint prefixes, picks the newest unmeasured checkpoint *across* all of
them, and opens one wandb run per arm from the single process so the arms still
plot as separate series on shared axes.

The x-axis is declared, not implied. ``wandb.log(..., step=update)`` silently
drops any record whose step is below the highest already seen, and this walks a
prefix newest-first, so every backfilled checkpoint was being thrown away --
three measurements in the jsonl, zero points on the curve. Every metric is now
logged against an explicit ``eval/ckpt_step`` step metric, which makes the
order records arrive in irrelevant, and every record already in the jsonl is
replayed into the run at startup so a fresh run carries the whole history.

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
from collections.abc import Iterator, Sequence
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
ENTITY = "will-rice"
PROJECT = "kaggriculture-2026"
# The scripted, state-conditioned opponent. It banks ~155k and cannot drift, so
# a rising score against it is a real improvement rather than a co-adaptation.
OPPONENT = "src/kaggriculture/economic_policy.py"
# Sixteen rather than the gate's 128: this runs continuously beside the training
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
# ``--once`` measures a checkpoint that has no training curve of its own, and a
# lone point at x=0 does not read as the line the arms are being compared
# against. The single measurement is published at both ends of the arms' axis so
# the series draws flat across it.
REFERENCE_SPAN = (0, 2_000)


def main() -> None:
    """Watch every arm's checkpoints, or measure one checkpoint and exit."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "prefixes",
        nargs="*",
        help="checkpoint stems to watch, e.g. toad-phase1c3-clone-teacher",
    )
    parser.add_argument(
        "--once",
        type=Path,
        default=None,
        help="measure this one checkpoint as a reference line and exit",
    )
    parser.add_argument("--name", default=None, help="wandb run name for --once")
    arguments = parser.parse_args()
    if bool(arguments.prefixes) == (arguments.once is not None):
        parser.error("pass either checkpoint prefixes to watch or --once CHECKPOINT")

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

    if arguments.once is not None:
        measure_once(
            arguments.once, arguments.name or f"eval-{arguments.once.stem}", banks
        )
        return
    watch(tuple(arguments.prefixes), banks)


def watch(prefixes: Sequence[str], banks: np.ndarray) -> None:
    """Measure the newest unmeasured checkpoint across every arm, forever.

    Args:
        prefixes: The checkpoint stems to watch, one per arm.
        banks: The corpus reference distribution.
    """
    runs = {prefix: open_run(f"eval-{prefix}", prefix) for prefix in prefixes}
    logs = {prefix: RUNS / f"eval_{prefix}.jsonl" for prefix in prefixes}
    LOGGER.info(
        "watching %d arms: %s",
        len(prefixes),
        ", ".join(f"{prefix} -> {runs[prefix].url}" for prefix in prefixes),
    )

    # Replay what is already on disk before measuring anything new, so the run
    # opens with the whole curve rather than growing one point at a time.
    seen: set[Path] = set()
    for prefix in prefixes:
        replayed = 0
        for record in recorded(prefix):
            publish(runs[prefix], record)
            seen.add(RUNS / record["checkpoint"])
            replayed += 1
        LOGGER.info("replayed %d record(s) for %s", replayed, prefix)

    while True:
        pending = [
            (checkpoint, prefix)
            for prefix in prefixes
            for checkpoint in RUNS.glob(f"{prefix}_*.pt")
            if checkpoint not in seen
        ]
        if not pending:
            time.sleep(POLL_SECONDS)
            continue
        # Newest first, by write time rather than by update number: update
        # numbers restart per arm, and a stale eval is worth less than a fresh
        # one, so when the evaluator falls behind it skips rather than queues.
        # Older checkpoints are backfilled only once the newest has been
        # measured, which is exactly the decreasing-step walk the explicit
        # ``eval/ckpt_step`` axis exists to survive.
        checkpoint, prefix = max(pending, key=lambda pair: pair[0].stat().st_mtime)
        seen.add(checkpoint)
        try:
            record = evaluate(checkpoint, banks, prefix, _update_of(checkpoint))
        except Exception:
            LOGGER.exception("evaluation of %s failed", checkpoint)
            continue
        with logs[prefix].open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        publish(runs[prefix], record)
        LOGGER.info(
            "%s update %d: ours %.0f vs economic_policy %.0f, "
            "win rate %.2f, percentile %.1f",
            prefix,
            record["update"],
            record["eval_bank_mean"],
            record["opponent_bank_mean"],
            record["win_rate"],
            record["eval_percentile"],
        )


def measure_once(checkpoint: Path, name: str, banks: np.ndarray) -> dict[str, float]:
    """Measure one checkpoint and publish it as a flat reference line.

    Args:
        checkpoint: The ``.pt`` to load. May be a bare state dict.
        name: The wandb run name to publish under.
        banks: The corpus reference distribution.

    Returns:
        The record, also appended to ``eval_{name}.jsonl``.
    """
    run = open_run(name, str(checkpoint))
    LOGGER.info("reference %s -> %s", checkpoint, run.url)
    record = evaluate(checkpoint, banks, name, REFERENCE_SPAN[0])
    with (RUNS / f"eval_{name}.jsonl").open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    for step in REFERENCE_SPAN:
        publish(run, {**record, "update": step})
    LOGGER.info(
        "%s: ours %.0f vs economic_policy %.0f, win rate %.2f, percentile %.1f",
        name,
        record["eval_bank_mean"],
        record["opponent_bank_mean"],
        record["win_rate"],
        record["eval_percentile"],
    )
    run.finish()
    return record


def open_run(name: str, arm: str) -> wandb.Run:
    """Start a wandb run whose eval metrics are plotted against checkpoint step.

    ``reinit="create_new"`` because one process holds one run per arm; without
    it the second ``init`` would take over the first.

    Args:
        name: The run name.
        arm: What this run is watching, recorded in the config.

    Returns:
        The run, with its x-axis declared.
    """
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=name,
        config={"opponent": OPPONENT, "games": GAMES, "arm": arm},
        reinit="create_new",
    )
    run.define_metric("eval/ckpt_step")
    run.define_metric("eval/*", step_metric="eval/ckpt_step")
    return run


def publish(run: wandb.Run, record: dict) -> None:
    """Log one eval record against the explicit checkpoint-step axis.

    No ``step=`` argument: that is the monotonic counter that dropped the
    backfill. The checkpoint's update number travels as a logged value instead,
    so records may arrive in any order.

    Args:
        run: The arm's wandb run.
        record: One eval record.
    """
    payload = {
        f"eval/{key.removeprefix('eval_')}": value for key, value in record.items()
    }
    payload["eval/ckpt_step"] = record["update"]
    run.log(payload)


def recorded(prefix: str) -> Iterator[dict]:
    """Yield every eval record already written for one arm, oldest update first.

    Reads across every log this arm has ever written -- the daemon opens a new
    file when it is relaunched -- and keeps one record per checkpoint, the last
    written winning.

    Args:
        prefix: The arm's checkpoint stem.

    Yields:
        The arm's records, ordered by update.
    """
    records: dict[str, dict] = {}
    for log in sorted(RUNS.glob(f"eval_{prefix}*.jsonl")):
        for line in log.read_text().splitlines():
            record = json.loads(line)
            records[record["checkpoint"]] = record
    yield from sorted(records.values(), key=lambda record: record["update"])


def evaluate(
    checkpoint: Path, banks: np.ndarray, arm: str, update: int
) -> dict[str, float]:
    """Play one checkpoint against the scripted opponent and score it.

    Args:
        checkpoint: The ``.pt`` to load.
        banks: The corpus reference distribution.
        arm: Which run produced this checkpoint.
        update: The x position this record belongs at.

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
        "arm": arm,
        "update": update,
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
    """Return the policy inside a checkpoint, in eval mode.

    The training checkpoints wrap the weights beside the optimizer and counters;
    the older self-play runs saved the bare state dict. Both are accepted, and
    the shape is read off the weights rather than assumed, because the arms and
    the old PPO lineage differ in width.

    Args:
        checkpoint: The ``.pt`` to load.

    Returns:
        The policy, ready to play.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = state["learner"] if "learner" in state else state
    channels = int(weights["stem.weight"].shape[0])
    blocks = 1 + max(int(k.split(".")[1]) for k in weights if k.startswith("blocks."))
    LOGGER.info("%s: %d blocks, %d channels", checkpoint.name, blocks, channels)
    # ``value_bound`` reshapes nothing and is not a parameter, so it does not
    # affect loading; the value head is not read during a rollout's action
    # choice either, which is why the arms and the unbounded PPO lineage can
    # share one constructor here.
    policy = Policy(blocks=blocks, channels=channels, value_bound=1.0)
    policy.load_state_dict(weights)
    return policy.eval()


def _update_of(checkpoint: Path) -> int:
    """Return the update number encoded in a checkpoint's name."""
    return int(checkpoint.stem.rsplit("_", 1)[-1])


if __name__ == "__main__":
    main()
