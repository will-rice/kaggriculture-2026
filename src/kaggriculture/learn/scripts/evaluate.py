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

**Completed sales are measured beside the bank.** A bank number says how much a
policy made and nothing about how it failed to. ``sales.sale_metrics`` reads
that off consecutive observations, so this drives the episode itself rather than
through ``rollout_many``: a ``Trajectory`` keeps the encoded tensors and throws
the observations away, and the shed and the market book live only in the
observation. Two states per environment are held at a time, never the season.

**Buying is watched from the same pairs.** ``sales.buy_units`` counts what
purchases credited to the shed, because the shaped reward's ``fuel`` term pays a
flat rate per unit and charges nothing for the coins, so an arm can raise its
return by shopping. That is a tripwire rather than a diagnostic: see
``BUY_UNITS_ALARM``.

**The opponent is measured out of the same episodes.** ``economic_policy`` banks
~158,000 to our ~3, so its trading profile is the target shape, and a bare bank
number does not say whether the gap is selling less often, selling smaller, or
buying less. Its ``private`` mapping is not shared, but it is stored on its own
seat, so scoring seat 1 alongside seat 0 costs one extra copy per turn and
yields the reference out of the *same* seeds, the same market and the same 719
turns -- which a second run against a different opponent could not. Those fields
ride every record under ``opponent_``.

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
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from kaggle_environments import make
from kaggle_environments.core import Environment

from kaggriculture.constants import ENVIRONMENT, EPISODE_STEPS
from kaggriculture.learn import corpus
from kaggriculture.learn.model import Policy, load_policy_weights

# The eval rollout drives the episode itself so that it can keep observations,
# but it does not decide anything itself: the masked sampling and the opponent
# actor come from `rollout`, so the policy this measures plays from exactly the
# distribution training samples from. A second implementation of masked
# sampling here would be a second thing to keep in step, and `rollout`'s module
# docstring is explicit that the mask must be applied before the softmax --
# which is one more reason not to spell it twice.
from kaggriculture.learn.rollout import (
    LEARNER,
    _agent_observation,
    _decide,
    _opponent_actor,
)
from kaggriculture.learn.rollout import (
    OPPONENT as OPPONENT_SEAT,
)
from kaggriculture.learn.sales import buy_units, sale_metrics
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
# Buying is free shaped reward and the gradient knows it. `toad_reward`'s `fuel`
# term pays a flat rate per unit of shed stock -- a fertilizer unit pays what a
# melon unit pays -- and in the baseline arm (`money_weight = 0.0`) the coins
# that bought the stock are never charged, so an arm can raise its shaped return
# by shopping. `_commit_unit` credits the shed on BUY_PRODUCT and BUY_ANIMAL
# without a capacity check, so purchased stock is not even held to the 100-unit
# cap.
#
# The ceiling: a unit is worth `FUEL_WEIGHT / NORMALISER` = 0.005 / 500 = 1e-5
# of shaped return, and 64 units a turn over 719 turns is 46,016 units, so
# 46,016 * 1e-5 = 0.460 -- against a shaped total of about 0.130, roughly 3.5x
# the entire signal. Today the term is ~1.5% of the signal, which is why the
# arms are running rather than restarted, but the gradient points uphill toward
# that ceiling the whole way.
#
# 1,300 units an episode is the line between the noise and an arm that has found
# it: under 3% of that ceiling, and far above what ordinary farming buys.
# Sustained above it, the arm is shopping rather than playing.
BUY_UNITS_ALARM = 1_300.0
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
            "win rate %.2f, percentile %.1f | ours %.1f sales/%.0f units at "
            "%.2f realisation, %.0f bought | theirs %.1f sales/%.0f units at "
            "%.2f realisation, %.0f bought",
            prefix,
            record["update"],
            record["eval_bank_mean"],
            record["opponent_bank_mean"],
            record["win_rate"],
            record["eval_percentile"],
            record["sales_per_episode"],
            record["units_per_episode"],
            record["realisation"],
            record["buy_units_per_episode"],
            record["opponent_sales_per_episode"],
            record["opponent_units_per_episode"],
            record["opponent_realisation"],
            record["opponent_buy_units_per_episode"],
        )
        if record["buy_units_per_episode"] > BUY_UNITS_ALARM:
            LOGGER.warning(
                "%s update %d buys %.0f units an episode, over the %.0f alarm: "
                "this arm may be farming the fuel term rather than playing",
                prefix,
                record["update"],
                record["buy_units_per_episode"],
                BUY_UNITS_ALARM,
            )


def measure_once(checkpoint: Path, name: str, banks: np.ndarray) -> dict[str, Any]:
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
        "%s: ours %.0f vs economic_policy %.0f, win rate %.2f, percentile %.1f, "
        "%.1f sales/%.0f units at %.2f realisation, %.0f buy-units/episode",
        name,
        record["eval_bank_mean"],
        record["opponent_bank_mean"],
        record["win_rate"],
        record["eval_percentile"],
        record["sales_per_episode"],
        record["units_per_episode"],
        record["realisation"],
        record["buy_units_per_episode"],
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
) -> dict[str, Any]:
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
    played = play(policy, seeds)

    wins = sum(
        1.0
        for ours, theirs in zip(played.ours, played.theirs, strict=True)
        if ours > theirs
    )
    mean = float(np.mean(played.ours))
    return {
        "arm": arm,
        "update": update,
        "checkpoint": checkpoint.name,
        "eval_bank_mean": mean,
        "eval_bank_max": float(np.max(played.ours)),
        "opponent_bank_mean": float(np.mean(played.theirs)),
        "win_rate": wins / len(played.ours),
        "eval_percentile": percentile(mean, banks),
        "games": len(played.ours),
        "seconds": round(time.monotonic() - started, 1),
        **played.tally.record(len(played.ours)),
        # The reference line, out of the same episodes rather than a second run:
        # same seeds, same market, same 719 turns, so a difference between the
        # two profiles is the policy and not the conditions. `economic_policy`
        # banks ~158,000 against our ~3, and until now nothing said what its
        # trading looks like -- how often it sells, how big a clear is, how much
        # it buys. Every arm's record now carries it, so the target shape is on
        # the same axes at every update instead of as one flat line.
        **played.opponent_tally.record(len(played.ours), prefix="opponent_"),
    }


@dataclass
class Tally:
    """What the turn pairs said, accumulated over any number of episodes.

    ``sale_metrics`` and ``buy_units`` both score a sequence of observations, and
    this adds one consecutive pair to a running total so the caller never has to
    hold a whole season. The totals are exact rather than an average of averages:
    a pair's proceeds are ``mean_sale_price * units`` and its market reference is
    ``mean_market_price * units``, both of which are the sums the function
    divided by, so accumulating them and dividing once at the end gives the same
    number as scoring the whole series in one call.

    Attributes:
        seat: Which seat this is scoring. The shed is only legible in that
            seat's own observation, so the pairs handed to ``add`` must be its
            own and this must be the index it sits at.
        sales: Completed clears -- one per good that cleared on a turn.
        units: How many items those clears moved.
        proceeds: Coins the clears banked.
        market: The volume-weighted market value of what was sold, priced at the
            book standing before each clear.
        bought: Units credited to the shed by purchases.
    """

    seat: int = LEARNER
    sales: float = 0.0
    units: float = 0.0
    proceeds: float = 0.0
    market: float = 0.0
    bought: float = 0.0

    def add(self, before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
        """Add one consecutive pair of this seat's own observations.

        Args:
            before: The state before the turn.
            after: The state after it.
        """
        pair = (before, after)
        metrics = sale_metrics(pair, self.seat)
        self.sales += metrics["sales"]
        self.units += metrics["units"]
        self.proceeds += metrics["mean_sale_price"] * metrics["units"]
        self.market += metrics["mean_market_price"] * metrics["units"]
        self.bought += buy_units(pair, self.seat)

    def record(self, episodes: int, prefix: str = "") -> dict[str, float]:
        """Return the eval-record fields for these totals.

        Args:
            episodes: How many episodes were tallied, so the counts read per
                episode rather than per group and stay comparable when ``GAMES``
                changes.
            prefix: Prepended to every key, so the opponent's profile can ride
                in the same flat record as ours without either shadowing the
                other.

        Returns:
            ``sales_per_episode``, ``units_per_episode``, ``mean_sale_price``,
            ``mean_market_price``, ``realisation`` -- realised over market,
            where 1.0 is par and below 1.0 is selling into a depressed book --
            and ``buy_units_per_episode``, each under ``prefix``.
        """
        sale = self.proceeds / self.units if self.units else 0.0
        market = self.market / self.units if self.units else 0.0
        return {
            f"{prefix}sales_per_episode": self.sales / episodes,
            f"{prefix}units_per_episode": self.units / episodes,
            f"{prefix}mean_sale_price": sale,
            f"{prefix}mean_market_price": market,
            f"{prefix}realisation": (sale / market) if market else 0.0,
            f"{prefix}buy_units_per_episode": self.bought / episodes,
        }


@dataclass(frozen=True)
class Games:
    """One group of evaluation episodes, played to the end.

    Attributes:
        ours: Each episode's terminal bank for the learner's seat.
        theirs: Each episode's terminal bank for the scripted opponent.
        tally: Our completed sales and purchases across the whole group.
        opponent_tally: The same, for the scripted opponent, out of the same
            episodes.
    """

    ours: list[float] = field(default_factory=list)
    theirs: list[float] = field(default_factory=list)
    tally: Tally = field(default_factory=Tally)
    opponent_tally: Tally = field(default_factory=lambda: Tally(seat=OPPONENT_SEAT))


def play(policy: Policy, seeds: Sequence[int]) -> Games:
    """Play a group of episodes against the scripted opponent, keeping observations.

    The lockstep loop of ``rollout_many`` without the trajectory: every
    environment is stepped together so one forward serves the group, and the
    decisions come from ``rollout``'s own sampler so this measures the same
    distribution training samples from. What it keeps instead of tensors is the
    part ``sale_metrics`` needs, and only the previous state of each
    environment -- one pair at a time, not the season.

    Args:
        policy: The checkpoint to measure, in eval mode.
        seeds: One seed per episode. The first also seeds the sampling stream,
            matching ``rollout_many``, so these episodes are the ones the gate
            would have played.

    Returns:
        The group's terminal banks and its completed-sale totals.
    """
    generator = torch.Generator().manual_seed(int(seeds[0]))
    environments = [
        make(ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS, "seed": seed})
        for seed in seeds
    ]
    for environment in environments:
        environment.reset(2)
    actors = [_opponent_actor(OPPONENT, environment) for environment in environments]

    games = Games()
    tallies = {LEARNER: games.tally, OPPONENT_SEAT: games.opponent_tally}
    previous = {
        (index, seat): snapshot(environment, seat)
        for index, environment in enumerate(environments)
        for seat in tallies
    }
    while not environments[0].done:
        turns = _decide(
            policy,
            [
                (environment.state[LEARNER].observation, LEARNER)
                for environment in environments
            ],
            generator,
        )
        for environment, turn, act in zip(environments, turns, actors, strict=True):
            environment.step(
                [turn.action, act(_agent_observation(environment, OPPONENT_SEAT))]
            )
        for index, environment in enumerate(environments):
            for seat, tally in tallies.items():
                seen = snapshot(environment, seat)
                tally.add(previous[index, seat], seen)
                previous[index, seat] = seen

    for environment in environments:
        terminal = environment.state[LEARNER].observation["farms"]
        games.ours.append(float(terminal[LEARNER]["money"]))
        games.theirs.append(float(terminal[OPPONENT_SEAT]["money"]))
    LOGGER.info(
        "played %d episodes: %.1f completed sales each",
        len(games.ours),
        games.tally.sales / len(games.ours),
    )
    return games


def snapshot(environment: Environment, seat: int) -> dict[str, Any]:
    """Return one seat's own observation, copied out of the state that mutates.

    ``Environment.step`` appends ``self.state`` without copying and the
    interpreter mutates the farms and the shed in place, so a kept reference to
    an observation reads as the terminal position at every turn -- every pair
    would then show no change and every sale would go uncounted. Only the three
    things ``sale_metrics`` reads are copied, so holding the previous state of
    sixteen environments costs a few hundred floats.

    The shed is the reason this takes a seat at all. ``private`` is not a shared
    property -- the engine gives each seat its own, and the specification agrees
    -- so the opponent's shed is legible only from the opponent's own stored
    observation. That is what makes the opponent measurable out of the very same
    episodes rather than out of a second run against different conditions.

    Args:
        environment: The episode to read.
        seat: Which seat's observation to copy.

    Returns:
        A standalone mapping with both farms' money, this seat's own shed, and
        the market book.
    """
    observation = environment.state[seat].observation
    return {
        "farms": [{"money": float(farm["money"])} for farm in observation["farms"]],
        "private": {"shed": dict(observation["private"]["shed"])},
        "market": {"prices": dict(observation["market"]["prices"])},
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
    load_policy_weights(policy, weights)
    return policy.eval()


def _update_of(checkpoint: Path) -> int:
    """Return the update number encoded in a checkpoint's name."""
    return int(checkpoint.stem.rsplit("_", 1)[-1])


if __name__ == "__main__":
    main()
