"""Measure what a season of simulator rollout actually costs, and against what.

What the previous version of this script measured, and why it was wrong
======================================================================

This file used to report a single headline -- "28,030 environment-steps/second
at batch 1024" -- that was quoted as a verified property of the batched
simulator. It was not. Two separate defects, either of which is enough to make
the number mean something other than its label.

**Its CUDA-graph mode replayed a step that never advanced state.** It captured
``observe`` + ``legal`` + ``step`` once against fixed input buffers and then
called ``cuda_graph.replay()`` in a loop. ``step`` is functional: it reads the
input state and returns a *new* one. Under replay the new state is written into
the graph's private pool and thrown away, and the next replay starts from the
same unchanged inputs. So the loop recomputed turn zero N times. The reported
figure was the cost of replaying a frozen computation, not the cost of a
rollout. A graph mode can be honest here, but only if the captured region ends
by copying the successor state back over the input buffers, which is what
``_capture`` below does and what ``verify_graph`` proves it does.

**Even the eager mode benchmarked the wrong work.** It timed ``observe`` +
``legal`` + ``step`` with every unit passing and every market order empty. A
real turn has a policy forward, actions sampled from it, market orders that
resolve against prices, and a state that changes underneath all of it. Passing
units skip the whole unit-phase branch structure and empty orders skip the
market phase almost entirely.

The consequence was concrete. A separate measurement of a *real* evaluation
loop found the simulator 8% slower than the reference engine at the gate's own
batch of 128, and winning only at much larger widths -- because ``step`` is
launch-bound, costing about the same per turn at batch 128 as at batch 512, so
its cost is paid per batch rather than per season. None of that is visible in a
component benchmark, and this project has been burned by drifting labels before
(a value loss read as accuracy, a collection-only figure read as end-to-end
throughput).

What this version measures
==========================

Every configuration advances real state across a full ``SEASON_TURNS``-turn
season with actions that change every turn, and reports the units a plan can be
budgeted in: seconds per season at a stated batch size, seasons per hour, and
seconds per game so the per-game cost curve is visible.

Four engines, all doing the same per-turn work:

``store``
    ``sim.rollout.collect_segment``, the path PPO actually collects on: it
    records every tensor the update reads. This is the real cost of collection.
``store-graph``
    That same ``collect_segment`` call, one whole segment of it captured as a
    CUDA graph and replayed. This is the row that answers what collection would
    cost captured, rather than what a stripped-down stand-in for it would.
``eager``
    The same per-turn work in a plain Python loop, without the trajectory
    bookkeeping. Kept because the gap to ``store`` is what the recording costs,
    measured rather than assumed.
``graph``
    That same body captured once and replayed, with the successor state copied
    back over the input buffers inside the capture so each replay genuinely
    advances the season.

``collect_segment`` used to be uncapturable, and the first version of this file
said so: it ended every call with ``int(illegal_count.cpu())``, and one host
synchronisation anywhere in a captured region aborts the capture. It also
rebuilt six price tables per ``potential`` call with ``torch.tensor(...,
device=cuda)``, an unpinned host-to-device copy that capture refuses for the
same reason. Both are gone -- ``Trajectory.illegal`` is a device tensor the
caller materialises when it logs it, and the tables are cached per device --
which is what makes ``store-graph`` a row rather than a hypothetical.

Two policy settings, reported separately rather than blended: ``network`` runs
the real ``Policy`` trunk in the loop, ``none`` substitutes ``ZeroPolicy``,
whose zero logits make the masked softmax uniform over legal moves -- so
actions still vary every turn and only the network's arithmetic is removed. The
difference between the two rows at a batch is the network's share.

The reference engine, ``learn.rollout.rollout_many``, is measured on the same
work -- mirror self-play, both seats recorded, a full season -- so the output
answers "is this faster than what we have" rather than "how fast is this in
isolation". It is measured in the shape the gate actually runs it: ``gate.GAMES``
episodes across ``gate.WORKERS`` spawned processes at ``gate.THREADS`` threads
each, all three imported rather than restated. Timing it in one process instead
would have flattered the simulator by roughly the worker count, against a
configuration nobody runs. ``versus_reference`` divides the reference's
wall-clock seconds per game by the simulator's.

What this benchmark does not claim
==================================

It is a cost measurement, not a fidelity one. The simulator currently fails its
nightly-size differential campaign against the reference engine, so the two
engines here are playing games that agree in shape and work but not yet in
every coin. That is being fixed separately and does not move these numbers: a
fidelity fix changes what the kernels compute, not how many are launched.
"""

import argparse
import json
import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
from typing import Sequence

import torch

from kaggriculture.constants import EPISODE_STEPS
from kaggriculture.learn.encoding import MARKET_SLOTS, MAX_UNITS, QUANTITIES, UNIT_OPS
from kaggriculture.learn.model import Policy
from kaggriculture.learn.rollout import rollout_many
from kaggriculture.learn.scripts.gate import DEVICE, GAMES, THREADS, WORKERS
from kaggriculture.sim.config import Config
from kaggriculture.sim.decode import decode_market_buckets
from kaggriculture.sim.engine import reset, step
from kaggriculture.sim.legality import legal
from kaggriculture.sim.observe import observe
from kaggriculture.sim.rollout import Trajectory, collect_segment
from kaggriculture.sim.state import SimState
from kaggriculture.sim.tensors import tensor_constant

LOGGER = logging.getLogger("benchmark_simulator")

# One season is `episodeSteps` states with a decision between each pair, so 719
# decisions. Read from the constant rather than written out, because a benchmark
# whose season length drifts from the game's is the failure this file exists to
# stop.
SEASON_TURNS = EPISODE_STEPS - 1

BATCHES = (128, 256, 1024, 4096)
ENGINES = ("store", "store-graph", "eager", "graph")

# `collect_segment` keeps every turn of a segment resident: at batch 4096 the
# board planes alone are 157 MB per turn. The segment length is therefore capped
# by a row budget rather than fixed, which does not change the per-turn cost --
# the engine's work is identical and only the number of stacks differs.
SEGMENT_TURNS = 32
SEGMENT_ROWS = 32 * 1024

# Held-out seeds. Nothing trains on these, so a run here never warms a cache
# that a training measurement would then benefit from.
SEED_BASE = 3_000_000_000

# The reference is measured in the configuration the gate actually runs it in --
# `gate.GAMES` episodes split across `gate.WORKERS` spawned processes, each
# playing its share through one `rollout_many` -- rather than in a single
# process. Both numbers are imported rather than restated so that a change to
# the gate moves the thing this benchmark is compared against, instead of
# quietly leaving the comparison measuring a configuration nobody runs.
REFERENCE_GAMES = GAMES
REFERENCE_WORKERS = WORKERS

# Graph equivalence is checked on a small batch because it is an exactness
# check, not a throughput one: every field of every state must match.
VERIFY_BATCH = 32
VERIFY_TURNS = 40
VERIFY_SEGMENT = 5
WARMUP_TURNS = 3
WARMUP_SEGMENTS = 2

# The logit gap a `PeakedPolicy` puts between adjacent options. Large enough
# that `exp` of the gap underflows to zero in float32, so the masked softmax is
# one-hot and the multinomial draw off it is deterministic whatever the RNG
# does -- which is what lets a sampling body be compared against an eager run
# turn for turn.
PEAK = 1e4


def main() -> None:
    """Run the sweep, verify the graph mode, and log one record per configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=list(BATCHES))
    parser.add_argument("--engines", nargs="+", default=list(ENGINES), choices=ENGINES)
    parser.add_argument(
        "--turns",
        type=int,
        default=SEASON_TURNS,
        help=(
            "turns to advance per configuration; anything short of "
            f"{SEASON_TURNS} marks the season figures extrapolated"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="omit the reference engine, leaving versus_reference unpopulated",
    )
    parser.add_argument("--reference-games", type=int, default=REFERENCE_GAMES)
    parser.add_argument("--reference-workers", type=int, default=REFERENCE_WORKERS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError(f"the simulator is a CUDA program; got {device}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    LOGGER.info(
        "device=%s (%s) torch=%s season_turns=%d measured_turns=%d",
        device,
        torch.cuda.get_device_name(device),
        torch.__version__,
        SEASON_TURNS,
        args.turns,
    )

    if "graph" in args.engines:
        verify_graph(device)
    if "store-graph" in args.engines:
        verify_segment_graph(device)

    reference = (
        None
        if args.skip_reference
        else benchmark_reference(
            args.reference_games, args.reference_workers, args.turns
        )
    )
    measurements = [] if reference is None else [reference]
    for batch in args.batches:
        for engine in args.engines:
            for policy_in_loop in (True, False):
                measurements.append(
                    benchmark(
                        device,
                        batch=batch,
                        turns=args.turns,
                        engine=engine,
                        policy_in_loop=policy_in_loop,
                        reference=reference,
                    )
                )
    report(measurements)


@dataclass(frozen=True)
class Measurement:
    """One timed configuration, carrying everything needed to read its number.

    Attributes:
        engine: ``store``, ``store-graph``, ``eager``, ``graph`` or
            ``reference``.
        batch: Seasons the configuration plays at once. For the simulator that
            is the tensor batch; for the reference it is the total number of
            episodes across all of its worker processes.
        workers: Processes the configuration used. One for every simulator row;
            the reference is measured with the gate's own worker count, because
            that is the throughput it would have to be beaten at.
        turns: Turns actually advanced.
        policy_in_loop: True when the real ``Policy`` trunk ran every turn.
        device: The CUDA device the work ran on.
        seconds: Wall clock for the whole configuration, after warmup.
        seconds_per_season: Wall clock for ``batch`` seasons of
            ``SEASON_TURNS`` turns. Equal to ``seconds`` when the full season
            was measured.
        seconds_per_game: ``seconds_per_season / batch`` -- the per-game cost
            curve, which is the whole point of sweeping the batch.
        seasons_per_hour: How many seasons an hour of this configuration buys.
        extrapolated: True when fewer than ``SEASON_TURNS`` turns were timed and
            the season figures were scaled up from them.
        load_average: The machine's one-minute load average when the
            configuration finished. Recorded because the simulator's eager loop
            is launch-bound -- its cost is a Python thread issuing kernels -- and
            the reference is outright CPU-bound, so a number taken on a busy box
            is a different number. A reader comparing two runs needs to see this
            rather than assume it.
        versus_reference: The reference engine's seconds per game divided by
            this configuration's. Above 1.0 means the simulator is faster.
            ``None`` when the reference was not measured.
    """

    engine: str
    batch: int
    workers: int
    turns: int
    policy_in_loop: bool
    device: str
    seconds: float
    seconds_per_season: float
    seconds_per_game: float
    seasons_per_hour: float
    extrapolated: bool
    load_average: float
    versus_reference: float | None


class ZeroPolicy(torch.nn.Module):
    """Stand-in that removes the network's arithmetic and nothing else.

    Its zero logits leave the masked softmax uniform over the legal moves, so
    every turn still samples a different action and the state still advances
    along a real trajectory. The gap between a ``network`` row and a ``none``
    row at the same batch is therefore the trunk's cost alone.
    """

    def forward(
        self, board: torch.Tensor, scalars: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return zero unit, quantity and market logits, and a zero value."""
        rows = len(board)
        device = board.device
        return (
            torch.zeros(rows, MAX_UNITS, len(UNIT_OPS), device=device),
            torch.zeros(rows, MAX_UNITS, len(QUANTITIES), device=device),
            torch.zeros(rows, len(MARKET_SLOTS) + 2, len(QUANTITIES), device=device),
            torch.zeros(rows, 1, device=device),
        )


class PeakedPolicy(ZeroPolicy):
    """Stand-in whose masked softmax is one-hot, so sampling is deterministic.

    ``collect_segment`` always samples; it has no argmax mode, and it should not
    grow one just to be verifiable. So the determinism is put in the policy
    instead: logits spaced ``PEAK`` apart make every masked distribution one-hot
    on its highest-numbered legal option, and a multinomial draw off a one-hot
    distribution returns that option whatever random number it drew. The
    captured segment and the eager segment therefore play the same actions and
    can be compared field by field, which is what ``verify_segment_graph``
    needs. The masks still change every turn, so the season is still a real one.
    """

    def forward(
        self, board: torch.Tensor, scalars: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return logits that peak on one option per slot, and a zero value."""
        units, quantities, market, value = super().forward(board, scalars, positions)
        qwidth = quantities.shape[-1]
        return (
            units + PEAK * torch.arange(units.shape[-1], device=units.device),
            quantities + PEAK * torch.arange(qwidth, device=quantities.device),
            market + PEAK * torch.arange(market.shape[-1], device=market.device),
            value,
        )


def benchmark(
    device: torch.device,
    *,
    batch: int,
    turns: int,
    engine: str,
    policy_in_loop: bool,
    reference: Measurement | None,
) -> Measurement:
    """Advance one configuration through ``turns`` turns and price the season.

    Args:
        device: The CUDA device.
        batch: Environments advanced together.
        turns: Turns to advance. State changes on every one of them.
            ``store-graph`` advances in whole captured segments, so it plays the
            largest multiple of its segment length that fits and its row is
            marked extrapolated; every other engine plays exactly ``turns``.
        engine: ``store``, ``store-graph``, ``eager`` or ``graph``.
        policy_in_loop: Whether to run the real ``Policy`` trunk.
        reference: The reference measurement, for ``versus_reference``.

    Returns:
        The configuration's ``Measurement``.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    policy = _policy(device, policy_in_loop)
    seeds = torch.arange(batch, dtype=torch.int64, device=device) + SEED_BASE
    generator = torch.Generator(device=device).manual_seed(SEED_BASE)

    warm = reset(Config(), seeds)
    for _ in range(WARMUP_TURNS):
        warm = turn(warm, policy, sample=True)
    del warm
    torch.cuda.synchronize()

    state = reset(Config(), seeds)
    measured = turns
    if engine == "graph":
        captured = _capture(state, policy)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(turns):
            captured.replay()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        advanced = int(state.step[0])
        del captured
    elif engine == "store-graph":
        length = segment_length(batch)
        replays = max(1, turns // length)
        measured = replays * length
        captured, trajectory = _capture_segment(state, policy, length, generator)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(replays):
            captured.replay()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        advanced = int(state.step[0])
        del captured, trajectory
    else:
        started = time.perf_counter()
        state = (
            _run_store(state, policy, turns, generator)
            if engine == "store"
            else _run_eager(state, policy, turns)
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        advanced = int(state.step[0])
    if advanced != measured:
        raise RuntimeError(
            f"{engine} at batch {batch} advanced {advanced} turns, not {measured}; "
            "the configuration is not stepping the season it claims to"
        )
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    del state, policy
    torch.cuda.empty_cache()

    seconds_per_season = elapsed * SEASON_TURNS / measured
    seconds_per_game = seconds_per_season / batch
    measurement = Measurement(
        engine=engine,
        batch=batch,
        workers=1,
        turns=measured,
        policy_in_loop=policy_in_loop,
        device=str(device),
        seconds=elapsed,
        seconds_per_season=seconds_per_season,
        seconds_per_game=seconds_per_game,
        seasons_per_hour=3600.0 / seconds_per_game,
        extrapolated=measured != SEASON_TURNS,
        load_average=os.getloadavg()[0],
        versus_reference=(
            None if reference is None else reference.seconds_per_game / seconds_per_game
        ),
    )
    LOGGER.info("%s peak_gpu_gb=%.2f", json.dumps(asdict(measurement)), peak)
    return measurement


def benchmark_reference(games: int, workers: int, turns: int) -> Measurement:
    """Time the reference engine on the same work, in the shape the gate runs it.

    ``games`` mirror self-play seasons -- both seats the same network, both
    recorded, one forward per turn for a worker's whole group -- split into
    ``workers`` spawned processes, which is exactly ``gate.measure``. Timing one
    process instead would price a configuration nobody runs and make the
    simulator look about eight times better than it is against the thing it
    would replace.

    The reference cannot be stopped part way: ``rollout_many`` drives each
    ``kaggle_environments`` episode until it reports done, so it always plays
    the full season regardless of ``turns``. That is stated rather than hidden.

    Args:
        games: Total episodes, split evenly across the workers.
        workers: Spawned processes playing them.
        turns: The sweep's turn count, logged when it differs from the season.

    Returns:
        The reference ``Measurement``.

    Raises:
        ValueError: If the games do not divide evenly across the workers, which
            would silently drop the remainder and price fewer seasons than the
            row claims.
        RuntimeError: If a worker played an episode of an unexpected length.
    """
    if games % workers:
        raise ValueError(f"{games} games do not divide across {workers} workers")
    if turns != SEASON_TURNS:
        LOGGER.info(
            "the reference always plays the whole %d-turn season; the sweep's "
            "%d-turn setting does not apply to it",
            SEASON_TURNS,
            turns,
        )
    share = games // workers
    assignments = [
        tuple(SEED_BASE + worker * share + index for index in range(share))
        for worker in range(workers)
    ]
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        played = list(pool.map(reference_worker, assignments))
    elapsed = time.perf_counter() - started
    if set(played) != {SEASON_TURNS}:
        raise RuntimeError(
            f"the reference played episodes of {sorted(set(played))} turns, not "
            f"{SEASON_TURNS}; the comparison would be against different work"
        )
    seconds_per_game = elapsed / (share * workers)
    measurement = Measurement(
        engine="reference",
        batch=share * workers,
        workers=workers,
        turns=SEASON_TURNS,
        policy_in_loop=True,
        device=DEVICE,
        seconds=elapsed,
        seconds_per_season=elapsed,
        seconds_per_game=seconds_per_game,
        seasons_per_hour=3600.0 / seconds_per_game,
        extrapolated=False,
        load_average=os.getloadavg()[0],
        versus_reference=1.0,
    )
    LOGGER.info(json.dumps(asdict(measurement)))
    return measurement


def reference_worker(seeds: Sequence[int]) -> int:
    """Play one worker's share of reference seasons. Runs in the worker process.

    A freshly initialised ``Policy`` rather than a checkpoint: the cost of a
    forward does not depend on what the weights are, and loading one would tie
    the benchmark to a file that may not exist.

    ``torch.set_num_threads(gate.THREADS)`` is not incidental. The gate's own
    workers set it, and without it eight processes each open a thread pool the
    width of the machine and spend the run fighting each other -- which would
    price the reference at whatever the contention happened to be rather than at
    what the gate costs. The constant is imported for the same reason the worker
    count is.

    Args:
        seeds: The worker's episodes.

    Returns:
        How many turns its first episode played, which the caller checks is a
        whole season.
    """
    torch.set_num_threads(THREADS)
    policy = Policy().to(DEVICE).eval()
    return int(rollout_many(policy, policy, seeds)[0].rewards.shape[0])


def turn(state: SimState, policy: torch.nn.Module, *, sample: bool) -> SimState:
    """Advance one turn: observe both seats, mask, decide, and step.

    The same per-turn work ``sim.rollout.collect_segment`` does, minus its
    trajectory bookkeeping. It is spelled out here rather than reused so that
    the cost of the bookkeeping is a measured gap -- ``store`` against ``eager``
    -- rather than an assumption. It was originally spelled out for a second
    reason that no longer holds: ``collect_segment`` used to end every call with
    ``int(illegal_count.cpu())``, a host synchronisation that aborts a CUDA
    graph capture, so this was the only body ``graph`` could capture. It is now
    capturable itself, which is what the ``store-graph`` engine times.

    Args:
        state: The batch to advance.
        policy: The network, or ``ZeroPolicy``.
        sample: Draw from the masked distribution, as a rollout does. False
            takes the masked argmax instead, which is what ``verify_graph``
            needs to compare two runs exactly.

    Returns:
        The successor state.
    """
    observed = [observe(state, seat) for seat in range(2)]
    boards = torch.stack([value[0] for value in observed], dim=1)
    scalars = torch.stack([value[1] for value in observed], dim=1)
    positions = torch.stack([value[2] for value in observed], dim=1)
    masks = [legal(state, seat) for seat in range(2)]
    unit_masks = torch.stack([value[0] for value in masks], dim=1)
    quantity_masks = torch.stack([value[1] for value in masks], dim=1)
    market_masks = torch.stack([value[2] for value in masks], dim=1)
    batch = state.batch_size
    with torch.no_grad():
        unit_logits, quantity_logits, market_logits, _value = policy(
            boards.flatten(0, 1), scalars.flatten(0, 1), positions.flatten(0, 1)
        )
    unit_logits = unit_logits.reshape(batch, 2, *unit_logits.shape[1:])
    quantity_logits = quantity_logits.reshape(batch, 2, *quantity_logits.shape[1:])
    market_logits = market_logits.reshape(batch, 2, *market_logits.shape[1:])
    chosen_units = _decide(unit_logits, unit_masks, sample=sample)
    chosen_quantities = _decide(quantity_logits, quantity_masks, sample=sample)
    chosen_market = _decide(market_logits, market_masks, sample=sample)
    # The timed turn plays the same three lanes the collector does. Handing
    # `step` a column of ones here instead would time a turn the training loop
    # no longer takes -- one masked softmax, one multinomial and one gather
    # short of it, on a tensor of the crew's width.
    return step(
        state,
        chosen_units.to(torch.int16),
        decode_market_buckets(chosen_market),
        tensor_constant(QUANTITIES, dtype=torch.int16, device=state.step.device)[
            chosen_quantities
        ],
    )


def _decide(logits: torch.Tensor, mask: torch.Tensor, *, sample: bool) -> torch.Tensor:
    """Return one chosen index per slot, sampled under the mask or argmaxed under it.

    Sampling uses the default CUDA generator rather than an explicit one so that
    ``torch.cuda.graph`` registers its state and each replay draws fresh
    randomness. A CPU generator would not reach these tensors and an
    unregistered CUDA one would replay the same draw forever, which is exactly
    the frozen-computation failure this file documents.

    Args:
        logits: ``(batch, 2, slots, options)`` head output.
        mask: The same shape, True where the option is legal.
        sample: Draw, or take the masked argmax.

    Returns:
        ``(batch, 2, slots)`` int64 indices.
    """
    masked = logits.masked_fill(~mask, -torch.inf)
    if not sample:
        return masked.argmax(dim=-1)
    rows = torch.log_softmax(masked, dim=-1).flatten(0, -2)
    return torch.multinomial(rows.exp(), 1).reshape(logits.shape[:-1])


def _policy(device: torch.device, policy_in_loop: bool) -> torch.nn.Module:
    """Return the trunk or the stand-in that removes it."""
    if policy_in_loop:
        return Policy().to(device).eval()
    return ZeroPolicy().to(device)


def _run_eager(state: SimState, policy: torch.nn.Module, turns: int) -> SimState:
    """Advance ``turns`` turns in a plain Python loop."""
    for _ in range(turns):
        state = turn(state, policy, sample=True)
    return state


def _run_store(
    state: SimState, policy: torch.nn.Module, turns: int, generator: torch.Generator
) -> SimState:
    """Advance ``turns`` turns through ``collect_segment``, discarding each segment.

    PPO collects a segment, updates on it and drops it, so dropping it here is
    what the real loop does with the memory. The segments are chained -- each
    starts from the state the last one ended in -- so the season advances
    continuously rather than restarting.

    Args:
        state: The batch to advance.
        policy: The network, or ``ZeroPolicy``.
        turns: Total turns across all segments.
        generator: The sampling stream.

    Returns:
        The state after the last segment.
    """
    length = segment_length(state.batch_size)
    remaining = turns
    while remaining > 0:
        state, _trajectory = collect_segment(
            state, policy, turns=min(length, remaining), generator=generator
        )
        remaining -= min(length, remaining)
    return state


def segment_length(batch: int) -> int:
    """Return the segment length used at a batch, by both ``store`` engines.

    Shared so that the captured row and the eager row collect segments of the
    same shape. A graph row measured against an eager row of a different
    segment length would be comparing two different amounts of work.

    Args:
        batch: Environments advanced together.

    Returns:
        Turns per segment.
    """
    return max(1, min(SEGMENT_TURNS, SEGMENT_ROWS // batch))


def _capture_segment(
    state: SimState, policy: torch.nn.Module, length: int, generator: torch.Generator
) -> tuple[torch.cuda.CUDAGraph, Trajectory]:
    """Capture one whole ``collect_segment`` call, so a replay collects a segment.

    The copy-back is the same mechanism ``_capture`` uses and is needed for the
    same reason: ``collect_segment`` returns a successor state and leaves its
    input alone, so without it every replay would re-collect the first segment
    of the season from unchanged buffers. Here it is a whole segment rather than
    a turn, so one replay advances ``length`` turns.

    ``generator`` is registered with the graph before capture. An unregistered
    generator would replay one fixed draw forever -- the frozen-computation
    failure this file exists to prevent -- and registering the explicit one lets
    the captured row sample from exactly the stream the eager row does.

    Args:
        state: The buffers the graph reads and writes; the season's live state.
        policy: The network, or ``ZeroPolicy``.
        length: Turns per captured segment.
        generator: The sampling stream, registered with the graph.

    Returns:
        The captured graph, and the ``Trajectory`` its replays overwrite.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        warm = state
        for _ in range(WARMUP_SEGMENTS):
            warm = collect_segment(warm, policy, turns=length, generator=generator)[0]
    torch.cuda.current_stream().wait_stream(side)
    del warm
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    graph.register_generator_state(generator)
    with torch.cuda.graph(graph):
        successor, trajectory = collect_segment(
            state, policy, turns=length, generator=generator
        )
        for field in fields(SimState):
            getattr(state, field.name).copy_(getattr(successor, field.name))
    torch.cuda.synchronize()
    return graph, trajectory


def _capture(state: SimState, policy: torch.nn.Module) -> torch.cuda.CUDAGraph:
    """Capture one state-advancing turn, so that replaying it plays the season.

    ``step`` is functional: it returns a new state and leaves its input alone.
    Captured naively that makes a graph whose every replay recomputes the same
    turn from the same buffers, which is precisely what the previous version of
    this script timed. The fix is the loop at the end of the capture: every
    field of the successor is copied back over the corresponding input buffer,
    inside the captured region and after every read of it, so replay ``n+1``
    starts from the state replay ``n`` produced.

    The warmup runs on a side stream because capture requires the allocator and
    every lazily built constant to be warm; ``sim.tensors.tensor_constant``
    caches per device for exactly this reason.

    Args:
        state: The buffers the graph will read and write. They are the season's
            live state afterwards -- reading ``state.step`` after ``n`` replays
            reports ``n``.
        policy: The network, or ``ZeroPolicy``.

    Returns:
        The captured graph.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        warm = state
        for _ in range(WARMUP_TURNS):
            warm = turn(warm, policy, sample=True)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        successor = turn(state, policy, sample=True)
        for field in fields(SimState):
            getattr(state, field.name).copy_(getattr(successor, field.name))
    torch.cuda.synchronize()
    return graph


def verify_graph(device: torch.device) -> None:
    """Prove the captured graph advances the season, before any graph number is quoted.

    Two checks, one for each way the old benchmark's graph mode was wrong.

    The first replays the graph ``VERIFY_TURNS`` times under a deterministic
    decision rule and compares every field of the resulting state against an
    eager run of the same length from the same seeds. Equality across all of
    them is a strong statement: a graph that replayed a frozen turn would sit at
    step 1, and one that advanced but computed something else would diverge in
    some plane.

    The second replays the sampling body and checks that successive replays
    choose different actions. A captured RNG that did not advance would give a
    state that moves but a policy that plays one fixed turn forever, which is
    still not a rollout.

    Args:
        device: The CUDA device.

    Raises:
        RuntimeError: If the graph and the eager loop disagree on any state
            field, or if the sampled actions never change.
    """
    policy = ZeroPolicy().to(device)
    seeds = torch.arange(VERIFY_BATCH, dtype=torch.int64, device=device) + SEED_BASE

    replayed = reset(Config(), seeds)
    graph = _capture_deterministic(replayed, policy)
    for _ in range(VERIFY_TURNS):
        graph.replay()
    torch.cuda.synchronize()

    stepped = reset(Config(), seeds)
    for _ in range(VERIFY_TURNS):
        stepped = turn(stepped, policy, sample=False)
    divergent = [
        field.name
        for field in fields(SimState)
        if not torch.equal(getattr(replayed, field.name), getattr(stepped, field.name))
    ]
    if divergent:
        raise RuntimeError(
            f"the captured graph diverged from the eager loop after "
            f"{VERIFY_TURNS} turns in: {', '.join(divergent)}"
        )
    del graph, replayed, stepped

    sampled = reset(Config(), seeds)
    graph = _capture(sampled, policy)
    banks = [sampled.money.clone()]
    for _ in range(VERIFY_TURNS):
        graph.replay()
        banks.append(sampled.money.clone())
    torch.cuda.synchronize()
    if int(sampled.step[0]) != VERIFY_TURNS:
        raise RuntimeError(
            f"{VERIFY_TURNS} replays advanced the season to step "
            f"{int(sampled.step[0])}; the captured step is not advancing state"
        )
    if all(torch.equal(banks[0], later) for later in banks[1:]):
        raise RuntimeError(
            "no replay moved either farm's bank, so the captured turn is not "
            "playing the game it is being timed on"
        )
    del graph, sampled
    torch.cuda.empty_cache()
    LOGGER.info(
        "graph verified: %d replays match %d eager turns on every state field, "
        "and the season advances under sampling",
        VERIFY_TURNS,
        VERIFY_TURNS,
    )


def _capture_deterministic(
    state: SimState, policy: torch.nn.Module
) -> torch.cuda.CUDAGraph:
    """Capture the argmax body, the one an eager run can be compared against exactly."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        warm = state
        for _ in range(WARMUP_TURNS):
            warm = turn(warm, policy, sample=False)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        successor = turn(state, policy, sample=False)
        for field in fields(SimState):
            getattr(state, field.name).copy_(getattr(successor, field.name))
    torch.cuda.synchronize()
    return graph


def verify_segment_graph(device: torch.device) -> None:
    """Prove a captured ``collect_segment`` collects what the eager one collects.

    ``verify_graph`` proves the *turn* body replays honestly. This proves the
    same for the body the training loop would actually capture, and it is a
    stronger statement, because a segment carries more than a state: replaying
    it must reproduce every recorded tensor as well.

    The first check replays a captured segment under ``PeakedPolicy``, whose
    one-hot distributions make sampling deterministic, and compares the result
    against the same number of eager ``collect_segment`` calls from the same
    seeds -- all 39 ``SimState`` fields and all 16 ``Trajectory`` fields,
    ``illegal`` included. A graph that replayed a frozen segment would sit at
    the first segment's clock; one that advanced but computed something else
    would differ in some plane or some recorded row.

    The second check replays the sampling body and requires the season clock to
    reach the right turn, a farm's bank to have moved, and two replays to have
    recorded different actions. A captured RNG that did not advance would give a
    state that moves but a policy that replays one fixed segment forever.

    Args:
        device: The CUDA device.

    Raises:
        RuntimeError: If the captured segment disagrees with the eager one on
            any state field or any recorded tensor, if the clock does not reach
            the turn the replays claim, or if the sampled actions never change.
    """
    seeds = torch.arange(VERIFY_BATCH, dtype=torch.int64, device=device) + SEED_BASE
    replays = VERIFY_TURNS // VERIFY_SEGMENT
    advanced = replays * VERIFY_SEGMENT

    peaked = PeakedPolicy().to(device)
    replayed = reset(Config(), seeds)
    graph, trajectory = _capture_segment(
        replayed,
        peaked,
        VERIFY_SEGMENT,
        torch.Generator(device=device).manual_seed(SEED_BASE),
    )
    for _ in range(replays):
        graph.replay()
    torch.cuda.synchronize()

    stepped = reset(Config(), seeds)
    eager = torch.Generator(device=device).manual_seed(SEED_BASE)
    for _ in range(replays):
        stepped, collected = collect_segment(
            stepped, peaked, turns=VERIFY_SEGMENT, generator=eager
        )
    divergent = [
        field.name
        for field in fields(SimState)
        if not torch.equal(getattr(replayed, field.name), getattr(stepped, field.name))
    ] + [
        f"trajectory.{field.name}"
        for field in fields(Trajectory)
        if not torch.equal(
            getattr(trajectory, field.name), getattr(collected, field.name)
        )
    ]
    if divergent:
        raise RuntimeError(
            f"the captured segment diverged from the eager one after {replays} "
            f"segments of {VERIFY_SEGMENT} turns in: {', '.join(divergent)}"
        )
    if int(replayed.step[0]) != advanced:
        raise RuntimeError(
            f"{replays} segment replays advanced the season to step "
            f"{int(replayed.step[0])}, not {advanced}"
        )
    del graph, trajectory, replayed, stepped, collected

    sampled = reset(Config(), seeds)
    graph, trajectory = _capture_segment(
        sampled,
        ZeroPolicy().to(device),
        VERIFY_SEGMENT,
        torch.Generator(device=device).manual_seed(SEED_BASE),
    )
    banks = [sampled.money.clone()]
    actions = []
    for _ in range(replays):
        graph.replay()
        banks.append(sampled.money.clone())
        actions.append(trajectory.unit_actions.clone())
    torch.cuda.synchronize()
    if int(sampled.step[0]) != advanced:
        raise RuntimeError(
            f"{replays} segment replays advanced the season to step "
            f"{int(sampled.step[0])}, not {advanced}; the captured segment is "
            "not advancing state"
        )
    if all(torch.equal(banks[0], later) for later in banks[1:]):
        raise RuntimeError(
            "no replay moved either farm's bank, so the captured segment is not "
            "playing the game it is being timed on"
        )
    if all(torch.equal(actions[0], later) for later in actions[1:]):
        raise RuntimeError(
            "every replay recorded the same actions, so the captured RNG is not "
            "advancing and the segment is one fixed segment replayed"
        )
    if int(trajectory.illegal):
        raise RuntimeError(
            f"the captured segment recorded {int(trajectory.illegal)} actions "
            "its own stored mask forbade"
        )
    del graph, trajectory, sampled
    torch.cuda.empty_cache()
    LOGGER.info(
        "segment graph verified: %d replays of a %d-turn captured segment match "
        "%d eager turns on every state field and every recorded tensor, and the "
        "season advances under sampling",
        replays,
        VERIFY_SEGMENT,
        advanced,
    )


def report(measurements: list[Measurement]) -> None:
    """Log the sweep as one table, with the configuration beside every number.

    Every column that changes the meaning of a row is printed on the row --
    engine, batch, turns, whether the policy ran, the device -- because a
    throughput figure quoted without its batch size is the error this file was
    rewritten to stop.

    Args:
        measurements: Every timed configuration, reference first if measured.
    """
    header = (
        f"{'engine':<10}{'batch':>7}{'proc':>6}{'turns':>7}{'policy':>8}"
        f"{'s/season':>11}{'s/game':>10}{'seasons/hr':>12}{'vs ref':>9}"
    )
    LOGGER.info("")
    LOGGER.info(header)
    LOGGER.info("-" * len(header))
    for measurement in measurements:
        versus = (
            "n/a"
            if measurement.versus_reference is None
            else f"{measurement.versus_reference:.2f}x"
        )
        LOGGER.info(
            "%-10s%7d%6d%7d%8s%11.1f%10.3f%12.0f%9s%s",
            measurement.engine,
            measurement.batch,
            measurement.workers,
            measurement.turns,
            "network" if measurement.policy_in_loop else "none",
            measurement.seconds_per_season,
            measurement.seconds_per_game,
            measurement.seasons_per_hour,
            versus,
            " (extrapolated)" if measurement.extrapolated else "",
        )
    references = [row for row in measurements if row.engine == "reference"]
    LOGGER.info("")
    LOGGER.info(
        "s/season is the wall clock for one configuration to play `batch` "
        "seasons of %d turns; s/game divides that by `batch`; `proc` is how "
        "many processes it used.",
        SEASON_TURNS,
    )
    if references:
        LOGGER.info(
            "vs ref above 1.0 means the simulator beats learn.rollout."
            "rollout_many per game, measured over %d episodes across %d "
            "processes on %s.",
            references[0].batch,
            references[0].workers,
            references[0].device,
        )
    else:
        LOGGER.info(
            "the reference engine was not measured in this run, so no row "
            "answers whether the simulator is faster than what we have."
        )


if __name__ == "__main__":
    main()
