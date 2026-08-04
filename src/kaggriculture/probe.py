"""Sandbox probe, logged on the first turn of a submitted episode.

Phase 0 of the strategy: before spending weeks on a policy that has to run
inside the competition sandbox, find out what the sandbox is. A Lux S2
competitor abandoned a trained model because no fast inference runtime could be
installed there, and read the error logs back as 404s.

The networks timed here are the ones we would actually ship, not a toy: the
~20M-parameter residual trunk Toad Brigade won Lux S1 with on a personal PC, and
the ~300M-parameter scale Lux S3's second place trained on an RTX 3090 — the
configuration our workstation strictly beats, and therefore the one worth
knowing about. Training scale and inference budget are separate constraints, and
this measures the second one.

Everything runs once, on turn 0, and comes back with
``kaggle competitions logs <episode> <index>``. The environment allows one
second per turn plus a 60-second overage pool for the whole episode, of which a
normal season spends none, so the probe is bounded to a fraction of that pool.
Each network's shape is logged *before* it is built, so a run that dies on the
largest one still says where it died — which is itself the answer.
"""

import logging
import os
import platform
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Optional

# The action head is board-shaped: one 10x10x22 read per unit-bearing cell.
BOARD_SIZE = 10
INPUT_PLANES = 32
ACTION_PLANES = 22

# (label, residual blocks, channels), smallest first — a run that dies on the
# large net has already logged the small one. Channel counts are chosen to land
# on the parameter counts these solutions reported; the actual count is logged.
TRUNKS = (
    ("toad-brigade-lux-s1-20M", 24, 224),
    ("lux-s3-second-place-300M", 24, 833),
)
PASSES = 20
BUDGET_SECONDS = 30.0


def report() -> None:
    """Log what the sandbox offers, within a bounded slice of the overage pool."""
    logger = probe_logger()
    deadline = time.monotonic() + BUDGET_SECONDS

    logger.info("PROBE python %s", sys.version.replace("\n", " "))
    logger.info("PROBE platform %s", platform.platform())
    logger.info(
        "PROBE cpus %s affinity %s memory_kb %s",
        os.cpu_count(),
        len(os.sched_getaffinity(0)),
        total_memory_kb(),
    )

    numpy = import_timed(logger, "numpy")
    if numpy is not None:
        logger.info("PROBE numpy version %s", numpy.__version__)

    torch = import_timed(logger, "torch")
    if torch is None:
        return
    logger.info(
        "PROBE torch version %s threads %d",
        torch.__version__,
        torch.get_num_threads(),
    )
    # Each trunk's cost predicts the next one's, because the board size is fixed
    # and these are all convolutions: time scales with parameters. A 300M-
    # parameter forward pass could outlast the whole overage pool on a sandbox
    # CPU, and an agent that overruns it forfeits, so the large trunk is built
    # only once the small one says it fits. A prediction that says it does not
    # fit is the Phase 0 answer, arrived at without betting the episode on it.
    measured: Optional[tuple[int, float]] = None
    for label, blocks, channels in TRUNKS:
        parameters = trunk_parameters(blocks, channels)
        remaining_ms = (deadline - time.monotonic()) * 1000
        predicted_ms = predict_ms(measured, parameters)
        if predicted_ms is not None and predicted_ms > remaining_ms:
            logger.info(
                "PROBE trunk %s skipped parameters %d predicted_ms %.0f left_ms %.0f",
                label,
                parameters,
                predicted_ms,
                remaining_ms,
            )
            continue
        median_ms = time_trunk(logger, torch, label, blocks, channels, deadline)
        measured = (parameters, median_ms)


def probe_logger() -> logging.Logger:
    """Return a logger whose output reaches the episode log.

    Nothing configures logging inside the sandbox, so a bare ``info`` call would
    be dropped by the last-resort handler's WARNING threshold and the probe
    would silently report nothing. Its own stderr handler is what
    ``kaggle competitions logs`` hands back.
    """
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def total_memory_kb() -> Optional[int]:
    """Return total system memory, which bounds how large a model can even load."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1])
    return None


def import_timed(logger: logging.Logger, name: str) -> Optional[ModuleType]:
    """Import a module, logging how long it took or that it is unavailable.

    An import that fails is the finding, not an error: whether torch exists in
    this sandbox is the single question Phase 0 exists to answer.
    """
    start = time.perf_counter()
    try:
        module = __import__(name)
    except ImportError as error:
        logger.info("PROBE import %s unavailable %s", name, error)
        return None
    logger.info("PROBE import %s seconds %.2f", name, time.perf_counter() - start)
    return module


def trunk_parameters(blocks: int, channels: int) -> int:
    """Return how many parameters ``time_trunk`` will build, without building them.

    Known before allocation, a 300M-parameter trunk is 1.2 GB of weights that
    the sandbox may not have; the count is what decides whether to try.
    """
    stem = INPUT_PLANES * channels * 9 + channels
    block = 2 * (channels * channels * 9 + channels)
    head = channels * ACTION_PLANES + ACTION_PLANES
    return stem + blocks * block + head


def predict_ms(
    measured: Optional[tuple[int, float]], parameters: int
) -> Optional[float]:
    """Return the forward-pass cost implied by an already-timed trunk."""
    if measured is None:
        return None
    timed_parameters, median_ms = measured
    return median_ms * parameters / timed_parameters


def time_trunk(
    logger: logging.Logger,
    torch: ModuleType,
    label: str,
    blocks: int,
    channels: int,
    deadline: float,
) -> float:
    """Build one candidate trunk and time single-batch forward passes of it.

    Batch one is the shape that matters: the agent plans one board per turn, and
    a throughput number measured at batch 32 would flatter the per-turn budget
    this has to fit inside. The network is built here rather than handed in
    because it exists only to be timed, and squeeze-excitation and the
    downscaling variant are Phase 2's decision — what is measured here is the
    cost of the convolutions, which is where the time and the parameters are.

    Returns:
        Median milliseconds per forward pass.
    """
    logger.info(
        "PROBE trunk %s building blocks %d channels %d", label, blocks, channels
    )
    layers = [torch.nn.Conv2d(INPUT_PLANES, channels, 3, padding=1), torch.nn.ReLU()]
    for _ in range(blocks):
        layers += [
            torch.nn.Conv2d(channels, channels, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels, channels, 3, padding=1),
            torch.nn.ReLU(),
        ]
    layers.append(torch.nn.Conv2d(channels, ACTION_PLANES, 1))
    network = torch.nn.Sequential(*layers)
    network.eval()
    parameters = sum(parameter.numel() for parameter in network.parameters())
    board = torch.zeros(1, INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)

    with torch.no_grad():
        network(board)  # warm up; the first pass pays for lazy kernel selection
        timings = []
        while len(timings) < PASSES and time.monotonic() < deadline:
            start = time.perf_counter()
            network(board)
            timings.append((time.perf_counter() - start) * 1000)

    timings.sort()
    median_ms = timings[len(timings) // 2]
    logger.info(
        "PROBE trunk %s parameters %d passes %d median_ms %.1f max_ms %.1f",
        label,
        parameters,
        len(timings),
        median_ms,
        timings[-1],
    )
    return median_ms
