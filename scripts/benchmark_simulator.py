"""Measure end-to-end batched simulator throughput on CUDA."""

import argparse
import json
import time
from typing import Any

import torch

from kaggriculture.learn.encoding import MARKET_SLOTS, MAX_UNITS, UNIT_OPS
from kaggriculture.sim.config import Config
from kaggriculture.sim.decode import decode_market_buckets
from kaggriculture.sim.engine import reset, step
from kaggriculture.sim.legality import legal
from kaggriculture.sim.observe import observe


def _iteration(state: Any, units: torch.Tensor, buckets: torch.Tensor) -> Any:  # noqa: ANN401
    for seat in range(2):
        observe(state, seat)
        legal(state, seat)
    return step(state, units, decode_market_buckets(buckets))


def benchmark(batch: int, iterations: int, *, graph: bool) -> dict[str, object]:
    """Benchmark one batch size, including observations and legality masks."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the simulator throughput gate")
    device = torch.device("cuda")
    seeds = torch.arange(batch, device=device, dtype=torch.int64) + 20260810
    state = reset(Config(), seeds)
    units = torch.zeros((batch, 2, MAX_UNITS), device=device, dtype=torch.int16)
    buckets = torch.zeros(
        (batch, 2, len(MARKET_SLOTS) + 2), device=device, dtype=torch.int64
    )
    units.fill_(UNIT_OPS.index("PASS"))

    for _ in range(5):
        state = _iteration(state, units, buckets)
    torch.cuda.synchronize()

    if graph:
        cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cuda_graph):
            captured = _iteration(state, units, buckets)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            cuda_graph.replay()
        torch.cuda.synchronize()
        _ = captured
    else:
        started = time.perf_counter()
        for _ in range(iterations):
            state = _iteration(state, units, buckets)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    steps_per_second = batch * iterations / elapsed
    return {
        "batch": batch,
        "cuda_graph": graph,
        "iterations": iterations,
        "seconds": elapsed,
        "environment_steps_per_second": steps_per_second,
        "trajectories_per_hour": steps_per_second * 3600 / 720,
    }


def main() -> None:
    """Run the requested benchmark matrix and print JSON lines."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, nargs="+", default=[64, 128, 256, 1024])
    parser.add_argument("--iterations", type=int, default=100)
    # CUDA graph capture is an optional accelerator, not a correctness
    # requirement, and it is the fragile half: a single host sync anywhere in
    # the step aborts capture and takes the whole run with it. Without these
    # flags one broken capture made the eager numbers unobtainable, which is
    # exactly when they are most wanted.
    parser.add_argument("--eager-only", action="store_true", help="skip capture")
    parser.add_argument("--graph-only", action="store_true", help="skip eager")
    args = parser.parse_args()
    modes = [False, True]
    if args.eager_only:
        modes = [False]
    if args.graph_only:
        modes = [True]
    for batch in args.batches:
        for graph in modes:
            print(json.dumps(benchmark(batch, args.iterations, graph=graph)))


if __name__ == "__main__":
    main()
