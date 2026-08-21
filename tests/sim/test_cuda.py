"""CUDA-residency and graph-capture tests for the simulator hot path."""
# ruff: noqa: D103

from dataclasses import fields

import pytest
import torch

from kaggriculture.learn.encoding import MARKET_SLOTS, MAX_UNITS, QUANTITIES, UNIT_OPS
from kaggriculture.sim.config import Config
from kaggriculture.sim.decode import decode_market_buckets
from kaggriculture.sim.engine import reset, step, unit_quantity_ones
from kaggriculture.sim.legality import legal
from kaggriculture.sim.observe import observe
from kaggriculture.sim.rng import select_day_words
from kaggriculture.sim.rollout import collect_segment
from kaggriculture.sim.state import SimState

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device is not visible"
)


def _actions(batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    units = torch.full(
        (batch, 2, MAX_UNITS),
        UNIT_OPS.index("PASS"),
        dtype=torch.int16,
        device=device,
    )
    buckets = torch.zeros(
        (batch, 2, len(MARKET_SLOTS) + 2), dtype=torch.int64, device=device
    )
    return units, buckets


class _ZeroPolicy(torch.nn.Module):
    """Zero logits, so the masked softmax stays uniform over the legal moves."""

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


def _iteration(state: SimState, units: torch.Tensor, buckets: torch.Tensor) -> SimState:
    for seat in range(2):
        observe(state, seat)
        legal(state, seat)
    return step(
        state,
        units,
        decode_market_buckets(buckets),
        unit_quantity_ones(state.batch_size, state.step.device),
    )


def test_cuda_uint32_day_selection_preserves_bits() -> None:
    device = torch.device("cuda")
    words = torch.tensor(
        [[[0, 2**32 - 1]], [[2**31, 17]]], dtype=torch.uint32, device=device
    )

    selected = select_day_words(words, torch.tensor([0, 0], device=device))

    assert selected.cpu().tolist() == [[0, 2**32 - 1], [2**31, 17]]


def test_cuda_step_matches_cpu_through_end_of_day() -> None:
    seeds = torch.tensor([241, 251], dtype=torch.int64)
    cpu = reset(Config(), seeds)
    gpu = reset(Config(), seeds.cuda())
    cpu_units, cpu_buckets = _actions(2, torch.device("cpu"))
    gpu_units, gpu_buckets = _actions(2, torch.device("cuda"))

    for _ in range(24):
        cpu = _iteration(cpu, cpu_units, cpu_buckets)
        gpu = _iteration(gpu, gpu_units, gpu_buckets)

    torch.cuda.synchronize()
    for field in fields(SimState):
        assert torch.equal(getattr(cpu, field.name), getattr(gpu, field.name).cpu()), (
            field.name
        )


def test_cuda_hot_path_is_graph_capturable() -> None:
    device = torch.device("cuda")
    state = reset(Config(), torch.tensor([257, 263], device=device))
    units, buckets = _actions(2, device)
    for _ in range(3):
        state = _iteration(state, units, buckets)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _iteration(state, units, buckets)
    graph.replay()
    torch.cuda.synchronize()

    assert captured.step.is_cuda


def test_cuda_collect_segment_is_graph_capturable_and_replays_the_season() -> None:
    """The collection path itself must capture, not just the bare hot path.

    Capture is where the simulator's speedup lives -- the loop is launch-bound,
    so a replayed segment costs a fraction of an issued one -- and a single host
    synchronisation anywhere inside ``collect_segment`` aborts it. This test is
    what fails if one comes back: reading ``Trajectory.illegal`` on the host, or
    rebuilding a price table with ``torch.tensor(..., device=cuda)`` inside the
    loop, both of which it used to do.

    The successor state is copied back over the input buffers inside the capture
    because ``collect_segment`` is functional. Without that, every replay would
    re-collect the season's first segment from unchanged inputs and the clock
    would never leave the first segment -- which is how a previous benchmark
    came to quote the cost of a computation that never advanced.
    """
    device = torch.device("cuda")
    length = 2
    replays = 3
    state = reset(Config(), torch.tensor([269, 271], device=device))
    policy = _ZeroPolicy().to(device)
    generator = torch.Generator(device=device).manual_seed(277)
    for _ in range(2):
        state = collect_segment(state, policy, turns=length, generator=generator)[0]
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    graph.register_generator_state(generator)
    with torch.cuda.graph(graph):
        successor, trajectory = collect_segment(
            state, policy, turns=length, generator=generator
        )
        for field in fields(SimState):
            getattr(state, field.name).copy_(getattr(successor, field.name))
    started = int(state.step[0])
    for _ in range(replays):
        graph.replay()
    torch.cuda.synchronize()

    assert int(state.step[0]) == started + replays * length
    assert trajectory.illegal.is_cuda
    assert int(trajectory.illegal) == 0
