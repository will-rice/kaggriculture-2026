"""CUDA-residency and graph-capture tests for the simulator hot path."""
# ruff: noqa: D103

from dataclasses import fields

import pytest
import torch

from kaggriculture.learn.encoding import MARKET_SLOTS, MAX_UNITS, UNIT_OPS
from kaggriculture.sim.config import Config
from kaggriculture.sim.decode import decode_market_buckets
from kaggriculture.sim.engine import reset, step
from kaggriculture.sim.legality import legal
from kaggriculture.sim.observe import observe
from kaggriculture.sim.rng import select_day_words
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


def _iteration(state: SimState, units: torch.Tensor, buckets: torch.Tensor) -> SimState:
    for seat in range(2):
        observe(state, seat)
        legal(state, seat)
    return step(state, units, decode_market_buckets(buckets))


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
