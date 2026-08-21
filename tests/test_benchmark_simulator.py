"""Tests for the simulator throughput benchmark.

The benchmark's own graph self-check needs a GPU and runs on every invocation,
so what is worth pinning here is the part a CPU can see: that the body being
timed genuinely advances the season with actions that vary, and that the
reference it is compared against is the configuration the gate actually runs.
Those are the failures the previous version of the script shipped -- a captured
step replayed against fixed buffers, and a loop of ``PASS`` actions with empty
market orders -- and neither of them raised anything at the time.
"""

from dataclasses import fields
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType

import torch

from kaggriculture.constants import EPISODE_STEPS
from kaggriculture.learn.encoding import MARKET_SLOTS, MAX_UNITS, QUANTITIES, UNIT_OPS
from kaggriculture.learn.scripts import gate
from kaggriculture.sim.config import Config
from kaggriculture.sim.engine import reset

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_simulator.py"
SEEDS = torch.arange(4, dtype=torch.int64)
TURNS = 24


def load_script() -> ModuleType:
    """Import the benchmark by path, since it lives outside the package."""
    module = ModuleType("benchmark_simulator")
    SourceFileLoader("benchmark_simulator", str(SCRIPT)).exec_module(module)
    return module


def test_the_timed_turn_advances_the_season() -> None:
    """The body the benchmark times must move the clock, once per turn.

    The previous script's graph mode replayed a step captured against fixed
    buffers, so its clock never left turn zero however many iterations it timed.
    """
    module = load_script()
    state = reset(Config(), SEEDS)
    for expected in range(TURNS):
        assert int(state.step[0]) == expected
        state = module.turn(state, module.ZeroPolicy(), sample=True)
    assert int(state.step[0]) == TURNS


def test_the_timed_turn_plays_actions_that_move_the_farms() -> None:
    """Units must actually move, so the timed work is not a season of ``PASS``.

    The previous script filled every unit slot with ``PASS`` and left every
    market order empty, which skips most of the unit and market phases. Timing
    that prices a turn nobody plays.

    Every turn is checked rather than only the last, because the day's end walks
    every unit back to the farmhouse: a segment a whole day long would finish
    where it started even though the farm spent all of it moving.
    """
    module = load_script()
    state = reset(Config(), SEEDS)
    start_x, start_y = state.unit_x.clone(), state.unit_y.clone()
    moved = False
    for _ in range(TURNS):
        state = module.turn(state, module.ZeroPolicy(), sample=True)
        moved = moved or not torch.equal(state.unit_x, start_x)
        moved = moved or not torch.equal(state.unit_y, start_y)
    assert moved, "no unit changed tile in a whole segment of sampled turns"


def test_the_timed_turn_leaves_its_input_state_alone() -> None:
    """``step`` is functional, which is exactly why the graph needs a copy-back.

    Had a turn mutated its input, the previous graph mode would have advanced by
    accident and the defect would never have shipped. It does not, so
    ``_capture`` writes the successor back over the input buffers before the
    capture closes, and this test is what says that requirement is still real.
    """
    module = load_script()
    state = reset(Config(), SEEDS)
    before = {field.name: getattr(state, field.name).clone() for field in fields(state)}
    module.turn(state, module.ZeroPolicy(), sample=True)
    mutated = [
        name
        for name, value in before.items()
        if not torch.equal(value, getattr(state, name))
    ]
    assert not mutated


def test_zero_policy_removes_the_network_and_nothing_else() -> None:
    """The no-policy rows must still sample from a live distribution.

    Zero logits leave the masked softmax uniform over the legal moves, so the
    difference between a ``network`` row and a ``none`` row is the trunk's
    arithmetic rather than the whole decision.
    """
    module = load_script()
    rows = 3
    units, quantities, market, value = module.ZeroPolicy()(
        torch.zeros(rows, 1, 10, 10), torch.zeros(rows, 1), torch.zeros(rows, MAX_UNITS)
    )
    assert units.shape == (rows, MAX_UNITS, len(UNIT_OPS))
    assert quantities.shape == (rows, MAX_UNITS, len(QUANTITIES))
    assert market.shape == (rows, len(MARKET_SLOTS) + 2, len(QUANTITIES))
    assert value.shape == (rows, 1)
    assert not units.any()
    assert not quantities.any()
    assert not market.any()


def test_the_season_length_is_the_game_s_own() -> None:
    """A benchmark whose season drifts from the game's prices the wrong thing."""
    assert load_script().SEASON_TURNS == EPISODE_STEPS - 1


def test_the_reference_is_measured_in_the_gate_s_shape() -> None:
    """The comparison has to be against the configuration the gate runs.

    Timing the reference in one process would flatter the simulator by roughly
    the worker count. The constants are imported from ``gate`` rather than
    restated so the comparison follows it, and this test is what fails if
    someone pins them here instead.
    """
    module = load_script()
    assert module.REFERENCE_GAMES == gate.GAMES
    assert module.REFERENCE_WORKERS == gate.WORKERS
