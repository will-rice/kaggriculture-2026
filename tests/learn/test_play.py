"""Tests for the wrapper that plays a checkpoint against the league."""

import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping

import pytest
import torch
from kaggle_environments import make
from kaggle_environments.agent import get_last_callable

from kaggriculture.constants import ENVIRONMENT
from kaggriculture.learn.encoding import UNIT_OPS
from kaggriculture.learn.model import Policy
from kaggriculture.learn.scripts import play as play_module
from kaggriculture.learn.scripts.play import agent

SOURCE = Path(play_module.__file__)
VERBS = {name.split(":")[0] for name in UNIT_OPS}


@pytest.fixture
def checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the wrapper at a freshly initialised policy of the real shape."""
    path = tmp_path / "policy.pt"
    torch.save(Policy().state_dict(), path)
    monkeypatch.setattr(play_module, "CHECKPOINT", path)
    play_module.model.cache_clear()
    yield path
    play_module.model.cache_clear()


def observation(hands: list[list[int]]) -> Mapping[str, Any]:
    """Return a real opening observation with ``hands`` hired onto seat 0."""
    environment = make(ENVIRONMENT)
    environment.reset()
    state = environment.state[0].observation
    state["farms"][0]["hands"] = hands
    return state


def test_the_agent_is_the_last_callable_in_the_file() -> None:
    """The environment execs an agent path and plays whatever callable ends it.

    A helper appended below ``agent`` would be played instead, and the episode
    would fail on turn zero with a signature error rather than anything that
    names the cause.
    """
    loaded = get_last_callable(SOURCE.read_text(), path=str(SOURCE))

    assert getattr(loaded, "__name__", None) == "agent"


def test_the_agent_returns_one_op_per_unit_on_the_farm(checkpoint: Path) -> None:
    """The action names the farmer and every hand, and nothing beyond them.

    The count comes from the observation, which is what the engine walks. An
    action shorter than the crew leaves hands idle; one longer attaches ops to
    units that do not exist.
    """
    action = agent(observation([[4, 3], [5, 4]]))

    assert set(action) == {"farmer", "hands", "market"}
    assert len(action["hands"]) == 2
    assert action["farmer"][0] in VERBS
    assert all(op[0] in VERBS for op in action["hands"])


def test_the_action_grows_with_the_crew(checkpoint: Path) -> None:
    """Hiring a hand must add an op, not leave the new hand standing there.

    Guards the unit count against being read from anywhere but the
    observation -- a constant, or the farmer alone, would pass every other test
    in this file.
    """
    lone = agent(observation([]))
    crew = agent(observation([[4, 3], [5, 4], [3, 4]]))

    assert lone["hands"] == []
    assert len(crew["hands"]) == 3


def test_the_agent_carries_no_market_orders(checkpoint: Path) -> None:
    """A stated limitation, guarded so it stays stated rather than assumed.

    ``UNIT_OPS`` has no market verbs, so the cloned agent farms and never
    trades. When the RL phase adds a market head this test fails, which is the
    moment to update what the wrapper claims about itself.
    """
    assert agent(observation([[4, 3]]))["market"] == []


def test_importing_the_wrapper_never_loads_wandb() -> None:
    """The play path must not reach the network, directly or through an import.

    The wrapper is not shipped -- ``package.py`` excludes ``learn`` -- but it is
    the shape a submitted checkpoint would take, and the sandbox has no network:
    an import that reaches out forfeits the episode on turn zero. Checked in a
    fresh interpreter, since the test session has already imported wandb
    through the tracking tests.
    """
    loaded = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, kaggriculture.learn.scripts.play as played;"
            " print('wandb' in sys.modules, played.CHECKPOINT.name)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert loaded.stdout.split() == ["False", "policy.pt"]
