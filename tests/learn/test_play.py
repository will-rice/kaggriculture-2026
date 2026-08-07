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
from kaggriculture.learn import play as play_module
from kaggriculture.learn.encoding import MARKET_SLOTS, MAX_ORDERS, QUANTITIES, UNIT_OPS
from kaggriculture.learn.model import Policy
from kaggriculture.learn.play import agent

SOURCE = Path(play_module.__file__)
VERBS = {name.split(":")[0] for name in UNIT_OPS}
MARKET_VERBS = {verb for verb, _ in MARKET_SLOTS} | {"HIRE", "BUY_LAND"}


@pytest.fixture
def checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the wrapper at a freshly initialised policy of the real shape.

    Seeded, so the untrained weights are the same ones every run. The market
    tests below read what those weights decode to, and an unseeded init would
    make them pass or fail by luck.
    """
    torch.manual_seed(0)
    path = tmp_path / "policy.pt"
    torch.save(Policy().state_dict(), path)
    monkeypatch.setattr(play_module, "CHECKPOINT", path)
    play_module.model.cache_clear()
    yield path
    play_module.model.cache_clear()


def observation(hands: list[list[int]]) -> Mapping[str, Any]:
    """Return a real opening observation with ``hands`` hired onto seat 0.

    The private inventory list is extended alongside, because the engine's
    ``_do_hire`` appends an inventory as it appends a hand and ``encode_board``
    reads one per unit. Staffing a farm without that is a state the game cannot
    produce, and a fixture in an impossible state is how a test comes to agree
    with a bug.
    """
    environment = make(ENVIRONMENT)
    environment.reset()
    state = environment.state[0].observation
    state["farms"][0]["hands"] = hands
    state["private"]["inventories"] = [{} for _ in range(1 + len(hands))]
    return state


@pytest.fixture
def checkpoint_without_a_value_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Policy]:
    """Point the wrapper at a checkpoint shaped like the real one on disk.

    ``policy.pt`` was behaviour-cloned before the value head existed, so it
    carries every trunk and head key but no ``value.*`` -- this is the exact
    shape ``model()`` has to load in the sandbox.
    """
    torch.manual_seed(0)
    trained = Policy()
    state = {
        name: tensor
        for name, tensor in trained.state_dict().items()
        if not name.startswith("value.")
    }
    path = tmp_path / "policy.pt"
    torch.save(state, path)
    monkeypatch.setattr(play_module, "CHECKPOINT", path)
    play_module.model.cache_clear()
    yield trained
    play_module.model.cache_clear()


def test_model_loads_a_checkpoint_that_predates_the_value_head(
    checkpoint_without_a_value_head: Policy,
) -> None:
    """The shipped checkpoint has no value head.

    Loading it must not fail or leave the trunk randomly initialised.
    """
    loaded = play_module.model()

    assert torch.equal(loaded.stem.weight, checkpoint_without_a_value_head.stem.weight)


def test_model_refuses_a_checkpoint_missing_more_than_the_value_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint missing a trunk key too must be refused, not silently loaded.

    ``strict=False`` alone would let this through with the trunk half-random
    and every other check still passing -- this proves the extra key check
    actually discriminates a genuinely broken checkpoint from one that merely
    predates the value head.
    """
    state = Policy().state_dict()
    del state["stem.weight"]
    path = tmp_path / "policy.pt"
    torch.save(state, path)
    monkeypatch.setattr(play_module, "CHECKPOINT", path)
    play_module.model.cache_clear()

    with pytest.raises(ValueError, match="stem.weight"):
        play_module.model()

    play_module.model.cache_clear()


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


def test_the_agent_emits_market_orders(checkpoint: Path) -> None:
    """An action with an empty market cannot bank a coin; that lost 400 games.

    The engine increases a farm's money in exactly one place -- crediting a
    completed ``SELL`` -- so a wrapper that computes the market logits and
    discards them, which this one used to, banks its opening 3,000 in every
    episode however accurate the unit head is. This is what fails if the second
    head is ever dropped from the returned action.
    """
    orders = agent(observation([[4, 3]]))["market"]

    assert orders
    assert len(orders) <= MAX_ORDERS


def test_every_market_order_is_one_the_engine_will_act_on(checkpoint: Path) -> None:
    """A malformed order is not rejected, it is silently skipped.

    ``_process_market`` parses each order and moves on, so a bad verb or a
    quantity outside the bucket table costs a slot out of the ten the turn is
    allowed and reports nothing.
    """
    orders = agent(observation([[4, 3]]))["market"]

    assert all(order[0] in MARKET_VERBS for order in orders)
    assert all(
        order[2] in QUANTITIES and order[2] > 0 for order in orders if len(order) == 3
    )


def test_importing_the_wrapper_never_loads_wandb() -> None:
    """The play path must not reach the network, directly or through an import.

    This module ships now, and the sandbox has no network: an import that
    reaches out forfeits the episode on turn zero. Checked in a fresh
    interpreter, since the test session has already imported wandb through the
    tracking tests. The checkpoint name is printed alongside to catch the
    import path silently reverting to one under ``/data``, which exists on the
    workstation and nowhere else.
    """
    loaded = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, kaggriculture.learn.play as played;"
            " print('wandb' in sys.modules, played.CHECKPOINT.name)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert loaded.stdout.split() == ["False", "policy.pt"]


def test_the_checkpoint_sits_inside_the_shipped_package() -> None:
    """One path has to serve both the local league and the unpacked archive.

    ``package.py`` copies the checkpoint to the same place relative to the
    package, so a path anywhere else -- ``/data``, the repo root -- would load
    locally and find nothing in the sandbox, and the submitted agent would play
    random weights with no error to say so.
    """
    assert play_module.CHECKPOINT.parent == SOURCE.parent
