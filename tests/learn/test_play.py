"""Tests for the wrapper that plays a checkpoint against the league."""

import copy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping

import pytest
import torch
from kaggle_environments import make
from kaggle_environments.agent import get_last_callable
from kaggle_environments.envs.kaggriculture import kaggriculture as engine

from kaggriculture.constants import (
    BOARD_SIZE,
    ENVIRONMENT,
    EPISODE_STEPS,
    SHED_CAPACITY,
    TURNS_PER_DAY,
)
from kaggriculture.learn import play as play_module
from kaggriculture.learn.encoding import (
    MARKET_SLOTS,
    MAX_ORDERS,
    QUANTITIES,
    UNIT_OPS,
    decode_units,
    encode_board,
    encode_positions,
    encode_scalars,
    unit_count,
)
from kaggriculture.learn.model import Policy
from kaggriculture.learn.play import agent

SOURCE = Path(play_module.__file__)
VERBS = {name.split(":")[0] for name in UNIT_OPS}
MARKET_VERBS = {verb for verb, _ in MARKET_SLOTS} | {"HIRE", "BUY_LAND"}

# The episode the acceptance test below plays. Seeded so the season, and
# therefore the set of ops the agent is asked to choose between, is the same one
# every run.
EPISODE_SEED = 42


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
def checkpoint_without_a_quantity_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Policy]:
    """Point the wrapper at a checkpoint that predates the quantity head only.

    Not the checkpoint actually shipped on disk: ``policy.pt`` predates the
    quantity-lane market widening too, so its ``trade_head`` is a different
    shape and it does not load at all -- see ``load_policy_weights``. This is
    the narrower gap that function is actually built to cross: a checkpoint
    whose shapes otherwise agree with the current ``Policy`` and is missing
    only the head this task added.
    """
    torch.manual_seed(0)
    trained = Policy()
    state = {
        name: tensor
        for name, tensor in trained.state_dict().items()
        if not name.startswith("quantity_head.")
    }
    path = tmp_path / "policy.pt"
    torch.save(state, path)
    monkeypatch.setattr(play_module, "CHECKPOINT", path)
    play_module.model.cache_clear()
    yield trained
    play_module.model.cache_clear()


def test_model_loads_a_checkpoint_that_predates_the_quantity_head(
    checkpoint_without_a_quantity_head: Policy,
) -> None:
    """A checkpoint missing only the quantity head must not fail to load.

    Loading it must not fail or leave the trunk randomly initialised.
    """
    loaded = play_module.model()

    assert torch.equal(
        loaded.stem.weight, checkpoint_without_a_quantity_head.stem.weight
    )


def test_model_refuses_a_checkpoint_missing_the_value_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint missing the value head is refused, not silently loaded.

    Old checkpoints -- the ones the value head used to predate -- do not
    reach this function at all any more: their ``trade_head`` shape mismatch
    raises straight out of ``load_state_dict``. So a checkpoint that reaches
    here missing ``value.*`` is not "merely old", it is corrupt or foreign,
    and ``strict=False`` alone would let it load with the value head randomly
    initialised while every other check kept passing.
    """
    state = {
        name: tensor
        for name, tensor in Policy().state_dict().items()
        if not name.startswith("value.")
    }
    path = tmp_path / "policy.pt"
    torch.save(state, path)
    monkeypatch.setattr(play_module, "CHECKPOINT", path)
    play_module.model.cache_clear()

    with pytest.raises(ValueError, match="value.weight"):
        play_module.model()

    play_module.model.cache_clear()


def test_model_refuses_a_checkpoint_missing_a_trunk_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint missing a trunk key must be refused, not silently loaded.

    ``strict=False`` alone would let this through with the trunk half-random
    and every other check still passing -- this proves the extra key check
    actually discriminates a genuinely broken checkpoint from one that merely
    predates the quantity head.
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


def _engine_acts_on(
    observation: Mapping[str, Any], seat: int, unit: int, op: list[Any]
) -> bool:
    """Return whether ``_apply_unit_action`` moves any state for one unit's op.

    The engine reports nothing when it refuses: an illegal op is a silent
    ``return`` out of a chain of guards, and the unit has spent its turn. So
    legality is read off the state instead -- copy the farm and the private
    mapping, apply the op, and ask whether anything moved. This is the same
    probe ``test_mask.py`` uses to hold the mask against the engine; it is
    repeated here because what is under test is different. There, it asks
    whether ``unit_mask`` was right about an op. Here, it asks whether the
    shipped agent ever *chooses* one the engine will drop.

    ``PASS`` is the one op for which "nothing moved" means accepted rather than
    refused, and it is the one op the engine always permits, so it is answered
    directly.

    Args:
        observation: The turn the op was chosen from.
        seat: Whose farm the unit belongs to.
        unit: Index into ``[farmer, *hands]``.
        op: The decoded op, as the action dict carries it.

    Returns:
        True if the engine acted on the op.
    """
    if op[0] == "PASS":
        return True
    farm = copy.deepcopy(observation["farms"][seat])
    private = copy.deepcopy(observation["private"])
    before = (copy.deepcopy(farm), copy.deepcopy(private))
    engine._apply_unit_action(
        farm,
        private,
        unit,
        op,
        BOARD_SIZE,
        observation["day"],
        TURNS_PER_DAY,
        SHED_CAPACITY,
    )
    return (farm, private) != before


def _market_state_after(
    observation: Mapping[str, Any], seat: int, orders: list[list[Any]]
) -> tuple[Any, ...]:
    """Return the state ``_process_market`` reaches from ``observation`` on ``orders``.

    Run through ``_process_market`` itself rather than through ``_commit_unit``
    in a loop, so the per-unit lockstep, the re-quoting of a moving price and
    the abort-on-refusal are the engine's own. The opponent is given a fresh
    private mapping and no orders: their real one is hidden from this seat, and
    an idle opponent is the only opponent a single-seat probe can honestly
    model.

    Args:
        observation: The turn the orders were chosen from.
        seat: Whose orders these are.
        orders: The order list to process, or a prefix of one.

    Returns:
        The farm, the private mapping and the market after processing --
        everything a market order can move, for comparing one prefix against
        the next.
    """
    farms = copy.deepcopy(observation["farms"])
    market = copy.deepcopy(observation["market"])
    privates = [engine._new_private(), engine._new_private()]
    privates[seat] = copy.deepcopy(observation["private"])
    state = [
        SimpleNamespace(
            observation=SimpleNamespace(
                farms=farms, market=market, private=privates[player]
            ),
            action={"market": list(orders) if player == seat else []},
        )
        for player in (0, 1)
    ]
    engine._process_market(state, SimpleNamespace(configuration={}))
    return (farms[seat], privates[seat], market)


def test_the_agent_refuses_an_op_the_unmasked_argmax_would_have_thrown_away(
    checkpoint: Path,
) -> None:
    """The bug, on the board the engine deals at the start of every game.

    This path used to select by ``logits.argmax(dim=-1)`` over raw logits. On
    the opening observation the seeded fixture policy's highest-scoring op for
    the farmer is ``PLANT:STRAWBERRY`` -- and the farm opens holding no
    strawberry seed, so ``_apply_unit_action`` falls out of its ``seeds`` guard
    without moving anything and the farmer has spent turn zero standing still.
    Nothing raises and nothing logs; the only trace is a season that banks less
    than the gate said it would.

    Both halves are asserted. That the unmasked argmax really does pick an op
    the engine discards is what makes this test discriminate: without it, a
    masked path that happened to agree with the raw one would pass and prove
    nothing. That the masked path picks an op the engine acts on is the fix.

    The unmasked selection is reproduced by handing ``decode_units`` an all-True
    mask, which is exactly the arithmetic the old signature performed, rather
    than by keeping a second copy of the decoder alive to be tested against.
    """
    state = observation([])
    seat = 0
    with torch.no_grad():
        unit_logits, _quantity_logits, _market_logits, _value = play_module.model()(
            encode_board(state, seat),
            encode_scalars(state, seat),
            encode_positions(state, seat),
        )
    units = unit_count(state, seat)

    unmasked = decode_units(
        unit_logits, units, torch.ones_like(unit_logits, dtype=torch.bool)
    )
    played = agent(state)

    assert unmasked["farmer"] == ["PLANT", "STRAWBERRY"]
    assert not _engine_acts_on(state, seat, 0, unmasked["farmer"])
    assert played["farmer"] != unmasked["farmer"]
    assert _engine_acts_on(state, seat, 0, played["farmer"])


@pytest.mark.slow
def test_every_op_a_full_episode_emits_is_one_the_engine_acts_on() -> None:
    """719 turns of the shipped checkpoint, every op put back through the engine.

    The unit heads' claim is absolute and is asserted that way: across a whole
    season, not one op the agent chooses may be silently discarded. A wasted op
    is a wasted turn for that unit, it costs nothing visible, and before the
    masks were threaded through this path 371 of 2,267 ops in this same episode
    were dropped.

    The market's claim is weaker, and the difference is the point rather than
    an exception being made for it. ``unit_mask`` gates each unit against a
    tile and an inventory nobody else is spending, so a per-slot mask makes the
    whole action legal. ``market_mask`` gates each slot against the farm's
    *whole* balance and the slots then spend from that one balance together, so
    orders that are each individually affordable can still overdraw -- which
    ``mask.py`` documents as a limit no per-slot mask can express. So rather
    than tolerate a count, every dropped order is required to be one the engine
    *would* have filled had it been the turn's only order. That pins the loss
    to the shared balance and to nothing else: a genuine mask error, an order
    for an item the shed does not hold or a quantity the market cannot fill,
    would fail in isolation too and fail this test.

    Non-vacuity is asserted alongside. A wrapper that emitted ``PASS`` for
    every unit and no orders at all would satisfy every claim above while
    playing no game, and that is precisely the degenerate agent masking could
    collapse into if a mask were ever inverted.
    """
    ops = 0
    orders = 0
    isolated_failures: list[tuple[int, list[Any]]] = []
    discarded_ops: list[tuple[int, int, list[Any]]] = []
    turn = 0

    def watched(raw_obs: Mapping[str, Any]) -> dict[str, Any]:
        """Play one turn and put everything it emits back through the engine."""
        nonlocal ops, orders, turn
        action = agent(raw_obs)
        seat = int(raw_obs["player"])
        turn += 1
        for unit, op in enumerate([action["farmer"], *action["hands"]]):
            ops += 1
            if not _engine_acts_on(raw_obs, seat, unit, op):
                discarded_ops.append((turn, unit, op))
        # Each order is isolated by replaying the turn's list one order longer
        # and asking whether the extra one moved anything. Comparing whole
        # states rather than counting per (verb, item) is what keeps a SELL and
        # a BUY_PRODUCT of the same item from netting to zero and reading as
        # two dropped orders when both were filled.
        previous = _market_state_after(raw_obs, seat, [])
        for index, order in enumerate(action["market"]):
            orders += 1
            current = _market_state_after(raw_obs, seat, action["market"][: index + 1])
            if current == previous:
                alone = _market_state_after(raw_obs, seat, [order])
                if alone == _market_state_after(raw_obs, seat, []):
                    isolated_failures.append((turn, order))
            previous = current
        return action

    environment = make(
        ENVIRONMENT,
        configuration={"episodeSteps": EPISODE_STEPS, "seed": EPISODE_SEED},
    )
    environment.run([watched, "starter"])

    assert turn == EPISODE_STEPS - 1
    assert discarded_ops == []
    assert isolated_failures == []
    # Non-vacuity, expressed as a floor rather than a recorded count. The farm
    # starts with a farmer and no hands, so an agent that never hired emits
    # exactly one op per turn and `ops == turn`; the degenerate all-PASS agent
    # this guards against sits precisely there. Requiring double that is a crew
    # that demonstrably grew.
    #
    # It used to read `> 3000`, which passed on kaggle-environments 1.32.3 and
    # failed on 1.32.6 at 2,344 ops -- the clone hires a smaller crew now that
    # the town's demand is an eighth of what it was. That is the economy
    # changing, not the masking breaking, and re-recording the magic number on
    # every engine bump would test nothing. The three assertions above are the
    # ones with content and none of them moved.
    assert ops > 2 * turn, f"only {ops} ops over {turn} turns, so the crew never grew"
    assert orders > 0, "an agent that never trades cannot bank a coin"
