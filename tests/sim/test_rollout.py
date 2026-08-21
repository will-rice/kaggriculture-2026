"""Tensor rollout primitive tests."""
# ruff: noqa: ANN001, ANN202, D103

from dataclasses import replace

import pytest
import torch
from kaggle_environments import make

from kaggriculture import economic_policy
from kaggriculture.learn.encoding import MARKET_SLOTS, QUANTITIES, UNIT_OPS
from kaggriculture.learn.progress import potential as reference_potential
from kaggriculture.sim.config import Config
from kaggriculture.sim.engine import reset, step, unit_quantity_ones
from kaggriculture.sim.rollout import collect_segment, potential, scripted_actions
from kaggriculture.sim.state import PRODUCT_NAMES, SHED_NAMES, pack
from tests.sim.conftest import assert_identical


def test_tensor_potential_matches_reference_components_for_both_seats() -> None:
    environment = make("kaggriculture", configuration={"seed": 193}, debug=True)
    environment.reset(2)
    observation = environment.state[0].observation
    observation.farms[0].tiles[1][1] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 0,
        "watered_today": True,
        "consecutive_unwatered": 0,
        "yield_units": 3,
        "max_lifespan_step": 100,
        "fertilized_until_day": -1,
    }
    observation.farms[1].tiles[2][2] = {
        "kind": "PASTURE",
        "animal": "COW",
        "placed_day": 0,
        "yield_units": 2,
        "consecutive_unfed": 0,
        "fed_today": True,
        "cared_today": False,
        "fertilizer_available": True,
        "pending_care_bonus": 0,
    }
    observation.farms[0].unlocked_quadrants = ["NW", "NE", "SW"]
    private0 = environment.state[0].observation.private
    private0.seeds["MELON"] = 2
    private0.shed["MILK"] = 4
    private0.inventories = [{"WHEAT": 2}]
    private1 = environment.state[1].observation.private
    private1.shed["EGG"] = 3
    environment.state[1].observation.farms = observation.farms
    state = pack([environment])

    for seat in range(2):
        actual = potential(state, seat)
        expected = torch.tensor(
            reference_potential(environment.state[seat].observation),
            dtype=torch.float32,
        )
        assert torch.equal(actual[0], expected)


class _UniformPolicy(torch.nn.Module):
    def forward(self, board, scalars, positions):
        batch = len(board)
        device = board.device
        return (
            torch.zeros(batch, 20, len(UNIT_OPS), device=device),
            torch.zeros(batch, len(MARKET_SLOTS) + 2, len(QUANTITIES), device=device),
            torch.zeros(batch, 1, device=device),
        )


class _SellWheatPolicy(_UniformPolicy):
    def forward(self, board, scalars, positions):
        units, markets, values = super().forward(board, scalars, positions)
        markets[:, 0, 1] = 100
        return units, markets, values


def test_collect_segment_keeps_actions_legal_and_tensors_on_device() -> None:
    state = reset(Config(), torch.tensor([197, 199]))

    next_state, trajectory = collect_segment(
        state,
        _UniformPolicy(),
        turns=2,
        generator=torch.Generator().manual_seed(211),
    )

    assert next_state.step.tolist() == [2, 2]
    assert trajectory.board.shape[:3] == (2, 2, 2)
    assert trajectory.unit_actions.shape == (2, 2, 2, 20)
    assert int(trajectory.illegal) == 0
    assert trajectory.illegal.device == state.step.device
    assert trajectory.board.device == state.step.device


def test_scripted_economic_policy_bridge_matches_reference_action() -> None:
    """The bridge tracks the reference turn by turn, well past its old limit.

    Seed 223 is the exact case that used to force this test to stop after a
    turn or two: on turn 14, ``economic_policy.agent`` issues a bulk PICKUP
    (``["PICKUP", "WHEAT", 2]``, in the second hand) once its worker
    inventory is non-empty, which the encoder used to refuse outright. The
    quantity lane replaces that refusal with faithful encoding, so the
    segment now runs straight through turn 14 and stays bit-identical to the
    reference on every turn, including that one.
    """
    environment = make("kaggriculture", configuration={"seed": 223}, debug=True)
    environment.reset(2)
    state = pack([environment])
    saw_the_bulk_pickup = False

    for _turn in range(15):
        opponent_units, opponent_quantities, markets = scripted_actions(
            state, economic_policy.agent
        )
        units = torch.full((1, 2, 20), UNIT_OPS.index("PASS"), dtype=torch.int16)
        units[:, 1].copy_(opponent_units)
        quantities = unit_quantity_ones(1)
        quantities[:, 1].copy_(opponent_quantities)
        reference_action = economic_policy.agent(environment.state[1].observation)

        if reference_action.get("hands") and reference_action["hands"][1] == [
            "PICKUP",
            "WHEAT",
            2,
        ]:
            saw_the_bulk_pickup = True
            # The second hand is unit slot 2 (farmer is 0, first hand is 1).
            assert int(quantities[0, 1, 2]) == 2

        environment.step(
            [
                {"farmer": ["PASS"], "hands": [], "market": []},
                reference_action,
            ]
        )
        state = step(state, units, markets, quantities)

        assert_identical(environment, state, 0)

    assert saw_the_bulk_pickup, "seed 223 stopped issuing the bulk PICKUP on turn 14"


def test_scripted_actions_encodes_a_bulk_pickup_from_the_opponent() -> None:
    """The scripted-opponent bridge carries a bulk quantity instead of refusing it.

    Before the quantity lane, a bulk PICKUP or PLACE (any quantity other than
    one) was outside the encoder's domain and ``encode_turn`` raised.
    ``economic_policy.agent`` issues exactly that once its shed and worker
    inventories are non-empty -- reproduced here directly, without needing to
    play the opponent out to the turn where it first does so.
    """
    state = reset(Config(), torch.tensor([223]))

    def bulk_pickup(observation: object) -> dict[str, object]:
        return {"farmer": ["PICKUP", "WHEAT", 2], "hands": [], "market": []}

    units, quantities, _markets = scripted_actions(state, bulk_pickup)

    assert int(units[0, 0]) == UNIT_OPS.index("PICKUP:WHEAT")
    assert int(quantities[0, 0]) == 2


def test_collect_segment_accepts_a_scripted_opponent() -> None:
    state = reset(Config(), torch.tensor([227]))

    next_state, trajectory = collect_segment(
        state,
        _UniformPolicy(),
        turns=2,
        generator=torch.Generator().manual_seed(229),
        opponent=economic_policy.agent,
    )

    assert next_state.step.tolist() == [2]
    assert int(trajectory.illegal) == 0


def test_terminal_segment_rewards_telescope_to_the_terminal_margin() -> None:
    state = reset(Config(), torch.tensor([233]))
    step_number = torch.tensor([718], dtype=torch.int32)
    shed = state.shed.clone()
    shed[0, 0, SHED_NAMES.index("WHEAT")] = 1
    state = replace(
        state,
        step=step_number,
        day=torch.tensor([29], dtype=torch.int32),
        hour=torch.tensor([22], dtype=torch.int32),
        shed=shed,
    )
    initial_margin = state.money[:, 0] - state.money[:, 1]

    final_state, trajectory = collect_segment(
        state,
        _SellWheatPolicy(),
        turns=1,
        generator=torch.Generator().manual_seed(239),
    )
    final_margin = final_state.money[:, 0] - final_state.money[:, 1]

    assert final_state.done.tolist() == [True]
    assert torch.equal(
        trajectory.rewards[:, :, 0].sum(0), (final_margin - initial_margin).float()
    )


def test_encode_turn_maps_the_action_grammar_to_simulator_codes() -> None:
    """One reader of the grammar, used by both the scripted bridge and routes."""
    from kaggriculture.sim.rollout import encode_turn

    encoded = encode_turn(
        {
            "farmer": ["PLANT", "WHEAT"],
            "hands": [["WATER"], ["PASS"]],
            "market": [["SELL", "WHEAT", 3], ["HIRE"]],
        }
    )

    assert encoded.units[0] == UNIT_OPS.index("PLANT:WHEAT")
    assert encoded.units[1] == UNIT_OPS.index("WATER")
    # SELL is order type 1 and WHEAT is index 0 of PRODUCT_NAMES.
    assert encoded.orders[0] == (1, PRODUCT_NAMES.index("WHEAT"), 3)
    # HIRE carries no item, so the item slot stays at the unused sentinel.
    assert encoded.orders[1] == (5, -1, 1)
    # Unused slots are the "no order" code.
    assert encoded.orders[2] == (0, -1, 0)


def test_unit_index_encodes_a_pickup_quantity_outside_the_old_domain() -> None:
    """The encoder carries a bulk PICKUP's quantity instead of refusing it.

    Silently collapsing ``["PICKUP", "FERTILIZER", 3]`` to a transfer of one
    is what let a recorded route replay through the simulator and diverge
    from its own recorded result without any error -- refusing it outright
    was the guard's fix, and faithful encoding replaces the refusal now that
    execution (task 1) applies the engine's own clamp.
    """
    from kaggriculture.sim.rollout import encode_turn

    encoded = encode_turn(
        {"farmer": ["PICKUP", "FERTILIZER", 3], "hands": [], "market": []}
    )

    assert encoded.units[0] == UNIT_OPS.index("PICKUP:FERTILIZER")
    assert encoded.quantities[0] == 3


def test_unit_index_encodes_a_place_quantity_outside_the_old_domain() -> None:
    """The same faithful encoding applies to PLACE, the other transfer verb."""
    from kaggriculture.sim.rollout import encode_turn

    encoded = encode_turn(
        {"farmer": ["PASS"], "hands": [["PLACE", "WHEAT", 2]], "market": []}
    )

    assert encoded.units[1] == UNIT_OPS.index("PLACE:WHEAT")
    assert encoded.quantities[1] == 2


def test_encode_order_raises_on_a_quantity_over_64() -> None:
    """The reference has no cap; clamping to 64 would silently alter the order.

    The RL action space never emits a quantity this large, so this never
    fires for it -- only for an order outside the simulator's domain.
    """
    from kaggriculture.sim.rollout import encode_turn

    with pytest.raises(ValueError, match="SELL.*WHEAT.*100"):
        encode_turn(
            {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", 100]]}
        )


def test_encode_order_raises_on_a_numeric_string_quantity() -> None:
    """The reference coerces a quantity via ``int()`` and executes it.

    Encoding it as the "aborted" code 7 instead would silently diverge from
    what the reference actually replays.
    """
    from kaggriculture.sim.rollout import encode_turn

    with pytest.raises(ValueError, match="SELL.*WHEAT.*3"):
        encode_turn(
            {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", "3"]]}
        )


def test_unit_index_accepts_the_in_domain_quantity_of_one() -> None:
    """Quantity 1 was the whole old domain, and it still works."""
    from kaggriculture.sim.rollout import encode_turn

    encoded = encode_turn(
        {"farmer": ["PICKUP", "FERTILIZER", 1], "hands": [], "market": []}
    )

    assert encoded.units[0] == UNIT_OPS.index("PICKUP:FERTILIZER")
    assert encoded.quantities[0] == 1


def test_unit_index_defaults_the_quantity_to_one_off_a_transfer_verb() -> None:
    """A slot that never named a transfer carries the "no bulk" quantity."""
    from kaggriculture.sim.rollout import encode_turn

    encoded = encode_turn(
        {"farmer": ["CARE"], "hands": [["FEED"], ["WATER"]], "market": []}
    )

    assert encoded.units[0] == UNIT_OPS.index("CARE")
    assert encoded.quantities[0] == 1
    assert encoded.quantities[1] == 1
    assert encoded.quantities[2] == 1


def test_unit_index_encodes_a_negative_or_zero_pickup_quantity_faithfully() -> None:
    """The engine's own clamp turns a non-positive quantity into a no-op.

    The encoder does not pre-empt that: it carries the quantity exactly as
    named, sign included, and leaves the clamp to execution.
    """
    from kaggriculture.sim.rollout import encode_turn

    encoded = encode_turn({"farmer": ["PICKUP", "WHEAT", 0], "hands": [], "market": []})
    assert encoded.units[0] == UNIT_OPS.index("PICKUP:WHEAT")
    assert encoded.quantities[0] == 0

    encoded = encode_turn(
        {"farmer": ["PICKUP", "WHEAT", -5], "hands": [], "market": []}
    )
    assert encoded.units[0] == UNIT_OPS.index("PICKUP:WHEAT")
    assert encoded.quantities[0] == -5


def test_unit_index_folds_an_unparseable_pickup_quantity_to_pass() -> None:
    """A quantity the reference's bare ``int(action[2])`` could not parse either.

    The reference has no try/except around that call for a unit action --
    unlike a market order, which the reference itself catches and aborts. The
    encoder still folds this to PASS rather than claiming to execute an
    action the reference could not, mirroring the tolerance the market-order
    path already documents.
    """
    from kaggriculture.sim.rollout import encode_turn

    encoded = encode_turn(
        {
            "farmer": ["PICKUP", "WHEAT", ["not", "a", "number"]],
            "hands": [],
            "market": [],
        }
    )

    assert encoded.units[0] == UNIT_OPS.index("PASS")
    assert encoded.quantities[0] == 1
