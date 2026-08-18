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
from kaggriculture.sim.engine import reset, step
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
    environment = make("kaggriculture", configuration={"seed": 223}, debug=True)
    environment.reset(2)
    state = pack([environment])
    opponent_units, markets = scripted_actions(state, economic_policy.agent)
    units = torch.full((1, 2, 20), UNIT_OPS.index("PASS"), dtype=torch.int16)
    units[:, 1].copy_(opponent_units)
    reference_action = economic_policy.agent(environment.state[1].observation)

    environment.step(
        [
            {"farmer": ["PASS"], "hands": [], "market": []},
            reference_action,
        ]
    )
    actual = step(state, units, markets)

    assert_identical(environment, actual, 0)


def test_scripted_actions_raises_on_a_bulk_pickup_from_the_opponent() -> None:
    """The scripted-opponent bridge is bound by the simulator's own domain.

    It has no quantity lane, so a bulk PICKUP or PLACE is outside it.
    ``economic_policy.agent`` issues exactly that once its shed and worker
    inventories hold more than the simulator's supported quantity of one --
    reproduced here directly, without needing to play the opponent out to the
    turn where it first does so. Before the guard, this bulk PICKUP silently
    executed as a transfer of one instead of raising.
    """
    state = reset(Config(), torch.tensor([223]))

    def bulk_pickup(observation: object) -> dict[str, object]:
        return {"farmer": ["PICKUP", "WHEAT", 2], "hands": [], "market": []}

    with pytest.raises(ValueError, match="PICKUP.*WHEAT.*2"):
        scripted_actions(state, bulk_pickup)


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


def test_unit_index_raises_on_a_pickup_quantity_outside_its_domain() -> None:
    """A unit action has no quantity lane; a route that needs one is out of domain.

    Silently collapsing ``["PICKUP", "FERTILIZER", 3]`` to a transfer of one is
    what let a recorded route replay through the simulator and diverge from its
    own recorded result without any error. The simulator must refuse instead.
    """
    from kaggriculture.sim.rollout import encode_turn

    with pytest.raises(ValueError, match="PICKUP.*FERTILIZER.*3"):
        encode_turn({"farmer": ["PICKUP", "FERTILIZER", 3], "hands": [], "market": []})


def test_unit_index_raises_on_a_place_quantity_outside_its_domain() -> None:
    """The same guard applies to PLACE, the other transfer verb."""
    from kaggriculture.sim.rollout import encode_turn

    with pytest.raises(ValueError, match="PLACE.*WHEAT.*2"):
        encode_turn(
            {"farmer": ["PASS"], "hands": [["PLACE", "WHEAT", 2]], "market": []}
        )


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
    """Quantity 1 is the whole supported domain, and it still works."""
    from kaggriculture.sim.rollout import encode_turn

    encoded = encode_turn(
        {"farmer": ["PICKUP", "FERTILIZER", 1], "hands": [], "market": []}
    )

    assert encoded.units[0] == UNIT_OPS.index("PICKUP:FERTILIZER")
