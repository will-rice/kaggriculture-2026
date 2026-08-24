"""Tensor rollout primitive tests."""
# ruff: noqa: ANN001, ANN202, D103

from dataclasses import fields, replace

import pytest
import torch
from kaggle_environments import make

from kaggriculture import economic_policy
from kaggriculture.learn.encoding import (
    CARRIED_SCALE,
    IGNORE,
    MARKET_SLOTS,
    MAX_TRANSFER,
    QUANTITIES,
    SEED_SCALE,
    UNIT_OPS,
)
from kaggriculture.learn.progress import potential as reference_potential
from kaggriculture.learn.toad.config import ModelConfig
from kaggriculture.learn.toad.model import PolicyOutput, PolicyState, StatefulPolicy
from kaggriculture.sim.config import Config
from kaggriculture.sim.engine import reset, step, unit_quantity_ones
from kaggriculture.sim.observe import observe
from kaggriculture.sim.rollout import (
    RolloutPolicyState,
    collect_segment,
    copy_policy_state_,
    potential,
    scripted_actions,
)
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
            torch.zeros(batch, 20, len(QUANTITIES), device=device),
            torch.zeros(batch, len(MARKET_SLOTS) + 2, len(QUANTITIES), device=device),
            torch.zeros(batch, 1, device=device),
        )


class _SellWheatPolicy(_UniformPolicy):
    def forward(self, board, scalars, positions):
        units, quantities, markets, values = super().forward(board, scalars, positions)
        markets[:, 0, 1] = 100
        return units, quantities, markets, values


class _RecordingStatefulPolicy(StatefulPolicy):
    """A real recurrent policy that retains its reset-adjusted actor inputs."""

    def __init__(self, *, belief: bool = True) -> None:
        super().__init__(
            ModelConfig.control(blocks=1, channels=4).model_copy(
                update={
                    "recurrent": True,
                    "recurrent_channels": 3,
                    "belief": belief,
                }
            )
        )
        self.recorded_inputs: list[PolicyState] = []

    def forward(
        self,
        board: torch.Tensor,
        scalars: torch.Tensor,
        positions: torch.Tensor,
        state: PolicyState | None = None,
        dones: torch.Tensor | None = None,
    ) -> PolicyOutput:
        output = super().forward(board, scalars, positions, state=state, dones=dones)
        assert output.input_state is not None
        self.recorded_inputs.append(output.input_state.detach())
        return output


def test_copy_policy_state_preserves_static_recurrent_buffers() -> None:
    target = PolicyState(
        hidden=torch.zeros(2, 3, 10, 10),
        cell=torch.zeros(2, 3, 10, 10),
        prior_belief=torch.zeros(2, 4),
    )
    source = PolicyState(
        hidden=torch.ones_like(target.hidden),
        cell=torch.full_like(target.cell, 2),
        prior_belief=torch.full_like(target.prior_belief, 3),
    )
    pointers = (
        target.hidden.data_ptr(),
        target.cell.data_ptr(),
        target.prior_belief.data_ptr(),
    )

    copied = copy_policy_state_(target, source)

    assert copied is target
    assert pointers == (
        target.hidden.data_ptr(),
        target.cell.data_ptr(),
        target.prior_belief.data_ptr(),
    )
    assert torch.equal(target.hidden, source.hidden)
    assert torch.equal(target.cell, source.cell)
    assert torch.equal(target.prior_belief, source.prior_belief)


def test_stateful_collect_segment_carries_resets_and_records_sparse_entry_state() -> (
    None
):
    """A terminal transition clears every actor-state row before the next action."""
    state = reset(Config(), torch.tensor([197, 199]))
    state = replace(
        state,
        step=torch.full((2,), 718, dtype=torch.int32),
        day=torch.full((2,), 29, dtype=torch.int32),
        hour=torch.full((2,), 22, dtype=torch.int32),
    )
    policy = _RecordingStatefulPolicy().eval()
    initial = policy.initial_state(4, like=state.money.float())
    assert initial is not None
    carried = PolicyState(
        hidden=torch.ones_like(initial.hidden),
        cell=torch.full_like(initial.cell, 2.0),
        prior_belief=torch.full_like(initial.prior_belief, 3.0),
    )

    next_state, next_policy_state, trajectory = collect_segment(
        state,
        policy,
        policy_state=carried,
        turns=2,
        generator=torch.Generator().manual_seed(211),
    )

    assert next_state.done.all()
    assert next_policy_state is not None
    assert trajectory.state_steps is not None
    assert trajectory.state_steps.tolist() == [0]
    assert trajectory.hidden is not None
    assert trajectory.cell is not None
    assert trajectory.prior_belief is not None
    assert trajectory.hidden.shape == (1, 2, 2, 3, 10, 10)
    assert trajectory.cell.shape == trajectory.hidden.shape
    assert trajectory.prior_belief.shape == (1, 2, 2, policy.config.belief_size)
    reset_input = policy.recorded_inputs[1]
    assert not torch.count_nonzero(reset_input.hidden)
    assert not torch.count_nonzero(reset_input.cell)
    assert not torch.count_nonzero(reset_input.prior_belief)


def test_stateful_collect_segment_records_exact_end_anchored_boundaries() -> None:
    """Changing sparse retention must not shift a learner segment by one turn."""
    state = reset(Config(), torch.tensor([223]))
    policy = _RecordingStatefulPolicy().eval()

    _state, _policy_state, trajectory = collect_segment(
        state,
        policy,
        turns=5,
        state_unroll_length=2,
        generator=torch.Generator().manual_seed(227),
    )

    assert trajectory.state_steps is not None
    assert trajectory.state_steps.tolist() == [1, 3]
    assert trajectory.hidden is not None
    assert trajectory.hidden.shape[:3] == (2, 1, 2)


def test_stateful_collect_segment_records_opponent_private_belief_targets() -> None:
    """Belief labels are simulator truth aligned to each acting seat, not inputs."""
    state = reset(Config(), torch.tensor([229]))
    shed = state.shed.clone()
    seeds = state.seeds.clone()
    inventories = state.inv_count.clone()
    shed[0, 1, SHED_NAMES.index("WHEAT")] = 7
    seeds[0, 1, 0] = 5
    inventories[0, 1, 0, SHED_NAMES.index(PRODUCT_NAMES[0])] = 3
    state = replace(state, shed=shed, seeds=seeds, inv_count=inventories)
    policy = _RecordingStatefulPolicy().eval()

    _state, _policy_state, trajectory = collect_segment(
        state,
        policy,
        turns=1,
        generator=torch.Generator().manual_seed(233),
    )

    assert trajectory.belief_targets is not None
    assert trajectory.belief_valid is not None
    seat_zero = trajectory.belief_targets[0, 0, 0]
    assert seat_zero[SHED_NAMES.index("WHEAT")] == pytest.approx(7 / 100)
    seed_start = len(SHED_NAMES)
    assert seat_zero[seed_start] == pytest.approx(5 / SEED_SCALE)
    carried_start = seed_start + seeds.shape[-1]
    assert seat_zero[carried_start] == pytest.approx(3 / CARRIED_SCALE)
    assert trajectory.belief_valid.all()


def test_collect_segment_keeps_actions_legal_and_tensors_on_device() -> None:
    state = reset(Config(), torch.tensor([197, 199]))

    next_state, _policy_state, trajectory = collect_segment(
        state,
        _UniformPolicy(),
        turns=2,
        generator=torch.Generator().manual_seed(211),
    )

    assert next_state.step.tolist() == [2, 2]
    assert trajectory.board.shape[:3] == (3, 2, 2)
    assert trajectory.unit_actions.shape == (2, 2, 2, 20)
    assert int(trajectory.illegal) == 0
    assert trajectory.illegal.device == state.step.device
    assert trajectory.board.device == state.step.device

    final_observations = [observe(next_state, seat) for seat in range(2)]
    assert torch.equal(
        trajectory.board[-1],
        torch.stack([entry[0] for entry in final_observations], dim=1),
    )
    assert torch.equal(
        trajectory.scalars[-1],
        torch.stack([entry[1] for entry in final_observations], dim=1),
    )
    assert torch.equal(
        trajectory.positions[-1],
        torch.stack([entry[2] for entry in final_observations], dim=1),
    )


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

    next_state, _policy_state, trajectory = collect_segment(
        state,
        _UniformPolicy(),
        turns=2,
        generator=torch.Generator().manual_seed(229),
        opponent=economic_policy.agent,
    )

    assert next_state.step.tolist() == [2]
    assert int(trajectory.illegal) == 0


class _CalledPolicy(_UniformPolicy):
    """A distinct neural opponent whose invocation is directly observable."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, board, scalars, positions):
        self.calls += 1
        return super().forward(board, scalars, positions)


def test_collect_segment_accepts_a_distinct_neural_opponent() -> None:
    """Frozen and teacher seats act from their own verified policy object."""
    state = reset(Config(), torch.tensor([231]))
    learner = _CalledPolicy()
    opponent = _CalledPolicy()

    next_state, _policy_state, trajectory = collect_segment(
        state,
        learner,
        turns=2,
        generator=torch.Generator().manual_seed(233),
        opponent_policy=opponent,
    )

    assert next_state.step.tolist() == [2]
    assert learner.calls == 2
    assert opponent.calls == 2
    assert int(trajectory.illegal) == 0


def test_stateful_neural_opponent_matches_one_long_segment_across_boundaries() -> None:
    """Both actor memories, not only the learner's, cross native source chunks."""
    model = ModelConfig.control(blocks=1, channels=4).model_copy(
        update={"recurrent": True, "recurrent_channels": 3, "belief": True}
    )
    long_learner = StatefulPolicy(model).eval()
    long_opponent = StatefulPolicy(model).eval()
    split_learner = StatefulPolicy(model).eval()
    split_opponent = StatefulPolicy(model).eval()
    split_learner.load_state_dict(long_learner.state_dict())
    split_opponent.load_state_dict(long_opponent.state_dict())
    long_state = reset(Config(), torch.tensor([241]))
    split_state = reset(Config(), torch.tensor([241]))

    long_state, long_policy_state, long = collect_segment(
        long_state,
        long_learner,
        turns=4,
        generator=torch.Generator().manual_seed(251),
        opponent_policy=long_opponent,
    )
    split_generator = torch.Generator().manual_seed(251)
    split_state, split_policy_state, first = collect_segment(
        split_state,
        split_learner,
        turns=2,
        generator=split_generator,
        opponent_policy=split_opponent,
    )
    split_state, split_policy_state, second = collect_segment(
        split_state,
        split_learner,
        policy_state=split_policy_state,
        turns=2,
        generator=split_generator,
        opponent_policy=split_opponent,
    )

    assert isinstance(long_policy_state, RolloutPolicyState)
    assert isinstance(split_policy_state, RolloutPolicyState)
    assert long_policy_state.opponent is not None
    assert split_policy_state.opponent is not None
    for field in fields(long_state):
        torch.testing.assert_close(
            getattr(split_state, field.name), getattr(long_state, field.name)
        )
    for name in (
        "unit_actions",
        "unit_quantities",
        "market_actions",
        "log_probs",
        "values",
        "rewards",
        "own",
        "shaped",
        "shaped_money",
        "margin",
        "sparse",
        "dones",
    ):
        torch.testing.assert_close(
            torch.cat((getattr(first, name), getattr(second, name))),
            getattr(long, name),
        )
    for name in ("hidden", "cell", "prior_belief"):
        torch.testing.assert_close(
            getattr(split_policy_state.learner, name),
            getattr(long_policy_state.learner, name),
        )
        torch.testing.assert_close(
            getattr(split_policy_state.opponent, name),
            getattr(long_policy_state.opponent, name),
        )


def test_stateful_neural_opponent_resets_after_a_segment_terminal() -> None:
    """A carried opponent state still obeys the next observation's done mask."""
    state = reset(Config(), torch.tensor([257]))
    state = replace(
        state,
        step=torch.tensor([718], dtype=torch.int32),
        day=torch.tensor([29], dtype=torch.int32),
        hour=torch.tensor([22], dtype=torch.int32),
    )
    learner = _RecordingStatefulPolicy().eval()
    opponent = _RecordingStatefulPolicy().eval()

    state, carried, _first = collect_segment(
        state,
        learner,
        turns=1,
        generator=torch.Generator().manual_seed(263),
        opponent_policy=opponent,
    )
    _state, carried, _second = collect_segment(
        state,
        learner,
        policy_state=carried,
        turns=1,
        generator=torch.Generator().manual_seed(269),
        opponent_policy=opponent,
    )

    assert isinstance(carried, RolloutPolicyState)
    reset_input = opponent.recorded_inputs[-1]
    assert not torch.count_nonzero(reset_input.hidden)
    assert not torch.count_nonzero(reset_input.cell)
    assert not torch.count_nonzero(reset_input.prior_belief)


def test_collect_segment_rejects_two_opponent_owners() -> None:
    """A seat cannot silently switch from a named script to neural weights."""
    state = reset(Config(), torch.tensor([237]))

    with pytest.raises(ValueError, match="mutually exclusive"):
        collect_segment(
            state,
            _UniformPolicy(),
            turns=1,
            opponent=economic_policy.agent,
            opponent_policy=_UniformPolicy(),
        )


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

    final_state, _policy_state, trajectory = collect_segment(
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


def test_encode_order_raises_on_a_buy_seed_quantity_over_the_axis() -> None:
    """BUY_SEED has no shed to bound it, so an over-axis request still raises.

    Seeds land in ``private["seeds"]``, never the shed, so unlike
    SELL/BUY_PRODUCT/BUY_ANIMAL (bounded by ``SHED_CAPACITY`` and safe to
    clamp -- see ``sim.rollout.SHED_BOUND_VERBS``), a BUY_SEED request this
    large has no structural bound to appeal to. Clamping it would risk
    silently truncating a genuinely large seed fill the reference would have
    executed in full, which is exactly the divergence this guard exists to
    prevent. 1,000,000 is the engine's own "sell everything" sentinel
    magnitude, reused here as a quantity always well past whatever
    ``QUANTITY_AXIS`` is. The RL action space never emits a quantity this
    large, so this never fires for it -- only for an order outside the
    simulator's domain.
    """
    from kaggriculture.sim.rollout import encode_turn

    with pytest.raises(ValueError, match="BUY_SEED.*WHEAT.*1000000"):
        encode_turn(
            {
                "farmer": ["PASS"],
                "hands": [],
                "market": [["BUY_SEED", "WHEAT", 1_000_000]],
            }
        )


def test_encode_order_clamps_a_sell_quantity_over_the_axis() -> None:
    """SELL is shed-bound, so an over-axis request clamps rather than raising.

    See ``sim.rollout.SHED_BOUND_VERBS`` for the equivalence argument: the
    engine can never sell more than the shed holds, which is always below
    ``QUANTITY_AXIS``, so clamping the request here changes nothing about
    what gets executed.
    """
    from kaggriculture.sim.rollout import QUANTITY_AXIS, encode_turn

    encoded = encode_turn(
        {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", 999]]}
    )

    assert encoded.orders[0] == (1, PRODUCT_NAMES.index("WHEAT"), QUANTITY_AXIS)


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


class _BulkPickupPolicy(_UniformPolicy):
    """A policy that picks up wheat three at a time and trades nothing.

    The market is pinned to bucket 0 rather than left uniform: with wheat in
    the shed a uniform market head sells some of it on the same turn, and the
    shed count this fixture reads would then be measuring two lanes at once.
    """

    def forward(self, board, scalars, positions):
        units, quantities, markets, values = super().forward(board, scalars, positions)
        units[:, :, UNIT_OPS.index("PICKUP:WHEAT")] = 100
        quantities[:, :, QUANTITIES.index(3)] = 100
        markets[:, :, 0] = 100
        return units, quantities, markets, values


def test_collect_segment_records_a_quantity_for_every_unit_slot() -> None:
    """The lane is sampled on-device, per unit, and stored beside the op.

    Two properties, and the second is what a shape assertion would miss. The
    buckets recorded for the units that exist have to land inside the range
    ``legality.quantity_mask`` permits -- one to twelve, the exact end of
    ``QUANTITIES`` -- and they have to *vary*, because a collector that quietly
    kept handing the engine ``unit_quantity_ones`` records a legal-looking
    column of ones and passes every range check ever written against it.

    The dead slots are checked in the other direction: their row of the mask
    holds the single-item bucket alone, so anything else there means the mask
    and the sample have come apart.
    """
    state = reset(Config(), torch.tensor([241, 251]))

    _next_state, _policy_state, trajectory = collect_segment(
        state,
        _UniformPolicy(),
        turns=4,
        generator=torch.Generator().manual_seed(257),
    )

    quantities = torch.tensor(QUANTITIES)[trajectory.unit_quantities]
    alive = trajectory.unit_actions != IGNORE

    assert trajectory.unit_quantities.shape == trajectory.unit_actions.shape
    assert int(quantities[alive].min()) >= 1
    assert int(quantities[alive].max()) <= MAX_TRANSFER
    assert (quantities[~alive] == 1).all()
    assert int(quantities[alive].max()) > 1


def test_collect_segment_spends_the_sampled_quantity_in_the_engine() -> None:
    """The sampled bucket has to reach ``step``, not just the trajectory.

    Three wheat out of a shed of five, in one turn, by one farmer standing on
    the shed-access tile the season opens on. A lane that recorded its samples
    and still handed the engine a column of ones leaves four in the shed here
    and one in the farmer's hands, which is the state this collector produced
    before the quantity head reached it.
    """
    wheat = SHED_NAMES.index("WHEAT")
    state = reset(Config(), torch.tensor([263]))
    shed = state.shed.clone()
    shed[:, :, wheat] = 5
    state = replace(state, shed=shed)

    next_state, _policy_state, _trajectory = collect_segment(
        replace(state),
        _BulkPickupPolicy(),
        turns=1,
        generator=torch.Generator().manual_seed(269),
    )

    assert int(next_state.shed[0, 0, wheat]) == 2
    assert int(next_state.inv_count[0, 0, 0, wheat]) == 3
