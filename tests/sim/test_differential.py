"""Differential assertion behavior tests."""
# ruff: noqa: D103

from dataclasses import replace

import pytest
import torch
from kaggle_environments import make

from kaggriculture.learn.encoding import (
    encode_board,
    encode_positions,
    encode_private_belief_target,
    encode_scalars,
)
from kaggriculture.learn.mask import market_mask, unit_mask, unit_quantity_mask
from kaggriculture.sim.engine import MarketActions, step, unit_quantity_ones
from kaggriculture.sim.legality import legal
from kaggriculture.sim.observe import observe
from kaggriculture.sim.rollout import belief_targets, encode_turn
from kaggriculture.sim.state import pack
from tests.sim.conftest import assert_identical


def test_assert_identical_accepts_an_exact_codec_round_trip() -> None:
    environment = make("kaggriculture", configuration={"seed": 163}, debug=True)
    environment.reset(2)

    assert_identical(environment, pack([environment]), 0)


def test_assert_identical_names_the_first_corrupted_field() -> None:
    environment = make("kaggriculture", configuration={"seed": 167}, debug=True)
    environment.reset(2)
    state = pack([environment])
    money = state.money.clone()
    money[0, 0] += 1
    corrupted = replace(state, money=money)

    with pytest.raises(AssertionError, match=r"seat\[0\]\.farms\[0\]\.money"):
        assert_identical(environment, corrupted, 0)


def test_fixed_legal_tape_matches_bulk_pickup_and_place_end_to_end() -> None:
    """A fixed tape agrees before and after both quantity-bearing unit ops."""
    environment = make("kaggriculture", configuration={"seed": 173}, debug=True)
    environment.reset(2)
    # Put the learning farmer on a shed-access tile with a non-empty shelf.
    # The first tape action takes two wheat; the second puts those exact two
    # back, so both quantity-bearing operations are exercised in one tiny
    # deterministic differential rather than inferred from a random policy.
    for agent in environment.state:
        agent.observation.farms[0].farmer = [4, 4]
    private = environment.state[0].observation.private
    private.shed["WHEAT"] = 3
    private.inventories = [{}]
    state = pack([environment])
    tape = (
        {"farmer": ["PICKUP", "WHEAT", 2], "hands": [], "market": []},
        {"farmer": ["PLACE", "WHEAT", 2], "hands": [], "market": []},
    )
    passed = {"farmer": ["PASS"], "hands": [], "market": []}

    before_banks: list[torch.Tensor] = []
    after_banks: list[torch.Tensor] = []
    for action in tape:
        before_banks.append(state.money.clone())
        native_observations = [observe(state, seat) for seat in range(2)]
        native_masks = [legal(state, seat) for seat in range(2)]
        native_beliefs = belief_targets(state)
        for seat in range(2):
            observation = environment.state[seat].observation
            assert torch.equal(
                native_observations[seat][0], encode_board(observation, seat)
            )
            assert torch.equal(
                native_observations[seat][1], encode_scalars(observation, seat)
            )
            assert torch.equal(
                native_observations[seat][2], encode_positions(observation, seat)
            )
            assert torch.equal(native_masks[seat][0], unit_mask(observation, seat))
            assert torch.equal(
                native_masks[seat][1], unit_quantity_mask(observation, seat)
            )
            assert torch.equal(native_masks[seat][2], market_mask(observation, seat))
            opposing_private = environment.state[1 - seat].observation
            assert torch.equal(
                native_beliefs[0, seat],
                encode_private_belief_target(opposing_private).tensor,
            )

        encoded = (encode_turn(action), encode_turn(passed))
        units = torch.tensor([[entry.units for entry in encoded]], dtype=torch.int16)
        quantities = unit_quantity_ones(1)
        quantities.copy_(
            torch.tensor([[entry.quantities for entry in encoded]], dtype=torch.int16)
        )
        markets = MarketActions.empty(1)
        for seat, entry in enumerate(encoded):
            for slot, (kind, item, quantity) in enumerate(entry.orders):
                markets.order_type[0, seat, slot] = kind
                markets.order_item[0, seat, slot] = item
                markets.order_qty[0, seat, slot] = quantity

        # The decoded tape preserves the bucket-independent engine quantity;
        # this is the datum that used to collapse every bulk transfer to one.
        assert int(quantities[0, 0, 0]) == 2
        environment.step([action, passed])
        state = step(state, units, markets, quantities)
        after_banks.append(state.money.clone())

        assert_identical(environment, state, 0)
        assert torch.equal(state.money, after_banks[-1])
        assert bool(state.done[0]) == all(
            str(agent.status) == "DONE" for agent in environment.state
        )

    assert torch.equal(before_banks[0], after_banks[-1])
    assert environment.state[0].observation.private.shed["WHEAT"] == 3
    assert environment.state[0].observation.private.inventories[0] == {}
