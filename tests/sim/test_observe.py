"""Batched observation parity tests."""
# ruff: noqa: D103

import torch
from kaggle_environments import make

from kaggriculture.learn.encoding import (
    encode_board,
    encode_positions,
    encode_scalars,
)
from kaggriculture.sim.observe import observe
from kaggriculture.sim.state import pack


def test_observe_matches_reference_encoder_for_both_seats_and_a_batch() -> None:
    environments = []
    for seed in (7, 19):
        environment = make("kaggriculture", configuration={"seed": seed}, debug=True)
        environment.reset(2)
        environments.append(environment)
    state = pack(environments)

    for seat in range(2):
        board, scalars, positions = observe(state, seat)
        for batch, environment in enumerate(environments):
            observation = environment.state[seat].observation
            assert torch.equal(
                board[batch : batch + 1], encode_board(observation, seat)
            )
            assert torch.equal(
                scalars[batch : batch + 1], encode_scalars(observation, seat)
            )
            assert torch.equal(
                positions[batch : batch + 1], encode_positions(observation, seat)
            )


def test_observe_returns_tensors_on_the_state_device() -> None:
    environment = make("kaggriculture", configuration={"seed": 3}, debug=True)
    environment.reset(2)
    state = pack([environment])

    outputs = observe(state, 0)

    assert all(output.device == state.step.device for output in outputs)


def test_observe_matches_reference_for_nonempty_tiles_units_and_private_state() -> None:
    environment = make("kaggriculture", configuration={"seed": 47}, debug=True)
    environment.reset(2)
    observation = environment.state[0].observation
    observation.day = 5
    observation.hour = 7
    observation.step = 127
    observation.farms[0].tiles[1][2] = {
        "kind": "PLANT",
        "crop": "TOMATO",
        "planted_day": 2,
        "watered_today": True,
        "consecutive_unwatered": 0,
        "yield_units": 3,
        "max_lifespan_step": -1,
        "fertilized_until_day": 5,
    }
    observation.farms[1].tiles[3][4] = {
        "kind": "PASTURE",
        "animal": "COW",
        "placed_day": 1,
        "yield_units": 2,
        "consecutive_unfed": 1,
        "fed_today": False,
        "cared_today": True,
        "fertilizer_available": True,
        "pending_care_bonus": 3,
    }
    observation.farms[0].hands = [[2, 1], [2, 1]]
    environment.state[0].observation.private.inventories = [
        {"WHEAT": 2},
        {"MILK": 3},
        {"WOOL": 4},
    ]
    for seat in range(1, 2):
        environment.state[seat].observation.farms = observation.farms
        environment.state[seat].observation.day = 5
        environment.state[seat].observation.hour = 7
    state = pack([environment])

    board, scalars, positions = observe(state, 0)

    assert torch.equal(board, encode_board(observation, 0))
    assert torch.equal(scalars, encode_scalars(observation, 0))
    assert torch.equal(positions, encode_positions(observation, 0))
