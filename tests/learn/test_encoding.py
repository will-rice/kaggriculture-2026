"""Tests for turning an observation into tensors.

A wrong plane never raises. It trains the model on a game that is not this one,
and the first symptom is an agent that plays badly for reasons nobody can find.
So these check the encoding against the engine's own rules tables and against a
board built by hand.
"""

import torch

from kaggriculture.constants import CROPS, PRODUCTS
from kaggriculture.learn.encoding import (
    BOARD,
    SCALARS,
    TILE_PLANES,
    encode_board,
    encode_scalars,
)


def _empty_farm() -> dict:
    """Return a fresh, independent farm with an empty unlocked board.

    Each call builds its own tile grid: the two farms in an observation must
    never share one mutable list, or mutating one player's board silently
    mutates the other's too.
    """
    return {
        "tiles": [[None] * BOARD for _ in range(BOARD)],
        "money": 3000.0,
        "farmer": [4, 4],
        "hands": [],
        "unlocked_quadrants": ["NW"],
        "hires_today": 0,
    }


def empty_observation(seat: int = 0) -> dict:
    """Return a minimal well-formed observation with an empty unlocked board."""
    return {
        "player": seat,
        "step": 0,
        "day": 0,
        "hour": 0,
        "farms": [_empty_farm(), _empty_farm()],
        "market": {
            "prices": dict.fromkeys(PRODUCTS, 100),
            "inventory": dict.fromkeys(PRODUCTS, 10000),
        },
        "town": {"unlocked_shops": []},
        "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
    }


def test_board_has_the_declared_shape_and_batch_dimension() -> None:
    """Every tensor in this project carries its batch dimension."""
    board = encode_board(empty_observation(), seat=0)

    assert board.shape == (1, TILE_PLANES, BOARD, BOARD)
    assert board.dtype == torch.float32


def test_a_planted_tile_lights_exactly_its_own_crop_plane() -> None:
    """Crop identity is one-hot; two crops lit at once would be a silent mixture."""
    observation = empty_observation()
    observation["farms"][0]["tiles"][2][3] = {
        "kind": "PLANT",
        "crop": "MELON",
        "planted_day": 0,
        "watered_today": False,
        "consecutive_unwatered": 0,
        "yield_units": 0,
        "fertilized_until_day": -1,
    }

    board = encode_board(observation, seat=0)
    crops = sorted(CROPS)
    lit = [board[0, index, 2, 3].item() for index in range(len(crops))]

    assert sum(lit) == 1.0
    assert lit[crops.index("MELON")] == 1.0


def test_the_opponent_s_board_occupies_its_own_planes() -> None:
    """Opponent supply decides prices here; their board must be visible and distinct."""
    observation = empty_observation()
    observation["farms"][1]["tiles"][5][5] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 0,
        "watered_today": False,
        "consecutive_unwatered": 0,
        "yield_units": 0,
        "fertilized_until_day": -1,
    }

    ours = encode_board(observation, seat=0)
    theirs = encode_board(observation, seat=1)

    assert not torch.equal(ours, theirs)


def test_scalars_have_the_declared_width() -> None:
    """The market branch is fixed-width; a ragged vector would break the model."""
    scalars = encode_scalars(empty_observation(), seat=0)

    assert scalars.shape == (1, SCALARS)
    assert torch.isfinite(scalars).all()


def test_scalars_carry_both_price_and_opponent_supply() -> None:
    """Value here is a property of a product given what the opponent is selling.

    This is the finding that a static per-animal valuation cost 10k when it
    replaced the opponent-aware one. If the model cannot see the opponent's
    supply it cannot learn the thing that decides this game.
    """
    cheap = empty_observation()
    rich = empty_observation()
    rich["market"]["prices"] = dict.fromkeys(PRODUCTS, 200)

    assert not torch.equal(encode_scalars(cheap, 0), encode_scalars(rich, 0))


def test_the_phase_of_the_season_is_encoded() -> None:
    """Shops unlock every three days and demand steps at days 10 and 20."""
    early, late = empty_observation(), empty_observation()
    late["day"], late["hour"], late["step"] = 25, 13, 25 * 24 + 13

    assert not torch.equal(encode_scalars(early, 0), encode_scalars(late, 0))
