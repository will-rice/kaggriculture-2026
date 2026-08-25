"""Compatibility contract for the submission-safe canonical observation bundle."""
# ruff: noqa: D103

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from typing import Any, Iterator

import pytest
import torch
from kaggle_environments.envs.kaggriculture import kaggriculture as engine

from kaggriculture.constants import MARKET_PARAMS, PRODUCTS, TURNS_PER_DAY
from kaggriculture.learn.corpus import CORPUS
from kaggriculture.learn.encoding import (
    encode_board as legacy_encode_board,
)
from kaggriculture.learn.encoding import (
    encode_positions as legacy_encode_positions,
)
from kaggriculture.learn.encoding import (
    encode_scalars as legacy_encode_scalars,
)
from kaggriculture.learn.mask import (
    market_mask as legacy_market_mask,
)
from kaggriculture.learn.mask import (
    unit_mask as legacy_unit_mask,
)
from kaggriculture.learn.mask import (
    unit_quantity_mask as legacy_unit_quantity_mask,
)


def _farm() -> dict[str, Any]:
    return {
        "tiles": [[None] * 10 for _ in range(10)],
        "money": 3_000.0,
        "farmer": [4, 4],
        "hands": [],
        "unlocked_quadrants": ["NW"],
        "hires_today": 0,
    }


def rich_observation() -> dict[str, Any]:
    observation = {
        "player": 0,
        "day": 3,
        "hour": 7,
        "farms": [_farm(), _farm()],
        "market": {
            "prices": {name: int(MARKET_PARAMS[name]["base"]) for name in PRODUCTS},
            "inventory": {name: int(MARKET_PARAMS[name]["I0"]) for name in PRODUCTS},
        },
        "town": {"unlocked_shops": []},
        "private": engine._new_private(),
    }
    observation["farms"][0]["money"] = 4_250.0
    observation["farms"][1]["money"] = 1_750.0
    observation["farms"][0]["hands"] = [[2, 3]]
    observation["private"]["inventories"] = [
        {"WHEAT": 5},
        {"CARROT": 2},
    ]
    observation["private"]["shed"]["WHEAT"] = 11
    observation["private"]["seeds"]["MELON"] = 9
    crop = engine._new_plant("MELON", 1, TURNS_PER_DAY)
    crop["consecutive_unwatered"] = 1
    observation["farms"][0]["tiles"][2][3] = crop
    animal = engine._new_animal("GOOSE", 0)
    animal["consecutive_unfed"] = 2
    animal["yield_units"] = 3
    observation["farms"][1]["tiles"][6][7] = animal
    observation["market"]["prices"]["WHEAT"] += 17
    observation["market"]["inventory"]["WHEAT"] -= 23
    return observation


def _assert_parity(observation: dict[str, Any], seat: int) -> None:
    from kaggriculture.features import encode_observation
    from kaggriculture.learn.encoding import to_torch

    expected = (
        legacy_encode_board(observation, seat),
        legacy_encode_scalars(observation, seat),
        legacy_encode_positions(observation, seat),
        legacy_unit_mask(observation, seat),
        legacy_unit_quantity_mask(observation, seat),
        legacy_market_mask(observation, seat),
    )
    actual = to_torch(encode_observation(observation, seat))

    assert torch.equal(actual.board, expected[0])
    assert torch.equal(actual.scalars, expected[1])
    assert torch.equal(actual.positions, expected[2])
    assert torch.equal(actual.unit_mask, expected[3])
    assert torch.equal(actual.quantity_mask, expected[4])
    assert torch.equal(actual.market_mask, expected[5])


def test_canonical_features_round_trip_to_identical_tensors() -> None:
    _assert_parity(rich_observation(), 0)


def test_collocated_hands_preserve_legacy_float32_accumulation() -> None:
    """Repeated tensor adds rounded after every hand, not only at conversion."""
    from kaggriculture.features import PLANE_INDEX, encode_observation
    from kaggriculture.learn.encoding import to_torch

    observation = rich_observation()
    observation["farms"][0]["hands"] = [[2, 3]] * 7
    observation["private"]["inventories"] = [{} for _ in range(8)]

    board = to_torch(encode_observation(observation, 0)).board

    assert board[0, PLANE_INDEX["unit:HANDS"], 3, 2].item() == 0.3500000238418579


def test_named_accessors_select_the_existing_normalized_layout() -> None:
    from kaggriculture.features import encode_observation

    encoded = encode_observation(rich_observation(), 0)

    assert encoded.money() == pytest.approx(0.425)
    assert encoded.money(opponent=True) == pytest.approx(0.175)
    assert encoded.phase() == pytest.approx((3 / 30, 7 / 24, 79 / 720))
    assert encoded.live_price("WHEAT") == pytest.approx(
        17 / float(MARKET_PARAMS["WHEAT"]["base"])
    )
    assert encoded.live_inventory("WHEAT") == pytest.approx(
        -23 / float(MARKET_PARAMS["WHEAT"]["T"])
    )
    assert encoded.seed_count("MELON") == 9
    assert encoded.shed_count("WHEAT") == 11
    assert encoded.carried_count("WHEAT") == 5
    assert encoded.carried_count("CARROT") == 2
    assert encoded.unit_position(0) == (4, 4)
    assert encoded.unit_position(1) == (2, 3)
    assert encoded.crop_at(3, 2) == "MELON"
    assert encoded.animal_at(7, 6, opponent=True) == "GOOSE"
    assert encoded.distress_at(3, 2) == pytest.approx(0.5)
    assert encoded.distress_at(7, 6, opponent=True) == pytest.approx(1.0)
    assert encoded.opponent_public_supply("EGG") == 3


def test_features_imports_without_torch_or_pydantic() -> None:
    script = """
import sys
import kaggriculture.features
assert 'torch' not in sys.modules
assert 'pydantic' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def _corpus_observations(limit: int) -> Iterator[tuple[dict[str, Any], int]]:
    yielded = 0
    for archive in sorted(CORPUS.glob("*.zip")):
        with zipfile.ZipFile(archive) as bundle:
            names = sorted(name for name in bundle.namelist() if name.endswith(".json"))
            for name in names:
                with bundle.open(name) as member:
                    steps = json.load(member)["steps"]
                stride = max(1, len(steps) // 32)
                for index in range(0, len(steps), stride):
                    for seat in (0, 1):
                        yield steps[index][seat]["observation"], seat
                        yielded += 1
                        if yielded >= limit:
                            return


@pytest.mark.slow
def test_canonical_features_are_bit_identical_across_the_replay_corpus() -> None:
    archives = tuple(CORPUS.glob("*.zip")) if CORPUS.is_dir() else ()
    if not archives:
        pytest.skip("replay corpus not present on this machine")
    sampled = 0
    for observation, seat in _corpus_observations(limit=2048):
        _assert_parity(observation, seat)
        sampled += 1
    assert sampled == 2048
