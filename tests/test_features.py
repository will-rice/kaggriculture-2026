"""Compatibility contract for the submission-safe canonical observation bundle."""
# ruff: noqa: D103

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from kaggriculture.constants import MARKET_PARAMS
from kaggriculture.learn.corpus import CORPUS
from tests.feature_golden_generator import (
    BASE_COMMIT,
    CORPUS_LIMIT,
    FIELDS,
    corpus_observations,
    rich_observation,
    tensor_digests,
)

_GOLDENS = json.loads(
    (Path(__file__).parent / "fixtures" / "feature_goldens.json").read_text()
)


def _tensor_row(observation: dict[str, Any], seat: int) -> dict[str, torch.Tensor]:
    from kaggriculture.features import encode_observation
    from kaggriculture.learn.encoding import to_torch

    actual = to_torch(encode_observation(observation, seat))
    return {field: getattr(actual, field) for field in FIELDS}


def _assert_rich_golden() -> None:
    assert tensor_digests([_tensor_row(rich_observation(), 0)]) == _GOLDENS["rich"]


def test_canonical_features_round_trip_to_identical_tensors() -> None:
    _assert_rich_golden()


def test_golden_provenance_is_the_exact_pre_refactor_commit() -> None:
    assert _GOLDENS["provenance"] == {
        "archives": ["kaggriculture-episodes-2026-07-30.zip"],
        "base_commit": BASE_COMMIT,
        "corpus_limit": CORPUS_LIMIT,
        "generator": "tests/feature_golden_generator.py",
    }


def test_parity_oracle_catches_a_canonical_only_scalar_perturbation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A defect in the canonical producer must not also alter the oracle."""
    import kaggriculture.features as features

    original = features._encode_scalar_values

    def perturbed(observation: dict[str, Any], seat: int) -> tuple[float, ...]:
        values = original(observation, seat)
        return (values[0] + 1.0, *values[1:])

    monkeypatch.setattr(features, "_encode_scalar_values", perturbed)

    with pytest.raises(AssertionError):
        _assert_rich_golden()


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


def test_named_facts_recover_opening_counts_without_changing_tensor_features() -> None:
    from kaggle_environments.envs.kaggriculture import kaggriculture as engine

    from kaggriculture.features import encode_observation

    observation = rich_observation()
    observation["farms"][0]["unlocked_quadrants"] = ["NW", "NE"]
    observation["farms"][0]["hands"] = [[2, 3], [3, 3]]
    observation["private"]["inventories"] = [{"GOOSE": 1}, {}, {}]
    observation["farms"][0]["tiles"][0][0] = engine._new_animal("GOOSE", 0)
    observation["farms"][0]["tiles"][0][1] = {"kind": "COOP"}

    encoded = encode_observation(observation, 0)

    assert encoded.day_count() == 3
    assert encoded.hand_count() == 2
    assert encoded.quadrant_count() == 2
    assert encoded.carried_item_count("GOOSE") == 1
    assert encoded.animal_count("GOOSE") == 2
    assert encoded.structure_count("COOP") == 2


def test_features_imports_without_torch_or_pydantic() -> None:
    script = """
import sys
import kaggriculture.features
assert 'torch' not in sys.modules
assert 'pydantic' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.slow
def test_canonical_features_are_bit_identical_across_the_replay_corpus() -> None:
    archives = tuple(CORPUS.glob("*.zip")) if CORPUS.is_dir() else ()
    if not archives:
        pytest.skip("replay corpus not present on this machine")
    rows: list[dict[str, torch.Tensor]] = []
    used_archives: list[str] = []
    for archive, observation, seat in corpus_observations(CORPUS):
        if archive not in used_archives:
            used_archives.append(archive)
        rows.append(_tensor_row(observation, seat))

    assert len(rows) == CORPUS_LIMIT
    assert used_archives == _GOLDENS["provenance"]["archives"]
    assert tensor_digests(rows) == _GOLDENS["corpus"]
