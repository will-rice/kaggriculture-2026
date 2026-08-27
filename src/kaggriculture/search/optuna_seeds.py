"""Audited warm starts for the direct semantic Optuna study."""

import hashlib
from pathlib import Path

from kaggriculture.hybrid.config import HybridConfig
from kaggriculture.search.evolution import SearchState
from kaggriculture.search.optuna_space import config_sha256

LEGACY_STATE_SHA256 = "c74ae9ff99fc6bd1d39c93f2edda8ac3a9ee1894d1422c2e02e9c6d02aab5c5b"

DELIBERATE_PAYLOADS: tuple[dict[str, object], ...] = (
    {
        "opening": {
            "phases": (
                {
                    "start_day": 0,
                    "target_hands": 3,
                    "target_quadrants": 1,
                    "primary_crop": "WHEAT",
                    "crop_targets": (0, 0, 0, 0, 8),
                    "animal_targets": (0, 0, 0),
                    "structure_targets": (0, 0),
                    "cash_reserve": 1000,
                    "inventory_reserve": 4,
                },
                {
                    "start_day": 12,
                    "target_hands": 6,
                    "target_quadrants": 2,
                    "primary_crop": "STRAWBERRY",
                    "crop_targets": (0, 0, 10, 0, 8),
                    "animal_targets": (0, 0, 0),
                    "structure_targets": (0, 0),
                    "cash_reserve": 1200,
                    "inventory_reserve": 6,
                },
                {
                    "start_day": 22,
                    "target_hands": 8,
                    "target_quadrants": 3,
                    "primary_crop": "STRAWBERRY",
                    "crop_targets": (0, 0, 18, 0, 8),
                    "animal_targets": (0, 0, 0),
                    "structure_targets": (0, 0),
                    "cash_reserve": 1500,
                    "inventory_reserve": 8,
                },
            )
        },
        "jobs": {
            "recovery": 9.0,
            "urgency": 7.0,
            "distance_penalty": 1.5,
            "production": 7.0,
            "transport": 6.0,
            "structure": 2.0,
        },
        "market": {
            "live_price": 7.0,
            "town_demand": 6.0,
            "opponent_supply": 4.0,
            "reserve_penalty": 9.0,
            "liquidation_urgency": 8.0,
        },
        "liquidation_start_day": 25,
    },
    {
        "opening": {
            "phases": (
                {
                    "start_day": 0,
                    "target_hands": 5,
                    "target_quadrants": 1,
                    "primary_crop": "WHEAT",
                    "crop_targets": (0, 0, 0, 0, 12),
                    "animal_targets": (0, 0, 0),
                    "structure_targets": (0, 0),
                    "cash_reserve": 200,
                    "inventory_reserve": 2,
                },
                {
                    "start_day": 10,
                    "target_hands": 10,
                    "target_quadrants": 3,
                    "primary_crop": "STRAWBERRY",
                    "crop_targets": (0, 0, 18, 0, 12),
                    "animal_targets": (0, 0, 0),
                    "structure_targets": (0, 0),
                    "cash_reserve": 300,
                    "inventory_reserve": 3,
                },
                {
                    "start_day": 20,
                    "target_hands": 15,
                    "target_quadrants": 5,
                    "primary_crop": "STRAWBERRY",
                    "crop_targets": (0, 0, 30, 0, 12),
                    "animal_targets": (0, 0, 0),
                    "structure_targets": (0, 0),
                    "cash_reserve": 500,
                    "inventory_reserve": 4,
                },
            )
        },
        "jobs": {
            "recovery": 7.0,
            "urgency": 8.0,
            "distance_penalty": 0.5,
            "production": 10.0,
            "transport": 7.0,
            "structure": 1.0,
        },
        "market": {
            "live_price": 5.0,
            "town_demand": 5.0,
            "opponent_supply": 3.0,
            "reserve_penalty": 3.0,
            "liquidation_urgency": 10.0,
        },
        "liquidation_start_day": 27,
    },
    {
        "opening": {
            "phases": (
                {
                    "start_day": 0,
                    "target_hands": 3,
                    "target_quadrants": 1,
                    "primary_crop": "WHEAT",
                    "crop_targets": (0, 0, 0, 0, 6),
                    "animal_targets": (0, 0, 0),
                    "structure_targets": (0, 0),
                    "cash_reserve": 1200,
                    "inventory_reserve": 5,
                },
                {
                    "start_day": 14,
                    "target_hands": 5,
                    "target_quadrants": 2,
                    "primary_crop": "STRAWBERRY",
                    "crop_targets": (0, 0, 8, 0, 6),
                    "animal_targets": (0, 0, 0),
                    "structure_targets": (0, 0),
                    "cash_reserve": 1800,
                    "inventory_reserve": 8,
                },
                {
                    "start_day": 24,
                    "target_hands": 6,
                    "target_quadrants": 2,
                    "primary_crop": "STRAWBERRY",
                    "crop_targets": (0, 0, 10, 0, 6),
                    "animal_targets": (0, 0, 0),
                    "structure_targets": (0, 0),
                    "cash_reserve": 2400,
                    "inventory_reserve": 10,
                },
            )
        },
        "jobs": {
            "recovery": 10.0,
            "urgency": 9.0,
            "distance_penalty": 2.0,
            "production": 5.0,
            "transport": 8.0,
            "structure": 1.0,
        },
        "market": {
            "live_price": 9.0,
            "town_demand": 8.0,
            "opponent_supply": 6.0,
            "reserve_penalty": 10.0,
            "liquidation_urgency": 10.0,
        },
        "liquidation_start_day": 23,
    },
)


def deliberate_seed_configs() -> tuple[HybridConfig, HybridConfig, HybridConfig]:
    """Return the three deliberately contrasting complete policy configurations."""
    first, second, third = (
        HybridConfig.model_validate(payload) for payload in DELIBERATE_PAYLOADS
    )
    return first, second, third


def warm_start_configs(
    path: Path, expected_sha256: str = LEGACY_STATE_SHA256
) -> tuple[HybridConfig, ...]:
    """Load the audited legacy elites and reject changed or duplicate inputs."""
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "legacy search state SHA-256 does not match the expected digest"
        )
    state = SearchState.load(path)
    if state.generation != 8:
        raise ValueError(
            "legacy search state must be the completed generation-8 artifact"
        )
    if len(state.elites) != 4:
        raise ValueError("legacy search state must contain exactly four elites")
    configs = (
        HybridConfig.default(),
        *(row.config for row in state.elites),
        *deliberate_seed_configs(),
    )
    if len({config_sha256(config) for config in configs}) != len(configs):
        raise ValueError(
            "warm starts must have distinct canonical configuration digests"
        )
    return configs
