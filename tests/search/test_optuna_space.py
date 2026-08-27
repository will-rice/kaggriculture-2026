"""Contract tests for the direct semantic Optuna parameter space."""

import optuna

from kaggriculture.hybrid.config import HybridConfig
from kaggriculture.search.optuna_space import (
    SPACE_MANIFEST,
    SPACE_SHA256,
    ParameterSpec,
    parameters_for_config,
    suggest_config,
)


def _expected_specs() -> tuple[ParameterSpec, ...]:
    """Return the hand-written public parameter table, in stable order."""
    return (
        ParameterSpec("opening.phase_0.target_hands", "int", 0, 19),
        ParameterSpec("opening.phase_0.target_quadrants", "int", 1, 5),
        ParameterSpec(
            "opening.phase_0.primary_crop",
            "categorical",
            choices=("CARROT", "MELON", "STRAWBERRY", "TOMATO", "WHEAT"),
        ),
        ParameterSpec("opening.phase_0.crop.CARROT", "int", 0, 100),
        ParameterSpec("opening.phase_0.crop.MELON", "int", 0, 100),
        ParameterSpec("opening.phase_0.crop.STRAWBERRY", "int", 0, 100),
        ParameterSpec("opening.phase_0.crop.TOMATO", "int", 0, 100),
        ParameterSpec("opening.phase_0.crop.WHEAT", "int", 0, 100),
        ParameterSpec("opening.phase_0.animal.COW", "int", 0, 100),
        ParameterSpec("opening.phase_0.animal.GOOSE", "int", 0, 100),
        ParameterSpec("opening.phase_0.animal.SHEEP", "int", 0, 100),
        ParameterSpec("opening.phase_0.structure.coop", "int", 0, 100),
        ParameterSpec("opening.phase_0.structure.pasture", "int", 0, 100),
        ParameterSpec("opening.phase_0.cash_reserve", "int", 0, 5_000, 100),
        ParameterSpec("opening.phase_0.inventory_reserve", "int", 0, 100),
        ParameterSpec("opening.phase_1.start_day", "int", 10, 19),
        ParameterSpec("opening.phase_1.target_hands", "int", 0, 19),
        ParameterSpec("opening.phase_1.target_quadrants", "int", 1, 5),
        ParameterSpec(
            "opening.phase_1.primary_crop",
            "categorical",
            choices=("CARROT", "MELON", "STRAWBERRY", "TOMATO", "WHEAT"),
        ),
        ParameterSpec("opening.phase_1.crop.CARROT", "int", 0, 100),
        ParameterSpec("opening.phase_1.crop.MELON", "int", 0, 100),
        ParameterSpec("opening.phase_1.crop.STRAWBERRY", "int", 0, 100),
        ParameterSpec("opening.phase_1.crop.TOMATO", "int", 0, 100),
        ParameterSpec("opening.phase_1.crop.WHEAT", "int", 0, 100),
        ParameterSpec("opening.phase_1.animal.COW", "int", 0, 100),
        ParameterSpec("opening.phase_1.animal.GOOSE", "int", 0, 100),
        ParameterSpec("opening.phase_1.animal.SHEEP", "int", 0, 100),
        ParameterSpec("opening.phase_1.structure.coop", "int", 0, 100),
        ParameterSpec("opening.phase_1.structure.pasture", "int", 0, 100),
        ParameterSpec("opening.phase_1.cash_reserve", "int", 0, 5_000, 100),
        ParameterSpec("opening.phase_1.inventory_reserve", "int", 0, 100),
        ParameterSpec("opening.phase_2.start_day", "int", 20, 29),
        ParameterSpec("opening.phase_2.target_hands", "int", 0, 19),
        ParameterSpec("opening.phase_2.target_quadrants", "int", 1, 5),
        ParameterSpec(
            "opening.phase_2.primary_crop",
            "categorical",
            choices=("CARROT", "MELON", "STRAWBERRY", "TOMATO", "WHEAT"),
        ),
        ParameterSpec("opening.phase_2.crop.CARROT", "int", 0, 100),
        ParameterSpec("opening.phase_2.crop.MELON", "int", 0, 100),
        ParameterSpec("opening.phase_2.crop.STRAWBERRY", "int", 0, 100),
        ParameterSpec("opening.phase_2.crop.TOMATO", "int", 0, 100),
        ParameterSpec("opening.phase_2.crop.WHEAT", "int", 0, 100),
        ParameterSpec("opening.phase_2.animal.COW", "int", 0, 100),
        ParameterSpec("opening.phase_2.animal.GOOSE", "int", 0, 100),
        ParameterSpec("opening.phase_2.animal.SHEEP", "int", 0, 100),
        ParameterSpec("opening.phase_2.structure.coop", "int", 0, 100),
        ParameterSpec("opening.phase_2.structure.pasture", "int", 0, 100),
        ParameterSpec("opening.phase_2.cash_reserve", "int", 0, 5_000, 100),
        ParameterSpec("opening.phase_2.inventory_reserve", "int", 0, 100),
        ParameterSpec("jobs.recovery", "float", 0.0, 10.0),
        ParameterSpec("jobs.urgency", "float", 0.0, 10.0),
        ParameterSpec("jobs.distance_penalty", "float", 0.0, 10.0),
        ParameterSpec("jobs.production", "float", 0.0, 10.0),
        ParameterSpec("jobs.transport", "float", 0.0, 10.0),
        ParameterSpec("jobs.structure", "float", 0.0, 10.0),
        ParameterSpec("market.live_price", "float", 0.0, 10.0),
        ParameterSpec("market.town_demand", "float", 0.0, 10.0),
        ParameterSpec("market.opponent_supply", "float", 0.0, 10.0),
        ParameterSpec("market.reserve_penalty", "float", 0.0, 10.0),
        ParameterSpec("market.liquidation_urgency", "float", 0.0, 10.0),
        ParameterSpec("liquidation_start_day", "int", 20, 29),
    )


def test_space_manifest_names_domains_and_hash_are_stable() -> None:
    """The public semantic names and domains remain one stable API."""
    assert SPACE_MANIFEST.parameters == _expected_specs()
    assert "opening.phase_0.start_day" not in {
        spec.name for spec in SPACE_MANIFEST.parameters
    }
    assert len(SPACE_SHA256) == 64


def test_fixed_trial_round_trips_every_semantic_parameter() -> None:
    """Every direct parameter reconstructs the complete default config."""
    expected = HybridConfig.default()
    trial = optuna.trial.FixedTrial(parameters_for_config(expected))
    actual = suggest_config(trial)
    assert actual == expected
    assert parameters_for_config(actual) == parameters_for_config(expected)


def test_margin_genome_is_not_part_of_optuna_space() -> None:
    """The Optuna API exposes policy values rather than legacy genome logits."""
    names = {spec.name for spec in SPACE_MANIFEST.parameters}
    assert all("logit" not in name and "genome" not in name for name in names)
    assert len(names) < 64
