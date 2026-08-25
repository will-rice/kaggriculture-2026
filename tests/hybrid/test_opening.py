"""Contracts for the guarded opening-target derivation."""
# ruff: noqa: D103

from __future__ import annotations

from copy import deepcopy

from kaggle_environments.envs.kaggriculture import kaggriculture as engine

from kaggriculture.constants import LAND_ORDER
from kaggriculture.features import (
    ANIMAL_NAMES,
    CROP_NAMES,
    PRODUCT_NAMES,
    STRUCTURE_KINDS,
    EncodedObservation,
    encode_observation,
)
from kaggriculture.hybrid.opening import opening_targets, select_phase
from kaggriculture.hybrid.runtime import (
    RuntimeConfig,
    RuntimeJobWeights,
    RuntimeMarketWeights,
    RuntimeOpeningPhase,
)
from tests.feature_golden_generator import rich_observation


def phase(
    *,
    start_day: int,
    target_hands: int = 0,
    target_quadrants: int = 1,
    crops: dict[str, int] | None = None,
    animals: dict[str, int] | None = None,
    structures: dict[str, int] | None = None,
    cash_reserve: int = 0,
    inventory_reserve: int = 0,
) -> RuntimeOpeningPhase:
    """Build one canonical-order runtime phase for an opening test."""
    crops = crops or {}
    animals = animals or {}
    structures = structures or {}
    return RuntimeOpeningPhase(
        start_day=start_day,
        target_hands=target_hands,
        target_quadrants=target_quadrants,
        primary_crop=CROP_NAMES[0],
        crop_targets=tuple(crops.get(crop, 0) for crop in CROP_NAMES),
        animal_targets=tuple(animals.get(animal, 0) for animal in ANIMAL_NAMES),
        structure_targets=(
            structures.get(STRUCTURE_KINDS[0], 0),
            structures.get(STRUCTURE_KINDS[1], 0),
        ),
        cash_reserve=cash_reserve,
        inventory_reserve=inventory_reserve,
    )


def runtime_config(
    *,
    phases: tuple[RuntimeOpeningPhase, RuntimeOpeningPhase, RuntimeOpeningPhase],
    liquidation_start_day: int = 26,
) -> RuntimeConfig:
    """Build a fully specified runtime config without authoring dependencies."""
    return RuntimeConfig(
        phases=phases,
        jobs=RuntimeJobWeights(
            recovery=0.0,
            urgency=0.0,
            distance_penalty=0.0,
            production=0.0,
            transport=0.0,
            structure=0.0,
        ),
        market=RuntimeMarketWeights(
            live_price=0.0,
            town_demand=0.0,
            opponent_supply=0.0,
            reserve_penalty=0.0,
            liquidation_urgency=0.0,
        ),
        liquidation_start_day=liquidation_start_day,
    )


def encoded_fixture(
    *,
    day: int,
    hands: int = 0,
    crops: dict[str, int] | None = None,
    land: int = 1,
    field_animals: dict[str, int] | None = None,
    shed_animals: dict[str, int] | None = None,
    carried_animals: dict[str, int] | None = None,
    structures: tuple[str, ...] = (),
    money: int = 3_000,
) -> EncodedObservation:
    """Return an encoded state containing only the requested owned assets."""
    observation = deepcopy(rich_observation())
    farm = observation["farms"][0]
    farm["tiles"] = [[None] * 10 for _ in range(10)]
    farm["hands"] = [[index + 1, 1] for index in range(hands)]
    farm["money"] = money
    farm["unlocked_quadrants"] = ["NW", *LAND_ORDER][:land]
    observation["day"] = day
    observation["hour"] = 0
    observation["private"]["inventories"] = [{} for _ in range(hands + 1)]
    for animal, count in (shed_animals or {}).items():
        observation["private"]["shed"][animal] = count
    for animal, count in (carried_animals or {}).items():
        observation["private"]["inventories"][0][animal] = count

    x = 0
    for crop, count in (crops or {}).items():
        for _ in range(count):
            farm["tiles"][0][x] = engine._new_plant(crop, 0, 0)
            x += 1
    for animal, count in (field_animals or {}).items():
        for _ in range(count):
            farm["tiles"][1][x] = engine._new_animal(animal, 0)
            x += 1
    for kind in structures:
        farm["tiles"][2][x] = {"kind": kind}
        x += 1
    return encode_observation(observation, 0)


def test_opening_targets_are_state_deficits_not_taped_actions() -> None:
    encoded = encoded_fixture(day=5, hands=2, crops={"WHEAT": 3})
    config = runtime_config(
        phases=(
            phase(start_day=0, target_hands=4, crops={"WHEAT": 5}),
            phase(start_day=10),
            phase(start_day=20),
        )
    )

    targets = opening_targets(encoded, config)

    assert targets.hands_needed == 2
    assert targets.crop_deficits["WHEAT"] == 2


def test_latest_phase_does_not_replay_already_satisfied_work() -> None:
    encoded = encoded_fixture(day=8, hands=5, crops={"WHEAT": 8})
    first = phase(start_day=0, target_hands=3, crops={"WHEAT": 4})
    expected = phase(start_day=7, target_hands=5, crops={"WHEAT": 8})
    config = runtime_config(phases=(first, expected, phase(start_day=20)))

    targets = opening_targets(encoded, config)

    assert select_phase(encoded, config) == expected
    assert targets.hands_needed == 0
    assert targets.crop_deficits["WHEAT"] == 0


def test_delayed_land_target_counts_already_unlocked_quadrants() -> None:
    encoded = encoded_fixture(day=8, land=2)
    config = runtime_config(
        phases=(
            phase(start_day=0, target_quadrants=1),
            phase(start_day=7, target_quadrants=3),
            phase(start_day=20),
        )
    )

    targets = opening_targets(encoded, config)

    assert targets.phase_start_day == 7
    assert targets.quadrants_needed == 1


def test_animal_and_structure_deficits_include_existing_owned_assets() -> None:
    encoded = encoded_fixture(
        day=5,
        field_animals={"GOOSE": 1},
        shed_animals={"GOOSE": 1},
        carried_animals={"GOOSE": 1},
        structures=("COOP", "PASTURE"),
    )
    config = runtime_config(
        phases=(
            phase(
                start_day=0,
                animals={"GOOSE": 3},
                structures={"COOP": 2, "PASTURE": 1},
            ),
            phase(start_day=10),
            phase(start_day=20),
        )
    )

    targets = opening_targets(encoded, config)

    assert targets.animal_deficits["GOOSE"] == 0
    assert targets.structure_deficits == {"COOP": 0, "PASTURE": 0}


def test_reserves_remain_explicit_when_current_cash_cannot_protect_them() -> None:
    encoded = encoded_fixture(day=5, money=200)
    config = runtime_config(
        phases=(
            phase(start_day=0, cash_reserve=500, inventory_reserve=3),
            phase(start_day=10),
            phase(start_day=20),
        )
    )

    targets = opening_targets(encoded, config)

    assert targets.protected_cash == 500
    assert targets.protected_inventory == dict.fromkeys(PRODUCT_NAMES, 3)


def test_liquidation_zeros_all_future_acquisition_and_build_deficits() -> None:
    encoded = encoded_fixture(day=26, hands=0)
    config = runtime_config(
        phases=(
            phase(start_day=0, target_hands=4, target_quadrants=3, crops={"WHEAT": 5}),
            phase(start_day=10),
            phase(
                start_day=20,
                target_hands=8,
                target_quadrants=4,
                crops={"WHEAT": 10},
                animals={"GOOSE": 3},
                structures={"COOP": 2},
                cash_reserve=700,
                inventory_reserve=6,
            ),
        ),
    )

    targets = opening_targets(encoded, config)

    assert targets.liquidating is True
    assert targets.hands_needed == 0
    assert targets.quadrants_needed == 0
    assert targets.crop_deficits == dict.fromkeys(CROP_NAMES, 0)
    assert targets.animal_deficits == dict.fromkeys(ANIMAL_NAMES, 0)
    assert targets.structure_deficits == dict.fromkeys(STRUCTURE_KINDS, 0)
    assert targets.protected_cash == 700
    assert targets.protected_inventory == dict.fromkeys(PRODUCT_NAMES, 6)
