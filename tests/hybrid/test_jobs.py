"""Behavior contracts for reactive, legal hybrid unit jobs."""
# ruff: noqa: D103

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import replace

from kaggle_environments.envs.kaggriculture import kaggriculture as engine

from kaggriculture.action_codec import QUANTITIES, UNIT_OPS
from kaggriculture.features import (
    ANIMAL_NAMES,
    CROP_NAMES,
    PRODUCT_NAMES,
    STRUCTURE_KINDS,
    EncodedObservation,
    encode_observation,
)
from kaggriculture.hybrid.config import HybridConfig, to_runtime
from kaggriculture.hybrid.jobs import UnitSelection, collect_jobs, select_units
from kaggriculture.hybrid.opening import OpeningTargets
from kaggriculture.hybrid.runtime import RuntimeConfig
from tests.feature_golden_generator import rich_observation


def runtime_config() -> RuntimeConfig:
    return to_runtime(HybridConfig.default())


def targets_fixture(
    *,
    crop_deficits: dict[str, int] | None = None,
    animal_deficits: dict[str, int] | None = None,
    structure_deficits: dict[str, int] | None = None,
    protected_inventory: dict[str, int] | None = None,
    liquidating: bool = False,
) -> OpeningTargets:
    return OpeningTargets(
        phase_start_day=0,
        hands_needed=0,
        quadrants_needed=0,
        crop_deficits={name: (crop_deficits or {}).get(name, 0) for name in CROP_NAMES},
        animal_deficits={
            name: (animal_deficits or {}).get(name, 0) for name in ANIMAL_NAMES
        },
        structure_deficits={
            name: (structure_deficits or {}).get(name, 0) for name in STRUCTURE_KINDS
        },
        protected_cash=0,
        protected_inventory={
            name: (protected_inventory or {}).get(name, 0) for name in PRODUCT_NAMES
        },
        liquidating=liquidating,
    )


def _plant(
    crop: str,
    *,
    day: int,
    planted_day: int = 0,
    watered: bool = False,
    distress: int = 0,
    yield_units: int = 0,
) -> dict[str, object]:
    tile = engine._new_plant(crop, planted_day, planted_day * 24)
    tile["watered_today"] = watered
    tile["consecutive_unwatered"] = distress
    tile["yield_units"] = yield_units
    if day >= 29:
        tile["max_lifespan_step"] = 720
    return tile


def _animal(
    animal: str,
    *,
    fed: bool = False,
    cared: bool = False,
    distress: int = 0,
    yield_units: int = 0,
) -> dict[str, object]:
    tile = engine._new_animal(animal, 0)
    tile["fed_today"] = fed
    tile["cared_today"] = cared
    tile["consecutive_unfed"] = distress
    tile["yield_units"] = yield_units
    return tile


def encoded_fixture(
    *,
    units: tuple[tuple[int, int], ...] = ((4, 4),),
    day: int = 3,
    weeds: tuple[tuple[int, int], ...] = (),
    animals: dict[tuple[int, int], dict[str, object]] | None = None,
    crops: dict[tuple[int, int], dict[str, object]] | None = None,
    structures: dict[tuple[int, int], str] | None = None,
    seeds: dict[str, int] | None = None,
    shed: dict[str, int] | None = None,
    inventories: tuple[dict[str, int], ...] | None = None,
) -> EncodedObservation:
    observation = deepcopy(rich_observation())
    farm = observation["farms"][0]
    farm["tiles"] = [[None] * 10 for _ in range(10)]
    farm["farmer"] = list(units[0])
    farm["hands"] = [list(position) for position in units[1:]]
    farm["unlocked_quadrants"] = ["NW", "NE", "SW", "SE"]
    observation["day"] = day
    observation["hour"] = 0
    private = engine._new_private()
    private["inventories"] = [
        dict(items) for items in (inventories or ({},) * len(units))
    ]
    for item, count in (seeds or {}).items():
        private["seeds"][item] = count
    for item, count in (shed or {}).items():
        private["shed"][item] = count
    observation["private"] = private
    for x, y in weeds:
        farm["tiles"][y][x] = {"kind": "WEED"}
    for (x, y), tile in (animals or {}).items():
        farm["tiles"][y][x] = tile
    for (x, y), tile in (crops or {}).items():
        farm["tiles"][y][x] = tile
    for (x, y), kind in (structures or {}).items():
        farm["tiles"][y][x] = {"kind": kind}
    return encode_observation(observation, 0)


def _ops(selected: UnitSelection) -> tuple[str, ...]:
    return tuple(UNIT_OPS[index] for index in selected.operation_indices)


def _quantities(selected: UnitSelection) -> tuple[int, ...]:
    return tuple(QUANTITIES[index] for index in selected.quantity_indices)


def test_weeds_and_at_risk_animals_preempt_new_planting() -> None:
    encoded = encoded_fixture(
        units=((4, 4), (1, 1)),
        weeds=((4, 4),),
        animals={(1, 1): _animal("GOOSE", distress=1)},
        seeds={"WHEAT": 5},
        inventories=({}, {"WHEAT": 1}),
    )
    targets = targets_fixture(crop_deficits={"WHEAT": 3})

    selected = select_units(encoded, targets, runtime_config())

    assert _ops(selected) == ("DIG", "FEED")


def test_assignment_is_nearest_unit_then_stable_unit_index() -> None:
    encoded = encoded_fixture(units=((0, 0), (2, 0)), weeds=((1, 0),))

    first = select_units(encoded, targets_fixture(), runtime_config())

    assert first == select_units(encoded, targets_fixture(), runtime_config())
    assert _ops(first) == ("EAST", "PASS")


def test_assignment_recomputes_nearest_pair_after_each_unit_claim() -> None:
    encoded = encoded_fixture(
        units=((0, 0), (9, 9)),
        weeds=((0, 0), (1, 0), (9, 8)),
    )

    selected = select_units(encoded, targets_fixture(), runtime_config())

    assert _ops(selected) == ("DIG", "NORTH")


def test_ranked_steps_at_board_boundary_keep_only_in_bounds_fallbacks() -> None:
    from kaggriculture.actions import ranked_steps_toward

    encoded = encoded_fixture(units=((0, 0),), weeds=((0, 2),))
    choices = tuple(choice[0] for choice in ranked_steps_toward((0, 0), (0, 2)))
    canonical_legal = tuple(
        name for name in choices if encoded.unit_mask[0][UNIT_OPS.index(name)]
    )

    assert choices == ("SOUTH", "EAST")
    assert canonical_legal == choices
    assert _ops(select_units(encoded, targets_fixture(), runtime_config())) == (
        "SOUTH",
    )


def test_bulk_feed_pickup_uses_need_without_spending_protected_inventory() -> None:
    encoded = encoded_fixture(
        units=((4, 4),),
        animals={
            (0, 0): _animal("GOOSE"),
            (1, 0): _animal("GOOSE"),
            (2, 0): _animal("GOOSE"),
        },
        shed={"WHEAT": 11},
    )
    targets = targets_fixture(protected_inventory={"WHEAT": 8})

    selected = select_units(encoded, targets, runtime_config())

    assert _ops(selected) == ("PICKUP:WHEAT",)
    assert _quantities(selected) == (3,)


def test_carried_feed_is_not_hauled_away_from_an_unfed_animal() -> None:
    encoded = encoded_fixture(
        units=((4, 4),),
        animals={(4, 4): _animal("GOOSE")},
        inventories=({"WHEAT": 12},),
    )

    selected = select_units(encoded, targets_fixture(), runtime_config())

    assert _ops(selected) == ("FEED",)


def test_bulk_place_uses_unit_carry_and_remaining_shed_room() -> None:
    encoded = encoded_fixture(
        units=((4, 4),),
        shed={"CARROT": 95},
        inventories=({"WHEAT": 8},),
    )

    selected = select_units(encoded, targets_fixture(), runtime_config())

    assert _ops(selected) == ("PLACE:WHEAT",)
    assert _quantities(selected) == (5,)


def test_displaced_worker_replans_route_then_places_its_animal() -> None:
    targets = targets_fixture()
    displaced = encoded_fixture(
        units=((0, 0),),
        structures={(3, 0): "PASTURE"},
        inventories=({"COW": 1},),
    )
    arrived = encoded_fixture(
        units=((3, 0),),
        structures={(3, 0): "PASTURE"},
        inventories=({"COW": 1},),
    )

    assert _ops(select_units(displaced, targets, runtime_config())) == ("EAST",)
    assert _ops(select_units(arrived, targets, runtime_config())) == ("PLACE:COW",)


def test_full_shed_and_missing_requested_seed_recover_with_pass() -> None:
    full_shed = encoded_fixture(
        units=((4, 4),),
        shed={"CARROT": 100},
        inventories=({"WHEAT": 4},),
    )
    missing_seed = encoded_fixture(units=((0, 0),), seeds={"CARROT": 5})

    assert _ops(select_units(full_shed, targets_fixture(), runtime_config())) == (
        "PASS",
    )
    assert _ops(
        select_units(
            missing_seed,
            targets_fixture(crop_deficits={"WHEAT": 2}),
            runtime_config(),
        )
    ) == ("PASS",)


def test_waiting_animal_is_picked_up_and_placed_only_in_matching_structure() -> None:
    at_shed = encoded_fixture(
        units=((4, 4),), shed={"COW": 1}, structures={(2, 2): "PASTURE"}
    )
    wrong_home = encoded_fixture(
        units=((2, 2),),
        structures={(2, 2): "COOP"},
        inventories=({"COW": 1},),
    )

    pickup = select_units(at_shed, targets_fixture(), runtime_config())

    assert _ops(pickup) == ("PICKUP:COW",)
    assert _quantities(pickup) == (1,)
    assert _ops(select_units(wrong_home, targets_fixture(), runtime_config())) == (
        "PASS",
    )


def test_two_waiting_animals_do_not_claim_the_same_bare_structure() -> None:
    encoded = encoded_fixture(
        units=((4, 4), (5, 4)),
        shed={"COW": 1, "SHEEP": 1},
        structures={(2, 2): "PASTURE"},
    )

    jobs = collect_jobs(encoded, targets_fixture(), runtime_config())
    pickups = [
        UNIT_OPS[job.operation_index]
        for job in jobs
        if UNIT_OPS[job.operation_index].startswith("PICKUP:")
    ]

    assert pickups == ["PICKUP:COW"]


def test_care_water_and_harvest_are_emitted_from_live_tile_urgency() -> None:
    encoded = encoded_fixture(
        units=((0, 0), (1, 0), (2, 0)),
        crops={
            (0, 0): _plant("WHEAT", day=3, distress=1),
            (2, 0): _plant("WHEAT", day=3, watered=True, yield_units=4),
        },
        animals={(1, 0): _animal("GOOSE", fed=True, cared=False)},
    )

    selected = select_units(encoded, targets_fixture(), runtime_config())

    assert _ops(selected) == ("WATER", "CARE", "HARVEST")


def test_healthy_crop_outside_its_bonus_window_does_not_waste_a_water_turn() -> None:
    encoded = encoded_fixture(
        units=((0, 0),),
        day=0,
        crops={(0, 0): _plant("WHEAT", day=0)},
    )

    selected = select_units(encoded, targets_fixture(), runtime_config())

    assert _ops(selected) == ("PASS",)


def test_final_day_harvest_and_haul_preempt_new_production() -> None:
    encoded = encoded_fixture(
        units=((0, 0), (4, 4)),
        day=29,
        crops={(0, 0): _plant("WHEAT", day=29, yield_units=4)},
        seeds={"WHEAT": 5},
        inventories=({}, {"WHEAT": 6}),
    )

    selected = select_units(
        encoded,
        targets_fixture(crop_deficits={"WHEAT": 3}, liquidating=True),
        runtime_config(),
    )

    assert _ops(selected) == ("HARVEST", "PLACE:WHEAT")
    assert _quantities(selected) == (1, 6)


def test_structure_and_crop_deficits_claim_distinct_empty_tiles() -> None:
    encoded = encoded_fixture(units=((0, 0), (1, 0)), seeds={"WHEAT": 1})

    selected = select_units(
        encoded,
        targets_fixture(crop_deficits={"WHEAT": 1}, structure_deficits={"PASTURE": 1}),
        runtime_config(),
    )

    assert set(_ops(selected)) == {"BUILD_PASTURE", "PLANT:WHEAT"}


def test_distance_penalty_can_prefer_useful_local_work() -> None:
    encoded = encoded_fixture(
        units=((0, 0),),
        crops={(9, 0): _plant("WHEAT", day=3, watered=True, yield_units=4)},
        seeds={"WHEAT": 1},
    )
    config = runtime_config()
    config = replace(config, jobs=replace(config.jobs, distance_penalty=10.0))

    selected = select_units(
        encoded,
        targets_fixture(crop_deficits={"WHEAT": 1}),
        config,
    )

    assert _ops(selected) == ("PLANT:WHEAT",)


def test_collect_jobs_are_typed_sorted_desires_not_raw_engine_actions() -> None:
    encoded = encoded_fixture(units=((0, 0),), weeds=((0, 0),))

    jobs = collect_jobs(encoded, targets_fixture(), runtime_config())

    assert jobs == tuple(sorted(jobs))
    assert jobs[0].position == (0, 0)
    assert UNIT_OPS[jobs[0].operation_index] == "DIG"


def test_selection_is_deterministic_and_mask_legal_across_varied_states() -> None:
    rng = random.Random(821)
    config = runtime_config()
    for _ in range(50):
        positions = tuple((rng.randrange(10), rng.randrange(10)) for _ in range(3))
        weed = (rng.randrange(10), rng.randrange(10))
        encoded = encoded_fixture(
            units=positions,
            weeds=(weed,),
            seeds={"WHEAT": rng.randrange(4)},
            shed={"WHEAT": rng.randrange(13)},
            inventories=tuple(
                {"WHEAT": count} if count else {}
                for count in (rng.randrange(5), rng.randrange(5), rng.randrange(5))
            ),
        )
        targets = targets_fixture(crop_deficits={"WHEAT": rng.randrange(4)})

        selected = select_units(encoded, targets, config)

        assert selected == select_units(encoded, targets, config)
        assert all(
            encoded.unit_mask[unit][operation] and encoded.quantity_mask[unit][quantity]
            for unit, (operation, quantity) in enumerate(
                zip(
                    selected.operation_indices,
                    selected.quantity_indices,
                    strict=True,
                )
            )
        )
