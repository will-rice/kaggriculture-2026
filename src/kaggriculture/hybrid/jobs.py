"""Reactive desired jobs and deterministic legal unit assignment."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from kaggriculture.action_codec import MAX_TRANSFER, QUANTITIES, UNIT_OPS
from kaggriculture.actions import distance, step_toward
from kaggriculture.constants import (
    ANIMALS,
    BOARD_SIZE,
    CROPS,
    LAST_DAY,
    SEASON_DAYS,
    SHED_CAPACITY,
)
from kaggriculture.features import (
    ANIMAL_NAMES,
    CROP_NAMES,
    PRODUCT_NAMES,
    SHED_NAMES,
    STRUCTURE_KINDS,
    EncodedObservation,
)
from kaggriculture.hybrid.opening import OpeningTargets
from kaggriculture.hybrid.runtime import RuntimeConfig

_OP = {name: index for index, name in enumerate(UNIT_OPS)}
_MOVES = tuple(name for name in ("NORTH", "SOUTH", "EAST", "WEST"))
_SHED_ACCESS = ((4, 4), (5, 4), (4, 5), (5, 5))

_FINAL_HAUL = 0
_FINAL_HARVEST = 1
_WEED = 2
_RISK_FEED = 3
_RISK_WATER = 4
_PLACE_ANIMAL = 5
_PICKUP_ANIMAL = 6
_FEED = 7
_WATER = 8
_HARVEST = 9
_CARE = 10
_COLLECT = 11
_TRANSPORT = 12
_BUILD = 13
_PLANT = 14
_KINDS = 15

_FINAL_PRIORITY = -1_000_000.0
_RISK_PRIORITY = -100_000.0


@dataclass(frozen=True, order=True)
class Job:
    """One scored desired operation at a board position."""

    priority: float
    distance: int
    stable_order: int
    position: tuple[int, int]
    operation_index: int
    quantity_index: int
    bound_unit: int | None = None


@dataclass(frozen=True)
class UnitSelection:
    """One legal operation and quantity vocabulary index per acting unit."""

    operation_indices: tuple[int, ...]
    quantity_indices: tuple[int, ...]


def _positions(encoded: EncodedObservation, plane: str) -> tuple[tuple[int, int], ...]:
    return tuple(
        (x, y)
        for y in range(BOARD_SIZE)
        for x in range(BOARD_SIZE)
        if encoded.plane(plane, x, y) != 0.0
    )


def _nearest_distance(
    encoded: EncodedObservation,
    position: tuple[int, int],
    bound_unit: int | None,
) -> int:
    if bound_unit is not None:
        return distance(encoded.unit_position(bound_unit), position)
    return min(
        distance(encoded.unit_position(unit), position) for unit in range(encoded.units)
    )


def _quantity_index(
    encoded: EncodedObservation, useful: int, bound_unit: int | None = None
) -> int:
    """Return the largest useful legal transfer bucket, never bucket zero."""
    unit = 0 if bound_unit is None else bound_unit
    limit = min(max(useful, 1), MAX_TRANSFER)
    candidates = [
        index
        for index, quantity in enumerate(QUANTITIES)
        if 0 < quantity <= limit and encoded.quantity_mask[unit][index]
    ]
    if candidates:
        return max(candidates, key=lambda index: QUANTITIES[index])
    return next(
        index for index, legal in enumerate(encoded.quantity_mask[unit]) if legal
    )


def _priority(score: float, *, final: bool = False, risk: bool = False) -> float:
    if final:
        return _FINAL_PRIORITY - score
    if risk:
        return _RISK_PRIORITY - score
    return -score


def _job(
    encoded: EncodedObservation,
    config: RuntimeConfig,
    *,
    kind: int,
    position: tuple[int, int],
    operation: str,
    score: float,
    useful_quantity: int = 1,
    bound_unit: int | None = None,
    final: bool = False,
    risk: bool = False,
) -> Job:
    yx_kind = ((position[1] * BOARD_SIZE + position[0]) * _KINDS) + kind
    locality = _nearest_distance(encoded, position, bound_unit)
    return Job(
        priority=_priority(score, final=final, risk=risk)
        + config.jobs.distance_penalty * locality,
        distance=locality,
        stable_order=yx_kind,
        position=position,
        operation_index=_OP[operation],
        quantity_index=_quantity_index(encoded, useful_quantity, bound_unit),
        bound_unit=bound_unit,
    )


def _crop_yield(encoded: EncodedObservation, crop: str, x: int, y: int) -> int:
    return round(
        encoded.plane("feature:YIELD_FRACTION", x, y) * int(CROPS[crop]["max_yield"])
    )


def _animal_yield(encoded: EncodedObservation, animal: str, x: int, y: int) -> int:
    return round(
        encoded.plane("feature:YIELD_FRACTION", x, y) * int(ANIMALS[animal]["max_held"])
    )


def _mature(encoded: EncodedObservation, crop: str, x: int, y: int) -> bool:
    age = round(encoded.plane("feature:AGE", x, y) * SEASON_DAYS)
    return age >= int(CROPS[crop]["first_yield_day"])


def _needs_water(
    encoded: EncodedObservation, crop: str, x: int, y: int, distress: float
) -> bool:
    if distress > 0.0:
        return True
    if bool(CROPS[crop]["ongoing"]):
        return False
    age = round(encoded.plane("feature:AGE", x, y) * SEASON_DAYS)
    start = (int(CROPS[crop]["max_yield_day"]) + 1) // 2
    return start <= age <= int(CROPS[crop]["max_yield_day"])


def _empty_sites(encoded: EncodedObservation) -> list[tuple[int, int]]:
    units = tuple(encoded.unit_position(unit) for unit in range(encoded.units))
    return sorted(
        _positions(encoded, "state:EMPTY"),
        key=lambda position: (
            min(distance(unit, position) for unit in units),
            min(distance(access, position) for access in _SHED_ACCESS),
            position[1],
            position[0],
        ),
    )


def _bare_structures(encoded: EncodedObservation, kind: str) -> list[tuple[int, int]]:
    return sorted(
        _positions(encoded, f"state:{kind}"),
        key=lambda position: (
            min(distance(access, position) for access in _SHED_ACCESS),
            position[1],
            position[0],
        ),
    )


def _animal_recovery_jobs(
    encoded: EncodedObservation, config: RuntimeConfig
) -> list[Job]:
    jobs: list[Job] = []
    claimed_homes: set[tuple[int, int]] = set()
    waiting_homes: dict[str, list[tuple[int, int]]] = {
        animal: _bare_structures(encoded, str(ANIMALS[animal]["structure"]))
        for animal in ANIMAL_NAMES
    }
    for unit in range(encoded.units):
        animal = next(
            (
                name
                for name in ANIMAL_NAMES
                if encoded.unit_carried_count(unit, name) > 0
            ),
            None,
        )
        if animal is None:
            continue
        homes = [home for home in waiting_homes[animal] if home not in claimed_homes]
        if not homes:
            continue
        source = encoded.unit_position(unit)
        home = min(homes, key=lambda pos: (distance(source, pos), pos[1], pos[0]))
        claimed_homes.add(home)
        jobs.append(
            _job(
                encoded,
                config,
                kind=_PLACE_ANIMAL,
                position=home,
                operation=f"PLACE:{animal}",
                score=config.jobs.recovery + config.jobs.transport,
                bound_unit=unit,
            )
        )

    for animal in ANIMAL_NAMES:
        homes = [home for home in waiting_homes[animal] if home not in claimed_homes]
        waiting = min(encoded.shed_count(animal), len(homes))
        for offset, home in enumerate(homes[:waiting]):
            claimed_homes.add(home)
            access = _SHED_ACCESS[offset % len(_SHED_ACCESS)]
            jobs.append(
                _job(
                    encoded,
                    config,
                    kind=_PICKUP_ANIMAL,
                    position=access,
                    operation=f"PICKUP:{animal}",
                    score=config.jobs.recovery + config.jobs.transport,
                )
            )
    return jobs


def _animal_work_jobs(
    encoded: EncodedObservation,
    targets: OpeningTargets,
    config: RuntimeConfig,
) -> list[Job]:
    jobs: list[Job] = []
    unfed: list[tuple[tuple[int, int], float]] = []
    for animal in ANIMAL_NAMES:
        for position in _positions(encoded, f"animal:{animal}"):
            x, y = position
            distress = encoded.distress_at(x, y)
            fed = encoded.plane("feature:ACTIVE_TODAY", x, y) != 0.0
            cared = encoded.plane("feature:CARED_TODAY", x, y) != 0.0
            held_yield = _animal_yield(encoded, animal, x, y)
            if held_yield:
                jobs.append(
                    _job(
                        encoded,
                        config,
                        kind=_HARVEST,
                        position=position,
                        operation="HARVEST",
                        score=config.jobs.production + config.jobs.urgency * held_yield,
                        final=encoded.day_count() >= LAST_DAY,
                    )
                )
            if encoded.day_count() < LAST_DAY and not fed:
                unfed.append((position, distress))
            if encoded.day_count() < LAST_DAY and not cared:
                bonus = encoded.plane("feature:CARE_BONUS", x, y)
                jobs.append(
                    _job(
                        encoded,
                        config,
                        kind=_CARE,
                        position=position,
                        operation="CARE",
                        score=config.jobs.production
                        + config.jobs.urgency * (1.0 + bonus),
                    )
                )
            if encoded.plane("feature:BONUS_READY", x, y) != 0.0:
                jobs.append(
                    _job(
                        encoded,
                        config,
                        kind=_COLLECT,
                        position=position,
                        operation="COLLECT_FERTILIZER",
                        score=config.jobs.production,
                    )
                )

    available_carriers = [
        unit
        for unit in range(encoded.units)
        if encoded.unit_carried_count(unit, "WHEAT") > 0
    ]
    for position, distress in sorted(
        unfed, key=lambda value: (-value[1], value[0][1], value[0][0])
    ):
        if not available_carriers:
            break
        unit = min(
            available_carriers,
            key=lambda candidate: (
                distance(encoded.unit_position(candidate), position),
                candidate,
            ),
        )
        available_carriers.remove(unit)
        jobs.append(
            _job(
                encoded,
                config,
                kind=_RISK_FEED if distress > 0.0 else _FEED,
                position=position,
                operation="FEED",
                score=config.jobs.recovery + config.jobs.urgency * (1.0 + distress),
                bound_unit=unit,
                risk=distress > 0.0,
            )
        )

    feed_need = max(len(unfed) - encoded.carried_item_count("WHEAT"), 0)
    reserve = targets.protected_inventory.get("WHEAT", 0)
    available = max(encoded.shed_count("WHEAT") - reserve, 0)
    useful = min(feed_need, available, MAX_TRANSFER)
    if useful:
        jobs.append(
            _job(
                encoded,
                config,
                kind=_RISK_FEED if any(risk > 0.0 for _, risk in unfed) else _FEED,
                position=_SHED_ACCESS[0],
                operation="PICKUP:WHEAT",
                score=config.jobs.recovery
                + config.jobs.transport
                + config.jobs.urgency * max((risk for _, risk in unfed), default=0.0),
                useful_quantity=useful,
                risk=any(risk > 0.0 for _, risk in unfed),
            )
        )
    return jobs


def _crop_work_jobs(encoded: EncodedObservation, config: RuntimeConfig) -> list[Job]:
    jobs: list[Job] = []
    final_day = encoded.day_count() >= LAST_DAY
    for crop in CROP_NAMES:
        for position in _positions(encoded, f"crop:{crop}"):
            x, y = position
            distress = encoded.distress_at(x, y)
            watered = encoded.plane("feature:ACTIVE_TODAY", x, y) != 0.0
            held_yield = _crop_yield(encoded, crop, x, y)
            harvestable = held_yield > 0 and _mature(encoded, crop, x, y)
            if (
                not watered
                and not final_day
                and _needs_water(encoded, crop, x, y, distress)
            ):
                jobs.append(
                    _job(
                        encoded,
                        config,
                        kind=_RISK_WATER if distress > 0.0 else _WATER,
                        position=position,
                        operation="WATER",
                        score=config.jobs.production
                        + config.jobs.urgency * (1.0 + distress),
                        risk=distress > 0.0,
                    )
                )
            elif harvestable:
                jobs.append(
                    _job(
                        encoded,
                        config,
                        kind=_FINAL_HARVEST if final_day else _HARVEST,
                        position=position,
                        operation="HARVEST",
                        score=config.jobs.production + config.jobs.urgency * held_yield,
                        final=final_day,
                    )
                )
    return jobs


def _transport_jobs(encoded: EncodedObservation, config: RuntimeConfig) -> list[Job]:
    room = max(SHED_CAPACITY - sum(encoded.shed_count(item) for item in SHED_NAMES), 0)
    jobs: list[Job] = []
    final_day = encoded.day_count() >= LAST_DAY
    feed_due = not final_day and any(
        encoded.plane("feature:ACTIVE_TODAY", x, y) == 0.0
        for animal in ANIMAL_NAMES
        for x, y in _positions(encoded, f"animal:{animal}")
    )
    for unit in range(encoded.units):
        if room <= 0:
            break
        source = encoded.unit_position(unit)
        access = min(
            _SHED_ACCESS,
            key=lambda position: (distance(source, position), position[1], position[0]),
        )
        for product in PRODUCT_NAMES:
            if product == "WHEAT" and feed_due:
                continue
            carried = encoded.unit_carried_count(unit, product)
            useful = min(carried, room, MAX_TRANSFER)
            if useful <= 0:
                continue
            jobs.append(
                _job(
                    encoded,
                    config,
                    kind=_FINAL_HAUL if final_day else _TRANSPORT,
                    position=access,
                    operation=f"PLACE:{product}",
                    score=config.jobs.transport * useful,
                    useful_quantity=useful,
                    bound_unit=unit,
                    final=final_day,
                )
            )
            room -= useful
            break
    return jobs


def _development_jobs(
    encoded: EncodedObservation,
    targets: OpeningTargets,
    config: RuntimeConfig,
) -> list[Job]:
    if targets.liquidating:
        return []
    jobs: list[Job] = []
    sites = _empty_sites(encoded)
    for kind in STRUCTURE_KINDS:
        wanted = min(targets.structure_deficits.get(kind, 0), len(sites))
        for _ in range(wanted):
            position = sites.pop(0)
            jobs.append(
                _job(
                    encoded,
                    config,
                    kind=_BUILD,
                    position=position,
                    operation=f"BUILD_{kind}",
                    score=config.jobs.structure * (1.0 + wanted),
                )
            )
    for crop in CROP_NAMES:
        wanted = min(
            targets.crop_deficits.get(crop, 0),
            encoded.seed_count(crop),
            len(sites),
        )
        for _ in range(wanted):
            position = sites.pop(0)
            jobs.append(
                _job(
                    encoded,
                    config,
                    kind=_PLANT,
                    position=position,
                    operation=f"PLANT:{crop}",
                    score=config.jobs.production * (1.0 + wanted),
                )
            )
    return jobs


def collect_jobs(
    encoded: EncodedObservation,
    targets: OpeningTargets,
    config: RuntimeConfig,
) -> tuple[Job, ...]:
    """Collect current desired categories, independently of past turn plans."""
    jobs: list[Job] = []
    for position in _positions(encoded, "state:WEED"):
        jobs.append(
            _job(
                encoded,
                config,
                kind=_WEED,
                position=position,
                operation="DIG",
                score=config.jobs.recovery + config.jobs.urgency,
                risk=True,
            )
        )
    jobs.extend(_animal_work_jobs(encoded, targets, config))
    jobs.extend(_crop_work_jobs(encoded, config))
    jobs.extend(_animal_recovery_jobs(encoded, config))
    jobs.extend(_transport_jobs(encoded, config))
    jobs.extend(_development_jobs(encoded, targets, config))
    return tuple(sorted(jobs))


def _legal_move_choices(
    encoded: EncodedObservation, unit: int, target: tuple[int, int]
) -> tuple[int, ...]:
    source = encoded.unit_position(unit)
    preferred = step_toward(source, target)[0]
    candidates: list[tuple[int, int, int]] = []
    for name in _MOVES:
        operation = _OP[name]
        if not encoded.unit_mask[unit][operation]:
            continue
        x, y = source
        if name == "NORTH":
            y -= 1
        elif name == "SOUTH":
            y += 1
        elif name == "EAST":
            x += 1
        else:
            x -= 1
        candidates.append((distance((x, y), target), name != preferred, operation))
    return tuple(operation for _, _, operation in sorted(candidates))


def _unit_can_take(encoded: EncodedObservation, unit: int, job: Job) -> bool:
    if job.bound_unit is not None and job.bound_unit != unit:
        return False
    if not encoded.quantity_mask[unit][job.quantity_index]:
        return False
    if encoded.unit_position(unit) == job.position:
        return encoded.unit_mask[unit][job.operation_index]
    return bool(_legal_move_choices(encoded, unit, job.position))


def assign_nearest_legal(
    encoded: EncodedObservation, jobs: Iterable[Job]
) -> dict[int, Job]:
    """Greedily bind ranked jobs to the nearest available capable unit."""
    assigned: dict[int, Job] = {}
    for job in jobs:
        candidates = [
            unit
            for unit in range(encoded.units)
            if unit not in assigned and _unit_can_take(encoded, unit, job)
        ]
        if not candidates:
            continue
        unit = min(
            candidates,
            key=lambda candidate: (
                distance(encoded.unit_position(candidate), job.position),
                candidate,
            ),
        )
        assigned[unit] = replace(
            job, distance=distance(encoded.unit_position(unit), job.position)
        )
    return assigned


def _fallback_quantity(encoded: EncodedObservation, unit: int) -> int:
    one = QUANTITIES.index(1)
    if encoded.quantity_mask[unit][one]:
        return one
    return next(
        index for index, legal in enumerate(encoded.quantity_mask[unit]) if legal
    )


def select_units(
    encoded: EncodedObservation,
    targets: OpeningTargets,
    config: RuntimeConfig,
) -> UnitSelection:
    """Select one deterministic legal operation and quantity for every unit."""
    assignments = assign_nearest_legal(encoded, collect_jobs(encoded, targets, config))
    operations: list[int] = []
    quantities: list[int] = []
    for unit in range(encoded.units):
        assignment = assignments.get(unit)
        candidates: list[tuple[int, int]] = []
        if assignment is not None:
            if encoded.unit_position(unit) == assignment.position:
                candidates.append(
                    (assignment.operation_index, assignment.quantity_index)
                )
            else:
                candidates.extend(
                    (operation, _fallback_quantity(encoded, unit))
                    for operation in _legal_move_choices(
                        encoded, unit, assignment.position
                    )
                )
        candidates.append((_OP["PASS"], _fallback_quantity(encoded, unit)))
        operation, quantity = next(
            candidate
            for candidate in candidates
            if encoded.unit_mask[unit][candidate[0]]
            and encoded.quantity_mask[unit][candidate[1]]
        )
        operations.append(operation)
        quantities.append(quantity)
    return UnitSelection(tuple(operations), tuple(quantities))
