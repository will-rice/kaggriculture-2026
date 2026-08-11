"""Pure-Torch PLANT guard and serialized unit-action phases."""

from dataclasses import fields

import torch

from kaggriculture.constants import ANIMALS, CROPS, SHED_CAPACITY, TURNS_PER_DAY
from kaggriculture.learn.encoding import MAX_UNITS, UNIT_OPS
from kaggriculture.sim.state import (
    ANIMAL_NAMES,
    CROP_NAMES,
    SHED_NAMES,
    SimState,
)
from kaggriculture.sim.tensors import tensor_constant

_OP = {name: index for index, name in enumerate(UNIT_OPS)}
_OP_CROP = tuple(
    CROP_NAMES.index(name.removeprefix("PLANT:")) if name.startswith("PLANT:") else -1
    for name in UNIT_OPS
)
_FERTILIZER = SHED_NAMES.index("FERTILIZER")
_WHEAT = SHED_NAMES.index("WHEAT")


def _clone(state: SimState) -> SimState:
    return SimState(
        **{field.name: getattr(state, field.name).clone() for field in fields(state)}
    )


def _planes(state: SimState, name: str) -> torch.Tensor:
    return getattr(state, name).reshape(state.batch_size * 2, 100)


def _gather(state: SimState, name: str, position: torch.Tensor) -> torch.Tensor:
    return _planes(state, name).gather(1, position[:, None]).squeeze(1)


def _set(
    state: SimState,
    name: str,
    position: torch.Tensor,
    mask: torch.Tensor,
    value: torch.Tensor | int | bool,
) -> None:
    plane = _planes(state, name)
    current = plane.gather(1, position[:, None]).squeeze(1)
    if isinstance(value, torch.Tensor):
        replacement = value.to(dtype=plane.dtype, device=plane.device)
        if replacement.ndim == 0:
            replacement = replacement.expand_as(current)
    else:
        replacement = torch.full_like(current, value)
    plane.scatter_(
        1, position[:, None], torch.where(mask, replacement, current)[:, None]
    )


def _clear(
    state: SimState, position: torch.Tensor, mask: torch.Tensor, kind: int
) -> None:
    _set(state, "kind", position, mask, kind)
    for name in ("occupant", "crop", "planted_day", "placed_day", "yield_units"):
        _set(state, name, position, mask, 0)
    for name in (
        "watered_today",
        "fed_today",
        "cared_today",
        "fertilizer_available",
    ):
        _set(state, name, position, mask, False)
    for name in (
        "consecutive_unwatered",
        "consecutive_unfed",
        "pending_care_bonus",
    ):
        _set(state, name, position, mask, 0)
    _set(state, "fertilized_until_day", position, mask, -1)
    _set(state, "max_lifespan_step", position, mask, -1)


def _inv_add(
    state: SimState,
    unit: int,
    item: int,
    amount: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    count = state.inv_count.reshape(-1, MAX_UNITS, len(SHED_NAMES))
    sequence = state.inv_seq.reshape(-1, MAX_UNITS, len(SHED_NAMES))
    tick = state.inv_tick.reshape(-1, MAX_UNITS)
    active = mask & (amount > 0)
    newly_present = active & (count[:, unit, item] == 0)
    tick[:, unit].add_(newly_present.to(tick.dtype))
    sequence[:, unit, item].copy_(
        torch.where(newly_present, tick[:, unit], sequence[:, unit, item])
    )
    count[:, unit, item].add_(torch.where(active, amount, 0).to(count.dtype))


def _inv_take(
    state: SimState, unit: int, item: int, amount: int, mask: torch.Tensor
) -> torch.Tensor:
    count = state.inv_count.reshape(-1, MAX_UNITS, len(SHED_NAMES))
    sequence = state.inv_seq.reshape(-1, MAX_UNITS, len(SHED_NAMES))
    accepted = mask & (count[:, unit, item] >= amount)
    count[:, unit, item].sub_(accepted.to(count.dtype) * amount)
    sequence[:, unit, item].copy_(
        torch.where(accepted & (count[:, unit, item] == 0), 0, sequence[:, unit, item])
    )
    return accepted


def _plant_guard(actions: torch.Tensor, state: SimState) -> torch.Tensor:
    flat = actions.reshape(-1, MAX_UNITS).clone()
    device = flat.device
    op_crop = tensor_constant(_OP_CROP, dtype=torch.int64, device=device)
    requested = op_crop[flat.to(torch.int64)]
    alive = state.alive.reshape(-1, MAX_UNITS)
    planting = (requested >= 0) & alive
    demand = torch.zeros(len(flat), len(CROP_NAMES), dtype=torch.int16, device=device)
    demand.scatter_add_(
        1,
        requested.clamp_min(0),
        planting.to(torch.int16),
    )
    seeds = state.seeds.reshape(-1, len(CROP_NAMES))
    blocked = demand > seeds
    blocked_unit = blocked.gather(1, requested.clamp_min(0)) & planting
    flat.copy_(torch.where(blocked_unit, _OP["PASS"], flat))
    return flat


def _drop(state: SimState, unit: int, mask: torch.Tensor) -> None:
    count = state.inv_count.reshape(-1, MAX_UNITS, len(SHED_NAMES))
    sequence = state.inv_seq.reshape(-1, MAX_UNITS, len(SHED_NAMES))
    held = count[:, unit]
    order = torch.where(sequence[:, unit] > 0, sequence[:, unit], 2**30).argsort(dim=1)
    ordered = held.gather(1, order)
    cumulative = ordered.cumsum(dim=1)
    shed = state.shed.reshape(-1, len(SHED_NAMES))
    room = (SHED_CAPACITY - shed.sum(dim=1)).clamp_min(0).to(torch.int64)
    accepted_cumulative = torch.minimum(cumulative, room[:, None])
    previous = torch.nn.functional.pad(accepted_cumulative[:, :-1], (1, 0))
    accepted_ordered = accepted_cumulative - previous
    accepted = torch.zeros_like(accepted_ordered).scatter(1, order, accepted_ordered)
    shed.add_((accepted * mask[:, None]).to(shed.dtype))
    count[:, unit].copy_(torch.where(mask[:, None], 0, held))
    sequence[:, unit].copy_(torch.where(mask[:, None], 0, sequence[:, unit]))


def apply_unit_phases(  # noqa: C901 - mirrors the reference's ordered dispatch
    original: SimState, unit_actions: torch.Tensor
) -> SimState:
    """Apply the atomic PLANT guard and all twenty serialized unit slots."""
    expected = (original.batch_size, 2, MAX_UNITS)
    if tuple(unit_actions.shape) != expected:
        raise ValueError(
            f"expected unit actions {expected}, got {tuple(unit_actions.shape)}"
        )
    state = _clone(original)
    actions = _plant_guard(unit_actions, state)
    x_all = state.unit_x.reshape(-1, MAX_UNITS)
    y_all = state.unit_y.reshape(-1, MAX_UNITS)
    alive_all = state.alive.reshape(-1, MAX_UNITS)
    day = state.day[:, None].expand(-1, 2).reshape(-1)
    seeds = state.seeds.reshape(-1, len(CROP_NAMES))
    shed = state.shed.reshape(-1, len(SHED_NAMES))

    for unit in range(MAX_UNITS):
        op = actions[:, unit].to(torch.int64)
        alive = alive_all[:, unit]
        x = x_all[:, unit].to(torch.int64)
        y = y_all[:, unit].to(torch.int64)
        for name, dx, dy in (
            ("NORTH", 0, -1),
            ("SOUTH", 0, 1),
            ("EAST", 1, 0),
            ("WEST", -1, 0),
        ):
            moving = alive & (op == _OP[name])
            valid = (
                moving & (x + dx >= 0) & (x + dx < 10) & (y + dy >= 0) & (y + dy < 10)
            )
            x_all[:, unit].copy_(torch.where(valid, x + dx, x_all[:, unit]))
            y_all[:, unit].copy_(torch.where(valid, y + dy, y_all[:, unit]))

        position = y * 10 + x
        kind = _gather(state, "kind", position)
        occupant = _gather(state, "occupant", position)
        access = alive & (((x == 4) | (x == 5)) & ((y == 4) | (y == 5)))
        _drop(state, unit, access & (op == _OP["DROP"]))

        for item, name in enumerate(SHED_NAMES):
            pickup = access & (op == _OP[f"PICKUP:{name}"]) & (shed[:, item] > 0)
            shed[:, item].sub_(pickup.to(shed.dtype))
            _inv_add(state, unit, item, torch.ones_like(op), pickup)

            place_op = alive & (op == _OP[f"PLACE:{name}"])
            animal_condition = torch.zeros_like(place_op)
            if name in ANIMAL_NAMES:
                structure = 4 if ANIMALS[name]["structure"] == "COOP" else 5
                animal_condition = place_op & (kind == structure) & (occupant == 0)
                placed = _inv_take(state, unit, item, 1, animal_condition)
                _set(state, "occupant", position, placed, ANIMAL_NAMES.index(name) + 1)
                _set(state, "placed_day", position, placed, day)
                _set(state, "yield_units", position, placed, 0)
                _set(state, "consecutive_unfed", position, placed, 0)
                _set(state, "fed_today", position, placed, False)
                _set(state, "cared_today", position, placed, False)
                _set(state, "fertilizer_available", position, placed, False)
                _set(state, "pending_care_bonus", position, placed, 0)
            room = shed.sum(dim=1) < SHED_CAPACITY
            shed_place = access & place_op & ~animal_condition & room
            moved = _inv_take(state, unit, item, 1, shed_place)
            shed[:, item].add_(moved.to(shed.dtype))

        unlocked = alive & (kind != 1)
        empty = unlocked & (kind == 0)
        for crop_index, name in enumerate(CROP_NAMES):
            planting = empty & (op == _OP[f"PLANT:{name}"]) & (seeds[:, crop_index] > 0)
            seeds[:, crop_index].sub_(planting.to(seeds.dtype))
            _clear(state, position, planting, 3)
            _set(state, "crop", position, planting, crop_index + 1)
            _set(state, "planted_day", position, planting, day)
            _set(state, "consecutive_unwatered", position, planting, 1)
            initial_yield = 0 if CROPS[name]["ongoing"] else 1
            _set(state, "yield_units", position, planting, initial_yield)
            lifespan = (
                -1
                if CROPS[name]["ongoing"]
                else (day + int(CROPS[name]["max_yield_day"]) + 1) * TURNS_PER_DAY
            )
            _set(state, "max_lifespan_step", position, planting, lifespan)

        plant = unlocked & (kind == 3)
        crop = _gather(state, "crop", position).to(torch.int64)
        watered = _gather(state, "watered_today", position)
        watering = plant & (op == _OP["WATER"]) & ~watered
        _set(state, "watered_today", position, watering, True)
        for crop_index, name in enumerate(CROP_NAMES):
            data = CROPS[name]
            if data["ongoing"]:
                continue
            age = day - _gather(state, "planted_day", position)
            window = (int(data["max_yield_day"]) + 1) // 2
            bonus_mask = (
                watering
                & (crop == crop_index + 1)
                & (age >= window)
                & (age <= int(data["max_yield_day"]))
            )
            fertilized = _gather(state, "fertilized_until_day", position) >= day
            current_yield = _gather(state, "yield_units", position)
            gained = torch.where(fertilized, 2, 1)
            updated = torch.minimum(
                current_yield + gained,
                torch.full_like(current_yield, int(data["max_yield"])),
            )
            _set(state, "yield_units", position, bonus_mask, updated)

        fertilizing = plant & (op == _OP["FERTILIZE"])
        fertilized = _inv_take(state, unit, _FERTILIZER, 1, fertilizing)
        until = torch.maximum(_gather(state, "fertilized_until_day", position), day + 2)
        _set(state, "fertilized_until_day", position, fertilized, until)

        tile_yield = _gather(state, "yield_units", position)
        harvesting = unlocked & (op == _OP["HARVEST"]) & (tile_yield > 0)
        for crop_index, name in enumerate(CROP_NAMES):
            mature = day - _gather(state, "planted_day", position) >= int(
                CROPS[name]["first_yield_day"]
            )
            selected = harvesting & (kind == 3) & (crop == crop_index + 1) & mature
            _inv_add(
                state,
                unit,
                SHED_NAMES.index(name),
                tile_yield.to(torch.int64),
                selected,
            )
            _set(state, "yield_units", position, selected, 0)
            if not CROPS[name]["ongoing"]:
                _clear(state, position, selected, 0)
        for animal_index, name in enumerate(ANIMAL_NAMES):
            selected = harvesting & (occupant == animal_index + 1)
            product = str(ANIMALS[name]["product"])
            _inv_add(
                state,
                unit,
                SHED_NAMES.index(product),
                tile_yield.to(torch.int64),
                selected,
            )
            _set(state, "yield_units", position, selected, 0)

        occupied = unlocked & (occupant > 0)
        feeding = (
            occupied & (op == _OP["FEED"]) & ~_gather(state, "fed_today", position)
        )
        fed = _inv_take(state, unit, _WHEAT, 1, feeding)
        _set(state, "fed_today", position, fed, True)
        collecting = (
            occupied
            & (op == _OP["COLLECT_FERTILIZER"])
            & _gather(state, "fertilizer_available", position)
        )
        _set(state, "fertilizer_available", position, collecting, False)
        _inv_add(state, unit, _FERTILIZER, torch.ones_like(op), collecting)
        caring = (
            occupied & (op == _OP["CARE"]) & ~_gather(state, "cared_today", position)
        )
        _set(state, "cared_today", position, caring, True)

        digging = unlocked & (op == _OP["DIG"]) & (kind != 0) & (occupant == 0)
        _clear(state, position, digging, 0)
        _clear(state, position, empty & (op == _OP["BUILD_COOP"]), 4)
        _clear(state, position, empty & (op == _OP["BUILD_PASTURE"]), 5)
    return state
