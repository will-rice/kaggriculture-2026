"""Pure-Torch PLANT guard and serialized unit-action phases.

Unit slots stay sequential because a later unit sees an earlier unit's
mutations, but every slot resolves the whole batch at once: operation, item,
crop, and animal properties are gathered from device rule tables instead of
being scanned in Python.
"""

from dataclasses import fields

import torch

from kaggriculture.constants import BOARD_SIZE, SHED_CAPACITY, TURNS_PER_DAY
from kaggriculture.learn.encoding import MAX_UNITS, UNIT_OPS
from kaggriculture.sim.state import (
    CROP_NAMES,
    SHED_NAMES,
    SimState,
)
from kaggriculture.sim.tables import (
    CROP_FIRST_YIELD_DAY,
    CROP_INITIAL_YIELD,
    CROP_MAX_YIELD,
    CROP_MAX_YIELD_DAY,
    CROP_ONGOING,
    CROP_SHED,
    CROP_WATER_WINDOW,
    UNIT_MOVE_X,
    UNIT_MOVE_Y,
    UNIT_MOVING,
    UNIT_PICKUP,
    UNIT_PLACE,
    UNIT_PLACE_KIND,
    UNIT_PLACE_OCCUPANT,
    UNIT_PLANT,
    animal_produce,
    crop_rules,
    unit_ops,
)
from kaggriculture.sim.tensors import tensor_constant

_OP = {name: index for index, name in enumerate(UNIT_OPS)}
_OP_CROP = tuple(
    CROP_NAMES.index(name.removeprefix("PLANT:")) if name.startswith("PLANT:") else -1
    for name in UNIT_OPS
)
_FERTILIZER = SHED_NAMES.index("FERTILIZER")
_WHEAT = SHED_NAMES.index("WHEAT")
_TILES = BOARD_SIZE * BOARD_SIZE


def _clone(state: SimState) -> SimState:
    return SimState(
        **{field.name: getattr(state, field.name).clone() for field in fields(state)}
    )


def _planes(state: SimState, name: str) -> torch.Tensor:
    return getattr(state, name).reshape(state.batch_size * 2, _TILES)


def _gather(state: SimState, name: str, position: torch.Tensor) -> torch.Tensor:
    return _planes(state, name).gather(1, position[:, None]).squeeze(1)


def _selector(position: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return a plane-shaped mask that selects one tile per row."""
    hit = torch.zeros(
        (position.shape[0], _TILES), dtype=torch.bool, device=position.device
    )
    return hit.scatter_(1, position[:, None], mask[:, None])


def _write(
    state: SimState,
    name: str,
    hit: torch.Tensor,
    value: torch.Tensor | int | bool,
) -> None:
    """Write one value into every tile a selector marks."""
    plane = _planes(state, name)
    if isinstance(value, torch.Tensor):
        plane.copy_(torch.where(hit, value[:, None].to(plane.dtype), plane))
    else:
        plane.masked_fill_(hit, value)


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
    state: SimState,
    position: torch.Tensor,
    mask: torch.Tensor,
    kind: torch.Tensor | int,
) -> None:
    hit = _selector(position, mask)
    _write(state, "kind", hit, kind)
    for name in ("occupant", "crop", "planted_day", "placed_day", "yield_units"):
        _write(state, name, hit, 0)
    for name in (
        "watered_today",
        "fed_today",
        "cared_today",
        "fertilizer_available",
    ):
        _write(state, name, hit, False)
    for name in (
        "consecutive_unwatered",
        "consecutive_unfed",
        "pending_care_bonus",
    ):
        _write(state, name, hit, 0)
    _write(state, "fertilized_until_day", hit, -1)
    _write(state, "max_lifespan_step", hit, -1)


def _inv_add(
    state: SimState,
    unit: int,
    item: torch.Tensor,
    amount: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    count = state.inv_count.reshape(-1, MAX_UNITS, len(SHED_NAMES))[:, unit]
    sequence = state.inv_seq.reshape(-1, MAX_UNITS, len(SHED_NAMES))[:, unit]
    tick = state.inv_tick.reshape(-1, MAX_UNITS)[:, unit]
    slot = item[:, None]
    held = count.gather(1, slot)
    active = (mask & (amount > 0))[:, None]
    newly_present = active & (held == 0)
    tick.add_(newly_present.squeeze(1).to(tick.dtype))
    sequence.scatter_(
        1,
        slot,
        torch.where(newly_present, tick[:, None], sequence.gather(1, slot)),
    )
    count.scatter_(
        1, slot, (held + torch.where(active, amount[:, None], 0)).to(count.dtype)
    )


def _inv_take(
    state: SimState, unit: int, item: torch.Tensor, amount: int, mask: torch.Tensor
) -> torch.Tensor:
    count = state.inv_count.reshape(-1, MAX_UNITS, len(SHED_NAMES))[:, unit]
    sequence = state.inv_seq.reshape(-1, MAX_UNITS, len(SHED_NAMES))[:, unit]
    slot = item[:, None]
    held = count.gather(1, slot)
    accepted = mask[:, None] & (held >= amount)
    remaining = held - accepted.to(held.dtype) * amount
    count.scatter_(1, slot, remaining)
    sequence.scatter_(
        1,
        slot,
        torch.where(accepted & (remaining == 0), 0, sequence.gather(1, slot)),
    )
    return accepted.squeeze(1)


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


def apply_unit_phases(original: SimState, unit_actions: torch.Tensor) -> SimState:
    """Apply the atomic PLANT guard and all twenty serialized unit slots."""
    expected = (original.batch_size, 2, MAX_UNITS)
    if tuple(unit_actions.shape) != expected:
        raise ValueError(
            f"expected unit actions {expected}, got {tuple(unit_actions.shape)}"
        )
    state = _clone(original)
    device = state.step.device
    actions = _plant_guard(unit_actions, state)
    rules = unit_ops(device)
    crops = crop_rules(device)
    produce = animal_produce(device)
    x_all = state.unit_x.reshape(-1, MAX_UNITS)
    y_all = state.unit_y.reshape(-1, MAX_UNITS)
    alive_all = state.alive.reshape(-1, MAX_UNITS)
    day = state.day[:, None].expand(-1, 2).reshape(-1)
    seeds = state.seeds.reshape(-1, len(CROP_NAMES))
    shed = state.shed.reshape(-1, len(SHED_NAMES))

    for unit in range(MAX_UNITS):
        op = actions[:, unit].to(torch.int64)
        rule = rules[op]
        alive = alive_all[:, unit]
        x = x_all[:, unit].to(torch.int64)
        y = y_all[:, unit].to(torch.int64)

        step_x = x + rule[:, UNIT_MOVE_X]
        step_y = y + rule[:, UNIT_MOVE_Y]
        moving = alive & (rule[:, UNIT_MOVING] > 0)
        valid = (
            moving
            & (step_x >= 0)
            & (step_x < BOARD_SIZE)
            & (step_y >= 0)
            & (step_y < BOARD_SIZE)
        )
        x_all[:, unit].copy_(torch.where(valid, step_x, x_all[:, unit]))
        y_all[:, unit].copy_(torch.where(valid, step_y, y_all[:, unit]))

        position = y * 10 + x
        kind = _gather(state, "kind", position)
        occupant = _gather(state, "occupant", position)
        access = alive & (((x == 4) | (x == 5)) & ((y == 4) | (y == 5)))
        _drop(state, unit, access & (op == _OP["DROP"]))

        carried = rule[:, UNIT_PICKUP]
        pickup_slot = carried.clamp_min(0)
        pickup = (
            access
            & (carried >= 0)
            & (shed.gather(1, pickup_slot[:, None]).squeeze(1) > 0)
        )
        shed.scatter_add_(1, pickup_slot[:, None], -pickup[:, None].to(shed.dtype))
        _inv_add(state, unit, pickup_slot, torch.ones_like(op), pickup)

        offered = rule[:, UNIT_PLACE]
        place_slot = offered.clamp_min(0)
        place_op = alive & (offered >= 0)
        structure = rule[:, UNIT_PLACE_KIND]
        animal_condition = place_op & (kind == structure) & (occupant == 0)
        room = shed.sum(dim=1) < SHED_CAPACITY
        shed_place = access & place_op & ~animal_condition & room
        taken = _inv_take(state, unit, place_slot, 1, animal_condition | shed_place)
        placed = taken & animal_condition
        settled = _selector(position, placed)
        _write(state, "occupant", settled, rule[:, UNIT_PLACE_OCCUPANT])
        _write(state, "placed_day", settled, day)
        _write(state, "yield_units", settled, 0)
        _write(state, "consecutive_unfed", settled, 0)
        _write(state, "fed_today", settled, False)
        _write(state, "cared_today", settled, False)
        _write(state, "fertilizer_available", settled, False)
        _write(state, "pending_care_bonus", settled, 0)
        shed.scatter_add_(
            1, place_slot[:, None], (taken & shed_place)[:, None].to(shed.dtype)
        )

        unlocked = alive & (kind != 1)
        empty = unlocked & (kind == 0)
        sown = rule[:, UNIT_PLANT]
        sown_slot = sown.clamp_min(0)
        sown_rule = crops[sown_slot]
        planting = (
            empty & (sown >= 0) & (seeds.gather(1, sown_slot[:, None]).squeeze(1) > 0)
        )
        seeds.scatter_add_(1, sown_slot[:, None], -planting[:, None].to(seeds.dtype))
        _clear(state, position, planting, 3)
        opened = _selector(position, planting)
        _write(state, "crop", opened, sown_slot + 1)
        _write(state, "planted_day", opened, day)
        _write(state, "consecutive_unwatered", opened, 1)
        _write(state, "yield_units", opened, sown_rule[:, CROP_INITIAL_YIELD])
        _write(
            state,
            "max_lifespan_step",
            opened,
            torch.where(
                sown_rule[:, CROP_ONGOING] > 0,
                -torch.ones_like(day),
                (day + sown_rule[:, CROP_MAX_YIELD_DAY] + 1) * TURNS_PER_DAY,
            ),
        )

        plant = unlocked & (kind == 3)
        crop = _gather(state, "crop", position).to(torch.int64)
        grown = crops[(crop - 1).clamp_min(0)]
        watered = _gather(state, "watered_today", position)
        watering = plant & (op == _OP["WATER"]) & ~watered
        _set(state, "watered_today", position, watering, True)
        age = day - _gather(state, "planted_day", position)
        bonus_mask = (
            watering
            & (crop >= 1)
            & (grown[:, CROP_ONGOING] == 0)
            & (age >= grown[:, CROP_WATER_WINDOW])
            & (age <= grown[:, CROP_MAX_YIELD_DAY])
        )
        fertilized = _gather(state, "fertilized_until_day", position) >= day
        current_yield = _gather(state, "yield_units", position)
        gained = torch.where(fertilized, 2, 1)
        updated = torch.minimum(
            current_yield + gained, grown[:, CROP_MAX_YIELD].to(current_yield.dtype)
        )
        _set(state, "yield_units", position, bonus_mask, updated)

        fertilizing = plant & (op == _OP["FERTILIZE"])
        applied = _inv_take(
            state, unit, torch.full_like(op, _FERTILIZER), 1, fertilizing
        )
        until = torch.maximum(_gather(state, "fertilized_until_day", position), day + 2)
        _set(state, "fertilized_until_day", position, applied, until)

        tile_yield = _gather(state, "yield_units", position)
        harvesting = unlocked & (op == _OP["HARVEST"]) & (tile_yield > 0)
        mature = (
            day - _gather(state, "planted_day", position)
            >= grown[:, CROP_FIRST_YIELD_DAY]
        )
        selected = harvesting & (kind == 3) & (crop >= 1) & mature
        collected = harvesting & (occupant > 0)
        harvested = torch.where(
            selected,
            grown[:, CROP_SHED],
            produce[(occupant.to(torch.int64) - 1).clamp_min(0)],
        )
        _inv_add(
            state,
            unit,
            harvested,
            tile_yield.to(torch.int64),
            selected | collected,
        )
        _set(state, "yield_units", position, selected | collected, 0)
        _clear(state, position, selected & (grown[:, CROP_ONGOING] == 0), 0)

        occupied = unlocked & (occupant > 0)
        feeding = (
            occupied & (op == _OP["FEED"]) & ~_gather(state, "fed_today", position)
        )
        fed = _inv_take(state, unit, torch.full_like(op, _WHEAT), 1, feeding)
        _set(state, "fed_today", position, fed, True)
        collecting = (
            occupied
            & (op == _OP["COLLECT_FERTILIZER"])
            & _gather(state, "fertilizer_available", position)
        )
        _set(state, "fertilizer_available", position, collecting, False)
        _inv_add(
            state,
            unit,
            torch.full_like(op, _FERTILIZER),
            torch.ones_like(op),
            collecting,
        )
        caring = (
            occupied & (op == _OP["CARE"]) & ~_gather(state, "cared_today", position)
        )
        _set(state, "cared_today", position, caring, True)

        digging = unlocked & (op == _OP["DIG"]) & (kind != 0) & (occupant == 0)
        coop = empty & (op == _OP["BUILD_COOP"])
        pasture = empty & (op == _OP["BUILD_PASTURE"])
        _clear(
            state,
            position,
            digging | coop | pasture,
            torch.where(coop, 4, torch.where(pasture, 5, 0)),
        )
    return state
