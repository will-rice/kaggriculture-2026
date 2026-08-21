"""Direct tensor legal-action masks for simulator state."""

import torch

from kaggriculture.constants import (
    ANIMALS,
    CROPS,
    LAND_ORDER,
    LAND_PRICES,
    SHED_CAPACITY,
)
from kaggriculture.learn.encoding import (
    HIRE_SLOT,
    LAND_SLOT,
    MARKET_SLOTS,
    MAX_ORDERS,
    MAX_UNITS,
    QUANTITIES,
    UNIT_OPS,
)
from kaggriculture.sim.market import QUANTITY_AXIS
from kaggriculture.sim.pricing import market_prices
from kaggriculture.sim.state import (
    ANIMAL_NAMES,
    CROP_NAMES,
    PRODUCT_NAMES,
    SHED_NAMES,
    SimState,
)
from kaggriculture.sim.tensors import tensor_constant

_OP = {name: index for index, name in enumerate(UNIT_OPS)}
_PRODUCT_SHED = {name: SHED_NAMES.index(name) for name in PRODUCT_NAMES}


def _at_unit(plane: torch.Tensor, flat: torch.Tensor) -> torch.Tensor:
    """Gather a ``(B,H,W)`` plane at each ``(B,U)`` flattened position."""
    return plane.flatten(1).gather(1, flat)


def _unit_mask(state: SimState, seat: int) -> torch.Tensor:
    batch = state.batch_size
    device = state.step.device
    mask = torch.zeros(
        batch, len(state.alive[0, seat]), len(UNIT_OPS), dtype=torch.bool, device=device
    )
    mask[..., _OP["PASS"]] = True
    alive = state.alive[:, seat]
    x = state.unit_x[:, seat].to(torch.int64)
    y = state.unit_y[:, seat].to(torch.int64)
    flat = y * 10 + x
    mask[..., _OP["NORTH"]] = alive & (y > 0)
    mask[..., _OP["SOUTH"]] = alive & (y < 9)
    mask[..., _OP["EAST"]] = alive & (x < 9)
    mask[..., _OP["WEST"]] = alive & (x > 0)

    access = alive & (((x == 4) | (x == 5)) & ((y == 4) | (y == 5)))
    inventory = state.inv_count[:, seat]
    shed_room = state.shed[:, seat].sum(dim=-1) < SHED_CAPACITY
    mask[..., _OP["DROP"]] = access & (inventory.sum(dim=-1) > 0)
    for item, name in enumerate(SHED_NAMES):
        mask[..., _OP[f"PICKUP:{name}"]] = access & (
            state.shed[:, seat, item, None] > 0
        )
        mask[..., _OP[f"PLACE:{name}"]] = (
            access & shed_room[:, None] & (inventory[..., item] > 0)
        )

    kind = _at_unit(state.kind[:, seat], flat)
    occupant = _at_unit(state.occupant[:, seat], flat)
    crop = _at_unit(state.crop[:, seat], flat)
    unlocked = alive & (kind != 1)
    empty = unlocked & (kind == 0)
    mask[..., _OP["BUILD_COOP"]] = empty
    mask[..., _OP["BUILD_PASTURE"]] = empty
    for crop_index, name in enumerate(CROP_NAMES):
        mask[..., _OP[f"PLANT:{name}"]] = empty & (
            state.seeds[:, seat, crop_index, None] > 0
        )

    plant = unlocked & (kind == 3)
    watered = _at_unit(state.watered_today[:, seat], flat)
    yields = _at_unit(state.yield_units[:, seat], flat)
    planted = _at_unit(state.planted_day[:, seat], flat)
    first_yield = tensor_constant(
        [0, *(int(CROPS[name]["first_yield_day"]) for name in CROP_NAMES)],
        dtype=torch.int16,
        device=device,
    )[crop.to(torch.int64)]
    mature = state.day[:, None] - planted >= first_yield
    fertilizer = SHED_NAMES.index("FERTILIZER")
    mask[..., _OP["WATER"]] = plant & ~watered
    mask[..., _OP["FERTILIZE"]] = plant & (inventory[..., fertilizer] > 0)
    mask[..., _OP["HARVEST"]] = plant & (yields > 0) & mature
    mask[..., _OP["DIG"]] = unlocked & ~(occupant > 0) & (kind != 0)

    animal = unlocked & (occupant > 0)
    fed = _at_unit(state.fed_today[:, seat], flat)
    cared = _at_unit(state.cared_today[:, seat], flat)
    fertilizer_ready = _at_unit(state.fertilizer_available[:, seat], flat)
    wheat = SHED_NAMES.index("WHEAT")
    mask[..., _OP["FEED"]] = animal & ~fed & (inventory[..., wheat] > 0)
    mask[..., _OP["CARE"]] = animal & ~cared
    mask[..., _OP["COLLECT_FERTILIZER"]] = animal & fertilizer_ready
    mask[..., _OP["HARVEST"]] |= animal & (yields > 0)

    structure_kind = {"COOP": 4, "PASTURE": 5}
    for name in ANIMAL_NAMES:
        item = SHED_NAMES.index(name)
        matching_bare = (
            unlocked
            & (kind == structure_kind[str(ANIMALS[name]["structure"])])
            & (occupant == 0)
            & (inventory[..., item] > 0)
        )
        mask[..., _OP[f"PLACE:{name}"]] |= matching_bare
    return mask


def _market_mask(state: SimState, seat: int) -> torch.Tensor:
    batch = state.batch_size
    device = state.step.device
    mask = torch.zeros(
        batch,
        len(MARKET_SLOTS) + 2,
        len(QUANTITIES),
        dtype=torch.bool,
        device=device,
    )
    mask[..., 0] = True
    money = state.money[:, seat]
    room = SHED_CAPACITY - state.shed[:, seat].sum(dim=-1)
    fillable = []
    product_index = {name: index for index, name in enumerate(PRODUCT_NAMES)}
    unit_offsets = torch.arange(1, QUANTITY_AXIS + 1, dtype=torch.int64, device=device)
    for verb, item in MARKET_SLOTS:
        if verb == "SELL":
            value = state.shed[:, seat, _PRODUCT_SHED[item]].to(torch.int64)
        elif verb == "BUY_SEED":
            value = money // int(CROPS[item]["seed"])
        elif verb == "BUY_ANIMAL":
            value = torch.minimum(
                money // int(ANIMALS[item]["cost"]), room.to(torch.int64)
            )
        else:
            index = product_index[item]
            levels = state.inventory[:, index, None] - unit_offsets
            costs = market_prices(levels, index).cumsum(dim=-1)
            value = ((costs <= money[:, None]) & (unit_offsets <= room[:, None])).sum(
                dim=-1
            )
        fillable.append(value)
    limits = torch.stack(fillable, dim=-1)
    quantities = tensor_constant(QUANTITIES, dtype=torch.int32, device=device)
    mask[:, : len(MARKET_SLOTS), 1:] = quantities[None, None, 1:] <= limits[..., None]

    remaining = money.clone()
    hired = torch.zeros(batch, dtype=torch.int64, device=device)
    crew_room = MAX_UNITS - state.hand_count[:, seat].to(torch.int64) - 1
    fib = (1, 1, 2, 3, 5, 8, 13, 21, 34, 55)
    for offset in range(MAX_ORDERS):
        hire_index = (state.hires_today[:, seat].to(torch.int64) + offset).clamp_max(9)
        costs = tensor_constant(fib, dtype=torch.int64, device=device)[hire_index]
        accepted = (hired < crew_room) & (remaining >= costs)
        remaining = torch.where(accepted, remaining - costs, remaining)
        hired += accepted
    mask[:, HIRE_SLOT, 1:] = quantities[None, 1:] <= hired[:, None]

    bought = state.quadrants[:, seat].to(torch.int64) - 1
    land_costs = tensor_constant(LAND_PRICES, dtype=torch.int64, device=device)
    has_land = bought < len(LAND_ORDER)
    cost = land_costs[bought.clamp_max(len(LAND_ORDER) - 1)]
    mask[:, LAND_SLOT, 1] = has_land & (money >= cost)
    return mask


def legal(state: SimState, seat: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return direct unit and market legality masks for one seat."""
    if seat not in (0, 1):
        raise ValueError(f"seat must be 0 or 1, got {seat}")
    return _unit_mask(state, seat), _market_mask(state, seat)
