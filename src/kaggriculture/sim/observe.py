"""Direct tensor observations for batched simulator state."""

import torch

from kaggriculture.constants import (
    ANIMALS,
    CROPS,
    EPISODE_STEPS,
    MARKET_PARAMS,
    SEASON_DAYS,
    SHED_CAPACITY,
    TURNS_PER_DAY,
)
from kaggriculture.features import (
    ANIMAL_NAMES,
    CARE_BONUS_SCALE,
    CARRIED_SCALE,
    CROP_NAMES,
    MAX_UNITS,
    NO_DEATH_SCHEDULED,
    PER_FARM_PLANES,
    PLANE_INDEX,
    PRODUCT_NAMES,
    SCALARS,
    SEED_SCALE,
    SHED_NAMES,
    SHOP_NAMES,
    TILE_PLANES,
    UNIT_CARRIED_SCALE,
)
from kaggriculture.sim.state import (
    SimState,
)
from kaggriculture.sim.tensors import tensor_constant

_PER_FARM_PLANES = PER_FARM_PLANES
_STATE_BASE = PLANE_INDEX["state:WEED"]
_FEATURE_BASE = PLANE_INDEX["feature:ACTIVE_TODAY"]
_UNIT_BASE = PLANE_INDEX["unit:FARMER"]
_PRODUCT_SHED_INDICES = tuple(SHED_NAMES.index(name) for name in PRODUCT_NAMES)


def _as_feature(value: torch.Tensor) -> torch.Tensor:
    """Match Python-double calculation followed by float32 assignment."""
    return value.to(torch.float64).to(torch.float32)


def _scatter_units(
    planes: torch.Tensor, state: SimState, player: int, base: int
) -> None:
    batch = state.batch_size
    flat = state.unit_y[:, player].to(torch.int64) * 10 + state.unit_x[:, player].to(
        torch.int64
    )
    farmer = planes[:, base + _UNIT_BASE].reshape(batch, 100)
    farmer.scatter_(1, flat[:, :1], 1.0)
    hand_values = state.alive[:, player, 1:].to(torch.float32) / MAX_UNITS
    hands = planes[:, base + _UNIT_BASE + 1].reshape(batch, 100)
    hands.scatter_add_(1, flat[:, 1:], hand_values)


def _write_farm(planes: torch.Tensor, state: SimState, player: int, base: int) -> None:
    kind = state.kind[:, player]
    crop = state.crop[:, player]
    occupant = state.occupant[:, player]
    plant = kind == 3
    animal = occupant > 0

    for index in range(len(CROP_NAMES)):
        planes[:, base + index] = (crop == index + 1).to(torch.float32)
    for index in range(len(ANIMAL_NAMES)):
        planes[:, base + len(CROP_NAMES) + index] = (occupant == index + 1).to(
            torch.float32
        )

    planes[:, base + _STATE_BASE] = (kind == 2).to(torch.float32)
    planes[:, base + _STATE_BASE + 1] = (kind == 1).to(torch.float32)
    planes[:, base + _STATE_BASE + 2] = (kind == 0).to(torch.float32)
    planes[:, base + _STATE_BASE + 3] = ((kind == 4) & ~animal).to(torch.float32)
    planes[:, base + _STATE_BASE + 4] = ((kind == 5) & ~animal).to(torch.float32)

    active = torch.where(
        plant, state.watered_today[:, player], state.fed_today[:, player]
    )
    cared = animal & state.cared_today[:, player]
    care_bonus = torch.where(
        animal,
        state.pending_care_bonus[:, player],
        torch.zeros_like(state.pending_care_bonus[:, player]),
    )
    distress = torch.where(
        plant,
        state.consecutive_unwatered[:, player],
        state.consecutive_unfed[:, player],
    )
    crop_caps = tensor_constant(
        [0, *(int(CROPS[name]["max_yield"]) for name in CROP_NAMES)],
        device=kind.device,
        dtype=torch.int16,
    )[crop.to(torch.int64)]
    animal_caps = tensor_constant(
        [0, *(int(ANIMALS[name]["max_held"]) for name in ANIMAL_NAMES)],
        device=kind.device,
        dtype=torch.int16,
    )[occupant.to(torch.int64)]
    capacity = torch.where(plant, crop_caps, animal_caps).clamp_min(1)
    relevant = plant | animal
    bonus = torch.where(
        plant,
        state.fertilized_until_day[:, player] >= state.day[:, None, None],
        state.fertilizer_available[:, player],
    )
    planted_age = state.day[:, None, None] - state.planted_day[:, player]
    placed_age = state.day[:, None, None] - state.placed_day[:, player]
    age = torch.where(plant, planted_age, placed_age)
    death = state.max_lifespan_step[:, player]
    lifespan = torch.where(
        death < 0,
        torch.full_like(death, NO_DEATH_SCHEDULED, dtype=torch.float64),
        (death.to(torch.float64) - state.step[:, None, None]) / EPISODE_STEPS,
    )
    lifespan = torch.where(plant, lifespan, torch.zeros_like(lifespan))

    values = (
        active.to(torch.float32),
        cared.to(torch.float32),
        _as_feature(care_bonus) / CARE_BONUS_SCALE,
        _as_feature(distress) / 2.0,
        _as_feature(state.yield_units[:, player]) / _as_feature(capacity),
        bonus.to(torch.float32),
        _as_feature(age) / SEASON_DAYS,
        lifespan.to(torch.float32),
    )
    mask = relevant.to(torch.float32)
    for offset, value in enumerate(values):
        planes[:, base + _FEATURE_BASE + offset] = value * mask
    _scatter_units(planes, state, player, base)


def _board(state: SimState, seat: int) -> torch.Tensor:
    planes = torch.zeros(
        state.batch_size,
        TILE_PLANES,
        10,
        10,
        dtype=torch.float32,
        device=state.step.device,
    )
    _write_farm(planes, state, seat, 0)
    _write_farm(planes, state, 1 - seat, _PER_FARM_PLANES)
    flat = state.unit_y[:, seat].to(torch.int64) * 10 + state.unit_x[:, seat].to(
        torch.int64
    )
    carried = (
        state.inv_count[:, seat].sum(dim=-1).to(torch.float32) / UNIT_CARRIED_SCALE
    )
    carried = carried * state.alive[:, seat]
    planes[:, _UNIT_BASE + 2].reshape(state.batch_size, 100).scatter_add_(
        1, flat, carried
    )
    return planes


def _scalars(state: SimState, seat: int) -> torch.Tensor:
    device = state.step.device
    product_shed_indices = tensor_constant(
        _PRODUCT_SHED_INDICES, dtype=torch.int64, device=device
    )
    bases = tensor_constant(
        [int(MARKET_PARAMS[name]["base"]) for name in PRODUCT_NAMES],
        dtype=torch.float64,
        device=device,
    )
    initials = tensor_constant(
        [int(MARKET_PARAMS[name]["I0"]) for name in PRODUCT_NAMES],
        dtype=torch.float64,
        device=device,
    )
    thresholds = tensor_constant(
        [int(MARKET_PARAMS[name]["T"]) for name in PRODUCT_NAMES],
        dtype=torch.float64,
        device=device,
    )
    other = 1 - seat
    parts = [
        (state.prices.to(torch.float64) - bases) / bases,
        (state.inventory.to(torch.float64) - initials) / thresholds,
        torch.stack(
            (
                state.money[:, seat] / 10_000.0,
                state.money[:, other] / 10_000.0,
                state.day / SEASON_DAYS,
                state.hour / TURNS_PER_DAY,
                state.step / EPISODE_STEPS,
                state.shop_count / len(SHOP_NAMES),
                state.quadrants[:, seat] / 4.0,
                state.quadrants[:, other] / 4.0,
                state.hand_count[:, seat] / 8.0,
                state.hand_count[:, other] / 8.0,
                state.hires_today[:, seat] / MAX_UNITS,
                state.hires_today[:, other] / MAX_UNITS,
            ),
            dim=-1,
        ).to(torch.float64),
        state.shed[:, seat].to(torch.float64) / SHED_CAPACITY,
        state.seeds[:, seat].to(torch.float64) / SEED_SCALE,
        state.inv_count[:, seat]
        .index_select(-1, product_shed_indices)
        .sum(dim=1)
        .to(torch.float64)
        / CARRIED_SCALE,
        torch.stack(
            [(state.shops == index).any(dim=1) for index in range(len(SHOP_NAMES))],
            dim=-1,
        ).to(torch.float64),
    ]
    scalars = torch.cat(parts, dim=-1).to(torch.float32)
    if scalars.shape[-1] != SCALARS:
        raise AssertionError(
            f"native scalar width {scalars.shape[-1]} != canonical {SCALARS}"
        )
    return scalars


def observe(
    state: SimState, seat: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return direct batched board, scalar, and position tensors."""
    if seat not in (0, 1):
        raise ValueError(f"seat must be 0 or 1, got {seat}")
    positions = state.unit_y[:, seat].to(torch.int64) * 10 + state.unit_x[:, seat].to(
        torch.int64
    )
    return _board(state, seat), _scalars(state, seat), positions
