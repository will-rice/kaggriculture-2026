"""Pure-Torch town, decay, end-of-day, RNG, and clock phases."""

from dataclasses import fields, replace

import torch

from kaggriculture.constants import (
    ANIMALS,
    CROPS,
    EPISODE_STEPS,
    SHED_CAPACITY,
    SHOPS,
    TOWN_CENTER_PRODUCTS,
    TURNS_PER_DAY,
)
from kaggriculture.sim.pricing import refresh_prices
from kaggriculture.sim.rng import random_from_words, select_day_words
from kaggriculture.sim.state import (
    ANIMAL_NAMES,
    CROP_NAMES,
    PRODUCT_NAMES,
    SHED_NAMES,
    SHOP_NAMES,
    SimState,
)
from kaggriculture.sim.tensors import tensor_constant


def _clone(state: SimState) -> SimState:
    return SimState(
        **{field.name: getattr(state, field.name).clone() for field in fields(state)}
    )


def _clear_tile(state: SimState, mask: torch.Tensor, kind: int) -> None:
    state.kind[mask] = kind
    for name in ("occupant", "crop", "planted_day", "placed_day", "yield_units"):
        getattr(state, name)[mask] = 0
    for name in (
        "watered_today",
        "fed_today",
        "cared_today",
        "fertilizer_available",
    ):
        getattr(state, name)[mask] = False
    for name in (
        "consecutive_unwatered",
        "consecutive_unfed",
        "pending_care_bonus",
    ):
        getattr(state, name)[mask] = 0
    state.fertilized_until_day[mask] = -1
    state.max_lifespan_step[mask] = -1


def _town_and_decay(state: SimState, active: torch.Tensor) -> None:
    shop_tick = active & (state.step % 4 == 0)
    center_tick = active & (state.step % 24 == 0)
    state.inventory.sub_(state.shop_drain.to(torch.int64) * shop_tick[:, None])
    center = tensor_constant(
        [int(name in TOWN_CENTER_PRODUCTS) for name in PRODUCT_NAMES],
        dtype=torch.int64,
        device=state.step.device,
    )
    state.inventory.sub_(center[None, :] * center_tick[:, None])
    refreshed = refresh_prices(state.inventory)
    state.prices.copy_(torch.where(active[:, None], refreshed, state.prices))

    step = state.step[:, None, None, None]
    decay = (
        active[:, None, None, None]
        & (state.kind == 3)
        & (state.max_lifespan_step >= 0)
        & (step >= state.max_lifespan_step)
        & ((step - state.max_lifespan_step) % 2 == 0)
    )
    state.yield_units.sub_(decay.to(state.yield_units.dtype))
    _clear_tile(state, decay & (state.yield_units <= 0), 2)


def _refresh_plants(state: SimState, end: torch.Tensor) -> None:
    device = state.step.device
    plant = end[:, None, None, None] & (state.kind == 3)
    was_watered = state.watered_today.clone()
    missed = torch.where(
        was_watered,
        torch.zeros_like(state.consecutive_unwatered),
        state.consecutive_unwatered + 1,
    )
    state.consecutive_unwatered.copy_(
        torch.where(plant, missed, state.consecutive_unwatered)
    )
    state.watered_today.logical_and_(~plant)
    dead = plant & (state.consecutive_unwatered >= 2)
    _clear_tile(state, dead, 2)
    plant &= ~dead

    ongoing_values = tensor_constant(
        [0, *(int(bool(CROPS[name]["ongoing"])) for name in CROP_NAMES)],
        dtype=torch.bool,
        device=device,
    )[state.crop.to(torch.int64)]
    first = tensor_constant(
        [0, *(int(CROPS[name]["first_yield_day"]) for name in CROP_NAMES)],
        dtype=torch.int16,
        device=device,
    )[state.crop.to(torch.int64)]
    interval = tensor_constant(
        [1, *(int(CROPS[name]["interval"]) for name in CROP_NAMES)],
        dtype=torch.int16,
        device=device,
    )[state.crop.to(torch.int64)].clamp_min(1)
    maximum = tensor_constant(
        [0, *(int(CROPS[name]["max_yield"]) for name in CROP_NAMES)],
        dtype=torch.int16,
        device=device,
    )[state.crop.to(torch.int64)]
    next_day = state.day[:, None, None, None] + 1
    since = next_day - state.planted_day - first
    count = torch.div(since, interval, rounding_mode="floor") + 1
    produce = (
        plant
        & ongoing_values
        & (since >= 0)
        & (since % interval == 0)
        & (count <= maximum)
    )
    fertilized = was_watered & (
        state.fertilized_until_day >= state.day[:, None, None, None]
    )
    addition = torch.where(fertilized, 2, 1).to(state.yield_units.dtype)
    produced = torch.minimum(maximum, state.yield_units + addition)
    state.yield_units.copy_(torch.where(produce, produced, state.yield_units))
    final = produce & (count == maximum)
    death_step = (next_day + 1) * TURNS_PER_DAY
    state.max_lifespan_step.copy_(
        torch.where(final, death_step.to(torch.int32), state.max_lifespan_step)
    )


def _refresh_animals(state: SimState, end: torch.Tensor) -> None:
    device = state.step.device
    animal = end[:, None, None, None] & (state.occupant > 0)
    was_fed = state.fed_today.clone()
    was_cared = state.cared_today.clone()
    unfed = torch.where(
        was_fed,
        torch.zeros_like(state.consecutive_unfed),
        state.consecutive_unfed + 1,
    )
    state.consecutive_unfed.copy_(torch.where(animal, unfed, state.consecutive_unfed))
    escaped = animal & (state.consecutive_unfed >= 2)
    occupant_before = state.occupant.clone()
    structure = tensor_constant(
        [
            0,
            *(
                4 if ANIMALS[name]["structure"] == "COOP" else 5
                for name in ANIMAL_NAMES
            ),
        ],
        dtype=torch.int8,
        device=device,
    )[occupant_before.to(torch.int64)]
    _clear_tile(state, escaped, 0)
    state.kind.copy_(torch.where(escaped, structure, state.kind))
    animal &= ~escaped

    first = tensor_constant(
        [0, *(int(ANIMALS[name]["first_yield_day"]) for name in ANIMAL_NAMES)],
        dtype=torch.int16,
        device=device,
    )[occupant_before.to(torch.int64)]
    interval = tensor_constant(
        [1, *(int(ANIMALS[name]["interval"]) for name in ANIMAL_NAMES)],
        dtype=torch.int16,
        device=device,
    )[occupant_before.to(torch.int64)]
    maximum = tensor_constant(
        [0, *(int(ANIMALS[name]["max_held"]) for name in ANIMAL_NAMES)],
        dtype=torch.int16,
        device=device,
    )[occupant_before.to(torch.int64)]
    next_day = state.day[:, None, None, None] + 1
    since = next_day - state.placed_day - first
    produce = animal & (since >= 0) & (since % interval == 0)
    bonus = torch.where(was_fed, state.pending_care_bonus, 0)
    produced = torch.minimum(maximum, state.yield_units + 1 + bonus)
    state.yield_units.copy_(torch.where(produce, produced, state.yield_units))
    state.pending_care_bonus.copy_(
        torch.where(
            produce,
            torch.zeros_like(state.pending_care_bonus),
            state.pending_care_bonus,
        )
    )
    cared_bonus = animal & was_cared & was_fed
    state.pending_care_bonus.add_(cared_bonus.to(state.pending_care_bonus.dtype))
    state.fertilizer_available.logical_or_(animal)
    state.fed_today.logical_and_(~animal)
    state.cared_today.logical_and_(~animal)


def _weeds_and_shop(state: SimState, end: torch.Tensor) -> None:
    batch = state.batch_size
    day = state.day.clamp(0, state.rng_words.shape[1] - 1).to(torch.int64)
    words = select_day_words(state.rng_words, day)
    randoms = random_from_words(words[:, :400]).reshape(batch, 200)
    empty = state.kind == 0
    flat_empty = empty.reshape(batch, 200)
    ranks = flat_empty.to(torch.int64).cumsum(dim=1) - 1
    draws = randoms.gather(1, ranks.clamp_min(0))
    spawn = flat_empty & (draws < 0.005) & end[:, None]
    flat_kind = state.kind.reshape(batch, 200)
    flat_kind.copy_(torch.where(spawn, 2, flat_kind))

    consumed = 2 * flat_empty.sum(dim=1)
    attempts = torch.arange(64, device=state.step.device)[None, :]
    offsets = (consumed[:, None] + attempts).clamp_max(words.shape[1] - 1)
    choices = (words.to(torch.int64).gather(1, offsets) >> 28).to(torch.int64)
    accepted = choices < len(SHOP_NAMES)
    first = accepted.to(torch.int64).argmax(dim=1)
    drawn = choices.gather(1, first[:, None]).squeeze(1)
    unlock = end & (((state.day + 1) % 3) == 0) & (state.shop_count < 8)
    slot = state.shop_count.to(torch.int64).clamp_max(7)
    shop_slots = torch.arange(8, device=state.step.device)[None, :]
    write = unlock[:, None] & (shop_slots == slot[:, None])
    state.shops.copy_(torch.where(write, drawn[:, None].to(torch.int8), state.shops))
    state.shop_count.add_(unlock.to(torch.int8))

    drains = torch.zeros(
        len(SHOP_NAMES), len(PRODUCT_NAMES), dtype=torch.int16, device=state.step.device
    )
    for shop_index, name in enumerate(SHOP_NAMES):
        amount = 2 if len(SHOPS[name]) == 1 else 1
        for product in SHOPS[name]:
            drains[shop_index, PRODUCT_NAMES.index(product)] += amount
    selected = drains[drawn]
    state.shop_drain.add_(selected * unlock[:, None])


def _drop_and_reset(state: SimState, end: torch.Tensor) -> None:
    batch = state.batch_size
    items = len(SHED_NAMES)
    units = state.inv_count.shape[2]
    counts = state.inv_count.reshape(batch, 2, units * items)
    sequence = state.inv_seq.reshape(batch, 2, units * items)
    unit_ids = torch.arange(units, device=state.step.device).repeat_interleave(items)
    order_key = unit_ids[None, None, :] * 1_000_000 + sequence
    order_key = torch.where(sequence > 0, order_key, torch.full_like(order_key, 2**62))
    order = order_key.argsort(dim=-1)
    ordered_counts = counts.gather(-1, order)
    cumulative = ordered_counts.cumsum(dim=-1)
    room = (SHED_CAPACITY - state.shed.sum(dim=-1)).clamp_min(0).to(torch.int64)
    taken_cumulative = torch.minimum(cumulative, room[..., None])
    previous = torch.nn.functional.pad(taken_cumulative[..., :-1], (1, 0))
    taken_ordered = taken_cumulative - previous
    taken = torch.zeros_like(taken_ordered).scatter(-1, order, taken_ordered)
    item_ids = torch.arange(items, device=state.step.device).repeat(units)
    additions = torch.zeros_like(state.shed, dtype=torch.int64)
    additions.scatter_add_(2, item_ids[None, None, :].expand(batch, 2, -1), taken)
    state.shed.add_((additions * end[:, None, None]).to(state.shed.dtype))

    reset = end[:, None, None]
    state.hand_count.copy_(torch.where(end[:, None], 0, state.hand_count))
    state.hires_today.copy_(torch.where(end[:, None], 0, state.hires_today))
    alive = torch.zeros_like(state.alive)
    alive[..., 0] = True
    state.alive.copy_(torch.where(reset, alive, state.alive))
    spawn = torch.zeros_like(state.unit_x)
    spawn[..., 0] = 4
    state.unit_x.copy_(torch.where(reset, spawn, state.unit_x))
    state.unit_y.copy_(torch.where(reset, spawn, state.unit_y))
    state.inv_count.copy_(torch.where(reset[..., None], 0, state.inv_count))
    state.inv_seq.copy_(torch.where(reset[..., None], 0, state.inv_seq))
    state.inv_tick.copy_(torch.where(reset, 0, state.inv_tick))


def apply_day_phases(original: SimState) -> SimState:
    """Apply phases 4–7 to a state after unit and market processing."""
    state = _clone(original)
    active = ~state.done
    _town_and_decay(state, active)
    end = active & ((state.step + 1) % TURNS_PER_DAY == 0)
    _refresh_plants(state, end)
    _refresh_animals(state, end)
    _weeds_and_shop(state, end)
    _drop_and_reset(state, end)
    next_step = state.step + active.to(state.step.dtype)
    state.step.copy_(next_step)
    state.day.copy_(torch.where(active, next_step // TURNS_PER_DAY, state.day))
    state.hour.copy_(torch.where(active, next_step % TURNS_PER_DAY, state.hour))
    finished = active & ((next_step - 1) >= EPISODE_STEPS - 2)
    state.done.logical_or_(finished)
    state.reward.copy_(
        torch.where(finished[:, None], state.money.to(torch.float64), state.reward)
    )
    return replace(state)
