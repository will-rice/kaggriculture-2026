"""On-device rollout utilities for the batched simulator."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from kaggriculture.constants import ANIMALS, CROPS, LAND_PRICES, MARKET_PARAMS
from kaggriculture.learn.encoding import IGNORE, MAX_UNITS, UNIT_OPS
from kaggriculture.sim.decode import decode_market_buckets
from kaggriculture.sim.engine import MarketActions, step
from kaggriculture.sim.legality import legal
from kaggriculture.sim.observe import observe
from kaggriculture.sim.state import (
    ANIMAL_NAMES,
    CROP_NAMES,
    PRODUCT_NAMES,
    SHED_NAMES,
    SimState,
    unpack,
)

GROWING = 0.4
ScriptedOpponent = Callable[[Mapping[str, Any]], Mapping[str, Any]]

_MARKET_TYPES = {
    "SELL": (1, PRODUCT_NAMES),
    "BUY_SEED": (2, CROP_NAMES),
    "BUY_PRODUCT": (3, ("WHEAT", "FERTILIZER")),
    "BUY_ANIMAL": (4, ANIMAL_NAMES),
}


@dataclass(frozen=True)
class Trajectory:
    """One fixed-length on-device segment, time-major and batch-preserving."""

    board: torch.Tensor
    scalars: torch.Tensor
    positions: torch.Tensor
    unit_actions: torch.Tensor
    market_actions: torch.Tensor
    unit_masks: torch.Tensor
    market_masks: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    own: torch.Tensor
    potentials: torch.Tensor
    dones: torch.Tensor
    illegal: int


def potential(state: SimState, seat: int) -> torch.Tensor:
    """Return the six potential components directly from tensor state."""
    if seat not in (0, 1):
        raise ValueError(f"seat must be 0 or 1, got {seat}")
    device = state.step.device
    seed_cost = torch.tensor(
        [float(CROPS[name]["seed"]) for name in CROP_NAMES], device=device
    )
    seeds = (state.seeds[:, seat].to(torch.float32) * seed_cost).sum(dim=-1)
    cumulative_land = torch.tensor(
        [0.0, *torch.tensor(LAND_PRICES, dtype=torch.float32).cumsum(0).tolist()],
        device=device,
    )
    land = cumulative_land[(state.quadrants[:, seat].to(torch.int64) - 1).clamp(0, 3)]

    crop_seed = torch.tensor(
        [0.0, *(float(CROPS[name]["seed"]) for name in CROP_NAMES)], device=device
    )[state.crop[:, seat].to(torch.int64)]
    crop_value = torch.tensor(
        [0.0, *(float(MARKET_PARAMS[name]["base"]) for name in CROP_NAMES)],
        device=device,
    )[state.crop[:, seat].to(torch.int64)]
    plant = state.kind[:, seat] == 3
    growing = torch.where(
        plant,
        crop_seed + GROWING * state.yield_units[:, seat] * crop_value,
        0.0,
    ).sum(dim=(1, 2))

    animal_cost = torch.tensor(
        [0.0, *(float(ANIMALS[name]["cost"]) for name in ANIMAL_NAMES)],
        device=device,
    )[state.occupant[:, seat].to(torch.int64)]
    animal_product_value = torch.tensor(
        [
            0.0,
            *(
                float(MARKET_PARAMS[str(ANIMALS[name]["product"])]["base"])
                for name in ANIMAL_NAMES
            ),
        ],
        device=device,
    )[state.occupant[:, seat].to(torch.int64)]
    occupied = state.occupant[:, seat] > 0
    livestock = torch.where(
        occupied,
        animal_cost + GROWING * state.yield_units[:, seat] * animal_product_value,
        0.0,
    ).sum(dim=(1, 2))

    item_value = []
    for name in SHED_NAMES:
        if name in MARKET_PARAMS:
            item_value.append(float(MARKET_PARAMS[name]["base"]))
        else:
            item_value.append(float(ANIMALS[name]["cost"]))
    values = torch.tensor(item_value, dtype=torch.float32, device=device)
    carried = (state.inv_count[:, seat].to(torch.float32) * values[None, None, :]).sum(
        dim=(1, 2)
    )
    stored = (state.shed[:, seat].to(torch.float32) * values).sum(dim=-1)
    return torch.stack((seeds, land, growing, livestock, carried, stored), dim=-1)


def _sample(
    logits: torch.Tensor, mask: torch.Tensor, generator: torch.Generator | None
) -> tuple[torch.Tensor, torch.Tensor]:
    log_prob = torch.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)
    flat = log_prob.flatten(0, -2)
    chosen = torch.multinomial(flat.exp(), 1, generator=generator)
    return (
        chosen.reshape(logits.shape[:-1]),
        flat.gather(1, chosen).reshape(logits.shape[:-1]),
    )


def _unit_index(action: object) -> int:
    if not isinstance(action, (list, tuple)) or not action:
        return UNIT_OPS.index("PASS")
    verb = str(action[0])
    name = (
        f"{verb}:{action[1]}"
        if verb in {"PLANT", "PICKUP", "PLACE"} and len(action) > 1
        else verb
    )
    return UNIT_OPS.index(name) if name in UNIT_OPS else UNIT_OPS.index("PASS")


def scripted_actions(  # noqa: C901 - mirrors the reference action grammar
    state: SimState, opponent: ScriptedOpponent, seat: int = 1
) -> tuple[torch.Tensor, MarketActions]:
    """Bridge a dict-based scripted opponent into fixed simulator tensors."""
    if seat not in (0, 1):
        raise ValueError(f"seat must be 0 or 1, got {seat}")
    device = state.step.device
    units = torch.full(
        (state.batch_size, MAX_UNITS),
        UNIT_OPS.index("PASS"),
        dtype=torch.int16,
        device=device,
    )
    markets = MarketActions.empty(state.batch_size, device=device)
    for batch in range(state.batch_size):
        action = opponent(unpack(state, batch, seat))
        units[batch, 0] = _unit_index(action.get("farmer", ["PASS"]))
        hands = action.get("hands", [])
        if isinstance(hands, list):
            for unit, unit_action in enumerate(hands[: MAX_UNITS - 1], start=1):
                units[batch, unit] = _unit_index(unit_action)
        orders = action.get("market", [])
        if not isinstance(orders, list):
            continue
        for slot, order in enumerate(orders[:10]):
            if not isinstance(order, (list, tuple)) or not order:
                markets.order_type[batch, seat, slot] = 7
                continue
            verb = str(order[0])
            if verb == "HIRE":
                markets.order_type[batch, seat, slot] = 5
                markets.order_qty[batch, seat, slot] = 1
            elif verb == "BUY_LAND":
                markets.order_type[batch, seat, slot] = 6
                markets.order_qty[batch, seat, slot] = 1
            elif verb in _MARKET_TYPES and len(order) >= 3:
                kind, catalogue = _MARKET_TYPES[verb]
                item = str(order[1])
                quantity = order[2]
                if item not in catalogue or not isinstance(quantity, int):
                    markets.order_type[batch, seat, slot] = 7
                    continue
                markets.order_type[batch, seat, slot] = kind
                markets.order_item[batch, seat, slot] = catalogue.index(item)
                markets.order_qty[batch, seat, slot] = max(0, min(64, quantity))
            else:
                markets.order_type[batch, seat, slot] = 7
    return units, markets


def collect_segment(
    state: SimState,
    policy: torch.nn.Module,
    *,
    turns: int = 32,
    generator: torch.Generator | None = None,
    opponent: ScriptedOpponent | None = None,
) -> tuple[SimState, Trajectory]:
    """Collect a fixed-length segment, optionally bridging a scripted seat 1."""
    records: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "board",
            "scalars",
            "positions",
            "unit_actions",
            "market_actions",
            "unit_masks",
            "market_masks",
            "log_probs",
            "values",
            "rewards",
            "own",
            "potentials",
            "dones",
        )
    }
    illegal_count = torch.zeros((), dtype=torch.int64, device=state.step.device)
    for _ in range(turns):
        observed = [observe(state, seat) for seat in range(2)]
        boards = torch.stack([value[0] for value in observed], dim=1)
        scalars = torch.stack([value[1] for value in observed], dim=1)
        positions = torch.stack([value[2] for value in observed], dim=1)
        masks = [legal(state, seat) for seat in range(2)]
        unit_masks = torch.stack([value[0] for value in masks], dim=1)
        market_masks = torch.stack([value[1] for value in masks], dim=1)
        batch = state.batch_size
        with torch.no_grad():
            unit_logits, market_logits, values = policy(
                boards.flatten(0, 1),
                scalars.flatten(0, 1),
                positions.flatten(0, 1),
            )
        unit_logits = unit_logits.reshape(batch, 2, *unit_logits.shape[1:])
        market_logits = market_logits.reshape(batch, 2, *market_logits.shape[1:])
        values = values.reshape(batch, 2, -1).squeeze(-1)
        chosen_units, unit_log = _sample(unit_logits, unit_masks, generator)
        chosen_market, market_log = _sample(market_logits, market_masks, generator)
        market_orders = decode_market_buckets(chosen_market)
        if opponent is not None:
            opponent_units, opponent_market = scripted_actions(state, opponent)
            chosen_units[:, 1].copy_(opponent_units)
            market_orders.order_type[:, 1].copy_(opponent_market.order_type[:, 1])
            market_orders.order_item[:, 1].copy_(opponent_market.order_item[:, 1])
            market_orders.order_qty[:, 1].copy_(opponent_market.order_qty[:, 1])
        alive = state.alive
        stored_units = torch.where(alive, chosen_units, IGNORE)
        joint_log = (unit_log * alive).sum(dim=-1) + market_log.sum(dim=-1)
        before_money = state.money.clone()
        before_margin = before_money - before_money.flip(1)
        before_potential = torch.stack(
            (potential(state, 0), potential(state, 1)), dim=1
        )
        next_state = step(state, chosen_units.to(torch.int16), market_orders)
        after_money = next_state.money
        after_margin = after_money - after_money.flip(1)
        records["board"].append(boards)
        records["scalars"].append(scalars)
        records["positions"].append(positions)
        records["unit_actions"].append(stored_units)
        records["market_actions"].append(chosen_market)
        records["unit_masks"].append(unit_masks)
        records["market_masks"].append(market_masks)
        records["log_probs"].append(joint_log)
        records["values"].append(values)
        records["rewards"].append((after_margin - before_margin).to(torch.float32))
        records["own"].append((after_money - before_money).to(torch.float32))
        records["potentials"].append(before_potential)
        records["dones"].append(next_state.done[:, None].expand(-1, 2))
        checked_seats = slice(0, 1) if opponent is not None else slice(None)
        illegal_count.add_(
            (
                ~unit_masks[:, checked_seats]
                .gather(-1, chosen_units[:, checked_seats, ..., None])
                .squeeze(-1)
                & alive[:, checked_seats]
            ).sum()
            + (
                ~market_masks[:, checked_seats]
                .gather(-1, chosen_market[:, checked_seats, ..., None])
                .squeeze(-1)
            ).sum()
        )
        state = next_state
    trajectory = Trajectory(
        **{name: torch.stack(values) for name, values in records.items()},
        illegal=int(illegal_count.cpu()),
    )
    return state, trajectory
