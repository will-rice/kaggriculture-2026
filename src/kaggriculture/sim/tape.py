"""Replaying recorded routes through the batched simulator, for exact banks.

This is the quantity lane's fidelity proof, and its scope is the *unit*
transfer lane specifically: a recorded top episode exercises bulk
PICKUP/PLACE on most of its transfers, a branch the RL action space
structurally never reaches, and ``replay`` reproduces such a route's own
bank exactly, because the engine is deterministic given a seed and both
seats' actions.

That guarantee holds only for routes whose *market* orders stay inside the
simulator's domain. A market fill above ``sim.market.QUANTITY_AXIS`` is
outside it, and ``encode_turn`` raises rather than silently clamping or
truncating one -- see ``sim.rollout._order_quantity``. This was not an edge
case at the axis's original width of 64: a scan of top-rated 1.32.7 episodes
found the large majority carry at least one market order above it, typically
in the 65-80 range, real fills the engine actually committed unit by unit --
not merely requested and then clamped down by stock or shed room. The axis
was widened to 165 for exactly that reason (see the comment beside
``QUANTITIES`` in ``learn.encoding`` for the measured distribution behind the
number), with real headroom above every fill measured. A request naming a
quantity past even that -- the engine's own "sell everything" sentinel is the
one seen in practice -- is still outside the domain, and ``encode_turn``
still raises on it rather than silently clamping. A route that does stay
inside the domain replays here to the exact bank it was harvested with, or
the lane is wrong.
"""

from collections.abc import Sequence

import torch

from kaggriculture.constants import EPISODE_STEPS, MAX_MARKET_ORDERS_PER_TURN
from kaggriculture.learn.encoding import MAX_UNITS
from kaggriculture.search.route import Route
from kaggriculture.sim.config import Config
from kaggriculture.sim.engine import MarketActions, reset, step
from kaggriculture.sim.rollout import encode_turn


def replay(
    seat_zero: Sequence[Route],
    seat_one: Sequence[Route],
    seeds: Sequence[int],
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Replay recorded routes through the batched simulator to their final banks.

    The batched analogue of ``search.arena.play``: every game advances
    together, one ``step`` call per turn, with no host synchronisation once
    both seats' routes are encoded. Mirrors the alignment
    ``search.arena._replay`` established at cost: the action recorded at
    ``steps[t]`` was chosen from the observation whose own ``step`` field
    reads ``t - 1``, so turn ``k`` of this loop executes ``route[k + 1]``, not
    ``route[k]``.

    Args:
        seat_zero: Seat 0's route per game, as ``route.from_episode``/``load``
            produce -- one action dict per turn, ``EPISODE_STEPS`` long.
        seat_one: Seat 1's route per game, same length as ``seat_zero``.
        seeds: One episode seed per game, e.g. ``episode["info"]["seed"]``.
        device: Device to run the batch on.

    Returns:
        ``(len(seeds), 2)`` int64 final banks, one row per game, in the order
        ``seeds`` was given.
    """
    batch = len(seeds)
    if len(seat_zero) != batch or len(seat_one) != batch:
        raise ValueError(
            f"seat_zero ({len(seat_zero)}), seat_one ({len(seat_one)}) and "
            f"seeds ({batch}) must be the same length"
        )
    for route in (*seat_zero, *seat_one):
        if len(route) != EPISODE_STEPS:
            raise ValueError(f"route must have {EPISODE_STEPS} turns, got {len(route)}")

    turns = EPISODE_STEPS - 1
    unit_actions = torch.empty((turns, batch, 2, MAX_UNITS), dtype=torch.int16)
    unit_quantities = torch.empty((turns, batch, 2, MAX_UNITS), dtype=torch.int16)
    order_type = torch.empty(
        (turns, batch, 2, MAX_MARKET_ORDERS_PER_TURN), dtype=torch.int8
    )
    order_item = torch.empty(
        (turns, batch, 2, MAX_MARKET_ORDERS_PER_TURN), dtype=torch.int8
    )
    order_qty = torch.empty(
        (turns, batch, 2, MAX_MARKET_ORDERS_PER_TURN), dtype=torch.int32
    )
    for game, (route_zero, route_one) in enumerate(
        zip(seat_zero, seat_one, strict=True)
    ):
        for seat, route in enumerate((route_zero, route_one)):
            for turn, action in enumerate(route[1:]):
                encoded = encode_turn(action)
                unit_actions[turn, game, seat] = torch.tensor(
                    encoded.units, dtype=torch.int16
                )
                unit_quantities[turn, game, seat] = torch.tensor(
                    encoded.quantities, dtype=torch.int16
                )
                for slot, (kind, item, quantity) in enumerate(encoded.orders):
                    order_type[turn, game, seat, slot] = kind
                    order_item[turn, game, seat, slot] = item
                    order_qty[turn, game, seat, slot] = quantity

    unit_actions = unit_actions.to(device)
    unit_quantities = unit_quantities.to(device)
    order_type = order_type.to(device)
    order_item = order_item.to(device)
    order_qty = order_qty.to(device)

    seeds_tensor = torch.tensor(seeds, dtype=torch.int64, device=device)
    state = reset(Config(), seeds_tensor, device=device)
    for turn in range(turns):
        market = MarketActions(
            order_type=order_type[turn],
            order_item=order_item[turn],
            order_qty=order_qty[turn],
        )
        state = step(state, unit_actions[turn], market, unit_quantities[turn])
    return state.money
