"""Replaying recorded routes through the batched simulator, for exact banks.

This is the quantity lane's fidelity proof, and its scope is the *unit*
transfer lane specifically: a recorded top episode exercises bulk
PICKUP/PLACE on most of its transfers, a branch the RL action space
structurally never reaches, and ``replay`` reproduces such a route's own
bank exactly, because the engine is deterministic given a seed and both
seats' actions.

That guarantee now covers ordinary top play's market orders, not just its
unit transfers. A scan of top-rated 1.32.7 episodes found that 85% (128 of
150) carry at least one market order *requesting* more than the axis's
original width of 64 -- a request, which stock or shed room can still clamp
down before anything is filled. A separate scan of what the engine actually
committed found a smaller share still crossed that line: 43% (65 of 150) had
at least one order the engine *filled* past 64, typically in the 65-80
range -- real fills, unit by unit, not requests (see ``learn.encoding``,
lines 1011-1014, for that count and the measured fill distribution). The
axis was widened to 165 for that fill share, but a follow-up scan found top
play also spells "sell everything" as a large, round request
-- ``SELL X 999`` and the like, in 4.75% of a 400-episode sample -- that
still named a quantity past even a 165-wide axis. For SELL, BUY_PRODUCT and
BUY_ANIMAL such a request now *clamps* onto the axis at encode time rather
than raising, because the reference engine's own fill for those three verbs
is bounded by ``SHED_CAPACITY`` regardless of the request, and
``SHED_CAPACITY`` is below ``QUANTITY_AXIS`` -- so the clamp changes nothing
about what actually gets sold or bought (see
``sim.rollout.SHED_BOUND_VERBS`` for the argument in full). BUY_SEED is not
shed-bound -- seeds never touch the shed, so a large BUY_SEED request has no
such structural ceiling to appeal to -- and a request past the axis for it
still raises rather than risk silently truncating a genuinely large fill.
The largest BUY_SEED request found in the 400-episode scan was 46, so this
has not yet been observed to bind. A route whose only over-axis orders are
shed-bound now replays here to the exact bank it was harvested with, or the
lane is wrong; a route naming an out-of-domain BUY_SEED quantity is still
outside what ``replay`` can encode.
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
