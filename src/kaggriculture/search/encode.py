"""Encoding a route into the tensors ``sim.engine.step`` consumes.

A route is the same for every game in a batch, so it is encoded once and the
per-turn rows are broadcast across the batch at replay time. That is the whole
reason a route replays without host synchronisation and a scripted Python
opponent does not.
"""

import dataclasses

import torch

from kaggriculture.learn.encoding import MAX_UNITS
from kaggriculture.search.route import Route
from kaggriculture.sim.rollout import encode_turn


@dataclasses.dataclass(frozen=True)
class EncodedRoute:
    """One route's whole season as device tensors, indexed by turn.

    Attributes:
        units: ``(turns, MAX_UNITS)`` int16 op indices.
        order_type: ``(turns, 10)`` int8 market order kinds.
        order_item: ``(turns, 10)`` int8 catalogue indices, -1 where unused.
        order_qty: ``(turns, 10)`` int32 quantities.
    """

    units: torch.Tensor
    order_type: torch.Tensor
    order_item: torch.Tensor
    order_qty: torch.Tensor


def encode_route(route: Route, device: torch.device | str) -> EncodedRoute:
    """Return ``route`` as per-turn tensors on ``device``.

    Args:
        route: One seat's season in the engine's action grammar.
        device: Where the tensors are wanted.

    Returns:
        The encoded route.
    """
    turns = [encode_turn(action) for action in route]
    units = torch.tensor(
        [turn.units for turn in turns], dtype=torch.int16, device=device
    )
    orders = [[list(order) for order in turn.orders] for turn in turns]
    stacked = torch.tensor(orders, dtype=torch.int64, device=device)
    return EncodedRoute(
        units=units.view(len(turns), MAX_UNITS),
        order_type=stacked[:, :, 0].to(torch.int8),
        order_item=stacked[:, :, 1].to(torch.int8),
        order_qty=stacked[:, :, 2].to(torch.int32),
    )
