"""Measure completed sales and price realisation from a sequence of observations.

Two different rewards follow from two different gaps, and nothing we log today
tells them apart. If sales stay rare while the bank climbs, the gap is knowing
*when* to sell and the fix is per-transaction shaping. If sales are frequent and
the bank stays low, the policy is dumping into its own depressed prices and the
fix is about price realisation. This module supplies the number that decides it.

It counts what the ENGINE completed, never what the policy emitted. The engine
drops orders it cannot fill, and this project has already been bitten by
conflating the two; a counter reading emitted orders would report sales as
frequent when they are rare, and send the reward design in exactly the wrong
direction.
"""

from collections.abc import Mapping, Sequence
from typing import Any


def sale_metrics(
    observations: Sequence[Mapping[str, Any]], seat: int
) -> dict[str, float]:
    """Return completed-sale counts and price realisation for one episode.

    A completed sale is inferred from the pair of facts only a completion
    produces: a good's shed stock falls while the bank rises on the same turn.
    An order the engine dropped moves neither, so it cannot be counted.

    Proceeds are attributed per turn rather than per good, because the bank is a
    single number and several goods can clear together. The realised price is
    therefore the turn's coins gained divided by the turn's units sold, and the
    market reference is volume-weighted so that a large clear of a cheap good
    does not read the same as one unit of an expensive one.

    Args:
        observations: One seat's observation at each state, in order. The shed
            lives in the unindexed ``private`` mapping, so these must be that
            seat's own observations.
        seat: Which seat to score.

    Returns:
        ``sales`` (completed clears), ``units``, ``mean_sale_price``,
        ``mean_market_price``, and ``price_realisation`` -- realised over
        market, where 1.0 is par and below 1.0 means selling into a depressed
        book.
    """
    sales = 0
    units = 0
    proceeds = 0.0
    weighted_market: list[float] = []
    for before, after in zip(observations[:-1], observations[1:], strict=True):
        gained = float(after["farms"][seat]["money"]) - float(
            before["farms"][seat]["money"]
        )
        if gained <= 0.0:
            continue
        shed_before = before["private"]["shed"]
        shed_after = after["private"]["shed"]
        sold = {
            good: shed_before[good] - shed_after.get(good, 0)
            for good in shed_before
            if shed_before[good] > shed_after.get(good, 0)
        }
        if not sold:
            continue
        prices = before["market"]["prices"]
        sales += len(sold)
        units += sum(sold.values())
        proceeds += gained
        for good, count in sold.items():
            weighted_market.extend([float(prices[good])] * count)

    mean_market = sum(weighted_market) / len(weighted_market) if weighted_market else 0.0
    mean_sale = proceeds / units if units else 0.0
    return {
        "sales": float(sales),
        "units": float(units),
        "mean_sale_price": mean_sale,
        "mean_market_price": mean_market,
        "price_realisation": (mean_sale / mean_market) if mean_market else 0.0,
    }
