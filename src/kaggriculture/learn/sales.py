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

``buy_units`` is the mirror image and is here rather than beside it because it
is read from the same consecutive pair, on the same completion discipline. It
answers a different question: ``toad_reward.Counts.fuel`` pays a flat rate per
unit of shed stock and charges nothing for the coins that bought it, so cheap
goods are free shaped reward, and this is the counter that says whether an arm
has found that out.
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

    Only goods the market prices are candidates. The shed also holds animals
    waiting to be placed, and an animal is not something the market buys.

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
        prices = before["market"]["prices"]
        # The market book is what defines a sellable good, so it is what the
        # scan iterates. The shed is wider than the book -- BUY_ANIMAL parks a
        # COW there until it is placed -- and an animal leaving the shed on a
        # turn the bank happens to rise is a placement, not a sale. Reading the
        # shed instead and pricing whatever fell out of it asks the book for a
        # COW, which is a KeyError, not a wrong number: it took down two of the
        # four evaluated arms the first time this ran against real episodes.
        sold = {
            good: shed_before[good] - shed_after.get(good, 0)
            for good in prices
            if shed_before[good] > shed_after.get(good, 0)
        }
        if not sold:
            continue
        sales += len(sold)
        units += sum(sold.values())
        proceeds += gained
        for good, count in sold.items():
            weighted_market.extend([float(prices[good])] * count)

    mean_market = (
        sum(weighted_market) / len(weighted_market) if weighted_market else 0.0
    )
    mean_sale = proceeds / units if units else 0.0
    return {
        "sales": float(sales),
        "units": float(units),
        "mean_sale_price": mean_sale,
        "mean_market_price": mean_market,
        "price_realisation": (mean_sale / mean_market) if mean_market else 0.0,
    }


def buy_units(observations: Sequence[Mapping[str, Any]], seat: int) -> float:
    """Return how many units purchases credited to the shed, over one episode.

    The mirror image of the sale inference, and for the same reason: the engine
    drops orders it cannot fill, so a purchase is read from what happened to the
    observation -- shed stock rising while the bank falls -- rather than from
    what the policy emitted.

    The bank falling is what separates a purchase from a harvest. ``HARVEST``
    credits the shed out of the field and costs nothing, so a turn whose bank did
    not fall contributes nothing here no matter how much stock appeared. Without
    that gate the counter would climb through ordinary farming and the tripwire
    it feeds would fire on a policy doing exactly what it should.

    Animals count. ``BUY_ANIMAL`` is one of the two verbs under watch, and unlike
    a sale a purchase needs no market price -- this is a unit count, which is
    precisely what makes it comparable to the ``fuel`` term it exists to watch.

    Attribution is per turn, as it is for sales: a turn that both harvests and
    buys credits every new unit to the purchase, because the observation cannot
    separate them. That inflates the count on mixed turns and the threshold it
    is compared against is three orders of magnitude above the noise, which is
    the trade this is making deliberately.

    Args:
        observations: One seat's observation at each state, in order. The shed
            lives in the unindexed ``private`` mapping, so these must be that
            seat's own observations.
        seat: Which seat to score.

    Returns:
        Units credited to the shed on turns the bank fell.
    """
    bought = 0
    for before, after in zip(observations[:-1], observations[1:], strict=True):
        spent = float(before["farms"][seat]["money"]) - float(
            after["farms"][seat]["money"]
        )
        if spent <= 0.0:
            continue
        shed_before = before["private"]["shed"]
        bought += sum(
            max(count - shed_before.get(good, 0), 0)
            for good, count in after["private"]["shed"].items()
        )
    return float(bought)
