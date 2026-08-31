"""The boundary a learned residual may not cross, and the merge that enforces it.

The residual proposes one quantity bucket per entry of ``ALLOWED_SLOTS``. Every
other part of the frozen controller's turn -- the farmer operation, the hands,
and the ``HIRE``, ``BUY_LAND``, ``BUY_SEED`` and ``BUY_ANIMAL`` orders -- is
copied through untouched. The failure this module exists to prevent is not a
crash: it is a residual that quietly degrades a controller that was winning, by
proposing something the engine then only partly fills.

So the merge is fail-closed by construction. Every path that cannot *prove* the
merged action legal returns the caller's own action object -- not a copy, the
same object -- with one typed ``FallbackReason``, and the whole body runs under
a blanket handler so that an exception raised anywhere inside it, including from
inside the decision the residual handed us, is a fallback rather than a
traceback out of a submission.

Three things are checked from the encoded observation rather than trusted:

* **Masks.** The canonical per-slot legality mask, computed at encoding time
  from the raw observation.
* **Held quantities, cash, and shed room.** Recovered exactly and walked over
  the merged queue in engine order. These duplicate what the mask already
  implies for a single slot, on purpose: the mask is computed per slot as if
  that slot were the only order of the turn, and ten of them together are not
  legal just because each of them is.

The exact recovery the cash walk depends on is not an assumption. Money is an
integer in the engine and is encoded as ``money / 10_000`` in float32, so
``round(money() * 10_000)`` returns it exactly for every bank this game can
reach; ``tests/market_residual/test_actions.py`` pins that against the raw
observations of a real episode.

Where the opponent could intervene, the walk is conservative rather than exact.
Sale proceeds are not credited and our own sales are not added to the book,
because the engine quotes both seats against the same pre-commit inventory and
what our sales actually earn depends on orders we cannot see. Both omissions
push the same way: they overstate what a buy costs and understate what we can
afford, so an action that passes this walk is affordable in the real engine too.
"""

import hashlib
import json
import operator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping, Sequence, SupportsIndex

from kaggriculture.action_codec import (
    MARKET_SLOTS,
    MAX_ORDERS,
    QUANTITIES,
    quantity_of,
)
from kaggriculture.constants import (
    ANIMALS,
    CROPS,
    LAND_ORDER,
    LAND_PRICES,
    MARKET_PARAMS,
    SHED_CAPACITY,
    hire_cost,
    market_price,
)
from kaggriculture.features import (
    MAX_UNITS,
    PRODUCT_NAMES,
    SHED_NAMES,
    EncodedObservation,
)
from kaggriculture.market_residual.schema import (
    ALLOWED_SLOTS,
    BUCKETS,
    LEARNABLE_VERBS,
)

ATOMIC_VERBS = frozenset({"HIRE", "BUY_LAND"})
COUNTED_VERBS = frozenset({"SELL", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL"})
SHED_FILLING_VERBS = frozenset({"BUY_PRODUCT", "BUY_ANIMAL"})

SLOT_OF: dict[tuple[str, str], int] = {
    slot_key: slot for slot, slot_key in enumerate(MARKET_SLOTS)
}
BUCKET_OF_QUANTITY: dict[int, int] = {
    quantity: bucket for bucket, quantity in enumerate(QUANTITIES)
}


class FallbackReason(StrEnum):
    """Why a merge returned the frozen controller's action untouched."""

    NONFINITE = "nonfinite"
    SHAPE = "shape"
    STATE = "state"
    ORDERS = "orders"
    MASK = "mask"
    CASH = "cash"
    HELD = "held"
    SHED = "shed"
    ERROR = "error"


class ResidualMode(StrEnum):
    """What the residual asked the merge to do with the market this turn."""

    USE_KAITO = "use_kaito"
    REPLACE = "replace"


@dataclass(frozen=True)
class ResidualDecision:
    """One residual head output, before anything has been proven about it."""

    mode: ResidualMode
    buckets: tuple[int, ...]
    finite: bool = True


@dataclass(frozen=True)
class MarketOrder:
    """One exactly parsed market order and the list the engine will read.

    ``order`` is the caller's own list for an order copied from the frozen
    controller, so a frozen order survives the merge as the identical object
    rather than as something rebuilt from a parse that could have lost a field.
    """

    verb: str
    item: str
    quantity: int
    order: list[Any]


@dataclass(frozen=True)
class MergeResult:
    """The action the seat will play, and why it is the one it is."""

    action: Mapping[str, Any]
    replaced: bool
    fallback: FallbackReason | None


def exact_cash(encoded: EncodedObservation) -> int:
    """Return the seat's exact bank, recovered from its normalized scalar.

    Args:
        encoded: The canonical observation encoding for the acting seat.

    Returns:
        The integer coin balance the engine holds for this seat.
    """
    return round(encoded.money() * 10_000)


def book_inventory(encoded: EncodedObservation, product: str) -> int:
    """Return the market's exact book inventory for one product.

    Args:
        encoded: The canonical observation encoding for the acting seat.
        product: A canonical product name.

    Returns:
        The integer inventory the engine prices this product against.
    """
    params = MARKET_PARAMS[product]
    return round(encoded.live_inventory(product) * int(params["T"]) + int(params["I0"]))


def exact_quantity(value: object) -> int | None:
    """Return ``value`` as an integer only when it already is one exactly.

    The engine coerces an order quantity with ``int()``, which turns 3.9 into
    three orders silently. Nothing here may rely on that: a quantity we cannot
    read exactly is a quantity we cannot price, mask-check or reproduce in a
    feature row, so it is refused instead of rounded.

    Args:
        value: The third field of a raw engine market order.

    Returns:
        The exact integer, or ``None`` if the value is not an exact integer.
    """
    if isinstance(value, bool) or not isinstance(value, SupportsIndex):
        return None
    return operator.index(value)


def parse_market_orders(action: Mapping[str, Any]) -> tuple[MarketOrder, ...] | None:
    """Parse an engine action's market queue exactly, or refuse it.

    Args:
        action: A frozen-controller action dictionary.

    Returns:
        Every order in queue order, or ``None`` if the queue is not a list of
        orders this vocabulary can represent without coercion.
    """
    if not isinstance(action, Mapping):
        return None
    queue = action.get("market", [])
    if not isinstance(queue, list):
        return None
    parsed: list[MarketOrder] = []
    for order in queue:
        if not isinstance(order, list) or not order:
            return None
        verb = order[0]
        if verb in ATOMIC_VERBS:
            if len(order) != 1:
                return None
            parsed.append(MarketOrder(verb, "", 1, order))
            continue
        if verb not in COUNTED_VERBS or len(order) != 3:
            return None
        item = order[1]
        if (verb, item) not in SLOT_OF:
            return None
        quantity = exact_quantity(order[2])
        if quantity is None or quantity <= 0:
            return None
        parsed.append(MarketOrder(verb, item, quantity, order))
    return tuple(parsed)


def kaito_market_buckets(action: Mapping[str, Any]) -> tuple[int, ...] | None:
    """Return the frozen controller's commodity proposal as canonical buckets.

    One bucket per ``ALLOWED_SLOTS`` entry, in that order. The parse is exact or
    it fails: a quantity outside ``QUANTITIES`` is not rounded to the nearest
    bucket, and a slot ordered twice in one turn is not summed. Either would let
    the feature row claim a proposal the controller did not make, and the
    residual would then be trained against, and merged over, a proposal that was
    never on the board. A real episode of the served controller reaches both
    cases, so this returning ``None`` is an ordinary turn, not an error.

    Args:
        action: A frozen-controller action dictionary.

    Returns:
        The bucket per allowed slot, or ``None`` when the commodity orders are
        not exactly representable.
    """
    parsed = parse_market_orders(action)
    if parsed is None:
        return None
    buckets = dict.fromkeys(ALLOWED_SLOTS, 0)
    seen: set[int] = set()
    for entry in parsed:
        if entry.verb not in LEARNABLE_VERBS:
            continue
        slot = SLOT_OF[(entry.verb, entry.item)]
        if slot in seen or entry.quantity not in BUCKET_OF_QUANTITY:
            return None
        seen.add(slot)
        buckets[slot] = BUCKET_OF_QUANTITY[entry.quantity]
    return tuple(buckets[slot] for slot in ALLOWED_SLOTS)


def event_fingerprint(
    encoded: EncodedObservation, kaito_buckets: Sequence[int], shops: Sequence[bool]
) -> str:
    """Return the digest of exactly the state a market trigger reads.

    Args:
        encoded: The canonical observation encoding for the acting seat.
        kaito_buckets: The frozen controller's proposal per allowed slot.
        shops: Whether each canonical shop is open, in schema order.

    Returns:
        A hex SHA-256 over canonical JSON of the trigger-visible state.
    """
    canonical = json.dumps(
        {
            "prices": [round(encoded.live_price(name) * 1e9) for name in PRODUCT_NAMES],
            "inventory": [book_inventory(encoded, name) for name in PRODUCT_NAMES],
            "supply": [encoded.opponent_public_supply(n) for n in PRODUCT_NAMES],
            "legal": [
                legal_slots(encoded)[index] for index in range(len(ALLOWED_SLOTS))
            ],
            "shops": [bool(value) for value in shops],
            "kaito": [int(bucket) for bucket in kaito_buckets],
        },
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def legal_slots(encoded: EncodedObservation) -> tuple[bool, ...]:
    """Return whether any nonzero quantity is mask-legal, per allowed slot.

    Args:
        encoded: The canonical observation encoding for the acting seat.

    Returns:
        One flag per ``ALLOWED_SLOTS`` entry, in that order.
    """
    return tuple(any(encoded.market_mask[slot][1:]) for slot in ALLOWED_SLOTS)


def merge_residual_action(
    encoded: EncodedObservation,
    action: Mapping[str, Any],
    decision: ResidualDecision,
) -> MergeResult:
    """Apply a residual decision only when the result is provably legal.

    Args:
        encoded: The canonical observation encoding for the acting seat.
        action: The frozen controller's action for this turn.
        decision: What the residual head asked for.

    Returns:
        Either the merged action, or the caller's own action object together
        with the single typed reason the merge refused it.
    """
    try:
        return _merge(encoded, action, decision)
    except Exception:
        return MergeResult(action, False, FallbackReason.ERROR)


def _merge(
    encoded: EncodedObservation,
    action: Mapping[str, Any],
    decision: ResidualDecision,
) -> MergeResult:
    """Merge without the outer guard, so every refusal has a named reason.

    Args:
        encoded: The canonical observation encoding for the acting seat.
        action: The frozen controller's action for this turn.
        decision: What the residual head asked for.

    Returns:
        The merged action, or the original with the reason it was kept.
    """
    if decision.mode is not ResidualMode.REPLACE:
        return MergeResult(action, False, None)
    if not decision.finite:
        return MergeResult(action, False, FallbackReason.NONFINITE)
    buckets = _exact_buckets(decision.buckets)
    if buckets is None:
        return MergeResult(action, False, FallbackReason.SHAPE)
    parsed = parse_market_orders(action)
    if parsed is None or kaito_market_buckets(action) is None:
        return MergeResult(action, False, FallbackReason.STATE)

    frozen = tuple(entry for entry in parsed if entry.verb not in LEARNABLE_VERBS)
    sells, buys = _replacement_orders(encoded, buckets)
    if sells is None or buys is None:
        return MergeResult(action, False, FallbackReason.MASK)
    merged = sells + frozen + buys
    if len(merged) > MAX_ORDERS:
        return MergeResult(action, False, FallbackReason.ORDERS)
    refused = _walk(encoded, merged)
    if refused is not None:
        return MergeResult(action, False, refused)
    return MergeResult(
        {**action, "market": [entry.order for entry in merged]},
        True,
        None,
    )


def _exact_buckets(buckets: Sequence[int]) -> tuple[int, ...] | None:
    """Return the residual's buckets when they are in schema, else ``None``.

    Args:
        buckets: One quantity bucket per ``ALLOWED_SLOTS`` entry.

    Returns:
        The exact buckets, or ``None`` if the shape or a value is out of schema.
    """
    if len(buckets) != len(ALLOWED_SLOTS):
        return None
    exact: list[int] = []
    for bucket in buckets:
        index = exact_quantity(bucket)
        if index is None or not 0 <= index < BUCKETS:
            return None
        exact.append(index)
    return tuple(exact)


def _replacement_orders(
    encoded: EncodedObservation, buckets: Sequence[int]
) -> tuple[tuple[MarketOrder, ...] | None, tuple[MarketOrder, ...] | None]:
    """Build the residual's sells and buys, refusing any the mask forbids.

    Sells are emitted ahead of the frozen orders and buys behind them, so the
    residual can only ever hand the frozen controller more cash and more shed
    room than it planned for, never less.

    Args:
        encoded: The canonical observation encoding for the acting seat.
        buckets: One in-schema quantity bucket per ``ALLOWED_SLOTS`` entry.

    Returns:
        The sell orders and the buy orders, or ``(None, None)`` if any nonzero
        bucket is illegal under the canonical mask.
    """
    sells: list[MarketOrder] = []
    buys: list[MarketOrder] = []
    for slot, bucket in zip(ALLOWED_SLOTS, buckets, strict=True):
        if bucket == 0:
            continue
        if not encoded.market_mask[slot][bucket]:
            return None, None
        verb, item = MARKET_SLOTS[slot]
        quantity = quantity_of(bucket)
        entry = MarketOrder(verb, item, quantity, [verb, item, quantity])
        (sells if verb == "SELL" else buys).append(entry)
    return tuple(sells), tuple(buys)


def _walk(
    encoded: EncodedObservation, merged: Sequence[MarketOrder]
) -> FallbackReason | None:
    """Replay the merged queue against exact state and report the first refusal.

    Args:
        encoded: The canonical observation encoding for the acting seat.
        merged: The full merged queue, in the order the engine will read it.

    Returns:
        The reason the queue cannot be proven legal, or ``None`` when it can.
    """
    refused = _shed_walk(encoded, merged)
    if refused is not None:
        return refused
    spend = _spend(encoded, merged)
    if spend is None:
        return FallbackReason.STATE
    return FallbackReason.CASH if spend > exact_cash(encoded) else None


def _shed_walk(
    encoded: EncodedObservation, merged: Sequence[MarketOrder]
) -> FallbackReason | None:
    """Track held stock and shed room across the merged queue in engine order.

    Args:
        encoded: The canonical observation encoding for the acting seat.
        merged: The full merged queue, in the order the engine will read it.

    Returns:
        ``HELD`` if a sale exceeds stock, ``SHED`` if a purchase overfills the
        shed, or ``None`` when neither happens.
    """
    shed = {item: encoded.shed_count(item) for item in SHED_NAMES}
    held = sum(shed.values())
    for entry in merged:
        if entry.verb == "SELL":
            if shed[entry.item] < entry.quantity:
                return FallbackReason.HELD
            shed[entry.item] -= entry.quantity
            held -= entry.quantity
        elif entry.verb in SHED_FILLING_VERBS:
            held += entry.quantity
            if held > SHED_CAPACITY:
                return FallbackReason.SHED
            shed[entry.item] += entry.quantity
    return None


def _spend(encoded: EncodedObservation, merged: Sequence[MarketOrder]) -> int | None:
    """Return what the merged queue costs, priced exactly and conservatively.

    Args:
        encoded: The canonical observation encoding for the acting seat.
        merged: The full merged queue, in the order the engine will read it.

    Returns:
        The total coin cost, or ``None`` if an order cannot be priced at all --
        a ``BUY_LAND`` with no quadrant left to unlock.
    """
    book = {name: book_inventory(encoded, name) for name in PRODUCT_NAMES}
    hires = round(encoded.scalar("our_hires_today") * MAX_UNITS)
    quadrants = encoded.quadrant_count() - 1
    spend = 0
    for entry in merged:
        if entry.verb == "HIRE":
            spend += hire_cost(hires)
            hires += 1
        elif entry.verb == "BUY_LAND":
            if quadrants >= len(LAND_ORDER):
                return None
            spend += LAND_PRICES[quadrants]
            quadrants += 1
        elif entry.verb == "BUY_SEED":
            spend += entry.quantity * int(CROPS[entry.item]["seed"])
        elif entry.verb == "BUY_ANIMAL":
            spend += entry.quantity * int(ANIMALS[entry.item]["cost"])
        elif entry.verb == "BUY_PRODUCT":
            spend += sum(
                market_price(entry.item, book[entry.item] - offset)
                for offset in range(1, entry.quantity + 1)
            )
            book[entry.item] -= entry.quantity
    return spend
