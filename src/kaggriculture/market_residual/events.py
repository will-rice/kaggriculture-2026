"""When the residual is asked for an opinion, and when the turn is not its turn.

A residual that ran every turn would be a second controller. This is the state
machine that decides which turns the market has actually moved on, so the
learned head is a reaction to something rather than a coin flip laid over a
controller that was already winning.

How dense that is, is a measurement rather than an intention. Against the served
controller at the declared thresholds it opens 982 of 1440 seat-turns, and
``tests/market_residual/test_events.py`` pins that share and the trigger it
rests on: opponent supply, whose two-unit threshold a growing opponent farm
crosses most turns. The heartbeat never expires against that controller. Anyone
tightening this machine should move ``EventConfig`` and re-measure, not reason
about it.

Two rules keep it honest.

The first is that a trigger fires on a *change* since the last event, not on a
level. Levels are never surprising: prices are always high or low, the shed is
always fuller than it was. The reference every threshold is measured against is
the state as of the last event this episode, carried in ``EventMemory``.

The second is the fingerprint. ``event_fingerprint`` hashes exactly the fields
the triggers read -- prices, book inventories, opponent supply, the legal slot
set, the open shops, and the frozen controller's own proposal -- and nothing
else. That makes an unchanged digest a proof rather than a heuristic: if it
matches the digest recorded at the last event, no state trigger can have a new
reason to fire, and the ones that do not read state at all, above all a frozen
controller repeating the same proposal turn after turn, are suppressed with it.
The clock triggers -- the heartbeat and the one-shot liquidation switch -- are
exempt, because their reason to fire is time passing, which the digest does not
and should not see.

The clock lives in ``EventMemory.turn`` and the caller advances it once per
engine turn with ``advance``. It is not read from the observation because
counterfactual collection replays the same observation at different points in an
episode, and a clock recovered from the board would make those replays
indistinguishable.
"""

from dataclasses import dataclass, replace
from typing import Any, Mapping

from kaggriculture.features import PRODUCT_NAMES, SHOP_NAMES, EncodedObservation
from kaggriculture.market_residual.actions import (
    event_fingerprint,
    kaito_market_buckets,
    legal_slots,
)


@dataclass(frozen=True)
class EventConfig:
    """The thresholds that decide when the market has moved enough to matter."""

    price_relative: float = 0.05
    inventory_relative: float = 0.10
    opponent_supply_units: int = 2
    heartbeat_turns: int = 24
    liquidation_day: int = 26


@dataclass(frozen=True)
class MarketEvent:
    """One turn the residual is asked about, and every reason it was asked."""

    turn: int
    day: int
    triggers: tuple[str, ...]
    fingerprint: str
    kaito_buckets: tuple[int, ...]


@dataclass(frozen=True)
class EventMemory:
    """The clock, and the state the next set of thresholds is measured against.

    Every reference field is the value observed at the last event of this
    episode, so a slow drift that never crosses a threshold between two adjacent
    turns still crosses it eventually.
    """

    turn: int
    last_event_turn: int | None
    fingerprint: str | None
    prices: tuple[float, ...]
    inventories: tuple[float, ...]
    supply: tuple[int, ...]
    shops: tuple[bool, ...]
    legal: tuple[bool, ...]
    liquidated: bool

    @classmethod
    def initial(cls) -> "EventMemory":
        """Return the memory an episode starts from, before any event."""
        return cls(
            turn=0,
            last_event_turn=None,
            fingerprint=None,
            prices=(),
            inventories=(),
            supply=(),
            shops=(),
            legal=(),
            liquidated=False,
        )

    def advance(self, turns: int) -> "EventMemory":
        """Return this memory with the clock moved forward.

        Args:
            turns: How many engine turns have passed.

        Returns:
            The same references and fingerprint at a later turn.
        """
        return replace(self, turn=self.turn + turns)


@dataclass(frozen=True)
class EventTransition:
    """The event this turn produced, if any, and the memory to carry forward."""

    event: MarketEvent | None
    memory: EventMemory


def detect_event(
    encoded: EncodedObservation,
    action: Mapping[str, Any],
    memory: EventMemory,
    config: EventConfig,
) -> EventTransition:
    """Decide whether this turn is a market event for the residual.

    A turn whose frozen proposal cannot be read exactly into canonical buckets
    is never an event: the feature row for it would have to claim a proposal the
    controller did not make, so the turn is left to the controller alone.

    Args:
        encoded: The canonical observation encoding for the acting seat.
        action: The frozen controller's action for this turn.
        memory: The clock and references carried from the last event.
        config: The thresholds in force for this run.

    Returns:
        The event and the memory it updates, or no event and the memory
        unchanged.
    """
    buckets = kaito_market_buckets(action)
    if buckets is None:
        return EventTransition(None, memory)

    shops = tuple(encoded.scalar(f"shop:{shop}") != 0.0 for shop in SHOP_NAMES)
    legal = legal_slots(encoded)
    fingerprint = event_fingerprint(encoded, buckets, shops)
    day = encoded.day_count()

    triggers: list[str] = []
    if fingerprint != memory.fingerprint:
        triggers.extend(_state_triggers(encoded, buckets, legal, shops, memory, config))
    if day >= config.liquidation_day and not memory.liquidated:
        triggers.append("liquidation")
    elapsed = (
        None if memory.last_event_turn is None else memory.turn - memory.last_event_turn
    )
    if elapsed is not None and elapsed >= config.heartbeat_turns:
        triggers.append("heartbeat")
    if not triggers:
        return EventTransition(None, memory)

    event = MarketEvent(
        turn=memory.turn,
        day=day,
        triggers=tuple(triggers),
        fingerprint=fingerprint,
        kaito_buckets=buckets,
    )
    return EventTransition(
        event,
        replace(
            memory,
            last_event_turn=memory.turn,
            fingerprint=fingerprint,
            prices=tuple(encoded.live_price(name) for name in PRODUCT_NAMES),
            inventories=tuple(encoded.live_inventory(name) for name in PRODUCT_NAMES),
            supply=tuple(encoded.opponent_public_supply(n) for n in PRODUCT_NAMES),
            shops=shops,
            legal=legal,
            liquidated=memory.liquidated or "liquidation" in triggers,
        ),
    )


def _state_triggers(
    encoded: EncodedObservation,
    buckets: tuple[int, ...],
    legal: tuple[bool, ...],
    shops: tuple[bool, ...],
    memory: EventMemory,
    config: EventConfig,
) -> tuple[str, ...]:
    """Return the triggers that read board state, in a fixed order.

    Args:
        encoded: The canonical observation encoding for the acting seat.
        buckets: The frozen controller's proposal per allowed slot.
        legal: Whether each allowed slot admits a nonzero quantity now.
        shops: Whether each canonical shop is open now.
        memory: The clock and references carried from the last event.
        config: The thresholds in force for this run.

    Returns:
        Every state trigger that fires this turn.
    """
    if memory.fingerprint is None:
        return ("first",)
    triggers: list[str] = []
    if any(bucket != 0 for bucket in buckets):
        triggers.append("kaito")
    if any(now and not before for now, before in zip(legal, memory.legal, strict=True)):
        triggers.append("slot")
    prices = tuple(encoded.live_price(name) for name in PRODUCT_NAMES)
    if _crossed(prices, memory.prices, config.price_relative):
        triggers.append("price")
    inventories = tuple(encoded.live_inventory(name) for name in PRODUCT_NAMES)
    if _crossed(inventories, memory.inventories, config.inventory_relative):
        triggers.append("inventory")
    if shops != memory.shops:
        triggers.append("shop")
    supply = tuple(encoded.opponent_public_supply(name) for name in PRODUCT_NAMES)
    if _crossed(supply, memory.supply, config.opponent_supply_units, strict=False):
        triggers.append("opponent_supply")
    return tuple(triggers)


def _crossed(
    now: tuple[float, ...],
    before: tuple[float, ...],
    threshold: float,
    strict: bool = True,
) -> bool:
    """Return whether any paired value moved past a threshold.

    Args:
        now: This turn's values.
        before: The values recorded at the last event.
        threshold: The move that counts.
        strict: Whether the move must exceed the threshold or may equal it.

    Returns:
        ``True`` when at least one pair crossed.
    """
    moves = (abs(a - b) for a, b in zip(now, before, strict=True))
    return any(move > threshold if strict else move >= threshold for move in moves)
