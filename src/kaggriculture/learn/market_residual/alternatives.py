"""The small set of things the seat could have done instead, at one event.

Every label the counterfactual dataset carries is a difference between two
actions, and this module names the second one. That makes it the place where a
quiet mistake is most expensive: collection stores an alternative *index*, the
learner reads that index back weeks later, and if the two sides disagree about
which action row three was, every label in the shard is attached to an action
nobody played. Nothing downstream raises on that. So three properties are
built in rather than checked afterwards.

**Deterministic.** The same event state yields the same rows, in the same
order, in any interpreter. Nothing here iterates a set or depends on dict
ordering derived from one; every ordering is an explicit sort with a total
tie-break, and the identity digest is taken over canonical JSON of the families
and buckets rather than over any Python repr.

**Bounded.** The cap is a declared budget, split across families, and
``AlternativeConfig`` refuses at construction any split that does not fit
inside it. A family that returned more than its share would overflow the total,
and the assembly raises rather than truncating: a silently shortened
alternative set is a dataset that is missing exactly the rows the ranking
thought were best.

**Legal.** Legality is not restated here. A candidate survives only if
``merge_residual_action`` -- the same fail-closed merge the submitted runtime
uses -- proves it, so an alternative is playable by construction and is
playable through the code path that will actually play it.

The generator is bound to its event. It recomputes ``event_fingerprint`` from
the state it was handed and refuses to run when that does not match the event's
own digest, so alternatives for one turn cannot be recorded against another.

The ranking is a proposal order, not a claim about value. ``quoted edge`` is
what the market's current quote pays over its neutral-inventory price: selling
into a bare market scores well, buying into a glutted one scores well, and the
per-slot quotes are read independently. Its only jobs are to be cheap, total,
and reproducible, so that the bounded set spends its budget on trades the board
is currently offering a price for. What the trade was actually worth is the
branch outcome, measured later.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from kaggriculture.action_codec import MARKET_SLOTS, QUANTITIES, quantity_of
from kaggriculture.constants import MARKET_PARAMS, market_price
from kaggriculture.features import SHOP_NAMES, EncodedObservation
from kaggriculture.market_residual.actions import (
    ResidualDecision,
    ResidualMode,
    book_inventory,
    event_fingerprint,
    kaito_market_buckets,
    merge_residual_action,
)
from kaggriculture.market_residual.events import MarketEvent
from kaggriculture.market_residual.schema import ALLOWED_SLOTS, BUCKETS

ALTERNATIVE_VERSION = 1

FAMILY_KAITO = "kaito"

ALTERNATIVE_FAMILIES: tuple[str, ...] = (
    FAMILY_KAITO,
    "cancel",
    "scale_down",
    "scale_up",
    "single",
    "ranked_multi",
)

# The three families that are one row each: doing nothing in the market, and
# the controller's own plan at half and at one and a half its size.
FIXED_FAMILIES = ALTERNATIVE_FAMILIES[1:4]

SCALE_DOWN = (1, 2)
SCALE_UP = (3, 2)


@dataclass(frozen=True)
class AlternativeConfig:
    """How many rows each family may contribute, and the total that allows.

    The per-family caps are the budget; ``max_alternatives`` is the hard cap
    the plan fixes. A split that does not fit inside the cap is refused here,
    at construction, so no caller can configure an overflow that only shows up
    on the one event dense enough to reach it.
    """

    max_single: int = 36
    max_ranked_multi: int = 8
    max_alternatives: int = 48

    def __post_init__(self) -> None:
        """Refuse a budget whose families could together exceed the cap.

        Raises:
            ValueError: If the declared per-family rows do not fit the cap.
        """
        declared = 1 + len(FIXED_FAMILIES) + self.max_single + self.max_ranked_multi
        if declared > self.max_alternatives:
            raise ValueError(
                f"alternative families declare {declared} rows, "
                f"above the cap of {self.max_alternatives}"
            )


@dataclass(frozen=True)
class Alternative:
    """One thing the seat could have done, and which family proposed it."""

    family: str
    buckets: tuple[int, ...]


@dataclass(frozen=True)
class AlternativeSet:
    """Every alternative for one market event, with Kaito's own plan first."""

    fingerprint: str
    rows: tuple[Alternative, ...]

    @property
    def sha256(self) -> str:
        """Return the identity digest over the event and every row, in order."""
        canonical = json.dumps(
            {
                "version": ALTERNATIVE_VERSION,
                "event": self.fingerprint,
                "rows": [[row.family, list(row.buckets)] for row in self.rows],
            },
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def generate_alternatives(
    event: MarketEvent,
    encoded: EncodedObservation,
    action: Mapping[str, Any],
    config: AlternativeConfig,
) -> AlternativeSet:
    """Return the bounded, legal, deterministic alternatives for one event.

    Row zero is always the frozen controller's own proposal, which is played by
    preserving its action rather than by replacing it and so is not subject to
    the replacement merge. Every other row is a replacement the merge has
    proved legal, deduplicated on its buckets and ordered lexicographically.

    Args:
        event: The event the residual was consulted at.
        encoded: The canonical observation encoding for the acting seat, which
            must be the state that event was detected on.
        action: The frozen controller's action for that turn.
        config: The row budget in force for this run.

    Returns:
        The alternatives, Kaito first and the rest in lexicographic order.

    Raises:
        ValueError: If the action's commodity orders cannot be read exactly, or
            if the state handed over is not the state the event was opened on.
        RuntimeError: If the assembled rows exceed the configured cap.
    """
    kaito = kaito_market_buckets(action)
    if kaito is None:
        raise ValueError("the controller's commodity orders are not exactly readable")
    if kaito != tuple(event.kaito_buckets):
        raise ValueError(
            f"action proposes {kaito}, event recorded {tuple(event.kaito_buckets)}"
        )
    shops = tuple(encoded.scalar(f"shop:{shop}") != 0.0 for shop in SHOP_NAMES)
    fingerprint = event_fingerprint(encoded, kaito, shops)
    if fingerprint != event.fingerprint:
        raise ValueError("encoded state does not match the event it was given with")

    edges = slot_edges(encoded)
    best = best_buckets(encoded, edges)
    candidates = (
        *_fixed(encoded, action, kaito),
        *_singles(encoded, action, kaito, edges, best, config),
        *_ranked_multi(encoded, action, kaito, edges, best, config),
    )

    rows = [Alternative(FAMILY_KAITO, kaito)]
    seen = {kaito}
    tail: list[Alternative] = []
    for family, buckets in candidates:
        if buckets in seen:
            continue
        seen.add(buckets)
        tail.append(Alternative(family, buckets))
    rows.extend(sorted(tail, key=lambda row: row.buckets))
    if len(rows) > config.max_alternatives:
        raise RuntimeError(
            f"generated {len(rows)} alternatives, above the cap of "
            f"{config.max_alternatives}"
        )
    return AlternativeSet(fingerprint, tuple(rows))


def slot_edges(encoded: EncodedObservation) -> tuple[tuple[int, ...], ...]:
    """Return what each quantity bucket earns over the neutral quote, per slot.

    A sale is scored by what the book pays for it above the price the product
    carries at its neutral inventory, and a purchase by what the book saves
    against that same price. Both are walked unit by unit exactly as the engine
    quotes them, so the marginal edge falls away as the order moves the book.

    Args:
        encoded: The canonical observation encoding for the acting seat.

    Returns:
        One row per ``ALLOWED_SLOTS`` entry, holding the cumulative edge of
        every quantity bucket in bucket order.
    """
    rows: list[tuple[int, ...]] = []
    for slot in ALLOWED_SLOTS:
        verb, item = MARKET_SLOTS[slot]
        inventory = book_inventory(encoded, item)
        reference = market_price(item, int(MARKET_PARAMS[item]["I0"]))
        cumulative = [0]
        running = 0
        unit = 0
        for bucket in range(1, BUCKETS):
            while unit < quantity_of(bucket):
                if verb == "SELL":
                    running += market_price(item, inventory + unit) - reference
                else:
                    running += reference - market_price(item, inventory - 1 - unit)
                unit += 1
            cumulative.append(running)
        rows.append(tuple(cumulative))
    return tuple(rows)


def best_buckets(
    encoded: EncodedObservation, edges: Sequence[Sequence[int]]
) -> tuple[int, ...]:
    """Return the mask-legal quantity with the most quoted edge, per slot.

    Args:
        encoded: The canonical observation encoding for the acting seat.
        edges: The cumulative per-bucket edge from ``slot_edges``.

    Returns:
        One bucket per ``ALLOWED_SLOTS`` entry: the smallest bucket carrying the
        greatest edge, or zero where the mask allows no nonzero quantity.
    """
    chosen: list[int] = []
    for index, slot in enumerate(ALLOWED_SLOTS):
        legal = [
            bucket for bucket in range(1, BUCKETS) if encoded.market_mask[slot][bucket]
        ]
        chosen.append(
            max(legal, key=lambda bucket: (edges[index][bucket], -bucket))
            if legal
            else 0
        )
    return tuple(chosen)


def _merges(
    encoded: EncodedObservation,
    action: Mapping[str, Any],
    buckets: tuple[int, ...],
) -> bool:
    """Return whether the runtime merge proves this replacement legal.

    Args:
        encoded: The canonical observation encoding for the acting seat.
        action: The frozen controller's action for that turn.
        buckets: One quantity bucket per ``ALLOWED_SLOTS`` entry.

    Returns:
        ``True`` when the merge would play the replacement.
    """
    decision = ResidualDecision(ResidualMode.REPLACE, buckets)
    return merge_residual_action(encoded, action, decision).replaced


def _fixed(
    encoded: EncodedObservation,
    action: Mapping[str, Any],
    kaito: tuple[int, ...],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Return the legal one-row families: cancelling, and rescaling the plan.

    Args:
        encoded: The canonical observation encoding for the acting seat.
        action: The frozen controller's action for that turn.
        kaito: The controller's proposal per allowed slot.

    Returns:
        Each of the three families that survives the merge, with its buckets.
    """
    proposed = (
        (FIXED_FAMILIES[0], (0,) * len(ALLOWED_SLOTS)),
        (FIXED_FAMILIES[1], _scaled(kaito, *SCALE_DOWN)),
        (FIXED_FAMILIES[2], _scaled(kaito, *SCALE_UP)),
    )
    return tuple(
        (family, buckets)
        for family, buckets in proposed
        if _merges(encoded, action, buckets)
    )


def _scaled(
    kaito: tuple[int, ...], numerator: int, denominator: int
) -> tuple[int, ...]:
    """Return the controller's quantities rescaled and requantized downward.

    The rescaled quantity is floored onto the canonical vocabulary rather than
    rounded to the nearest bucket, so scaling a plan down never asks the market
    for more than the plan did.

    Args:
        kaito: The controller's proposal per allowed slot.
        numerator: The scale's numerator.
        denominator: The scale's denominator.

    Returns:
        One bucket per allowed slot, in the same order.
    """
    return tuple(
        _floor_bucket(quantity_of(bucket) * numerator // denominator)
        for bucket in kaito
    )


def _floor_bucket(quantity: int) -> int:
    """Return the largest canonical bucket that asks for no more than ``quantity``.

    Args:
        quantity: A non-negative unit count.

    Returns:
        The bucket index whose quantity is the greatest one not above it.
    """
    return max(bucket for bucket, size in enumerate(QUANTITIES) if size <= quantity)


def _singles(
    encoded: EncodedObservation,
    action: Mapping[str, Any],
    kaito: tuple[int, ...],
    edges: Sequence[Sequence[int]],
    best: Sequence[int],
    config: AlternativeConfig,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Return single-slot deviations from the plan, ranked by quoted edge.

    Each candidate changes exactly one slot and leaves the rest of the
    controller's proposal alone. The quantities tried are the two neighbours of
    what the controller asked for and the price-ranked best quantity that slot
    currently admits.

    Args:
        encoded: The canonical observation encoding for the acting seat.
        action: The frozen controller's action for that turn.
        kaito: The controller's proposal per allowed slot.
        edges: The cumulative per-bucket edge from ``slot_edges``.
        best: The price-ranked quantity per slot from ``best_buckets``.
        config: The row budget in force for this run.

    Returns:
        At most ``config.max_single`` legal candidates, best edge first.
    """
    scored: list[tuple[int, int, int, tuple[int, ...]]] = []
    for index, slot in enumerate(ALLOWED_SLOTS):
        wanted = dict.fromkeys((kaito[index] - 1, kaito[index] + 1, best[index]))
        for target in wanted:
            if not 1 <= target < BUCKETS or not encoded.market_mask[slot][target]:
                continue
            buckets = kaito[:index] + (target,) + kaito[index + 1 :]
            if not _merges(encoded, action, buckets):
                continue
            scored.append((edges[index][target], index, target, buckets))
    scored.sort(key=lambda entry: (-entry[0], entry[1], entry[2]))
    return tuple(
        (ALTERNATIVE_FAMILIES[4], entry[3]) for entry in scored[: config.max_single]
    )


def _ranked_multi(
    encoded: EncodedObservation,
    action: Mapping[str, Any],
    kaito: tuple[int, ...],
    edges: Sequence[Sequence[int]],
    best: Sequence[int],
    config: AlternativeConfig,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Return nested multi-slot combinations of the best-quoted trades.

    The slots are ranked once by the edge of their best legal quantity, and the
    combinations are the nested prefixes of that ranking laid over the
    controller's plan. Nesting keeps the family a chain rather than a search:
    the number of combinations grows with the cap, never with the slot count.

    Args:
        encoded: The canonical observation encoding for the acting seat.
        action: The frozen controller's action for that turn.
        kaito: The controller's proposal per allowed slot.
        edges: The cumulative per-bucket edge from ``slot_edges``.
        best: The price-ranked quantity per slot from ``best_buckets``.
        config: The row budget in force for this run.

    Returns:
        At most ``config.max_ranked_multi`` legal combinations, smallest first.
    """
    ranked = sorted(
        (index for index in range(len(ALLOWED_SLOTS)) if best[index] > 0),
        key=lambda index: (-edges[index][best[index]], index),
    )
    combinations: list[tuple[str, tuple[int, ...]]] = []
    for size in range(2, 2 + config.max_ranked_multi):
        if size > len(ranked):
            break
        buckets = list(kaito)
        for index in ranked[:size]:
            buckets[index] = best[index]
        candidate = tuple(buckets)
        if _merges(encoded, action, candidate):
            combinations.append((ALTERNATIVE_FAMILIES[5], candidate))
    return tuple(combinations)
