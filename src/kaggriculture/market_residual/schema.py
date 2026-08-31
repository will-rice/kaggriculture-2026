"""Frozen market-residual layout, learnable slot set, and seed banks.

Three things are fixed here because they are the parts a later mistake cannot
recover from.

The learnable slots are the only market slots the residual may touch. Logistics
and investment -- ``HIRE``, ``BUY_LAND``, ``BUY_SEED``, ``BUY_ANIMAL`` -- stay
with the frozen controller, so they are absent from this set by construction
rather than by a check somewhere downstream.

The feature layout is an ordered list of named blocks and widths, and its
SHA-256 is taken over exactly that -- canonical JSON of names and widths, never
a Python repr, which would change with an unrelated refactor and stay the same
across a rename. The digest is part of run identity: a model trained under one
column order and served under another does not raise, it plays a different
game.

The seed banks are disjoint by declaration and are checked at import time, not
only in a test, so a later widening cannot quietly void a gate. Two ranges are
refused outright by every bank: ``840000..840127``, spent on route promotion
decisions that are already made, and ``700000..700063``, the held-out exam every
promotion this project has taken was graded on. A training bank that reaches
into either is not training against unseen play, it is one more round of the
selection those banks exist to check.
"""

import hashlib
import itertools
import json
from dataclasses import dataclass
from typing import Mapping, Sequence

from kaggriculture.action_codec import MARKET_SLOTS, QUANTITIES
from kaggriculture.constants import SEASON_DAYS, TURNS_PER_DAY
from kaggriculture.features import PRODUCT_NAMES, SHOP_NAMES

SCHEMA_VERSION = 1

LEARNABLE_VERBS = frozenset({"SELL", "BUY_PRODUCT"})

ALLOWED_SLOTS: tuple[int, ...] = tuple(
    index for index, (verb, _item) in enumerate(MARKET_SLOTS) if verb in LEARNABLE_VERBS
)

BUCKETS = len(QUANTITIES)


def kaito_block(slot: int) -> str:
    """Return the block holding the frozen controller's bucket for one slot.

    Args:
        slot: An index into ``action_codec.MARKET_SLOTS``.

    Returns:
        The canonical block name for that slot's proposed quantity.
    """
    verb, item = MARKET_SLOTS[slot]
    return f"kaito:{verb}:{item}"


BLOCKS: tuple[tuple[str, int], ...] = (
    ("day", SEASON_DAYS),
    ("hour", TURNS_PER_DAY),
    *((f"shop_open:{shop}", 1) for shop in SHOP_NAMES),
    *((f"held:{product}", BUCKETS) for product in PRODUCT_NAMES),
    *((f"price:{product}", 1) for product in PRODUCT_NAMES),
    *((f"inventory:{product}", 1) for product in PRODUCT_NAMES),
    *((f"carried:{product}", 1) for product in PRODUCT_NAMES),
    ("shed_free", BUCKETS),
    ("cash", 1),
    ("opponent_cash", 1),
    *((f"opponent_supply:{product}", BUCKETS) for product in PRODUCT_NAMES),
    *((kaito_block(slot), BUCKETS) for slot in ALLOWED_SLOTS),
    *((f"delta_price:{product}", 1) for product in PRODUCT_NAMES),
    *((f"delta_inventory:{product}", 1) for product in PRODUCT_NAMES),
    ("delta_cash", 1),
    ("has_previous_event", 1),
)

if len({name for name, _width in BLOCKS}) != len(BLOCKS):
    raise RuntimeError("market feature block names must be unique")


@dataclass(frozen=True)
class MarketFeatureSchema:
    """One immutable market feature layout as ordered named blocks."""

    blocks: tuple[tuple[str, int], ...]

    @classmethod
    def current(cls) -> "MarketFeatureSchema":
        """Return the single layout every consumer of the residual shares."""
        return CURRENT

    @property
    def width(self) -> int:
        """Return the number of columns in one encoded row."""
        return sum(width for _name, width in self.blocks)

    @property
    def sha256(self) -> str:
        """Return the identity digest over canonical block names and widths."""
        canonical = json.dumps(
            {
                "version": SCHEMA_VERSION,
                "blocks": [[name, width] for name, width in self.blocks],
            },
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def day_slice(self) -> slice:
        """Return the columns holding the one-hot season day."""
        return self.span("day")

    @property
    def hour_slice(self) -> slice:
        """Return the columns holding the one-hot hour of the day."""
        return self.span("hour")

    @property
    def kaito_bucket_slices(self) -> tuple[slice, ...]:
        """Return one column range per ``ALLOWED_SLOTS`` entry, in that order."""
        return tuple(self.span(kaito_block(slot)) for slot in ALLOWED_SLOTS)

    def span(self, name: str) -> slice:
        """Return the columns one named block occupies.

        Args:
            name: A block name declared by this schema.

        Returns:
            The half-open column range of that block.

        Raises:
            KeyError: If this schema declares no such block.
        """
        start = 0
        for block, width in self.blocks:
            if block == name:
                return slice(start, start + width)
            start += width
        raise KeyError(f"no market feature block named {name!r}")


CURRENT = MarketFeatureSchema(BLOCKS)

# Spent on route-promotion decisions that have already been made. Replaying one
# grades a new agent on an exam whose answers are in the record.
CONSUMED_PROMOTION_SEEDS = frozenset(range(840_000, 840_128))

# Mirrors `search.scripts.holdout.GATE_SEEDS`. It is restated rather than
# imported because this package is the submission runtime and may not depend on
# the search tree; `tests/market_residual/test_schema.py` pins the two equal, so
# moving the gate moves this with it.
HELD_OUT_GATE_SEEDS = frozenset(range(700_000, 700_064))

SEED_BANKS: dict[str, tuple[int, ...]] = {
    "counterfactual_train": tuple(range(860_000, 860_512)),
    "counterfactual_temporal": tuple(range(861_000, 861_128)),
    "online_train": tuple(range(870_000, 871_024)),
    "online_gate": tuple(range(872_000, 872_064)),
    "temporal_frontier": tuple(range(880_000, 880_128)),
    "export_determinism": tuple(range(895_000, 895_008)),
    "promotion": tuple(range(890_000, 890_128)),
}


def check_seed_banks(banks: Mapping[str, Sequence[int]]) -> None:
    """Raise unless a set of seed banks is disjoint and reuses nothing spent.

    Called on ``SEED_BANKS`` at import, so widening a bank later cannot quietly
    void a gate, and callable on a candidate set so the rule itself has a test.

    Args:
        banks: Candidate seed banks keyed by name.

    Raises:
        RuntimeError: If two banks share a seed, or a bank reuses a consumed
            promotion seed or a held-out gate seed.
    """
    overlapping = sorted(
        (left, right)
        for left, right in itertools.combinations(sorted(banks), 2)
        if set(banks[left]) & set(banks[right])
    )
    if overlapping:
        raise RuntimeError(
            f"market residual seed banks must be disjoint: {overlapping}"
        )
    consumed = sorted(
        name for name, seeds in banks.items() if set(seeds) & CONSUMED_PROMOTION_SEEDS
    )
    if consumed:
        raise RuntimeError(
            f"market residual seed banks reuse consumed promotion seeds: {consumed}"
        )
    gated = sorted(
        name for name, seeds in banks.items() if set(seeds) & HELD_OUT_GATE_SEEDS
    )
    if gated:
        raise RuntimeError(
            f"market residual seed banks reuse held-out gate seeds: {gated}"
        )


check_seed_banks(SEED_BANKS)


def validate_seed_bank(name: str, seeds: Sequence[int]) -> tuple[int, ...]:
    """Return ``seeds`` when every one of them belongs to the named bank.

    Args:
        name: A bank declared in ``SEED_BANKS``.
        seeds: The seeds a run is about to play.

    Returns:
        The validated seeds, in the order given.

    Raises:
        ValueError: If the bank is unknown, or a seed is a consumed promotion
            seed, a held-out gate seed, or outside the named bank.
    """
    if name not in SEED_BANKS:
        raise ValueError(f"unknown seed bank {name!r}")
    requested = tuple(seeds)
    consumed = sorted(set(requested) & CONSUMED_PROMOTION_SEEDS)
    if consumed:
        raise ValueError(f"consumed promotion seed requested for {name}: {consumed}")
    gate = sorted(set(requested) & HELD_OUT_GATE_SEEDS)
    if gate:
        raise ValueError(f"held-out gate seed requested for {name}: {gate}")
    outside = sorted(set(requested) - set(SEED_BANKS[name]))
    if outside:
        raise ValueError(f"seed outside the {name} bank: {outside}")
    return requested
