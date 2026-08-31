"""The one canonical market feature vector shared by every consumer.

Collection, Torch training and the exported NumPy runtime all call
``market_feature_vector``. A second implementation that agreed with this one on
the day it was written is the failure this module exists to prevent, so the row
is assembled by walking the declared schema and asking for each block by name:
a block this file forgets to build raises here rather than shifting every
column after it by its width.
"""

from dataclasses import dataclass
from typing import Sequence

from kaggriculture.action_codec import bucket_of
from kaggriculture.constants import SEASON_DAYS, SHED_CAPACITY, TURNS_PER_DAY
from kaggriculture.features import (
    PRODUCT_NAMES,
    SHED_NAMES,
    SHOP_NAMES,
    EncodedObservation,
)
from kaggriculture.market_residual.schema import (
    ALLOWED_SLOTS,
    BUCKETS,
    MarketFeatureSchema,
    kaito_block,
)


@dataclass(frozen=True)
class MarketFeatureVector:
    """One encoded market decision and the layout it was encoded under."""

    schema: MarketFeatureSchema
    values: tuple[float, ...]

    def block(self, name: str) -> tuple[float, ...]:
        """Return the columns of one named block.

        Args:
            name: A block name declared by this row's schema.

        Returns:
            The block's columns, in schema order.
        """
        return self.values[self.schema.span(name)]


def _one_hot(value: int, width: int, label: str) -> tuple[float, ...]:
    """Return the one-hot row for an integer input.

    Args:
        value: The index to set.
        width: The number of columns in the block.
        label: The block name, used to name the failure.

    Returns:
        ``width`` columns with exactly one 1.0.

    Raises:
        ValueError: If ``value`` is outside the block.
    """
    if not 0 <= value < width:
        raise ValueError(f"{label}={value} outside [0, {width})")
    return tuple(float(index == value) for index in range(width))


def market_feature_vector(
    encoded: EncodedObservation,
    kaito_buckets: Sequence[int],
    previous: MarketFeatureVector | None,
) -> MarketFeatureVector:
    """Encode one market decision into the canonical feature row.

    Args:
        encoded: The canonical observation encoding for the acting seat.
        kaito_buckets: The frozen controller's proposed quantity bucket for
            every entry of ``ALLOWED_SLOTS``, in that order.
        previous: The row built at the previous market event of this episode,
            or ``None`` at its first event.

    Returns:
        The immutable feature row and the schema it was built under.

    Raises:
        ValueError: If ``kaito_buckets`` has the wrong length, a value falls
            outside its block, or ``previous`` was built under another schema.
    """
    schema = MarketFeatureSchema.current()
    if len(kaito_buckets) != len(ALLOWED_SLOTS):
        raise ValueError(
            f"expected {len(ALLOWED_SLOTS)} Kaito buckets, got {len(kaito_buckets)}"
        )
    if previous is not None and previous.schema.sha256 != schema.sha256:
        raise ValueError("previous market row was built under another schema")

    cash = encoded.money()
    free = SHED_CAPACITY - sum(encoded.shed_count(item) for item in SHED_NAMES)
    prices = {product: encoded.live_price(product) for product in PRODUCT_NAMES}
    inventories = {
        product: encoded.live_inventory(product) for product in PRODUCT_NAMES
    }
    columns: dict[str, tuple[float, ...]] = {
        "day": _one_hot(encoded.day_count(), SEASON_DAYS, "day"),
        "hour": _one_hot(
            round(encoded.scalar("hour") * TURNS_PER_DAY), TURNS_PER_DAY, "hour"
        ),
        "shed_free": _one_hot(bucket_of(free), BUCKETS, "shed_free"),
        "cash": (cash,),
        "opponent_cash": (encoded.money(opponent=True),),
        "has_previous_event": (float(previous is not None),),
        "delta_cash": (0.0 if previous is None else cash - previous.block("cash")[0],),
    }
    columns.update(
        {f"shop_open:{shop}": (encoded.scalar(f"shop:{shop}"),) for shop in SHOP_NAMES}
    )
    columns.update({f"price:{name}": (value,) for name, value in prices.items()})
    columns.update(
        {f"inventory:{name}": (value,) for name, value in inventories.items()}
    )
    columns.update(
        {
            f"carried:{product}": (encoded.scalar(f"carried:{product}"),)
            for product in PRODUCT_NAMES
        }
    )
    columns.update(
        {
            f"held:{product}": _one_hot(
                bucket_of(encoded.shed_count(product)), BUCKETS, f"held:{product}"
            )
            for product in PRODUCT_NAMES
        }
    )
    columns.update(
        {
            f"opponent_supply:{product}": _one_hot(
                bucket_of(encoded.opponent_public_supply(product)),
                BUCKETS,
                f"opponent_supply:{product}",
            )
            for product in PRODUCT_NAMES
        }
    )
    columns.update(
        {
            kaito_block(slot): _one_hot(bucket, BUCKETS, kaito_block(slot))
            for slot, bucket in zip(ALLOWED_SLOTS, kaito_buckets, strict=True)
        }
    )
    columns.update(
        {
            f"delta_{field}:{product}": (
                0.0
                if previous is None
                else value - previous.block(f"{field}:{product}")[0],
            )
            for field, live in (("price", prices), ("inventory", inventories))
            for product, value in live.items()
        }
    )

    row: list[float] = []
    for name, width in schema.blocks:
        column = columns[name]
        if len(column) != width:
            raise ValueError(f"block {name} produced {len(column)} of {width} columns")
        row.extend(column)
    return MarketFeatureVector(schema=schema, values=tuple(row))
