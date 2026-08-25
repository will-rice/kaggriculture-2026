"""Fixed all-opponent objective and measured-frontier strength tiers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from kaggriculture.search.frontier import FrontierReport


@dataclass(frozen=True, init=False)
class StrengthWeights:
    """Immutable positive integer weights for one exact named league."""

    values: Mapping[str, int]

    def __init__(self, values: Mapping[str, int]) -> None:
        """Copy and validate a mapping so callers cannot mutate search identity."""
        copied = dict(values)
        if not copied:
            raise ValueError("strength weights must not be empty")
        if any(type(name) is not str or not name for name in copied):
            raise ValueError("strength weight names must be non-empty strings")
        if any(type(weight) is not int or weight <= 0 for weight in copied.values()):
            raise ValueError("strength weights must be positive integers")
        object.__setattr__(
            self, "values", MappingProxyType(dict(sorted(copied.items())))
        )

    def as_dict(self) -> dict[str, int]:
        """Return a JSON-ready copy in the fixed league order."""
        return dict(self.values)


@dataclass(frozen=True)
class Fitness:
    """Primary objective components and hard eligibility ruling."""

    value: float
    weighted_mean: float
    worst: float
    eligible: bool
    failures: int


def score_fitness(
    rates: Mapping[str, float],
    weights: StrengthWeights,
    *,
    failures: int = 0,
) -> Fitness:
    """Score every named matchup, making any execution failure ineligible."""
    if set(rates) != set(weights.values):
        raise ValueError("matchup names must exactly match strength weight names")
    if type(failures) is not int or failures < 0:
        raise ValueError("failures must be a non-negative integer")
    copied_rates = dict(rates)
    if any(
        not isinstance(rate, (int, float))
        or isinstance(rate, bool)
        or not math.isfinite(rate)
        or not 0.0 <= rate <= 1.0
        for rate in copied_rates.values()
    ):
        raise ValueError("matchup rates must be finite values between zero and one")
    total_weight = sum(weights.values.values())
    weighted_mean = (
        sum(
            float(copied_rates[name]) * weight
            for name, weight in weights.values.items()
        )
        / total_weight
    )
    worst = min(float(rate) for rate in copied_rates.values())
    eligible = failures == 0
    return Fitness(
        value=0.70 * weighted_mean + 0.30 * worst if eligible else -math.inf,
        weighted_mean=weighted_mean,
        worst=worst,
        eligible=eligible,
        failures=failures,
    )


def strength_weights(report: FrontierReport) -> StrengthWeights:
    """Assign fixed 4/3/2/1 quartiles from a measured strongest-first report."""
    names = tuple(row.name for row in report.rows)
    if not names:
        raise ValueError("frontier report must contain at least one ranked row")
    if len(names) != len(set(names)):
        raise ValueError("frontier report row names must be unique")
    tiers = (4, 3, 2, 1)
    count = len(names)
    values = {
        name: tiers[min(index * len(tiers) // count, len(tiers) - 1)]
        for index, name in enumerate(names)
    }
    for name in names:
        if name.startswith("boatlee_v14"):
            values[name] = max(values[name], 3)
    return StrengthWeights(values)
