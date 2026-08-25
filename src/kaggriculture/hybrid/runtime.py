"""Frozen, dependency-free hybrid configuration used by the turn loop."""

from dataclasses import asdict, dataclass
from typing import Mapping, Self, cast


@dataclass(frozen=True)
class RuntimeOpeningPhase:
    """One validated opening milestone in runtime form."""

    start_day: int
    target_hands: int
    target_quadrants: int
    primary_crop: str
    crop_targets: tuple[int, ...]
    animal_targets: tuple[int, ...]
    structure_targets: tuple[int, int]
    cash_reserve: int
    inventory_reserve: int


@dataclass(frozen=True)
class RuntimeJobWeights:
    """Runtime weights for guarded unit-job ranking."""

    recovery: float
    urgency: float
    distance_penalty: float
    production: float
    transport: float
    structure: float


@dataclass(frozen=True)
class RuntimeMarketWeights:
    """Runtime weights for live market ranking."""

    live_price: float
    town_demand: float
    opponent_supply: float
    reserve_penalty: float
    liquidation_urgency: float


_PHASE_KEYS = frozenset(
    {
        "start_day",
        "target_hands",
        "target_quadrants",
        "primary_crop",
        "crop_targets",
        "animal_targets",
        "structure_targets",
        "cash_reserve",
        "inventory_reserve",
    }
)
_JOB_KEYS = frozenset(
    {
        "recovery",
        "urgency",
        "distance_penalty",
        "production",
        "transport",
        "structure",
    }
)
_MARKET_KEYS = frozenset(
    {
        "live_price",
        "town_demand",
        "opponent_supply",
        "reserve_penalty",
        "liquidation_urgency",
    }
)
_RUNTIME_KEYS = frozenset({"phases", "jobs", "market", "liquidation_start_day"})


def _require_keys(
    value: object, expected: frozenset[str], label: str
) -> dict[str, object]:
    """Return one generated dictionary after checking its exact schema."""
    if type(value) is not dict:
        raise TypeError(f"{label} must be a dictionary")
    if set(value) != expected:
        raise ValueError(f"{label} keys must be {sorted(expected)}")
    return cast(dict[str, object], value)


def _require_mapping_keys(
    value: Mapping[str, object], expected: frozenset[str], label: str
) -> Mapping[str, object]:
    """Return a top-level payload mapping after checking its exact keys."""
    if set(value) != expected:
        raise ValueError(f"{label} keys must be {sorted(expected)}")
    return value


def _require_int(value: object, label: str) -> int:
    """Return a generated integer without accepting booleans or coercions."""
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    return value


def _require_float(value: object, label: str) -> float:
    """Return a generated float without accepting booleans or coercions."""
    if type(value) is not float:
        raise TypeError(f"{label} must be a float")
    return value


def _require_int_tuple(
    value: object, length: int | None, label: str
) -> tuple[int, ...]:
    """Return one exact immutable integer vector from a generated payload."""
    if type(value) is not tuple:
        raise TypeError(f"{label} must be a tuple of integers")
    items = value
    if length is not None and len(items) != length:
        raise TypeError(f"{label} must be a tuple of {length} integers")
    return tuple(_require_int(item, label) for item in items)


def _phase_from_payload(value: object) -> RuntimeOpeningPhase:
    """Build one runtime phase after recursive exact-schema checks."""
    payload = _require_keys(value, _PHASE_KEYS, "phase")
    primary_crop = payload["primary_crop"]
    if type(primary_crop) is not str:
        raise TypeError("phase primary_crop must be a string")
    structure_targets = _require_int_tuple(
        payload["structure_targets"], 2, "phase structure_targets"
    )
    return RuntimeOpeningPhase(
        start_day=_require_int(payload["start_day"], "phase start_day"),
        target_hands=_require_int(payload["target_hands"], "phase target_hands"),
        target_quadrants=_require_int(
            payload["target_quadrants"], "phase target_quadrants"
        ),
        primary_crop=primary_crop,
        crop_targets=_require_int_tuple(
            payload["crop_targets"], None, "phase crop_targets"
        ),
        animal_targets=_require_int_tuple(
            payload["animal_targets"], None, "phase animal_targets"
        ),
        structure_targets=(structure_targets[0], structure_targets[1]),
        cash_reserve=_require_int(payload["cash_reserve"], "phase cash_reserve"),
        inventory_reserve=_require_int(
            payload["inventory_reserve"], "phase inventory_reserve"
        ),
    )


def _weights_from_payload(
    value: object, expected: frozenset[str], label: str
) -> dict[str, float]:
    """Validate and return one exact runtime float-weight dictionary."""
    payload = _require_keys(value, expected, label)
    return {name: _require_float(payload[name], f"{label} {name}") for name in expected}


@dataclass(frozen=True)
class RuntimeConfig:
    """The complete submission-safe hybrid strategy configuration."""

    phases: tuple[RuntimeOpeningPhase, ...]
    jobs: RuntimeJobWeights
    market: RuntimeMarketWeights
    liquidation_start_day: int

    def to_payload(self) -> dict[str, object]:
        """Return the exact literal payload accepted by ``from_payload``."""
        return {
            "phases": [asdict(phase) for phase in self.phases],
            "jobs": asdict(self.jobs),
            "market": asdict(self.market),
            "liquidation_start_day": self.liquidation_start_day,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> Self:
        """Restore a frozen runtime config from a generated literal payload."""
        outer = _require_mapping_keys(payload, _RUNTIME_KEYS, "runtime payload")
        phases = outer["phases"]
        if type(phases) is not list:
            raise TypeError("phases must be a list of dictionaries")
        jobs = _weights_from_payload(outer["jobs"], _JOB_KEYS, "job weights")
        market = _weights_from_payload(outer["market"], _MARKET_KEYS, "market weights")
        return cls(
            phases=tuple(_phase_from_payload(phase) for phase in phases),
            jobs=RuntimeJobWeights(
                recovery=jobs["recovery"],
                urgency=jobs["urgency"],
                distance_penalty=jobs["distance_penalty"],
                production=jobs["production"],
                transport=jobs["transport"],
                structure=jobs["structure"],
            ),
            market=RuntimeMarketWeights(
                live_price=market["live_price"],
                town_demand=market["town_demand"],
                opponent_supply=market["opponent_supply"],
                reserve_penalty=market["reserve_penalty"],
                liquidation_urgency=market["liquidation_urgency"],
            ),
            liquidation_start_day=_require_int(
                outer["liquidation_start_day"], "liquidation_start_day"
            ),
        )
