"""One-hot evolutionary genome codec for strict hybrid configurations."""

import copy
import math
from dataclasses import dataclass
from typing import Sequence, cast

from kaggriculture.features import ANIMAL_NAMES, CROP_NAMES
from kaggriculture.hybrid.config import HybridConfig
from kaggriculture.hybrid.schema import PHASE_START_DAYS

HAND_TARGETS = tuple(range(20))
QUADRANT_TARGETS = tuple(range(1, 6))
TARGET_COUNTS = tuple(range(101))
CASH_RESERVES = tuple(range(0, 5_001, 100))
INVENTORY_RESERVES = tuple(range(101))
LIQUIDATION_DAYS = tuple(range(20, 30))
JOB_WEIGHT_FIELDS = (
    "recovery",
    "urgency",
    "distance_penalty",
    "production",
    "transport",
    "structure",
)
MARKET_WEIGHT_FIELDS = (
    "live_price",
    "town_demand",
    "opponent_supply",
    "reserve_penalty",
    "liquidation_urgency",
)


@dataclass(frozen=True)
class DiscreteGroup:
    """One categorical argmax group in the flat optimizer vector."""

    path: tuple[str | int, ...]
    values: tuple[object, ...]
    start: int

    @property
    def width(self) -> int:
        """Return the contiguous logit width occupied by this category."""
        return len(self.values)


@dataclass(frozen=True)
class ContinuousGene:
    """One genuinely continuous weight encoded over the unit interval."""

    path: tuple[str | int, ...]
    low: float
    high: float
    index: int


def assign_path(
    payload: dict[str, object], path: tuple[str | int, ...], value: object
) -> None:
    """Set one nested JSON-compatible value by its fixed schema path."""
    current: object = payload
    for part in path[:-1]:
        if isinstance(part, str):
            if type(current) is not dict:
                raise TypeError("string paths require dictionary containers")
            current = cast(dict[str, object], current)[part]
        else:
            if type(current) is not list:
                raise TypeError("integer paths require list containers")
            current = cast(list[object], current)[part]
    final = path[-1]
    if isinstance(final, str):
        if type(current) is not dict:
            raise TypeError("string paths require dictionary containers")
        cast(dict[str, object], current)[final] = value
    else:
        if type(current) is not list:
            raise TypeError("integer paths require list containers")
        cast(list[object], current)[final] = value


def value_at(payload: dict[str, object], path: tuple[str | int, ...]) -> object:
    """Return one nested JSON-compatible value by its fixed schema path."""
    current: object = payload
    for part in path:
        if isinstance(part, str):
            if type(current) is not dict:
                raise TypeError("string paths require dictionary containers")
            current = cast(dict[str, object], current)[part]
        else:
            if type(current) is not list:
                raise TypeError("integer paths require list containers")
            current = cast(list[object], current)[part]
    return current


def authoring_payload(payload: dict[str, object]) -> dict[str, object]:
    """Convert the mutable JSON template back into strict tuple authoring values."""
    opening = payload["opening"]
    if type(opening) is not dict:
        raise TypeError("opening must be a dictionary")
    opening_payload = cast(dict[str, object], opening)
    phases = opening_payload["phases"]
    if type(phases) is not list:
        raise TypeError("opening phases must be a list")
    strict_phases: list[dict[str, object]] = []
    for phase in phases:
        if type(phase) is not dict:
            raise TypeError("opening phase must be a dictionary")
        phase_payload = cast(dict[str, object], phase)
        for name in ("crop_targets", "animal_targets", "structure_targets"):
            values = phase_payload[name]
            if type(values) is not list:
                raise TypeError(f"{name} must be a list")
            phase_payload[name] = tuple(cast(list[object], values))
        strict_phases.append(phase_payload)
    opening_payload["phases"] = tuple(strict_phases)
    return payload


@dataclass(frozen=True)
class GenomeCodec:
    """Encode and decode the strict mixed discrete/continuous configuration."""

    template: dict[str, object]
    discrete_groups: tuple[DiscreteGroup, ...]
    continuous_genes: tuple[ContinuousGene, ...]

    @property
    def width(self) -> int:
        """Return the exact required flat-vector width."""
        discrete_width = sum(group.width for group in self.discrete_groups)
        return discrete_width + len(self.continuous_genes)

    @classmethod
    def default(cls) -> "GenomeCodec":
        """Build the fixed-schema codec for the baseline three-phase template."""
        template = HybridConfig.default().model_dump(mode="json")
        groups: list[DiscreteGroup] = []
        cursor = 0

        def add(path: tuple[str | int, ...], values: tuple[object, ...]) -> None:
            nonlocal cursor
            groups.append(DiscreteGroup(path=path, values=values, start=cursor))
            cursor += len(values)

        phases = template["opening"]["phases"]
        if type(phases) is not list or len(phases) != len(PHASE_START_DAYS):
            raise ValueError("default template must hold three opening phases")
        for phase_index in range(len(phases)):
            prefix: tuple[str | int, ...] = ("opening", "phases", phase_index)
            add((*prefix, "start_day"), PHASE_START_DAYS[phase_index])
            add((*prefix, "target_hands"), HAND_TARGETS)
            add((*prefix, "target_quadrants"), QUADRANT_TARGETS)
            add((*prefix, "primary_crop"), tuple(CROP_NAMES))
            for target_index in range(len(CROP_NAMES)):
                add((*prefix, "crop_targets", target_index), TARGET_COUNTS)
            for target_index in range(len(ANIMAL_NAMES)):
                add((*prefix, "animal_targets", target_index), TARGET_COUNTS)
            for target_index in range(2):
                add((*prefix, "structure_targets", target_index), TARGET_COUNTS)
            add((*prefix, "cash_reserve"), CASH_RESERVES)
            add((*prefix, "inventory_reserve"), INVENTORY_RESERVES)
        add(("liquidation_start_day",), LIQUIDATION_DAYS)

        continuous = tuple(
            ContinuousGene(
                path=("jobs", name), low=0.0, high=10.0, index=cursor + index
            )
            for index, name in enumerate(JOB_WEIGHT_FIELDS)
        ) + tuple(
            ContinuousGene(
                path=("market", name),
                low=0.0,
                high=10.0,
                index=cursor + len(JOB_WEIGHT_FIELDS) + index,
            )
            for index, name in enumerate(MARKET_WEIGHT_FIELDS)
        )
        return cls(
            template=template,
            discrete_groups=tuple(groups),
            continuous_genes=continuous,
        )

    def decode(self, vector: Sequence[float]) -> HybridConfig:
        """Decode finite logits into one fully validated hybrid configuration."""
        if len(vector) != self.width or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in vector
        ):
            raise ValueError(f"genome must hold {self.width} finite values")
        payload = copy.deepcopy(self.template)
        for group in self.discrete_groups:
            logits = vector[group.start : group.start + group.width]
            selected = max(
                range(group.width), key=lambda index: (logits[index], -index)
            )
            assign_path(payload, group.path, group.values[selected])
        for gene in self.continuous_genes:
            unit = min(max(float(vector[gene.index]), 0.0), 1.0)
            assign_path(payload, gene.path, gene.low + unit * (gene.high - gene.low))
        return HybridConfig.model_validate(authoring_payload(payload))

    def encode(self, config: HybridConfig) -> tuple[float, ...]:
        """Return exact one-hots plus inverse-linear continuous gene values."""
        payload = config.model_dump(mode="json")
        vector = [0.0] * self.width
        for group in self.discrete_groups:
            selected = value_at(payload, group.path)
            try:
                selected_index = group.values.index(selected)
            except ValueError as error:
                raise ValueError(
                    f"configuration value at {group.path} is not encodable"
                ) from error
            vector[group.start + selected_index] = 1.0
        for gene in self.continuous_genes:
            value = value_at(payload, gene.path)
            if type(value) is not float:
                raise TypeError(
                    f"continuous configuration value at {gene.path} must be a float"
                )
            vector[gene.index] = (value - gene.low) / (gene.high - gene.low)
        return tuple(vector)


def encode(config: HybridConfig) -> tuple[float, ...]:
    """Encode with the fixed default hybrid genome schema."""
    return GenomeCodec.default().encode(config)


def decode(vector: Sequence[float]) -> HybridConfig:
    """Decode with the fixed default hybrid genome schema."""
    return GenomeCodec.default().decode(vector)
