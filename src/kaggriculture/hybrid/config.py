"""Strict offline authoring configuration for the hybrid rule agent."""

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kaggriculture.features import ANIMAL_NAMES, CROP_NAMES
from kaggriculture.hybrid.runtime import (
    RuntimeConfig,
    RuntimeJobWeights,
    RuntimeMarketWeights,
    RuntimeOpeningPhase,
)

TARGET_COUNT = Annotated[int, Field(ge=0, le=100)]


class OpeningPhaseConfig(BaseModel):
    """One validated target milestone in the opening program."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False, strict=True
    )

    start_day: int = Field(ge=0, le=29)
    target_hands: int = Field(ge=0, le=19)
    target_quadrants: int = Field(ge=1, le=5)
    primary_crop: str
    crop_targets: tuple[TARGET_COUNT, ...]
    animal_targets: tuple[TARGET_COUNT, ...]
    structure_targets: tuple[TARGET_COUNT, TARGET_COUNT]
    cash_reserve: int = Field(ge=0, le=5_000)
    inventory_reserve: int = Field(ge=0, le=100)

    @field_validator("primary_crop")
    @classmethod
    def primary_crop_is_engine_defined(cls, value: str) -> str:
        """Reject crops outside the canonical engine-derived schema."""
        if value not in CROP_NAMES:
            raise ValueError("primary_crop must be a canonical crop name")
        return value

    @field_validator("cash_reserve")
    @classmethod
    def cash_reserve_is_a_search_domain_value(cls, value: int) -> int:
        """Keep authoring values exactly representable by a one-hot domain."""
        if value % 100:
            raise ValueError("cash_reserve must be a multiple of 100")
        return value


class OpeningConfig(BaseModel):
    """The ordered fixed-schema opening milestones."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False, strict=True
    )

    phases: tuple[OpeningPhaseConfig, ...] = Field(min_length=1)


class JobWeightsConfig(BaseModel):
    """Bounded continuous priorities for reactive unit-job ranking."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False, strict=True
    )

    recovery: float = Field(ge=0.0, le=10.0)
    urgency: float = Field(ge=0.0, le=10.0)
    distance_penalty: float = Field(ge=0.0, le=10.0)
    production: float = Field(ge=0.0, le=10.0)
    transport: float = Field(ge=0.0, le=10.0)
    structure: float = Field(ge=0.0, le=10.0)


class MarketConfig(BaseModel):
    """Bounded continuous priorities for reactive market ranking."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False, strict=True
    )

    live_price: float = Field(ge=0.0, le=10.0)
    town_demand: float = Field(ge=0.0, le=10.0)
    opponent_supply: float = Field(ge=0.0, le=10.0)
    reserve_penalty: float = Field(ge=0.0, le=10.0)
    liquidation_urgency: float = Field(ge=0.0, le=10.0)


class HybridConfig(BaseModel):
    """Fully validated authoring configuration for one hybrid candidate."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False, strict=True
    )

    opening: OpeningConfig
    jobs: JobWeightsConfig
    market: MarketConfig
    liquidation_start_day: int = Field(ge=20, le=29)

    @model_validator(mode="after")
    def ordered_phases_and_fixed_schema(self) -> Self:
        """Reject duplicate, unordered, or schema-incompatible milestones."""
        starts = tuple(phase.start_day for phase in self.opening.phases)
        if starts != tuple(sorted(set(starts))):
            raise ValueError("opening phases must have unique increasing start_day")
        for phase in self.opening.phases:
            if len(phase.crop_targets) != len(CROP_NAMES):
                raise ValueError("crop_targets must follow the fixed crop schema")
            if len(phase.animal_targets) != len(ANIMAL_NAMES):
                raise ValueError("animal_targets must follow the fixed animal schema")
        return self

    @classmethod
    def default(cls) -> Self:
        """Return the compact, valid baseline searched by the default codec."""
        return cls(
            opening=OpeningConfig(
                phases=(
                    OpeningPhaseConfig(
                        start_day=0,
                        target_hands=4,
                        target_quadrants=1,
                        primary_crop="WHEAT",
                        crop_targets=(0, 0, 0, 0, 5),
                        animal_targets=(0, 0, 0),
                        structure_targets=(0, 0),
                        cash_reserve=300,
                        inventory_reserve=2,
                    ),
                    OpeningPhaseConfig(
                        start_day=10,
                        target_hands=8,
                        target_quadrants=2,
                        primary_crop="STRAWBERRY",
                        crop_targets=(0, 0, 8, 0, 8),
                        animal_targets=(0, 0, 0),
                        structure_targets=(0, 0),
                        cash_reserve=500,
                        inventory_reserve=4,
                    ),
                    OpeningPhaseConfig(
                        start_day=20,
                        target_hands=10,
                        target_quadrants=3,
                        primary_crop="STRAWBERRY",
                        crop_targets=(0, 0, 16, 0, 8),
                        animal_targets=(0, 0, 0),
                        structure_targets=(0, 0),
                        cash_reserve=700,
                        inventory_reserve=6,
                    ),
                )
            ),
            jobs=JobWeightsConfig(
                recovery=8.0,
                urgency=6.0,
                distance_penalty=1.0,
                production=5.0,
                transport=4.0,
                structure=3.0,
            ),
            market=MarketConfig(
                live_price=5.0,
                town_demand=3.0,
                opponent_supply=2.0,
                reserve_penalty=7.0,
                liquidation_urgency=9.0,
            ),
            liquidation_start_day=26,
        )

    @classmethod
    def from_runtime(cls, runtime: RuntimeConfig) -> Self:
        """Recover the exact validated authoring model from frozen runtime data."""
        return cls.model_validate(
            {
                "opening": {"phases": runtime.phases},
                "jobs": runtime.jobs,
                "market": runtime.market,
                "liquidation_start_day": runtime.liquidation_start_day,
            },
            from_attributes=True,
        )


def to_runtime(config: HybridConfig) -> RuntimeConfig:
    """Freeze a validated authoring config into its Pydantic-free runtime form."""
    return RuntimeConfig(
        phases=tuple(
            RuntimeOpeningPhase(
                start_day=phase.start_day,
                target_hands=phase.target_hands,
                target_quadrants=phase.target_quadrants,
                primary_crop=phase.primary_crop,
                crop_targets=phase.crop_targets,
                animal_targets=phase.animal_targets,
                structure_targets=phase.structure_targets,
                cash_reserve=phase.cash_reserve,
                inventory_reserve=phase.inventory_reserve,
            )
            for phase in config.opening.phases
        ),
        jobs=RuntimeJobWeights(
            recovery=config.jobs.recovery,
            urgency=config.jobs.urgency,
            distance_penalty=config.jobs.distance_penalty,
            production=config.jobs.production,
            transport=config.jobs.transport,
            structure=config.jobs.structure,
        ),
        market=RuntimeMarketWeights(
            live_price=config.market.live_price,
            town_demand=config.market.town_demand,
            opponent_supply=config.market.opponent_supply,
            reserve_penalty=config.market.reserve_penalty,
            liquidation_urgency=config.market.liquidation_urgency,
        ),
        liquidation_start_day=config.liquidation_start_day,
    )
