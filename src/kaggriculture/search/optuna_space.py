"""Stable direct semantic Optuna space for validated hybrid policies."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal, Protocol

from kaggriculture.features import ANIMAL_NAMES, CROP_NAMES
from kaggriculture.hybrid.config import (
    HybridConfig,
    JobWeightsConfig,
    MarketConfig,
    OpeningConfig,
    OpeningPhaseConfig,
)
from kaggriculture.hybrid.schema import PHASE_START_DAYS

ParameterValue = int | float | str
ParameterKind = Literal["int", "float", "categorical"]

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


class TrialLike(Protocol):
    """The Optuna trial operations used by this semantic-space adapter."""

    def suggest_int(self, name: str, low: int, high: int, *, step: int = 1) -> int:
        """Return an integer parameter value from its declared domain."""

    def suggest_float(self, name: str, low: float, high: float) -> float:
        """Return a floating-point parameter value from its declared domain."""

    def suggest_categorical(self, name: str, choices: Sequence[str]) -> str:
        """Return one named category from its declared choices."""


@dataclass(frozen=True)
class ParameterSpec:
    """One stable, named public Optuna parameter definition."""

    name: str
    kind: ParameterKind
    low: int | float | None = None
    high: int | float | None = None
    step: int | None = None
    choices: tuple[str, ...] = ()


@dataclass(frozen=True)
class SpaceManifest:
    """The ordered, digestible public schema of the semantic search space."""

    parameters: tuple[ParameterSpec, ...]


def _space_parameters() -> tuple[ParameterSpec, ...]:
    """Build the fixed parameter table in the same order used for suggestion."""
    parameters: list[ParameterSpec] = []
    for index, allowed_days in enumerate(PHASE_START_DAYS):
        prefix = f"opening.phase_{index}"
        if len(allowed_days) > 1:
            parameters.append(
                ParameterSpec(
                    name=f"{prefix}.start_day",
                    kind="int",
                    low=allowed_days[0],
                    high=allowed_days[-1],
                )
            )
        parameters.extend(
            (
                ParameterSpec(f"{prefix}.target_hands", "int", 0, 19),
                ParameterSpec(f"{prefix}.target_quadrants", "int", 1, 5),
                ParameterSpec(
                    f"{prefix}.primary_crop",
                    "categorical",
                    choices=tuple(CROP_NAMES),
                ),
            )
        )
        parameters.extend(
            ParameterSpec(f"{prefix}.crop.{name}", "int", 0, 100) for name in CROP_NAMES
        )
        parameters.extend(
            ParameterSpec(f"{prefix}.animal.{name}", "int", 0, 100)
            for name in ANIMAL_NAMES
        )
        parameters.extend(
            (
                ParameterSpec(f"{prefix}.structure.coop", "int", 0, 100),
                ParameterSpec(f"{prefix}.structure.pasture", "int", 0, 100),
                ParameterSpec(f"{prefix}.cash_reserve", "int", 0, 5_000, step=100),
                ParameterSpec(f"{prefix}.inventory_reserve", "int", 0, 100),
            )
        )
    parameters.extend(
        ParameterSpec(f"jobs.{name}", "float", 0.0, 10.0) for name in JOB_WEIGHT_FIELDS
    )
    parameters.extend(
        ParameterSpec(f"market.{name}", "float", 0.0, 10.0)
        for name in MARKET_WEIGHT_FIELDS
    )
    parameters.append(ParameterSpec("liquidation_start_day", "int", 20, 29))
    return tuple(parameters)


SPACE_MANIFEST = SpaceManifest(parameters=_space_parameters())


def _canonical_json(value: object) -> str:
    """Return compact, deterministic JSON suitable for content addressing."""
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


SPACE_SHA256 = hashlib.sha256(
    _canonical_json(
        {"parameters": [asdict(spec) for spec in SPACE_MANIFEST.parameters]}
    ).encode()
).hexdigest()


def _suggest_weights(
    trial: TrialLike, prefix: str, fields: Sequence[str]
) -> dict[str, float]:
    """Suggest all continuous ranking weights for one configuration section."""
    return {name: trial.suggest_float(f"{prefix}.{name}", 0.0, 10.0) for name in fields}


def suggest_config(trial: TrialLike) -> HybridConfig:
    """Suggest every public semantic value and return its strict configuration."""
    phases: list[OpeningPhaseConfig] = []
    for index, allowed_days in enumerate(PHASE_START_DAYS):
        prefix = f"opening.phase_{index}"
        start_day = (
            allowed_days[0]
            if len(allowed_days) == 1
            else trial.suggest_int(
                f"{prefix}.start_day", allowed_days[0], allowed_days[-1]
            )
        )
        phases.append(
            OpeningPhaseConfig(
                start_day=start_day,
                target_hands=trial.suggest_int(f"{prefix}.target_hands", 0, 19),
                target_quadrants=trial.suggest_int(f"{prefix}.target_quadrants", 1, 5),
                primary_crop=trial.suggest_categorical(
                    f"{prefix}.primary_crop", tuple(CROP_NAMES)
                ),
                crop_targets=tuple(
                    trial.suggest_int(f"{prefix}.crop.{name}", 0, 100)
                    for name in CROP_NAMES
                ),
                animal_targets=tuple(
                    trial.suggest_int(f"{prefix}.animal.{name}", 0, 100)
                    for name in ANIMAL_NAMES
                ),
                structure_targets=(
                    trial.suggest_int(f"{prefix}.structure.coop", 0, 100),
                    trial.suggest_int(f"{prefix}.structure.pasture", 0, 100),
                ),
                cash_reserve=trial.suggest_int(
                    f"{prefix}.cash_reserve", 0, 5_000, step=100
                ),
                inventory_reserve=trial.suggest_int(
                    f"{prefix}.inventory_reserve", 0, 100
                ),
            )
        )
    first_phase, second_phase, third_phase = phases
    return HybridConfig(
        opening=OpeningConfig(phases=(first_phase, second_phase, third_phase)),
        jobs=JobWeightsConfig(**_suggest_weights(trial, "jobs", JOB_WEIGHT_FIELDS)),
        market=MarketConfig(**_suggest_weights(trial, "market", MARKET_WEIGHT_FIELDS)),
        liquidation_start_day=trial.suggest_int("liquidation_start_day", 20, 29),
    )


def parameters_for_config(config: HybridConfig) -> dict[str, ParameterValue]:
    """Return the exact public parameter values for one validated config."""
    parameters: dict[str, ParameterValue] = {}
    for index, (phase, allowed_days) in enumerate(
        zip(config.opening.phases, PHASE_START_DAYS, strict=True)
    ):
        prefix = f"opening.phase_{index}"
        if len(allowed_days) > 1:
            parameters[f"{prefix}.start_day"] = phase.start_day
        parameters[f"{prefix}.target_hands"] = phase.target_hands
        parameters[f"{prefix}.target_quadrants"] = phase.target_quadrants
        parameters[f"{prefix}.primary_crop"] = phase.primary_crop
        parameters.update(
            {
                f"{prefix}.crop.{name}": target
                for name, target in zip(CROP_NAMES, phase.crop_targets, strict=True)
            }
        )
        parameters.update(
            {
                f"{prefix}.animal.{name}": target
                for name, target in zip(ANIMAL_NAMES, phase.animal_targets, strict=True)
            }
        )
        parameters[f"{prefix}.structure.coop"] = phase.structure_targets[0]
        parameters[f"{prefix}.structure.pasture"] = phase.structure_targets[1]
        parameters[f"{prefix}.cash_reserve"] = phase.cash_reserve
        parameters[f"{prefix}.inventory_reserve"] = phase.inventory_reserve
    parameters.update(
        {f"jobs.{name}": getattr(config.jobs, name) for name in JOB_WEIGHT_FIELDS}
    )
    parameters.update(
        {
            f"market.{name}": getattr(config.market, name)
            for name in MARKET_WEIGHT_FIELDS
        }
    )
    parameters["liquidation_start_day"] = config.liquidation_start_day
    return parameters


def config_sha256(config: HybridConfig) -> str:
    """Return the canonical content digest used to reject duplicate starts."""
    return hashlib.sha256(
        _canonical_json(config.model_dump(mode="json")).encode()
    ).hexdigest()
