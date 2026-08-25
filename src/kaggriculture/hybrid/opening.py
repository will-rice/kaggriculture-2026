"""Pure guarded opening-target derivation for the hybrid rule agent."""

from dataclasses import dataclass
from typing import Mapping

from kaggriculture.features import (
    ANIMAL_NAMES,
    CROP_NAMES,
    PRODUCT_NAMES,
    STRUCTURE_KINDS,
    EncodedObservation,
)
from kaggriculture.hybrid.runtime import RuntimeConfig, RuntimeOpeningPhase


@dataclass(frozen=True)
class OpeningTargets:
    """The currently unsatisfied, reserve-aware targets for one opening phase."""

    phase_start_day: int
    hands_needed: int
    quadrants_needed: int
    crop_deficits: Mapping[str, int]
    animal_deficits: Mapping[str, int]
    structure_deficits: Mapping[str, int]
    protected_cash: int
    protected_inventory: Mapping[str, int]
    liquidating: bool


def select_phase(
    encoded: EncodedObservation, config: RuntimeConfig
) -> RuntimeOpeningPhase:
    """Return the latest scheduled phase whose start day has arrived."""
    day = encoded.day_count()
    active = tuple(phase for phase in config.phases if phase.start_day <= day)
    if not active:
        raise RuntimeError("validated runtime must include an active day-zero phase")
    return max(active, key=lambda phase: phase.start_day)


def opening_targets(
    encoded: EncodedObservation, config: RuntimeConfig
) -> OpeningTargets:
    """Derive state deficits without replaying missed opening actions."""
    phase = select_phase(encoded, config)
    liquidating = encoded.day_count() >= config.liquidation_start_day
    hands_needed = max(phase.target_hands - encoded.hand_count(), 0)
    quadrants_needed = max(phase.target_quadrants - encoded.quadrant_count(), 0)
    crop_deficits = {
        crop: max(phase.crop_targets[index] - encoded.crop_count(crop), 0)
        for index, crop in enumerate(CROP_NAMES)
    }
    animal_deficits = {
        animal: max(phase.animal_targets[index] - encoded.animal_count(animal), 0)
        for index, animal in enumerate(ANIMAL_NAMES)
    }
    structure_deficits = {
        structure: max(
            phase.structure_targets[index] - encoded.structure_count(structure), 0
        )
        for index, structure in enumerate(STRUCTURE_KINDS)
    }
    if liquidating:
        hands_needed = 0
        quadrants_needed = 0
        crop_deficits = dict.fromkeys(CROP_NAMES, 0)
        animal_deficits = dict.fromkeys(ANIMAL_NAMES, 0)
        structure_deficits = dict.fromkeys(STRUCTURE_KINDS, 0)
    return OpeningTargets(
        phase_start_day=phase.start_day,
        hands_needed=hands_needed,
        quadrants_needed=quadrants_needed,
        crop_deficits=crop_deficits,
        animal_deficits=animal_deficits,
        structure_deficits=structure_deficits,
        protected_cash=phase.cash_reserve,
        protected_inventory=dict.fromkeys(PRODUCT_NAMES, phase.inventory_reserve),
        liquidating=liquidating,
    )
