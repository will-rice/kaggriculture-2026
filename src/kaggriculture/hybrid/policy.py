"""Single-path, dependency-light hybrid policy and conservative boundary."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from kaggriculture.action_codec import (
    SelectedActions,
    decode_selected,
    safe_pass_action,
)
from kaggriculture.features import EncodedObservation, encode_observation
from kaggriculture.hybrid.jobs import select_units
from kaggriculture.hybrid.market import select_market
from kaggriculture.hybrid.opening import opening_targets
from kaggriculture.hybrid.runtime import (
    RuntimeConfig,
    RuntimeJobWeights,
    RuntimeMarketWeights,
    RuntimeOpeningPhase,
)

DEFAULT_RUNTIME_CONFIG = RuntimeConfig(
    phases=(
        RuntimeOpeningPhase(
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
        RuntimeOpeningPhase(
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
        RuntimeOpeningPhase(
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
    ),
    jobs=RuntimeJobWeights(
        recovery=8.0,
        urgency=6.0,
        distance_penalty=1.0,
        production=5.0,
        transport=4.0,
        structure=3.0,
    ),
    market=RuntimeMarketWeights(
        live_price=5.0,
        town_demand=3.0,
        opponent_supply=2.0,
        reserve_penalty=7.0,
        liquidation_urgency=9.0,
    ),
    liquidation_start_day=26,
)


class HybridPolicy:
    """Pure per-turn hybrid controller over one encoded observation."""

    def __init__(self, runtime: RuntimeConfig) -> None:
        self.runtime = runtime

    def decide(self, encoded: EncodedObservation) -> dict[str, Any]:
        """Derive targets, select legal indices, and decode exactly once."""
        targets = opening_targets(encoded, self.runtime)
        units = select_units(encoded, targets, self.runtime)
        market = select_market(encoded, targets, self.runtime)
        return decode_selected(
            SelectedActions(
                units.operation_indices,
                units.quantity_indices,
                market.indices,
            )
        )


class AgentCallable(Protocol):
    """One- or two-argument callable accepted by the reference engine."""

    def __call__(
        self,
        observation: Mapping[str, Any],
        configuration: object | None = None,
    ) -> dict[str, Any]:
        """Return one canonical engine action for a raw observation."""
        ...


def build_agent(runtime: RuntimeConfig) -> AgentCallable:
    """Bind one immutable runtime and the narrow raw-observation boundary."""
    controller = HybridPolicy(runtime)

    def agent(
        observation: Mapping[str, Any], configuration: object | None = None
    ) -> dict[str, Any]:
        try:
            seat = int(observation.get("player", 0))
            return controller.decide(encode_observation(observation, seat))
        except (KeyError, TypeError, ValueError, IndexError):
            return safe_pass_action()

    return agent


agent = build_agent(DEFAULT_RUNTIME_CONFIG)
