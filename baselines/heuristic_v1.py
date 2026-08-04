"""The melon monoculture that first went on the ladder, frozen as an opponent.

Pinned as a configuration rather than a copy of the policy, so it keeps
exercising the shipped code path. It is kept in the league because it is a
genuinely different shape of opponent — no livestock, one crop, a small crew —
and a change that only beats livestock builds has not been shown to generalise.
"""

from typing import Any, Mapping

from kaggriculture.observation import Observation
from kaggriculture.policy import Strategy, plan

STRATEGY = Strategy(
    crops={"MELON": 36},
    herd={},
    max_hands=8,
    tiles_per_unit=4,
    land_reserve=800,
)


def agent(raw_obs: Mapping[str, Any]) -> dict[str, Any]:
    """Return this turn's action dict for the environment."""
    return plan(Observation.parse(raw_obs), STRATEGY).to_action()
