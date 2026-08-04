"""A reactive reconstruction of the build 75% of the ladder plays.

Measured from a replay: wheat and melon from day three, strawberry ramped to
forty, all three quadrants by day twelve, eight cows and six sheep in place by
day twelve. The recorded tape at
``/data/kaggriculture/baselines/meta_tape.py`` plays the same build but
open-loop, so it cannot respond to anything we do. Beating the recording can be
achieved by exploiting its blindness; beating this one cannot, which is why both
belong in the league.
"""

from typing import Any, Mapping

from kaggriculture.observation import Observation
from kaggriculture.policy import Strategy, plan

STRATEGY = Strategy(
    crops={"WHEAT": 11, "MELON": 11, "STRAWBERRY": 40},
    herd={"COW": 8, "SHEEP": 6},
    max_hands=12,
    tiles_per_unit=5,
)


def agent(raw_obs: Mapping[str, Any]) -> dict[str, Any]:
    """Return this turn's action dict for the environment."""
    return plan(Observation.parse(raw_obs), STRATEGY).to_action()
