"""The tuned job-list heuristic, frozen on the day it stopped being the agent.

This is where a day of measurement against the recorded tape left our own
policy: the meta build's crop mix, a herd of fourteen, five tiles per unit, and
a sell rate that clears the shed by the season's end rather than dumping it in
the final turn. It banked 54k against the tape where the melon monoculture it
replaced banked 28k.

It is kept as a league opponent because it lost 20-0 to the vendored economic
policy that replaced it, at 51k against that policy's 148k. An opponent we have
comprehensively beaten is still worth keeping: a change that improves against
the tape while regressing against this one has traded a known strength for an
unknown, and the league is what makes that visible.

Every field is pinned explicitly, including the ones that currently match the
package defaults. A baseline that inherits defaults is not frozen — it follows
whatever we tune next, and a league opponent that moves with us measures
nothing. The earlier `meta_build` baseline was deleted for exactly that reason.
"""

from typing import Any, Mapping

from kaggriculture.observation import Observation
from kaggriculture.policy import Strategy, plan

STRATEGY = Strategy(
    crops={"WHEAT": 11, "MELON": 11, "STRAWBERRY": 40},
    herd={"COW": 8, "SHEEP": 6},
    max_hands=12,
    tiles_per_unit=5,
    hire_before_hour=4,
    wage_share=0.1,
    cash_reserve=300,
    land_reserve=400,
    last_land_day=20,
    animals_per_turn=2,
    working_capital=400,
    feed_days=3,
    sell_rate=2,
    buy_slots=4,
    haul_threshold=12,
    flush_hour=12,
)


def agent(raw_obs: Mapping[str, Any]) -> dict[str, Any]:
    """Return this turn's action dict for the environment."""
    return plan(Observation.parse(raw_obs), STRATEGY).to_action()
