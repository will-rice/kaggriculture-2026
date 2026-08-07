"""The single richest harvested route, replayed with nothing to switch to.

A gate rung, not a submission. ``routes.play`` normally chooses among 190
routes per board and is free to change its mind mid-season, which is the whole
reason it beats the recorded tape; this file hands ``RouteAgent`` exactly one
prototype, so there is nothing to switch to and the season is the fixed best
route or the fallback. That is the "fixed best route" the plan's gate names,
and it is a genuinely different agent from the one ``main.py`` serves: locking
on banks *more* coins and loses every game to the tape, measured over 100 seeds
when ``HYSTERESIS_MARGIN`` was swept.

The richest route is picked by ``bank``, the same key ``dedupe`` orders by, so
"best" here means what it means everywhere else in this package.

``agent`` is the last callable defined, because ``kaggle_environments`` execs an
agent file and plays whatever callable it finds last.
"""

import functools
from typing import Any, Mapping

from kaggriculture.routes import STORE
from kaggriculture.routes.play import RouteAgent
from kaggriculture.routes.store import load


@functools.lru_cache(maxsize=1)
def best_route() -> RouteAgent:
    """Return the process's agent over the one richest route.

    Cached for the same reason ``routes.play.route_agent`` is: the store costs
    seconds to decode and one process plays many episodes. Safe to share
    because ``RouteAgent.act`` clears its incumbent route on turn zero.

    Returns:
        A ``RouteAgent`` holding a single prototype.
    """
    return RouteAgent([max(load(STORE), key=lambda prototype: prototype.bank)])


def agent(
    observation: Mapping[str, Any], config: Mapping[str, int] | None = None
) -> dict[str, Any]:
    """Return this turn's action from the one route this agent knows.

    Args:
        observation: One turn's observation, as the environment hands it over.
        config: The episode configuration, or ``None`` for the defaults.

    Returns:
        The action dict the environment consumes.
    """
    return best_route().act(observation, config)
