"""Replaying routes against each other on the reference engine.

The engine is deterministic given a seed and both seats' actions, so a route
replayed here reproduces the episode it was harvested from exactly. That is the
property the arena rests on, and it is why the arena runs on the reference
engine rather than on the batched simulator: the simulator's action encoding
carries no quantity for PICKUP/PLACE, which 78% of a real route's transfers use.

Episodes are independent, so seeds are played across a process pool. Each worker
runs one whole episode; the routes are plain lists of dicts and pickle without
help.
"""

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from kaggle_environments import make

from kaggriculture.constants import ENVIRONMENT, EPISODE_STEPS
from kaggriculture.search.route import Route

_Action = dict[str, Any]
_Observation = Mapping[str, Any]
_Configuration = Mapping[str, Any] | None
_Agent = Callable[[_Observation, _Configuration], _Action]


def play(
    seat_zero: Route,
    seat_one: Route,
    seeds: Sequence[int],
    workers: int | None = None,
) -> list[tuple[int, int]]:
    """Play two routes against each other across ``seeds``.

    Args:
        seat_zero: The route to play in seat 0.
        seat_one: The route to play in seat 1.
        seeds: One episode seed per game.
        workers: Processes to spread the games over, or None for the default.

    Returns:
        One ``(bank_seat_zero, bank_seat_one)`` per seed, in the order given.
    """
    work = [(seat_zero, seat_one, int(seed)) for seed in seeds]
    if not work:
        return []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_one, work))


def _one(work: tuple[Route, Route, int]) -> tuple[int, int]:
    """Play a single episode. Runs in a subprocess."""
    seat_zero, seat_one, seed = work
    environment = make(
        ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS, "seed": seed}
    )
    environment.run([_replay(seat_zero), _replay(seat_one)])
    final = environment.steps[-1]
    if final[0].reward is None or final[1].reward is None:
        statuses = (final[0].status, final[1].status)
        raise RuntimeError(
            f"seed {seed} finished with a missing reward (statuses={statuses}); "
            "a route that no longer matches the episode it plays against, or a "
            "run that timed out or errored, is not a 0-0 result."
        )
    return (int(final[0].reward), int(final[1].reward))


def _replay(route: Route) -> _Agent:
    """Return an agent that plays ``route``, indexed by the engine's own clock.

    Reading ``observation["step"]`` rather than counting turns internally is what
    keeps the replay aligned with the recording: the route was harvested by step
    index, so it is replayed by step index, and no assumption about whether an
    action belongs to the state before or after it can creep in.

    The index carries a fixed offset of one. ``environment.steps[t]["action"]``
    is the action an agent submitted while looking at the observation whose own
    ``step`` field reads ``t - 1`` -- it lands at the index of the state it
    produced, one ahead of the state it was chosen from. ``route.from_episode``
    keeps that same indexing, since it reads straight from ``episode["steps"]``,
    so the action to submit when the engine hands this agent an observation at
    step ``k`` is ``route[k + 1]``, not ``route[k]``.

    There is no PASS fallback for a step past the end of ``route``: ``play``
    always configures ``episodeSteps`` to match ``constants.EPISODE_STEPS``,
    which is exactly the length ``route.from_episode`` and ``route.load``
    produce, so the engine never asks this agent for a step the route does not
    cover. A shorter route is not a case to paper over -- it means the route
    does not describe a full season, and indexing past its end raises
    ``IndexError`` rather than silently padding it with PASS.
    """

    def agent(
        observation: _Observation, configuration: _Configuration = None
    ) -> _Action:
        return route[int(observation["step"]) + 1]

    return agent
