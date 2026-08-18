"""Replaying routes against each other on the reference engine.

The engine is deterministic given a seed and both seats' actions, so a route
replayed here reproduces the episode it was harvested from exactly. That is the
property the arena rests on, and it is why the arena runs on the reference
engine rather than on the batched simulator: the simulator's action encoding
carries no quantity for PICKUP/PLACE, which 78% of a real route's transfers use.

An opponent is either a route (a recording, replayed by our own closure) or a
path to an agent file (a policy, loaded and run by the engine itself). The
reference engine runs agents, not just tapes, so it can host an opponent that
reacts to the candidate -- the one check a frozen-recording league cannot give.

Episodes are independent, so seeds are played across a process pool. Each worker
runs one whole episode; routes are plain lists of dicts and agent paths are
plain strings, both of which pickle without help.
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

Opponent = Route | str


def play(
    seat_zero: Opponent,
    seat_one: Opponent,
    seeds: Sequence[int],
    workers: int | None = None,
) -> list[tuple[int, int]]:
    """Play two opponents against each other across ``seeds``.

    Args:
        seat_zero: The route or agent path to play in seat 0.
        seat_one: The route or agent path to play in seat 1.
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


def outcomes(
    candidate: Route,
    league: Mapping[str, Opponent],
    seeds: Sequence[int],
    workers: int | None = None,
) -> list[float]:
    """Return one score per game the candidate plays against the league.

    Every seed is played twice, once with the candidate in each seat, because
    the seats are not symmetric: they hold different quadrants and their market
    orders pair by queue position. Scoring one ordering only would measure the
    seat as much as the route.

    A tie counts as half a win, which is what the ladder's rating does with it.

    This is the one place the win/tie/loss rule is applied; ``evaluate`` and
    any caller that needs per-game scores (a paired comparison, for instance)
    both read from here rather than from a second implementation of it.

    Args:
        candidate: The route being scored.
        league: Opponents by name; each a route or a path to an agent file.
        seeds: Episode seeds; each is played in both orderings.
        workers: Processes to spread games over, or None for the default.

    Returns:
        ``1.0``/``0.5``/``0.0`` per game (win/tie/loss for the candidate), in
        a fixed, deterministic order: league member (``league``'s own
        iteration order), then seed, then seat ordering (candidate in seat
        zero, then candidate in seat one). Each league member contributes
        ``2 * len(seeds)`` consecutive entries.
    """
    scores: list[float] = []
    for _, opponent in league.items():
        first = play(candidate, opponent, seeds, workers)
        second = play(opponent, candidate, seeds, workers)
        for (ours_first, theirs_first), (theirs_second, ours_second) in zip(
            first, second, strict=True
        ):
            scores.append(_win(ours_first, theirs_first))
            scores.append(_win(ours_second, theirs_second))
    return scores


def summarize(
    scores: Sequence[float], league: Mapping[str, Opponent], seeds: Sequence[int]
) -> dict[str, float]:
    """Group a flat ``outcomes`` list back into one win rate per league member.

    Exposed separately from ``evaluate`` so a caller that already holds an
    ``outcomes`` list -- a paired hill-climb comparison, for instance -- can
    get the same per-member breakdown without replaying the games.

    Args:
        scores: An ``outcomes`` list, in that function's game order.
        league: The same league ``outcomes`` was called with, same order.
        seeds: The same seeds ``outcomes`` was called with.

    Returns:
        One win rate per league member, over ``2 * len(seeds)`` games each.
    """
    games = 2 * len(seeds)
    return {
        name: sum(scores[index * games : (index + 1) * games]) / games
        for index, name in enumerate(league)
    }


def evaluate(
    candidate: Route,
    league: Mapping[str, Opponent],
    seeds: Sequence[int],
    workers: int | None = None,
) -> dict[str, float]:
    """Return the candidate's win rate against each league member.

    Args:
        candidate: The route being scored.
        league: Opponents by name; each a route or a path to an agent file.
        seeds: Episode seeds; each is played in both orderings.
        workers: Processes to spread games over, or None for the default.

    Returns:
        One win rate per league member, over ``2 * len(seeds)`` games each.
    """
    return summarize(outcomes(candidate, league, seeds, workers), league, seeds)


def _win(ours: int, theirs: int) -> float:
    """Return the candidate's score for one game: 1.0/0.5/0.0 win/tie/loss."""
    return 1.0 if ours > theirs else 0.5 if ours == theirs else 0.0


def _side(opponent: Opponent) -> str | _Agent:
    """Return what ``env.run`` should be handed for one seat.

    A ``str`` is a path to an agent file, which the engine loads and runs
    itself; a route is replayed by our own closure. Both occupy the same
    position in ``run``.
    """
    return opponent if isinstance(opponent, str) else _replay(opponent)


def _one(work: tuple[Opponent, Opponent, int]) -> tuple[int, int]:
    """Play a single episode. Runs in a subprocess."""
    seat_zero, seat_one, seed = work
    environment = make(
        ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS, "seed": seed}
    )
    environment.run([_side(seat_zero), _side(seat_one)])
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
