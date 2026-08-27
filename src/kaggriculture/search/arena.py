"""Replaying routes against each other on the reference engine.

The engine is deterministic given a seed and both seats' actions, so a route
replayed here reproduces the episode it was harvested from exactly. That is the
property the arena rests on, and it is why the arena runs on the reference
engine rather than on the batched simulator: the simulator's action encoding
carries no quantity for PICKUP/PLACE, which 78% of a real route's transfers use.

An opponent is a route (replayed by our own closure), a path to an agent file,
or immutable hybrid runtime data whose closure is built inside the worker. The
reference engine runs agents, not just tapes, so it can host an opponent that
reacts to the candidate -- the one check a frozen-recording league cannot give.

Episodes are independent, so seeds are played across a process pool. Each worker
runs one whole episode; routes are plain lists of dicts and agent paths are
plain strings, both of which pickle without help.
"""

import json
import math
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from kaggle_environments import make

from kaggriculture.constants import ENVIRONMENT, EPISODE_STEPS
from kaggriculture.hybrid.policy import build_agent
from kaggriculture.hybrid.runtime import RuntimeConfig
from kaggriculture.search.route import Route

_Action = dict[str, Any]
_Observation = Mapping[str, Any]
_Configuration = Mapping[str, Any] | None
_Agent = Callable[[_Observation, _Configuration], _Action]


@dataclass(frozen=True)
class HybridOpponent:
    """Picklable runtime data for constructing a hybrid agent in a worker."""

    runtime: RuntimeConfig


Opponent = Route | str | HybridOpponent


@dataclass(frozen=True, order=True)
class GameKey:
    """Canonical identity for one candidate-relative arena episode."""

    opponent: str
    seed: int
    seat: int

    def __post_init__(self) -> None:
        """Reject provenance that cannot identify a legal game cell."""
        if type(self.opponent) is not str or not self.opponent:
            raise ValueError("game opponent must be a nonempty string")
        if type(self.seed) is not int:
            raise ValueError("game seed must be an integer")
        if type(self.seat) is not int or self.seat not in (0, 1):
            raise ValueError("game seat must be 0 or 1")


@dataclass(frozen=True)
class GameTask:
    """Picklable input for one candidate-relative arena episode."""

    key: GameKey
    candidate: HybridOpponent
    opponent: Opponent


@dataclass(frozen=True)
class GameResult:
    """One arena episode with explicit provenance and failure preservation."""

    key: GameKey
    ours: int | None
    theirs: int | None
    runtime_seconds: float
    failure: str | None = None

    def __post_init__(self) -> None:
        """Reject malformed timing, bank, and failure combinations."""
        if (
            type(self.runtime_seconds) not in (int, float)
            or not math.isfinite(self.runtime_seconds)
            or self.runtime_seconds < 0
        ):
            raise ValueError("game runtime_seconds must be finite and nonnegative")
        if self.failure is None:
            if type(self.ours) is not int or type(self.theirs) is not int:
                raise ValueError("successful game result requires both banks")
        elif type(self.failure) is not str or not self.failure:
            raise ValueError("game failure must be a nonempty string")
        elif self.ours is not None or self.theirs is not None:
            raise ValueError("failed game result cannot contain banks")


def run_game_task(task: GameTask) -> GameResult:
    """Run one task while retaining provenance when the engine raises."""
    started = perf_counter()
    try:
        seats = (
            (task.candidate, task.opponent)
            if task.key.seat == 0
            else (task.opponent, task.candidate)
        )
        left, right = _run_banks(*seats, task.key.seed)
        ours, theirs = (left, right) if task.key.seat == 0 else (right, left)
        return GameResult(task.key, ours, theirs, perf_counter() - started)
    except Exception as error:
        return GameResult(
            task.key,
            None,
            None,
            perf_counter() - started,
            f"{type(error).__name__}: {error}",
        )


class OutcomeScores(list[float]):
    """Win points with the paired bank margins and duration that produced them."""

    def __init__(self) -> None:
        super().__init__()
        self.margins: list[int] = []
        self.normalized_margins: list[float] = []
        self.failures: list[str] = []
        self.runtime_seconds = 0.0


ActionProvenance = tuple[str, int, int]


def action_traces(
    candidate: HybridOpponent,
    league: Mapping[str, Opponent],
    seeds: Sequence[int],
    workers: int | None = None,
) -> dict[ActionProvenance, tuple[str, ...]]:
    """Return exact canonical candidate actions keyed by opponent, seed, and seat."""
    seed_tuple = tuple(seeds)
    if not seed_tuple or len(seed_tuple) != len(set(seed_tuple)):
        raise ValueError("action-trace seeds must be nonempty and unique")
    if any(type(seed) is not int for seed in seed_tuple):
        raise TypeError("action-trace seeds must be integers")
    league_copy = dict(league)
    if not league_copy or any(
        type(name) is not str or not name for name in league_copy
    ):
        raise ValueError("action-trace league requires unique nonempty names")
    work = [
        (name, candidate, opponent, seed, seat)
        for name, opponent in league_copy.items()
        for seed in seed_tuple
        for seat in (0, 1)
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_one_trace, work))
    evidence: dict[ActionProvenance, tuple[str, ...]] = {}
    for provenance, trace in rows:
        if provenance in evidence:
            raise ValueError(f"duplicate action-trace provenance: {provenance}")
        evidence[provenance] = trace
    expected = {
        (name, seed, seat)
        for name in league_copy
        for seed in seed_tuple
        for seat in (0, 1)
    }
    if set(evidence) != expected:
        raise ValueError("action-trace provenance is missing or unexpected")
    return evidence


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
    candidate: Opponent,
    league: Mapping[str, Opponent],
    seeds: Sequence[int],
    workers: int | None = None,
) -> OutcomeScores:
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
    started = perf_counter()
    scores = OutcomeScores()
    for _, opponent in league.items():
        first = play(candidate, opponent, seeds, workers)
        second = play(opponent, candidate, seeds, workers)
        for (ours_first, theirs_first), (theirs_second, ours_second) in zip(
            first, second, strict=True
        ):
            scores.append(_win(ours_first, theirs_first))
            scores.append(_win(ours_second, theirs_second))
            scores.margins.append(ours_first - theirs_first)
            scores.margins.append(ours_second - theirs_second)
            scores.normalized_margins.append(
                _normalized_margin(ours_first, theirs_first)
            )
            scores.normalized_margins.append(
                _normalized_margin(ours_second, theirs_second)
            )
    scores.runtime_seconds = perf_counter() - started
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
    candidate: Opponent,
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


def _normalized_margin(ours: int, theirs: int) -> float:
    """Scale a paired bank margin without changing its sign."""
    return (ours - theirs) / max(abs(ours) + abs(theirs), 1)


def _side(opponent: Opponent) -> str | _Agent:
    """Return what ``env.run`` should be handed for one seat.

    A ``HybridOpponent`` becomes a closure only after its runtime data reaches
    the worker. A ``str`` is a path the engine loads; a route is replayed.
    """
    if isinstance(opponent, HybridOpponent):
        return build_agent(opponent.runtime)
    return opponent if isinstance(opponent, str) else _replay(opponent)


def _one(work: tuple[Opponent, Opponent, int]) -> tuple[int, int]:
    """Play a legacy single episode. Runs in a subprocess."""
    seat_zero, seat_one, seed = work
    return _run_banks(seat_zero, seat_one, seed)


def _run_banks(seat_zero: Opponent, seat_one: Opponent, seed: int) -> tuple[int, int]:
    """Run the reference engine once and return its seat-zero/seat-one banks."""
    environment = make(
        ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS, "seed": seed}
    )
    environment.run([_side(seat_zero), _side(seat_one)])
    final = environment.steps[-1]
    statuses = (final[0].status, final[1].status)
    if (
        statuses != ("DONE", "DONE")
        or final[0].reward is None
        or final[1].reward is None
    ):
        raise RuntimeError(
            f"seed {seed} did not finish cleanly (statuses={statuses}, "
            f"rewards={(final[0].reward, final[1].reward)}); "
            "a route that no longer matches the episode it plays against, or a "
            "run that timed out, errored, or forfeited, is not an ordinary result."
        )
    return (int(final[0].reward), int(final[1].reward))


def _one_trace(
    work: tuple[str, HybridOpponent, Opponent, int, int],
) -> tuple[ActionProvenance, tuple[str, ...]]:
    """Play one determinism episode and retain the hybrid's exact action stream."""
    name, candidate, opponent, seed, seat = work
    trace: list[str] = []
    policy = build_agent(candidate.runtime)

    def traced(
        observation: _Observation, configuration: _Configuration = None
    ) -> _Action:
        action = policy(observation, configuration)
        trace.append(
            json.dumps(action, allow_nan=False, separators=(",", ":"), sort_keys=True)
        )
        return action

    environment = make(
        ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS, "seed": seed}
    )
    if seat == 0:
        environment.run([traced, _side(opponent)])
    else:
        environment.run([_side(opponent), traced])
    final = environment.steps[-1]
    statuses = (final[0].status, final[1].status)
    if statuses != ("DONE", "DONE") or any(state.reward is None for state in final):
        raise RuntimeError(
            f"determinism seed {seed} did not finish cleanly (statuses={statuses})"
        )
    return (name, seed, seat), tuple(trace)


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
