"""Paired holdout intervals and atomic hybrid-promotion rulings."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from time import perf_counter

from kaggriculture.search import arena
from kaggriculture.search.arena import HybridOpponent, Opponent
from kaggriculture.search.evolution import (
    DEVELOPMENT_SEEDS,
    PROMOTION_SEEDS,
    SCREENING_SEEDS,
)
from kaggriculture.search.fitness import StrengthWeights
from kaggriculture.search.scripts.frontier_round_robin import FRONTIER_SEEDS

DETERMINISM_SEEDS = tuple(range(850_100, 850_104))
assert all(
    set(DETERMINISM_SEEDS).isdisjoint(seeds)
    for seeds in (FRONTIER_SEEDS, SCREENING_SEEDS, DEVELOPMENT_SEEDS, PROMOTION_SEEDS)
), "determinism seeds must be disjoint from all search and promotion seeds"


@dataclass(frozen=True)
class Interval:
    """A point estimate and its two-sided 95% bootstrap bounds."""

    mean: float
    low: float
    high: float

    def __post_init__(self) -> None:
        """Reject intervals that cannot be valid statistical evidence."""
        values = (self.mean, self.low, self.high)
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError("interval values must be finite numbers")
        if not self.low <= self.mean <= self.high:
            raise ValueError("interval bounds must contain the mean")


@dataclass(frozen=True)
class GameCounts:
    """Auditable win/draw/loss counts for one direct matchup."""

    wins: int = 0
    draws: int = 0
    losses: int = 0

    def __post_init__(self) -> None:
        """Reject negative or coerced result counts."""
        if any(type(value) is not int or value < 0 for value in vars(self).values()):
            raise ValueError("game counts must be non-negative integers")


@dataclass(frozen=True)
class PromotionVerdict:
    """Complete evidence for one all-or-nothing promotion ruling."""

    boatlee: Interval
    frontier: Interval
    league_delta: Interval
    matchup_deltas: Mapping[str, float]
    failures: int
    deterministic: bool
    boatlee_pairs: tuple[tuple[float, float], ...] = ()
    frontier_pairs: tuple[tuple[float, float], ...] = ()
    league_delta_pairs: tuple[tuple[float, float], ...] = ()
    boatlee_counts: GameCounts = GameCounts()
    frontier_counts: GameCounts = GameCounts()
    boatlee_margin_pairs: tuple[tuple[float, float], ...] = ()
    frontier_margin_pairs: tuple[tuple[float, float], ...] = ()
    candidate_matchup_pairs: Mapping[str, tuple[tuple[float, float], ...]] = field(
        default_factory=dict
    )
    incumbent_matchup_pairs: Mapping[str, tuple[tuple[float, float], ...]] = field(
        default_factory=dict
    )
    candidate_matchup_margin_pairs: Mapping[str, tuple[tuple[float, float], ...]] = (
        field(default_factory=dict)
    )
    incumbent_matchup_margin_pairs: Mapping[str, tuple[tuple[float, float], ...]] = (
        field(default_factory=dict)
    )
    failure_details: tuple[str, ...] = ()
    runtime_seconds: float = 0.0

    def __post_init__(self) -> None:
        """Copy and validate all evidence before any gate is reported."""
        deltas = dict(self.matchup_deltas)
        if any(
            type(name) is not str
            or not name
            or not isinstance(delta, (int, float))
            or isinstance(delta, bool)
            or not math.isfinite(delta)
            for name, delta in deltas.items()
        ):
            raise ValueError("matchup deltas require named finite values")
        if type(self.failures) is not int or self.failures < 0:
            raise ValueError("failures must be a non-negative integer")
        if type(self.deterministic) is not bool:
            raise TypeError("deterministic must be a boolean")
        if not math.isfinite(self.runtime_seconds) or self.runtime_seconds < 0.0:
            raise ValueError("runtime_seconds must be finite and non-negative")
        if len(self.failure_details) != self.failures:
            if self.failure_details:
                raise ValueError("failure count must match failure details")
        object.__setattr__(self, "matchup_deltas", deltas)

    @property
    def reasons(self) -> tuple[str, ...]:
        """Name every failed condition; no aggregate can hide another gate."""
        reasons: list[str] = []
        if self.boatlee.low <= 0.5:
            reasons.append(f"Boatlee lower bound {self.boatlee.low:.3f} is not > 0.500")
        if self.frontier.low <= 0.5:
            reasons.append(
                f"frontier lower bound {self.frontier.low:.3f} is not > 0.500"
            )
        if self.league_delta.low <= 0.0:
            reasons.append(
                f"league delta lower bound {self.league_delta.low:.3f} is not > 0.000"
            )
        reasons.extend(
            f"{name} regressed by {-delta:.3f} (> 0.050)"
            for name, delta in self.matchup_deltas.items()
            if delta < -0.05
        )
        if self.failures:
            reasons.append(f"{self.failures} execution failure(s) recorded")
        if not self.deterministic:
            reasons.append("candidate outputs were nondeterministic on replay")
        return tuple(reasons)

    @property
    def passed(self) -> bool:
        """Return the single atomic ruling after considering every gate."""
        return not self.reasons


def bootstrap_interval(
    paired_seat_points: Sequence[tuple[float, float]],
    *,
    draws: int = 10_000,
    seed: int = 20_260_825,
) -> Interval:
    """Bootstrap whole seeds so a seed's two seat orderings never split."""
    if type(draws) is not int or draws < 2:
        raise ValueError("draws must be an integer of at least two")
    if type(seed) is not int:
        raise TypeError("bootstrap seed must be an integer")
    if not paired_seat_points:
        raise ValueError("at least one paired seed is required")
    per_seed: list[float] = []
    for index, pair in enumerate(paired_seat_points):
        if type(pair) is not tuple or len(pair) != 2:
            raise TypeError(f"seed pair {index} must contain exactly two seats")
        if any(
            not isinstance(point, (int, float))
            or isinstance(point, bool)
            or not math.isfinite(point)
            for point in pair
        ):
            raise ValueError(f"seed pair {index} must contain finite points")
        per_seed.append(statistics.fmean(pair))
    rng = random.Random(seed)
    estimates = sorted(
        statistics.fmean(rng.choices(per_seed, k=len(per_seed))) for _ in range(draws)
    )
    low_index = int(draws * 0.025)
    high_index = min(int(draws * 0.975), draws - 1)
    return Interval(
        mean=statistics.fmean(per_seed),
        low=estimates[low_index],
        high=estimates[high_index],
    )


@dataclass(frozen=True)
class _Run:
    pairs: tuple[tuple[float, float], ...]
    margin_pairs: tuple[tuple[float, float], ...]
    failures: tuple[str, ...]
    runtime_seconds: float


def evaluate_promotion(
    candidate: HybridOpponent,
    incumbent: Opponent,
    boatlee: Opponent,
    frontier: Opponent,
    league: Mapping[str, Opponent],
    seeds: Sequence[int],
    workers: int,
    *,
    deterministic: bool,
    weights: StrengthWeights | None = None,
) -> PromotionVerdict:
    """Build complete paired evidence, then apply every promotion gate at once."""
    seed_tuple = _validated_seeds(seeds)
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("workers must be between 1 and 16 while Toad is running")
    if type(deterministic) is not bool:
        raise TypeError("deterministic evidence must be a boolean")
    league_copy = dict(league)
    if not league_copy:
        raise ValueError("promotion league must not be empty")
    if any(type(name) is not str or not name for name in league_copy):
        raise ValueError("promotion league names must be non-empty strings")
    exact_weights = weights or StrengthWeights(dict.fromkeys(league_copy, 1))
    if set(exact_weights.values) != set(league_copy):
        raise ValueError("promotion weights must exactly match league names")

    failures: list[str] = []
    runtime_seconds = 0.0
    candidate_runs: dict[str, _Run] = {}
    incumbent_runs: dict[str, _Run] = {}
    for name, opponent in league_copy.items():
        candidate_run = _arena_run(candidate, name, opponent, seed_tuple, workers)
        incumbent_run = _arena_run(incumbent, name, opponent, seed_tuple, workers)
        candidate_runs[name] = candidate_run
        incumbent_runs[name] = incumbent_run
        failures.extend(
            f"candidate vs {name}: {item}" for item in candidate_run.failures
        )
        failures.extend(
            f"incumbent vs {name}: {item}" for item in incumbent_run.failures
        )
        runtime_seconds += candidate_run.runtime_seconds + incumbent_run.runtime_seconds

    boatlee_name = _league_name(league_copy, boatlee, "Boatlee")
    frontier_name = _league_name(league_copy, frontier, "frontier")
    boatlee_run = candidate_runs[boatlee_name]
    frontier_run = candidate_runs[frontier_name]

    matchup_deltas = {
        name: statistics.fmean(
            candidate_point - incumbent_point
            for candidate_pair, incumbent_pair in zip(
                candidate_runs[name].pairs,
                incumbent_runs[name].pairs,
                strict=True,
            )
            for candidate_point, incumbent_point in zip(
                candidate_pair, incumbent_pair, strict=True
            )
        )
        for name in league_copy
    }
    total_weight = sum(exact_weights.values.values())
    league_delta_pairs = tuple(
        (
            float(
                sum(
                    exact_weights.values[name]
                    * (
                        candidate_runs[name].pairs[seed_index][0]
                        - incumbent_runs[name].pairs[seed_index][0]
                    )
                    for name in league_copy
                )
                / total_weight
            ),
            float(
                sum(
                    exact_weights.values[name]
                    * (
                        candidate_runs[name].pairs[seed_index][1]
                        - incumbent_runs[name].pairs[seed_index][1]
                    )
                    for name in league_copy
                )
                / total_weight
            ),
        )
        for seed_index in range(len(seed_tuple))
    )

    return PromotionVerdict(
        boatlee=bootstrap_interval(boatlee_run.pairs),
        frontier=bootstrap_interval(frontier_run.pairs),
        league_delta=bootstrap_interval(league_delta_pairs),
        matchup_deltas=matchup_deltas,
        failures=len(failures),
        deterministic=deterministic,
        boatlee_pairs=boatlee_run.pairs,
        frontier_pairs=frontier_run.pairs,
        league_delta_pairs=league_delta_pairs,
        boatlee_counts=_game_counts(boatlee_run.pairs),
        frontier_counts=_game_counts(frontier_run.pairs),
        boatlee_margin_pairs=boatlee_run.margin_pairs,
        frontier_margin_pairs=frontier_run.margin_pairs,
        candidate_matchup_pairs={
            name: run.pairs for name, run in candidate_runs.items()
        },
        incumbent_matchup_pairs={
            name: run.pairs for name, run in incumbent_runs.items()
        },
        candidate_matchup_margin_pairs={
            name: run.margin_pairs for name, run in candidate_runs.items()
        },
        incumbent_matchup_margin_pairs={
            name: run.margin_pairs for name, run in incumbent_runs.items()
        },
        failure_details=tuple(failures),
        runtime_seconds=runtime_seconds,
    )


def _validated_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    seed_tuple = tuple(seeds)
    if not seed_tuple:
        raise ValueError("at least one promotion seed is required")
    if any(type(seed) is not int for seed in seed_tuple):
        raise TypeError("promotion seeds must be integers")
    if len(seed_tuple) != len(set(seed_tuple)):
        raise ValueError("promotion seed provenance contains duplicates")
    return seed_tuple


def check_determinism(
    candidate: HybridOpponent,
    league: Mapping[str, Opponent],
    workers: int,
    *,
    seeds: Sequence[int] = DETERMINISM_SEEDS,
) -> bool:
    """Compare exact action/provenance traces on a non-holdout seed bank."""
    seed_tuple = _validated_seeds(seeds)
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("workers must be between 1 and 16 while Toad is running")
    first = arena.action_traces(candidate, league, seed_tuple, workers)
    replay = arena.action_traces(candidate, league, seed_tuple, workers)
    _validate_trace_bank(first, league, seed_tuple)
    _validate_trace_bank(replay, league, seed_tuple)
    return first == replay


def _validate_trace_bank(
    traces: Mapping[arena.ActionProvenance, tuple[str, ...]],
    league: Mapping[str, Opponent],
    seeds: Sequence[int],
) -> None:
    expected = {
        (name, seed, seat) for name in league for seed in seeds for seat in (0, 1)
    }
    if set(traces) != expected:
        raise ValueError("determinism action-trace provenance is incomplete")
    if any(
        type(trace) is not tuple or any(type(action) is not str for action in trace)
        for trace in traces.values()
    ):
        raise TypeError("determinism action traces must be tuples of canonical actions")


def _league_name(league: Mapping[str, Opponent], target: Opponent, label: str) -> str:
    matches = [name for name, opponent in league.items() if opponent == target]
    if len(matches) != 1:
        raise ValueError(
            f"{label} opponent must occur exactly once in promotion league"
        )
    return matches[0]


def _arena_run(
    candidate: Opponent,
    name: str,
    opponent: Opponent,
    seeds: tuple[int, ...],
    workers: int,
) -> _Run:
    expected = 2 * len(seeds)
    started = perf_counter()
    try:
        scores = arena.outcomes(candidate, {name: opponent}, seeds, workers)
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        return _Run(
            pairs=tuple((0.0, 0.0) for _ in seeds),
            margin_pairs=tuple((0.0, 0.0) for _ in seeds),
            failures=(failure,),
            runtime_seconds=perf_counter() - started,
        )
    if len(scores) != expected:
        raise ValueError(
            f"{name}: arena returned {len(scores)} rows; expected {expected} rows"
        )
    values = tuple(float(score) for score in scores)
    if any(
        not math.isfinite(score) or score not in {0.0, 0.5, 1.0} for score in values
    ):
        raise ValueError(f"{name}: arena returned malformed win points")
    raw_failures = getattr(scores, "failures", ())
    if not isinstance(raw_failures, Sequence) or isinstance(raw_failures, (str, bytes)):
        raise TypeError(f"{name}: arena failures must be a sequence")
    failures = tuple(str(item) for item in raw_failures)
    raw_margins = getattr(scores, "margins", None)
    if not isinstance(raw_margins, Sequence) or isinstance(raw_margins, (str, bytes)):
        raise TypeError(f"{name}: arena margins must be a sequence")
    if len(raw_margins) != expected:
        raise ValueError(
            f"{name}: arena returned {len(raw_margins)} margins; expected {expected}"
        )
    margins = tuple(float(margin) for margin in raw_margins)
    if any(not math.isfinite(margin) for margin in margins):
        raise ValueError(f"{name}: arena returned non-finite margins")
    runtime = getattr(scores, "runtime_seconds", 0.0)
    if (
        not isinstance(runtime, (int, float))
        or isinstance(runtime, bool)
        or not math.isfinite(runtime)
        or runtime < 0.0
    ):
        raise ValueError(f"{name}: arena runtime must be finite and non-negative")
    pairs = tuple((values[index], values[index + 1]) for index in range(0, expected, 2))
    margin_pairs = tuple(
        (margins[index], margins[index + 1]) for index in range(0, expected, 2)
    )
    return _Run(pairs, margin_pairs, failures, float(runtime))


def _game_counts(pairs: Sequence[tuple[float, float]]) -> GameCounts:
    """Count exact win points without recomputing any game result."""
    points = tuple(point for pair in pairs for point in pair)
    return GameCounts(
        wins=points.count(1.0),
        draws=points.count(0.5),
        losses=points.count(0.0),
    )
