"""Deterministic progressive-game protocol for the hybrid Optuna study."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kaggriculture.hybrid.config import HybridConfig
from kaggriculture.search.arena import (
    GameKey,
    GameResult,
    GameTask,
    HybridOpponent,
    Opponent,
)
from kaggriculture.search.evolution import (
    DEVELOPMENT_SEEDS,
    PROMOTION_SEEDS,
    SCREENING_SEEDS,
)
from kaggriculture.search.fitness import StrengthWeights, score_fitness
from kaggriculture.search.frontier import FrontierReport
from kaggriculture.search.optuna_space import config_sha256
from kaggriculture.search.promotion import DETERMINISM_SEEDS
from kaggriculture.search.scripts.frontier_round_robin import FRONTIER_SEEDS

RUNG_1_SEEDS = tuple(range(860_000, 860_004))
RUNG_2_SEEDS = tuple(range(860_000, 860_016))
RUNG_3_SEEDS = tuple(range(860_000, 860_032))
_PROTECTED_SEED_BANKS = (
    FRONTIER_SEEDS,
    SCREENING_SEEDS,
    DEVELOPMENT_SEEDS,
    DETERMINISM_SEEDS,
    PROMOTION_SEEDS,
)
assert all(
    set(RUNG_3_SEEDS).isdisjoint(protected) for protected in _PROTECTED_SEED_BANKS
), "Optuna development seeds must remain disjoint from protected seed banks"


@dataclass(frozen=True)
class RungSpec:
    """The immutable panel and resource boundary for one progressive rung."""

    rung: int
    resource_step: int
    opponents: tuple[str, ...]
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class PanelSet:
    """The exact cumulative opponent panels selected from one frontier report."""

    rung_1: tuple[str, ...]
    rung_2: tuple[str, ...]
    rung_3: tuple[str, ...]


class MatchupEvidence(BaseModel):
    """One opponent's fully derived rung metrics."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, allow_inf_nan=False
    )

    opponent: str
    wins: int = Field(ge=0)
    draws: int = Field(ge=0)
    losses: int = Field(ge=0)
    games: int = Field(gt=0)
    win_points: float = Field(ge=0.0, le=1.0)
    paired_normalized_margin: float = Field(ge=-1.0, le=1.0)
    runtime_seconds: float = Field(ge=0.0)


class GameEvidence(BaseModel):
    """The durable candidate-relative result for one exactly named game cell."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, allow_inf_nan=False
    )

    opponent: str
    seed: int
    seat: Literal[0, 1]
    ours: int | None
    theirs: int | None
    runtime_seconds: float = Field(ge=0.0)
    failure: str | None = None

    @model_validator(mode="after")
    def banks_match_failure_status(self) -> Self:
        """Keep successful and failed game records mutually unambiguous."""
        if self.failure is None:
            if self.ours is None or self.theirs is None:
                raise ValueError("successful game evidence requires both banks")
        elif not self.failure:
            raise ValueError("failed game evidence requires a nonempty failure")
        elif self.ours is not None or self.theirs is not None:
            raise ValueError("failed game evidence cannot contain banks")
        return self


class RungScore(BaseModel):
    """The finite optimization score for one clean rung."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, allow_inf_nan=False
    )

    primary: float = Field(ge=0.0, le=1.0)
    dense_margin: float = Field(ge=-1.0, le=1.0)
    objective: float = Field(ge=0.0, le=1.000001)


class IneligibleEvidenceError(ValueError):
    """A rung contains execution failures and cannot receive a score."""


class RungEvidence(BaseModel):
    """Canonical, replayable evidence for a complete progressive rung."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, allow_inf_nan=False
    )

    schema_version: Literal[2] = 2
    trial_number: int = Field(ge=0)
    rung: Literal[1, 2, 3]
    resource_step: Literal[1, 4, 16]
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    opponents: tuple[str, ...]
    seeds: tuple[int, ...]
    games: tuple[GameEvidence, ...]
    matchups: dict[str, MatchupEvidence]
    strength_weights: tuple[tuple[str, int], ...]
    arena_wall_seconds: float = Field(ge=0.0)
    trial_wall_seconds: float = Field(ge=0.0)
    primary: float | None
    dense_margin: float | None
    objective: float | None
    failures: tuple[str, ...]

    @model_validator(mode="after")
    def is_derived_from_raw_games(self) -> Self:
        """Reject summaries and scores that diverge from durable raw game evidence."""
        if self.trial_wall_seconds < self.arena_wall_seconds:
            raise ValueError("trial wall time cannot be below arena wall time")
        spec = RungSpec(
            self.rung,
            self.resource_step,
            self.opponents,
            self.seeds,
        )
        _validate_spec(spec)
        expected_keys = tuple(
            (key.opponent, key.seed, key.seat) for key in _expected_keys(spec)
        )
        actual_keys = tuple(
            (game.opponent, game.seed, game.seat) for game in self.games
        )
        if actual_keys != expected_keys:
            raise ValueError("rung evidence game keys differ from the expected cells")
        expected_failures = _failure_messages(self.games)
        if self.failures != expected_failures:
            raise ValueError("rung evidence failures differ from its raw games")
        expected_matchups = {
            opponent: _matchup_evidence(
                opponent,
                tuple(game for game in self.games if game.opponent == opponent),
            )
            for opponent in self.opponents
        }
        if self.matchups != expected_matchups:
            raise ValueError("rung evidence matchups differ from its raw games")
        weights = _evidence_panel_weights(spec, self.strength_weights)
        values = (self.primary, self.dense_margin, self.objective)
        if self.failures and any(value is not None for value in values):
            raise ValueError("failed rung evidence must not contain scores")
        if not self.failures and any(value is None for value in values):
            raise ValueError("clean rung evidence must contain all scores")
        if not self.failures:
            assert self.primary is not None
            assert self.dense_margin is not None
            assert self.objective is not None
            expected_score = _score_matchups(expected_matchups, weights)
            actual_score = RungScore(
                primary=self.primary,
                dense_margin=self.dense_margin,
                objective=self.objective,
            )
            if actual_score != expected_score:
                raise ValueError("rung evidence scores differ from its raw games")
        return self


def derive_panels(report: FrontierReport) -> PanelSet:
    """Select the fixed 3/6/11 panels from a strongest-first verified report."""
    ranked = tuple(row.name for row in report.rows)
    if len(ranked) != 11:
        raise ValueError("frontier report must contain exactly 11 ranked opponents")
    if len(ranked) != len(set(ranked)):
        raise ValueError("frontier report ranked opponent names must be unique")
    if report.frontier_name != ranked[0]:
        raise ValueError(
            "frontier report frontier_name must be the strongest-ranked name"
        )
    for required in ("economic_policy", "boatlee_v14_current"):
        if required not in ranked:
            raise ValueError(f"frontier report is missing required opponent {required}")

    weakest = tuple(reversed(ranked))
    rung_1 = _fill_distinct(("economic_policy", *weakest), 3)
    median = ranked[len(ranked) // 2]
    rung_2 = _fill_distinct(
        (*rung_1, "boatlee_v14_current", report.frontier_name, median, *weakest), 6
    )
    return PanelSet(rung_1=rung_1, rung_2=rung_2, rung_3=ranked)


def rung_specs(panels: PanelSet) -> tuple[RungSpec, RungSpec, RungSpec]:
    """Return the fixed resource schedule for an already-derived panel set."""
    _validate_panels(panels)
    return (
        RungSpec(1, 1, panels.rung_1, RUNG_1_SEEDS),
        RungSpec(2, 4, panels.rung_2, RUNG_2_SEEDS),
        RungSpec(3, 16, panels.rung_3, RUNG_3_SEEDS),
    )


def missing_game_tasks(
    candidate: HybridOpponent,
    league: Mapping[str, Opponent],
    opponents: Sequence[str],
    seeds: Sequence[int],
    completed: Mapping[GameKey, GameResult],
) -> tuple[GameTask, ...]:
    """Build canonical tasks for only cells absent from prior cumulative evidence."""
    opponent_names = _distinct_names(opponents, "opponents")
    seed_values = _distinct_seeds(seeds)
    missing_league_names = set(opponent_names) - set(league)
    if missing_league_names:
        raise ValueError(
            "league is missing requested opponents: "
            + ", ".join(sorted(missing_league_names))
        )
    _validate_completed(completed)
    return tuple(
        GameTask(GameKey(name, seed, seat), candidate, league[name])
        for name in opponent_names
        for seed in seed_values
        for seat in (0, 1)
        if GameKey(name, seed, seat) not in completed
    )


def _fill_distinct(candidates: Sequence[str], count: int) -> tuple[str, ...]:
    """Keep first appearances and fail rather than returning an undersized panel."""
    selected: list[str] = []
    for name in candidates:
        if name not in selected:
            selected.append(name)
        if len(selected) == count:
            return tuple(selected)
    raise ValueError(f"could not select {count} distinct opponents")


def _validate_panels(panels: PanelSet) -> None:
    """Reject panel facts that cannot describe the mandated cumulative schedule."""
    if not isinstance(panels, PanelSet):
        raise ValueError("panels must be a PanelSet")
    rung_1 = _distinct_names(panels.rung_1, "rung 1 panel")
    rung_2 = _distinct_names(panels.rung_2, "rung 2 panel")
    rung_3 = _distinct_names(panels.rung_3, "rung 3 panel")
    if len(rung_1) != 3 or len(rung_2) != 6 or len(rung_3) != 11:
        raise ValueError("panels must contain exactly 3, 6, and 11 opponents")
    if not set(rung_1) <= set(rung_2) <= set(rung_3):
        raise ValueError("rung panels must be cumulative")


def _distinct_names(values: Sequence[str], label: str) -> tuple[str, ...]:
    """Validate one ordered, nonempty set of canonical opponent names."""
    names = tuple(values)
    if not names or any(type(name) is not str or not name for name in names):
        raise ValueError(f"{label} must contain nonempty string names")
    if len(names) != len(set(names)):
        raise ValueError(f"{label} names must be unique")
    return names


def _distinct_seeds(values: Sequence[int]) -> tuple[int, ...]:
    """Validate one ordered, nonempty set of integer episode seeds."""
    seeds = tuple(values)
    if not seeds or any(type(seed) is not int for seed in seeds):
        raise ValueError("seeds must contain integers")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")
    return seeds


def _validate_completed(completed: Mapping[GameKey, GameResult]) -> None:
    """Reject prior evidence whose mapping key diverges from the result provenance."""
    for key, result in completed.items():
        if not isinstance(key, GameKey) or not isinstance(result, GameResult):
            raise ValueError(
                "completed games must map GameKey values to GameResult rows"
            )
        if result.key != key:
            raise ValueError(
                "completed game result provenance differs from its mapping key"
            )


def build_rung_evidence(
    trial_number: int,
    config: HybridConfig,
    spec: RungSpec,
    completed: Mapping[GameKey, GameResult],
    weights: StrengthWeights,
    *,
    arena_wall_seconds: float = 0.0,
    trial_wall_seconds: float = 0.0,
) -> RungEvidence:
    """Derive complete canonical rung evidence from exactly the requested cells."""
    if type(trial_number) is not int or trial_number < 0:
        raise ValueError("trial number must be a non-negative integer")
    if not isinstance(config, HybridConfig):
        raise ValueError("config must be a HybridConfig")
    _validate_spec(spec)
    panel_weights = _panel_weights(spec, weights)
    expected_keys = _expected_keys(spec)
    _validate_completed(completed)
    if set(completed) != set(expected_keys):
        raise ValueError("completed games must contain exactly the expected rung cells")

    ordered_rows = tuple(completed[key] for key in expected_keys)
    games = tuple(_game_evidence(row) for row in ordered_rows)
    failures = _failure_messages(games)
    matchups = {
        opponent: _matchup_evidence(
            opponent,
            tuple(game for game in games if game.opponent == opponent),
        )
        for opponent in spec.opponents
    }
    primary: float | None = None
    dense_margin: float | None = None
    objective: float | None = None
    if not failures:
        score = _score_matchups(matchups, panel_weights)
        primary = score.primary
        dense_margin = score.dense_margin
        objective = score.objective
    return RungEvidence(
        trial_number=trial_number,
        rung=cast(Literal[1, 2, 3], spec.rung),
        resource_step=cast(Literal[1, 4, 16], spec.resource_step),
        config_sha256=config_sha256(config),
        opponents=spec.opponents,
        seeds=spec.seeds,
        games=games,
        matchups=matchups,
        strength_weights=tuple(
            (name, panel_weights.values[name]) for name in spec.opponents
        ),
        arena_wall_seconds=arena_wall_seconds,
        trial_wall_seconds=trial_wall_seconds,
        primary=primary,
        dense_margin=dense_margin,
        objective=objective,
        failures=failures,
    )


def score_evidence(evidence: RungEvidence, weights: StrengthWeights) -> RungScore:
    """Return the lexicographic scalar only when the complete rung is clean."""
    if not isinstance(evidence, RungEvidence):
        raise ValueError("evidence must be a RungEvidence")
    evidence = _validated_evidence(evidence)
    spec = RungSpec(
        evidence.rung,
        evidence.resource_step,
        evidence.opponents,
        evidence.seeds,
    )
    _validate_spec(spec)
    panel_weights = _panel_weights(spec, weights)
    if panel_weights != _evidence_panel_weights(spec, evidence.strength_weights):
        raise ValueError("evidence strength weights differ from the supplied weights")
    if evidence.failures:
        raise IneligibleEvidenceError("; ".join(evidence.failures))
    score = _score_matchups(evidence.matchups, panel_weights)
    if (
        evidence.primary != score.primary
        or evidence.dense_margin != score.dense_margin
        or evidence.objective != score.objective
    ):
        raise ValueError("rung evidence scores differ from its matchup aggregates")
    return score


def minimum_primary_increment(spec: RungSpec, weights: StrengthWeights) -> float:
    """Return a conservative positive primary delta for one result half-step."""
    _validate_spec(spec)
    panel_weights = _panel_weights(spec, weights)
    return _minimum_primary_increment(spec, panel_weights)


def _minimum_primary_increment(spec: RungSpec, weights: StrengthWeights) -> float:
    """Calculate the bound after the caller has selected exact panel weights."""
    return (
        0.70
        * min(weights.values.values())
        / sum(weights.values.values())
        * 0.5
        / (2 * len(spec.seeds))
    )


def objective_value(primary: float, dense_margin: float) -> float:
    """Encode lexicographic primary/margin ranking as one Optuna scalar."""
    _unit_float(primary, "primary")
    _bounded_float(dense_margin, -1.0, 1.0, "dense margin")
    return primary + 1e-6 * ((dense_margin + 1.0) / 2.0)


def evidence_path(root: Path, trial_number: int, rung: int) -> Path:
    """Return the sole durable path for one trial/rung evidence record."""
    if type(trial_number) is not int or trial_number < 0:
        raise ValueError("trial number must be a non-negative integer")
    if type(rung) is not int or rung not in (1, 2, 3):
        raise ValueError("rung must be 1, 2, or 3")
    return root / "evidence" / f"trial-{trial_number}-rung-{rung}.json"


def rung_summary_attributes(
    evidence: RungEvidence,
    path: Path,
    digest: str,
    root: Path,
) -> dict[str, object]:
    """Derive the exact SQLite summary recoverable from one canonical rung."""
    if not isinstance(evidence, RungEvidence):
        raise ValueError("evidence must be a RungEvidence")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("evidence digest must be lowercase SHA-256")
    expected_path = evidence_path(root, evidence.trial_number, evidence.rung)
    if path != expected_path:
        raise ValueError("evidence summary path is not canonical")
    attributes: dict[str, object] = {
        "config_sha256": evidence.config_sha256,
        "rung": evidence.rung,
        "resource_step": evidence.resource_step,
        "primary": evidence.primary,
        "dense_margin": evidence.dense_margin,
        "objective": evidence.objective,
        "games": len(evidence.games),
        "failures": len(evidence.failures),
        "runtime_seconds": sum(game.runtime_seconds for game in evidence.games),
        "arena_wall_seconds": evidence.arena_wall_seconds,
        "trial_wall_seconds": evidence.trial_wall_seconds,
        "evidence_path": str(path.relative_to(root)),
        "evidence_sha256": digest,
    }
    for name, matchup in evidence.matchups.items():
        prefix = f"opponent/{name}"
        attributes.update(
            {
                f"{prefix}/wins": matchup.wins,
                f"{prefix}/draws": matchup.draws,
                f"{prefix}/losses": matchup.losses,
                f"{prefix}/games": matchup.games,
                f"{prefix}/win_points": matchup.win_points,
                f"{prefix}/paired_normalized_margin": (
                    matchup.paired_normalized_margin
                ),
                f"{prefix}/runtime_seconds": matchup.runtime_seconds,
            }
        )
    return attributes


def write_rung_evidence_atomic(root: Path, evidence: RungEvidence) -> tuple[Path, str]:
    """Fsync canonical evidence before atomically replacing its published path."""
    if not isinstance(evidence, RungEvidence):
        raise ValueError("evidence must be a RungEvidence")
    evidence = _validated_evidence(evidence)
    output = evidence_path(root, evidence.trial_number, evidence.rung)
    output.parent.mkdir(parents=True, exist_ok=True)
    source = (
        json.dumps(
            evidence.model_dump(mode="json"),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(source)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)  # noqa: PTH105
        _fsync_directory(output.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output, hashlib.sha256(source).hexdigest()


def _validate_spec(spec: RungSpec) -> None:
    """Validate a self-consistent rung boundary before consuming its facts."""
    if not isinstance(spec, RungSpec):
        raise ValueError("spec must be a RungSpec")
    resource_steps = {1: 1, 2: 4, 3: 16}
    if type(spec.rung) is not int or spec.rung not in resource_steps:
        raise ValueError("rung must be 1, 2, or 3")
    if spec.resource_step != resource_steps[spec.rung]:
        raise ValueError("rung resource step is invalid")
    _distinct_names(spec.opponents, "rung opponents")
    _distinct_seeds(spec.seeds)


def _panel_weights(spec: RungSpec, weights: StrengthWeights) -> StrengthWeights:
    """Project full-league weights to the exact current panel without changing them."""
    if not isinstance(weights, StrengthWeights):
        raise ValueError("weights must be StrengthWeights")
    missing = set(spec.opponents) - set(weights.values)
    if missing:
        raise ValueError(
            "strength weights are missing rung opponents: " + ", ".join(sorted(missing))
        )
    panel_weights = StrengthWeights(
        {name: weights.values[name] for name in spec.opponents}
    )
    if not _float_primary_increment_dominates_dense_range(spec, panel_weights):
        raise ValueError("strength weights violate lexicographic dominance")
    return panel_weights


def _float_primary_increment_dominates_dense_range(
    spec: RungSpec, weights: StrengthWeights
) -> bool:
    """Reserve a ULP envelope for two rounded primary/objective evaluations.

    A comparison can evaluate two weighted primary scores and two scalar
    objectives. For a panel of ``n`` matchups, their independent weighted
    sums/multiplications account for at most ``4 * n`` unit-roundoff steps;
    the remaining scalar operations fit in a further 16 steps. Every primary
    value lies in ``[0, 1]``, so one step is conservatively bounded by
    ``math.ulp(1.0)``. This rejects an algebraically-safe increment that would
    collapse to the dense-margin range in binary floating-point arithmetic.
    """
    rounding_envelope = math.ulp(1.0) * (4 * len(spec.opponents) + 16)
    return _minimum_primary_increment(spec, weights) - rounding_envelope > 1e-6


def _evidence_panel_weights(
    spec: RungSpec, pairs: tuple[tuple[str, int], ...]
) -> StrengthWeights:
    """Recover and validate the exact per-panel weights persisted with evidence."""
    if tuple(name for name, _ in pairs) != spec.opponents:
        raise ValueError("evidence strength weights must follow the opponent panel")
    return _panel_weights(spec, StrengthWeights(dict(pairs)))


def _validated_evidence(evidence: RungEvidence) -> RungEvidence:
    """Re-run strict model validation after a frozen model's nested data may mutate."""
    return RungEvidence.model_validate(evidence.model_dump())


def _expected_keys(spec: RungSpec) -> tuple[GameKey, ...]:
    """Enumerate expected cells in durable opponent/seed/seat order."""
    return tuple(
        GameKey(opponent, seed, seat)
        for opponent in spec.opponents
        for seed in spec.seeds
        for seat in (0, 1)
    )


def _game_evidence(row: GameResult) -> GameEvidence:
    """Copy one typed arena row into the frozen evidence schema."""
    return GameEvidence(
        opponent=row.key.opponent,
        seed=row.key.seed,
        seat=cast(Literal[0, 1], row.key.seat),
        ours=row.ours,
        theirs=row.theirs,
        runtime_seconds=float(row.runtime_seconds),
        failure=row.failure,
    )


def _failure_messages(games: Sequence[GameEvidence]) -> tuple[str, ...]:
    """Format failure provenance directly from the canonical ordered game records."""
    return tuple(
        f"{game.opponent}/{game.seed}/seat{game.seat}: {game.failure}"
        for game in games
        if game.failure is not None
    )


def _matchup_evidence(opponent: str, rows: Sequence[GameEvidence]) -> MatchupEvidence:
    """Aggregate one complete opponent block without inventing failed outcomes."""
    successes = tuple(
        (row, _successful_banks(row)) for row in rows if row.failure is None
    )
    wins = sum(ours > theirs for _, (ours, theirs) in successes)
    draws = sum(ours == theirs for _, (ours, theirs) in successes)
    losses = sum(ours < theirs for _, (ours, theirs) in successes)
    games = len(rows)
    return MatchupEvidence(
        opponent=opponent,
        wins=wins,
        draws=draws,
        losses=losses,
        games=games,
        win_points=(wins + 0.5 * draws) / games,
        paired_normalized_margin=sum(
            _normalized_margin(ours, theirs) for _, (ours, theirs) in successes
        )
        / games,
        runtime_seconds=sum(float(row.runtime_seconds) for row in rows),
    )


def _successful_banks(row: GameEvidence) -> tuple[int, int]:
    """Return successful banks while defending the evidence-row invariant."""
    if row.ours is None or row.theirs is None:
        raise ValueError("successful game evidence is missing bank values")
    return row.ours, row.theirs


def _score_matchups(
    matchups: Mapping[str, MatchupEvidence], weights: StrengthWeights
) -> RungScore:
    """Calculate exact primary and dense components from validated matchups."""
    if set(matchups) != set(weights.values):
        raise ValueError("matchup names must exactly match strength weights")
    if any(matchup.opponent != name for name, matchup in matchups.items()):
        raise ValueError("matchup evidence names must match their mapping keys")
    fitness = score_fitness(
        {name: matchup.win_points for name, matchup in matchups.items()}, weights
    )
    total_weight = sum(weights.values.values())
    dense_margin = (
        sum(
            matchups[name].paired_normalized_margin * weight
            for name, weight in weights.values.items()
        )
        / total_weight
    )
    primary = fitness.value
    return RungScore(
        primary=primary,
        dense_margin=dense_margin,
        objective=objective_value(primary, dense_margin),
    )


def _normalized_margin(ours: int, theirs: int) -> float:
    """Match the arena's bounded candidate-relative bank-margin calculation."""
    return (ours - theirs) / max(abs(ours) + abs(theirs), 1)


def _unit_float(value: float, label: str) -> None:
    """Reject non-finite values outside a unit-interval objective component."""
    _bounded_float(value, 0.0, 1.0, label)


def _bounded_float(value: float, low: float, high: float, label: str) -> None:
    """Reject Boolean, non-finite, and out-of-range scalar inputs."""
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not low <= value <= high
    ):
        raise ValueError(f"{label} must be finite and between {low} and {high}")


def _fsync_directory(directory: Path) -> None:
    """Persist the renamed directory entry, not only the temporary file bytes."""
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
