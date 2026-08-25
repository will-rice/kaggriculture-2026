"""Verify an exact public-agent archive and measure its local field order."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    PositiveInt,
    model_validator,
)

from kaggriculture.search import arena


class FrontierIntegrityError(RuntimeError):
    """Raised when an archived frontier artifact cannot be trusted."""


class FrontierArtifact(BaseModel):
    """One executable agent with enough provenance to reproduce it exactly."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9_]+$")
    source_url: HttpUrl | None = None
    notebook_version: PositiveInt | None = None
    historical_score: float | None = None
    provenance: str = Field(min_length=1)
    relative_path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    archive_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def complete_notebook_identity(self) -> Self:
        """Keep Kaggle provenance complete enough to resolve a past version."""
        if (self.source_url is None) != (self.notebook_version is None):
            raise ValueError("source_url and notebook_version must be set together")
        if self.source_url is not None and self.archive_sha256 is None:
            raise ValueError("Kaggle artifacts require archive_sha256")
        return self


class FrontierManifest(BaseModel):
    """Immutable list of exact sources evaluated by a frontier run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    engine: str = Field(min_length=1)
    artifacts: tuple[FrontierArtifact, ...]

    @model_validator(mode="after")
    def distinct_artifact_names(self) -> Self:
        """Prevent a dict lookup from silently discarding a manifest entry."""
        names = [artifact.name for artifact in self.artifacts]
        if len(names) != len(set(names)):
            raise ValueError("frontier artifact names must be unique")
        return self


@dataclass(frozen=True)
class VerifiedFrontier:
    """Manifest paths that have passed byte-for-byte source validation."""

    engine: str
    opponents: Mapping[str, str]
    artifacts: tuple[FrontierArtifact, ...]


@dataclass(frozen=True)
class FrontierMatchup:
    """A candidate's complete seat-swapped result against one opponent."""

    opponent: str
    games: int
    win_points: float
    wins: int
    draws: int
    losses: int


@dataclass(frozen=True)
class FrontierRow:
    """One policy's field summary and all of its individual matchups."""

    name: str
    games: int
    field_win_points: float
    worst_matchup_win_points: float
    matchups: Mapping[str, float]
    matchup_results: tuple[FrontierMatchup, ...]


@dataclass(frozen=True)
class FrontierReport:
    """A reproducible non-transitive round-robin matrix and its ordering."""

    engine: str
    seeds: tuple[int, ...]
    frontier_name: str
    rows: tuple[FrontierRow, ...]


def verify_frontier(manifest_path: Path, artifact_root: Path) -> VerifiedFrontier:
    """Load a manifest only when every runnable source has its declared digest."""
    manifest = FrontierManifest.model_validate_json(manifest_path.read_text())
    root = artifact_root.resolve()
    resolved: dict[str, str] = {}
    for artifact in manifest.artifacts:
        source = (root / artifact.relative_path).resolve()
        if root not in source.parents:
            raise FrontierIntegrityError(f"{artifact.name}: path escapes artifact root")
        try:
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
        except OSError as error:
            raise FrontierIntegrityError(
                f"{artifact.name}: cannot read {artifact.relative_path}: {error}"
            ) from error
        if digest != artifact.sha256:
            raise FrontierIntegrityError(
                f"{artifact.name}: sha256 {digest} != {artifact.sha256}"
            )
        resolved[artifact.name] = str(source)
    return VerifiedFrontier(manifest.engine, resolved, manifest.artifacts)


def rank_frontier(
    frontier: VerifiedFrontier, seeds: Sequence[int], workers: int
) -> FrontierReport:
    """Rank a verified field by all-pairs, seat-swapped average win points."""
    seed_tuple = tuple(int(seed) for seed in seeds)
    names = tuple(frontier.opponents)
    matchup_results: dict[str, list[FrontierMatchup]] = {name: [] for name in names}
    games_per_matchup = 2 * len(seed_tuple)
    for candidate_index, name in enumerate(names):
        candidate = frontier.opponents[name]
        for opponent_name in names[candidate_index + 1 :]:
            scores = arena.outcomes(
                candidate,
                {opponent_name: frontier.opponents[opponent_name]},
                seed_tuple,
                workers,
            )
            if len(scores) != games_per_matchup:
                raise FrontierIntegrityError(
                    f"{name} vs {opponent_name}: arena returned {len(scores)} scores "
                    f"for {games_per_matchup} games"
                )
            matchup_results[name].append(_matchup(opponent_name, scores))
            matchup_results[opponent_name].append(
                _matchup(name, tuple(1.0 - score for score in scores))
            )
    rows: list[FrontierRow] = []
    for name in names:
        results = tuple(matchup_results[name])
        matchups = {result.opponent: result.win_points for result in results}
        games = sum(result.games for result in results)
        field_points = (
            sum(result.win_points * result.games for result in results) / games
            if games
            else 0.0
        )
        worst_points = min(matchups.values(), default=0.0)
        rows.append(
            FrontierRow(
                name=name,
                games=games,
                field_win_points=field_points,
                worst_matchup_win_points=worst_points,
                matchups=matchups,
                matchup_results=results,
            )
        )
    ordered = tuple(
        sorted(
            rows,
            key=lambda row: (
                -row.field_win_points,
                -row.worst_matchup_win_points,
                row.name,
            ),
        )
    )
    if not ordered:
        raise FrontierIntegrityError("frontier manifest contains no artifacts")
    return FrontierReport(
        engine=frontier.engine,
        seeds=seed_tuple,
        frontier_name=ordered[0].name,
        rows=ordered,
    )


def _matchup(opponent: str, scores: Sequence[float]) -> FrontierMatchup:
    """Summarize one flat, seat-swapped block returned by ``arena.outcomes``."""
    return FrontierMatchup(
        opponent=opponent,
        games=len(scores),
        win_points=sum(scores) / len(scores) if scores else 0.0,
        wins=sum(score == 1.0 for score in scores),
        draws=sum(score == 0.5 for score in scores),
        losses=sum(score == 0.0 for score in scores),
    )
