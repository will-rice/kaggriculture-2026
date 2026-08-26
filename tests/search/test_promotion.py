"""Exact paired evidence and atomic rulings for hybrid promotion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import pytest

from kaggriculture.hybrid.config import HybridConfig, to_runtime
from kaggriculture.search import arena
from kaggriculture.search.arena import HybridOpponent, Opponent
from kaggriculture.search.evolution import DEVELOPMENT_SEEDS, SCREENING_SEEDS
from kaggriculture.search.fitness import StrengthWeights
from kaggriculture.search.promotion import (
    PROMOTION_SEEDS,
    GameCounts,
    Interval,
    PromotionVerdict,
    bootstrap_interval,
    evaluate_promotion,
)
from kaggriculture.search.scripts.frontier_round_robin import FRONTIER_SEEDS


class _Scores(list[float]):
    """Complete deterministic arena result returned by a fake reference engine."""

    def __init__(
        self,
        values: Sequence[float],
        *,
        failures: Sequence[str] = (),
        margins: Sequence[float] | None = None,
    ) -> None:
        super().__init__(values)
        self.failures = list(failures)
        self.margins = list(margins if margins is not None else (0.0,) * len(values))
        self.runtime_seconds = 0.25


@dataclass(frozen=True)
class _Fixture:
    candidate: HybridOpponent
    incumbent: Opponent
    boatlee: Opponent
    frontier: Opponent
    league: Mapping[str, Opponent]


@pytest.fixture
def opponents() -> _Fixture:
    """Return distinct identities for every gate role and league member."""
    return _Fixture(
        candidate=HybridOpponent(to_runtime(HybridConfig.default())),
        incumbent="incumbent.py",
        boatlee="boatlee.py",
        frontier="frontier.py",
        league={"strong": "strong.py", "old_meta": "old.py"},
    )


def test_promotion_requires_every_gate_not_only_the_aggregate() -> None:
    """A single matchup regression cannot be hidden by passing aggregates."""
    verdict = PromotionVerdict(
        boatlee=Interval(0.55, 0.51, 0.59),
        frontier=Interval(0.54, 0.501, 0.58),
        league_delta=Interval(0.03, 0.01, 0.05),
        matchup_deltas={"old_meta": -0.051},
        failures=0,
        deterministic=True,
    )

    assert verdict.passed is False
    assert verdict.reasons == ("old_meta regressed by 0.051 (> 0.050)",)


@pytest.mark.parametrize(
    ("changes", "reason_fragment"),
    [
        ({"boatlee": Interval(0.51, 0.50, 0.52)}, "Boatlee"),
        ({"frontier": Interval(0.51, 0.50, 0.52)}, "frontier"),
        ({"league_delta": Interval(0.01, 0.0, 0.02)}, "league delta"),
        ({"failures": 1}, "failure"),
        ({"deterministic": False}, "nondeterministic"),
    ],
)
def test_each_atomic_gate_can_fail_a_strong_candidate(
    changes: Mapping[str, object], reason_fragment: str
) -> None:
    """Removing any one hard gate must make an otherwise passing verdict fail."""
    fields: dict[str, object] = {
        "boatlee": Interval(0.60, 0.51, 0.70),
        "frontier": Interval(0.59, 0.501, 0.68),
        "league_delta": Interval(0.04, 0.001, 0.08),
        "matchup_deltas": {"old_meta": -0.05},
        "failures": 0,
        "deterministic": True,
    }
    fields.update(changes)

    verdict = PromotionVerdict(
        boatlee=fields["boatlee"],
        frontier=fields["frontier"],
        league_delta=fields["league_delta"],
        matchup_deltas=cast(Mapping[str, float], fields["matchup_deltas"]),
        failures=cast(int, fields["failures"]),
        deterministic=cast(bool, fields["deterministic"]),
    )

    assert verdict.passed is False
    assert any(reason_fragment in reason for reason in verdict.reasons)


def test_bootstrap_pairs_both_seats_of_each_seed() -> None:
    """Changing the random draw cannot split the two seats of one seed."""
    samples = ((1.0, 0.0), (0.5, 0.5), (1.0, 1.0))

    first = bootstrap_interval(samples, draws=10_000, seed=20_260_825)
    second = bootstrap_interval(samples, draws=10_000, seed=20_260_825)

    assert first == second
    assert first.mean == pytest.approx(2.0 / 3.0)


@pytest.mark.parametrize(
    "samples",
    [(), ((1.0,),), ((1.0, float("nan")),), ((1.0, True),)],
)
def test_bootstrap_rejects_incomplete_or_nonfinite_pairs(
    samples: Sequence[tuple[float, ...]],
) -> None:
    """Malformed seat provenance cannot silently enter an interval."""
    with pytest.raises((TypeError, ValueError)):
        bootstrap_interval(cast(Sequence[tuple[float, float]], samples))


def test_holdout_seeds_are_disjoint_from_every_search_seed() -> None:
    """The exact 128 promotion seeds remain untouched by all search stages."""
    assert set(PROMOTION_SEEDS).isdisjoint(FRONTIER_SEEDS)
    assert set(PROMOTION_SEEDS).isdisjoint(SCREENING_SEEDS)
    assert set(PROMOTION_SEEDS).isdisjoint(DEVELOPMENT_SEEDS)
    assert len(PROMOTION_SEEDS) == 128
    assert len(set(PROMOTION_SEEDS)) == len(PROMOTION_SEEDS)


def test_evaluation_pairs_identical_candidate_and_incumbent_rows(
    monkeypatch: pytest.MonkeyPatch, opponents: _Fixture
) -> None:
    """Weighted deltas use the same opponent, seed, and seat rows on both sides."""
    rows = {
        "boatlee.py": (1.0, 1.0, 1.0, 1.0),
        "frontier.py": (1.0, 1.0, 1.0, 1.0),
        "strong.py": (1.0, 0.0, 1.0, 0.0),
        "old.py": (0.5, 0.5, 0.5, 0.5),
    }
    incumbent_rows = {
        "strong.py": (0.0, 0.0, 0.0, 0.0),
        "old.py": (0.5, 0.5, 0.5, 0.5),
    }
    calls: list[tuple[Opponent, str, tuple[int, ...], int]] = []

    def fake_outcomes(
        candidate: Opponent,
        league: Mapping[str, Opponent],
        seeds: Sequence[int],
        workers: int | None = None,
    ) -> _Scores:
        [(name, opponent)] = league.items()
        calls.append((candidate, name, tuple(seeds), int(workers or 0)))
        values = (
            incumbent_rows[str(opponent)]
            if candidate == opponents.incumbent
            else rows[str(opponent)]
        )
        return _Scores(values)

    monkeypatch.setattr(arena, "outcomes", fake_outcomes)

    verdict = evaluate_promotion(
        opponents.candidate,
        opponents.incumbent,
        opponents.boatlee,
        opponents.frontier,
        opponents.league,
        (7, 9),
        3,
        weights=StrengthWeights({"strong": 3, "old_meta": 1}),
    )

    assert verdict.boatlee.mean == 1.0
    assert verdict.frontier.mean == 1.0
    assert verdict.boatlee_counts == GameCounts(wins=4, draws=0, losses=0)
    assert verdict.frontier_counts == GameCounts(wins=4, draws=0, losses=0)
    assert verdict.matchup_deltas == {"strong": 0.5, "old_meta": 0.0}
    assert verdict.league_delta.mean == pytest.approx(0.375)
    assert verdict.failures == 0
    assert verdict.deterministic is True
    assert verdict.passed is True
    assert all(call[2:] == ((7, 9), 3) for call in calls)


def test_evaluation_retains_direct_and_league_paired_bank_margins(
    monkeypatch: pytest.MonkeyPatch, opponents: _Fixture
) -> None:
    """The verdict carries the raw paired margins needed to audit every matchup."""

    def fake_outcomes(
        candidate: Opponent,
        league: Mapping[str, Opponent],
        seeds: Sequence[int],
        workers: int | None = None,
    ) -> _Scores:
        del seeds, workers
        [(name, _)] = league.items()
        margins = (10.0, 20.0) if candidate == opponents.candidate else (3.0, 4.0)
        points = (1.0, 1.0) if candidate == opponents.candidate else (0.0, 0.0)
        return _Scores(points, margins=margins)

    monkeypatch.setattr(arena, "outcomes", fake_outcomes)

    verdict = evaluate_promotion(
        opponents.candidate,
        opponents.incumbent,
        opponents.boatlee,
        opponents.frontier,
        {"strong": opponents.league["strong"]},
        (7,),
        1,
        weights=StrengthWeights({"strong": 4}),
    )

    assert verdict.boatlee_margin_pairs == ((10.0, 20.0),)
    assert verdict.frontier_margin_pairs == ((10.0, 20.0),)
    assert verdict.candidate_matchup_margin_pairs == {"strong": ((10.0, 20.0),)}
    assert verdict.incumbent_matchup_margin_pairs == {"strong": ((3.0, 4.0),)}


def test_evaluation_detects_nondeterministic_candidate_rows(
    monkeypatch: pytest.MonkeyPatch, opponents: _Fixture
) -> None:
    """A replay disagreement is a hard gate even if every first run wins."""
    candidate_call = 0

    def fake_outcomes(
        candidate: Opponent,
        league: Mapping[str, Opponent],
        seeds: Sequence[int],
        workers: int | None = None,
    ) -> _Scores:
        nonlocal candidate_call
        del league, seeds, workers
        if candidate == opponents.incumbent:
            return _Scores((0.0, 0.0))
        candidate_call += 1
        return _Scores((1.0, 1.0) if candidate_call % 2 else (1.0, 0.5))

    monkeypatch.setattr(arena, "outcomes", fake_outcomes)

    verdict = evaluate_promotion(
        opponents.candidate,
        opponents.incumbent,
        opponents.boatlee,
        opponents.frontier,
        {"strong": "strong.py"},
        (7,),
        1,
        weights=StrengthWeights({"strong": 4}),
    )

    assert verdict.deterministic is False
    assert any("nondeterministic" in reason for reason in verdict.reasons)


@pytest.mark.parametrize("seeds", [(7, 7), ()])
def test_evaluation_rejects_duplicate_or_missing_seed_provenance(
    seeds: Sequence[int], opponents: _Fixture
) -> None:
    """A holdout row cannot be identified if seed provenance is ambiguous."""
    with pytest.raises(ValueError, match="seed"):
        evaluate_promotion(
            opponents.candidate,
            opponents.incumbent,
            opponents.boatlee,
            opponents.frontier,
            opponents.league,
            seeds,
            1,
            weights=StrengthWeights({"strong": 3, "old_meta": 1}),
        )


def test_evaluation_rejects_missing_game_rows(
    monkeypatch: pytest.MonkeyPatch, opponents: _Fixture
) -> None:
    """An aggregate over an incomplete seat pair is not promotion evidence."""
    monkeypatch.setattr(arena, "outcomes", lambda *args, **kwargs: _Scores((1.0,)))

    with pytest.raises(ValueError, match="2 rows"):
        evaluate_promotion(
            opponents.candidate,
            opponents.incumbent,
            opponents.boatlee,
            opponents.frontier,
            opponents.league,
            (7,),
            1,
            weights=StrengthWeights({"strong": 3, "old_meta": 1}),
        )


def test_execution_failure_is_recorded_without_hiding_remaining_matchups(
    monkeypatch: pytest.MonkeyPatch, opponents: _Fixture
) -> None:
    """One failed matchup hard-fails the verdict after all evidence is collected."""
    calls: list[tuple[Opponent, str]] = []

    def fake_outcomes(
        candidate: Opponent,
        league: Mapping[str, Opponent],
        seeds: Sequence[int],
        workers: int | None = None,
    ) -> _Scores:
        del seeds, workers
        [(name, _)] = league.items()
        calls.append((candidate, name))
        if candidate == opponents.candidate and name == "old_meta":
            raise RuntimeError("forfeit")
        return _Scores((1.0, 1.0))

    monkeypatch.setattr(arena, "outcomes", fake_outcomes)

    verdict = evaluate_promotion(
        opponents.candidate,
        opponents.incumbent,
        opponents.boatlee,
        opponents.frontier,
        opponents.league,
        (7,),
        1,
        weights=StrengthWeights({"strong": 3, "old_meta": 1}),
    )

    assert verdict.passed is False
    assert verdict.failures == 1
    assert "old_meta: RuntimeError: forfeit" in verdict.failure_details[0]
    assert (opponents.incumbent, "old_meta") in calls
