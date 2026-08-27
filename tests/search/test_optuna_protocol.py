"""Progressive Optuna rung protocol contracts."""

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from kaggriculture.hybrid.config import HybridConfig, to_runtime
from kaggriculture.search.arena import GameKey, GameResult, HybridOpponent
from kaggriculture.search.evolution import (
    DEVELOPMENT_SEEDS,
    PROMOTION_SEEDS,
    SCREENING_SEEDS,
)
from kaggriculture.search.fitness import StrengthWeights
from kaggriculture.search.frontier import FrontierReport, FrontierRow
from kaggriculture.search.optuna_protocol import (
    RUNG_1_SEEDS,
    RUNG_2_SEEDS,
    RUNG_3_SEEDS,
    GameEvidence,
    IneligibleEvidenceError,
    RungSpec,
    build_rung_evidence,
    derive_panels,
    evidence_path,
    minimum_primary_increment,
    missing_game_tasks,
    objective_value,
    rung_specs,
    score_evidence,
    write_rung_evidence_atomic,
)
from kaggriculture.search.promotion import DETERMINISM_SEEDS
from kaggriculture.search.scripts.frontier_round_robin import FRONTIER_SEEDS

NAMES = (
    "frontier_policy",
    "strong_policy",
    "boatlee_v14_current",
    "economic_policy",
    "middle_policy",
    "sixth_policy",
    "seventh_policy",
    "eighth_policy",
    "ninth_policy",
    "tenth_policy",
    "weakest_policy",
)


def _row(name: str) -> FrontierRow:
    return FrontierRow(
        name=name,
        games=0,
        field_win_points=0.0,
        worst_matchup_win_points=0.0,
        paired_margin=0.0,
        failures=(),
        runtime_seconds=0.0,
        matchups={},
        matchup_results=(),
    )


@pytest.fixture
def frontier_report() -> FrontierReport:
    """Return an exact strongest-first public frontier fixture."""
    return FrontierReport(
        engine="1.32.7",
        seeds=(1,),
        frontier_name="frontier_policy",
        failures=(),
        runtime_seconds=0.0,
        rows=tuple(_row(name) for name in NAMES),
    )


def test_nested_seed_banks_are_fixed_and_protected() -> None:
    """Changing a tuning seed must fail before it can overlap a protected split."""
    assert RUNG_1_SEEDS == tuple(range(860_000, 860_004))
    assert RUNG_2_SEEDS == tuple(range(860_000, 860_016))
    assert RUNG_3_SEEDS == tuple(range(860_000, 860_032))
    assert set(RUNG_1_SEEDS) < set(RUNG_2_SEEDS) < set(RUNG_3_SEEDS)
    for protected in (
        FRONTIER_SEEDS,
        SCREENING_SEEDS,
        DEVELOPMENT_SEEDS,
        DETERMINISM_SEEDS,
        PROMOTION_SEEDS,
    ):
        assert set(RUNG_3_SEEDS).isdisjoint(protected)


def test_panels_are_exact_three_six_eleven_from_ranked_report(
    frontier_report: FrontierReport,
) -> None:
    """Rank ordering determines the three cumulative panels without ambiguity."""
    panels = derive_panels(frontier_report)

    assert panels.rung_1 == ("economic_policy", "weakest_policy", "tenth_policy")
    assert len(panels.rung_2) == 6
    assert {"boatlee_v14_current", frontier_report.frontier_name} <= set(panels.rung_2)
    assert panels.rung_3 == NAMES


def test_panels_reject_missing_required_or_ambiguous_ranked_names(
    frontier_report: FrontierReport,
) -> None:
    """A shortened or duplicate public report cannot silently select a panel."""
    without_economic = replace(
        frontier_report,
        rows=tuple(
            row for row in frontier_report.rows if row.name != "economic_policy"
        ),
    )

    with pytest.raises(ValueError, match="exactly 11"):
        derive_panels(without_economic)
    with pytest.raises(ValueError, match="unique"):
        derive_panels(
            replace(
                frontier_report,
                rows=(*frontier_report.rows[:-1], _row(NAMES[0])),
            )
        )


def test_later_rung_runs_only_missing_cells(frontier_report: FrontierReport) -> None:
    """A promoted trial reuses rung-one game cells instead of replaying them."""
    panels = derive_panels(frontier_report)
    candidate = HybridOpponent(to_runtime(HybridConfig.default()))
    league = dict.fromkeys(panels.rung_3, "opponent.py")
    prior = {
        GameKey(opponent, seed, seat): GameResult(
            GameKey(opponent, seed, seat), 1, 0, 0.0
        )
        for opponent in panels.rung_1
        for seed in RUNG_1_SEEDS
        for seat in (0, 1)
    }

    tasks = missing_game_tasks(candidate, league, panels.rung_2, RUNG_2_SEEDS, prior)

    assert len(tasks) == 168
    assert not set(prior) & {task.key for task in tasks}


RUNG = RungSpec(1, 1, ("economic_policy",), (860_000, 860_001))
WEIGHTS = StrengthWeights({"economic_policy": 3})


def _rows(*, failed: bool = False) -> dict[GameKey, GameResult]:
    """Return four hand-checked candidate-relative game rows for one matchup."""
    rows = {
        GameKey("economic_policy", 860_000, 0): GameResult(
            GameKey("economic_policy", 860_000, 0), 150, 50, 0.1
        ),
        GameKey("economic_policy", 860_000, 1): GameResult(
            GameKey("economic_policy", 860_000, 1), 100, 100, 0.2
        ),
        GameKey("economic_policy", 860_001, 0): GameResult(
            GameKey("economic_policy", 860_001, 0), 1, 3, 0.3
        ),
        GameKey("economic_policy", 860_001, 1): GameResult(
            GameKey("economic_policy", 860_001, 1), 2, 1, 0.4
        ),
    }
    if failed:
        key = GameKey("economic_policy", 860_000, 0)
        rows[key] = GameResult(key, None, None, 0.1, "boom")
    return rows


def test_rung_evidence_aggregates_counts_margin_runtime_and_scores() -> None:
    """Wrong raw game aggregation must change the persisted rung evidence."""
    evidence = build_rung_evidence(7, HybridConfig.default(), RUNG, _rows(), WEIGHTS)
    economic = evidence.matchups["economic_policy"]

    assert (economic.wins, economic.draws, economic.losses, economic.games) == (
        2,
        1,
        1,
        4,
    )
    assert economic.win_points == pytest.approx(0.625)
    assert economic.paired_normalized_margin == pytest.approx(1 / 12)
    assert economic.runtime_seconds == pytest.approx(1.0)
    assert evidence.failures == ()
    assert (evidence.primary, evidence.dense_margin) == pytest.approx((0.625, 1 / 12))
    assert evidence.objective == pytest.approx(objective_value(0.625, 1 / 12))


def test_any_failed_game_makes_rung_ineligible() -> None:
    """A recorded engine failure cannot receive an optimization value."""
    evidence = build_rung_evidence(
        7, HybridConfig.default(), RUNG, _rows(failed=True), WEIGHTS
    )

    assert evidence.failures == ("economic_policy/860000/seat0: boom",)
    assert (evidence.primary, evidence.dense_margin, evidence.objective) == (
        None,
        None,
        None,
    )
    with pytest.raises(IneligibleEvidenceError, match="boom"):
        score_evidence(evidence, WEIGHTS)


@pytest.mark.parametrize("rung", (1, 2, 3))
def test_one_primary_increment_beats_maximum_margin_difference(
    rung: int, frontier_report: FrontierReport
) -> None:
    """The dense margin epsilon may order ties but cannot outrank win points."""
    panels = derive_panels(frontier_report)
    specs = rung_specs(panels)
    all_weights = {name: index + 1 for index, name in enumerate(NAMES)}
    spec = specs[rung - 1]
    weights = StrengthWeights(all_weights)
    minimum_increment = minimum_primary_increment(spec, weights)

    assert minimum_increment > 1e-6
    assert objective_value(0.4 + minimum_increment, -1.0) > objective_value(0.4, 1.0)


@pytest.mark.parametrize("rung", (1, 2, 3))
def test_each_single_game_result_half_step_exceeds_the_conservative_bound(
    rung: int, frontier_report: FrontierReport
) -> None:
    """A one-cell loss/draw change cannot be smaller than the documented bound."""
    panels = derive_panels(frontier_report)
    spec = rung_specs(panels)[rung - 1]
    weights = StrengthWeights({name: index + 1 for index, name in enumerate(NAMES)})
    keys = tuple(
        GameKey(opponent, seed, seat)
        for opponent in spec.opponents
        for seed in spec.seeds
        for seat in (0, 1)
    )
    losses = {key: GameResult(key, 0, 1, 0.0) for key in keys}
    draws = {key: GameResult(key, 1, 1, 0.0) for key in keys}
    lower_score = score_evidence(
        build_rung_evidence(7, HybridConfig.default(), spec, losses, weights), weights
    )
    draw_score = score_evidence(
        build_rung_evidence(7, HybridConfig.default(), spec, draws, weights), weights
    )
    bound = minimum_primary_increment(spec, weights)

    observed_deltas: list[float] = []
    for opponent in spec.opponents:
        key = GameKey(opponent, spec.seeds[0], 0)
        loss_to_draw = dict(losses)
        loss_to_draw[key] = GameResult(key, 1, 1, 0.0)
        draw_to_win = dict(draws)
        draw_to_win[key] = GameResult(key, 1, 0, 0.0)
        observed_deltas.extend(
            (
                score_evidence(
                    build_rung_evidence(
                        7, HybridConfig.default(), spec, loss_to_draw, weights
                    ),
                    weights,
                ).primary
                - lower_score.primary,
                score_evidence(
                    build_rung_evidence(
                        7, HybridConfig.default(), spec, draw_to_win, weights
                    ),
                    weights,
                ).primary
                - draw_score.primary,
            )
        )

    assert all(
        delta > 0.0
        and (delta >= bound or math.isclose(delta, bound, rel_tol=1e-12, abs_tol=1e-15))
        for delta in observed_deltas
    )


def test_evidence_models_are_strict_and_atomic_writes_are_canonical(
    tmp_path: Path,
) -> None:
    """Evidence must reject coerced inputs and hash the exact durable JSON bytes."""
    with pytest.raises(ValidationError):
        GameEvidence(
            opponent="economic_policy",
            seed=True,
            seat=0,
            ours=1,
            theirs=0,
            runtime_seconds=0.0,
        )
    with pytest.raises(ValidationError, match="successful game evidence"):
        GameEvidence(
            opponent="economic_policy",
            seed=860_000,
            seat=0,
            ours=1,
            theirs=None,
            runtime_seconds=0.0,
        )

    evidence = build_rung_evidence(7, HybridConfig.default(), RUNG, _rows(), WEIGHTS)
    path, digest = write_rung_evidence_atomic(tmp_path, evidence)
    source = path.read_bytes()

    assert path == evidence_path(tmp_path, 7, 1)
    assert digest == hashlib.sha256(source).hexdigest()
    assert (
        source
        == (
            json.dumps(
                evidence.model_dump(mode="json"),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode()
    )
