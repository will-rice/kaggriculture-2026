"""Read-only W&B telemetry for durable hybrid-search generations."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kaggriculture.hybrid.config import HybridConfig
from kaggriculture.search.evolution import (
    DEVELOPMENT_SEEDS,
    SCREENING_SEEDS,
    CandidateEvaluation,
    EvolutionConfig,
    SearchState,
    initial_state,
)
from kaggriculture.search.fitness import StrengthWeights, score_fitness
from kaggriculture.search.genome import GenomeCodec
from kaggriculture.search.scripts import hybrid_wandb
from kaggriculture.search.scripts.hybrid_wandb import (
    WandbSink,
    generation_metrics,
    publish_unseen,
    wandb_config,
)


class RecordingSink:
    """External-logger seam that records the rows the sidecar publishes."""

    def __init__(self) -> None:
        self.rows: list[tuple[int, dict[str, float]]] = []

    def log(self, step: int, metrics: dict[str, float]) -> None:
        """Record one externally visible generation row."""
        self.rows.append((step, metrics))


def _candidate(
    codec: GenomeCodec,
    *,
    stage: str,
    rates: dict[str, float],
    margin: float,
) -> CandidateEvaluation:
    config = HybridConfig.default()
    seeds = SCREENING_SEEDS if stage == "screening" else DEVELOPMENT_SEEDS
    weights = StrengthWeights({"frontier": 4, "boatlee_v14": 3})
    return CandidateEvaluation(
        genome=codec.encode(config),
        config=config,
        stage=stage,
        seeds=seeds,
        matchup_rates=rates,
        matchup_games={name: 2 * len(seeds) for name in rates},
        fitness=score_fitness(rates, weights),
        paired_normalized_margin=margin,
        failures=(),
    )


def _state(*, generations: int = 2, seed: int = 73) -> SearchState:
    codec = GenomeCodec.default()
    config = EvolutionConfig(
        artifact_mode="test",
        manifest_sha256=None,
        population=1,
        elites=1,
        generations=4,
        mutation_sigma=0.35,
        seed=seed,
        workers=1,
        engine="1.32.7",
    )
    state = initial_state(
        codec,
        (HybridConfig.default(),),
        {"frontier": "frontier-agent", "boatlee_v14": "boatlee-agent"},
        StrengthWeights({"frontier": 4, "boatlee_v14": 3}),
        config,
    )
    fixtures = (
        ({"frontier": 0.25, "boatlee_v14": 0.50}, -0.50),
        ({"frontier": 0.50, "boatlee_v14": 0.75}, -0.25),
        ({"frontier": 0.75, "boatlee_v14": 1.00}, 0.125),
    )
    rng = random.Random(seed)
    for generation, (rates, margin) in enumerate(fixtures[:generations]):
        screened = _candidate(codec, stage="screening", rates=rates, margin=margin)
        developed = _candidate(codec, stage="development", rates=rates, margin=margin)
        state = state.advance(
            generation,
            (screened,),
            (developed,),
            rng.getstate(),
        )
    return state


def test_generation_metrics_expose_search_objective_and_every_matchup() -> None:
    """Dropping objective components or an opponent must break dashboard parity."""
    record = _state(generations=1).history[0]

    metrics = generation_metrics(record, completed=1, target=4)

    assert metrics == {
        "search/generation": 1.0,
        "search/progress": 0.25,
        "search/remaining_generations": 3.0,
        "screening/candidates": 1.0,
        "screening/eligible": 1.0,
        "screening/failures": 0.0,
        "development/candidates": 1.0,
        "development/eligible": 1.0,
        "development/failures": 0.0,
        "best/fitness": pytest.approx(0.325),
        "best/weighted_mean": pytest.approx(2.5 / 7.0),
        "best/worst": 0.25,
        "best/margin": -0.5,
        "best/matchup/boatlee_v14": 0.5,
        "best/matchup/frontier": 0.25,
        "elite/0/fitness": pytest.approx(0.325),
        "elite/0/weighted_mean": pytest.approx(2.5 / 7.0),
        "elite/0/worst": 0.25,
        "elite/0/margin": -0.5,
    }


def test_publish_backfills_once_then_logs_future_timing_without_duplicates(
    tmp_path: Path,
) -> None:
    """Restarting the sidecar must not duplicate durable generation steps."""
    state_path = tmp_path / "search-state.json"
    cursor_path = tmp_path / "search-wandb-cursor.json"
    state_path.write_text(_state(generations=2).to_json())
    sink = RecordingSink()

    identity = hybrid_wandb.search_identity(SearchState.load(state_path))
    assert (
        publish_unseen(
            state_path,
            cursor_path,
            sink,
            expected_identity=identity,
            now=1_000.0,
        )
        == 2
    )
    assert [step for step, _ in sink.rows] == [1, 2]
    assert (
        publish_unseen(
            state_path,
            cursor_path,
            sink,
            expected_identity=identity,
            now=1_010.0,
        )
        == 0
    )
    assert [step for step, _ in sink.rows] == [1, 2]

    state_path.write_text(_state(generations=3).to_json())
    assert (
        publish_unseen(
            state_path,
            cursor_path,
            sink,
            expected_identity=identity,
            now=1_030.0,
        )
        == 1
    )
    step, metrics = sink.rows[-1]
    assert step == 3
    assert metrics["monitor/generation_wall_seconds"] == 30.0
    assert metrics["monitor/eta_seconds"] == 30.0
    assert json.loads(cursor_path.read_text())["last_generation"] == 3
    assert not tuple(tmp_path.glob(".search-wandb-cursor.json.*.tmp"))


def test_cursor_refuses_a_different_search_identity(tmp_path: Path) -> None:
    """A reused cursor must never suppress generations from another search."""
    state_path = tmp_path / "search-state.json"
    cursor_path = tmp_path / "cursor.json"
    state_path.write_text(_state(generations=1, seed=73).to_json())
    identity = hybrid_wandb.search_identity(SearchState.load(state_path))
    publish_unseen(
        state_path,
        cursor_path,
        RecordingSink(),
        expected_identity=identity,
        now=1_000.0,
    )
    state_path.write_text(_state(generations=1, seed=74).to_json())

    with pytest.raises(ValueError, match="identity"):
        publish_unseen(
            state_path,
            cursor_path,
            RecordingSink(),
            expected_identity=identity,
            now=1_010.0,
        )


def test_wandb_config_binds_the_complete_search_identity() -> None:
    """The W&B run must identify the exact search rather than only its name."""
    state = _state(generations=1)

    config = wandb_config(state, Path("run/hybrid/search-state.json"), "abc123")

    assert config["commit"] == "abc123"
    assert config["state_path"] == "run/hybrid/search-state.json"
    assert config["engine"] == "1.32.7"
    assert config["target_generations"] == 4
    assert config["search_seed"] == 73
    assert config["manifest_sha256"] is None
    assert config["genome_schema_sha256"] == state.identity.genome_schema_sha256
    assert config["strength_weights"] == {"boatlee_v14": 3, "frontier": 4}
    assert config["league"] == ["boatlee_v14", "frontier"]


def test_publisher_does_not_advance_cursor_when_logging_fails(tmp_path: Path) -> None:
    """A W&B failure must leave the failed generation eligible for retry."""
    state_path = tmp_path / "search-state.json"
    cursor_path = tmp_path / "cursor.json"
    state_path.write_text(_state(generations=1).to_json())

    class FailingSink:
        def log(self, step: int, metrics: dict[str, float]) -> None:
            raise RuntimeError("wandb unavailable")

    with pytest.raises(RuntimeError, match="wandb unavailable"):
        publish_unseen(
            state_path,
            cursor_path,
            FailingSink(),
            expected_identity=hybrid_wandb.search_identity(
                SearchState.load(state_path)
            ),
            now=1_000.0,
        )

    assert not cursor_path.exists()


@pytest.mark.parametrize("alias", ["same", "hardlink"])
def test_cursor_cannot_alias_or_overwrite_search_state(
    tmp_path: Path, alias: str
) -> None:
    """A cursor path mistake must leave the authoritative state byte-exact."""
    state_path = tmp_path / "search-state.json"
    source = _state(generations=1).to_json()
    state_path.write_text(source)
    cursor_path = state_path
    if alias == "hardlink":
        cursor_path = tmp_path / "cursor.json"
        cursor_path.hardlink_to(state_path)

    with pytest.raises(ValueError, match="cursor.*state"):
        publish_unseen(
            state_path,
            cursor_path,
            RecordingSink(),
            expected_identity=hybrid_wandb.search_identity(
                SearchState.load(state_path)
            ),
            now=1_000.0,
        )

    assert state_path.read_text() == source


def test_wandb_sink_commits_and_finishes_every_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cursor acknowledgment must follow a committed and flushed W&B row."""

    class Run:
        def __init__(self) -> None:
            self.events: list[object] = []

        def define_metric(self, name: str, **kwargs: str) -> None:
            self.events.append(("define", name, kwargs))

        def log(self, metrics: dict[str, float], *, step: int, commit: bool) -> None:
            self.events.append(("log", metrics, step, commit))

        def finish(self) -> None:
            self.events.append("finish")

    run = Run()
    init_calls: list[dict[str, object]] = []

    def init(**kwargs: object) -> Run:
        init_calls.append(kwargs)
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    state = _state(generations=1)
    sink = WandbSink(
        state,
        Path("run/hybrid/search-state.json"),
        "abc123",
        entity="will-rice",
        project="kaggriculture-2026",
        name=None,
    )

    sink.log(6, {"search/generation": 6.0})

    assert len(init_calls) == 1
    assert run.events[-2:] == [
        ("log", {"search/generation": 6.0}, 6, True),
        "finish",
    ]


def test_wandb_sink_finishes_when_metric_definition_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An opened W&B run must close even when dashboard setup raises."""

    class Run:
        finished = False

        def define_metric(self, name: str, **kwargs: str) -> None:
            raise RuntimeError("metric definition failed")

        def finish(self) -> None:
            self.finished = True

    run = Run()
    monkeypatch.setitem(
        sys.modules, "wandb", SimpleNamespace(init=lambda **_kwargs: run)
    )
    sink = WandbSink(
        _state(generations=1),
        Path("run/hybrid/search-state.json"),
        "abc123",
        entity="will-rice",
        project="kaggriculture-2026",
        name=None,
    )

    with pytest.raises(RuntimeError, match="metric definition failed"):
        sink.log(1, {"search/generation": 1.0})

    assert run.finished is True


def test_expected_identity_rejects_state_swap_before_first_generation(
    tmp_path: Path,
) -> None:
    """Generation-zero startup must stay bound to its originally opened search."""
    state_path = tmp_path / "search-state.json"
    cursor_path = tmp_path / "cursor.json"
    first = _state(generations=0, seed=73)
    second = _state(generations=0, seed=74)
    state_path.write_text(second.to_json())

    with pytest.raises(ValueError, match="opened search identity"):
        publish_unseen(
            state_path,
            cursor_path,
            RecordingSink(),
            expected_identity=hybrid_wandb.search_identity(first),
            now=1_000.0,
        )

    assert not cursor_path.exists()


def test_partial_backfill_resumes_after_last_cursor_commit(tmp_path: Path) -> None:
    """A failed second row must retry only that row, not the committed first."""
    state_path = tmp_path / "search-state.json"
    cursor_path = tmp_path / "cursor.json"
    state_path.write_text(_state(generations=2).to_json())
    identity = hybrid_wandb.search_identity(SearchState.load(state_path))

    class FailSecond(RecordingSink):
        def log(self, step: int, metrics: dict[str, float]) -> None:
            if step == 2:
                raise RuntimeError("second row failed")
            super().log(step, metrics)

    with pytest.raises(RuntimeError, match="second row failed"):
        publish_unseen(
            state_path,
            cursor_path,
            FailSecond(),
            expected_identity=identity,
            now=1_000.0,
        )

    assert json.loads(cursor_path.read_text())["last_generation"] == 1
    retry = RecordingSink()
    assert (
        publish_unseen(
            state_path,
            cursor_path,
            retry,
            expected_identity=identity,
            now=1_010.0,
        )
        == 1
    )
    assert [step for step, _metrics in retry.rows] == [2]


def test_clock_regression_omits_future_timing_metrics(tmp_path: Path) -> None:
    """A corrected system clock must not emit negative duration or ETA."""
    state_path = tmp_path / "search-state.json"
    cursor_path = tmp_path / "cursor.json"
    state_path.write_text(_state(generations=1).to_json())
    identity = hybrid_wandb.search_identity(SearchState.load(state_path))
    publish_unseen(
        state_path,
        cursor_path,
        RecordingSink(),
        expected_identity=identity,
        now=1_000.0,
    )
    state_path.write_text(_state(generations=2).to_json())
    sink = RecordingSink()

    publish_unseen(
        state_path,
        cursor_path,
        sink,
        expected_identity=identity,
        now=999.0,
    )

    assert "monitor/generation_wall_seconds" not in sink.rows[0][1]
    assert "monitor/eta_seconds" not in sink.rows[0][1]


def test_eta_overflow_is_omitted_instead_of_logging_infinity(tmp_path: Path) -> None:
    """A finite elapsed interval must not create a non-finite derived ETA."""
    state_path = tmp_path / "search-state.json"
    cursor_path = tmp_path / "cursor.json"
    state_path.write_text(_state(generations=1).to_json())
    identity = hybrid_wandb.search_identity(SearchState.load(state_path))
    publish_unseen(
        state_path,
        cursor_path,
        RecordingSink(),
        expected_identity=identity,
        now=0.0,
    )
    state_path.write_text(_state(generations=2).to_json())
    sink = RecordingSink()

    publish_unseen(
        state_path,
        cursor_path,
        sink,
        expected_identity=identity,
        now=1e308,
    )

    assert sink.rows[0][1]["monitor/generation_wall_seconds"] == 1e308
    assert "monitor/eta_seconds" not in sink.rows[0][1]
