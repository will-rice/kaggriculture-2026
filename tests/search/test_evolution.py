"""Deterministic progressive hybrid evolution and resume contracts."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import pytest

from kaggriculture.hybrid.config import HybridConfig
from kaggriculture.search import evolution
from kaggriculture.search import frontier as frontier_module
from kaggriculture.search.arena import HybridOpponent
from kaggriculture.search.evolution import (
    DEVELOPMENT_SEEDS,
    FRONTIER_SEEDS,
    PROMOTION_SEEDS,
    SCREENING_SEEDS,
    EvolutionConfig,
    SearchState,
    evolve,
    top_eligible,
)
from kaggriculture.search.fitness import StrengthWeights
from kaggriculture.search.frontier import (
    FrontierArtifact,
    FrontierManifest,
    rank_frontier,
    verify_frontier,
)
from kaggriculture.search.genome import GenomeCodec
from kaggriculture.search.scripts import hybrid_search


class MeasuredScores(list[float]):
    """Arena-like deterministic outcomes with auditable tie-break telemetry."""

    def __init__(self, values: list[float], margin: float) -> None:
        super().__init__(values)
        self.normalized_margins = (margin,) * len(values)
        self.failures: tuple[str, ...] = ()
        self.runtime_seconds = 0.125


class UntimedScores(list[float]):
    """Arena results that deliberately exercise the real timing fallback."""

    def __init__(self, values: list[float], margin: float) -> None:
        super().__init__(values)
        self.normalized_margins = (margin,) * len(values)
        self.failures: tuple[str, ...] = ()


def _deterministic_arena(
    codec: GenomeCodec,
    calls: list[tuple[str, tuple[str, ...], tuple[int, ...], int | None]],
) -> Callable[
    [HybridOpponent, dict[str, str], tuple[int, ...], int | None], MeasuredScores
]:
    def outcomes(
        candidate: HybridOpponent,
        league: dict[str, str],
        seeds: tuple[int, ...],
        workers: int | None,
    ) -> MeasuredScores:
        assert isinstance(candidate, HybridOpponent)
        payload = json.dumps(
            candidate.runtime.to_payload(), sort_keys=True, separators=(",", ":")
        )
        digest = hashlib.sha256(payload.encode()).hexdigest()
        calls.append((digest, tuple(league), tuple(seeds), workers))
        name = next(iter(league))
        signal = int(digest[:12], 16) + sum(seeds) + sum(name.encode())
        point = (0.0, 0.5, 1.0)[signal % 3]
        # ``codec`` is deliberately captured and exercised: a runtime produced
        # by evolution must round-trip to a genome under the supplied schema.
        codec.encode(HybridConfig.from_runtime(candidate.runtime))
        return MeasuredScores([point] * (2 * len(seeds)), (signal % 101) / 100.0)

    return outcomes


def _untimed_deterministic_arena(
    codec: GenomeCodec,
) -> Callable[
    [HybridOpponent, dict[str, str], tuple[int, ...], int | None], UntimedScores
]:
    measured = _deterministic_arena(codec, [])

    def outcomes(
        candidate: HybridOpponent,
        league: dict[str, str],
        seeds: tuple[int, ...],
        workers: int | None,
    ) -> UntimedScores:
        result = measured(candidate, league, seeds, workers)
        return UntimedScores(list(result), result.normalized_margins[0])

    return outcomes


def _config(**changes: object) -> EvolutionConfig:
    base = EvolutionConfig(
        artifact_mode="test",
        manifest_sha256=None,
        population=4,
        elites=1,
        generations=1,
        mutation_sigma=0.35,
        seed=73,
        workers=2,
        engine="1.32.7",
    )
    return replace(base, **changes)


def _league() -> dict[str, str]:
    return {"frontier": "frontier-agent", "boatlee_v14_current": "v14-agent"}


def _weights() -> StrengthWeights:
    return StrengthWeights({"frontier": 4, "boatlee_v14_current": 3})


def test_evolution_config_requires_an_explicit_artifact_certification_mode() -> None:
    """A default constructor must not silently claim an all-zero manifest identity."""
    with pytest.raises((TypeError, ValueError), match="artifact|certif"):
        cast(Callable[[], EvolutionConfig], EvolutionConfig)()


def test_noncertifying_search_state_cannot_write_finalists(tmp_path: Path) -> None:
    """Unit-test state may persist only when it cannot masquerade as a finalist."""
    state = evolution.initial_state(
        GenomeCodec.default(),
        (HybridConfig.default(),),
        _league(),
        _weights(),
        _config(generations=0),
    )

    with pytest.raises(ValueError, match="non-certifying"):
        evolution.write_finalists(state, tmp_path / "finalists.json")

    assert not (tmp_path / "finalists.json").exists()


def test_search_seed_sets_are_fixed_and_pairwise_disjoint() -> None:
    """Search may never leak frontier or untouched holdout seeds across splits."""
    sets = tuple(
        map(set, (FRONTIER_SEEDS, SCREENING_SEEDS, DEVELOPMENT_SEEDS, PROMOTION_SEEDS))
    )

    assert len(SCREENING_SEEDS) == 8
    assert len(DEVELOPMENT_SEEDS) == 32
    assert len(PROMOTION_SEEDS) == 128
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(sets)
        for right in sets[index + 1 :]
    )


def test_progressive_evaluation_visits_every_member_for_every_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """All four candidates screen on 8 seeds; only two survivors develop on 32."""
    codec = GenomeCodec.default()
    calls: list[tuple[str, tuple[str, ...], tuple[int, ...], int | None]] = []
    monkeypatch.setattr(evolution.arena, "outcomes", _deterministic_arena(codec, calls))

    state = evolve(
        codec=codec,
        initial_configs=(HybridConfig.default(),),
        league=_league(),
        weights=_weights(),
        config=_config(),
        output=tmp_path / "state.json",
    )

    screening = [call for call in calls if call[2] == SCREENING_SEEDS]
    development = [call for call in calls if call[2] == DEVELOPMENT_SEEDS]
    assert len(screening) == 4 * len(_league())
    assert len(development) == 2 * len(_league())
    assert all(len(seeds) == 8 for _, _, seeds, _ in screening)
    assert all(len(seeds) == 32 for _, _, seeds, _ in development)
    assert all(len(names) == 1 for _, names, _, _ in calls)
    assert all(workers == 2 for *_, workers in calls)
    assert state.generation == 1
    assert state.history[0].screened[0].matchup_rates.keys() == _league().keys()


def test_failed_matchup_is_ineligible_but_remaining_league_is_still_played(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One exception hard-fails fitness without hiding the other matchup record."""
    codec = GenomeCodec.default()
    calls: list[str] = []

    def partly_failed(
        candidate: HybridOpponent,
        league: dict[str, str],
        seeds: tuple[int, ...],
        workers: int | None,
    ) -> MeasuredScores:
        name = next(iter(league))
        calls.append(name)
        if name == "frontier":
            raise RuntimeError("illegal action")
        return MeasuredScores([1.0] * (2 * len(seeds)), 1.0)

    monkeypatch.setattr(evolution.arena, "outcomes", partly_failed)

    (evaluated,) = evolution.evaluate_population(
        (codec.encode(HybridConfig.default()),),
        codec,
        SCREENING_SEEDS,
        _league(),
        _weights(),
        workers=1,
        stage="screening",
    )

    assert calls == ["frontier", "boatlee_v14_current"]
    assert evaluated.matchup_rates == {
        "frontier": 0.0,
        "boatlee_v14_current": 1.0,
    }
    assert evaluated.fitness is not None
    assert evaluated.fitness.eligible is False
    assert evaluated.failures == ("frontier: RuntimeError: illegal action",)


def test_uninterrupted_and_interrupted_resume_are_byte_for_byte_identical(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Restoring generation two must restore RNG state before generation three."""
    codec = GenomeCodec.default()
    monkeypatch.setattr(evolution.arena, "outcomes", _deterministic_arena(codec, []))
    config = _config(generations=4)
    full_path = tmp_path / "full.json"
    resumed_path = tmp_path / "resumed.json"

    full = evolve(
        codec=codec,
        initial_configs=(HybridConfig.default(),),
        league=_league(),
        weights=_weights(),
        config=config,
        output=full_path,
    )

    real_save = evolution.save_state_atomic

    def interrupt_after_second_save(state: SearchState, output: Path) -> None:
        real_save(state, output)
        if state.generation == 2:
            raise InterruptedError("test-controlled interruption")

    monkeypatch.setattr(evolution, "save_state_atomic", interrupt_after_second_save)
    with pytest.raises(InterruptedError, match="test-controlled"):
        evolve(
            codec=codec,
            initial_configs=(HybridConfig.default(),),
            league=_league(),
            weights=_weights(),
            config=config,
            output=resumed_path,
        )
    interrupted = SearchState.load(resumed_path)
    assert interrupted.generation == 2

    monkeypatch.setattr(evolution, "save_state_atomic", real_save)
    resumed = evolve(
        codec=codec,
        initial_configs=(HybridConfig.default(),),
        league=_league(),
        weights=_weights(),
        config=config,
        output=resumed_path,
        resume=interrupted,
    )

    assert resumed == full
    assert resumed_path.read_bytes() == full_path.read_bytes()


def test_wall_clock_fallback_cannot_change_canonical_search_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two identical searches differ only in timing and must serialize identically."""
    codec = GenomeCodec.default()
    monkeypatch.setattr(
        evolution.arena, "outcomes", _untimed_deterministic_arena(codec)
    )
    config = _config(generations=2)

    evolve(
        codec=codec,
        initial_configs=(HybridConfig.default(),),
        league=_league(),
        weights=_weights(),
        config=config,
        output=tmp_path / "first.json",
    )

    evolve(
        codec=codec,
        initial_configs=(HybridConfig.default(),),
        league=_league(),
        weights=_weights(),
        config=config,
        output=tmp_path / "second.json",
    )

    assert (tmp_path / "first.json").read_bytes() == (
        tmp_path / "second.json"
    ).read_bytes()


def test_timing_changes_across_interruption_cannot_change_resumed_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A resumed run remains identical when each process has a different clock."""
    codec = GenomeCodec.default()
    monkeypatch.setattr(
        evolution.arena, "outcomes", _untimed_deterministic_arena(codec)
    )
    config = _config(generations=3)
    full_path = tmp_path / "full-untimed.json"
    resumed_path = tmp_path / "resumed-untimed.json"

    evolve(
        codec=codec,
        initial_configs=(HybridConfig.default(),),
        league=_league(),
        weights=_weights(),
        config=config,
        output=full_path,
    )

    real_save = evolution.save_state_atomic

    def interrupt_after_first_save(state: SearchState, output: Path) -> None:
        real_save(state, output)
        if state.generation == 1:
            raise InterruptedError("timing boundary")

    monkeypatch.setattr(evolution, "save_state_atomic", interrupt_after_first_save)
    with pytest.raises(InterruptedError, match="timing boundary"):
        evolve(
            codec=codec,
            initial_configs=(HybridConfig.default(),),
            league=_league(),
            weights=_weights(),
            config=config,
            output=resumed_path,
        )

    monkeypatch.setattr(evolution, "save_state_atomic", real_save)
    evolve(
        codec=codec,
        initial_configs=(HybridConfig.default(),),
        league=_league(),
        weights=_weights(),
        config=config,
        output=resumed_path,
        resume=SearchState.load(resumed_path),
    )

    assert resumed_path.read_bytes() == full_path.read_bytes()


def test_resume_rejects_a_genome_that_no_longer_decodes_to_its_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A valid outer JSON shape cannot conceal corrupted candidate identity."""
    codec = GenomeCodec.default()
    monkeypatch.setattr(evolution.arena, "outcomes", _deterministic_arena(codec, []))
    config = _config()
    state = evolve(
        codec=codec,
        initial_configs=(HybridConfig.default(),),
        league=_league(),
        weights=_weights(),
        config=config,
        output=tmp_path / "state.json",
    )
    corrupted_elite = replace(state.elites[0], genome=state.elites[0].genome[:-1])
    corrupted = replace(state, elites=(corrupted_elite,))

    with pytest.raises(ValueError, match="resume candidate genome/config"):
        evolve(
            codec=codec,
            initial_configs=(HybridConfig.default(),),
            league=_league(),
            weights=_weights(),
            config=config,
            output=tmp_path / "corrupted.json",
            resume=corrupted,
        )


def _state_for_history_forgery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[GenomeCodec, EvolutionConfig, SearchState]:
    codec = GenomeCodec.default()
    monkeypatch.setattr(evolution.arena, "outcomes", _deterministic_arena(codec, []))
    config = _config(generations=2)
    state = evolve(
        codec=codec,
        initial_configs=(HybridConfig.default(),),
        league=_league(),
        weights=_weights(),
        config=config,
        output=tmp_path / "valid-state.json",
    )
    return codec, config, state


def _assert_forged_resume_is_atomic(
    *,
    codec: GenomeCodec,
    config: EvolutionConfig,
    state: SearchState,
    output: Path,
    message: str,
) -> None:
    output.write_bytes(b"previous-valid-state\n")
    with pytest.raises(ValueError, match=message):
        evolve(
            codec=codec,
            initial_configs=(HybridConfig.default(),),
            league=_league(),
            weights=_weights(),
            config=config,
            output=output,
            resume=state,
        )
    assert output.read_bytes() == b"previous-valid-state\n"


def test_resume_rejects_screening_results_forged_as_development(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An 8-seed result cannot become a 32-seed survivor by local consistency."""
    codec, config, state = _state_for_history_forgery(monkeypatch, tmp_path)
    record = state.history[-1]
    developed = record.screened[: len(record.developed)]
    forged_record = replace(record, developed=developed)
    forged = replace(
        state,
        history=(*state.history[:-1], forged_record),
        elites=top_eligible(developed, count=config.elites),
    )

    _assert_forged_resume_is_atomic(
        codec=codec,
        config=config,
        state=forged,
        output=tmp_path / "stage-forgery.json",
        message="development stage|development seeds|game count",
    )


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ({"seeds": tuple(range(32))}, "development seeds"),
        ({"matchup_games": {"frontier": 2, "boatlee_v14_current": 64}}, "game count"),
    ),
)
def test_resume_rejects_forged_development_seed_and_game_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    change: dict[str, object],
    message: str,
) -> None:
    """A locally plausible result must retain its exact evaluation schedule."""
    codec, config, state = _state_for_history_forgery(monkeypatch, tmp_path)
    record = state.history[-1]
    changed = replace(record.developed[0], **change)
    developed = (changed, *record.developed[1:])
    forged_record = replace(record, developed=developed)
    forged = replace(
        state,
        history=(*state.history[:-1], forged_record),
        elites=top_eligible(developed, count=config.elites),
    )

    _assert_forged_resume_is_atomic(
        codec=codec,
        config=config,
        state=forged,
        output=tmp_path / f"{message.replace(' ', '-')}-forgery.json",
        message=message,
    )


def test_resume_rejects_a_forged_development_survivor_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Development order must be the exact top-eligible screening selection."""
    codec, config, state = _state_for_history_forgery(monkeypatch, tmp_path)
    record = state.history[-1]
    developed = tuple(reversed(record.developed))
    forged_record = replace(record, developed=developed)
    forged = replace(
        state,
        history=(*state.history[:-1], forged_record),
        elites=top_eligible(developed, count=config.elites),
    )

    _assert_forged_resume_is_atomic(
        codec=codec,
        config=config,
        state=forged,
        output=tmp_path / "survivor-forgery.json",
        message="development survivor",
    )


def test_resume_rejects_a_forged_spawned_population(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every screened genome and its order must replay from the saved RNG stream."""
    codec, config, state = _state_for_history_forgery(monkeypatch, tmp_path)
    first = state.history[0]
    screened = tuple(reversed(first.screened))
    forged_first = replace(first, screened=screened)
    forged = replace(state, history=(forged_first, *state.history[1:]))

    _assert_forged_resume_is_atomic(
        codec=codec,
        config=config,
        state=forged,
        output=tmp_path / "spawn-forgery.json",
        message="spawned population",
    )


def test_resume_rejects_a_forged_rng_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The terminal RNG state must be exactly the history-replayed stream."""
    codec, config, state = _state_for_history_forgery(monkeypatch, tmp_path)
    forged = replace(state, rng_state=random.Random(999).getstate())

    _assert_forged_resume_is_atomic(
        codec=codec,
        config=config,
        state=forged,
        output=tmp_path / "rng-forgery.json",
        message="RNG state",
    )


def test_resume_rejects_every_changed_search_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Manifest, engine, seeds, schema, league, weights, and config are immutable."""
    codec = GenomeCodec.default()
    monkeypatch.setattr(evolution.arena, "outcomes", _deterministic_arena(codec, []))
    config = _config()
    state = evolve(
        codec=codec,
        initial_configs=(HybridConfig.default(),),
        league=_league(),
        weights=_weights(),
        config=config,
        output=tmp_path / "state.json",
    )

    cases = (
        (codec, _league(), _weights(), replace(config, workers=1), "evolution config"),
        (
            codec,
            _league(),
            _weights(),
            replace(config, engine="1.32.8"),
            "engine",
        ),
        (
            codec,
            {"frontier": "changed", "boatlee_v14_current": "v14-agent"},
            _weights(),
            config,
            "league identity",
        ),
        (
            codec,
            _league(),
            StrengthWeights({"frontier": 3, "boatlee_v14_current": 3}),
            config,
            "strength weights",
        ),
        (
            replace(codec, template={**codec.template, "revision": 2}),
            _league(),
            _weights(),
            config,
            "genome schema",
        ),
    )
    for changed_codec, league, weights, changed_config, message in cases:
        with pytest.raises(ValueError, match=message):
            evolve(
                codec=changed_codec,
                initial_configs=(HybridConfig.default(),),
                league=league,
                weights=weights,
                config=changed_config,
                output=tmp_path / "rejected.json",
                resume=state,
            )

    monkeypatch.setattr(evolution, "SCREENING_SEEDS", tuple(range(821_000, 821_008)))
    with pytest.raises(ValueError, match="screening seeds"):
        evolve(
            codec=codec,
            initial_configs=(HybridConfig.default(),),
            league=_league(),
            weights=_weights(),
            config=config,
            output=tmp_path / "rejected-seeds.json",
            resume=state,
        )


def test_atomic_save_leaves_the_previous_state_when_replacement_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A crash before replace cannot corrupt the last complete checkpoint."""
    output = tmp_path / "state.json"
    output.write_bytes(b"previous-complete-state\n")
    state = evolution.initial_state(
        GenomeCodec.default(),
        (HybridConfig.default(),),
        _league(),
        _weights(),
        _config(),
    )

    def fail_replace(source: Path, destination: str | Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated"):
        evolution.save_state_atomic(state, output)

    assert output.read_bytes() == b"previous-complete-state\n"
    assert not tuple(tmp_path.glob(".state.json.*.tmp"))


def test_search_modules_import_without_torch_or_cuda() -> None:
    """CPU search must not initialize Torch, much less a CUDA context."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json,sys; "
                "import kaggriculture.search.evolution; "
                "import kaggriculture.search.scripts.hybrid_search; "
                "print(json.dumps(sorted(sys.modules)))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    imported = set(json.loads(result.stdout))

    assert "torch" not in imported
    assert not any(name.startswith("torch.cuda") for name in imported)


def _frontier_cli_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path, Path, dict[str, Any]]:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(parents=True)
    artifacts: list[FrontierArtifact] = []
    for name in _league():
        source = artifact_root / f"{name}.py"
        source.write_text(f"# exact {name} test source\n")
        artifacts.append(
            FrontierArtifact(
                name=name,
                provenance="complete CLI fixture",
                relative_path=Path(source.name),
                sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            )
        )
    manifest = FrontierManifest(engine="1.32.7", artifacts=tuple(artifacts))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json())
    verified = verify_frontier(manifest_path, artifact_root)

    monkeypatch.setattr(
        frontier_module.arena,
        "outcomes",
        lambda candidate, league, seeds, workers: MeasuredScores(
            [1.0, 0.5] * len(seeds), 0.25
        ),
    )
    report = rank_frontier(verified, FRONTIER_SEEDS, workers=1)
    payload = asdict(report)
    payload.setdefault(
        "manifest_sha256", hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )
    payload.setdefault(
        "source_sha256", {artifact.name: artifact.sha256 for artifact in artifacts}
    )
    report_path = tmp_path / "frontier.json"
    report_path.write_text(json.dumps(payload))
    return manifest_path, artifact_root, report_path, payload


def _run_hybrid_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    manifest: Path,
    artifact_root: Path,
    report: Path,
) -> dict[str, object]:
    captured: dict[str, object] = {}
    monkeypatch.setattr(hybrid_search.os, "nice", lambda value: None)
    monkeypatch.setattr(
        hybrid_search, "evolve", lambda **kwargs: captured.update(kwargs)
    )
    monkeypatch.setattr(hybrid_search, "write_finalists", lambda *args: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "hybrid_search",
            "--manifest",
            str(manifest),
            "--artifact-root",
            str(artifact_root),
            "--frontier-report",
            str(report),
            "--output",
            str(tmp_path / "state.json"),
            "--finalists",
            str(tmp_path / "finalists.json"),
        ],
    )
    hybrid_search.main()
    return captured


def test_verified_certification_can_write_a_certified_finalist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The guarded production path remains usable after source re-verification."""
    manifest, artifact_root, _, _ = _frontier_cli_fixture(monkeypatch, tmp_path)
    frontier = verify_frontier(manifest, artifact_root)
    config = EvolutionConfig(
        artifact_mode="certified",
        manifest_sha256=frontier.manifest_sha256,
        population=1,
        elites=1,
        generations=0,
        workers=1,
        engine=frontier.engine,
    )
    state = evolution.initial_state(
        GenomeCodec.default(),
        (HybridConfig.default(),),
        frontier.opponents,
        _weights(),
        config,
        evolution.certify_frontier(frontier),
    )
    output = tmp_path / "certified-finalists.json"

    evolution.write_finalists(state, output)

    assert json.loads(output.read_text())["identity"]["certified"] is True


def test_cli_accepts_a_complete_report_bound_to_verified_sources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A real manifest, verified source bytes, and complete matrix may seed search."""
    manifest, artifact_root, report, _ = _frontier_cli_fixture(monkeypatch, tmp_path)

    captured = _run_hybrid_cli(monkeypatch, tmp_path, manifest, artifact_root, report)

    assert cast(Mapping[str, object], captured["league"]).keys() == _league().keys()


def test_cli_rejects_a_report_bound_to_another_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Matching names and seeds cannot hide a stale manifest provenance hash."""
    manifest, artifact_root, report, payload = _frontier_cli_fixture(
        monkeypatch, tmp_path
    )
    payload["manifest_sha256"] = "b" * 64
    report.write_text(json.dumps(payload))

    with pytest.raises(SystemExit, match="manifest sha256"):
        _run_hybrid_cli(monkeypatch, tmp_path, manifest, artifact_root, report)


def test_cli_rejects_a_report_bound_to_other_source_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stale exact-agent digest cannot derive current-league strength weights."""
    manifest, artifact_root, report, payload = _frontier_cli_fixture(
        monkeypatch, tmp_path
    )
    sources = dict(payload["source_sha256"])
    sources["frontier"] = "c" * 64
    payload["source_sha256"] = sources
    report.write_text(json.dumps(payload))

    with pytest.raises(SystemExit, match="source sha256"):
        _run_hybrid_cli(monkeypatch, tmp_path, manifest, artifact_root, report)


def test_cli_rejects_tampered_rank_order_and_incomplete_pair_coverage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Weights require the canonical rank order and one complete row per pair."""
    manifest, artifact_root, report, payload = _frontier_cli_fixture(
        monkeypatch, tmp_path
    )
    payload["rows"] = list(reversed(payload["rows"]))
    report.write_text(json.dumps(payload))
    with pytest.raises(SystemExit, match="rank order"):
        _run_hybrid_cli(monkeypatch, tmp_path, manifest, artifact_root, report)

    _, _, report, payload = _frontier_cli_fixture(monkeypatch, tmp_path / "second")
    payload["rows"][0]["matchup_results"] = []
    report.write_text(json.dumps(payload))
    with pytest.raises(SystemExit, match="all-pairs matchup coverage"):
        _run_hybrid_cli(
            monkeypatch,
            tmp_path,
            tmp_path / "second" / "manifest.json",
            tmp_path / "second" / "artifacts",
            report,
        )


def test_cli_rejects_a_frontier_matchup_with_the_wrong_game_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every pair must contain both seats for every fixed frontier seed."""
    manifest, artifact_root, report, payload = _frontier_cli_fixture(
        monkeypatch, tmp_path
    )
    payload["rows"][0]["matchup_results"][0]["games"] = 2
    report.write_text(json.dumps(payload))

    with pytest.raises(SystemExit, match="128 games"):
        _run_hybrid_cli(monkeypatch, tmp_path, manifest, artifact_root, report)


def test_cli_caps_workers_lowers_priority_and_serializes_exact_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The executable search protects Toad and binds verified source identity."""
    manifest, artifact_root, report_path, _ = _frontier_cli_fixture(
        monkeypatch, tmp_path
    )
    captured: dict[str, object] = {}
    lowered: list[int] = []
    monkeypatch.setattr(hybrid_search.os, "nice", lowered.append)
    monkeypatch.setattr(
        hybrid_search, "evolve", lambda **kwargs: captured.update(kwargs)
    )
    monkeypatch.setattr(hybrid_search, "write_finalists", lambda *args: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "hybrid_search",
            "--manifest",
            str(manifest),
            "--artifact-root",
            str(artifact_root),
            "--frontier-report",
            str(report_path),
            "--workers",
            "4",
            "--output",
            str(tmp_path / "state.json"),
            "--finalists",
            str(tmp_path / "finalists.json"),
        ],
    )

    hybrid_search.main()

    assert lowered == [10]
    config = captured["config"]
    assert isinstance(config, EvolutionConfig)
    assert config.workers == 4
    assert config.engine == "1.32.7"
    assert config.manifest_sha256 == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert config.artifact_mode == "certified"
    assert captured["certification"] is not None

    monkeypatch.setattr(
        "sys.argv",
        [
            "hybrid_search",
            "--workers",
            "17",
            "--frontier-report",
            str(report_path),
            "--output",
            str(tmp_path / "unused-state.json"),
            "--finalists",
            str(tmp_path / "unused-finalists.json"),
        ],
    )
    with pytest.raises(SystemExit, match="between 1 and 16"):
        hybrid_search.main()
