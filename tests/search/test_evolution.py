"""Deterministic progressive hybrid evolution and resume contracts."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from kaggriculture.hybrid.config import HybridConfig
from kaggriculture.search import evolution
from kaggriculture.search.arena import HybridOpponent
from kaggriculture.search.evolution import (
    DEVELOPMENT_SEEDS,
    FRONTIER_SEEDS,
    PROMOTION_SEEDS,
    SCREENING_SEEDS,
    EvolutionConfig,
    SearchState,
    evolve,
)
from kaggriculture.search.fitness import StrengthWeights
from kaggriculture.search.frontier import (
    FrontierArtifact,
    FrontierReport,
    FrontierRow,
    VerifiedFrontier,
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


def _config(**changes: object) -> EvolutionConfig:
    base = EvolutionConfig(
        population=4,
        elites=1,
        generations=1,
        mutation_sigma=0.35,
        seed=73,
        workers=2,
        engine="1.32.7",
        manifest_sha256="a" * 64,
    )
    return replace(base, **changes)


def _league() -> dict[str, str]:
    return {"frontier": "frontier-agent", "boatlee_v14_current": "v14-agent"}


def _weights() -> StrengthWeights:
    return StrengthWeights({"frontier": 4, "boatlee_v14_current": 3})


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
            replace(config, manifest_sha256="b" * 64),
            "manifest sha256",
        ),
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


def test_cli_caps_workers_lowers_priority_and_serializes_exact_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The executable search protects Toad and binds verified source identity."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"exact":"manifest bytes"}\n')
    report = FrontierReport(
        engine="1.32.7",
        seeds=FRONTIER_SEEDS,
        frontier_name="frontier",
        failures=(),
        runtime_seconds=0.0,
        rows=tuple(
            FrontierRow(
                name=name,
                games=1,
                field_win_points=0.5,
                worst_matchup_win_points=0.5,
                paired_margin=0.0,
                failures=(),
                runtime_seconds=0.0,
                matchups={},
                matchup_results=(),
            )
            for name in _league()
        ),
    )
    report_path = tmp_path / "frontier.json"
    report_path.write_text(json.dumps(asdict(report)))
    frontier = VerifiedFrontier(
        engine="1.32.7",
        opponents=_league(),
        artifacts=tuple(
            FrontierArtifact(
                name=name,
                provenance="fixture",
                relative_path=Path(f"{name}.py"),
                sha256=index * 64,
            )
            for index, name in zip("ab", _league(), strict=True)
        ),
    )
    captured: dict[str, object] = {}
    lowered: list[int] = []
    monkeypatch.setattr(hybrid_search.os, "nice", lowered.append)
    monkeypatch.setattr(hybrid_search, "verify_frontier", lambda *_: frontier)
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
