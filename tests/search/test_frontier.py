"""Integrity and field-ranking checks for the published opponent frontier."""

from pathlib import Path

import pytest

from kaggriculture.search import frontier
from kaggriculture.search.frontier import (
    FrontierArtifact,
    FrontierIntegrityError,
    FrontierManifest,
    VerifiedFrontier,
    rank_frontier,
    verify_frontier,
)
from kaggriculture.search.scripts import frontier_round_robin


@pytest.fixture
def verified_frontier(tmp_path: Path) -> VerifiedFrontier:
    """A two-policy field whose paths need not exist for mocked ranking."""
    return VerifiedFrontier(
        engine="1.32.7",
        opponents={"a": str(tmp_path / "a.py"), "b": str(tmp_path / "b.py")},
        artifacts=(
            FrontierArtifact(
                name="a",
                provenance="test fixture",
                relative_path=Path("a.py"),
                sha256="a" * 64,
            ),
            FrontierArtifact(
                name="b",
                provenance="test fixture",
                relative_path=Path("b.py"),
                sha256="b" * 64,
            ),
        ),
    )


def test_verify_frontier_rejects_changed_source_bytes(tmp_path: Path) -> None:
    """A manifest must reject a source changed after it was archived."""
    source = tmp_path / "kaito-v27.py"
    source.write_text("def agent(obs, config=None): return {}\n")
    manifest = FrontierManifest(
        engine="1.32.7",
        artifacts=(
            FrontierArtifact(
                name="kaito_v27",
                source_url="https://www.kaggle.com/code/kaitofukami/25-27-strict-future-v27-midgame-meta-reset",
                notebook_version=4,
                historical_score=3090.1,
                provenance="Kaggle notebook version 4, extracted last callable",
                relative_path=Path("kaito-v27.py"),
                sha256="0" * 64,
                archive_sha256="1" * 64,
            ),
        ),
    )
    path = tmp_path / "manifest.json"
    path.write_text(manifest.model_dump_json())

    with pytest.raises(FrontierIntegrityError, match="kaito_v27.*sha256"):
        verify_frontier(path, tmp_path)


def test_verify_frontier_rejects_a_path_outside_the_artifact_root(
    tmp_path: Path,
) -> None:
    """A malformed manifest must never load a policy outside its archive root."""
    manifest = FrontierManifest(
        engine="1.32.7",
        artifacts=(
            FrontierArtifact(
                name="outside",
                provenance="test fixture",
                relative_path=Path("../outside.py"),
                sha256="0" * 64,
            ),
        ),
    )
    path = tmp_path / "manifest.json"
    path.write_text(manifest.model_dump_json())

    with pytest.raises(FrontierIntegrityError, match="outside.*escapes"):
        verify_frontier(path, tmp_path)


def test_rank_frontier_uses_seat_swapped_field_win_points(
    monkeypatch: pytest.MonkeyPatch, verified_frontier: VerifiedFrontier
) -> None:
    """Field rank uses both seats, not only a candidate's favourable quadrant."""
    monkeypatch.setattr(
        frontier.arena,
        "outcomes",
        lambda candidate, league, seeds, workers: {
            "a.py": [1.0, 1.0, 0.5, 0.5],
            "b.py": [0.0, 0.0, 0.5, 0.5],
        }[Path(str(candidate)).name],
    )

    report = rank_frontier(verified_frontier, seeds=(11, 12), workers=1)

    assert report.rows[0].name == "a"
    assert report.frontier_name == "a"
    assert report.rows[0].games == 4
    assert report.rows[0].field_win_points == pytest.approx(0.75)
    assert report.rows[0].matchups == {"b": 0.75}


def test_rank_frontier_breaks_equal_field_and_worst_scores_by_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Stable ties ensure independently generated reports have the same leader."""
    verified = VerifiedFrontier(
        engine="1.32.7",
        opponents={name: str(tmp_path / f"{name}.py") for name in ("a", "b", "c")},
        artifacts=(),
    )
    scores = {
        "a.py": [1.0, 0.0],
        "b.py": [1.0, 0.0],
        "c.py": [1.0, 0.0],
    }
    monkeypatch.setattr(
        frontier.arena,
        "outcomes",
        lambda candidate, league, seeds, workers: scores[Path(str(candidate)).name],
    )

    report = rank_frontier(verified, seeds=(7,), workers=1)

    assert [row.name for row in report.rows] == ["a", "b", "c"]


def test_rank_frontier_plays_each_unordered_pair_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A 64-seed field has 128 games per pair, not a duplicate mirror block."""
    verified = VerifiedFrontier(
        engine="1.32.7",
        opponents={name: str(tmp_path / f"{name}.py") for name in ("a", "b", "c")},
        artifacts=(),
    )
    calls: list[tuple[str, tuple[str, ...]]] = []

    def outcomes_once(
        candidate: str, league: dict[str, str], seeds: tuple[int, ...], workers: int
    ) -> list[float]:
        calls.append((Path(candidate).stem, tuple(league)))
        return [1.0, 0.0] * len(league)

    monkeypatch.setattr(frontier.arena, "outcomes", outcomes_once)

    report = rank_frontier(verified, seeds=(101,), workers=1)

    assert calls == [("a", ("b",)), ("a", ("c",)), ("b", ("c",))]
    assert report.rows[0].games == 4
    assert {row.name: row.matchups for row in report.rows}["b"]["a"] == 0.5


def test_rank_frontier_serializes_pair_margins_failures_and_runtime(
    monkeypatch: pytest.MonkeyPatch, verified_frontier: VerifiedFrontier
) -> None:
    """The field report retains diagnostics needed to audit a noisy frontier run."""

    class MeasuredScores(list[float]):
        margins = (120, -20)
        failures = ("seat zero timeout",)
        runtime_seconds = 0.125

    monkeypatch.setattr(
        frontier.arena,
        "outcomes",
        lambda candidate, league, seeds, workers: MeasuredScores([1.0, 0.5]),
    )

    report = rank_frontier(verified_frontier, seeds=(11,), workers=1)
    matchup = report.rows[0].matchup_results[0]

    assert matchup.paired_margin == pytest.approx(50.0)
    assert matchup.failures == ("seat zero timeout",)
    assert matchup.runtime_seconds == pytest.approx(0.125)
    assert report.rows[0].paired_margin == pytest.approx(50.0)
    assert report.failures == ("a vs b: seat zero timeout",)
    assert report.runtime_seconds == pytest.approx(0.125)


def test_verify_frontier_rejects_a_manifest_for_another_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Sources measured under one engine must not silently run under another."""
    source = tmp_path / "agent.py"
    source.write_text("def agent(obs, config=None): return {}\n")
    manifest = FrontierManifest(
        engine="1.32.7",
        artifacts=(
            FrontierArtifact(
                name="agent",
                provenance="test fixture",
                relative_path=Path("agent.py"),
                sha256=frontier.hashlib.sha256(source.read_bytes()).hexdigest(),
            ),
        ),
    )
    path = tmp_path / "manifest.json"
    path.write_text(manifest.model_dump_json())
    monkeypatch.setattr(frontier.kaggle_environments, "__version__", "1.32.8")

    with pytest.raises(FrontierIntegrityError, match="engine 1.32.7.*1.32.8"):
        verify_frontier(path, tmp_path)


def test_round_robin_cli_lowers_its_own_cpu_priority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A direct CLI invocation protects the active training job by default."""
    output = tmp_path / "report.json"
    lowered: list[int] = []
    monkeypatch.setattr(frontier_round_robin.os, "nice", lowered.append)
    monkeypatch.setattr(frontier_round_robin, "verify_frontier", lambda *_: object())
    monkeypatch.setattr(
        frontier_round_robin,
        "rank_frontier",
        lambda *_: frontier.FrontierReport(
            engine="1.32.7",
            seeds=(),
            frontier_name="a",
            failures=(),
            runtime_seconds=0.0,
            rows=(),
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["frontier_round_robin", "--output", str(output)],
    )

    frontier_round_robin.main()

    assert lowered == [10]
