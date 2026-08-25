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
