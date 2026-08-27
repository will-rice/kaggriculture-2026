"""Production CLI preflight and read-only validation contracts."""

# Test names are the behavioral documentation.
# ruff: noqa: D103

from __future__ import annotations

import hashlib
import json
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest
from optuna.storages import RDBStorage

from kaggriculture.search.evolution import SnapshotLeague
from kaggriculture.search.fitness import StrengthWeights
from kaggriculture.search.frontier import VerifiedFrontier
from kaggriculture.search.optuna_protocol import PanelSet
from kaggriculture.search.optuna_search import SearchSummary
from kaggriculture.search.scripts import hybrid_optuna as cli

from .test_optuna_search import identity_fixture, warm_configs

BASE_ARGS = ("--frontier-report", "frontier.json")


def test_cli_worker_and_stop_ranges_are_strict() -> None:
    assert cli.parse_args([*BASE_ARGS, "--workers", "32"]).workers == 32
    assert cli.parse_args([*BASE_ARGS, "--stop-after", "512"]).stop_after == 512
    with pytest.raises(SystemExit):
        cli.parse_args([*BASE_ARGS, "--workers", "33"])
    with pytest.raises(SystemExit):
        cli.parse_args([*BASE_ARGS, "--stop-after", "0"])


def test_cli_verifies_legacy_and_frontier_before_reading_report_or_opening_pool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    configs = warm_configs()
    monkeypatch.setattr(
        cli,
        "warm_start_configs",
        lambda _path: calls.append("legacy") or configs,
    )
    monkeypatch.setattr(
        cli,
        "verify_frontier",
        lambda *_: calls.append("verify") or SimpleNamespace(),
    )
    monkeypatch.setattr(
        cli,
        "PersistentArena",
        lambda *_: pytest.fail("arena opened during failed preflight"),
    )
    args = cli.parse_args(
        [
            "--frontier-report",
            str(tmp_path / "missing.json"),
            "--legacy-state",
            str(tmp_path / "legacy.json"),
        ]
    )
    with pytest.raises(SystemExit, match="frontier report"):
        cli.run(args)
    assert calls == ["legacy", "verify"]


def test_validate_only_is_read_only_and_never_opens_arena_or_wandb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    configs = warm_configs()
    identity = identity_fixture(configs)
    root = tmp_path / "study"
    root.mkdir()
    legacy = tmp_path / "legacy.json"
    legacy.write_bytes(b"legacy bytes")
    sqlite = root / "study.sqlite3"
    sqlite.write_bytes(b"immutable sqlite bytes")
    (root / "identity.json").write_bytes(b"immutable identity bytes")
    before = {path: path.read_bytes() for path in (sqlite, root / "identity.json")}
    study = optuna.create_study()
    monkeypatch.setattr(
        cli,
        "warm_start_configs",
        lambda _path: calls.append("legacy") or configs,
    )
    monkeypatch.setattr(
        cli,
        "verify_frontier",
        lambda *_: calls.append("frontier") or SimpleNamespace(),
    )
    monkeypatch.setattr(
        cli,
        "_load_frontier_report",
        lambda _path: calls.append("report") or SimpleNamespace(),
    )
    monkeypatch.setattr(
        cli,
        "_validate_frontier_report",
        lambda *_: calls.append("report_validate"),
    )
    monkeypatch.setattr(
        cli,
        "_protocol_inputs",
        lambda _report: (
            calls.append("seeds_panels_weights")
            or (SimpleNamespace(), SimpleNamespace())
        ),
    )
    monkeypatch.setattr(
        cli,
        "_preflight_output_ownership",
        lambda *_: calls.append("ownership"),
    )
    monkeypatch.setattr(
        cli,
        "snapshot_frontier",
        lambda *_, **kwargs: (
            calls.append(f"snapshot:{kwargs['require_existing']}")
            or SimpleNamespace(opponents={})
        ),
    )
    monkeypatch.setattr(
        cli,
        "build_identity",
        lambda *_: calls.append("identity") or identity,
    )
    monkeypatch.setattr(
        cli,
        "validate_study_paths",
        lambda *_: calls.append("paths") or SimpleNamespace(),
    )
    monkeypatch.setattr(
        cli,
        "_readonly_study",
        lambda *_: nullcontext(study),
    )
    monkeypatch.setattr(
        cli,
        "validate_study_evidence",
        lambda *_: calls.append("evidence"),
    )
    monkeypatch.setattr(
        cli,
        "PersistentArena",
        lambda *_: pytest.fail("validate-only created workers"),
    )
    monkeypatch.setattr(
        cli,
        "run_search",
        lambda *_args, **_kwargs: pytest.fail("validate-only started search"),
    )
    args = cli.parse_args(
        [
            "--frontier-report",
            str(tmp_path / "report.json"),
            "--legacy-state",
            str(legacy),
            "--root",
            str(root),
            "--validate-only",
        ]
    )
    cli.run(args)
    assert calls == [
        "legacy",
        "frontier",
        "report",
        "report_validate",
        "seeds_panels_weights",
        "ownership",
        "snapshot:True",
        "identity",
        "paths",
        "evidence",
    ]
    assert {path: path.read_bytes() for path in before} == before


def test_normal_cli_reports_cpu_only_and_passes_one_arena_factory_after_preflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[str] = []
    configs = warm_configs()
    identity = identity_fixture(configs)
    study = optuna.create_study()
    snapshot = SimpleNamespace(opponents={})
    legacy = tmp_path / "legacy.json"
    legacy.write_bytes(b"legacy bytes")
    monkeypatch.setattr(os, "nice", lambda *_: pytest.fail("CLI called os.nice"))
    monkeypatch.setattr(cli, "warm_start_configs", lambda _path: configs)
    monkeypatch.setattr(cli, "verify_frontier", lambda *_: SimpleNamespace())
    monkeypatch.setattr(cli, "_load_frontier_report", lambda _path: SimpleNamespace())
    monkeypatch.setattr(cli, "_validate_frontier_report", lambda *_: None)
    monkeypatch.setattr(
        cli,
        "_protocol_inputs",
        lambda _report: (
            PanelSet(
                identity.panels.rung_1,
                identity.panels.rung_2,
                identity.panels.rung_3,
            ),
            StrengthWeights(dict(identity.strength_weights)),
        ),
    )
    monkeypatch.setattr(cli, "_preflight_output_ownership", lambda *_: None)
    monkeypatch.setattr(cli, "snapshot_frontier", lambda *_, **__: snapshot)
    monkeypatch.setattr(cli, "build_identity", lambda *_: identity)
    monkeypatch.setattr(cli, "validate_study_paths", lambda *_: SimpleNamespace())
    monkeypatch.setattr(cli, "open_study", lambda *_: study)
    monkeypatch.setattr(cli, "reconcile_running_trials", lambda _study: ())
    monkeypatch.setattr(cli, "validate_study_evidence", lambda *_: None)
    arena_factory = object()
    monkeypatch.setattr(cli, "PersistentArena", arena_factory)

    def fake_run_search(*_args: object, **kwargs: object) -> SearchSummary:
        calls.append("search")
        assert kwargs["arena_factory"] is arena_factory
        return SearchSummary(
            started_trials=0,
            terminal_trials=0,
            complete_trials=0,
            pruned_trials=0,
            failed_trials=0,
            stopped_reason="stop_after",
            best_trial=None,
        )

    monkeypatch.setattr(cli, "run_search", fake_run_search)
    args = cli.parse_args(
        [
            "--frontier-report",
            str(tmp_path / "report.json"),
            "--legacy-state",
            str(legacy),
            "--root",
            str(tmp_path / "study"),
            "--workers",
            "32",
            "--stop-after",
            "1",
            "--no-wandb",
        ]
    )
    cli.run(args)
    output = capsys.readouterr().out
    assert calls == ["search"]
    assert '"device": "cpu"' in output
    assert '"workers": 32' in output


def test_output_preflight_rejects_symlink_traversal_before_snapshot_creation(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    source = tmp_path / "opponent.py"
    source.write_text("def agent(*_): return None\n")
    legacy = tmp_path / "legacy.json"
    legacy.write_text("{}")
    frontier = VerifiedFrontier(
        engine="test",
        opponents={"opponent": str(source)},
        artifacts=(),
    )
    paths = cli.StudyPaths.from_root(linked_parent / "study")

    with pytest.raises(SystemExit, match="symlink"):
        cli._preflight_output_ownership(paths, frontier, legacy)
    assert not (real_parent / "study").exists()


def test_readonly_study_uses_a_backup_without_changing_original_bytes(
    tmp_path: Path,
) -> None:
    configs = warm_configs()
    legacy = tmp_path / "legacy.json"
    legacy.write_bytes(b"legacy bytes")
    identity = identity_fixture(configs).model_copy(
        update={"legacy_state_sha256": hashlib.sha256(legacy.read_bytes()).hexdigest()}
    )
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    snapshot = SnapshotLeague(str(snapshot_root), ())
    paths = cli.StudyPaths.from_root(tmp_path / "study")
    paths.root.mkdir()
    paths.identity.write_text(
        json.dumps(
            identity.model_dump(mode="json"), separators=(",", ":"), sort_keys=True
        )
    )
    study = optuna.create_study(
        storage=RDBStorage(f"sqlite:///{paths.sqlite.absolute()}"),
        study_name=identity.study_name,
        direction="maximize",
    )
    study.set_user_attr("study_identity_sha256", identity.digest)
    before = {
        paths.identity: paths.identity.read_bytes(),
        paths.sqlite: paths.sqlite.read_bytes(),
    }

    with cli._readonly_study(paths, identity, snapshot, legacy) as loaded:
        assert loaded.study_name == identity.study_name
        assert loaded.user_attrs["study_identity_sha256"] == identity.digest

    assert {path: path.read_bytes() for path in before} == before
