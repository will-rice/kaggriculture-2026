"""Contracts for the counterfactual collection command.

Nothing here plays a season. That is deliberate: every property this file pins
is a property of the *plan* -- which cells will be played, under which identity,
against which opponent, out of which seed bank -- and a plan that has to play a
game to be checked is a plan that has already spent the budget it was supposed
to bound. The one fixture that needs real rows re-stamps the rows the shared
branch fixtures already produced, so the reconcile path is exercised against a
shard the strict loader accepts rather than against a hand-built one.

The forbidden sentinels are the load-bearing part. Two of these tests assert
that a refusal happens *before* the first season is recorded, and they can only
assert that by making the recording itself an error.
"""
# ruff: noqa: D103

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from kaggriculture.learn.market_residual.alternatives import AlternativeConfig
from kaggriculture.learn.market_residual.artifacts import (
    CounterfactualRow,
    CounterfactualShard,
    ShardIdentity,
    counterfactual_rows,
    write_shard_atomic,
)
from kaggriculture.learn.market_residual.counterfactual import (
    CounterfactualIntegrityError,
    CounterfactualSnapshot,
    SeasonIdentity,
)
from kaggriculture.market_residual.baseline import SERVED
from kaggriculture.scripts import market_counterfactuals as cli

SMOKE_SEED = 860_000


def forbidden(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 - a sentinel
    """Fail loudly if a code path that must play no games is reached.

    Args:
        *args: Ignored.
        **kwargs: Ignored.

    Raises:
        AssertionError: Always.
    """
    raise AssertionError("this run must not touch the engine")


def arguments(root: Path, *extra: str) -> list[str]:
    """Return a complete argument vector for one collection root.

    Args:
        root: The collection root.
        *extra: Arguments appended after the defaults.

    Returns:
        The argument vector.
    """
    return [
        str(root),
        "--seed-start",
        str(SMOKE_SEED),
        "--seed-count",
        "1",
        "--workers",
        "1",
        "--max-events",
        "1",
        "--max-alternatives",
        "8",
        *extra,
    ]


def shard_digests(root: Path) -> dict[str, str]:
    """Return the digest of every published artifact under one root.

    Args:
        root: The collection root.

    Returns:
        A mapping from relative path to SHA-256.
    """
    shards = root / cli.SHARD_DIRECTORY
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(shards.rglob("*"))
        if path.is_file()
    }


@pytest.fixture
def live_root(tmp_path: Path) -> Path:
    """Return a created collection root with no cell published yet."""
    root = tmp_path / "collection"
    cli.main(arguments(root, "--create", "--validate-only"))
    return root


def test_the_cli_requires_exactly_one_mode(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        cli.main(arguments(tmp_path / "a"))
    with pytest.raises(SystemExit):
        cli.main(arguments(tmp_path / "a", "--create", "--resume"))


def test_a_held_out_gate_seed_never_reaches_a_game(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "record_season", forbidden)
    with pytest.raises(ValueError, match="held-out gate seed"):
        cli.main(
            [
                str(tmp_path / "gate"),
                "--create",
                "--seed-start",
                "700000",
                "--seed-count",
                "2",
            ]
        )


def test_a_gate_bank_is_not_a_collection_bank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "record_season", forbidden)
    with pytest.raises(ValueError, match="not a collection bank"):
        cli.main(
            [
                str(tmp_path / "gate"),
                "--create",
                "--seed-bank",
                "online_gate",
                "--seed-start",
                "872000",
                "--seed-count",
                "2",
            ]
        )


def test_a_route_replaying_opponent_is_refused_before_any_game(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "record_season", forbidden)
    with pytest.raises(cli.UnrestorableOpponentError, match="from the board"):
        cli.main(arguments(tmp_path / "r", "--create", "--opponent", "route"))


def test_a_frontier_artifact_opponent_is_refused_with_its_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "record_season", forbidden)
    with pytest.raises(cli.UnrestorableOpponentError, match="by path"):
        cli.main(arguments(tmp_path / "f", "--create", "--opponent", "kaito_v27"))


def test_an_unknown_opponent_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "record_season", forbidden)
    with pytest.raises(cli.UnrestorableOpponentError, match="unknown opponent"):
        cli.main(arguments(tmp_path / "u", "--create", "--opponent", "nobody"))


def test_workers_outside_the_bound_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="workers"):
        cli.main(arguments(tmp_path / "w", "--create", "--workers", "33"))


def test_the_shipped_budget_is_what_the_default_cap_produces() -> None:
    assert cli.alternative_budget(48) == AlternativeConfig()


def test_a_smaller_cap_keeps_the_shipped_family_mix() -> None:
    budget = cli.alternative_budget(16)
    assert budget.max_alternatives == 16
    assert 1 + 3 + budget.max_single + budget.max_ranked_multi == 16
    assert budget.max_single > budget.max_ranked_multi


def test_a_cap_too_small_for_every_family_is_refused() -> None:
    with pytest.raises(ValueError, match="alternatives"):
        cli.alternative_budget(cli.MINIMUM_ALTERNATIVES - 1)


def test_event_selection_is_bounded_and_spread() -> None:
    events = tuple(range(100))
    chosen = cli.selected_events(events, 4)
    assert chosen == (0, 25, 50, 75)
    assert cli.selected_events(events[:3], 4) == (0, 1, 2)


def test_a_second_create_on_a_live_root_is_refused(live_root: Path) -> None:
    with pytest.raises(cli.CollectionParameterError, match="already collecting"):
        cli.main(arguments(live_root, "--create", "--validate-only"))


def test_a_resume_of_different_parameters_is_refused(live_root: Path) -> None:
    with pytest.raises(cli.CollectionParameterError, match="different parameters"):
        cli.main(
            arguments(live_root, "--resume", "--validate-only", "--max-events", "2")
        )


def test_a_resume_of_an_uncreated_root_is_refused(tmp_path: Path) -> None:
    with pytest.raises(cli.CollectionParameterError, match="never created"):
        cli.main(arguments(tmp_path / "cold", "--resume", "--validate-only"))


def test_a_corrupted_parameter_file_is_refused(live_root: Path) -> None:
    path = live_root / cli.PARAMETERS_NAME
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["fields"]["max_events"] = 99
    path.write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(cli.CollectionParameterError, match="does not match its own"):
        cli.main(arguments(live_root, "--resume", "--validate-only"))


def test_a_second_holder_of_the_root_lock_is_refused(live_root: Path) -> None:
    with cli.root_lock(live_root):
        with pytest.raises(cli.CollectionLockError, match="held by another"):
            cli.main(arguments(live_root, "--resume", "--validate-only"))


def test_a_resume_quarantines_an_orphan_temporary(live_root: Path) -> None:
    orphan = live_root / cli.SHARD_DIRECTORY / "seed860000-seat0-abc.jsonl.tmp-7-ff"
    orphan.write_bytes(b"half a shard")
    cli.main(arguments(live_root, "--resume", "--validate-only"))
    assert not orphan.exists()
    assert (live_root / cli.QUARANTINE_DIRECTORY / orphan.name).read_bytes() == (
        b"half a shard"
    )


@pytest.fixture(scope="session")
def late_rows(late_snapshot: CounterfactualSnapshot) -> tuple[CounterfactualRow, ...]:
    """Branch one real event once, for every reconcile test in this file.

    Args:
        late_snapshot: The shared late-season branchable instant.

    Returns:
        Every alternative at that event, scored against the controller.
    """
    return counterfactual_rows(late_snapshot)


@pytest.fixture
def published_root(
    tmp_path: Path,
    branch_identity: SeasonIdentity,
    late_rows: tuple[CounterfactualRow, ...],
) -> Path:
    """Return a root holding one fully published cell of the planned run.

    The rows are the shared branch fixture's real rows, re-stamped under the
    identity this command's own planner produces, so the reconcile path meets a
    shard the strict loader accepts without paying for a second season.

    Args:
        tmp_path: The temporary directory.
        branch_identity: The identity the shared fixtures were branched under.
        late_rows: One real branched event's rows.

    Returns:
        The collection root.
    """
    root = tmp_path / "published"
    cli.main(arguments(root, "--create", "--validate-only"))
    planned = cli.collection_plan(
        cli.parser().parse_args(arguments(root, "--resume"))
    ).identities
    target = next(
        identity
        for identity in planned
        if (identity.seed, identity.seat)
        == (branch_identity.seed, branch_identity.seat)
    )
    rows = tuple(
        row.model_copy(update={"identity_sha256": target.sha256}) for row in late_rows
    )
    write_shard_atomic(
        root / cli.SHARD_DIRECTORY,
        CounterfactualShard(identity=ShardIdentity.of(target), rows=rows),
    )
    return root


def test_validate_only_runs_no_games_and_changes_no_artifact(
    published_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "record_season", forbidden)
    monkeypatch.setattr(cli, "counterfactual_rows", forbidden)
    before = shard_digests(published_root)
    cli.main(arguments(published_root, "--resume", "--validate-only"))
    report = json.loads((published_root / cli.REPORT_NAME).read_text(encoding="utf-8"))
    assert shard_digests(published_root) == before
    assert report["cells"] == {
        "planned": 2,
        "reconciled": 1,
        "outstanding": 1,
        "collected": 0,
    }
    assert report["rows"]["reconciled"] > 0


def test_the_report_separates_every_phase_the_budget_is_spent_on(
    published_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "record_season", forbidden)
    cli.main(arguments(published_root, "--resume", "--validate-only"))
    report = json.loads((published_root / cli.REPORT_NAME).read_text(encoding="utf-8"))
    assert set(report["timings"]) == {
        "engine_seconds",
        "feature_seconds",
        "policy_seconds",
        "snapshot_seconds",
        "branch_seconds",
        "fsync_seconds",
    }
    assert report["machine"]["load_average"][0] >= 0.0
    assert report["throughput"]["floor_events_per_hour"] == (
        cli.THROUGHPUT_FLOOR_EVENTS_PER_HOUR
    )


def test_a_published_shard_that_lost_a_byte_fails_the_reconcile_closed(
    published_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "record_season", forbidden)
    shard = next((published_root / cli.SHARD_DIRECTORY).glob("*.jsonl"))
    shard.write_bytes(shard.read_bytes()[:-1])
    with pytest.raises(CounterfactualIntegrityError):
        cli.main(arguments(published_root, "--resume", "--validate-only"))


def test_a_shard_without_its_manifest_is_not_reconciled(
    published_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "record_season", forbidden)
    next((published_root / cli.SHARD_DIRECTORY).glob("*.sha256")).unlink()
    cli.main(arguments(published_root, "--resume", "--validate-only"))
    report = json.loads((published_root / cli.REPORT_NAME).read_text(encoding="utf-8"))
    assert report["cells"]["reconciled"] == 0
    assert report["cells"]["outstanding"] == 2


def test_the_planned_identity_carries_the_verified_frontier_provenance(
    tmp_path: Path,
) -> None:
    plan = cli.collection_plan(
        cli.parser().parse_args(arguments(tmp_path / "p", "--resume"))
    )
    assert plan.provenance["engine"] == "1.32.7"
    assert len(plan.provenance["manifest_sha256"]) == 64
    assert len(plan.provenance["frontier_generation_sha256"]) == 64
    assert plan.provenance["frontier_name"]


def test_every_planned_cell_is_a_seat_of_a_banked_seed(tmp_path: Path) -> None:
    argv = arguments(tmp_path / "p", "--resume", "--seed-count", "8")
    plan = cli.collection_plan(cli.parser().parse_args(argv))
    cells = {(identity.seed, identity.seat) for identity in plan.identities}
    assert len(cells) == 16
    assert {seed for seed, _ in cells} == set(range(SMOKE_SEED, SMOKE_SEED + 8))
    assert all(identity.learner == SERVED for identity in plan.identities)


def test_throughput_below_the_floor_is_a_failure() -> None:
    assert cli.events_per_hour(events=1, wall_seconds=3600.0) == pytest.approx(1.0)
    with pytest.raises(cli.CollectionThroughputError, match="events/hour"):
        cli.check_throughput(events=1, wall_seconds=3600.0)


def test_a_run_that_branched_nothing_is_not_graded() -> None:
    cli.check_throughput(events=0, wall_seconds=1.0)


def parse_fails(argv: Sequence[str]) -> bool:
    """Return whether the parser refuses an argument vector.

    Args:
        argv: The arguments to parse.

    Returns:
        ``True`` when parsing exits.
    """
    try:
        cli.parser().parse_args(list(argv))
    except SystemExit:
        return True
    return False


def test_the_parser_refuses_a_modeless_vector() -> None:
    assert parse_fails([])
    assert parse_fails(["--create", "--resume"])
