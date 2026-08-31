"""Contracts for immutable counterfactual publication.

Every test here damages a shard the way a real failure would and demands the
loader refuse it: a repeated row, an event whose controller anchor is gone, a
seed that is not this cell's, a manifest that does not match the bytes, an
outcome that records a failure, and an identity edited without its digest.

The rows under test are real. They come from a real branch of a real season, so
the schema is exercised on the numbers collection actually produces -- exact
integer banks, a scoring rule that lands on 0.0, 0.5 or 1.0, and a paired delta
that must be the difference of two floats already written down.
"""
# ruff: noqa: D103

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from kaggriculture.learn.market_residual.alternatives import FAMILY_KAITO
from kaggriculture.learn.market_residual.artifacts import (
    MANIFEST_SUFFIX,
    CounterfactualImmutabilityError,
    CounterfactualRow,
    CounterfactualShard,
    ShardIdentity,
    counterfactual_rows,
    load_shard_strict,
    shard_bytes,
    shard_path,
    write_shard_atomic,
)
from kaggriculture.learn.market_residual.counterfactual import (
    CounterfactualIntegrityError,
    CounterfactualSnapshot,
    SeasonIdentity,
)

DAMAGES = ("duplicate", "missing_kaito", "seed", "hash", "status", "identity")


@pytest.fixture(scope="session")
def late_shard(
    branch_identity: SeasonIdentity, late_snapshot: CounterfactualSnapshot
) -> CounterfactualShard:
    """Branch one real event and hold its rows as a publishable shard."""
    return CounterfactualShard(
        identity=ShardIdentity.of(branch_identity),
        rows=counterfactual_rows(late_snapshot),
    )


def test_a_shard_holds_one_anchored_row_per_alternative(
    late_shard: CounterfactualShard, branch_identity: SeasonIdentity
) -> None:
    rows = late_shard.rows
    assert len(rows) > 1
    assert rows[0].family == FAMILY_KAITO
    assert rows[0].alternative_id == 0
    assert rows[0].win_point_delta == 0.0
    assert {row.alternative_id for row in rows} == set(range(len(rows)))
    assert {row.identity_sha256 for row in rows} == {branch_identity.sha256}
    assert {row.seed for row in rows} == {branch_identity.seed}
    for row in rows:
        assert row.kaito == rows[0].kaito
        assert row.win_point_delta == row.alternative.win_points - row.kaito.win_points
        assert row.margin_delta == row.alternative.margin - row.kaito.margin


def test_a_published_shard_round_trips_byte_identically(
    tmp_path: Path, late_shard: CounterfactualShard, branch_identity: SeasonIdentity
) -> None:
    path = write_shard_atomic(tmp_path, late_shard)
    loaded = load_shard_strict(path, branch_identity)

    assert path == shard_path(tmp_path, late_shard.identity)
    assert loaded == late_shard
    assert shard_bytes(loaded) == path.read_bytes()
    assert sorted(entry.name for entry in tmp_path.iterdir()) == sorted(
        [path.name, path.with_suffix(MANIFEST_SUFFIX).name]
    ), "publication left a temporary behind"


def test_the_manifest_publishes_the_digest_of_the_bytes(
    tmp_path: Path, late_shard: CounterfactualShard
) -> None:
    path = write_shard_atomic(tmp_path, late_shard)
    manifest = path.with_suffix(MANIFEST_SUFFIX)

    assert (
        manifest.read_text(encoding="utf-8").strip()
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )


def test_republishing_the_same_bytes_is_a_no_op(
    tmp_path: Path, late_shard: CounterfactualShard, branch_identity: SeasonIdentity
) -> None:
    first = write_shard_atomic(tmp_path, late_shard)
    before = first.read_bytes()
    second = write_shard_atomic(tmp_path, late_shard)

    assert first == second
    assert second.read_bytes() == before
    assert load_shard_strict(second, branch_identity) == late_shard


def test_a_second_write_of_different_bytes_is_refused(
    tmp_path: Path, late_shard: CounterfactualShard
) -> None:
    """The one thing a published address may never do is change its mind."""
    path = write_shard_atomic(tmp_path, late_shard)
    before = path.read_bytes()
    shorter = CounterfactualShard(
        identity=late_shard.identity, rows=late_shard.rows[:1]
    )

    assert shard_path(tmp_path, shorter.identity) == path, (
        "the cell must address the same file, or nothing is being overwritten"
    )
    with pytest.raises(CounterfactualImmutabilityError, match="different bytes"):
        write_shard_atomic(tmp_path, shorter)
    assert path.read_bytes() == before


def test_a_shard_without_its_manifest_is_not_published(
    tmp_path: Path, late_shard: CounterfactualShard, branch_identity: SeasonIdentity
) -> None:
    """A crash between the two links leaves bytes that are not evidence yet."""
    path = write_shard_atomic(tmp_path, late_shard)
    path.with_suffix(MANIFEST_SUFFIX).unlink()

    with pytest.raises(CounterfactualIntegrityError, match="no published manifest"):
        load_shard_strict(path, branch_identity)


@pytest.mark.parametrize("damage", DAMAGES)
def test_a_damaged_shard_rejects_without_partial_rows(
    tmp_path: Path,
    late_shard: CounterfactualShard,
    branch_identity: SeasonIdentity,
    damage: str,
) -> None:
    path = write_shard_atomic(tmp_path, late_shard)
    assert load_shard_strict(path, branch_identity) == late_shard

    damage_shard(path, damage)

    with pytest.raises(CounterfactualIntegrityError):
        load_shard_strict(path, branch_identity)


def test_a_shard_from_another_cell_is_refused(
    tmp_path: Path, late_shard: CounterfactualShard, branch_identity: SeasonIdentity
) -> None:
    path = write_shard_atomic(tmp_path, late_shard)
    other = replace(branch_identity, seed=branch_identity.seed + 1)

    with pytest.raises(CounterfactualIntegrityError, match="was produced under"):
        load_shard_strict(path, other)


def test_a_row_that_moves_the_anchor_off_row_zero_is_refused(
    late_shard: CounterfactualShard,
) -> None:
    anchor = late_shard.rows[0].model_dump(mode="json")

    with pytest.raises(ValueError, match="not the anchor's position"):
        parse_row({**anchor, "alternative_id": 3})
    with pytest.raises(ValueError, match="not the anchor's position"):
        parse_row({**anchor, "family": "single"})


def test_a_row_whose_delta_is_not_the_paired_difference_is_refused(
    late_shard: CounterfactualShard,
) -> None:
    replacement = next(
        row for row in late_shard.rows if row.family != FAMILY_KAITO
    ).model_dump(mode="json")

    with pytest.raises(ValueError, match="win point delta"):
        parse_row(
            {**replacement, "win_point_delta": replacement["win_point_delta"] + 0.5}
        )


def parse_row(payload: dict[str, object]) -> CounterfactualRow:
    """Parse one row the way a published shard is parsed, from its own JSON.

    Args:
        payload: A row's canonical fields.

    Returns:
        The validated row.
    """
    return CounterfactualRow.model_validate_json(json.dumps(payload))


def test_an_identity_edited_without_its_digest_is_refused(
    branch_identity: SeasonIdentity,
) -> None:
    published = ShardIdentity.of(branch_identity).model_dump(mode="json")
    published["fields"]["seed"] += 1

    with pytest.raises(ValueError, match="does not match its own fields"):
        ShardIdentity.model_validate(published)


def damage_shard(path: Path, damage: str) -> None:
    """Corrupt one published shard the way a real failure would.

    Every damage except ``hash`` republishes a matching manifest, so each case
    is refused by the check it is aimed at rather than by the digest.

    Args:
        path: The published shard.
        damage: Which corruption to apply.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    manifest = path.with_suffix(MANIFEST_SUFFIX)
    if damage == "hash":
        manifest.unlink()
        manifest.write_text("0" * 64 + "\n", encoding="utf-8")
        return
    if damage == "duplicate":
        lines.append(lines[-1])
    elif damage == "missing_kaito":
        lines = [lines[0], *lines[2:]]
    elif damage == "seed":
        row = json.loads(lines[1])
        row["seed"] += 1
        lines[1] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    elif damage == "status":
        row = json.loads(lines[1])
        row["failure"] = "timeout"
        lines[1] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    elif damage == "identity":
        identity = json.loads(lines[0])
        identity["fields"]["seed"] += 1
        lines[0] = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    else:
        raise ValueError(f"unknown damage {damage!r}")
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    path.unlink()
    path.write_bytes(payload)
    manifest.unlink()
    manifest.write_text(hashlib.sha256(payload).hexdigest() + "\n", encoding="utf-8")
