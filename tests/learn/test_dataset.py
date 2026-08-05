"""Tests for the sharded training data, built from the real corpus."""

import json
import logging
import zipfile
from pathlib import Path

import pytest
import torch

from kaggriculture.constants import PRODUCTS
from kaggriculture.learn import dataset as dataset_module
from kaggriculture.learn.corpus import CORPUS, Sample
from kaggriculture.learn.dataset import Shards, build_shard
from kaggriculture.learn.encoding import (
    BOARD,
    MAX_UNITS,
    SCALARS,
    TILE_PLANES,
    encode_units,
)

ARCHIVE = CORPUS / "kaggriculture-episodes-2026-08-03.zip"

_needs_corpus = pytest.mark.skipif(
    not ARCHIVE.exists(), reason="replay corpus not present on this machine"
)


def one_sample() -> Sample:
    """Return a sample naming a real episode in the corpus."""
    with zipfile.ZipFile(ARCHIVE) as bundle:
        name = next(n for n in bundle.namelist() if n.endswith(".json"))
    return Sample(archive=ARCHIVE.name, name=name, seat=0, rating=2600.0)


def _empty_farm() -> dict:
    """Return a fresh, independent farm dict for a synthetic observation."""
    return {
        "tiles": [[None] * BOARD for _ in range(BOARD)],
        "money": 3000.0,
        "hands": [],
        "unlocked_quadrants": ["NW"],
    }


def _observation() -> dict:
    """Return a minimal well-formed observation for a synthetic episode."""
    return {
        "day": 0,
        "hour": 0,
        "step": 0,
        "farms": [_empty_farm(), _empty_farm()],
        "market": {
            "prices": dict.fromkeys(PRODUCTS, 100),
            "inventory": dict.fromkeys(PRODUCTS, 10000),
        },
        "town": {"unlocked_shops": []},
    }


def _write_synthetic_archive(tmp_path: Path, name: str, episode: dict) -> Path:
    """Write one episode as a single-member zip archive and return its path."""
    archive = tmp_path / name
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("42.json", json.dumps(episode))
    return archive


@pytest.mark.slow
@_needs_corpus
def test_a_shard_round_trips_into_tensors_of_the_declared_shape(tmp_path: Path) -> None:
    """The model's input shape is fixed; a ragged shard would fail at train time."""
    destination = tmp_path / "shard-000.npz"

    rows = build_shard([one_sample()], destination, stride=64)

    assert rows > 0
    board, scalars, labels = Shards([destination])[0]
    assert board.shape == (TILE_PLANES, BOARD, BOARD)
    assert scalars.shape == (SCALARS,)
    assert labels.shape == (MAX_UNITS,)
    assert board.dtype == torch.float32


@pytest.mark.slow
@_needs_corpus
def test_stride_controls_how_many_turns_are_kept(tmp_path: Path) -> None:
    """720 turns per seat is more correlated data than it is information."""
    dense = build_shard([one_sample()], tmp_path / "dense.npz", stride=8)
    sparse = build_shard([one_sample()], tmp_path / "sparse.npz", stride=64)

    assert dense > sparse


@pytest.mark.slow
@_needs_corpus
def test_labels_are_the_action_taken_from_the_state_not_the_one_that_made_it(
    tmp_path: Path,
) -> None:
    """The label must be the decision that follows an observation, not precedes it.

    ``kaggle_environments``' interpreter mutates the state object carried
    alongside the action it is applying, so a recorded ``steps[i][seat]`` holds
    the state *after* ``action[i]`` ran -- not before. Pairing them by index
    therefore asks the model to predict an action from the world that action
    already created, which is label leakage during training and a distribution
    it never sees at inference. The correct pair is
    ``(observation[i], action[i + 1])``; ``action[0]`` is a reset filler with
    nothing to predict and is dropped.

    The obvious version of this test cannot fail. Sampled sparsely, most
    consecutive actions are both all-``PASS`` and encode identically, so the
    right and wrong pairings agree and the assertion passes either way. This
    one seeks out an index where the two genuinely differ and pins down both
    directions.
    """
    sample = one_sample()
    with zipfile.ZipFile(ARCHIVE) as bundle, bundle.open(sample.name) as member:
        steps = json.load(member)["steps"]

    turn = next(
        i
        for i in range(1, len(steps) - 1)
        if not torch.equal(
            encode_units(steps[i][sample.seat]["action"] or {})[0],
            encode_units(steps[i + 1][sample.seat]["action"] or {})[0],
        )
    )
    destination = tmp_path / "aligned.npz"
    build_shard([sample], destination, stride=1)

    _, _, labels = Shards([destination])[turn]

    assert torch.equal(labels, encode_units(steps[turn + 1][sample.seat]["action"])[0])
    assert not torch.equal(labels, encode_units(steps[turn][sample.seat]["action"])[0])


def test_a_turn_over_max_units_is_skipped_not_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """MAX_UNITS is an empirical bound; one oversized turn must not abort the build.

    Built against a synthetic archive so it runs on every machine, not only one
    with the corpus, and stays fast enough for the default test run. Four steps
    give three ``(observation[i], action[i + 1])`` rows: the first and third
    actions are ordinary, the middle one exceeds ``MAX_UNITS`` and must be
    skipped, counted and named in a warning rather than aborting the other two.
    """
    reset_action = {"farmer": ["PASS"], "hands": [], "market": []}
    fine_action = {"farmer": ["PASS"], "hands": [], "market": []}
    over_action = {
        "farmer": ["PASS"],
        "hands": [["PASS"] for _ in range(MAX_UNITS)],
        "market": [],
    }
    # steps[i][seat] is the shape json.load gives a real episode; seat 1 is
    # never read by this test, so its entry is left empty.
    episode = {
        "steps": [
            [{"action": reset_action, "observation": _observation()}, {}],
            [{"action": fine_action, "observation": _observation()}, {}],
            [{"action": over_action, "observation": _observation()}, {}],
            [{"action": fine_action, "observation": _observation()}, {}],
        ]
    }

    archive = _write_synthetic_archive(tmp_path, "synthetic.zip", episode)
    monkeypatch.setattr(dataset_module, "CORPUS", tmp_path)

    sample = Sample(archive=archive.name, name="42.json", seat=0, rating=2600.0)
    with caplog.at_level(logging.WARNING, logger=dataset_module.LOGGER.name):
        rows = build_shard([sample], tmp_path / "shard.npz", stride=1)

    assert rows == 2
    warnings = [
        record for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert "42.json" in warnings[0].getMessage()
    assert "MAX_UNITS" in warnings[0].getMessage()


def test_the_shard_cap_splits_a_large_build_into_numbered_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full build must never hold more than one shard's worth of tensors.

    ``ROWS_PER_SHARD`` is patched down to 1 so a four-turn synthetic episode
    forces multiple flushes, proving shards actually split rather than one
    giant file always being written regardless of the cap.
    """
    fine_action = {"farmer": ["PASS"], "hands": [], "market": []}
    episode = {
        "steps": [
            [{"action": fine_action, "observation": _observation()}, {}]
            for _ in range(5)
        ]
    }

    archive = _write_synthetic_archive(tmp_path, "synthetic.zip", episode)
    monkeypatch.setattr(dataset_module, "CORPUS", tmp_path)
    monkeypatch.setattr(dataset_module, "ROWS_PER_SHARD", 1)

    sample = Sample(archive=archive.name, name="42.json", seat=0, rating=2600.0)
    destination = tmp_path / "shard.npz"
    rows = build_shard([sample], destination, stride=1)

    assert rows == 4
    shard_paths = [destination] + [
        destination.with_name(f"shard-{index:03d}.npz") for index in range(1, 4)
    ]
    for path in shard_paths:
        assert path.exists()
    assert Shards(shard_paths).boards.shape[0] == 4
