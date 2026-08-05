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
    unit_count,
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


def _empty_farm(farmer: tuple[int, int] = (0, 0), hands: int = 0) -> dict:
    """Return a fresh, independent farm dict for a synthetic observation."""
    return {
        "tiles": [[None] * BOARD for _ in range(BOARD)],
        "money": 3000.0,
        "farmer": list(farmer),
        "hands": [[1, 1] for _ in range(hands)],
        "unlocked_quadrants": ["NW"],
    }


def _observation(farmer: tuple[int, int] = (0, 0), hands: int = 0) -> dict:
    """Return a minimal well-formed observation for a synthetic episode.

    ``farmer`` places seat 0's farmer and ``hands`` staffs it, so an episode can
    be built in which a unit stands somewhere different on each step, or in
    which one turn has more units on the board than ``MAX_UNITS`` covers.
    """
    return {
        "day": 0,
        "hour": 0,
        "step": 0,
        "farms": [_empty_farm(farmer, hands), _empty_farm()],
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
    board, scalars, positions, labels = Shards([destination])[0]
    assert board.shape == (TILE_PLANES, BOARD, BOARD)
    assert scalars.shape == (SCALARS,)
    assert positions.shape == (MAX_UNITS,)
    assert labels.shape == (MAX_UNITS,)
    assert board.dtype == torch.float32
    assert positions.dtype == torch.int64
    assert positions.ge(0).all() and positions.lt(BOARD * BOARD).all()


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

    def encoded(state: int, action: int) -> torch.Tensor:
        """Label the action at ``action`` against the units standing at ``state``."""
        observation = steps[state][sample.seat]["observation"]
        units = unit_count(observation, sample.seat)
        return encode_units(steps[action][sample.seat]["action"], units)[0]

    turn = next(
        i
        for i in range(1, len(steps) - 1)
        if not torch.equal(encoded(i, i), encoded(i, i + 1))
    )
    destination = tmp_path / "aligned.npz"
    build_shard([sample], destination, stride=1)

    _, _, _, labels = Shards([destination])[turn]

    assert torch.equal(labels, encoded(turn, turn + 1))
    assert not torch.equal(labels, encoded(turn, turn))


def test_positions_come_from_the_state_the_decision_was_made_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Positions are input, so they are read one step earlier than the label.

    A row pairs ``observation[i]`` with ``action[i + 1]``, and it is tempting to
    read positions alongside the label. That would place every unit where it
    ended up rather than where it decided from -- a hand that walked north has
    already left the tile whose surroundings made it choose NORTH -- and the
    model would be trained to justify moves after the fact.

    The synthetic episode walks seat 0's farmer to a different tile on every
    step, so the right and wrong sources cannot agree by accident.
    """
    action = {"farmer": ["NORTH"], "hands": [], "market": []}
    episode = {
        "steps": [
            [{"action": action, "observation": _observation(farmer=(1, 1))}, {}],
            [{"action": action, "observation": _observation(farmer=(7, 3))}, {}],
            [{"action": action, "observation": _observation(farmer=(9, 9))}, {}],
        ]
    }

    archive = _write_synthetic_archive(tmp_path, "synthetic.zip", episode)
    monkeypatch.setattr(dataset_module, "CORPUS", tmp_path)

    sample = Sample(archive=archive.name, name="42.json", seat=0, rating=2600.0)
    destination = tmp_path / "shard.npz"
    rows = build_shard([sample], destination, stride=1)
    shards = Shards([destination])

    assert rows == 2
    assert shards[0][2][0].item() == 1 * BOARD + 1
    assert shards[1][2][0].item() == 7 * BOARD + 3


def test_a_turn_over_max_units_is_skipped_not_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """MAX_UNITS is an empirical bound; one oversized turn must not abort the build.

    Built against a synthetic archive so it runs on every machine, not only one
    with the corpus, and stays fast enough for the default test run. Four steps
    give three ``(observation[i], action[i + 1])`` rows: the first and third
    observations are ordinary, the middle one puts more units on the board than
    ``MAX_UNITS`` covers, and its row must be skipped, counted and named in a
    warning rather than aborting the other two.

    The oversized count is in the *observation*, not the action, because the
    observation is what the engine walks -- an action listing ops for hands the
    farm does not have is ordinary in the corpus and no-ops in the engine.
    """
    action = {"farmer": ["PASS"], "hands": [], "market": []}
    # steps[i][seat] is the shape json.load gives a real episode; seat 1 is
    # never read by this test, so its entry is left empty.
    episode = {
        "steps": [
            [{"action": action, "observation": _observation()}, {}],
            [{"action": action, "observation": _observation(hands=MAX_UNITS)}, {}],
            [{"action": action, "observation": _observation()}, {}],
            [{"action": action, "observation": _observation()}, {}],
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
