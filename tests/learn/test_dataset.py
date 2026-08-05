"""Tests for the sharded training data, built from the real corpus."""

import json
import zipfile
from pathlib import Path

import pytest
import torch

from kaggriculture.learn.corpus import CORPUS, Sample
from kaggriculture.learn.dataset import Shards, build_shard
from kaggriculture.learn.encoding import MAX_UNITS, SCALARS, TILE_PLANES, encode_units

ARCHIVE = CORPUS / "kaggriculture-episodes-2026-08-03.zip"

_needs_corpus = pytest.mark.skipif(
    not ARCHIVE.exists(), reason="replay corpus not present on this machine"
)


def one_sample() -> Sample:
    """Return a sample naming a real episode in the corpus."""
    with zipfile.ZipFile(ARCHIVE) as bundle:
        name = next(n for n in bundle.namelist() if n.endswith(".json"))
    return Sample(archive=ARCHIVE.name, name=name, seat=0, rating=2600.0)


@pytest.mark.slow
@_needs_corpus
def test_a_shard_round_trips_into_tensors_of_the_declared_shape(tmp_path: Path) -> None:
    """The model's input shape is fixed; a ragged shard would fail at train time."""
    destination = tmp_path / "shard-000.npz"

    rows = build_shard([one_sample()], destination, stride=64)

    assert rows > 0
    board, scalars, labels = Shards([destination])[0]
    assert board.shape == (TILE_PLANES, 10, 10)
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
def test_labels_align_with_the_turn_they_were_recorded_on(tmp_path: Path) -> None:
    """An off-by-one between observation and action trains on the wrong pairing.

    A replay step holds the state *before* its action was interpreted, so the
    action belonging to an observation is recorded on that same index — not the
    next one. Getting this backwards is invisible: the model still trains, it
    just learns to predict the previous turn's decision.
    """
    sample = one_sample()
    destination = tmp_path / "aligned.npz"
    build_shard([sample], destination, stride=719)

    with zipfile.ZipFile(ARCHIVE) as bundle, bundle.open(sample.name) as member:
        episode = json.load(member)

    expected = encode_units(episode["steps"][0][0]["action"] or {})
    _, _, labels = Shards([destination])[0]

    assert torch.equal(labels, expected[0])


def test_a_turn_over_max_units_is_skipped_not_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MAX_UNITS is an empirical bound; one oversized turn must not abort the build.

    Built against a synthetic archive so it runs on every machine, not only one
    with the corpus, and stays fast enough for the default test run.
    """
    from kaggriculture.constants import PRODUCTS
    from kaggriculture.learn import dataset as dataset_module

    def _empty_farm() -> dict:
        return {
            "tiles": [[None] * 10 for _ in range(10)],
            "money": 3000.0,
            "hands": [],
            "unlocked_quadrants": ["NW"],
        }

    def observation() -> dict:
        return {
            "day": 0,
            "hour": 0,
            "step": 0,
            "farms": [
                _empty_farm(),
                _empty_farm(),
            ],
            "market": {
                "prices": dict.fromkeys(PRODUCTS, 100),
                "inventory": dict.fromkeys(PRODUCTS, 10000),
            },
            "town": {"unlocked_shops": []},
        }

    fine_action = {"farmer": ["PASS"], "hands": [], "market": []}
    over_action = {
        "farmer": ["PASS"],
        "hands": [["PASS"] for _ in range(MAX_UNITS)],
        "market": [],
    }
    episode = {
        "steps": [
            [{"action": fine_action, "observation": observation()}, {}],
            [{"action": over_action, "observation": observation()}, {}],
            [{"action": fine_action, "observation": observation()}, {}],
        ]
    }

    archive = tmp_path / "synthetic.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("42.json", json.dumps(episode))
    monkeypatch.setattr(dataset_module, "CORPUS", tmp_path)

    sample = Sample(archive=archive.name, name="42.json", seat=0, rating=2600.0)
    rows = build_shard([sample], tmp_path / "shard.npz", stride=1)

    assert rows == 2
