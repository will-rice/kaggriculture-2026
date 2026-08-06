"""Tests for shard selection and the masked objective behaviour cloning trains on."""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from kaggriculture.learn.encoding import (
    BOARD,
    IGNORE,
    MARKET_SLOTS,
    MAX_UNITS,
    SCALARS,
    TILE_PLANES,
    UNIT_OPS,
)
from kaggriculture.learn.model import Policy
from kaggriculture.learn.scripts import train as train_module
from kaggriculture.learn.scripts.train import built_shards, evaluate, unit_loss

ACTING = 3


def _shards(directory: Path) -> None:
    """Write the file names a real build leaves behind, contents unread."""
    for name in ["train.npz", "holdout.npz"] + [
        f"train-{index:03d}.npz" for index in range(1, 8)
    ]:
        (directory / name).touch()


def test_built_shards_takes_every_shard_and_never_the_holdout(tmp_path: Path) -> None:
    """The holdout sits in the same directory under a name no glob excludes for free.

    ``train.npz`` carries no index and the rest do, so the pattern that
    describes them all also describes ``holdout.npz``. Training on the holdout
    would not raise or look wrong; it would report an excellent validation
    number that means nothing, which is the failure this names.
    """
    _shards(tmp_path)

    training, held = built_shards(tmp_path)

    assert training == [tmp_path / "train.npz"] + [
        tmp_path / f"train-{index:03d}.npz" for index in range(1, 8)
    ]
    assert held == [tmp_path / "holdout.npz"]


def test_built_shards_reads_a_holdout_that_has_split(tmp_path: Path) -> None:
    """The holdout is one file today only because it fits under the row cap.

    It is written by the same builder as the training shards, so the nightly
    corpus will eventually push it past ``ROWS_PER_SHARD`` and it will grow a
    numbered tail. Rejecting that tail would stop training with a confusing
    error; silently dropping it would shrink the validation set without saying
    so.
    """
    _shards(tmp_path)
    (tmp_path / "holdout-001.npz").touch()

    training, held = built_shards(tmp_path)

    assert held == [tmp_path / "holdout.npz", tmp_path / "holdout-001.npz"]
    assert tmp_path / "holdout-001.npz" not in training


def test_built_shards_refuses_a_shard_it_cannot_account_for(tmp_path: Path) -> None:
    """A shard left out of training is as silent as a holdout swept into it."""
    _shards(tmp_path)
    (tmp_path / "train-extra.npz").touch()

    with pytest.raises(ValueError, match="train-extra.npz"):
        built_shards(tmp_path)


def test_built_shards_refuses_a_directory_with_no_first_shard(tmp_path: Path) -> None:
    """An empty directory must say so, not train on nothing."""
    with pytest.raises(FileNotFoundError, match="train.npz"):
        built_shards(tmp_path)


def test_the_loss_masks_the_slots_ignore_names_rather_than_torch_s_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``IGNORE`` is -100, and so is ``cross_entropy``'s default ``ignore_index``.

    That coincidence makes the obvious masking test worthless: it passes
    whether or not the argument is passed, because torch masks -100 anyway. So
    this test moves ``IGNORE`` to a real class index and asserts the loss
    follows it. Code that leans on the default scores those slots as class
    zero, and the perturbation below moves the loss -- which the final
    assertion measures directly, so the guard cannot pass by both sides being
    equal for some unrelated reason.
    """
    sentinel = 0
    monkeypatch.setattr(train_module, "IGNORE", sentinel)
    torch.manual_seed(0)
    logits = torch.randn(4, MAX_UNITS, len(UNIT_OPS))
    labels = torch.full((4, MAX_UNITS), sentinel, dtype=torch.int64)
    labels[:, :ACTING] = 1

    perturbed = logits.clone()
    perturbed[:, ACTING:, :] = torch.randn(4, MAX_UNITS - ACTING, len(UNIT_OPS))

    assert torch.equal(unit_loss(logits, labels), unit_loss(perturbed, labels))
    assert not torch.equal(
        torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), labels.flatten(), ignore_index=IGNORE
        ),
        torch.nn.functional.cross_entropy(
            perturbed.flatten(0, 1), labels.flatten(), ignore_index=IGNORE
        ),
    )


def test_the_loss_averages_over_acting_units_and_not_over_slots() -> None:
    """A mean taken over every slot is a mean over mostly padding.

    ``MAX_UNITS`` is 20 and a typical turn acts three units, so a loss divided
    by 20 rather than by 3 is off by most of its magnitude and its gradient
    points at whatever the padded slots happen to hold.
    """
    torch.manual_seed(0)
    logits = torch.randn(4, MAX_UNITS, len(UNIT_OPS))
    labels = torch.full((4, MAX_UNITS), IGNORE, dtype=torch.int64)
    labels[:, :ACTING] = torch.randint(0, len(UNIT_OPS), (4, ACTING))

    acting_only = torch.nn.functional.cross_entropy(
        logits[:, :ACTING].flatten(0, 1), labels[:, :ACTING].flatten()
    )
    every_slot = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), labels.clamp(min=0).flatten()
    )

    assert torch.allclose(unit_loss(logits, labels), acting_only)
    assert not torch.allclose(unit_loss(logits, labels), every_slot)


def test_evaluate_scores_only_the_units_that_acted() -> None:
    """Accuracy over padded slots measures how well we predict nobody.

    The expected figures are computed from the model's own predictions rather
    than assumed, so this asserts what ``evaluate`` counts, not how well an
    untrained network guesses.
    """
    torch.manual_seed(0)
    model = Policy(blocks=1, channels=8).eval()
    rows = 6
    board = torch.randn(rows, TILE_PLANES, BOARD, BOARD)
    scalars = torch.randn(rows, SCALARS)
    positions = torch.randint(0, BOARD * BOARD, (rows, MAX_UNITS))
    with torch.no_grad():
        predicted = model(board, scalars, positions).argmax(dim=-1)

    # Half the acting slots agree with the model and half deliberately do not,
    # so the answer is exactly 0.5 by construction. Counting the padded slots
    # too would divide by 120 instead of 18 and could only land on 0.5 by
    # coincidence.
    labels = torch.full((rows, MAX_UNITS), IGNORE, dtype=torch.int64)
    labels[:, :ACTING] = predicted[:, :ACTING]
    labels[rows // 2 :, :ACTING] = (predicted[rows // 2 :, :ACTING] + 1) % len(UNIT_OPS)
    market = torch.zeros(rows, len(MARKET_SLOTS) + 2, dtype=torch.int64)
    loader = DataLoader(
        TensorDataset(board, scalars, positions, labels, market), batch_size=4
    )

    with torch.no_grad():
        logits = model(board, scalars, positions)

    loss, accuracy = evaluate(model, loader, "cpu")

    assert accuracy == pytest.approx(0.5)
    assert loss == pytest.approx(float(unit_loss(logits, labels)), rel=1e-5)
