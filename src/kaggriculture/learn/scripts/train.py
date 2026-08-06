"""Behaviour-clone the reference policy on the imitation dataset.

Every knob is a constant rather than a flag. A run whose hyperparameters came
from the shell is one whose numbers cannot be reproduced from the repository,
and the checkpoint this writes is measured against the league at a commit --
so the commit has to say what produced it.

The shards are named explicitly. The build writes ``train.npz``,
``train-001.npz`` ... alongside ``holdout.npz`` and its own numbered tail in one
directory, so the obvious ``glob("*.npz")`` would train on the holdout and leave
a validation number that means nothing while looking excellent. ``built_shards``
names what it takes for each and refuses a directory holding an ``.npz`` it
cannot account for -- which is also what catches a leftover shard from an older,
larger build.
"""

import logging
from pathlib import Path

import torch
from lightning import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from kaggriculture.learn.dataset import Shards
from kaggriculture.learn.encoding import IGNORE, UNIT_OPS
from kaggriculture.learn.model import Policy
from kaggriculture.learn.scripts.build import HOLDOUT, SHARDS, TRAIN
from kaggriculture.learn.scripts.play import CHECKPOINT
from kaggriculture.scripts.tracking import ENTITY, PROJECT, commit

LOGGER = logging.getLogger(__name__)

EPOCHS = 8
BATCH = 512
LEARNING_RATE = 3e-4
SEED = 0
WORKERS = 8


def main() -> None:
    """Train the policy on the built shards and save the checkpoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    seed_everything(SEED, workers=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    shards, holdout_shards = built_shards(SHARDS)
    training = Shards(shards)
    held = Shards(holdout_shards)
    LOGGER.info(
        "%d rows over %d shards, %d holdout rows, on %s",
        len(training),
        len(shards),
        len(held),
        device,
    )

    loader = DataLoader(
        training, batch_size=BATCH, shuffle=True, num_workers=WORKERS, pin_memory=True
    )
    holdout = DataLoader(held, batch_size=BATCH, num_workers=WORKERS, pin_memory=True)

    model = Policy().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    parameters = sum(p.numel() for p in model.parameters())
    revision = commit()
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        job_type="behaviour-cloning",
        name=f"bc-{parameters // 1_000_000}M-seed{SEED}-{revision}",
        config={
            "epochs": EPOCHS,
            "batch": BATCH,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "rows": len(training),
            "holdout_rows": len(held),
            "shards": [path.name for path in shards],
            "parameters": parameters,
            "ops": len(UNIT_OPS),
            "commit": revision,
        },
    )

    for epoch in range(EPOCHS):
        model.train()
        total = correct = counted = 0.0
        for board, scalars, positions, labels, _market in tqdm(
            loader, desc=f"epoch {epoch}"
        ):
            labels = labels.to(device)
            logits, _market_logits = model(
                board.to(device), scalars.to(device), positions.to(device)
            )
            loss = unit_loss(logits, labels)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            acting = int((labels != IGNORE).sum().item())
            total += loss.item() * acting
            correct += float(
                ((logits.argmax(dim=-1) == labels) & (labels != IGNORE)).sum().item()
            )
            counted += acting
            run.log({"loss/step": loss.item()})

        held_loss, held_accuracy = evaluate(model, holdout, device)
        LOGGER.info(
            "epoch %d train loss %.4f accuracy %.4f | holdout loss %.4f accuracy %.4f",
            epoch,
            total / counted,
            correct / counted,
            held_loss,
            held_accuracy,
        )
        run.log(
            {
                "epoch": epoch,
                "loss/train": total / counted,
                "accuracy/train": correct / counted,
                "loss/holdout": held_loss,
                "accuracy/holdout": held_accuracy,
            }
        )

    torch.save(model.state_dict(), CHECKPOINT)
    LOGGER.info("saved %s", CHECKPOINT)
    run.finish()


def shards_named(directory: Path, stem: str) -> list[Path]:
    """Return one build's shards under ``stem``, first shard first.

    A build writes ``<stem>.npz`` and then ``<stem>-001.npz`` onwards once it
    passes the row cap, so no single glob describes one set without also
    describing the other's. Both training and holdout are written this way: the
    holdout is smaller and currently fits one file, but it is the same builder
    and it will split as the nightly corpus grows.

    Args:
        directory: Where ``build.py`` wrote its shards.
        stem: The shard family to collect, ``TRAIN`` or ``HOLDOUT``.

    Returns:
        The shards under ``stem``, first shard first.

    Raises:
        FileNotFoundError: If the first shard is missing.
    """
    first = directory / f"{stem}.npz"
    if not first.is_file():
        raise FileNotFoundError(f"no {stem}.npz in {directory}")
    return [first, *sorted(directory.glob(f"{stem}-[0-9][0-9][0-9].npz"))]


def built_shards(directory: Path) -> tuple[list[Path], list[Path]]:
    """Return the training and holdout shards, refusing anything unaccounted for.

    Selecting by name and then rejecting any ``.npz`` the selection cannot
    explain makes both failures loud: a holdout swept into training, and a shard
    silently left out of it. A leftover file from an older, larger build has the
    same shapes as a real shard and would otherwise concatenate in silence.

    Args:
        directory: Where ``build.py`` wrote its shards.

    Returns:
        The training shards and the holdout shards.

    Raises:
        ValueError: If the directory holds an ``.npz`` belonging to neither.
    """
    training = shards_named(directory, TRAIN)
    held = shards_named(directory, HOLDOUT)
    unaccounted = set(directory.glob("*.npz")) - set(training) - set(held)
    if unaccounted:
        raise ValueError(
            f"{directory} holds {sorted(path.name for path in unaccounted)}, which "
            f"belongs to neither the {TRAIN} nor the {HOLDOUT} shards"
        )
    return training, held


def unit_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return the mean cross entropy over the slots that hold a real unit.

    Most slots of most rows are padding -- a full farm is rare and ``MAX_UNITS``
    has headroom above the widest crew in the corpus -- so a loss that scored
    every slot would be dominated by slots holding nobody and would train the
    model to emit padding.

    ``ignore_index`` is passed explicitly and reads the module's ``IGNORE``
    rather than leaning on ``cross_entropy``'s default, which happens to be the
    same number. Relying on the coincidence would make the masking depend on a
    library default agreeing with our vocabulary forever, and it is exactly the
    kind of agreement that hides until the day one of them moves.

    Args:
        logits: ``(batch, MAX_UNITS, len(UNIT_OPS))`` per-unit scores.
        labels: ``(batch, MAX_UNITS)`` labels, padded slots set to ``IGNORE``.

    Returns:
        A scalar loss averaged over acting units.
    """
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), labels.flatten(), ignore_index=IGNORE
    )


def evaluate(model: Policy, loader: DataLoader, device: str) -> tuple[float, float]:
    """Return mean loss and top-1 accuracy over acting units.

    Both are weighted by how many units actually acted in each batch, not by
    batch count, so a short final batch does not count for as much as a full
    one. Padded slots are excluded from both numbers for the same reason they
    are excluded from the loss: predicting an absent hand's op is not a skill.

    The market labels ride along in every batch but are not yet scored here:
    the policy now has a market head, but no loss reads it yet, so its logits
    are computed and discarded on every call.

    Args:
        model: The policy to score.
        loader: Batches of ``(board, scalars, positions, labels, market)``.
        device: Where to run the forward pass.

    Returns:
        Mean cross entropy and top-1 accuracy, both over acting units.
    """
    model.eval()
    total = correct = counted = 0.0
    with torch.no_grad():
        for board, scalars, positions, labels, _market in loader:
            labels = labels.to(device)
            logits, _market_logits = model(
                board.to(device), scalars.to(device), positions.to(device)
            )
            acting = int((labels != IGNORE).sum().item())
            total += unit_loss(logits, labels).item() * acting
            correct += float(
                ((logits.argmax(dim=-1) == labels) & (labels != IGNORE)).sum().item()
            )
            counted += acting
    return total / counted, correct / counted


if __name__ == "__main__":
    main()
