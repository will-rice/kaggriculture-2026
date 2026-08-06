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
from kaggriculture.learn import CHECKPOINT
from kaggriculture.learn.dataset import Shards
from kaggriculture.learn.encoding import IGNORE, MARKET_SLOTS, UNIT_OPS
from kaggriculture.learn.model import Policy
from kaggriculture.learn.scripts.build import HOLDOUT, SHARDS, TRAIN
from kaggriculture.scripts.tracking import ENTITY, PROJECT, commit

LOGGER = logging.getLogger(__name__)

EPOCHS = 8
BATCH = 512
LEARNING_RATE = 3e-4
SEED = 0
WORKERS = 8

# The unit head sees roughly four real per-row targets (one turn's acting
# units) against the market head's len(MARKET_SLOTS) + 2 = 21 slots, most of
# which sit at bucket 0. Summed unweighted, the wider, mostly-zero head would
# dominate the gradient and the narrower one would starve. 1.0 is a starting
# point measured against nothing yet -- it has not been tuned against a run --
# and the first run's two accuracy curves, not this comment, are the evidence
# for changing it.
MARKET_WEIGHT = 1.0


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
            "market_slots": len(MARKET_SLOTS) + 2,
            "market_weight": MARKET_WEIGHT,
            "commit": revision,
        },
    )

    for epoch in range(EPOCHS):
        model.train()
        unit_total = unit_correct = counted = 0.0
        market_total = market_correct = slotted = 0.0
        for board, scalars, positions, labels, market in tqdm(
            loader, desc=f"epoch {epoch}"
        ):
            labels = labels.to(device)
            market = market.to(device)
            logits, market_logits = model(
                board.to(device), scalars.to(device), positions.to(device)
            )
            units, trades = (
                unit_loss(logits, labels),
                market_loss(market_logits, market),
            )
            loss = units + MARKET_WEIGHT * trades
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            acting = int((labels != IGNORE).sum().item())
            unit_total += units.item() * acting
            unit_correct += float(
                ((logits.argmax(dim=-1) == labels) & (labels != IGNORE)).sum().item()
            )
            counted += acting

            slots = market.numel()
            market_total += trades.item() * slots
            market_correct += float(
                (market_logits.argmax(dim=-1) == market).sum().item()
            )
            slotted += slots

            run.log(
                {
                    "loss/step": loss.item(),
                    "loss/units/step": units.item(),
                    "loss/market/step": trades.item(),
                }
            )

        train_metrics = {
            "loss/units": unit_total / counted,
            "accuracy/units": unit_correct / counted,
            "loss/market": market_total / slotted,
            "accuracy/market": market_correct / slotted,
        }
        held_metrics = evaluate(model, holdout, device)
        LOGGER.info(
            "epoch %d train units loss %.4f accuracy %.4f market loss %.4f "
            "accuracy %.4f | holdout units loss %.4f accuracy %.4f market loss "
            "%.4f accuracy %.4f",
            epoch,
            train_metrics["loss/units"],
            train_metrics["accuracy/units"],
            train_metrics["loss/market"],
            train_metrics["accuracy/market"],
            held_metrics["loss/units"],
            held_metrics["accuracy/units"],
            held_metrics["loss/market"],
            held_metrics["accuracy/market"],
        )
        run.log(
            {
                "epoch": epoch,
                **{f"train/{key}": value for key, value in train_metrics.items()},
                **{f"holdout/{key}": value for key, value in held_metrics.items()},
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


def market_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return the mean cross entropy over every market slot.

    Unlike ``unit_loss``, this passes no ``ignore_index`` at all: every market
    slot -- sell this product, buy that seed, hire, buy land -- is a real
    decision on every turn, and there is no slot analogous to a hand not yet
    hired. Bucket 0 means "trade nothing", and the teacher genuinely chooses it
    on 35.9% of training rows and 34.0% of holdout rows; masking it the way
    ``unit_loss`` masks padding would teach the model that not trading is
    unobserved rather than chosen.

    That is lower than the 46.3% of raw corpus turns quoted in
    ``encode_market``, and the two are not in conflict: these rows are filtered
    by ladder rating and strided, and the stronger agents that survive the
    filter trade more often. Each figure names its own population because an
    unqualified one invites exactly the "which is it?" the two used to provoke.

    ``IGNORE`` is -100, which happens to also be ``cross_entropy``'s own
    default ``ignore_index``. A test that calls ``cross_entropy`` directly with
    unlabelled ``ignore_index`` would therefore pass whether or not masking
    ever happened -- it proves nothing about this function -- so the guard has
    to call ``market_loss`` itself.

    Args:
        logits: ``(batch, len(MARKET_SLOTS) + 2, len(QUANTITIES))`` per-slot
            scores.
        labels: ``(batch, len(MARKET_SLOTS) + 2)`` bucket labels; every entry
            is a genuine target, none of them padding.

    Returns:
        A scalar loss averaged over every slot of every row.
    """
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten())


def evaluate(model: Policy, loader: DataLoader, device: str) -> dict[str, float]:
    """Return mean loss and top-1 accuracy for each head, kept separate.

    A single combined number would hide which of the two heads is failing, so
    the two are never averaged together. Both are weighted by how many slots
    actually contributed -- acting units for the unit head, every market slot
    for the market head -- not by batch count, so a short final batch does not
    count for as much as a full one. Padded unit slots are excluded from the
    unit numbers for the same reason they are excluded from ``unit_loss``:
    predicting an absent hand's op is not a skill. No market slot is excluded,
    for the same reason ``market_loss`` masks nothing.

    Args:
        model: The policy to score.
        loader: Batches of ``(board, scalars, positions, labels, market)``.
        device: Where to run the forward pass.

    Returns:
        A dict with ``loss/units``, ``accuracy/units``, ``loss/market`` and
        ``accuracy/market``.
    """
    model.eval()
    unit_total = unit_correct = counted = 0.0
    market_total = market_correct = slotted = 0.0
    with torch.no_grad():
        for board, scalars, positions, labels, market in loader:
            labels = labels.to(device)
            market = market.to(device)
            logits, market_logits = model(
                board.to(device), scalars.to(device), positions.to(device)
            )
            acting = int((labels != IGNORE).sum().item())
            unit_total += unit_loss(logits, labels).item() * acting
            unit_correct += float(
                ((logits.argmax(dim=-1) == labels) & (labels != IGNORE)).sum().item()
            )
            counted += acting

            slots = market.numel()
            market_total += market_loss(market_logits, market).item() * slots
            market_correct += float(
                (market_logits.argmax(dim=-1) == market).sum().item()
            )
            slotted += slots
    return {
        "loss/units": unit_total / counted,
        "accuracy/units": unit_correct / counted,
        "loss/market": market_total / slotted,
        "accuracy/market": market_correct / slotted,
    }


if __name__ == "__main__":
    main()
