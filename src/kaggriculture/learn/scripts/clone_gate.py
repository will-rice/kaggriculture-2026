"""The clone gate: can behaviour cloning reproduce our own strongest agent?

Our best agent is a route replayer indexed by the engine's clock -- it plays
action *k* on step *k* -- and that indexing is why it cannot be improved in
place: routes harvested from real games score 0.000 when replayed on another
board, and every single-constant edit to the schedule scores 0.000 because
there is no slack in it. A neural clone is state-conditioned instead, so in
principle it can hold the same strategy while adapting to a board the route was
never solved for.

This script measures one thing and stops. **Distillation cannot exceed its
teacher**, so nothing here is expected to beat anything; the question is only
whether a clone can *match* the teacher, because that decides whether RL from
this initialisation is worth weeks. The interpretation is fixed before any
number exists, in ``VERDICTS`` below, so a result cannot be re-read into a
better one after the fact.

Four stages:

``record``  Play the teacher against a spread of opponents on seeds outside the
            exam bank, both seat orderings, and keep every (observation,
            action) pair the teacher's seat produced.
``train``   Supervised cross-entropy on those pairs, three heads: which op each
            unit takes, how much a transferring unit moves, and what to trade.
            Accuracy is reported on held-out *seasons*, never held-out turns of
            a season the model trained on -- 719 turns of one season are far
            more correlated than they are informative, and splitting inside one
            would report memorisation.
``gate``    Play the clone against the teacher itself over the first 24 exam
            seeds, both seats, on the reference engine. 48 games, Wilson
            interval, and a verdict read off ``VERDICTS``.
``dagger``  Optional, and not part of the pre-registered gate: play the clone
            and label its own states with the teacher's answer, then ``train``
            and ``gate`` again. It separates a policy class that cannot express
            the teacher from one that can but never sees its own mistakes.

Every hyperparameter is a constant rather than a flag, for the reason
``scripts/train.py`` gives: a run whose numbers came from the shell is a run
that cannot be reproduced from the repository.
"""

import argparse
import json
import logging
from dataclasses import dataclass

import torch
from lightning import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

from kaggriculture.learn import CLONE_CHECKPOINT
from kaggriculture.learn.clone import (
    CLONE_AGENT,
    CLONE_DIR,
    HOLDOUT,
    TEACHER,
    Recording,
    TeacherShards,
    build,
    built_shards,
    dagger,
    read_manifest,
    season_slices,
    win_rate,
)
from kaggriculture.learn.encoding import IGNORE
from kaggriculture.learn.model import Policy
from kaggriculture.report import wilson_interval
from kaggriculture.search import arena
from kaggriculture.search.scripts.holdout import GATE_SEEDS

LOGGER = logging.getLogger(__name__)

EPOCHS = 20
BATCH = 512
LEARNING_RATE = 3e-4
SEED = 0
WORKERS = 8

# 24 of the 64 exam seeds, both seat orderings: 48 games. The gate needs to
# separate ~0.5 from <0.1, which 48 games does comfortably (a Wilson interval
# at 0.5 is about +/-0.14 wide), and the remaining 40 seeds stay unspent for
# whatever this result licenses next.
GATE_GAMES_SEEDS = GATE_SEEDS[:24]

# A win rate says which farm banked more; it does not say whether the loser
# banked 40,000 or nothing at all, and those are different findings about the
# same 0.000. `arena.outcomes` keeps only paired margins, so a small block of
# the same games is replayed through `arena.play`, which returns both banks, to
# put the clone's own farming on the record beside the rate.
BANK_SEEDS = GATE_GAMES_SEEDS[:8]


@dataclass(frozen=True)
class Verdict:
    """One band of the gate's pre-registered interpretation."""

    floor: float
    name: str
    reading: str


# Fixed before any result exists. The bands come from the dispatch, not from
# the numbers: a rate that lands between two of them is reported as the lower
# band, because rounding a result upward into a better verdict is exactly the
# move this table exists to prevent.
VERDICTS: tuple[Verdict, ...] = (
    Verdict(
        0.4,
        "ALIVE",
        "the clone reproduces the teacher; RL from this initialisation is the "
        "next step",
    ),
    Verdict(
        0.2,
        "PARTIAL",
        "the clone holds part of the teacher; the per-head accuracies say "
        "which head is failing",
    ),
    Verdict(
        0.1,
        "WEAK",
        "the clone is far below the teacher and above the floor; cloning has "
        "not reproduced the strategy",
    ),
    Verdict(
        0.0,
        "FAILED",
        "cloning does not reproduce the teacher; the direction closes here",
    ),
)


def main() -> None:
    """Run one stage of the gate from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("record", "dagger", "train", "gate"))
    parser.add_argument(
        "--workers", type=int, default=None, help="processes for engine episodes"
    )
    arguments = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if arguments.stage == "record":
        build(CLONE_DIR, arguments.workers)
    elif arguments.stage == "dagger":
        dagger(CLONE_DIR, arguments.workers)
    elif arguments.stage == "train":
        train()
    else:
        gate(arguments.workers)


def train() -> None:
    """Clone the recorded teacher and report per-head accuracy on held-out seasons."""
    seed_everything(SEED, workers=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    training_shards, holdout_shards = built_shards(CLONE_DIR)
    training = TeacherShards(training_shards)
    held = TeacherShards(holdout_shards)
    held_manifest = read_manifest(CLONE_DIR, HOLDOUT)
    LOGGER.info(
        "%d training rows, %d holdout rows over %d seasons, on %s",
        len(training),
        len(held),
        len(held_manifest.seasons),
        device,
    )

    loader = DataLoader(
        training, batch_size=BATCH, shuffle=True, num_workers=WORKERS, pin_memory=True
    )
    holdout = DataLoader(held, batch_size=BATCH, num_workers=WORKERS, pin_memory=True)

    model = Policy().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    history = []
    for epoch in range(EPOCHS):
        model.train()
        for batch in tqdm(loader, desc=f"epoch {epoch}"):
            board, scalars, positions, ops, quantities, market = (
                tensor.to(device, non_blocking=True) for tensor in batch
            )
            op_logits, quantity_logits, market_logits, _value = model(
                board, scalars, positions
            )
            loss = (
                head_loss(op_logits, ops)
                + head_loss(quantity_logits, quantities)
                + head_loss(market_logits, market)
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        measured = accuracy(model, holdout, device)
        history.append({"epoch": epoch, **measured})
        LOGGER.info("epoch %d holdout %s", epoch, json.dumps(measured))

    CLONE_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CLONE_CHECKPOINT)
    LOGGER.info("saved %s", CLONE_CHECKPOINT)

    report = {
        "teacher": TEACHER,
        "training_rows": len(training),
        "holdout_rows": len(held),
        "holdout_seasons": len(held_manifest.seasons),
        "teacher_holdout_win_rate": win_rate(held_manifest),
        "epochs": history,
        "final": history[-1],
        "divergence": divergence(model, held, held_manifest, device),
    }
    (CLONE_DIR / "clone.json").write_text(json.dumps(report, indent=1))
    LOGGER.info("wrote %s", CLONE_DIR / "clone.json")


def head_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return the mean cross entropy over the slots carrying a real label.

    One function serves all three heads because all three are the same
    question -- which row of a fixed vocabulary did the teacher choose -- and
    all three mark an absent decision with ``IGNORE``. A unit slot is
    ``IGNORE`` when the farm has no hand standing there; a quantity slot is
    ``IGNORE`` when the unit's op reads no count; a market slot never is,
    because every one of the 21 slots is a genuine decision on every turn and
    "trade nothing" is the meaningful class 0. Passing ``ignore_index`` on the
    market head therefore changes nothing about it, and a single loss keeps the
    three heads from drifting into three subtly different maskings.

    ``ignore_index`` reads the module's own ``IGNORE`` rather than leaning on
    ``cross_entropy``'s default, which happens to be the same number. Relying
    on that coincidence makes the masking depend on a library default agreeing
    with our vocabulary forever.

    Args:
        logits: ``(batch, slots, options)`` scores.
        labels: ``(batch, slots)`` labels, unchosen slots set to ``IGNORE``.

    Returns:
        A scalar loss averaged over the slots that carry a label.
    """
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), labels.flatten(), ignore_index=IGNORE
    )


def accuracy(model: Policy, loader: DataLoader, device: str) -> dict[str, float]:
    """Return top-1 agreement with the teacher, one number per head.

    The heads are never averaged together: a single combined number would hide
    which one is failing, and which one is failing is the whole content of the
    partial band of ``VERDICTS``.

    Five numbers, because three of them can each be high while the agent is
    still useless. ``op`` and ``quantity`` count only the slots that carry a
    label. ``market`` counts every slot, which is flattering -- most slots hold
    "trade nothing" on most turns -- so ``market_traded`` counts only the slots
    where the teacher actually placed an order, and ``market_row`` counts a
    turn's whole 21-slot market as one prediction that is either right or
    wrong. The engine credits money in exactly one place, a completed SELL, so
    the trading numbers are the ones that decide whether the clone can bank
    anything at all.

    Args:
        model: The policy to score.
        loader: Batches of one recorded row.
        device: Where to run the forward pass.

    Returns:
        Agreement per head, plus the two stricter market readings.
    """
    model.eval()
    counts = {key: [0.0, 0.0] for key in ("op", "quantity", "market", "market_traded")}
    rows = matched = 0.0
    with torch.no_grad():
        for batch in loader:
            board, scalars, positions, ops, quantities, market = (
                tensor.to(device, non_blocking=True) for tensor in batch
            )
            op_logits, quantity_logits, market_logits, _value = model(
                board, scalars, positions
            )
            for key, logits, labels, keep in (
                ("op", op_logits, ops, ops != IGNORE),
                ("quantity", quantity_logits, quantities, quantities != IGNORE),
                (
                    "market",
                    market_logits,
                    market,
                    torch.ones_like(market, dtype=torch.bool),
                ),
                ("market_traded", market_logits, market, market != 0),
            ):
                hit = (logits.argmax(dim=-1) == labels) & keep
                counts[key][0] += float(hit.sum().item())
                counts[key][1] += float(keep.sum().item())
            exact = ((market_logits.argmax(dim=-1) == market).all(dim=-1)).sum()
            matched += float(exact.item())
            rows += float(market.shape[0])
    scores = {key: hit / total for key, (hit, total) in counts.items()}
    scores["market_row"] = matched / rows
    return scores


def divergence(
    model: Policy, held: TeacherShards, manifest: Recording, device: str
) -> dict[str, float]:
    """Return op agreement split by how well the teacher's route did that season.

    The reason to want a state-conditioned policy at all is that the route
    handles some boards badly, so the question worth one extra comparison is
    whether the clone's play *diverges* from the teacher's exactly there. If
    agreement is uniform across good and bad seasons the clone is a faithful
    copy with nothing to add; if it drops on the seasons the teacher lost, the
    clone is generalising to boards the schedule was never solved for -- which
    could be value or could be noise, but is the thing to look at next.

    Seasons are split at the teacher's own result rather than at a margin
    threshold, because winning is the competition's condition and a chosen
    threshold would be one more free parameter.

    Args:
        model: The trained clone.
        held: The held-out rows, in the manifest's own season order.
        manifest: What each held-out season was and how it ended.
        device: Where to run the forward pass.

    Returns:
        Op agreement over won seasons, over lost seasons, and their counts.
    """
    model.eval()
    tallies = {"won": [0.0, 0.0], "lost": [0.0, 0.0]}
    seasons = {"won": 0, "lost": 0}
    with torch.no_grad():
        for season, rows in season_slices(manifest):
            bucket = "won" if season.ours > season.theirs else "lost"
            seasons[bucket] += 1
            board, scalars, positions, ops = (
                tensor[rows].to(device) for tensor in held.tensors[:4]
            )
            for start in range(0, board.shape[0], BATCH):
                window = slice(start, start + BATCH)
                op_logits, _quantity, _market, _value = model(
                    board[window], scalars[window], positions[window]
                )
                keep = ops[window] != IGNORE
                hit = (op_logits.argmax(dim=-1) == ops[window]) & keep
                tallies[bucket][0] += float(hit.sum().item())
                tallies[bucket][1] += float(keep.sum().item())
    return {
        "op_accuracy_won_seasons": tallies["won"][0] / max(tallies["won"][1], 1.0),
        "op_accuracy_lost_seasons": tallies["lost"][0] / max(tallies["lost"][1], 1.0),
        "won_seasons": float(seasons["won"]),
        "lost_seasons": float(seasons["lost"]),
    }


def gate(workers: int | None) -> None:
    """Play the clone against the teacher on the reference engine and report.

    ``arena.outcomes`` plays every seed in both seat orderings, because the
    seats are not symmetric -- they hold different quadrants and their market
    orders pair by queue position -- so scoring one ordering would measure the
    seat as much as the agent. It is also the one place in this project that
    applies the win/tie/loss rule, so this gate cannot disagree with any other
    measurement about what a win is.

    Args:
        workers: Processes to spread the 48 games over, or None for the default.
    """
    if not CLONE_CHECKPOINT.is_file():
        raise FileNotFoundError(f"no clone checkpoint at {CLONE_CHECKPOINT}")
    scores = arena.outcomes(
        CLONE_AGENT, {"teacher": TEACHER}, GATE_GAMES_SEEDS, workers
    )
    if scores.failures:
        raise RuntimeError(f"the engine did not play every game: {scores.failures}")
    rate = sum(scores) / len(scores)
    low, high = wilson_interval(sum(scores), len(scores))
    verdict = next(band for band in VERDICTS if rate >= band.floor)
    clone_banks, teacher_banks = banks(workers)
    report = {
        "clone": CLONE_AGENT,
        "teacher": TEACHER,
        "seeds": list(GATE_GAMES_SEEDS),
        "games": len(scores),
        "wins": sum(scores),
        "rate": rate,
        "wilson_95": [low, high],
        "median_margin": sorted(scores.margins)[len(scores.margins) // 2],
        "clone_mean_bank": sum(clone_banks) / len(clone_banks),
        "teacher_mean_bank": sum(teacher_banks) / len(teacher_banks),
        "clone_banks": clone_banks,
        "verdict": verdict.name,
        "reading": verdict.reading,
        "elapsed_s": scores.runtime_seconds,
    }
    (CLONE_DIR / "gate.json").write_text(json.dumps(report, indent=1))
    LOGGER.info("%s", json.dumps(report, indent=1))


def banks(workers: int | None) -> tuple[list[int], list[int]]:
    """Return both farms' final banks over ``BANK_SEEDS``, in both seat orderings.

    The win rate the gate reports is a comparison, and a comparison cannot tell
    a clone that farmed well and lost narrowly from one that banked nothing at
    all. Both read 0.000. These are the absolute numbers that separate them.

    Args:
        workers: Processes to spread the games over, or None for the default.

    Returns:
        The clone's banks and the teacher's, paired game for game.
    """
    first = arena.play(CLONE_AGENT, TEACHER, BANK_SEEDS, workers)
    second = arena.play(TEACHER, CLONE_AGENT, BANK_SEEDS, workers)
    clone = [ours for ours, _ in first] + [ours for _, ours in second]
    teacher = [theirs for _, theirs in first] + [theirs for theirs, _ in second]
    return clone, teacher


if __name__ == "__main__":
    main()
