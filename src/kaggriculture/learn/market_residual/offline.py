"""Offline training for the market residual, from counterfactual shards.

The dataset is a set of *cells* — one recorded season each — and inside every
cell a handful of events carry counterfactual labels: the frozen controller's
own play (the anchor) and each scored replacement, with the paired win-point
and margin deltas a finished branch measured. The head is recurrent, so a cell
is trained as the whole ordered event sequence the served runtime would see,
and labels attach to positions inside it; cells are shuffled between batches,
event order inside a cell never is.

Five loss terms, all reported separately and summed unweighted:

* **Ranking.** Pairwise logistic loss over one event's arms, ordered by
  ``win_point_delta + 0.1 * margin_delta``, scored by the policy's own joint
  log-probability of playing each arm. The anchor is an arm like any other, so
  outranking the controller is something a replacement must earn.
* **Q.** Huber regression of the value head to the taught action's combined
  delta — zero when the target is deferring, the chosen arm's delta otherwise.
* **Mode.** Cross-entropy pulling the mode head to ``REPLACE`` only where the
  best arm's *win-point* delta clears the ``0.01`` indifference band, and to
  ``USE_KAITO`` everywhere the evidence is weaker. This band, at training
  time, is the approved mechanism holding exported activation inside its band;
  nothing downstream may clamp the decision after the fact.
* **Quantity.** Cross-entropy to the taught arm's buckets, only on events
  whose target is ``REPLACE``.
* **Auxiliary.** Huber regression of the auxiliary head to the *next* event's
  per-product price, book inventory, and opponent-supply bucket, on the same
  normalized scales the feature row itself carries.

Rows that flip a lost season into a won one (or the reverse) carry ``4x``
weight in every term they touch.

The quantity mask collated here is all-true on purpose: the served runtime
applies no legality mask — an illegal argmax costs a fail-closed deferral at
the merge — and a head trained under a mask the runtime cannot apply would be
gated as one policy and served as another. The field exists so the loss code
and Task 9 share one signature when a mask is worth having.
"""

import hashlib
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields, replace
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy
import torch
from tqdm import tqdm

from kaggriculture.features import PRODUCT_NAMES, encode_observation
from kaggriculture.learn.market_residual.alternatives import AlternativeConfig
from kaggriculture.learn.market_residual.artifacts import (
    CounterfactualRow,
    ShardIdentity,
    load_shard_strict,
    shard_path,
)
from kaggriculture.learn.market_residual.counterfactual import (
    CounterfactualIntegrityError,
    SeasonIdentity,
    canonical_digest,
    record_season,
    season_events,
)
from kaggriculture.learn.market_residual.model import (
    MarketResidualNet,
    PolicyHeads,
    log_probability,
    mask_quantity_logits,
)
from kaggriculture.market_residual.actions import ResidualMode
from kaggriculture.market_residual.baseline import SERVED
from kaggriculture.market_residual.events import EventConfig
from kaggriculture.market_residual.features import (
    MarketFeatureVector,
    market_feature_vector,
)
from kaggriculture.market_residual.numpy_policy import MODES, model_digest
from kaggriculture.market_residual.schema import BUCKETS

USE_KAITO_INDEX = MODES.index(ResidualMode.USE_KAITO)
REPLACE_INDEX = MODES.index(ResidualMode.REPLACE)

# A replacement is taught as the mode target only when its measured win-point
# delta clears this band; anything weaker is taught as deferring to the
# controller. This is the training-time mechanism that holds exported
# activation down, per the carried ruling: never clamp the decision post hoc.
INDIFFERENCE_BAND = 0.01

# The margin tiebreak's weight in the combined per-arm target.
MARGIN_WEIGHT = 0.1

# Rows that flip a lost season into a won one, or the reverse, carry this
# weight in the ranking pairs and event terms they touch.
FLIP_WEIGHT = 4.0

# The exported head must replace on this share of held-out events: seldom
# enough to stay a residual, often enough to exist at all.
ACTIVATION_BAND = (0.01, 0.50)

# Held-out pairwise ranking accuracy the synthetic price-crossing gate demands.
SYNTHETIC_RANKING_GATE = 0.95

# Share of held-out profitable events where the head must rank some profitable
# replacement above the controller's own play, and the share of equal-outcome
# events where its greedy decision must still defer.
EXPLOIT_GATE = 0.75
RETENTION_GATE = 0.5

LAST_CHECKPOINT = "last.ckpt"
BEST_CHECKPOINT = "best.ckpt"
REPORT_NAME = "offline-report.json"


class PretrainDriftError(RuntimeError):
    """Raised when a resume meets a root trained under different terms."""


class PretrainGateError(RuntimeError):
    """Raised when a finished pretraining fails a declared gate."""


class PretrainNonFiniteError(RuntimeError):
    """Raised the moment any head or loss stops being finite."""


@dataclass(frozen=True)
class PretrainConfig:
    """Everything one pretraining run is parameterised by.

    Every field is part of the drift binding: a resume under any changed value
    is refused rather than silently continued.
    """

    epochs: int = 20
    batch_cells: int = 32
    learning_rate: float = 1e-3
    gradient_clip: float = 1.0
    seed: int = 0
    device: str = "cuda"

    def __post_init__(self) -> None:
        """Refuse a run that could never produce a checkpoint.

        Raises:
            ValueError: If there is not at least one epoch.
        """
        if self.epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {self.epochs}")


@dataclass(frozen=True)
class EventSequence:
    """One cell: every event the season opened, and the labels a shard holds.

    Arms are grouped by event, anchor first within its group, and
    ``arm_event`` ascends: collation and the gates rely on that order.
    """

    identity_sha256: str
    seed: int
    seat: int
    features: torch.Tensor
    auxiliary_targets: torch.Tensor
    auxiliary_valid: torch.Tensor
    arm_event: torch.Tensor
    arm_mode: torch.Tensor
    arm_buckets: torch.Tensor
    arm_win_delta: torch.Tensor
    arm_q: torch.Tensor
    arm_flip: torch.Tensor


@dataclass(frozen=True)
class OfflineBatch:
    """Padded event sequences and every index the loss terms consume.

    ``arm_*`` tensors are flat over every labeled arm in the batch;
    ``pair_better``/``pair_worse`` index into them and never pair arms of two
    different events. ``event_*`` tensors are flat over labeled events and
    carry the taught targets collation derived from the arms.
    """

    features: torch.Tensor
    valid: torch.Tensor
    mask: torch.Tensor
    auxiliary_target: torch.Tensor
    auxiliary_valid: torch.Tensor
    arm_step: torch.Tensor
    arm_batch: torch.Tensor
    arm_mode: torch.Tensor
    arm_buckets: torch.Tensor
    arm_win_delta: torch.Tensor
    arm_q: torch.Tensor
    pair_better: torch.Tensor
    pair_worse: torch.Tensor
    pair_weight: torch.Tensor
    event_step: torch.Tensor
    event_batch: torch.Tensor
    mode_target: torch.Tensor
    quantity_target: torch.Tensor
    q_target: torch.Tensor
    event_weight: torch.Tensor

    def to(self, device: torch.device | str) -> "OfflineBatch":
        """Return this batch with every tensor on one device.

        Args:
            device: Where the model runs.

        Returns:
            The moved batch.
        """
        return replace(
            self,
            **{
                field.name: getattr(self, field.name).to(device)
                for field in fields(self)
            },
        )


@dataclass(frozen=True)
class OfflineLoss:
    """Every term of one offline update, and what the heads looked like.

    The tensors stay on the graph; the floats are detached measurements.
    ``total`` is the unweighted sum of the five terms.
    """

    total: torch.Tensor
    ranking: torch.Tensor
    q: torch.Tensor
    mode: torch.Tensor
    quantity: torch.Tensor
    auxiliary: torch.Tensor
    ranking_accuracy: float
    activation: float
    calibration: float
    finite: bool

    def metrics(self) -> dict[str, float]:
        """Return every term and measurement as plain floats."""
        return {
            "total": float(self.total.item()),
            "ranking": float(self.ranking.item()),
            "q": float(self.q.item()),
            "mode": float(self.mode.item()),
            "quantity": float(self.quantity.item()),
            "auxiliary": float(self.auxiliary.item()),
            "ranking_accuracy": self.ranking_accuracy,
            "activation": self.activation,
            "calibration": self.calibration,
        }


def collate_event_sequences(
    sequences: Sequence[EventSequence],
) -> OfflineBatch:
    """Pack cells into one padded batch and derive every taught target.

    Cells become batch columns in the order given; events keep their recurrent
    order inside each column. Per labeled event, the taught action is the arm
    with the highest combined delta — the controller's own play on ties — and
    it becomes a ``REPLACE`` target only when its win-point delta clears the
    indifference band.

    Args:
        sequences: The cells of one batch.

    Returns:
        The batch.

    Raises:
        ValueError: If a labeled event's first arm is not the anchor.
    """
    steps = max(sequence.features.shape[0] for sequence in sequences)
    width = sequences[0].features.shape[1]
    auxiliary_width = sequences[0].auxiliary_targets.shape[1]
    slots = sequences[0].arm_buckets.shape[1] if sequences else 0
    features = torch.zeros(steps, len(sequences), width)
    valid = torch.zeros(steps, len(sequences), dtype=torch.bool)
    auxiliary_target = torch.zeros(steps, len(sequences), auxiliary_width)
    auxiliary_valid = torch.zeros(steps, len(sequences), dtype=torch.bool)

    arm_step: list[torch.Tensor] = []
    arm_batch: list[torch.Tensor] = []
    pair_better: list[int] = []
    pair_worse: list[int] = []
    pair_weight: list[float] = []
    event_step: list[int] = []
    event_batch: list[int] = []
    mode_target: list[int] = []
    quantity_target: list[torch.Tensor] = []
    q_target: list[float] = []
    event_weight: list[float] = []

    offset = 0
    for column, sequence in enumerate(sequences):
        events = sequence.features.shape[0]
        features[:events, column] = sequence.features
        valid[:events, column] = True
        auxiliary_target[:events, column] = sequence.auxiliary_targets
        auxiliary_valid[:events, column] = sequence.auxiliary_valid
        arm_step.append(sequence.arm_event)
        arm_batch.append(torch.full_like(sequence.arm_event, column))
        for group in _event_groups(sequence.arm_event):
            if int(sequence.arm_mode[group[0]]) != USE_KAITO_INDEX:
                raise ValueError(
                    f"labeled event {int(sequence.arm_event[group[0]])} of cell "
                    f"{sequence.identity_sha256} does not lead with the anchor"
                )
            quality = sequence.arm_q[group]
            chosen = group[int((quality == quality.max()).nonzero()[0])]
            replacing = (
                int(sequence.arm_mode[chosen]) == REPLACE_INDEX
                and float(sequence.arm_win_delta[chosen]) > INDIFFERENCE_BAND
            )
            event_step.append(int(sequence.arm_event[chosen]))
            event_batch.append(column)
            mode_target.append(REPLACE_INDEX if replacing else USE_KAITO_INDEX)
            quantity_target.append(
                sequence.arm_buckets[chosen if replacing else group[0]]
            )
            q_target.append(float(sequence.arm_q[chosen]) if replacing else 0.0)
            event_weight.append(
                FLIP_WEIGHT if replacing and bool(sequence.arm_flip[chosen]) else 1.0
            )
            for better in group:
                for worse in group:
                    if float(sequence.arm_q[better]) > float(sequence.arm_q[worse]):
                        pair_better.append(offset + better)
                        pair_worse.append(offset + worse)
                        pair_weight.append(
                            FLIP_WEIGHT
                            if bool(sequence.arm_flip[better])
                            or bool(sequence.arm_flip[worse])
                            else 1.0
                        )
        offset += len(sequence.arm_event)

    return OfflineBatch(
        features=features,
        valid=valid,
        mask=torch.ones(steps, len(sequences), slots, BUCKETS, dtype=torch.bool),
        auxiliary_target=auxiliary_target,
        auxiliary_valid=auxiliary_valid,
        arm_step=torch.cat(arm_step),
        arm_batch=torch.cat(arm_batch),
        arm_mode=torch.cat([sequence.arm_mode for sequence in sequences]),
        arm_buckets=torch.cat([sequence.arm_buckets for sequence in sequences]),
        arm_win_delta=torch.cat([sequence.arm_win_delta for sequence in sequences]),
        arm_q=torch.cat([sequence.arm_q for sequence in sequences]),
        pair_better=torch.tensor(pair_better, dtype=torch.long),
        pair_worse=torch.tensor(pair_worse, dtype=torch.long),
        pair_weight=torch.tensor(pair_weight),
        event_step=torch.tensor(event_step, dtype=torch.long),
        event_batch=torch.tensor(event_batch, dtype=torch.long),
        mode_target=torch.tensor(mode_target, dtype=torch.long),
        quantity_target=(
            torch.stack(quantity_target)
            if quantity_target
            else torch.zeros(0, slots, dtype=torch.long)
        ),
        q_target=torch.tensor(q_target),
        event_weight=torch.tensor(event_weight),
    )


def _event_groups(arm_event: torch.Tensor) -> list[list[int]]:
    """Split one cell's arm rows into runs sharing an event.

    Args:
        arm_event: Ascending event positions, one per arm.

    Returns:
        Consecutive index runs, one per labeled event.
    """
    groups: list[list[int]] = []
    for index, event in enumerate(arm_event.tolist()):
        if groups and arm_event[groups[-1][0]] == event:
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups


def offline_loss(heads: PolicyHeads, batch: OfflineBatch) -> OfflineLoss:
    """Return every offline term for one forward pass over one batch.

    Args:
        heads: The model outputs for ``batch.features``.
        batch: The collated cells, on the same device as the heads.

    Returns:
        The loss report; ``total`` carries the graph.
    """
    zero = heads.value.new_zeros(())
    scores = log_probability(
        PolicyHeads(
            mode_logits=heads.mode_logits[batch.arm_step, batch.arm_batch],
            quantity_logits=heads.quantity_logits[batch.arm_step, batch.arm_batch],
            value=heads.value[batch.arm_step, batch.arm_batch],
            auxiliary=heads.auxiliary[batch.arm_step, batch.arm_batch],
        ),
        batch.mask[batch.arm_step, batch.arm_batch],
        batch.arm_mode,
        batch.arm_buckets,
    )
    if len(batch.pair_better):
        margins = scores[batch.pair_better] - scores[batch.pair_worse]
        ranking = (
            torch.nn.functional.softplus(-margins) * batch.pair_weight
        ).sum() / batch.pair_weight.sum()
        ranking_accuracy = float((margins > 0).float().mean().item())
    else:
        ranking, ranking_accuracy = zero, 1.0

    if len(batch.event_step):
        event_modes = heads.mode_logits[batch.event_step, batch.event_batch]
        mode = _weighted_mean(
            torch.nn.functional.cross_entropy(
                event_modes, batch.mode_target, reduction="none"
            ),
            batch.event_weight,
        )
        value = heads.value[batch.event_step, batch.event_batch]
        q = _weighted_mean(
            torch.nn.functional.huber_loss(value, batch.q_target, reduction="none"),
            batch.event_weight,
        )
        calibration = float((value - batch.q_target).abs().mean().item())
        replacing = batch.mode_target == REPLACE_INDEX
        if bool(replacing.any()):
            logits = mask_quantity_logits(
                heads.quantity_logits[
                    batch.event_step[replacing], batch.event_batch[replacing]
                ],
                batch.mask[batch.event_step[replacing], batch.event_batch[replacing]],
            )
            per_slot = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1),
                batch.quantity_target[replacing].flatten(),
                reduction="none",
            ).reshape(logits.shape[:2])
            quantity = _weighted_mean(
                per_slot.mean(dim=-1), batch.event_weight[replacing]
            )
        else:
            quantity = zero
    else:
        mode = q = quantity = zero
        calibration = 0.0

    auxiliary_errors = torch.nn.functional.huber_loss(
        heads.auxiliary, batch.auxiliary_target, reduction="none"
    ).mean(dim=-1)
    auxiliary = (
        auxiliary_errors[batch.auxiliary_valid].mean()
        if bool(batch.auxiliary_valid.any())
        else zero
    )

    activation = float(
        (heads.mode_logits.argmax(dim=-1) == REPLACE_INDEX)[batch.valid]
        .float()
        .mean()
        .item()
    )
    finite = all(
        bool(torch.isfinite(tensor[batch.valid]).all())
        for tensor in (
            heads.mode_logits,
            heads.quantity_logits,
            heads.value,
            heads.auxiliary,
        )
    )
    total = ranking + q + mode + quantity + auxiliary
    return OfflineLoss(
        total=total,
        ranking=ranking,
        q=q,
        mode=mode,
        quantity=quantity,
        auxiliary=auxiliary,
        ranking_accuracy=ranking_accuracy,
        activation=activation,
        calibration=calibration,
        finite=finite,
    )


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Return the weighted mean of one term's per-element losses.

    Args:
        values: Per-element losses.
        weights: Their sample weights.

    Returns:
        ``sum(values * weights) / sum(weights)``.
    """
    return (values * weights).sum() / weights.sum()


def pretrain(
    model: MarketResidualNet,
    train_sequences: Sequence[EventSequence],
    select_sequences: Sequence[EventSequence],
    config: PretrainConfig,
    root: Path,
    log: Callable[[int, dict[str, float]], None] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Train the head offline, resumably, and report what it reached.

    One epoch shuffles whole cells into minibatches, never events inside a
    cell. After every epoch both checkpoints are written atomically —
    ``last.ckpt`` with optimizer, RNG, epoch, history and the W&B run identity,
    ``best.ckpt`` whenever the selection loss improves — so an interruption
    anywhere resumes at the epoch boundary it last completed. On the way out
    the model is left holding the best-selection weights, the gates are
    measured on the selection cells, and ``offline-report.json`` is published.

    Args:
        model: The head to train; mutated, and left at the best-selection
            weights.
        train_sequences: The training cells.
        select_sequences: The temporally held-out selection cells.
        config: The run's parameterisation.
        root: The owned pretraining root.
        log: Called after each epoch with its metrics row, when given.
        run_id: The W&B run identity to store in every checkpoint.

    Returns:
        The report that was written to the root.

    Raises:
        PretrainDriftError: If the root was trained under a different config
            or different data.
        PretrainNonFiniteError: If any loss stops being finite.
    """
    device = torch.device(config.device)
    model.to(device)
    binding = canonical_digest(
        {
            "config": asdict(config),
            "model": model_digest(model.config),
            "train": sequences_digest(train_sequences),
            "select": sequences_digest(select_sequences),
        }
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    root.mkdir(parents=True, exist_ok=True)
    epoch = 0
    best: dict[str, float] = {"epoch": 0, "loss": float("inf")}
    history: list[dict[str, float]] = []
    last_path = root / LAST_CHECKPOINT
    if last_path.exists():
        stored = load_checkpoint(last_path)
        if stored["binding_sha256"] != binding:
            raise PretrainDriftError(
                f"{root} was trained under binding {stored['binding_sha256']}, "
                f"this run computes {binding}"
            )
        model.load_state_dict(stored["model"])
        optimizer.load_state_dict(stored["optimizer"])
        _restore_rng(stored["rng"])
        epoch = stored["epoch"]
        best = stored["best"]
        history = stored["history"]
        run_id = stored["wandb_run_id"] if run_id is None else run_id

    while epoch < config.epochs:
        epoch += 1
        model.train()
        order = torch.randperm(len(train_sequences)).tolist()
        train_rows: list[dict[str, float]] = []
        for start in range(0, len(order), config.batch_cells):
            cells = [
                train_sequences[i] for i in order[start : start + config.batch_cells]
            ]
            batch = collate_event_sequences(cells).to(device)
            heads, _state = model(batch.features, None)
            report = offline_loss(heads, batch)
            if not report.finite or not bool(torch.isfinite(report.total)):
                raise PretrainNonFiniteError(
                    f"epoch {epoch} produced a nonfinite head or loss"
                )
            optimizer.zero_grad()
            report.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            train_rows.append(report.metrics())
        selection = evaluate(model, select_sequences, config.batch_cells, device)
        row: dict[str, float] = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in _mean_rows(train_rows).items()})
        row.update({f"select_{k}": v for k, v in selection.items()})
        history.append(row)
        if selection["total"] < best["loss"]:
            best = {"epoch": epoch, "loss": selection["total"]}
            save_checkpoint(
                root / BEST_CHECKPOINT,
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "binding_sha256": binding,
                },
            )
        save_checkpoint(
            last_path,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": _capture_rng(),
                "epoch": epoch,
                "best": best,
                "history": history,
                "binding_sha256": binding,
                "wandb_run_id": run_id,
            },
        )
        if log is not None:
            log(epoch, row)

    model.load_state_dict(load_checkpoint(root / BEST_CHECKPOINT)["model"])
    gates = selection_gates(model, select_sequences, device)
    report_document: dict[str, Any] = {
        "config": asdict(config),
        "binding_sha256": binding,
        "epoch": epoch,
        "best_epoch": best["epoch"],
        "history": history,
        "gates": gates,
        "wandb_run_id": run_id,
    }
    write_json_atomic(root / REPORT_NAME, report_document)
    return report_document


def evaluate(
    model: MarketResidualNet,
    sequences: Sequence[EventSequence],
    batch_cells: int,
    device: torch.device,
) -> dict[str, float]:
    """Return the mean loss metrics over one dataset, without training.

    Args:
        model: The head to evaluate.
        sequences: The cells to evaluate on.
        batch_cells: How many cells to forward at once.
        device: Where the model runs.

    Returns:
        Per-term means, weighted by cells per chunk.
    """
    model.eval()
    rows: list[dict[str, float]] = []
    weights: list[float] = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_cells):
            cells = sequences[start : start + batch_cells]
            batch = collate_event_sequences(cells).to(device)
            heads, _state = model(batch.features, None)
            rows.append(offline_loss(heads, batch).metrics())
            weights.append(float(len(cells)))
    return _mean_rows(rows, weights)


def _mean_rows(
    rows: Sequence[dict[str, float]], weights: Sequence[float] | None = None
) -> dict[str, float]:
    """Average per-batch metric rows into one row.

    Args:
        rows: Per-batch metrics.
        weights: Per-row weights; uniform when omitted.

    Returns:
        The weighted mean of every key.
    """
    scale = list(weights) if weights is not None else [1.0] * len(rows)
    total = sum(scale)
    return {
        key: sum(row[key] * weight for row, weight in zip(rows, scale, strict=True))
        / total
        for key in rows[0]
    }


def selection_gates(
    model: MarketResidualNet,
    sequences: Sequence[EventSequence],
    device: torch.device | str,
) -> dict[str, float]:
    """Measure every declared gate on held-out cells.

    Args:
        model: The trained head.
        sequences: The held-out cells.
        device: Where the model runs.

    Returns:
        ``exploit_rate``: on events holding a profitable arm (win-point delta
        above the band), how often some profitable arm outscores the anchor.
        ``equal_retention``: on events whose best arm changes nothing, how
        often the greedy mode still defers. ``expected_delta``: the mean
        win-point delta of the top-scored arm over every labeled event.
        ``activation``: the greedy replacement share over every event.
        ``ranking_accuracy``: pairwise order agreement. ``nonfinite``: how
        many head values were not finite. Empty denominators score their gate
        as vacuously passed.
    """
    model.eval()
    exploit = [0, 0]
    retention = [0, 0]
    deltas: list[float] = []
    ordered = [0, 0]
    replaced = [0, 0]
    nonfinite = 0
    with torch.no_grad():
        for sequence in sequences:
            batch = collate_event_sequences((sequence,)).to(device)
            heads, _state = model(batch.features, None)
            report = offline_loss(heads, batch)
            nonfinite += 0 if report.finite else 1
            scores = log_probability(
                PolicyHeads(
                    mode_logits=heads.mode_logits[batch.arm_step, batch.arm_batch],
                    quantity_logits=heads.quantity_logits[
                        batch.arm_step, batch.arm_batch
                    ],
                    value=heads.value[batch.arm_step, batch.arm_batch],
                    auxiliary=heads.auxiliary[batch.arm_step, batch.arm_batch],
                ),
                batch.mask[batch.arm_step, batch.arm_batch],
                batch.arm_mode,
                batch.arm_buckets,
            )
            greedy = heads.mode_logits.argmax(dim=-1)
            replaced[0] += int((greedy == REPLACE_INDEX)[batch.valid].sum())
            replaced[1] += int(batch.valid.sum())
            margins = scores[batch.pair_better] - scores[batch.pair_worse]
            ordered[0] += int((margins > 0).sum())
            ordered[1] += len(margins)
            for group in _event_groups(sequence.arm_event):
                group_scores = scores[group]
                wins = sequence.arm_win_delta[group]
                deltas.append(float(wins[int(group_scores.argmax())]))
                profitable = wins > INDIFFERENCE_BAND
                if bool(profitable.any()):
                    exploit[1] += 1
                    exploit[0] += int(group_scores[profitable].max() > group_scores[0])
                if float(wins.max()) == 0.0:
                    retention[1] += 1
                    step = int(sequence.arm_event[group[0]])
                    retention[0] += int(greedy[step, 0]) == USE_KAITO_INDEX
    return {
        "exploit_rate": exploit[0] / exploit[1] if exploit[1] else 1.0,
        "equal_retention": retention[0] / retention[1] if retention[1] else 1.0,
        "expected_delta": sum(deltas) / len(deltas) if deltas else 0.0,
        "activation": replaced[0] / replaced[1],
        "ranking_accuracy": ordered[0] / ordered[1] if ordered[1] else 1.0,
        "nonfinite": nonfinite,
    }


def check_gates(gates: dict[str, float]) -> None:
    """Hold a finished run to every declared band, or say which ones it broke.

    Args:
        gates: The measurements ``selection_gates`` produced.

    Raises:
        PretrainGateError: Naming every band the run is outside.
    """
    failures = []
    if gates["exploit_rate"] < EXPLOIT_GATE:
        failures.append(f"exploit_rate {gates['exploit_rate']:.3f} < {EXPLOIT_GATE}")
    if gates["equal_retention"] < RETENTION_GATE:
        failures.append(
            f"equal_retention {gates['equal_retention']:.3f} < {RETENTION_GATE}"
        )
    if gates["expected_delta"] < 0.0:
        failures.append(f"expected_delta {gates['expected_delta']:.4f} < 0")
    if not ACTIVATION_BAND[0] <= gates["activation"] <= ACTIVATION_BAND[1]:
        failures.append(
            f"activation {gates['activation']:.3f} outside {ACTIVATION_BAND}"
        )
    if gates["nonfinite"]:
        failures.append(f"{gates['nonfinite']} cells produced nonfinite heads")
    if failures:
        raise PretrainGateError("; ".join(failures))


def sequences_digest(sequences: Sequence[EventSequence]) -> str:
    """Return one digest over exactly the tensors a run would train on.

    Args:
        sequences: The dataset, in order.

    Returns:
        A SHA-256 hex digest; any changed value, row or order changes it.
    """
    digest = hashlib.sha256()
    for sequence in sequences:
        digest.update(sequence.identity_sha256.encode("utf-8"))
        digest.update(f"{sequence.seed}:{sequence.seat}".encode("utf-8"))
        for field in fields(sequence):
            value = getattr(sequence, field.name)
            if isinstance(value, torch.Tensor):
                digest.update(value.numpy(force=True).tobytes())
    return digest.hexdigest()


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Publish one checkpoint atomically: a torn write can never be loaded.

    Args:
        path: The checkpoint address.
        payload: What to store.
    """
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load one checkpoint onto the CPU.

    Args:
        path: The checkpoint address.

    Returns:
        The stored payload.
    """
    return torch.load(path, map_location="cpu", weights_only=False)


def write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    """Publish one JSON document atomically.

    Args:
        path: The document address.
        document: What to store.
    """
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _capture_rng() -> dict[str, Any]:
    """Return every random state a resume must put back."""
    return {
        "python": random.getstate(),
        "numpy": numpy.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(states: dict[str, Any]) -> None:
    """Put back every random state a checkpoint captured.

    Args:
        states: What ``_capture_rng`` stored.
    """
    random.setstate(states["python"])
    numpy.random.set_state(states["numpy"])
    torch.random.set_rng_state(states["torch"])
    if torch.cuda.is_available() and states["cuda"]:
        torch.cuda.set_rng_state_all(states["cuda"])


def collection_identities(root: Path) -> tuple[SeasonIdentity, ...]:
    """Rebuild every cell identity a collection root was bound to.

    The stored parameters name the bank, seeds and configs; the identities are
    reconstructed through ``SeasonIdentity.current`` — which re-checks the
    installed engine — and must reproduce the exact cell digests the root was
    created with, so a drifted engine, controller or config is refused here.

    Args:
        root: A collection root the counterfactual CLI produced.

    Returns:
        One identity per seat-cell, in the root's own seed-then-seat order.

    Raises:
        CounterfactualIntegrityError: If the parameters fail their own digest,
            name a controller other than the served baseline, or no longer
            reproduce the recorded cell digests.
    """
    stored = json.loads(
        (root / "collection-parameters.json").read_text(encoding="utf-8")
    )
    if stored["sha256"] != canonical_digest(stored["fields"]):
        raise CounterfactualIntegrityError(
            f"{root} collection parameters do not match their own digest"
        )
    parameters = stored["fields"]
    for side in ("learner_sha256", "opponent_sha256"):
        if parameters[side] != SERVED.sha256:
            raise CounterfactualIntegrityError(
                f"{root} was collected against {side[:-7]} {parameters[side]}, "
                f"this trainer serves {SERVED.sha256}"
            )
    seats: tuple[Literal[0, 1], ...] = (0, 1)
    identities = tuple(
        SeasonIdentity.current(
            learner=SERVED,
            opponent=SERVED,
            seed_bank=parameters["seed_bank"],
            seed=seed,
            seat=seat,
            event_config=EventConfig(**parameters["event_config"]),
            alternative_config=AlternativeConfig(**parameters["alternative_config"]),
        )
        for seed in parameters["seeds"]
        for seat in seats
    )
    if [identity.sha256 for identity in identities] != parameters["cells"]:
        raise CounterfactualIntegrityError(
            f"{root} cell digests no longer reproduce; the engine, controller "
            "or configuration has drifted since collection"
        )
    return identities


def cell_sequence(root: Path, identity: SeasonIdentity) -> EventSequence:
    """Rebuild one cell's event sequence and attach its shard's labels.

    The season is replayed through the reference engine exactly as collection
    played it, the event machine re-opens the same events, and the canonical
    feature row is chained event to event the way the served runtime chains
    it. Every labeled row is checked against the replayed event's fingerprint,
    so a label can never attach to a different instant than it was measured
    at.

    Args:
        root: The collection root holding the shard.
        identity: The cell to rebuild.

    Returns:
        The sequence.

    Raises:
        CounterfactualIntegrityError: If a shard row does not match the
            replayed season.
    """
    shard = load_shard_strict(
        shard_path(root / "shards", ShardIdentity.of(identity)), identity
    )
    season = record_season(identity)
    events = season_events(season)

    vectors: list[MarketFeatureVector] = []
    for event in events:
        encoded = encode_observation(
            season.learner.observations[event.event.turn], identity.seat
        )
        vectors.append(
            market_feature_vector(
                encoded,
                event.event.kaito_buckets,
                vectors[-1] if vectors else None,
            )
        )
    features = torch.tensor([vector.values for vector in vectors])

    auxiliary = torch.zeros(len(events), 3 * len(PRODUCT_NAMES))
    for index, vector in enumerate(vectors[1:]):
        auxiliary[index] = torch.tensor(_auxiliary_row(vector))
    auxiliary_valid = torch.arange(len(events)) < len(events) - 1

    labeled = sorted(shard.rows, key=lambda row: (row.event_index, row.alternative_id))
    arm_event: list[int] = []
    arm_mode: list[int] = []
    arm_buckets: list[tuple[int, ...]] = []
    arm_win: list[float] = []
    arm_q: list[float] = []
    arm_flip: list[bool] = []
    for row in labeled:
        recorded = events[row.event_index]
        if recorded.event.fingerprint != row.event_fingerprint:
            raise CounterfactualIntegrityError(
                f"seed {identity.seed} seat {identity.seat} event "
                f"{row.event_index} replays fingerprint "
                f"{recorded.event.fingerprint}, shard holds {row.event_fingerprint}"
            )
        arm_event.append(row.event_index)
        arm_mode.append(USE_KAITO_INDEX if row.alternative_id == 0 else REPLACE_INDEX)
        arm_buckets.append(row.buckets)
        arm_win.append(row.win_point_delta)
        arm_q.append(row.win_point_delta + MARGIN_WEIGHT * row.margin_delta)
        arm_flip.append(_flips_the_season(row))
    return EventSequence(
        identity_sha256=identity.sha256,
        seed=identity.seed,
        seat=identity.seat,
        features=features,
        auxiliary_targets=auxiliary,
        auxiliary_valid=auxiliary_valid,
        arm_event=torch.tensor(arm_event, dtype=torch.long),
        arm_mode=torch.tensor(arm_mode, dtype=torch.long),
        arm_buckets=torch.tensor(arm_buckets, dtype=torch.long),
        arm_win_delta=torch.tensor(arm_win),
        arm_q=torch.tensor(arm_q),
        arm_flip=torch.tensor(arm_flip),
    )


def _auxiliary_row(vector: MarketFeatureVector) -> list[float]:
    """Read one feature row back into the auxiliary head's target layout.

    Args:
        vector: The next event's canonical feature row.

    Returns:
        Per product: the normalized live price, the normalized book
        inventory, and the opponent-supply bucket scaled into ``[0, 1]``.
    """
    row: list[float] = []
    for prefix in ("price", "inventory"):
        for product in PRODUCT_NAMES:
            row.append(vector.block(f"{prefix}:{product}")[0])
    for product in PRODUCT_NAMES:
        block = vector.block(f"opponent_supply:{product}")
        row.append(block.index(1.0) / (BUCKETS - 1))
    return row


def _flips_the_season(row: CounterfactualRow) -> bool:
    """Say whether one arm turns a lost season into a won one, or back.

    Args:
        row: The scored arm.

    Returns:
        True when the pair's win points sit strictly on opposite sides of a
        tie — a change of degree (tie to win, loss to tie) is not a flip.
    """
    low = min(row.kaito.win_points, row.alternative.win_points)
    high = max(row.kaito.win_points, row.alternative.win_points)
    return low < 0.5 < high


def load_sequences(root: Path, workers: int) -> tuple[EventSequence, ...]:
    """Rebuild every cell of one collection root, in parallel.

    Args:
        root: The collection root.
        workers: How many replay processes to run.

    Returns:
        The sequences, in the root's own cell order.
    """
    identities = collection_identities(root)
    # Spawned, not forked: by the time a gate test reaches this loader the
    # process has already trained under torch and carries its thread pool, and
    # a forked child of a threaded process can inherit a held lock and hang
    # before it runs a line. A spawned worker pays a clean import instead.
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(cell_sequence, root, identity) for identity in identities
        ]
        return tuple(
            future.result()
            for future in tqdm(futures, desc=f"cells:{root.name}", total=len(futures))
        )
