"""Offline losses: exact term arithmetic, collation order, and the gates.

The fast tests pin the loss arithmetic to hand-computable numbers: heads built
from literal tensors, sequences with two or three arms whose paired deltas are
chosen so every softplus, Huber and cross-entropy term can be checked with
``pytest.approx``. The slow tests are the two known-exploit gates the plan
names: a synthetic price-crossing task the head must rank at 95%, and real
smoke-collection shards on which it must prefer a profitable alternative over
the frozen controller on three quarters of held-out cells.
"""

import math
from pathlib import Path

import pytest
import torch

from kaggriculture.features import PRODUCT_NAMES
from kaggriculture.learn.market_residual.model import (
    MarketResidualNet,
    ModelConfig,
    PolicyHeads,
)
from kaggriculture.learn.market_residual.offline import (
    EXPLOIT_GATE,
    FLIP_WEIGHT,
    INDIFFERENCE_BAND,
    MARGIN_WEIGHT,
    REPLACE_INDEX,
    SYNTHETIC_RANKING_GATE,
    USE_KAITO_INDEX,
    EventSequence,
    PretrainConfig,
    collate_event_sequences,
    offline_loss,
    pretrain,
    selection_gates,
)
from kaggriculture.market_residual.schema import MarketFeatureSchema

CONFIG = ModelConfig.current()


def manual_sequence(
    labels: dict[int, tuple[tuple[float, float, bool], ...]],
    events: int = 4,
    feature_fill: float = 0.1,
) -> EventSequence:
    """Build one cell's sequence with hand-chosen counterfactual labels.

    Args:
        labels: Event position mapped to its replacement arms, each a
            ``(win_point_delta, margin_delta, flip)`` triple. The anchor arm is
            added automatically, first, at delta zero.
        events: How many events the sequence runs for.
        feature_fill: The constant every feature column carries.

    Returns:
        The sequence, with deterministic per-arm buckets ``arm_index % buckets``.
    """
    arm_event: list[int] = []
    arm_mode: list[int] = []
    arm_buckets: list[list[int]] = []
    arm_win: list[float] = []
    arm_q: list[float] = []
    arm_flip: list[bool] = []
    for position in sorted(labels):
        rows = ((0.0, 0.0, False), *labels[position])
        for index, (win, margin, flip) in enumerate(rows):
            arm_event.append(position)
            arm_mode.append(USE_KAITO_INDEX if index == 0 else REPLACE_INDEX)
            arm_buckets.append([index % CONFIG.buckets] * CONFIG.slots)
            arm_win.append(win)
            arm_q.append(win + MARGIN_WEIGHT * margin)
            arm_flip.append(flip)
    return EventSequence(
        identity_sha256="manual",
        seed=0,
        seat=0,
        features=torch.full((events, CONFIG.input_size), feature_fill),
        auxiliary_targets=torch.full((events, CONFIG.auxiliary_size), 0.3),
        auxiliary_valid=torch.arange(events) < events - 1,
        arm_event=torch.tensor(arm_event, dtype=torch.long),
        arm_mode=torch.tensor(arm_mode, dtype=torch.long),
        arm_buckets=torch.tensor(arm_buckets, dtype=torch.long),
        arm_win_delta=torch.tensor(arm_win),
        arm_q=torch.tensor(arm_q),
        arm_flip=torch.tensor(arm_flip),
    )


def zero_heads(steps: int, batch: int) -> PolicyHeads:
    """Return all-zero heads, whose softmaxes are exactly uniform."""
    return PolicyHeads(
        mode_logits=torch.zeros(steps, batch, 2),
        quantity_logits=torch.zeros(steps, batch, CONFIG.slots, CONFIG.buckets),
        value=torch.zeros(steps, batch),
        auxiliary=torch.zeros(steps, batch, CONFIG.auxiliary_size),
    )


def softplus(value: float) -> float:
    """Return ``log(1 + exp(value))`` for a hand-computed expectation."""
    return math.log1p(math.exp(value))


def test_the_declared_values_are_the_plan_values() -> None:
    """The plan's exact numbers are pinned as literals, not self-references."""
    from kaggriculture.learn.market_residual.offline import (
        ACTIVATION_BAND,
    )
    from kaggriculture.learn.market_residual.offline import (
        EXPLOIT_GATE as EXPLOIT,
    )
    from kaggriculture.learn.market_residual.offline import (
        SYNTHETIC_RANKING_GATE as RANKING,
    )

    assert INDIFFERENCE_BAND == 0.01
    assert MARGIN_WEIGHT == 0.1
    assert FLIP_WEIGHT == 4.0
    assert ACTIVATION_BAND == (0.01, 0.50)
    assert RANKING == 0.95
    assert EXPLOIT == 0.75


def test_decisive_flip_outranks_kaito() -> None:
    """A decisive flip produces ranking signal and finite, nonzero gradients."""
    torch.manual_seed(0)
    model = MarketResidualNet(CONFIG)
    batch = collate_event_sequences((manual_sequence({1: ((1.0, 0.4, True),)}),))
    heads, _state = model(batch.features, None)
    report = offline_loss(heads, batch)
    report.total.backward()
    assert report.ranking.item() > 0
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_weak_evidence_targets_use_kaito() -> None:
    """Equal-outcome arms teach deferring, whatever their margins say."""
    batch = collate_event_sequences(
        (manual_sequence({0: ((0.0, 0.2, False), (0.0, -0.1, False))}),)
    )
    assert batch.mode_target.item() == USE_KAITO_INDEX
    assert batch.q_target.item() == 0.0


def test_the_indifference_band_is_exact() -> None:
    """Only a win-point delta above the band teaches a replacement."""
    on_band = collate_event_sequences(
        (manual_sequence({0: ((INDIFFERENCE_BAND, 0.0, False),)}),)
    )
    assert on_band.mode_target.item() == USE_KAITO_INDEX
    above = collate_event_sequences(
        (manual_sequence({0: ((INDIFFERENCE_BAND + 0.01, 0.5, False),)}),)
    )
    assert above.mode_target.item() == REPLACE_INDEX
    assert above.quantity_target[0].tolist() == [1 % CONFIG.buckets] * CONFIG.slots
    assert above.q_target.item() == pytest.approx(
        INDIFFERENCE_BAND + 0.01 + MARGIN_WEIGHT * 0.5
    )


def test_flips_carry_four_times_the_weight() -> None:
    """Flip arms weight the event terms and every pair they sit in."""
    batch = collate_event_sequences(
        (
            manual_sequence(
                {
                    0: ((1.0, 0.2, True),),
                    2: ((0.5, 0.1, False), (-0.5, -0.1, False)),
                }
            ),
        )
    )
    assert batch.event_weight.tolist() == [4.0, 1.0]
    # Event 0 pairs: (flip arm, anchor) -> 4.0. Event 2 pairs, none flipped:
    # (good, anchor), (good, bad), (anchor, bad) -> 1.0 each.
    assert sorted(batch.pair_weight.tolist()) == [1.0, 1.0, 1.0, 4.0]


def test_ranking_pairs_never_cross_events() -> None:
    """Arms are only ever ranked against arms of their own event."""
    batch = collate_event_sequences(
        (
            manual_sequence({0: ((0.5, 0.0, False),)}),
            manual_sequence({1: ((-0.5, 0.0, False),)}),
        )
    )
    assert len(batch.pair_better.tolist()) == 2
    same_event = batch.arm_step[batch.pair_better] == batch.arm_step[batch.pair_worse]
    same_cell = batch.arm_batch[batch.pair_better] == batch.arm_batch[batch.pair_worse]
    assert bool((same_event & same_cell).all())


def test_collation_preserves_order_and_padding() -> None:
    """Cells become columns; event order and padding are exact."""
    short = manual_sequence({0: ((0.5, 0.0, False),)}, events=2, feature_fill=0.5)
    long = manual_sequence({3: ((0.5, 0.0, False),)}, events=4, feature_fill=0.25)
    batch = collate_event_sequences((short, long))
    assert batch.features.shape == (4, 2, CONFIG.input_size)
    assert batch.valid.tolist() == [
        [True, True],
        [True, True],
        [False, True],
        [False, True],
    ]
    assert batch.features[1, 0, 0].item() == 0.5
    assert batch.features[2, 0].abs().sum().item() == 0.0
    assert batch.features[3, 1, 0].item() == 0.25
    assert batch.arm_step.tolist() == [0, 0, 3, 3]
    assert batch.arm_batch.tolist() == [0, 0, 1, 1]
    assert batch.auxiliary_valid.tolist() == [
        [True, True],
        [False, True],
        [False, True],
        [False, False],
    ]


def test_every_term_is_the_hand_computed_number() -> None:
    """Uniform heads produce exactly the arithmetic the docstring claims."""
    batch = collate_event_sequences(
        (manual_sequence({1: ((0.5, 0.2, False), (-0.5, -0.2, False))}, events=3),)
    )
    report = offline_loss(zero_heads(3, 1), batch)
    anchor = -math.log(2.0)
    replacement = anchor + CONFIG.slots * -math.log(CONFIG.buckets)
    expected_ranking = (
        softplus(anchor - replacement)  # the good arm over the anchor
        + softplus(0.0)  # the good arm over the bad arm, identical scores
        + softplus(replacement - anchor)  # the anchor over the bad arm
    ) / 3.0
    assert report.ranking.item() == pytest.approx(expected_ranking, rel=1e-5)
    assert report.q.item() == pytest.approx(0.5 * 0.52**2, rel=1e-5)
    assert report.mode.item() == pytest.approx(math.log(2.0), rel=1e-5)
    assert report.quantity.item() == pytest.approx(math.log(CONFIG.buckets), rel=1e-5)
    assert report.auxiliary.item() == pytest.approx(0.5 * 0.3**2, rel=1e-5)
    assert report.total.item() == pytest.approx(
        report.ranking.item()
        + report.q.item()
        + report.mode.item()
        + report.quantity.item()
        + report.auxiliary.item(),
        rel=1e-6,
    )
    assert report.calibration == pytest.approx(0.52, rel=1e-5)
    assert report.finite


def test_ranking_penalizes_an_outranked_better_arm() -> None:
    """An unordered pair is charged its full logistic margin."""
    # One profitable replacement, one pair. Under uniform heads the anchor's
    # single mode term outscores the replacement's joint log-probability by
    # exactly the quantity terms, and the logistic loss must charge for it;
    # a flipped margin sign would instead score this near zero.
    batch = collate_event_sequences(
        (manual_sequence({0: ((0.5, 0.0, False),)}, events=2),)
    )
    report = offline_loss(zero_heads(2, 1), batch)
    expected = softplus(CONFIG.slots * math.log(CONFIG.buckets))
    assert report.ranking.item() == pytest.approx(expected, rel=1e-5)
    assert report.ranking_accuracy == 0.0


def test_auxiliary_loss_ignores_invalid_steps() -> None:
    """Targets past the last event never reach the auxiliary term."""
    sequence = manual_sequence({0: ((0.5, 0.0, False),)}, events=3)
    poisoned = manual_sequence({0: ((0.5, 0.0, False),)}, events=3)
    poisoned.auxiliary_targets[-1] = 1e6
    baseline = offline_loss(zero_heads(3, 1), collate_event_sequences((sequence,)))
    altered = offline_loss(zero_heads(3, 1), collate_event_sequences((poisoned,)))
    assert altered.auxiliary.item() == pytest.approx(baseline.auxiliary.item())


def test_activation_and_finite_are_read_from_the_heads() -> None:
    """Activation counts greedy replacements; one NaN flips finite."""
    batch = collate_event_sequences(
        (manual_sequence({0: ((0.5, 0.0, False),)}, events=4),)
    )
    replacing = zero_heads(4, 1)
    replacing.mode_logits[:2, :, REPLACE_INDEX] = 5.0
    assert offline_loss(replacing, batch).activation == pytest.approx(0.5)
    poisoned = zero_heads(4, 1)
    poisoned.value[0, 0] = float("nan")
    assert not offline_loss(poisoned, batch).finite


def synthetic_price_crossing(
    cells: int, events: int, seed: int, price_column: int
) -> tuple[EventSequence, ...]:
    """Build the synthetic exploit: one price column decides every label.

    Each event carries an anchor and one replacement whose paired win-point
    delta is ``+0.5`` when the price column reads above ``0.55`` and ``-0.5``
    below ``0.45``, so a head that reads the board reaches perfect ranking and
    a head that reads nothing stays at chance. Prices inside the band stay
    unlabeled: the gate measures whether the crossing is read at all, not how
    finely a boundary is interpolated.

    Args:
        cells: How many sequences to build.
        events: Events per sequence.
        seed: The generator seed.
        price_column: Which feature column carries the deciding price.

    Returns:
        The sequences.
    """
    generator = torch.Generator().manual_seed(seed)
    sequences = []
    for _cell in range(cells):
        features = 0.05 * torch.rand((events, CONFIG.input_size), generator=generator)
        prices = torch.rand((events,), generator=generator)
        features[:, price_column] = prices
        labels: dict[int, tuple[tuple[float, float, bool], ...]] = {
            position: ((0.5 if prices[position] > 0.55 else -0.5, 0.0, False),)
            for position in range(events)
            if abs(float(prices[position]) - 0.5) > 0.05
        }
        sequence = manual_sequence(labels, events=events)
        sequence.features.copy_(features)
        sequences.append(sequence)
    return tuple(sequences)


@pytest.mark.slow
def test_the_synthetic_price_crossing_gate_holds(tmp_path: Path) -> None:
    """The head learns a one-column price rule to the 95% gate."""
    column = MarketFeatureSchema.current().span(f"price:{PRODUCT_NAMES[0]}").start
    train = synthetic_price_crossing(96, 12, seed=1, price_column=column)
    held_out = synthetic_price_crossing(24, 12, seed=2, price_column=column)
    torch.manual_seed(0)
    model = MarketResidualNet(CONFIG)
    config = PretrainConfig(epochs=60, batch_cells=32, learning_rate=3e-3, device="cpu")
    pretrain(model, train, held_out, config, tmp_path)
    gates = selection_gates(model, held_out, "cpu")
    assert gates["ranking_accuracy"] >= SYNTHETIC_RANKING_GATE


@pytest.mark.slow
def test_the_real_market_exploit_gate_holds(
    smoke_sequences: tuple[EventSequence, ...], tmp_path: Path
) -> None:
    """Trained on real shards, the head exploits held-out cells."""
    train = tuple(s for s in smoke_sequences if s.seed < 860_006)
    held_out = tuple(s for s in smoke_sequences if s.seed >= 860_006)
    assert len(train) == 12 and len(held_out) == 4
    torch.manual_seed(0)
    model = MarketResidualNet(CONFIG)
    config = PretrainConfig(epochs=120, batch_cells=4, learning_rate=3e-3, device="cpu")
    pretrain(model, train, held_out, config, tmp_path)
    gates = selection_gates(model, held_out, "cpu")
    assert gates["exploit_rate"] >= EXPLOIT_GATE
    assert gates["equal_retention"] >= 0.5
    assert gates["nonfinite"] == 0
