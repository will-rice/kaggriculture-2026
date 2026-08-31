"""What the Torch head has to be before anything is trained into it.

Three of these are cheap to state and expensive to discover late.

The head is *recurrent*. A GRU whose state is dropped between events is not a
bug that raises; it is a head that can never learn timing, and the symptom
arrives months later as "the residual never learned when to sell". So the
recurrence is asserted directly: the same rows fed with a carried state must
not produce the outputs they produce from a fresh one.

The head runs in FP32 whatever the surrounding trainer is doing. The rest of
this project trains under BF16 autocast, and a mode logit, a joint log
probability or a value target computed in BF16 has about three decimal digits.
That is not enough for the paired win-point deltas this residual is regressed
against, and it is not what the exported NumPy artifact would reproduce.

Illegal quantities are masked with the *finite* float32 minimum rather than
``-inf``. Every allowed slot can be masked down to a single legal bucket, and
with ``-inf`` a row whose every bucket is illegal makes ``log_softmax`` return
NaN, which propagates into the loss and stops the run. A finite minimum makes
that row uniform instead, and keeps every logit the merge boundary inspects
finite, so ``ResidualDecision.finite`` means what it says.
"""

import dataclasses

import pytest
import torch

from kaggriculture.learn.market_residual.model import (
    MarketResidualNet,
    ModelConfig,
    entropy,
    greedy_decision,
    log_probability,
    mask_quantity_logits,
)
from kaggriculture.market_residual.actions import ResidualMode
from kaggriculture.market_residual.features import MarketFeatureVector
from kaggriculture.market_residual.numpy_policy import MASK_FILL
from kaggriculture.market_residual.schema import ALLOWED_SLOTS, BUCKETS

PARAMETER_BUDGET = 1_000_000

# Long enough that a dropped hidden state is unmistakable, short enough that
# every test in this file runs on CPU in well under a second.
WINDOW = 64


@pytest.fixture(scope="module")
def config() -> ModelConfig:
    """Return the declared head configuration for the current schema."""
    return ModelConfig.current()


@pytest.fixture(scope="module")
def model(config: ModelConfig) -> MarketResidualNet:
    """Return one deterministically initialised head in evaluation mode."""
    torch.manual_seed(0)
    return MarketResidualNet(config).eval()


def sequence(rows: tuple[MarketFeatureVector, ...]) -> torch.Tensor:
    """Return real feature rows as one ``(steps, 1, width)`` FP32 batch.

    Args:
        rows: Market feature rows in the order one seat produced them.

    Returns:
        The rows stacked on the time axis of a single-sequence batch.
    """
    return torch.tensor([[row.values] for row in rows], dtype=torch.float32)


def test_the_head_is_recurrent_and_stays_under_the_parameter_budget(
    model: MarketResidualNet, event_rows: tuple[MarketFeatureVector, ...]
) -> None:
    """Carried state must change the answer, and the head must stay small."""
    first = sequence(event_rows[:WINDOW])
    second = sequence(event_rows[WINDOW : 2 * WINDOW])

    with torch.no_grad():
        _, state = model(first, None)
        carried, _ = model(second, state)
        fresh, _ = model(second, None)

    assert not torch.equal(carried.mode_logits, fresh.mode_logits)
    assert not torch.equal(carried.value, fresh.value)
    assert sum(parameter.numel() for parameter in model.parameters()) < PARAMETER_BUDGET


def test_the_head_emits_one_output_per_declared_head(
    model: MarketResidualNet,
    config: ModelConfig,
    event_rows: tuple[MarketFeatureVector, ...],
) -> None:
    """Every head is shaped by the schema, not by a literal written here."""
    rows = sequence(event_rows[:WINDOW])

    with torch.no_grad():
        heads, state = model(rows, None)

    assert heads.mode_logits.shape == (WINDOW, 1, len(ResidualMode))
    assert heads.quantity_logits.shape == (WINDOW, 1, len(ALLOWED_SLOTS), BUCKETS)
    assert heads.value.shape == (WINDOW, 1)
    assert heads.auxiliary.shape == (WINDOW, 1, config.auxiliary_size)
    assert state.shape == (1, 1, config.hidden_size)


def test_the_recurrent_state_is_the_one_the_next_call_continues_from(
    model: MarketResidualNet, event_rows: tuple[MarketFeatureVector, ...]
) -> None:
    """Splitting a sequence in two must equal running it whole."""
    rows = event_rows[: 2 * WINDOW]

    with torch.no_grad():
        whole, _ = model(sequence(rows), None)
        head, state = model(sequence(rows[:WINDOW]), None)
        tail, _ = model(sequence(rows[WINDOW:]), state)

    assert torch.allclose(whole.mode_logits[:WINDOW], head.mode_logits, atol=1e-6)
    assert torch.allclose(whole.mode_logits[WINDOW:], tail.mode_logits, atol=1e-6)


def test_illegal_quantities_are_masked_with_a_finite_minimum(
    model: MarketResidualNet, event_rows: tuple[MarketFeatureVector, ...]
) -> None:
    """Masking must zero the probability without ever producing a NaN."""
    rows = sequence(event_rows[:WINDOW])
    generator = torch.Generator().manual_seed(1)
    mask = torch.zeros(WINDOW, 1, len(ALLOWED_SLOTS), BUCKETS, dtype=torch.bool)
    mask.bernoulli_(0.3, generator=generator)
    # One slot with nothing legal at all: the case `-inf` turns into NaN.
    mask[:, :, 0, :] = False

    with torch.no_grad():
        heads, _ = model(rows, None)
    masked = mask_quantity_logits(heads.quantity_logits, mask)
    probabilities = torch.softmax(masked, dim=-1)

    assert torch.equal(masked[mask], heads.quantity_logits[mask])
    assert torch.equal(
        masked[~mask], torch.full_like(masked[~mask], MASK_FILL, dtype=masked.dtype)
    )
    assert torch.isfinite(masked).all()
    assert torch.isfinite(probabilities).all()
    assert float(probabilities[:, :, 1:, :][~mask[:, :, 1:, :]].max()) == 0.0


def test_the_joint_log_probability_is_the_factorised_one(
    model: MarketResidualNet, event_rows: tuple[MarketFeatureVector, ...]
) -> None:
    """Replacing costs the mode term plus one term per slot; deferring does not."""
    rows = sequence(event_rows[:WINDOW])
    mask = torch.ones(WINDOW, 1, len(ALLOWED_SLOTS), BUCKETS, dtype=torch.bool)
    generator = torch.Generator().manual_seed(2)
    modes = torch.randint(0, len(ResidualMode), (WINDOW, 1), generator=generator)
    buckets = torch.randint(
        0, BUCKETS, (WINDOW, 1, len(ALLOWED_SLOTS)), generator=generator
    )

    with torch.no_grad():
        heads, _ = model(rows, None)
    joint = log_probability(heads, mask, modes, buckets)

    mode_terms = torch.log_softmax(heads.mode_logits, dim=-1).gather(
        -1, modes[..., None]
    )[..., 0]
    quantity_terms = (
        torch.log_softmax(heads.quantity_logits, dim=-1)
        .gather(-1, buckets[..., None])[..., 0]
        .sum(dim=-1)
    )
    replacing = modes == list(ResidualMode).index(ResidualMode.REPLACE)
    expected = mode_terms + torch.where(replacing, quantity_terms, 0.0)

    assert torch.allclose(joint, expected, atol=1e-6)
    assert joint.dtype == torch.float32


def test_entropy_counts_the_quantity_heads_only_as_often_as_they_are_used(
    model: MarketResidualNet, event_rows: tuple[MarketFeatureVector, ...]
) -> None:
    """A head that never replaces has the entropy of its mode alone."""
    rows = sequence(event_rows[:WINDOW])
    mask = torch.ones(WINDOW, 1, len(ALLOWED_SLOTS), BUCKETS, dtype=torch.bool)

    with torch.no_grad():
        heads, _ = model(rows, None)
    replace = list(ResidualMode).index(ResidualMode.REPLACE)
    certain_defer = heads.mode_logits.clone()
    certain_defer[..., replace] = -30.0
    certain_defer[..., 1 - replace] = 30.0

    mixed = entropy(heads, mask)
    deferring = entropy(dataclasses.replace(heads, mode_logits=certain_defer), mask)

    assert (mixed > deferring).all()
    assert torch.allclose(deferring, torch.zeros_like(deferring), atol=1e-6)


def test_bf16_autocast_does_not_reach_the_head_outputs(
    model: MarketResidualNet, event_rows: tuple[MarketFeatureVector, ...]
) -> None:
    """The trainer's precision is not allowed to change what the head computes."""
    rows = sequence(event_rows[:WINDOW])

    with torch.no_grad():
        plain, plain_state = model(rows, None)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            cast, cast_state = model(rows, None)

    assert cast.mode_logits.dtype == torch.float32
    assert cast.value.dtype == torch.float32
    assert cast.auxiliary.dtype == torch.float32
    assert cast_state.dtype == torch.float32
    assert torch.equal(plain.mode_logits, cast.mode_logits)
    assert torch.equal(plain.value, cast.value)
    assert torch.equal(plain_state, cast_state)


def test_a_greedy_decision_reads_the_masked_logits_it_was_given(
    model: MarketResidualNet, event_rows: tuple[MarketFeatureVector, ...]
) -> None:
    """The argmax decode may never name a bucket its mask forbade."""
    rows = sequence(event_rows[:WINDOW])
    generator = torch.Generator().manual_seed(3)
    mask = torch.zeros(WINDOW, 1, len(ALLOWED_SLOTS), BUCKETS, dtype=torch.bool)
    mask.bernoulli_(0.3, generator=generator)
    mask[..., 0] = True

    with torch.no_grad():
        heads, _ = model(rows, None)
    masked = mask_quantity_logits(heads.quantity_logits, mask)

    for step in range(WINDOW):
        decision = greedy_decision(heads.mode_logits[step, 0], masked[step, 0])
        assert decision.finite
        assert len(decision.buckets) == len(ALLOWED_SLOTS)
        for slot, bucket in enumerate(decision.buckets):
            assert bool(mask[step, 0, slot, bucket])


def test_a_nonfinite_logit_is_reported_rather_than_decoded(
    model: MarketResidualNet, event_rows: tuple[MarketFeatureVector, ...]
) -> None:
    """A head that has gone NaN must reach the merge as a refusal, not a bucket."""
    rows = sequence(event_rows[:1])

    with torch.no_grad():
        heads, _ = model(rows, None)
    broken = heads.mode_logits[0, 0].clone()
    broken[0] = torch.nan

    assert not greedy_decision(broken, heads.quantity_logits[0, 0]).finite


def test_the_head_refuses_a_row_of_the_wrong_width(config: ModelConfig) -> None:
    """A schema change must break loudly here rather than shift every column."""
    torch.manual_seed(0)
    model = MarketResidualNet(config).eval()

    with pytest.raises(RuntimeError):
        model(torch.zeros(2, 1, config.input_size - 1), None)
