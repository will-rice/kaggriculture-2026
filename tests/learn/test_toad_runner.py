"""Tests that go through the runner, not just the loss it calls.

The value warmup shipped with ``baseline_only`` passed to a ``_step`` that did
not accept it. Every loss test still passed, because they call ``losses``
directly and never cross the runner boundary -- so arm C' launched, crashed on
its first batch, and burned the slot. These call what the runner calls.
"""

import copy
from typing import Any

import pytest
import torch

from kaggriculture.learn.encoding import (
    MARKET_SLOTS,
    MAX_UNITS,
    QUANTITIES,
    SCALARS,
    TILE_PLANES,
    UNIT_OPS,
)
from kaggriculture.learn.model import Policy
from kaggriculture.learn.scripts import toad_phase1
from kaggriculture.learn.toad_loss import ADAM_EPS, LEARNING_RATE, TEACHER_KL_COST


def _segment(turns: int = 16, transfers: bool = True) -> dict[str, torch.Tensor]:
    """Return one synthetic unroll shaped exactly as ``_segments`` produces.

    The observed fields carry one row more than the acted ones. That extra state
    is the one the value target bootstraps from -- it is deliberately outside the
    segment's own decisions -- so a stand-in built at equal lengths would not be
    the shape the runner is handed. See ``toad_phase1._segments``.

    ``transfers`` decides whether any unit played a ``PICKUP``. The quantity
    head is scored only on the slots that did, so a fixture that never
    transfers and a fixture that does are the two halves of that condition,
    and the runner tests below need both to say anything about it.

    Args:
        turns: Acted rows in the unroll.
        transfers: Whether unit 0 spends its quantity bucket on every turn.

    Returns:
        One segment, keyed exactly as ``_segments`` keys them.
    """
    slots = len(MARKET_SLOTS) + 2
    unit_actions = torch.zeros(turns, MAX_UNITS, dtype=torch.int64)
    if transfers:
        unit_actions[:, 0] = UNIT_OPS.index("PICKUP:WHEAT")
    return {
        "board": torch.randn(turns + 1, TILE_PLANES, 10, 10),
        "scalars": torch.randn(turns + 1, SCALARS),
        "positions": torch.zeros(turns + 1, MAX_UNITS, dtype=torch.int64),
        "unit_actions": unit_actions,
        "unit_quantities": torch.full((turns, MAX_UNITS), QUANTITIES.index(3)),
        "market_actions": torch.zeros(turns, slots, dtype=torch.int64),
        "unit_masks": torch.ones(turns, MAX_UNITS, len(UNIT_OPS), dtype=torch.bool),
        "unit_quantity_masks": torch.ones(
            turns, MAX_UNITS, len(QUANTITIES), dtype=torch.bool
        ),
        "market_masks": torch.ones(turns, slots, len(QUANTITIES), dtype=torch.bool),
        "log_probs": torch.full((turns,), -1.0),
        "shaped": torch.randn(turns) * 0.01,
        "shaped_money": torch.randn(turns) * 0.01,
        "dones": torch.zeros(turns, dtype=torch.bool),
    }


def _policy() -> tuple[Policy, torch.optim.Optimizer]:
    """Return a tiny policy and its optimizer."""
    policy = Policy(blocks=1, channels=16, value_bound=toad_phase1.VALUE_BOUND)
    return policy, torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS)


def test_the_runner_can_take_a_warmup_step() -> None:
    """During warmup the runner's own step must optimise the baseline alone."""
    policy, optimizer = _policy()
    segments = [_segment() for _ in range(toad_phase1.BATCH_SEGMENTS)]
    warm = toad_phase1._step(
        policy, optimizer, segments, "cpu", "shaped_money", baseline_only=True
    )
    assert warm["total"] == warm["baseline"]


def test_the_runner_takes_a_full_step_after_warmup() -> None:
    """Outside warmup the policy terms must be back in the total."""
    policy, optimizer = _policy()
    segments = [_segment() for _ in range(toad_phase1.BATCH_SEGMENTS)]
    full = toad_phase1._step(
        policy, optimizer, segments, "cpu", "shaped_money", baseline_only=False
    )
    assert full["total"] != full["baseline"]


def _moved(policy: Policy, before: torch.Tensor) -> bool:
    """Return whether the quantity head's weights changed."""
    return not torch.equal(policy.quantity_head.weight, before)


def test_the_runner_trains_the_quantity_head_only_where_a_transfer_acted() -> None:
    """The head has to get a gradient, and only from the slots that spent a bucket.

    Three cases, because two of them can each be passed by a different broken
    runner. A ``_step`` that never scores the quantity head leaves the weights
    where they were on the first case; one that scores it unconditionally moves
    them on the second, where no unit played a ``PICKUP`` at all and the
    engine would have read no count; and one that folds the head into the value
    warmup moves them on the third, which is meant to be the critic alone.
    """
    policy, optimizer = _policy()
    acted = [_segment() for _ in range(toad_phase1.BATCH_SEGMENTS)]
    quiet = [_segment(transfers=False) for _ in range(toad_phase1.BATCH_SEGMENTS)]

    before = policy.quantity_head.weight.detach().clone()
    toad_phase1._step(policy, optimizer, acted, "cpu", "shaped_money")
    trained = _moved(policy, before)

    policy, optimizer = _policy()
    before = policy.quantity_head.weight.detach().clone()
    toad_phase1._step(policy, optimizer, quiet, "cpu", "shaped_money")
    scored_nothing = not _moved(policy, before)

    policy, optimizer = _policy()
    before = policy.quantity_head.weight.detach().clone()
    toad_phase1._step(
        policy, optimizer, acted, "cpu", "shaped_money", baseline_only=True
    )
    warmup_left_it = not _moved(policy, before)

    assert trained
    assert scored_nothing
    assert warmup_left_it


def test_the_teacher_kl_covers_the_quantity_head_only_if_the_teacher_has_one() -> None:
    """A teacher whose checkpoint predates the head has a random one, not a taught one.

    ``Policy`` always *carries* a quantity head, so the object cannot answer
    this question -- only the state dict it was loaded from can, which is why
    ``_teacher`` records the answer beside the network instead of leaving the
    caller to guess. Pulling the learner toward an untrained head is not a
    weaker regulariser than leaving it alone; it is a pull toward noise.

    Measured on the KL term itself rather than on the total, so a run that
    happened to have a small teacher cost could not hide the difference.
    """
    segments = [_segment() for _ in range(toad_phase1.BATCH_SEGMENTS)]
    frozen = Policy(blocks=1, channels=16, value_bound=toad_phase1.VALUE_BOUND).eval()

    covered, uncovered = (
        toad_phase1._step(
            *_policy(),
            segments,
            "cpu",
            "shaped_money",
            teacher=toad_phase1.Teacher(policy=frozen, quantity=carries),
            teacher_kl_cost=1.0,
        )["teacher"]
        for carries in (True, False)
    )

    assert uncovered > 0.0
    assert covered > uncovered


def test_the_warmup_budget_is_counted_down_in_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_update` must report how many batches it consumed, or warmup never ends."""
    policy, optimizer = _policy()

    class _Fake:
        dones = torch.zeros(64, dtype=torch.bool)

    segments = [_segment() for _ in range(8)]
    # `monkeypatch` rather than assign-and-restore: it undoes the patch even if
    # the assertions below raise, which the hand-rolled `try/finally` did too but
    # only as long as nobody edited it.
    monkeypatch.setattr(toad_phase1, "_segments", lambda trajectory: segments)  # noqa: ARG005
    # `_update` reads only `dones` off each trajectory, which is the whole point
    # of the stand-in; it is not a `Trajectory` and does not need to be.
    batch: list[Any] = [_Fake()]
    terms, consumed = toad_phase1._update(
        policy, optimizer, batch, "cpu", "shaped_money", warmup_left=10**9
    )
    assert consumed == len(segments) // toad_phase1.BATCH_SEGMENTS
    # Every batch was inside the warmup budget, so the mean total is the mean baseline.
    assert terms["total"] == terms["baseline"]


def _fitted(segments: list[dict[str, torch.Tensor]], value_passes: int) -> float:
    """Return the value loss left on ``segments`` after one round of training.

    The probe is a copy, so measuring does not train: ``_step`` computes its
    terms at the weights it is handed and only then steps, which makes the
    returned ``baseline`` a readout of the fit rather than of the update.

    Args:
        segments: The round's unrolls.
        value_passes: Extra value-only passes to run inside ``_update``.

    Returns:
        Mean ``baseline`` loss over every batch of ``segments``.
    """
    torch.manual_seed(0)
    policy, optimizer = _policy()

    class _Fake:
        dones = torch.zeros(64, dtype=torch.bool)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(toad_phase1, "_segments", lambda trajectory: segments)  # noqa: ARG005
        batch: list[Any] = [_Fake()]
        toad_phase1._update(
            policy, optimizer, batch, "cpu", "shaped", value_passes=value_passes
        )
    probe = copy.deepcopy(policy)
    spare = torch.optim.Adam(probe.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS)
    losses = [
        toad_phase1._step(
            probe,
            spare,
            segments[start : start + toad_phase1.BATCH_SEGMENTS],
            "cpu",
            "shaped",
            baseline_only=True,
        )["baseline"]
        for start in range(0, len(segments), toad_phase1.BATCH_SEGMENTS)
    ]
    return sum(losses) / len(losses)


def test_extra_value_passes_leave_the_critic_better_fitted() -> None:
    """The point of ``value_passes`` is the fit, so pin the fit and not the count.

    A test that counted ``_step`` calls would pass on an implementation that
    shuffled without stepping, or that stepped a detached value head. This
    measures what the extra passes are for: the value loss left on the same
    round afterwards. Measured 2026-08-15 -- one look per sample is the whole
    gap between a critic at -0.103 within-step and the same target, trained over
    the same data, at +0.276.

    Verified by mutation: making the passes a no-op fails this and nothing else.
    It does **not** catch passes that replay the *full* loss -- the value term is
    inside that total, so the fit improves either way. That property has no
    behavioural signature here and is guarded structurally by
    ``test_the_extra_passes_never_replay_the_policy_gradient``.
    """
    torch.manual_seed(1)
    segments = [_segment() for _ in range(16)]
    assert _fitted(segments, value_passes=5) < _fitted(segments, value_passes=0)


def test_no_value_passes_reproduces_the_earlier_arms() -> None:
    """Zero extra passes must leave the round bit-identical to before the change.

    Ten arms and their controls were run without this knob, and the corrected
    arm they are compared against is one of them. If the default moved them, the
    comparison would be against a configuration that never ran.
    """
    torch.manual_seed(1)
    segments = [_segment() for _ in range(8)]
    torch.manual_seed(0)
    policy, optimizer = _policy()
    reference = [
        toad_phase1._step(
            policy,
            optimizer,
            segments[start : start + toad_phase1.BATCH_SEGMENTS],
            "cpu",
            "shaped",
        )
        for start in range(0, len(segments), toad_phase1.BATCH_SEGMENTS)
    ]
    expected = sum(terms["baseline"] for terms in reference) / len(reference)
    torch.manual_seed(0)
    fresh, fresh_optimizer = _policy()

    class _Fake:
        dones = torch.zeros(64, dtype=torch.bool)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(toad_phase1, "_segments", lambda trajectory: segments)  # noqa: ARG005
        batch: list[Any] = [_Fake()]
        terms, _ = toad_phase1._update(fresh, fresh_optimizer, batch, "cpu", "shaped")
    assert terms["baseline"] == pytest.approx(expected)
    # With no extra passes there is nothing else to report, so the new term must
    # fall back to the round's own baseline rather than to a misleading zero.
    assert terms["baseline_passes"] == pytest.approx(expected)


def test_the_extra_passes_never_replay_the_policy_gradient() -> None:
    """Every extra pass must be value-only, and no behavioural test can say so.

    Replaying a round through the full loss would fit the value head just as
    well -- the baseline term is inside that total -- so the fit test above
    cannot tell the two apart, and it was checked by mutation that it does not.
    What it would change is the algorithm: the policy gradient would take five
    steps on one round of stale actions instead of one, which is off-policy
    repetition V-trace's clipping was never asked to cover. The property is
    structural, so it is pinned structurally rather than left unguarded.
    """
    torch.manual_seed(1)
    segments = [_segment() for _ in range(8)]
    policy, optimizer = _policy()
    seen: list[bool] = []
    real = toad_phase1._step

    def _recording(
        learner: Policy,
        optimizer: torch.optim.Optimizer,
        batch: list[dict[str, torch.Tensor]],
        device: str,
        field: str,
        baseline_only: bool = False,
        teacher: toad_phase1.Teacher | None = None,
        teacher_kl_cost: float = TEACHER_KL_COST,
    ) -> dict[str, float]:
        seen.append(baseline_only)
        return real(
            learner,
            optimizer,
            batch,
            device,
            field,
            baseline_only,
            teacher,
            teacher_kl_cost,
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(toad_phase1, "_step", _recording)
        toad_phase1._value_passes(policy, optimizer, segments, "cpu", "shaped", 3)
    assert seen
    assert all(seen)
