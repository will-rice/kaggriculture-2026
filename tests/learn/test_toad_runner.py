"""Tests that go through the runner, not just the loss it calls.

The value warmup shipped with ``baseline_only`` passed to a ``_step`` that did
not accept it. Every loss test still passed, because they call ``losses``
directly and never cross the runner boundary -- so arm C' launched, crashed on
its first batch, and burned the slot. These call what the runner calls.
"""

import copy
from pathlib import Path
from typing import Any, cast

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
from kaggriculture.learn.rollout import Trajectory
from kaggriculture.learn.scripts import toad_phase1
from kaggriculture.learn.toad_loss import (
    ADAM_EPS,
    ENTROPY_COST,
    LEARNING_RATE,
    TEACHER_KL_COST,
)


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


def _trajectory(turns: int = 4, final_margin: float = 1.0) -> Trajectory:
    """Return a minimal ``Trajectory``, shaped exactly as ``rollout_many`` builds one.

    Only what ``_record``, ``_population`` and ``_critic`` read needs to vary
    between calls, so every other field is a fixed, cheap stand-in; the point
    of this fixture is the shape, not the values.

    Args:
        turns: Acted rows.
        final_margin: This seat's terminal margin, so callers can build both a
            decisive seat and a tied one.

    Returns:
        One synthetic recorded seat.
    """
    slots = len(MARKET_SLOTS) + 2
    dones = torch.zeros(turns, dtype=torch.bool)
    dones[-1] = True
    rewards = torch.full((turns,), 0.1)
    return Trajectory(
        board=torch.zeros(turns, TILE_PLANES, 10, 10),
        scalars=torch.zeros(turns, SCALARS),
        positions=torch.zeros(turns, MAX_UNITS, dtype=torch.int64),
        unit_actions=torch.zeros(turns, MAX_UNITS, dtype=torch.int64),
        unit_quantities=torch.ones(turns, MAX_UNITS, dtype=torch.int64),
        market_actions=torch.zeros(turns, slots, dtype=torch.int64),
        unit_masks=torch.ones(turns, MAX_UNITS, len(UNIT_OPS), dtype=torch.bool),
        unit_quantity_masks=torch.ones(
            turns, MAX_UNITS, len(QUANTITIES), dtype=torch.bool
        ),
        market_masks=torch.ones(turns, slots, len(QUANTITIES), dtype=torch.bool),
        log_probs=torch.zeros(turns),
        values=torch.zeros(turns),
        rewards=rewards,
        own=rewards,
        shaped=rewards,
        shaped_money=rewards,
        margin=rewards,
        potentials=torch.zeros(turns, 1),
        dones=dones,
        final_margin=final_margin,
        final_bank=10.0,
        final_capital=2.0,
        illegal=0,
        sales=1.0,
        units_sold=1.0,
        mean_sale_price=1.0,
        realisation=1.0,
        bought=1.0,
    )


def _record(
    mirror_batch: list[Trajectory], econ_batch: list[Trajectory]
) -> dict[str, object]:
    """Call ``_record`` with the counters and loss terms it does not test."""
    terms = {
        "vtrace_pg": 0.0,
        "upgo_pg": 0.0,
        "baseline": 0.0,
        "entropy": 0.0,
        "teacher": 0.0,
        "total": 0.0,
        "baseline_passes": 0.0,
    }
    return toad_phase1._record(
        mirror_batch,
        econ_batch,
        "shaped",
        update=1,
        steps=10,
        hours=0.1,
        lr=1e-4,
        warming=False,
        warmup_left=0,
        terms=terms,
    )


def test_every_logged_metric_carries_a_role_prefix() -> None:
    """Every key ``_record`` returns must be grouped under one of the four roles.

    wandb sections a dashboard by the text before a key's first ``/``, which is
    the entire mechanism that lets a reader see what role a number plays
    before its name is read. A metric added without a prefix falls outside
    every section and outside this guarantee, so this pins the invariant
    structurally rather than trusting the next edit to remember it.

    It also pins that every logged key has a one-line definition shipped into
    the run (``_log_definitions``) -- the fix for THE INCIDENT, where the
    correct reading of two metrics existed only in a source comment.
    """
    record = _record(
        mirror_batch=[_trajectory(), _trajectory()], econ_batch=[_trajectory()]
    )
    assert record
    assert all(key.startswith(toad_phase1.METRIC_PREFIXES) for key in record)
    assert set(record) == set(toad_phase1.METRIC_DEFINITIONS)


def test_mirror_decisive_rate_is_the_raw_win_rate_doubled() -> None:
    """``win_rate_mirror`` was 0.5 by construction; the logged key must not be.

    Self-play records both seats and exactly one wins, so the raw decisive
    fraction has a hard 0.5 ceiling. Doubling it is what turns that into a
    real 0-1 read -- "all decisive" rather than "as decisive as self-play can
    ever look" -- and this pins the factor of two rather than trusting a
    future edit not to drop it.
    """
    # One seat of every self-play episode wins (`final_margin > 0`) and the
    # other loses, so the raw rate over both seats tops out at 0.5 -- exactly
    # what "every game decisive, no ties" looks like. Doubled, that is 1.0.
    decisive = [_trajectory(final_margin=1.0), _trajectory(final_margin=-1.0)]
    record = _record(mirror_batch=decisive, econ_batch=[])
    assert record["diag/mirror_decisive_rate"] == pytest.approx(1.0)

    tied = [_trajectory(final_margin=0.0), _trajectory(final_margin=0.0)]
    record = _record(mirror_batch=tied, econ_batch=[])
    assert record["diag/mirror_decisive_rate"] == pytest.approx(0.0)


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


def test_no_teacher_flag_means_no_teacher() -> None:
    """Phase 1 runs teacher-free; the recipe sets teacher_kl_cost to 0.

    Every prior arm passed --teacher in phase 1 at phase 2's cost, anchoring a
    from-random policy to a behaviour clone the recipe uses nowhere. The
    default must be no teacher at all, not a clone.
    """
    arguments = toad_phase1._parser().parse_args([])
    assert arguments.teacher is None
    assert toad_phase1._teacher(arguments, "cpu") is None


def test_teacher_loads_the_named_checkpoint(tmp_path: Path) -> None:
    """A phase names its teacher; phases 2+ each anchor to a different one.

    Asserted on the loaded parameters themselves, not on whether a
    ``Teacher`` came back: a ``_teacher`` still hardwired to
    ``kaggriculture.learn.CHECKPOINT`` also returns one, so only comparing
    values against the path actually named tells the two apart.

    Also asserted frozen: a KL anchor that is not held fixed is not an
    anchor. ``Policy`` has no BatchNorm or Dropout, so train/eval mode has no
    numeric effect, and the teacher never enters the optimiser's param
    groups -- so dropping either ``.eval()`` or ``.requires_grad_(False)``
    inside ``_teacher`` would pass every check above while leaving the
    teacher free to drift.
    """
    named = Policy(
        blocks=toad_phase1.BLOCKS, channels=16, value_bound=toad_phase1.VALUE_BOUND
    )
    checkpoint = tmp_path / "phase1_final.pt"
    torch.save(named.state_dict(), checkpoint)

    arguments = toad_phase1._parser().parse_args(
        ["--channels", "16", "--teacher", str(checkpoint)]
    )
    teacher = toad_phase1._teacher(arguments, "cpu")

    assert teacher is not None
    named_state = named.state_dict()
    loaded_state = teacher.policy.state_dict()
    assert named_state.keys() == loaded_state.keys()
    for key, value in named_state.items():
        assert torch.equal(value, loaded_state[key]), key

    assert not teacher.policy.training
    assert not any(parameter.requires_grad for parameter in teacher.policy.parameters())


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


def test_the_default_arm_trains_on_the_reward_that_prices_coins() -> None:
    """The score constituent must be inside the dense reward the learner reads.

    In Lux, ``city`` *is* the score -- ``GameResultReward.compute_player_reward``
    ranks on ``city_tile_count * 10000 + unit_count`` -- so their heaviest
    per-step component pays, densely, for the very quantity that decides their
    game. Here the score is coins banked, and a default that omits the coin
    delta leaves our shaped reward with no dense term on the score at all. That
    is not a sparser reproduction of their recipe; it is a reproduction with
    their largest component deleted.

    Asserted through ``_step``, which is where a field name becomes the tensor
    the loss regresses on. ``_field`` returning the right string while ``_step``
    stacked another series would pass a namespace check.
    """
    segments = [_segment() for _ in range(2)]
    policy = Policy(blocks=1, channels=16, value_bound=toad_phase1.VALUE_BOUND)

    def run(field: str) -> dict[str, float]:
        learner = copy.deepcopy(policy)
        optimizer = torch.optim.Adam(
            learner.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS
        )
        return toad_phase1._step(learner, optimizer, segments, "cpu", field)

    default = run(toad_phase1._field(toad_phase1._parser().parse_args([])))
    assert default == run("shaped_money")
    assert default != run("shaped")


def test_the_money_free_arm_is_still_one_flag_away() -> None:
    """The ablation must remain runnable in the direction that omits the coins.

    Toad's own component set, with nothing paying for the score, is the control
    this reproduction is measured against; making it the default was the error,
    making it unreachable would be a second one.
    """
    arguments = toad_phase1._parser().parse_args(["--no-money"])
    assert toad_phase1._field(arguments) == "shaped"


def test_omitting_lr_and_entropy_cost_reproduces_todays_constants() -> None:
    """Every existing invocation must behave identically once the flags exist.

    A sweep is worthless if adding its own knobs quietly moved the arms that
    do not pass them, so the flags' defaults must be the constants they
    override -- not a copy of today's value that could drift from them.
    """
    arguments = toad_phase1._parser().parse_args([])
    assert arguments.lr == LEARNING_RATE
    assert arguments.entropy_cost == ENTROPY_COST


def test_lr_flag_reaches_the_optimizer() -> None:
    """``--lr`` must set the rate the optimizer actually steps with.

    Asserted on the optimizer's own ``param_groups``, not on the parsed
    namespace: a runner that parses ``--lr`` and builds the optimizer from
    ``LEARNING_RATE`` regardless would still pass a namespace-only check.
    """
    policy = Policy(blocks=1, channels=16, value_bound=toad_phase1.VALUE_BOUND)
    arguments = toad_phase1._parser().parse_args(["--lr", "3e-4"])
    optimizer = toad_phase1._optimizer(policy, arguments.lr)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)
    assert optimizer.param_groups[0]["lr"] != LEARNING_RATE


def test_lr_flag_moves_the_schedules_own_output() -> None:
    """``--lr`` must set the base the ``LambdaLR`` schedule scales, not be shadowed.

    ``_decay`` returns a multiplier relative to whatever base the optimizer
    was built with, so a schedule that silently pinned its own base rate
    would report the same current rate regardless of ``--lr``. Two policies,
    two bases, the same schedule shape, and the ratio between the two must
    survive the schedule's first step.
    """
    default_policy = Policy(blocks=1, channels=16, value_bound=toad_phase1.VALUE_BOUND)
    swept_policy = Policy(blocks=1, channels=16, value_bound=toad_phase1.VALUE_BOUND)
    default_optimizer = toad_phase1._optimizer(default_policy, LEARNING_RATE)
    swept_optimizer = toad_phase1._optimizer(swept_policy, 3e-4)
    default_schedule = torch.optim.lr_scheduler.LambdaLR(
        default_optimizer, toad_phase1._decay(0.0)
    )
    swept_schedule = torch.optim.lr_scheduler.LambdaLR(
        swept_optimizer, toad_phase1._decay(0.0)
    )
    default_schedule.step()
    swept_schedule.step()
    ratio = swept_schedule.get_last_lr()[0] / default_schedule.get_last_lr()[0]
    assert ratio == pytest.approx(3e-4 / LEARNING_RATE)


def test_entropy_cost_flag_reaches_the_loss() -> None:
    """``--entropy-cost`` must reach ``losses``, not stop at the parsed namespace.

    Zeroing the coefficient must zero the reported entropy term exactly; a
    ``_step`` that parses the flag and keeps computing with ``ENTROPY_COST``
    would report the same nonzero term either way.
    """
    torch.manual_seed(0)
    policy, optimizer = _policy()
    segments = [_segment() for _ in range(toad_phase1.BATCH_SEGMENTS)]
    default_terms = toad_phase1._step(
        policy, optimizer, segments, "cpu", "shaped_money"
    )
    assert default_terms["entropy"] != 0.0

    torch.manual_seed(0)
    policy, optimizer = _policy()
    zeroed_terms = toad_phase1._step(
        policy, optimizer, segments, "cpu", "shaped_money", entropy_cost=0.0
    )
    assert zeroed_terms["entropy"] == 0.0


def test_lr_and_entropy_cost_reach_the_recorded_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A swept hyperparameter absent from the run's config is useless for analysis.

    ``--teacher-kl-cost`` and ``--value-passes`` are recorded the same way;
    this pins ``--lr`` and ``--entropy-cost`` alongside them without touching
    the network by faking ``wandb.init`` and capturing what it was called with.
    """
    captured: dict[str, object] = {}

    class _FakeRun:
        url = "http://example.invalid/fake-run"

        def log(self, *_args: object, **_kwargs: object) -> None:
            pass

    def _fake_init(**kwargs: object) -> _FakeRun:
        captured.update(kwargs)
        return _FakeRun()

    monkeypatch.setattr(toad_phase1.wandb, "init", _fake_init)
    arguments = toad_phase1._parser().parse_args(
        ["--lr", "3e-4", "--entropy-cost", "0.02"]
    )
    toad_phase1._start_run(arguments, "shaped")
    config = cast(dict[str, object], captured["config"])
    assert config["lr"] == pytest.approx(3e-4)
    assert config["entropy_cost"] == pytest.approx(0.02)
