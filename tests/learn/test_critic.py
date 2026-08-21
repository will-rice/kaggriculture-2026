"""The critic's target must know the season ends, and the run must say so.

Ten arms read a falling ``baseline`` term as the critic learning. It was not:
that term is the value head's distance from its own bootstrapped target, and the
target was built so that the *infinite*-horizon extrapolation of the local reward
rate was a fixed point of it. A critic can sit at 0.998 explained variance
against such a target while explaining the real return twelvefold worse than a
constant, and that is exactly what was measured on 2026-08-09.

Two defects put it there, both in ``toad_phase1``'s segmentation rather than in
the vendored loss the ledger's faithfulness audit checked:

* the ragged tail of every episode was dropped, taking the only ``done`` in the
  season with it, so no target ever knew the season ends;
* each segment bootstrapped from a state *inside* itself, so no state outside a
  16-turn window could enter the target and the horizon could never propagate.

These tests pin the property rather than the implementation: the true
finite-horizon return must be a fixed point of the target the runner builds, and
the infinite-horizon extrapolation must not be.
"""

import pytest
import torch

from kaggriculture.learn.critic import critic_scores, explained_variance, monte_carlo
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
from kaggriculture.learn.toad.core import td_lambda
from kaggriculture.learn.toad_loss import ADAM_EPS, DISCOUNTING, LEARNING_RATE, LMB

TURNS = 719
REWARD = 1e-3
SLOTS = len(MARKET_SLOTS) + 2
# A correct critic against a correct target has a smooth-L1 distance of zero, so
# the floor here is float noise over 64 elements of a quadratic; the mutations
# these tests exist to catch land six orders of magnitude above it.
EXACT = 1e-8


def _episode(turns: int = TURNS) -> Trajectory:
    """Return a trajectory with a constant reward and one terminal turn.

    A constant reward is what makes the arithmetic checkable by hand: the true
    finite-horizon return-to-go with ``n`` turns left is ``r (1 - g^n)/(1 - g)``
    and the infinite-horizon extrapolation is ``r / (1 - g)``, which at Toad's
    gamma of 0.999 differ about twofold at turn 0 and eightfold at turn 600.

    Column 0 of ``scalars`` is the turn index, so a test can read which rows of
    the episode a segment was cut from without depending on how ``_segments``
    happens to be written.
    """
    dones = torch.zeros(turns, dtype=torch.bool)
    dones[-1] = True
    rewards = torch.full((turns,), REWARD)
    scalars = torch.zeros(turns, SCALARS)
    scalars[:, 0] = torch.arange(turns, dtype=torch.float32)
    return Trajectory(
        board=torch.zeros(turns, TILE_PLANES, 10, 10),
        scalars=scalars,
        positions=torch.zeros(turns, MAX_UNITS, dtype=torch.int64),
        unit_actions=torch.zeros(turns, MAX_UNITS, dtype=torch.int64),
        market_actions=torch.zeros(turns, SLOTS, dtype=torch.int64),
        unit_masks=torch.ones(turns, MAX_UNITS, len(UNIT_OPS), dtype=torch.bool),
        market_masks=torch.ones(turns, SLOTS, len(QUANTITIES), dtype=torch.bool),
        log_probs=torch.zeros(turns),
        values=torch.zeros(turns),
        rewards=rewards,
        own=rewards,
        shaped=rewards,
        shaped_money=rewards,
        margin=rewards,
        potentials=torch.zeros(turns, 1),
        dones=dones,
        final_margin=0.0,
        final_bank=0.0,
        final_capital=0.0,
        illegal=0,
        sales=0.0,
        units_sold=0.0,
        mean_sale_price=0.0,
        realisation=0.0,
        bought=0.0,
    )


def _rows(segment: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return the episode row index of every state a segment carries."""
    return segment["scalars"][:, 0].to(torch.int64)


def _finite_horizon(turns: int = TURNS) -> torch.Tensor:
    """Return the exact discounted return-to-go of a constant-reward season."""
    remaining = torch.arange(turns, 0, -1, dtype=torch.float32)
    return REWARD * (1.0 - DISCOUNTING**remaining) / (1.0 - DISCOUNTING)


def _targets(values: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return each segment's TD(lambda) target beside the values it was built from.

    The segmentation and the bootstrap come from ``toad_phase1`` itself, so a
    change to either is a change to what these tests measure. The network is
    replaced by an oracle -- ``values`` indexed at each row -- because the
    question is about the target's fixed point and not about what a trunk can fit.
    """
    episode = _episode()
    pairs = []
    for segment in toad_phase1._segments(episode):
        rows = _rows(segment)[: segment["dones"].shape[0]]
        bootstrap = values[int(_rows(segment)[-1])]
        target = td_lambda.td_lambda(
            rewards=segment["margin"],
            values=values[rows],
            bootstrap_value=bootstrap,
            discounts=(~segment["dones"]).float() * DISCOUNTING,
            lmb=LMB,
        ).vs
        pairs.append((target, values[rows]))
    return pairs


def test_the_true_return_is_a_fixed_point_of_the_value_target() -> None:
    """A critic that is exactly right must have nothing to learn.

    This is the property the whole loss depends on and it did not hold: with the
    terminal turn dropped and the bootstrap sealed inside the segment, the true
    finite-horizon return was pushed *away* from wherever it started.
    """
    values = _finite_horizon()
    for target, held in _targets(values):
        assert torch.allclose(target, held, atol=1e-6)


def test_the_infinite_horizon_extrapolation_is_not_a_fixed_point() -> None:
    """The critic the arms actually learned must be visibly wrong to the target.

    ``r / (1 - gamma)`` is what a 16-turn window bootstrapped from itself
    converges to, and it is 1.95x the true return at turn 0 rising to 8.91x at
    turn 600. If the target cannot tell it apart from the truth, nothing in
    training can.
    """
    values = torch.full((TURNS,), REWARD / (1.0 - DISCOUNTING))
    moved = [float((target - held).abs().max()) for target, held in _targets(values)]
    assert max(moved) > 1e-3


def test_every_segment_covers_a_turn_and_the_last_one_ends_the_season() -> None:
    """The terminal turn must be an acted row, or no target ever sees a done."""
    segments = toad_phase1._segments(_episode())
    acted = torch.cat([_rows(segment)[:-1] for segment in segments])
    assert int(acted.max()) == TURNS - 1
    assert any(bool(segment["dones"].any()) for segment in segments)


def test_a_segment_bootstraps_from_the_state_after_its_last_action() -> None:
    """The bootstrap state must be outside the segment, or nothing couples them."""
    segments = toad_phase1._segments(_episode())
    for segment in segments[:-1]:
        rows = _rows(segment)
        assert int(rows[-1]) == int(rows[-2]) + 1
        assert segment["scalars"].shape[0] == segment["dones"].shape[0] + 1
    # The season's last segment has no state after it, and its done deletes the
    # bootstrap from the target, so repeating the final state is safe.
    assert int(_rows(segments[-1])[-1]) == TURNS - 1


class _Oracle(Policy):
    """A critic whose value head emits a prescribed number for each turn.

    ``_step`` gives nothing back about the target it built except the
    ``baseline`` term, and that is enough to pin it: smooth L1 between a critic
    and a target is zero exactly when the two agree, so feeding a critic we
    already know the right answer for turns the loss into a readout of the
    target. It subclasses ``Policy`` because that is what the runner is typed to
    take, and it keeps a real gradient path so the runner's backward pass,
    gradient clipping and optimizer step all run exactly as they do in training.
    """

    def __init__(self, values: torch.Tensor) -> None:
        """Build the oracle around one value per turn of the season."""
        super().__init__(blocks=1, channels=8, value_bound=None)
        self.values = values

    def forward(
        self, board: torch.Tensor, scalars: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return flat logits and the prescribed value for each row's turn."""
        rows = board.shape[0]
        slack = self.value.bias.sum() * 0.0
        turn = scalars[:, 0].to(torch.int64)
        return (
            torch.zeros(rows, MAX_UNITS, len(UNIT_OPS)) + slack,
            torch.zeros(rows, MAX_UNITS, len(QUANTITIES)) + slack,
            torch.zeros(rows, SLOTS, len(QUANTITIES)) + slack,
            self.values[turn] + slack,
        )


def _baseline(values: torch.Tensor, last: bool = True) -> float:
    """Return the ``baseline`` term the runner's own ``_step`` builds for a critic.

    This goes through ``toad_phase1._step``, not through a reconstruction of it,
    so it pins what the runner **consumes**. Asserting on what ``_segments``
    *produces* is not the same thing and does not catch a ``_step`` that ignores
    the state it is handed.

    Args:
        values: One value per turn of the season, the critic's output.
        last: Score the segments that end the season rather than the ones that
            open it. The terminal is where a target either knows the horizon or
            does not.

    Returns:
        The smooth-L1 value loss over one batch of segments.
    """
    segments = toad_phase1._segments(_episode())
    batch = segments[-toad_phase1.BATCH_SEGMENTS :] if last else segments[:4]
    oracle = _Oracle(values)
    optimizer = torch.optim.Adam(oracle.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS)
    return toad_phase1._step(oracle, optimizer, batch, "cpu", "margin")["baseline"]


def test_the_runner_leaves_a_correct_critic_alone() -> None:
    """A critic that is exactly right must have nothing to learn *through _step*.

    This is the end-to-end form of the fixed-point property, and it is the one
    that catches a bootstrap taken from inside the segment: discarding the
    after-state before reading it moves the target off the true return by ~1e-3
    a row, which is six orders of magnitude above the floor here.
    """
    assert _baseline(_finite_horizon()) < EXACT
    assert _baseline(_finite_horizon(), last=False) < EXACT


def test_the_runner_rejects_the_infinite_horizon_critic() -> None:
    """The critic the arms actually learned must be visibly wrong *through _step*.

    ``r / (1 - gamma)`` is a zero-error fixed point of a target that never sees
    a ``done``, so this is the assertion that goes red if the season's terminal
    turn stops reaching the learner -- which is exactly what dropping the ragged
    tail did.
    """
    assert _baseline(torch.full((TURNS,), REWARD / (1.0 - DISCOUNTING))) > EXACT * 100.0


def test_monte_carlo_matches_a_hand_rolled_backward_scan() -> None:
    """The ground truth must not disagree with a scan anyone can check by eye."""
    episode = _episode()
    scanned = torch.zeros(TURNS)
    running = 0.0
    for turn in range(TURNS - 1, -1, -1):
        running = float(episode.margin[turn]) + (
            0.0 if bool(episode.dones[turn]) else DISCOUNTING * running
        )
        scanned[turn] = running
    assert torch.allclose(
        monte_carlo(episode.margin, episode.dones), scanned, atol=1e-5
    )
    assert torch.allclose(
        monte_carlo(episode.margin, episode.dones), _finite_horizon(), atol=1e-5
    )


def test_explained_variance_is_zero_for_a_critic_that_knows_only_the_mean() -> None:
    """Zero is the score to beat, so a negative reading means worse than constant.

    Seeded, and compared to a tolerance rather than exactly: the mean is
    subtracted in floating point, so the residual is zero only to rounding, and
    an exact comparison passes or fails on whatever RNG state the rest of the
    suite happened to leave behind.
    """
    returns = torch.randn(256, generator=torch.Generator().manual_seed(0))
    assert explained_variance(
        returns, torch.full_like(returns, float(returns.mean()))
    ) == pytest.approx(0.0, abs=1e-6)
    assert explained_variance(returns, -returns) < 0.0
    assert explained_variance(returns, returns) == pytest.approx(1.0, abs=1e-6)


def test_the_round_reports_explained_variance_against_the_real_return() -> None:
    """A run must log the critic's accuracy, not only its self-consistency.

    Without this the only critic number on the dashboard is ``baseline``, which
    is what ten arms read as progress while the critic explained nothing.
    """
    episode = _episode()
    values = _finite_horizon()
    scores = critic_scores(
        [values, values * 0.5], [episode.margin] * 2, [episode.dones] * 2
    )
    assert scores["ev_vs_return"] > 0.0
    assert set(scores) == {"ev_vs_return", "ev_vs_return_within_turn"}
    assert all(key in critic_scores([], [], []) for key in scores)
