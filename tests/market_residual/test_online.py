"""What the online phase must be true of before it is allowed to spend a GPU.

The three properties pinned here are the ones whose failure would be silent.

**The paired reward is a difference on one cell.** A reward that compared the
candidate against anything but the frozen controller on the same seed, seat and
opponent would be measuring the cell as much as the policy.

**Only market events carry policy rows.** The residual is asked for an opinion
on roughly seven of every ten turns; a trajectory that carried the other three
would put a policy gradient on turns the residual never decided.

**A freshly initialised policy defers.** This is the floor the whole run rests
on: at zero activation the served agent *is* the frozen controller, which
scores exactly 0.5 against itself, so every subsequent point is a real gain
rather than a recovery. Every offline checkpoint we hold loses 128 of 128
games against that same controller, which is why this run starts here instead.
"""

from dataclasses import replace
from pathlib import Path
from typing import Sequence

import pytest
import torch

from kaggriculture.learn.market_residual import online
from kaggriculture.learn.market_residual.model import MarketResidualNet, ModelConfig
from kaggriculture.learn.market_residual.online import (
    Cell,
    EventTrajectory,
    OnlineConfig,
    PairedEpisode,
    collate_paired_episodes,
    initialize_deferring,
    terminal_reward,
    train_online,
    vtrace_market_loss,
)
from kaggriculture.market_residual.features import MarketFeatureVector
from kaggriculture.market_residual.schema import ALLOWED_SLOTS, MarketFeatureSchema

WIDTH = MarketFeatureSchema.current().width
SLOTS = len(ALLOWED_SLOTS)


def trajectory(
    events: int,
    steps: Sequence[int] | None = None,
    generator: torch.Generator | None = None,
) -> EventTrajectory:
    """Return a synthetic trajectory of the declared width.

    Args:
        events: How many market events the episode opened.
        steps: The engine turns those events happened on.
        generator: The source of the feature values.

    Returns:
        One trajectory, deferring at every event.
    """
    return EventTrajectory(
        steps=tuple(range(events)) if steps is None else tuple(steps),
        features=torch.rand(events, WIDTH, generator=generator),
        modes=torch.zeros(events, dtype=torch.int64),
        buckets=torch.zeros(events, SLOTS, dtype=torch.int64),
        behavior_log_prob=torch.full((events,), -0.01),
        executed=torch.ones(events, dtype=torch.bool),
    )


def paired_episode(
    candidate_points: float,
    kaito_points: float,
    margin: float,
    events: int = 4,
    steps: Sequence[int] | None = None,
    generator: torch.Generator | None = None,
) -> PairedEpisode:
    """Return a paired cell whose two arms scored what the caller asked for.

    The banks are chosen so that ``win_points`` and ``normalized_margin`` read
    back exactly the requested numbers, because the point of the assertion is
    the arithmetic on top of them, not a second copy of that arithmetic.

    Args:
        candidate_points: The candidate arm's win points.
        kaito_points: The control arm's win points.
        margin: The normalized bank margin delta between the two arms.
        events: How many market events the candidate arm opened.
        steps: The engine turns those events happened on.
        generator: The source of the feature values.

    Returns:
        The paired episode.
    """
    return PairedEpisode(
        opponent="v56",
        seed=870_000,
        seat=0,
        trajectory=trajectory(events, steps, generator),
        candidate_points=candidate_points,
        control_points=kaito_points,
        normalized_margin_delta=margin,
        candidate_bank=0,
        control_bank=0,
        replacements=0,
        merged=0,
        fallbacks={},
    )


def test_reward_is_candidate_minus_kaito_on_same_cell() -> None:
    """The reward is the paired win-point difference plus a scaled margin.

    The margin term is bounded by 0.2 and the smallest nonzero win-point
    difference is 0.5, so it can separate two cells that tied on win points
    and can never turn a decided game into its opposite.
    """
    pair = paired_episode(candidate_points=1.0, kaito_points=0.0, margin=0.2)
    assert terminal_reward(pair) == pytest.approx(1.02)
    assert terminal_reward(paired_episode(0.5, 0.5, 0.0)) == pytest.approx(0.0)
    assert terminal_reward(paired_episode(0.0, 1.0, -2.0)) == pytest.approx(-1.2)


def test_non_event_turns_create_no_policy_rows() -> None:
    """A trajectory holds one row per market event, at the turn it happened."""
    episode = paired_episode(1.0, 0.0, 0.0, events=3, steps=(4, 20, 44))
    assert episode.trajectory.steps == (4, 20, 44)
    assert episode.trajectory.features.shape == (3, WIDTH)
    batch = collate_paired_episodes([episode])
    assert batch.valid.shape == (3, 1)
    assert bool(batch.valid.all())


def test_vtrace_clips_raw_importance_overflow() -> None:
    """An importance ratio that overflows float is clipped, not propagated."""
    model = MarketResidualNet(ModelConfig.current())
    episode = paired_episode(1.0, 0.0, 0.0, events=6)
    batch = collate_paired_episodes([episode])
    batch = online.replace_behavior_log_prob(batch, -1000.0)
    report = vtrace_market_loss(batch, model)
    assert torch.isfinite(report.total)
    assert report.max_rho == pytest.approx(1.0)


def test_a_refused_replacement_earns_no_policy_gradient() -> None:
    """A proposal the merge boundary refused never reached the board.

    Without this the run has a free action: an illegal proposal plays exactly
    what deferring plays, so it scores at the mean of the episodes that changed
    nothing, which is above the mean of all episodes because the average
    deviation is harmful. The policy would then learn to propose illegal
    replacements everywhere, and the activation number would stop meaning
    anything.
    """
    episode = paired_episode(1.0, 0.0, 0.0, events=4)
    refused = EventTrajectory(
        steps=episode.trajectory.steps,
        features=episode.trajectory.features,
        modes=torch.tensor([0, 1, 1, 0]),
        buckets=episode.trajectory.buckets,
        behavior_log_prob=episode.trajectory.behavior_log_prob,
        executed=torch.tensor([True, False, True, True]),
    )
    batch = collate_paired_episodes([replace(episode, trajectory=refused)])
    assert batch.pg_weight[:, 0].tolist() == [1.0, 0.0, 1.0, 1.0]


def test_the_terminal_step_stops_the_return_and_padding_cannot_leak() -> None:
    """Discounting ends the episode, so a shorter arm cannot read a longer one."""
    batch = collate_paired_episodes(
        [
            paired_episode(1.0, 0.0, 0.0, events=2),
            paired_episode(0.0, 1.0, 0.0, events=5),
        ]
    )
    assert batch.valid[:, 0].tolist() == [True, True, False, False, False]
    assert batch.discounts[:, 0].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert batch.rewards[:, 0].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]
    assert batch.rewards[:, 1].tolist() == [0.0, 0.0, 0.0, 0.0, -1.0]


def test_the_auxiliary_target_is_the_next_event_read_off_the_schema() -> None:
    """The auxiliary head is regressed against the next event's own columns."""
    schema = MarketFeatureSchema.current()
    episode = paired_episode(1.0, 0.0, 0.0, events=3)
    batch = collate_paired_episodes([episode])
    features = episode.trajectory.features
    assert batch.auxiliary_valid[:, 0].tolist() == [True, True, False]
    expected = online.auxiliary_targets(features)
    assert torch.equal(batch.auxiliary_target[0, 0], expected[1])
    assert float(expected[1][0]) == pytest.approx(
        float(features[1][schema.span("price:CARROT")][0])
    )


def test_a_fresh_policy_defers_over_a_real_season() -> None:
    """The initialised policy plays the frozen controller, and is proved to.

    Not "activation is small on a fixture" -- a whole real season against the
    served controller, sampling exactly as collection samples, with the banks
    checked equal. Zero activation means the wrapper returned the controller's
    own action on every turn, and the tie is what the frozen controller scores
    against itself.
    """
    config = OnlineConfig(updates=1, episodes_per_update=1, device="cpu")
    model = MarketResidualNet(ModelConfig.current())
    initialize_deferring(model, config)
    result = online.play_candidate(
        model,
        Cell(opponent="v56", path=online.SERVED_PATH, seed=870_000, seat=0),
        seed=0,
    )
    assert result.trajectory.features.shape[0] > 400
    assert result.replacements / result.trajectory.features.shape[0] < 0.01
    assert result.candidate_bank == result.opponent_bank


def test_a_fresh_policy_proposes_the_one_order_the_board_always_admits(
    event_rows: tuple[MarketFeatureVector, ...],
) -> None:
    """Exploration starts legal and small, on the rows of a real season.

    Without this the mode bias alone would look like a working initialisation
    while every replacement it ever sampled was refused: a head left uniform
    over 21 buckets in each of 11 allowed slots asks for eleven orders at once,
    and 59% of those slots admit no legal quantity at all at the moment they are
    looked at. The all-zero proposal is the one the merge boundary can always
    prove, so it is the mode, and the nonzero draws are tilted toward the small
    quantities a board is most likely to hold stock for.
    """
    config = OnlineConfig(updates=1, episodes_per_update=1, device="cpu")
    model = MarketResidualNet(ModelConfig.current())
    initialize_deferring(model, config)
    features = torch.tensor([[row.values] for row in event_rows[:64]])
    with torch.no_grad():
        heads, _state = model(features, None)

    replace = torch.softmax(heads.mode_logits, dim=-1)[..., 1]
    assert float(replace.max()) < 0.01

    quantities = torch.softmax(heads.quantity_logits, dim=-1)
    assert float(quantities[..., 0].min()) > 0.9
    assert float(quantities[..., 0].prod(dim=-1).mean()) > 0.5
    assert float(quantities[..., 1].mean()) > 5 * float(quantities[..., -1].mean())


def test_split_resume_matches_uninterrupted(tmp_path: Path) -> None:
    """Two updates in one process equal the same two across a restart.

    The whole point of an owned root is that an interruption is not a silent
    change of run. Equality is checked on the parameters themselves, because a
    resume that restored the model but not the optimizer or the RNG would still
    produce a plausible-looking loss curve.
    """
    assert run_two_updates(tmp_path / "split", split=True) == run_two_updates(
        tmp_path / "whole", split=False
    )


def run_two_updates(root: Path, split: bool) -> list[float]:
    """Run two updates, optionally across a simulated restart.

    Args:
        root: A fresh online root.
        split: Whether to stop after one update and resume in a new call.

    Returns:
        The trained parameters, flattened.
    """
    generator = torch.Generator().manual_seed(11)
    fixed = [
        paired_episode(1.0, 0.0, 0.2, events=5, generator=generator),
        paired_episode(0.0, 1.0, -0.2, events=7, generator=generator),
    ]

    def run(stop_after: int | None) -> MarketResidualNet:
        """Train two updates, optionally losing the process after the first."""

        def collect(update: int, chosen: Sequence[Cell]) -> tuple[PairedEpisode, ...]:
            del chosen
            if stop_after is not None and update > stop_after:
                raise InterruptedRunError
            return tuple(fixed)

        torch.manual_seed(3)
        model = MarketResidualNet(ModelConfig.current())
        config = OnlineConfig(updates=2, episodes_per_update=2, device="cpu", seed=5)
        initialize_deferring(model, config)
        try:
            train_online(model, config, root, cells(), collect)
        except InterruptedRunError:
            pass
        return model

    if split:
        run(stop_after=1)
    model = run(stop_after=None)
    trained = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    return [round(float(value), 6) for value in trained]


def test_a_root_refuses_to_continue_a_run_it_was_not_trained_as(
    tmp_path: Path,
) -> None:
    """Reopening a root under other cells or other hyperparameters is refused.

    The checkpoint would load and the loss would look ordinary, so the run
    would silently become a different experiment measured on the first one's
    history.
    """
    generator = torch.Generator().manual_seed(11)
    fixed = (paired_episode(1.0, 0.0, 0.2, events=5, generator=generator),)

    def collect(update: int, chosen: Sequence[Cell]) -> tuple[PairedEpisode, ...]:
        del update, chosen
        return fixed

    def run(settings: OnlineConfig, bank: Sequence[Cell]) -> None:
        torch.manual_seed(3)
        model = MarketResidualNet(ModelConfig.current())
        initialize_deferring(model, settings)
        train_online(model, settings, tmp_path, bank, collect)

    settings = OnlineConfig(updates=1, episodes_per_update=1, device="cpu", seed=5)
    run(settings, cells())
    with pytest.raises(online.OnlineDriftError):
        run(settings, cells()[:1])
    with pytest.raises(online.OnlineDriftError):
        run(
            OnlineConfig(
                updates=1,
                episodes_per_update=1,
                device="cpu",
                seed=5,
                learning_rate=1e-2,
            ),
            cells(),
        )


class InterruptedRunError(Exception):
    """Stands in for the process dying between two updates."""


def cells() -> tuple[Cell, ...]:
    """Return a two-cell bank that needs no engine to enumerate."""
    return (
        Cell(opponent="v56", path=online.SERVED_PATH, seed=870_000, seat=0),
        Cell(opponent="v56", path=online.SERVED_PATH, seed=870_001, seat=1),
    )
