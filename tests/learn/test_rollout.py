"""Tests for the self-play rollouts PPO learns from.

Every test here plays whole episodes, which is what makes them worth running:
the failures they exist for -- a trajectory that stops short of the season, a
reward that does not telescope, a log-probability from a distribution nobody
sampled -- are all invisible on a single turn.

The policy is deliberately a small one. ``rollout`` reads every shape it needs
from ``encoding``'s constants and nothing from ``model``'s ``BLOCKS`` or
``CHANNELS``, so a one-block trunk exercises the identical code path at about a
fifth of the cost, and ``tests/learn/test_model.py`` is where the shipped trunk
is pinned. At the full 8x256 the three rollouts below would cost a minute of
wall clock each and would end up behind ``pytest.mark.slow``, which is where
guards go to stop running.
"""

import functools

import pytest
import torch
from kaggle_environments import make

from kaggriculture.constants import ENVIRONMENT, EPISODE_STEPS, TURNS_PER_DAY
from kaggriculture.learn.encoding import (
    IGNORE,
    MARKET_SLOTS,
    MAX_UNITS,
    PRODUCT_NAMES,
    QUANTITIES,
    UNIT_OPS,
    decode_market,
    decode_units,
    encode_board,
    encode_positions,
    encode_scalars,
    unit_count,
)
from kaggriculture.learn.mask import market_mask, unit_mask
from kaggriculture.learn.model import Policy
from kaggriculture.learn.rollout import Trajectory, rollout

# Far enough in that the two farms have diverged. The opening position is
# identical for both seats -- same tiles, same money, same empty shed -- so a
# rollout that encoded the opponent's seat would reproduce ours exactly on turn
# 0 and every assertion about it would pass. That symmetry is how a transposed
# coordinate survived two phases of this project.
DIVERGED = 200

# Where our money and theirs sit in ``encode_scalars``' vector: it opens with a
# price and an inventory signal per product and puts the two banks immediately
# after, each divided by 10,000. Written from that layout rather than searched
# for, so a reordering of the scalars fails this file loudly instead of quietly
# comparing the reward against a shop flag.
MONEY = 2 * len(PRODUCT_NAMES)


@functools.lru_cache(maxsize=1)
def _untrained() -> Policy:
    """Return a seeded, untrained policy, built once for the whole module.

    Untrained on purpose. These tests are about whether a rollout records what
    it played, not about whether the play is any good, and the behaviour-cloned
    checkpoint on disk banks nothing and hires on nearly every turn -- a test
    that leaned on its choices would be pinning that bug.

    Seeded so the episode is the same one every run: several tests below read
    specific turns of it.
    """
    torch.manual_seed(0)
    return Policy(blocks=1, channels=32).eval()


@pytest.fixture(scope="module")
def trajectory() -> Trajectory:
    """Return one episode against the environment's own starter agent."""
    return rollout(_untrained(), "starter", seed=0)


def test_a_trajectory_covers_every_acting_turn() -> None:
    """719 decisions; a short trajectory silently truncates the season."""
    trajectory = rollout(_untrained(), "starter", seed=0)

    assert len(trajectory.rewards) == 719


def test_reward_is_the_change_in_bank_differential() -> None:
    """Dense, sums to the terminal margin, and is the win condition itself.

    Terminal bank alone is one scalar after 719 decisions. This is the same
    quantity, delivered per turn.
    """
    trajectory = rollout(_untrained(), "starter", seed=0)

    assert trajectory.rewards.sum() == pytest.approx(trajectory.final_margin, abs=1.0)


def test_sampled_actions_are_always_legal() -> None:
    """The mask is applied before sampling, not checked after."""
    trajectory = rollout(_untrained(), "starter", seed=0)

    assert trajectory.illegal == 0


def test_the_masks_forbid_most_of_the_action_space(trajectory: Trajectory) -> None:
    """``illegal == 0`` says nothing if the mask permits everything.

    A mask stored as all-True would satisfy every legality assertion in this
    file while gating no logit at all, and the policy would spend its budget
    rediscovering that it cannot HARVEST bare soil. So the stored masks are
    checked to be doing work: most of the space is forbidden on a typical turn,
    and no row is ever empty, which is the condition ``-inf`` masking needs to
    avoid softmaxing a row to NaN.
    """
    assert trajectory.unit_masks.float().mean() < 0.5
    assert trajectory.market_masks.float().mean() < 0.5
    assert bool(trajectory.unit_masks.any(dim=-1).all())
    assert bool(trajectory.market_masks.any(dim=-1).all())


def test_the_stored_log_prob_is_the_masked_distribution_s(
    trajectory: Trajectory,
) -> None:
    """PPO's ratio is ``exp(new - old)``; this is the state where it must be 1.

    The update recomputes the log-probability from the stored state, the stored
    mask and the stored action. If the rollout stored a log-probability from
    the *unmasked* softmax instead -- sampling from the masked distribution and
    then reading the probability off the wrong one -- every ratio in the first
    epoch would be off by the mask's surviving probability mass, which is a
    different number every turn and never raises.

    Recomputed here exactly the way the update will, at a turn deep enough that
    the crew has grown past one unit.
    """
    with torch.no_grad():
        unit_logits, market_logits, value = _untrained()(
            trajectory.board[DIVERGED : DIVERGED + 1],
            trajectory.scalars[DIVERGED : DIVERGED + 1],
            trajectory.positions[DIVERGED : DIVERGED + 1],
        )
    units = torch.log_softmax(
        unit_logits.masked_fill(
            ~trajectory.unit_masks[DIVERGED : DIVERGED + 1], -torch.inf
        ),
        dim=-1,
    )
    market = torch.log_softmax(
        market_logits.masked_fill(
            ~trajectory.market_masks[DIVERGED : DIVERGED + 1], -torch.inf
        ),
        dim=-1,
    )
    acted = trajectory.unit_actions[DIVERGED] != IGNORE
    expected = (
        units[0, acted]
        .gather(1, trajectory.unit_actions[DIVERGED][acted][:, None])
        .sum()
        + market[0].gather(1, trajectory.market_actions[DIVERGED][:, None]).sum()
    )

    assert float(trajectory.log_probs[DIVERGED]) == pytest.approx(
        float(expected), abs=1e-4
    )
    assert float(trajectory.values[DIVERGED]) == pytest.approx(
        float(value[0]), abs=1e-4
    )


def test_the_recorded_state_is_the_state_the_action_was_taken_from(
    trajectory: Trajectory,
) -> None:
    """Replays the episode from the stored actions and meets the same position.

    Guards three things at once, all of them silent if wrong: that the board,
    scalars, positions and masks belong to the turn *before* the action rather
    than after it; that they were encoded for our seat rather than the
    opponent's; and that the stored action indices decode to the ops that were
    actually played. ``starter`` has no randomness, so replaying seat 1 from
    the same seed reproduces the identical episode.

    Checked at turn 200 rather than turn 0 because the opening position is
    symmetric between the two seats, so nothing about it can tell our encoding
    apart from the opponent's -- the assertion below that the opponent's mask
    now differs is what establishes that the fixture has stopped being a
    mirror.
    """
    environment = make(
        ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS, "seed": 0}
    )
    environment.reset(2)
    starter = environment.agents["starter"]
    for turn in range(DIVERGED):
        acted = trajectory.unit_actions[turn] != IGNORE
        action = decode_units(
            _one_hot(trajectory.unit_actions[turn].clamp(min=0), len(UNIT_OPS)),
            int(acted.sum()),
        )
        action["market"] = decode_market(
            _one_hot(trajectory.market_actions[turn], len(QUANTITIES))
        )
        environment.step([action, starter(environment.state[1].observation)])

    observation = environment.state[0].observation

    assert torch.equal(
        encode_board(observation, 0), trajectory.board[DIVERGED : DIVERGED + 1]
    )
    assert torch.equal(
        encode_scalars(observation, 0), trajectory.scalars[DIVERGED : DIVERGED + 1]
    )
    assert torch.equal(
        encode_positions(observation, 0), trajectory.positions[DIVERGED : DIVERGED + 1]
    )
    assert torch.equal(
        unit_mask(observation, 0), trajectory.unit_masks[DIVERGED : DIVERGED + 1]
    )
    assert torch.equal(
        market_mask(observation, 0), trajectory.market_masks[DIVERGED : DIVERGED + 1]
    )
    assert not torch.equal(
        unit_mask(environment.state[1].observation, 1),
        trajectory.unit_masks[DIVERGED : DIVERGED + 1],
    )
    assert int((trajectory.unit_actions[DIVERGED] != IGNORE).sum()) == unit_count(
        observation, 0
    )


def _one_hot(chosen: torch.Tensor, options: int) -> torch.Tensor:
    """Return one turn's stored indices as logits the decoders can argmax."""
    return torch.nn.functional.one_hot(chosen[None], options).float()


def test_the_reward_is_a_differential_and_not_our_own_bank(
    trajectory: Trajectory,
) -> None:
    """Our bank alone rates a 6,000-against-20,000 season as a good one.

    The ladder scores the margin, so the reward has to be the margin's
    increments. A rollout that recorded our own bank change instead would still
    be dense, would still telescope to a terminal scalar, and would pass every
    other test in this file -- it would just be optimising a different game.

    Checked against both banks as the encoder recorded them, turn by turn,
    rather than against the terminal margin alone: the sum identity above is
    satisfied by any sequence with the right total, and only the per-turn
    comparison says the reward lands on the turn that earned it. The second
    assertion is what makes the first one mean something -- it fails if
    ``starter`` never banked a coin, which would make the two candidate rewards
    the same sequence.
    """
    banks = trajectory.scalars[:, MONEY : MONEY + 2] * 10_000.0
    margins = banks[:, 0] - banks[:, 1]

    assert torch.allclose(trajectory.rewards[:-1], margins[1:] - margins[:-1], atol=1.0)
    assert not torch.allclose(
        trajectory.rewards[:-1], banks[1:, 0] - banks[:-1, 0], atol=1.0
    )


def test_only_the_final_turn_is_done(trajectory: Trajectory) -> None:
    """GAE bootstraps across every turn the flag does not stop it at.

    A ``dones`` that is True everywhere throws away the return beyond one step;
    one that is False everywhere bootstraps a value off the end of the season.
    """
    assert bool(trajectory.dones[-1])
    assert int(trajectory.dones.sum()) == 1


def test_every_tensor_covers_the_same_turns(trajectory: Trajectory) -> None:
    """One index has to select one decision across all of them.

    Nothing in the update checks this. A head stacked a turn short would line
    every reward up against the state after the action that earned it, and
    train on a consistent off-by-one.
    """
    turns = len(trajectory.rewards)

    assert trajectory.board.shape[0] == turns
    assert trajectory.scalars.shape[0] == turns
    assert trajectory.positions.shape == (turns, MAX_UNITS)
    assert trajectory.unit_actions.shape == (turns, MAX_UNITS)
    assert trajectory.market_actions.shape == (turns, len(MARKET_SLOTS) + 2)
    assert trajectory.unit_masks.shape == (turns, MAX_UNITS, len(UNIT_OPS))
    assert trajectory.market_masks.shape == (
        turns,
        len(MARKET_SLOTS) + 2,
        len(QUANTITIES),
    )
    assert trajectory.log_probs.shape == (turns,)
    assert trajectory.values.shape == (turns,)
    assert trajectory.dones.shape == (turns,)


def test_the_crew_is_padded_exactly_as_the_engine_staffs_it(
    trajectory: Trajectory,
) -> None:
    """A padded slot holds no decision, and scoring one trains a choice nobody made.

    ``IGNORE`` is the same sentinel ``encode_units`` writes, so the PPO loss
    and the behaviour-cloning loss mask the same slots.

    Hands are not permanent. ``_end_of_day`` empties ``farm["hands"]`` and
    resets ``hires_today`` every twenty-four turns, so the crew climbs through
    a day and drops back to the lone farmer on the hour -- a rollout that
    carried a slot count forward, or padded from a running maximum, would
    attach ops to hands the engine has already sent home and the engine would
    no-op every one of them.
    """
    real = (trajectory.unit_actions != IGNORE).sum(dim=1)
    grew = real[1:] >= real[:-1]
    dawn = torch.arange(1, len(real)) % TURNS_PER_DAY == 0

    assert bool((real[::TURNS_PER_DAY] == 1).all())
    assert bool((grew | dawn).all())
    assert int(real.min()) == 1
    assert int(real.max()) <= MAX_UNITS


def test_self_play_runs_the_policy_on_both_seats() -> None:
    """Self-play is the whole point; a policy opponent must play a full season.

    The opponent branch is a different code path from the named-agent one, and
    it is the one training will actually use. Played at a different seed so
    this is not the module fixture's episode with a new opponent bolted on.
    """
    trajectory = rollout(_untrained(), _untrained(), seed=7)

    assert len(trajectory.rewards) == 719
    assert trajectory.illegal == 0
    assert trajectory.rewards.sum() == pytest.approx(trajectory.final_margin, abs=1.0)
