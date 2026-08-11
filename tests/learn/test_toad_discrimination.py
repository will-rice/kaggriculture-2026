"""Checks against the failure mode that produces a plausible curve and no alarm.

The bugs that cost this reproduction a run so far all failed loudly -- a 1e17
loss, an OOM, a hang. The dangerous class is the one that trains something and
looks fine: a V-trace that ignores its rewards, or a reward that reads the wrong
field. Both are cheap to rule out and neither is covered by the loss tests, which
only check the arithmetic against itself.
"""

from typing import Any

import pytest
import torch
from kaggle_environments import make

from kaggriculture.constants import ENVIRONMENT
from kaggriculture.learn import toad_loss, toad_reward
from kaggriculture.learn.encoding import SCALARS, TILE_PLANES
from kaggriculture.learn.model import Policy

TURNS, BATCH = 8, 3


def test_vtrace_advantages_track_the_rewards() -> None:
    """An inert V-trace would return something reward-independent.

    If the advantages do not move when the rewards move, the learner is
    optimising a constant and every curve it draws is meaningless. This is the
    single cheapest guard against a silently decorative off-policy correction.
    """
    generator = torch.Generator().manual_seed(0)
    # Annotated because `losses` also takes bools and weights: inferred as
    # `dict[str, Tensor]`, the `**shared` below is a type error against every
    # non-tensor parameter it does not actually supply.
    shared: dict[str, Any] = {
        "behaviour_log_probs": torch.full((TURNS, BATCH), -1.0),
        "learner_log_probs": torch.full((TURNS, BATCH), -1.0),
        "negative_entropy": torch.zeros(TURNS, BATCH),
        "values": torch.randn(TURNS, BATCH, generator=generator),
        "bootstrap_value": torch.randn(BATCH, generator=generator),
        "dones": torch.zeros(TURNS, BATCH, dtype=torch.bool),
    }
    low = toad_loss.losses(rewards=torch.full((TURNS, BATCH), 0.01), **shared)
    high = toad_loss.losses(rewards=torch.full((TURNS, BATCH), 5.0), **shared)
    # Bigger rewards must mean bigger returns, hence a different policy gradient
    # and a different baseline target.
    assert not torch.allclose(low.vtrace_pg, high.vtrace_pg)
    assert high.baseline > low.baseline


def test_the_value_head_is_bounded_to_the_reward_range() -> None:
    """Toad's baseline is confined to the reward space; ours must be too.

    An unbounded head is what diverged the first run, and it diverged silently
    for two updates before the magnitude gave it away.
    """
    policy = Policy(blocks=1, channels=16, value_bound=1.0)
    # `Policy` builds its stem and market head from these two constants, so
    # reading them back off the modules only reintroduces `nn.Sequential`'s
    # untyped indexing. `test_toad_runner` already sizes its inputs this way.
    board = torch.randn(4, TILE_PLANES, 10, 10) * 50.0
    scalars = torch.randn(4, SCALARS) * 50.0
    positions = torch.zeros(4, 20, dtype=torch.int64)
    _, _, value = policy(board, scalars, positions)
    assert bool((value.abs() <= 1.0).all())
    # And the default stays unbounded, so the PPO path is unchanged.
    plain = Policy(blocks=1, channels=16)
    _, _, wide = plain(board, scalars, positions)
    assert plain.value_bound is None
    assert torch.isfinite(wide).all()


def test_the_reward_reads_the_fields_it_claims_to() -> None:
    """Assert the counts against a hand-built state with known contents.

    Stated as literals rather than as expressions over the observation, so that
    reading the wrong key -- another farm's money, the town's shops instead of
    ours, tiles transposed -- fails here instead of quietly shaping the policy
    toward the wrong thing.
    """
    environment = make(ENVIRONMENT, debug=True)
    environment.reset(2)
    observation = environment.state[0].observation

    opening = toad_reward.counts(observation, 0)
    # One 5x5 quadrant unlocked, nothing planted, farmer only, empty shed.
    assert opening.city == 25
    assert opening.unit == 1
    assert opening.fuel == 0

    # Hire two hands and stock the shed with a known amount.
    farms = [dict(farm) for farm in observation["farms"]]
    farms[0]["hands"] = [{"id": "a"}, {"id": "b"}]
    shed = dict.fromkeys(observation["private"]["shed"], 0)
    shed[next(iter(shed))] = 7
    changed = {
        **observation,
        "farms": farms,
        "private": {**observation["private"], "shed": shed},
    }
    after = toad_reward.counts(changed, 0)
    assert after.unit == 3
    assert after.fuel == 7

    # One turn of that change is worth, by hand:
    #   unit  0.5 * (3 - 1)      = 1.0
    #   fuel  0.005 * (7 - 0)    = 0.035
    #   step  0.005              = 0.005
    #   total 1.04 / 500         = 0.00208
    reward = toad_reward.StatefulMultiReward(observation, 0)
    assert reward.step(changed, done=False) == pytest.approx(0.00208)


def test_the_reward_reads_our_own_seat_not_the_other() -> None:
    """Seat 1's counts must come from seat 1's farm.

    A reward that silently scores the opponent would still train, still produce
    a curve, and be wrong in a way no loss test could see.
    """
    environment = make(ENVIRONMENT, debug=True)
    environment.reset(2)
    observation = environment.state[0].observation
    farms = [dict(farm) for farm in observation["farms"]]
    farms[0]["hands"] = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    changed = {**observation, "farms": farms}
    assert toad_reward.counts(changed, 0).unit == 4
    assert toad_reward.counts(changed, 1).unit == 1
