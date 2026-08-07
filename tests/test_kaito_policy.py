"""Behaviour tests for the vendored Top-30 route memory.

The file is 96.7% base85-encoded data and is deliberately left unlinted, so
nothing about it is readable at a glance. These pin what it does rather than
what it says: that the embedded tables decode to the shapes the retrieval logic
expects, that it emits a well-formed action every turn, and that it stays inside
the sandbox's per-turn budget.
"""

import time

import pytest
from kaggle_environments import make

from kaggriculture.kaito_policy import agent


def first_observation() -> dict:
    """Return turn zero of a real episode, from the engine rather than by hand."""
    env = make("kaggriculture", debug=False)
    env.reset(num_agents=2)
    return env.steps[0][0]["observation"]


def test_the_agent_emits_a_well_formed_action() -> None:
    """A malformed action is discarded by the engine, costing a turn in silence."""
    action = agent(first_observation())

    assert set(action) == {"farmer", "hands", "market"}
    assert isinstance(action["farmer"], list)
    assert isinstance(action["hands"], list)
    assert isinstance(action["market"], list)


def test_the_embedded_tables_decode() -> None:
    """The route store is base85 inside zlib inside the source.

    A truncated copy of the blob would still import and still return actions;
    the failure would surface only as bad play, which is exactly the kind of
    silent breakage this file's unreadability invites.
    """
    from kaggriculture import kaito_policy

    assert len(kaito_policy._PROTOTYPES) > 0
    assert len(kaito_policy._ACTIONS) > 0


def test_a_turn_fits_the_sandbox_budget() -> None:
    """One second a turn on two cores, with a 60 second pool for the episode."""
    observation = first_observation()
    agent(observation)

    start = time.perf_counter()
    for _ in range(20):
        agent(observation)
    elapsed = (time.perf_counter() - start) / 20

    assert elapsed < 0.1, f"{elapsed * 1000:.0f}ms per turn leaves no margin"


@pytest.mark.slow
def test_it_plays_a_full_episode_and_banks() -> None:
    """The whole point: finish 719 turns and bank far above the 3,000 opening."""
    env = make("kaggriculture", debug=False)
    env.run([lambda observation, *rest: agent(observation), "starter"])

    banked = env.steps[-1][0]["observation"]["farms"][0]["money"]

    assert banked > 100_000, f"banked only {banked:,.0f}"
