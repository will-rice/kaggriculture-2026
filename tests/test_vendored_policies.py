"""Behaviour tests for the vendored public agents newer than v21.1.

Every file here is a decoded copy of a published kernel that shipped as a
base85 blob, so nothing about them is readable at a glance and nothing here
asserts what they say. These pin what they do: that the embedded tables decode
to the shapes the replay logic expects, that a well-formed action comes back
every turn, and that a turn fits the sandbox budget -- the same three
properties ``test_kaito_policy.py`` pins for v21.1.

v23 gets one test the others cannot have. It is the only vendored agent that
reads the engine configuration, and it uses it to choose between two whole
route tables: the ladder's move to ``townCenterSellInterval`` 24 in
kaggle-environments 1.32.6 is the entire reason the file exists. A copy whose
rebalance branch had been truncated would still import, still play, and still
pass every other test in this file, while silently playing the wrong season on
the only engine that scores us.
"""

import time

import pytest
from kaggle_environments import make

from kaggriculture import boatlee_v14_policy, kaito_v22_policy, kaito_v23_policy

MODULES = (kaito_v22_policy, kaito_v23_policy, boatlee_v14_policy)
IDS = ("v22", "v23", "boatlee_v14")


def first_observation() -> dict:
    """Return turn zero of a real episode, from the engine rather than by hand."""
    environment = make("kaggriculture", debug=False)
    environment.reset(num_agents=2)
    return environment.steps[0][0]["observation"]


@pytest.mark.parametrize("module", MODULES, ids=IDS)
def test_the_agent_emits_a_well_formed_action(module) -> None:
    """A malformed action is discarded by the engine, costing a turn in silence."""
    action = module.agent(first_observation())

    assert set(action) == {"farmer", "hands", "market"}
    assert isinstance(action["farmer"], list)
    assert isinstance(action["hands"], list)
    assert isinstance(action["market"], list)


@pytest.mark.parametrize("module", MODULES, ids=IDS)
def test_a_turn_fits_the_sandbox_budget(module) -> None:
    """One second a turn on two cores, with a 60 second pool for the episode."""
    observation = first_observation()
    module.agent(observation)

    start = time.perf_counter()
    for _ in range(20):
        module.agent(observation)
    elapsed = (time.perf_counter() - start) / 20

    assert elapsed < 0.1, f"{elapsed * 1000:.0f}ms per turn leaves no margin"


def test_the_single_table_routes_decode_to_a_full_season() -> None:
    """A truncated blob would still import and would only show up as bad play."""
    assert len(kaito_v22_policy._ACTIONS) == 719
    assert len(boatlee_v14_policy._ACTIONS) == 719


def test_both_v23_routes_decode_to_a_full_season() -> None:
    """Either table being short would strand the agent on one engine only."""
    assert len(kaito_v23_policy._LEGACY_ACTIONS) == 719
    assert len(kaito_v23_policy._REBALANCE_ACTIONS) == 719


@pytest.mark.parametrize(
    ("interval", "expected"),
    [(12, "legacy"), (23, "legacy"), (24, "rebalance"), (48, "rebalance")],
)
def test_v23_switches_route_on_the_town_centre_interval(
    interval: int, expected: str
) -> None:
    """1.32.3 reports 12 and 1.32.6 reports 24; the boundary is what it keys on."""
    assert kaito_v23_policy._regime({"townCenterSellInterval": interval}) == expected


def test_v23_falls_back_to_the_legacy_route_without_a_configuration() -> None:
    """Called with one argument it must still play, not raise."""
    assert kaito_v23_policy._regime(None) == "legacy"

    action = kaito_v23_policy.agent(first_observation())

    assert set(action) == {"farmer", "hands", "market"}


@pytest.mark.slow
@pytest.mark.parametrize("module", MODULES, ids=IDS)
def test_it_plays_a_full_episode_and_banks(module) -> None:
    """The whole point: finish 719 turns and bank far above the 3,000 opening."""
    environment = make("kaggriculture", debug=False)
    environment.run([lambda observation, *rest: module.agent(observation), "starter"])

    banked = environment.steps[-1][0]["observation"]["farms"][0]["money"]

    assert banked > 100_000, f"banked only {banked:,.0f}"
