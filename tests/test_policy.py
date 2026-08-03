"""Behavioural tests for the baseline policy, played against the real environment."""

import pytest
from kaggle_environments import make

from kaggriculture.policy import STRATEGY, agent, needs_water
from kaggriculture.constants import CROPS, EPISODE_STEPS, MAX_MARKET_ORDERS_PER_TURN, STARTING_MONEY


@pytest.fixture(scope="module")
def episode() -> list:
    """Return the recorded steps of one full season against the starter agent."""
    env = make("kaggriculture", configuration={"episodeSteps": EPISODE_STEPS, "seed": 7})
    env.run([agent, "starter"])
    return env.steps


def turns(episode: list) -> list[tuple]:
    """Pair each observation with the action our player chose from it.

    A recorded step holds the state *after* its action was interpreted, so the
    action belonging to an observation is the one recorded on the next step.
    """
    return [
        (observed[0].observation, acted[0].action)
        for observed, acted in zip(episode, episode[1:])
    ]


def test_agent_finishes_without_error(episode: list) -> None:
    """A full season runs to completion with no invalid action or timeout."""
    assert len(episode) == EPISODE_STEPS
    assert [state.status for state in episode[-1]] == ["DONE", "DONE"]


def test_agent_beats_the_starter_baseline(episode: list) -> None:
    """The policy ends the season richer than the built-in starter agent."""
    ours, theirs = (state.reward for state in episode[-1])

    assert ours > theirs
    assert ours > STARTING_MONEY


def test_actions_stay_within_the_market_order_cap(episode: list) -> None:
    """No turn queues more market orders, or more hand ops, than are honoured."""
    for observed, acted in turns(episode):
        assert len(acted["market"]) <= MAX_MARKET_ORDERS_PER_TURN
        assert len(acted["hands"]) == len(observed.farms[0]["hands"])


def test_plant_requests_never_exceed_seeds(episode: list) -> None:
    """Over-requesting a crop voids every plant that turn, so it must never happen."""
    for observed, acted in turns(episode):
        planted = [op[1] for op in [acted["farmer"], *acted["hands"]] if op[0] == "PLANT"]
        seeds = observed.private["seeds"]
        for crop in set(planted):
            assert planted.count(crop) <= seeds.get(crop, 0)


def test_season_ends_with_produce_sold(episode: list) -> None:
    """Unsold stock scores nothing, so the shed is emptied before the last turn."""
    shed = episode[-1][0].observation.private["shed"]

    assert sum(shed.values()) == 0


def test_new_plants_are_watered_on_their_planting_day() -> None:
    """A plant starts one dry day down, so skipping day zero would kill it."""
    crop = STRATEGY.crop
    fresh = {
        "crop": crop,
        "planted_day": 3,
        "watered_today": False,
        "consecutive_unwatered": 1,
    }

    assert needs_water(fresh, day=3)


def test_watering_is_skipped_outside_the_bonus_window() -> None:
    """A plant watered yesterday and outside its window can safely wait a turn."""
    crop = STRATEGY.crop
    idle = {
        "crop": crop,
        "planted_day": 0,
        "watered_today": False,
        "consecutive_unwatered": 0,
    }
    day = (CROPS[crop]["max_yield_day"] + 1) // 2 - 1

    assert not needs_water(idle, day=day)
