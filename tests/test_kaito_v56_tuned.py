"""The tuned controller is the vendored one with two thresholds moved.

The win this file records was measured, not reasoned: 0.9922 over 128 games
against the vendored v56. What a unit test can hold is that the two numbers
are still the measured ones, that nothing else about the configuration
drifted, and that the tuned agent genuinely plays differently from the
vendored one -- a config change that stopped reaching the board would leave
every other assertion here green.
"""

import dataclasses

from kaggle_environments import make

from kaggriculture import kaito_v56_policy as vendored
from kaggriculture import kaito_v56_tuned_policy as tuned

# A seed outside every declared bank in `market_residual.schema.SEED_BANKS`,
# so this test spends no seed a measurement might later need.
PROBE_SEED = 4247


def test_the_tuned_thresholds_are_the_measured_ones() -> None:
    """The two constants are the values the 128-game gate was run at."""
    assert tuned.PREEMPT_HORIZON == 7
    assert tuned.PREEMPT_MINIMUM_PRICE_RATIO == 0.225
    assert tuned.TUNED_CONFIG.preempt_horizon == 7
    assert tuned.TUNED_CONFIG.preempt_minimum_price_ratio == 0.225


def test_nothing_but_those_two_fields_moved() -> None:
    """Every other configuration field still holds the vendored value."""
    moved = {"preempt_horizon", "preempt_minimum_price_ratio"}
    for field in dataclasses.fields(vendored._V54_CONFIG):
        if field.name in moved:
            continue
        assert getattr(tuned.TUNED_CONFIG, field.name) == getattr(
            vendored._V54_CONFIG, field.name
        ), field.name


def test_the_tuned_agent_actually_plays_differently() -> None:
    """The changed thresholds reach the board, on a season neither has seen.

    Without this, a tuned config that never altered an action would satisfy
    every other assertion in this file while playing exactly the vendored
    agent -- which is the failure the gate exists to catch and a unit test
    can catch far more cheaply.
    """
    actions = []
    for seat in (tuned.agent, vendored.kaggle_agent_v56):
        environment = make(
            "kaggriculture", configuration={"seed": PROBE_SEED}, debug=False
        )
        environment.run([seat, vendored.kaggle_agent_v56])
        actions.append(
            [step[0].get("action") for step in environment.steps if "action" in step[0]]
        )
    assert actions[0] != actions[1]
