"""Tests for the shape of what we send to Weights & Biases."""

import pytest

from kaggriculture.config import HarnessConfig
from kaggriculture.report import Standing
from kaggriculture.scripts.tracking import run_config, run_metrics


def standing(opponent: str, win_rate: float) -> Standing:
    """Return a standing with the fields the metric builder reads."""
    return Standing(
        opponent=opponent,
        games=100,
        wins=int(win_rate * 100),
        losses=100 - int(win_rate * 100),
        ties=0,
        errors=0,
        win_rate=win_rate,
        low=win_rate - 0.05,
        high=win_rate + 0.05,
        bank=50000.0,
        opponent_bank=170000.0,
    )


def test_run_config_carries_every_strategy_knob() -> None:
    """A sweep is only comparable if the configuration that produced it is recorded."""
    config = run_config(HarnessConfig(), agent="baselines/heuristic_v2.py")

    assert config["crops"] == {"WHEAT": 11, "MELON": 11, "STRAWBERRY": 40}
    assert config["sell_rate"] == 2
    assert config["tiles_per_unit"] == 5
    assert config["agent"] == "baselines/heuristic_v2.py"
    assert "games" in config and "seed" in config


def test_metric_keys_survive_opponents_named_by_path() -> None:
    """League members are file paths; slashes would nest them into separate charts."""
    metrics = run_metrics([standing("baselines/heuristic_v2.py", 0.4)])

    assert metrics["win_rate/heuristic_v2"] == 0.4
    assert metrics["bank/heuristic_v2"] == 50000.0
    assert not any("/" in key.split("/", 1)[1] for key in metrics)


def test_league_win_rate_is_the_headline_number() -> None:
    """One number decides whether a change ships, and it spans the whole league."""
    metrics = run_metrics([standing("a.py", 0.4), standing("b.py", 0.6)])

    assert metrics["win_rate/league"] == 0.5


def test_opponents_sharing_a_basename_raise_instead_of_colliding() -> None:
    """A silent overwrite would drop a row from a record meant to be trustworthy."""
    with pytest.raises(ValueError) as excinfo:
        run_metrics([standing("dir1/foo.py", 0.4), standing("dir2/foo.py", 0.6)])

    assert "dir1/foo.py" in str(excinfo.value)
    assert "dir2/foo.py" in str(excinfo.value)


def test_config_describes_the_agent_that_ran_not_the_package_default() -> None:
    """The runner used to log the package default whatever --agent selected.

    That is the failure this module exists to prevent, and it happened: the
    Phase 1 gate evaluated a frozen baseline and recorded the knobs of an
    entirely different farm. The strategy is now read from the agent, so the two
    cannot disagree.
    """
    from kaggriculture.policy import STRATEGY

    logged = run_config(HarnessConfig(), agent="baselines/heuristic_v1.py")

    assert logged["crops"] == {"MELON": 36}
    assert logged["herd"] == {}
    assert logged["max_hands"] == 8
    assert logged["crops"] != dict(STRATEGY.crops)


def test_an_agent_without_a_strategy_records_none_rather_than_borrowing_one() -> None:
    """The served agent is not parameterised by Strategy; it must not claim to be."""
    logged = run_config(HarnessConfig(), agent="main.py")

    assert "crops" not in logged
    assert "sell_rate" not in logged
    assert logged["agent"] == "main.py"


def test_a_builtin_opponent_has_no_strategy() -> None:
    """Built-ins are named rather than pathed, so there is no module to read."""
    logged = run_config(HarnessConfig(), agent="starter")

    assert "crops" not in logged
    assert logged["agent"] == "starter"
