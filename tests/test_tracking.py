"""Tests for the shape of what we send to Weights & Biases."""

import pytest

from kaggriculture.config import HarnessConfig
from kaggriculture.report import Standing
from kaggriculture.scripts.tracking import (
    ensure_reproducible,
    job_type,
    run_config,
    run_metrics,
    run_name,
)


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
    config = run_config(HarnessConfig(), "baselines/heuristic_v2.py", "abc1234")

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

    logged = run_config(HarnessConfig(), "baselines/heuristic_v1.py", "abc1234")

    assert logged["crops"] == {"MELON": 36}
    assert logged["herd"] == {}
    assert logged["max_hands"] == 8
    assert logged["crops"] != dict(STRATEGY.crops)


def test_an_agent_without_a_strategy_records_none_rather_than_borrowing_one() -> None:
    """The served agent is not parameterised by Strategy; it must not claim to be."""
    logged = run_config(HarnessConfig(), "main.py", "abc1234")

    assert "crops" not in logged
    assert "sell_rate" not in logged
    assert logged["agent"] == "main.py"


def test_a_builtin_opponent_has_no_strategy() -> None:
    """Built-ins are named rather than pathed, so there is no module to read."""
    logged = run_config(HarnessConfig(), "starter", "abc1234")

    assert "crops" not in logged
    assert logged["agent"] == "starter"


def test_a_league_run_is_named_for_the_agent_and_the_commit() -> None:
    """`icy-pond-3` tells a reader nothing later; this tells them everything."""
    name = run_name(HarnessConfig(), agent="main.py", commit="bc7b4ce")

    assert name == "main-vs-league-bc7b4ce"


def test_a_head_to_head_run_names_its_opponents() -> None:
    """A gate run is a specific matchup and its name should say which."""
    config = HarnessConfig(opponents=("/data/kaggriculture/baselines/meta_tape.py",))

    name = run_name(config, agent="baselines/heuristic_v1.py", commit="82a4130")

    assert name == "heuristic_v1-vs-meta_tape-82a4130"


def test_opponent_order_does_not_change_the_name() -> None:
    """The same matchup written two ways must group as one thing, not two."""
    forward = HarnessConfig(opponents=("starter", "pass"))
    reverse = HarnessConfig(opponents=("pass", "starter"))

    assert run_name(forward, "main.py", "abc1234") == run_name(
        reverse, "main.py", "abc1234"
    )


def test_job_type_separates_league_runs_from_ad_hoc_matchups() -> None:
    """League runs are the acceptance record; ad-hoc sweeps are exploration."""
    assert job_type(HarnessConfig()) == "league"
    assert job_type(HarnessConfig(opponents=("starter",))) == "head-to-head"


def test_a_clean_tree_records_its_revision() -> None:
    """The ordinary case: nothing uncommitted, so the commit describes the code."""
    assert ensure_reproducible("", "bc7b4ce") == "bc7b4ce"


def test_a_dirty_tree_refuses_to_be_recorded() -> None:
    """Numbers from uncommitted code are attributable to no revision at all."""
    status = " M src/kaggriculture/policy.py\n?? experiment.py"

    with pytest.raises(RuntimeError) as raised:
        ensure_reproducible(status, "bc7b4ce")

    assert "dirty working tree" in str(raised.value)
    assert "policy.py" in str(raised.value)


def test_the_refusal_names_what_is_uncommitted() -> None:
    """A refusal the user cannot act on is just an obstacle."""
    with pytest.raises(RuntimeError) as raised:
        ensure_reproducible(" M tests/test_policy.py", "abc1234")

    assert "tests/test_policy.py" in str(raised.value)
    assert "abc1234" in str(raised.value)
