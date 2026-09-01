"""What the online command must refuse, and what its verdict is allowed to say.

The production run is hours of engine time on an owned root, so the failures
worth a test here are the ones that would only surface after those hours: a run
quietly demoted to the CPU, a root reopened under different parameters, a batch
that happened to contain no game against the opponent the gate is decided by,
and a verdict that read an inconclusive interval as a pass.
"""

import json
from pathlib import Path

import pytest

from kaggriculture.learn.market_residual import online
from kaggriculture.learn.market_residual.online import OnlineConfig, sample_cells
from kaggriculture.market_residual.schema import (
    HELD_OUT_GATE_SEEDS,
    SEED_BANKS,
)
from kaggriculture.scripts import market_train
from kaggriculture.scripts.market_train import (
    OPPONENTS,
    OnlineParameterError,
    build_cells,
    online_parameters,
    reconcile_parameters,
    verdict,
)
from kaggriculture.search.scripts import holdout


def config(
    updates: int = 4,
    episodes_per_update: int = 10,
    learning_rate: float = 3e-4,
    workers: int = 48,
) -> OnlineConfig:
    """Return a small online configuration.

    Args:
        updates: How many updates the run would take.
        episodes_per_update: How many paired seasons each update plays.
        learning_rate: The optimizer step size.
        workers: How many actor processes the run would own.

    Returns:
        The configuration.
    """
    return OnlineConfig(
        updates=updates,
        episodes_per_update=episodes_per_update,
        learning_rate=learning_rate,
        workers=workers,
        device="cpu",
    )


def test_the_opponent_league_is_the_one_that_can_separate_two_agents() -> None:
    """Every opponent is on disk, and the served floor is the one gate plays.

    ``online`` restates the served controller's path rather than importing it
    from the search tree, so the two are pinned equal here: moving the gate
    moves collection with it.
    """
    assert online.SERVED_PATH == holdout.SERVED
    for name, path, weight in OPPONENTS:
        assert Path(path).exists(), f"{name} is not on disk at {path}"
        assert weight > 0.0
    assert sum(weight for _n, _p, weight in OPPONENTS) == pytest.approx(1.0)


def test_cells_cover_both_seats_and_never_touch_the_exam_seeds() -> None:
    """A cell bank is opponents by bank seeds by seats, and nothing else."""
    cells = build_cells(config(), seeds=8)
    assert len(cells) == len(OPPONENTS) * 8 * 2
    assert len(set(cells)) == len(cells)
    seeds = {cell.seed for cell in cells}
    assert seeds <= set(SEED_BANKS["online_train"])
    assert not seeds & HELD_OUT_GATE_SEEDS
    assert {cell.seat for cell in cells} == {0, 1}


def test_every_batch_holds_games_against_the_opponent_the_gate_decides_on() -> None:
    """Opponent shares are allocated, not sampled, so none can go missing."""
    settings = config(episodes_per_update=10)
    cells = build_cells(settings, seeds=16)
    for update in range(1, 6):
        batch = sample_cells(cells, settings, update)
        assert len(batch) == settings.episodes_per_update
        counts = {name: 0 for name, _path, _weight in OPPONENTS}
        for cell in batch:
            counts[cell.opponent] += 1
        assert counts["v56"] == 4
        assert all(count > 0 for count in counts.values())


def test_sampling_a_batch_does_not_depend_on_being_interrupted() -> None:
    """The same update plays the same cells however the run reached it."""
    settings = config()
    cells = build_cells(settings, seeds=16)
    assert sample_cells(cells, settings, 3) == sample_cells(cells, settings, 3)
    assert sample_cells(cells, settings, 3) != sample_cells(cells, settings, 4)


def test_the_production_command_refuses_anything_but_one_owned_gpu(
    tmp_path: Path,
) -> None:
    """A run demoted to the CPU or split over two GPUs is not this run."""
    with pytest.raises(OnlineParameterError, match="must be cuda"):
        market_train.main([str(tmp_path), "--create", "--device", "cpu"])
    with pytest.raises(OnlineParameterError, match="--devices must be 1"):
        market_train.main([str(tmp_path), "--create", "--devices", "2"])
    assert not (tmp_path / market_train.PARAMETERS_NAME).exists()


def test_create_refuses_a_claimed_root_and_resume_an_unclaimed_one(
    tmp_path: Path,
) -> None:
    """A root is bound to one parameterisation, once, and says so if it moves."""
    settings = config()
    cells = build_cells(settings, seeds=4)
    wanted = online_parameters(settings, cells)
    with pytest.raises(OnlineParameterError, match="never created"):
        reconcile_parameters(tmp_path, wanted, create=False)
    reconcile_parameters(tmp_path, wanted, create=True)
    with pytest.raises(OnlineParameterError, match="already claimed"):
        reconcile_parameters(tmp_path, wanted, create=True)
    assert reconcile_parameters(tmp_path, wanted, create=False) == wanted

    moved = online_parameters(config(learning_rate=1e-3), cells)
    with pytest.raises(OnlineParameterError, match="different parameters"):
        reconcile_parameters(tmp_path, moved, create=False)

    stored = json.loads((tmp_path / market_train.PARAMETERS_NAME).read_text())
    stored["fields"]["version"] = 99
    (tmp_path / market_train.PARAMETERS_NAME).write_text(json.dumps(stored))
    with pytest.raises(OnlineParameterError, match="own digest"):
        reconcile_parameters(tmp_path, wanted, create=False)


def test_the_machine_is_not_part_of_what_a_root_is_bound_to() -> None:
    """Moving a root to another GPU or worker budget is a legal resume."""
    cells = build_cells(config(), seeds=4)
    assert online_parameters(config(workers=8), cells) == online_parameters(
        config(workers=64), cells
    )
    assert online_parameters(config(updates=4), cells) != online_parameters(
        config(updates=8), cells
    )


def test_an_inconclusive_interval_is_not_a_pass_and_not_a_failure() -> None:
    """The three branches were fixed before any of them had a number in it."""
    assert verdict(0.52, 0.64) == "improves"
    assert verdict(0.0, 0.0291) == "harms"
    assert verdict(0.41, 0.59) == "no measured effect"
    assert verdict(0.5, 0.5) == "no measured effect"
