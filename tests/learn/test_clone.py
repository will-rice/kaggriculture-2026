"""What the clone gate must not get wrong.

The gate's answer is a win rate, and a win rate is only worth reading if the
rows it was trained on are the rows the teacher decided from, if the seasons
it was graded on are not the seasons it saw, and if the agent the engine
played is the one the checkpoint belongs to. Each of those fails silently, so
each has a test here.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from kaggle_environments.agent import get_last_callable

from kaggriculture.action_codec import bucket_of
from kaggriculture.constants import EPISODE_STEPS
from kaggriculture.features import MAX_UNITS, SCALAR_INDEX
from kaggriculture.learn import CLONE_CHECKPOINT, clone, play
from kaggriculture.learn.clone import (
    CLONE_ACTOR,
    HOLDOUT_SEEDS,
    TEACHER_ACTOR,
    TRAIN,
    TRAIN_SEEDS,
    Game,
    Recording,
    Season,
    built_shards,
    record,
    season_slices,
    win_rate,
)
from kaggriculture.learn.encoding import IGNORE, encode_unit_quantities
from kaggriculture.learn.scripts.clone_gate import VERDICTS
from kaggriculture.search.scripts.holdout import GATE_SEEDS


def season(rows: int, ours: int, theirs: int) -> Season:
    """Return a manifest entry standing for one recorded season."""
    return Season(
        opponent="v56",
        seed=1,
        seat=0,
        actor=TEACHER_ACTOR,
        ours=ours,
        theirs=theirs,
        rows=rows,
        skipped=0,
    )


def test_only_a_transfer_carries_a_quantity_label() -> None:
    """A count is the transfer verbs' argument; no other op has one to choose."""
    labels = encode_unit_quantities(
        {"farmer": ["PICKUP", "WHEAT", 5], "hands": [["WATER"], ["PLANT", "MELON"]]},
        units=3,
    )
    assert labels[0, 0].item() == bucket_of(5)
    assert labels[0, 1].item() == IGNORE
    assert labels[0, 2].item() == IGNORE


def test_padded_slots_carry_no_quantity_label() -> None:
    """A hand the farm has not hired did not choose to move nothing."""
    labels = encode_unit_quantities(
        {"farmer": ["PLACE", "WHEAT", 2], "hands": [["PICKUP", "COW", 1]]}, units=1
    )
    assert labels[0, 0].item() == bucket_of(2)
    assert (labels[0, 1:] == IGNORE).all()
    assert labels.shape == (1, MAX_UNITS)


def test_a_transfer_with_no_count_is_labelled_the_way_the_engine_reads_it() -> None:
    """`_apply_unit_action` reads a missing count as 1, so the label is bucket 1.

    Labelling it 0 would teach the clone that a real one-item transfer was no
    transfer at all, and bucket 0 emits nothing on the way back out.
    """
    labels = encode_unit_quantities({"farmer": ["PICKUP", "WHEAT"], "hands": []}, 1)
    assert labels[0, 0].item() == bucket_of(1)
    assert labels[0, 0].item() != 0


def test_no_clone_season_is_ever_played_on_an_exam_seed() -> None:
    """The gate is void the moment a training seed appears in the exam bank."""
    assert not set(TRAIN_SEEDS + HOLDOUT_SEEDS) & set(GATE_SEEDS)
    assert not set(TRAIN_SEEDS) & set(HOLDOUT_SEEDS)


def test_built_shards_refuses_a_shard_it_cannot_account_for(tmp_path: Path) -> None:
    """A leftover shard has real shapes and would concatenate in silence."""
    for name in (f"{TRAIN}.npz", "holdout.npz"):
        np.savez(tmp_path / name, boards=np.zeros((1, 1)))
    built_shards(tmp_path)
    np.savez(tmp_path / "leftover.npz", boards=np.zeros((1, 1)))
    with pytest.raises(ValueError, match="leftover.npz"):
        built_shards(tmp_path)


def test_season_slices_partition_the_rows_in_manifest_order() -> None:
    """Rows are concatenated in the order the seasons were played, contiguously."""
    manifest = Recording(
        teacher="t", seasons=[season(3, 1, 0), season(2, 0, 1), season(4, 1, 0)]
    )
    covered = [index for _, rows in season_slices(manifest) for index in range(9)[rows]]
    assert covered == list(range(9))


def test_win_rate_counts_a_tie_as_half() -> None:
    """The ladder does, and so does `arena`; a third rule here would disagree."""
    manifest = Recording(
        teacher="t", seasons=[season(1, 5, 4), season(1, 4, 4), season(1, 3, 4)]
    )
    assert win_rate(manifest) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("rate", "name"),
    [(1.0, "ALIVE"), (0.4, "ALIVE"), (0.39, "PARTIAL"), (0.1, "WEAK"), (0.0, "FAILED")],
)
def test_a_rate_between_two_bands_reads_as_the_lower_one(
    rate: float, name: str
) -> None:
    """Pre-registration is only pre-registration if it cannot round upward."""
    assert next(band for band in VERDICTS if rate >= band.floor).name == name


def test_the_engine_loads_the_clone_agent_and_points_it_at_the_clone_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`get_last_callable` takes the last callable, and the rebind has to survive it.

    The file imports a checkpoint constant and a module before defining
    ``agent``; if either of those were left callable and defined afterwards, the
    engine would play it instead and the gate would score something that is not
    the clone.
    """
    monkeypatch.setattr(play, "CHECKPOINT", play.CHECKPOINT)
    path = clone.CLONE_AGENT
    loaded = get_last_callable(Path(path).read_text(), path=path)
    assert loaded is play.agent
    assert play.CHECKPOINT == CLONE_CHECKPOINT


def test_a_season_must_be_driven_by_the_teacher_or_the_clone() -> None:
    """The actor decides whose states get labelled, so it cannot be a free string.

    A typo would otherwise fall through to whichever branch was written last
    and the round would silently be the wrong experiment.
    """
    with pytest.raises(ValueError, match="teacher or the clone"):
        record(Game(opponent="v56", seed=TRAIN_SEEDS[0], seat=0, actor="expert"))


@pytest.mark.slow
def test_an_actor_that_never_decides_is_not_a_recorded_season(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raising agent still finishes DONE, so the status alone proves nothing.

    ``kaggle_environments`` catches an agent's exception, logs it, and hands
    the interpreter an error object in place of an action. The interpreter
    ignores it, the episode runs to the horizon and both seats report DONE with
    the farm untouched -- and the recording is empty. This is the check that
    makes that loud.
    """
    broken = tmp_path / "broken.py"
    broken.write_text("def agent(observation):\n    raise RuntimeError('no')\n")
    monkeypatch.setattr(clone, "CLONE_AGENT", str(broken))
    with pytest.raises(RuntimeError, match="did not decide every turn"):
        record(Game(opponent="v56", seed=TRAIN_SEEDS[0], seat=0, actor=CLONE_ACTOR))


@pytest.mark.slow
def test_a_recorded_row_is_the_turn_the_teacher_decided_from() -> None:
    """Row *i* must hold step *i*'s state, with no turn dropped or shifted.

    ``dataset.py`` has a whole module docstring about the off-by-one that
    reading a replay back invites. This path cannot make that mistake -- it
    encodes inside the decision -- and this is the check that says so: the
    engine's own step counter, recovered from the row's scalars, has to equal
    the row's index for every one of a season's turns.
    """
    outcome, arrays = record(Game(opponent="v56", seed=TRAIN_SEEDS[0], seat=0))
    scalars = torch.from_numpy(arrays["scalars"])
    steps = scalars[:, SCALAR_INDEX["step"]] * EPISODE_STEPS
    assert outcome.rows == EPISODE_STEPS - 1
    assert outcome.skipped == 0
    assert torch.allclose(steps, torch.arange(outcome.rows, dtype=torch.float32))
    assert (torch.from_numpy(arrays["labels"])[:, 0] != IGNORE).all()
