"""Recording our own teacher's decisions, and the dataset built from them.

``dataset.py`` clones the *published* corpus: episodes other people played,
read out of an archive, where the only way to know what an agent saw is to
reconstruct it from the recorded steps. That reconstruction is delicate --
the interpreter mutates the state it is handed, so a recorded ``steps[i]``
holds the world *after* its own action -- and the whole module docstring
there is about getting the off-by-one right.

None of that applies here, because here we are playing the teacher ourselves.
The observation is encoded inside the agent closure, at the moment the teacher
is looking at it and before it has returned, and the action is the object the
closure is about to hand back. There is no index to align and nothing to
reconstruct: the pair is the decision. That is why this module records rather
than re-deriving, and why it does not reuse ``_encode_episode``.

What it does reuse is every encoder: ``encode_observation`` for the input,
``encode_units``, ``encode_unit_quantities`` and ``encode_market`` for the
labels. Play, training and this recording therefore cannot disagree about what
a row means -- ``learn.play`` reads the same ``encode_observation`` on the same
observation shape.

**Four labels are stored, not three.** The op vocabulary carries no count, so a
clone trained on ops alone moves exactly one item per transfer; the teacher's
transfers are 1 in only a third of cases. ``encode_unit_quantities`` is the
fourth label and it is why this module writes a six-array shard where
``dataset.py`` writes five.

Seeds are drawn from a block that no other part of this project searches on:
``hillclimb.SEED_POOL`` is 500,000-600,000 and ``holdout.GATE_SEEDS`` is
700,000-700,063, so training here can never touch the exam the clone is
graded on. The disjointness is asserted at import time rather than left to a
test, because a widened seed block would otherwise void the gate silently.
"""

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from kaggle_environments import make
from kaggle_environments.agent import get_last_callable
from pydantic import BaseModel
from torch.utils.data import Dataset
from tqdm import tqdm

from kaggriculture.constants import ENVIRONMENT, EPISODE_STEPS
from kaggriculture.features import TooManyUnitsError, encode_observation, unit_count
from kaggriculture.learn import CLONE_CHECKPOINT
from kaggriculture.learn.encoding import (
    encode_market,
    encode_unit_quantities,
    encode_units,
)
from kaggriculture.search.scripts.hillclimb import SEED_POOL as _SEARCH_SEED_POOL
from kaggriculture.search.scripts.holdout import GATE_SEEDS as _GATE_SEEDS

LOGGER = logging.getLogger(__name__)

CLONE_DIR = CLONE_CHECKPOINT.parent

# The agent being cloned. This is the strongest thing we field: v58's routes
# and router with two market thresholds moved (see the agents README).
TEACHER = "/data/kaggriculture/agents/tuned_v58_h7_ratio225.py"

# A spread rather than one opponent. The teacher is a clock-indexed route
# replayer, so the states it visits are largely a function of its own schedule;
# what varies them is the board (the seed) and what the other farm does to the
# market. Playing only our own lineage would give a clone that has never seen
# a farm that prices differently.
OPPONENTS: dict[str, str] = {
    "v56": "src/kaggriculture/kaito_v56_policy.py",
    "v58": "/data/kaggriculture/agents/v58_original.py",
    "tuned_v56": "/data/kaggriculture/agents/tuned_v56_h7_ratio225.py",
    "tetsutani_shopforge": "/data/kaggriculture/opponents/tetsutani_shopforge/main.py",
    "indarkarhana_top10": "/data/kaggriculture/opponents/indarkarhana_top10/main.py",
    "lynnsakurai_v5": "/data/kaggriculture/opponents/lynnsakurai_v5/main.py",
    "yhay81_router2929": "/data/kaggriculture/opponents/yhay81_router2929/main.py",
    "salemali7_2900": "/data/kaggriculture/opponents/salemali7_2900/main.py",
}

TRAIN_SEEDS: tuple[int, ...] = tuple(range(800_000, 800_020))
HOLDOUT_SEEDS: tuple[int, ...] = tuple(range(800_100, 800_105))

if set(TRAIN_SEEDS) & set(HOLDOUT_SEEDS):
    raise RuntimeError("clone training and holdout seasons share a seed")
if set(TRAIN_SEEDS + HOLDOUT_SEEDS) & set(_GATE_SEEDS):
    raise RuntimeError(
        "clone seeds intersect holdout.GATE_SEEDS: a season the clone trained "
        "on cannot also be the exam it is graded on"
    )
if set(TRAIN_SEEDS + HOLDOUT_SEEDS) & set(_SEARCH_SEED_POOL):
    raise RuntimeError("clone seeds intersect hillclimb.SEED_POOL")

TRAIN = "train"
HOLDOUT = "holdout"

# One row is 48 board planes of 10x10 float32 (19.2 KB) plus a few hundred
# bytes of scalars, positions and labels, so a shard is ~1 GB at this cap. A
# season is 719 rows; this caps a shard at roughly 70 of them.
ROWS_PER_SHARD = 50_000

_ARRAYS = ("boards", "scalars", "positions", "labels", "quantities", "markets")

# Who drives a recorded season. The labels are the teacher's either way --
# that is what makes this a clone -- and the only thing this changes is whose
# states get labelled. ``TEACHER_ACTOR`` gives plain behaviour cloning, on the
# teacher's own distribution. ``CLONE_ACTOR`` gives one DAgger round: the
# clone plays, and the teacher answers "what should you have done here", which
# is the one question a fixed dataset of the teacher's own seasons can never
# ask. It is the measurement that separates a policy class that cannot express
# the teacher from a policy class that can but never sees the states its own
# mistakes lead to.
TEACHER_ACTOR = "teacher"
CLONE_ACTOR = "clone"

# The clone as the engine's loader sees it: `learn.play` with the checkpoint
# rebound. Named here rather than in the gate script so that the recording and
# the gate cannot end up playing two different files.
CLONE_AGENT = "src/kaggriculture/learn/clone_play.py"


@dataclass(frozen=True)
class Game:
    """One recorded season: ``actor`` in ``seat`` against ``opponent``.

    The teacher labels every turn whoever is acting, so a ``CLONE_ACTOR``
    season is a DAgger round and a ``TEACHER_ACTOR`` season is ordinary
    behaviour cloning.
    """

    opponent: str
    seed: int
    seat: int
    actor: str = TEACHER_ACTOR


class Season(BaseModel):
    """What one recorded season was, and how it ended.

    The banks come from the engine's own terminal rewards, so a season this
    module reports on is a season the engine actually played to the end. They
    are kept alongside the rows because "did the clone agree with the teacher
    where the teacher was doing badly" is a question about the season, not
    about any one turn, and it cannot be asked after the fact from the tensors.
    """

    opponent: str
    seed: int
    seat: int
    actor: str
    ours: int
    theirs: int
    rows: int
    skipped: int


class Recording(BaseModel):
    """The manifest written beside a build's shards."""

    teacher: str
    seasons: list[Season]


def record(game: Game) -> tuple[Season, dict[str, np.ndarray]]:
    """Play one season with ``game.actor`` in ``game.seat`` and label every turn.

    The acting agent is wrapped in a closure that encodes the observation it was
    handed and the action the *teacher* would take from it, in that order,
    inside the same call. No step index is read and ``environment.steps`` is
    never walked for a label: the pair recorded is literally the input and the
    teacher's output for one decision.

    The episode is run through ``Environment.run`` rather than stepped, because
    ``run`` is what ``arena`` uses and the gate has to play the same engine
    path the recording did. A season that did not finish cleanly, or that did
    not record a decision for every turn, raises rather than contributing a
    truncated set of rows.

    A turn whose acting-unit count exceeds ``MAX_UNITS`` is skipped and
    counted, the way ``dataset.build_shard`` skips one, so an unexpectedly
    non-zero count surfaces as a finding about the bound rather than a silent
    loss. The teacher's observed maximum is 13 acting units against a bound of
    20, so this is expected to stay zero.

    Args:
        game: Which opponent, seed, seat and actor to play.

    Returns:
        The season's outcome and its row arrays, keyed by ``_ARRAYS``.

    Raises:
        ValueError: If ``game.actor`` names neither the teacher nor the clone.
        RuntimeError: If the episode did not end with both seats ``DONE``, or
            did not record a decision for every turn.
    """
    teacher = get_last_callable(Path(TEACHER).read_text(), path=TEACHER)
    if game.actor == TEACHER_ACTOR:
        actor = teacher
    elif game.actor == CLONE_ACTOR:
        actor = get_last_callable(Path(CLONE_AGENT).read_text(), path=CLONE_AGENT)
    else:
        raise ValueError(f"a season is driven by the teacher or the clone: {game}")
    rows: list[list[np.ndarray]] = [[] for _ in _ARRAYS]
    skipped = 0

    def recorder(
        observation: Mapping[str, Any],
        configuration: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        nonlocal skipped
        action = teacher(observation, configuration)
        # `learn.play.agent` takes the observation alone, where the teacher
        # takes the engine's two arguments. The engine itself dispatches on
        # `co_argcount`; called directly, the arity has to be right here.
        played = action if actor is teacher else actor(observation)
        seat = int(observation["player"])
        units = unit_count(observation, seat)
        try:
            labels = encode_units(action, units)
            quantities = encode_unit_quantities(action, units)
        except TooManyUnitsError as error:
            LOGGER.warning("skipping a turn of %s: %s", game, error)
            skipped += 1
            return played
        encoded = encode_observation(observation, seat)
        for store, value in zip(
            rows,
            (
                np.asarray(encoded.board, dtype=np.float32)[None],
                np.asarray(encoded.scalars, dtype=np.float32)[None],
                np.asarray(encoded.positions, dtype=np.int64)[None],
                labels.numpy(force=True),
                quantities.numpy(force=True),
                encode_market(action).numpy(force=True),
            ),
            strict=True,
        ):
            store.append(value)
        return played

    environment = make(
        ENVIRONMENT,
        configuration={"episodeSteps": EPISODE_STEPS, "seed": game.seed},
    )
    other = OPPONENTS[game.opponent]
    environment.run([recorder, other] if game.seat == 0 else [other, recorder])
    final = environment.steps[-1]
    statuses = (final[0].status, final[1].status)
    if statuses != ("DONE", "DONE") or any(state.reward is None for state in final):
        raise RuntimeError(f"{game} did not finish cleanly (statuses={statuses})")
    # A season that reports DONE is not yet a season that was recorded. An
    # agent that raises every turn is caught by `kaggle_environments`' own
    # wrapper, which logs it and hands the interpreter an error object in place
    # of an action; the interpreter ignores it, the episode runs to the horizon
    # and every seat finishes DONE with the farm untouched. The only symptom is
    # a recording with no rows in it, which is exactly the shape a silent
    # failure takes here -- so the turn count is checked, not the status alone.
    if len(rows[0]) + skipped != EPISODE_STEPS - 1:
        raise RuntimeError(
            f"{game} recorded {len(rows[0])} rows and skipped {skipped} of "
            f"{EPISODE_STEPS - 1} turns; the acting agent did not decide every "
            "turn it was asked to"
        )
    banks = (int(final[0].reward), int(final[1].reward))
    season = Season(
        opponent=game.opponent,
        seed=game.seed,
        seat=game.seat,
        actor=game.actor,
        ours=banks[game.seat],
        theirs=banks[1 - game.seat],
        rows=len(rows[0]),
        skipped=skipped,
    )
    return season, {
        name: np.concatenate(store) for name, store in zip(_ARRAYS, rows, strict=True)
    }


def games(seeds: tuple[int, ...], actor: str = TEACHER_ACTOR) -> list[Game]:
    """Return every opponent, seed and seat ordering for one seed block."""
    return [
        Game(opponent=opponent, seed=seed, seat=seat, actor=actor)
        for opponent in OPPONENTS
        for seed in seeds
        for seat in (0, 1)
    ]


def write_shards(
    schedule: list[Game],
    destination: Path,
    workers: int | None = None,
    start: int = 0,
) -> Recording:
    """Play every game in ``schedule`` and write its rows to numbered shards.

    Seasons are independent and each is a whole episode of pure-Python engine
    work, so they are fanned out across processes. Rows are flushed as soon as
    ``ROWS_PER_SHARD`` is reached, so a build never holds more than one shard's
    tensors in memory at once.

    Args:
        schedule: The seasons to play.
        destination: Path for the first (or only) shard.
        workers: Processes to spread seasons over, or None for the default.
        start: The first shard index to write. A DAgger round appends to a
            family that already has shards in it, and passing the index it may
            begin at is what keeps it from overwriting them. 0 writes
            ``destination`` itself, which is the only name ``shards_named``
            requires to exist.

    Returns:
        The manifest of what was played.
    """
    stores: list[list[np.ndarray]] = [[] for _ in _ARRAYS]
    held = 0
    shard_index = start
    seasons: list[Season] = []

    def flush() -> None:
        nonlocal shard_index, held
        if not held:
            return
        path = (
            destination
            if shard_index == 0
            else destination.with_name(
                f"{destination.stem}-{shard_index:03d}{destination.suffix}"
            )
        )
        stacked = {
            name: np.concatenate(store)
            for name, store in zip(_ARRAYS, stores, strict=True)
        }
        # Named one by one rather than splatted: `np.savez` takes its own
        # `allow_pickle` as a keyword alongside the arrays, so a splatted
        # mapping is a mapping that could name it.
        np.savez(
            path,
            boards=stacked["boards"],
            scalars=stacked["scalars"],
            positions=stacked["positions"],
            labels=stacked["labels"],
            quantities=stacked["quantities"],
            markets=stacked["markets"],
        )
        LOGGER.info("%s: %d rows", path.name, held)
        shard_index += 1
        held = 0
        for store in stores:
            store.clear()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for season, arrays in tqdm(
            pool.map(record, schedule), total=len(schedule), unit="season"
        ):
            seasons.append(season)
            for name, store in zip(_ARRAYS, stores, strict=True):
                store.append(arrays[name])
            held += season.rows
            if held >= ROWS_PER_SHARD:
                flush()
    flush()
    return Recording(teacher=TEACHER, seasons=seasons)


# Where a DAgger round's shards start numbering, far above anything the
# teacher-driven build writes, so the two blocks stay legible on disk and a
# round can never overwrite the data it is meant to be added to.
DAGGER_SHARD_START = 100


def dagger(directory: Path, workers: int | None = None) -> Recording:
    """Play the current clone and label its own states with the teacher's answer.

    This is what a fixed dataset cannot provide. Behaviour cloning only ever
    sees the states the teacher's own play reaches; the moment the clone makes
    a mistake it is somewhere the teacher never went, and nothing in the
    training set says what to do there. One DAgger round labels exactly those
    states, and comparing a gate before and after it separates two very
    different failures that produce the same 0.000: a policy class that cannot
    express the teacher at all, and one that can but has never been shown how
    to recover.

    The shards are appended to the training family rather than replacing it --
    DAgger trains on the union, not on the newest round alone -- and the
    holdout is untouched, so accuracy stays comparable across rounds.

    Args:
        directory: Where the teacher-driven build already wrote its shards.
        workers: Processes to spread seasons over, or None for the default.

    Returns:
        The manifest of the round.
    """
    recording = write_shards(
        games(TRAIN_SEEDS, CLONE_ACTOR),
        directory / f"{TRAIN}.npz",
        workers,
        start=DAGGER_SHARD_START,
    )
    (directory / "dagger.json").write_text(recording.model_dump_json(indent=1))
    LOGGER.info(
        "dagger: %d seasons, %d rows, clone win rate %.3f",
        len(recording.seasons),
        sum(season.rows for season in recording.seasons),
        win_rate(recording),
    )
    return recording


def build(directory: Path, workers: int | None = None) -> None:
    """Clear a previous build and record both seed blocks into it.

    Clearing is bound to writing for the same reason it is in
    ``scripts.build.write_dataset``: a leftover shard from a longer previous
    build is named like a real one and carries identical shapes, so it would
    concatenate into training without complaint.

    Args:
        directory: Where to write the shards and the manifests.
        workers: Processes to spread seasons over, or None for the default.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stale = sorted(directory.glob("*.npz"))
    for shard in stale:
        shard.unlink()
    if stale:
        LOGGER.info("cleared %d shard(s) from a previous build", len(stale))
    for stem, seeds in ((TRAIN, TRAIN_SEEDS), (HOLDOUT, HOLDOUT_SEEDS)):
        recording = write_shards(games(seeds), directory / f"{stem}.npz", workers)
        (directory / f"{stem}.json").write_text(recording.model_dump_json(indent=1))
        LOGGER.info(
            "%s: %d seasons, %d rows, teacher win rate %.3f",
            stem,
            len(recording.seasons),
            sum(season.rows for season in recording.seasons),
            win_rate(recording),
        )


def win_rate(recording: Recording) -> float:
    """Return the actor's win rate over a recording, ties counting a half."""
    if not recording.seasons:
        return 0.0
    points = sum(
        1.0
        if season.ours > season.theirs
        else 0.5
        if season.ours == season.theirs
        else 0.0
        for season in recording.seasons
    )
    return points / len(recording.seasons)


def shards_named(directory: Path, stem: str) -> list[Path]:
    """Return one build's shards under ``stem``, first shard first.

    Args:
        directory: Where ``build`` wrote its shards.
        stem: ``TRAIN`` or ``HOLDOUT``.

    Returns:
        The shards under ``stem``, first shard first.

    Raises:
        FileNotFoundError: If the first shard is missing.
    """
    first = directory / f"{stem}.npz"
    if not first.is_file():
        raise FileNotFoundError(f"no {stem}.npz in {directory}")
    return [first, *sorted(directory.glob(f"{stem}-[0-9][0-9][0-9].npz"))]


def built_shards(directory: Path) -> tuple[list[Path], list[Path]]:
    """Return the training and holdout shards, refusing anything unaccounted for.

    Args:
        directory: Where ``build`` wrote its shards.

    Returns:
        The training shards and the holdout shards.

    Raises:
        ValueError: If the directory holds an ``.npz`` belonging to neither.
    """
    training = shards_named(directory, TRAIN)
    held = shards_named(directory, HOLDOUT)
    unaccounted = set(directory.glob("*.npz")) - set(training) - set(held)
    if unaccounted:
        raise ValueError(
            f"{directory} holds {sorted(path.name for path in unaccounted)}, which "
            f"belongs to neither the {TRAIN} nor the {HOLDOUT} shards"
        )
    return training, held


def read_manifest(directory: Path, stem: str) -> Recording:
    """Return the manifest written beside one shard family."""
    return Recording.model_validate_json((directory / f"{stem}.json").read_text())


class TeacherShards(Dataset):
    """A dataset over one or more recorded shards, held in memory."""

    def __init__(self, paths: list[Path]) -> None:
        """Load every shard named by ``paths``."""
        loaded: list[list[np.ndarray]] = [[] for _ in _ARRAYS]
        for path in paths:
            with np.load(path) as data:
                for store, name in zip(loaded, _ARRAYS, strict=True):
                    store.append(data[name])
        self.tensors = tuple(
            torch.from_numpy(np.concatenate(store)) for store in loaded
        )

    def __len__(self) -> int:
        """Return how many rows this dataset holds."""
        return int(self.tensors[0].shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        """Return one row's board, scalars, positions and three label sets."""
        return tuple(tensor[index] for tensor in self.tensors)


def season_slices(recording: Recording) -> Iterator[tuple[Season, slice]]:
    """Yield each season alongside the rows it contributed, in build order.

    Rows are concatenated in the order ``write_shards`` played them, which is
    the order of ``Recording.seasons``, so a season's rows are one contiguous
    block and no per-row season column has to be stored.

    Args:
        recording: The manifest written beside the shards.

    Yields:
        Each season and the slice of the concatenated dataset holding it.
    """
    start = 0
    for season in recording.seasons:
        yield season, slice(start, start + season.rows)
        start += season.rows
