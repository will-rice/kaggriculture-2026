"""Two evaluations: a fast one for ranking, a deep one that decides.

Fast plays fresh non-exam seeds through the harness; deep plays the sealed
exam block through the field gate, both seats, every pool opponent and the
held-out set, and reports Wilson intervals. Every opponent counts the same
-- the gate asks whether a candidate beats each of them, not what it
averages -- so both scores are the plain mean of the per-opponent rates and
neither is the number a promotion turns on.

Neither plays a program against itself. A champion is a member of the pool
it is re-scored on, and its own bytes in the other seat are a structural
0.5 that no metric should carry: ``opponents`` drops that entry before any
game is played, so the mean score, the per-opponent rates, ``field``
and the intervals are all over real opponents and nothing downstream has to
know the mirror ever existed.
"""

import random
from pathlib import Path

from pydantic import BaseModel

from kaggriculture.campaign import config, field_gate, harness, roster
from kaggriculture.campaign.pool import Pool
from kaggriculture.report import wilson_interval

VENDORED = list(roster.TRAINING)
HELD_OUT = list(roster.HELD_OUT)


class FastResult(BaseModel):
    """A ranking measurement on seeds nothing else has seen.

    Attributes:
        fitness: Mean win rate over the pool.
        rates: Win rate per pool opponent, ties as half.
        seeds: The seeds drawn for this call.
    """

    fitness: float
    rates: dict[str, float]
    seeds: list[int]


class DeepResult(BaseModel):
    """The gate's verdict on one program over the sealed exam block.

    Attributes:
        program_id: The program this measures.
        score: Mean win rate over the pool opponents only.
        low: Conservative lower bound on ``score``: the mean of the
            per-opponent Wilson lower bounds. Each of those holds at 95%, so
            their mean is at least as wide as the exact interval would be.
        high: Conservative upper bound on ``score``, by the same mean of the
            per-opponent Wilson upper bounds.
        rates: Win rate per pool opponent.
        intervals: Wilson interval per pool opponent.
        field: Mean rate over the vendored opponents still in the pool.
        held_out: Win rate per held-out opponent; never part of ``score``.
        games: Games played against each opponent.
    """

    program_id: str
    score: float
    low: float
    high: float
    rates: dict[str, float]
    intervals: dict[str, tuple[float, float]]
    field: float
    held_out: dict[str, float]
    games: int


def _mean(rates: dict[str, float]) -> float:
    """The mean of ``rates``; every opponent counts the same."""
    return sum(rates.values()) / len(rates)


def _rates(games: list[harness.Game], names: list[str]) -> dict[str, float]:
    """Win rate per opponent over the games played against it, ties as half."""
    out = {}
    for name in names:
        mine = [g for g in games if g.opponent == name]
        out[name] = sum(
            1.0 if g.ours > g.theirs else 0.5 if g.ours == g.theirs else 0.0
            for g in mine
        ) / len(mine)
    return out


def opponents(pool: Pool, program_id: str, agent: Path) -> Pool:
    """``pool`` without ``program_id``; ``pool`` itself if it is not in it.

    A champion is a member of the pool it is scored on, and its own bytes in
    the other seat are a structural 0.5 that says nothing about the program.
    The match is by name, which is enough because the only caller that scores
    a pool member passes the pool's own key as the ``program_id``, and the
    pool registers that key against the very file being scored.

    Args:
        pool: The opponents as they stand.
        program_id: The program being scored.
        agent: The file being scored.

    Returns:
        The opponents to play.

    Raises:
        ValueError: The pool holds that name against a different file, which
            is a disagreement this cannot settle by looking.
    """
    if program_id not in pool.opponents:
        return pool
    if Path(pool.opponents[program_id]) != agent:
        raise ValueError(
            f"the pool's {program_id} is {pool.opponents[program_id]}, not the "
            f"{agent} being scored"
        )
    return Pool(opponents={n: p for n, p in pool.opponents.items() if n != program_id})


def fast(
    agent: Path, program_id: str, pool: Pool, rng: random.Random, workers: int
) -> FastResult:
    """Mean win rate over ``FAST_SEEDS`` fresh seeds, both seats.

    The seeds are drawn per call from ``FAST_SEED_RANGE``, which excludes the
    exam block, so ranking pressure never touches the seeds the gate decides
    on and no two candidates are ranked on a block that could be memorised.

    Args:
        agent: The candidate's ``main.py``.
        program_id: The program being scored, so it is not among them.
        pool: The opponents to measure against.
        rng: The generator the seeds are drawn from.
        workers: Processes to fan the games over.

    Returns:
        The mean fitness, the per-opponent rates, and the seeds drawn.

    Raises:
        RuntimeError: A side raised during a game. A crashed candidate is a
            failed evaluation, never a zero score, so this propagates.
    """
    measured = opponents(pool, program_id, agent)
    seeds = rng.sample(config.FAST_SEED_RANGE, config.FAST_SEEDS)
    games = harness.play(agent, measured.names(), seeds, workers)
    rates = _rates(games, measured.names())
    return FastResult(fitness=_mean(rates), rates=rates, seeds=seeds)


def deep(agent: Path, program_id: str, pool: Pool, workers: int) -> DeepResult:
    """The gate: exam block x both seats x pool and held-out, with intervals.

    Playing a candidate executes it, and a candidate is evolved source that
    may write files; ``harness._one`` runs every game in a scratch directory
    in its own process, which is where that is contained. Nothing here moves
    the working directory: the loop runs deep evaluations concurrently with
    each other and with fast ones, and the working directory is one per
    process, so a move here would be a race rather than an isolation.

    Args:
        agent: The candidate's ``main.py``.
        program_id: The program this measurement belongs to, and the pool
            entry it is therefore not played against.
        pool: The opponents that count towards the score.
        workers: Processes to fan the games over.

    Returns:
        The mean score with its interval, the per-opponent rates and
        intervals, the vendored-field rate, and the held-out rates.

    Raises:
        RuntimeError: A side raised during a game. A crashed candidate is a
            failed evaluation, never a zero score, so this propagates.
        ValueError: The pool holds no vendored opponent, so there is no field
            to average over.
    """
    measured = opponents(pool, program_id, agent)
    names = measured.names()
    played = names + [name for name in HELD_OUT if name not in names]
    rates = field_gate.score_field(
        agent, seeds=config.EXAM_SEEDS, workers=workers, opponents=played
    )
    games = 2 * len(config.EXAM_SEEDS)
    intervals = {name: wilson_interval(rates[name] * games, games) for name in names}
    vendored = [name for name in VENDORED if name in names]
    if not vendored:
        raise ValueError("pool holds no vendored opponent; field is undefined")
    return DeepResult(
        program_id=program_id,
        score=_mean({name: rates[name] for name in names}),
        low=_mean({n: bounds[0] for n, bounds in intervals.items()}),
        high=_mean({n: bounds[1] for n, bounds in intervals.items()}),
        rates={name: rates[name] for name in names},
        intervals=intervals,
        field=sum(rates[name] for name in vendored) / len(vendored),
        held_out={name: rates[name] for name in HELD_OUT},
        games=games,
    )
