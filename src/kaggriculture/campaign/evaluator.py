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

    It is also the whole of what a model is told about the program it is
    asked to improve: the rates and margins are the verdict, and ``states``
    is the game behind it.

    Attributes:
        fitness: Mean win rate over the pool, which grows as champions join
            it, so it ranks the database but does not compare across time.
        field: Mean win rate over the vendored incumbents alone. They never
            change, so this is the one number that means the same thing on
            the first session and the thousandth -- it is what "the best is
            rising" is measured on.
        rates: Win rate per pool opponent, ties as half.
        margins: Bank margin per pool opponent.
        seeds: The seeds drawn for this call.
        hardest: The opponent with the lowest win rate.
        states: One game against ``hardest``, day by day -- the one it lost
            by the most, which is a lost game whenever it lost any.
    """

    fitness: float
    field: float
    rates: dict[str, float]
    margins: dict[str, harness.Margin]
    seeds: list[int]
    hardest: str
    states: list[harness.Day]


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
        margins: Bank margin per pool opponent. Defaulted, so a result
            written before margins existed still loads out of the log.
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
    margins: dict[str, harness.Margin] = {}
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

    Every game is played with its day table recorded, because one of them
    is what the loop shows a model of how its program played: the game
    against the opponent it does worst against that it lost by the most.
    Taking the minimum margin is what makes that a lost game whenever one
    exists -- every loss is below every tie and every win -- and the
    narrowest win when the program lost nothing at all.

    Args:
        agent: The candidate's ``main.py``.
        program_id: The program being scored, so it is not among them.
        pool: The opponents to measure against.
        rng: The generator the seeds are drawn from.
        workers: Processes to fan the games over.

    Returns:
        The mean fitness, the per-opponent rates and margins, the seeds
        drawn, and one game against the hardest opponent day by day.

    Raises:
        RuntimeError: A side raised during a game. A crashed candidate is a
            failed evaluation, never a zero score, so this propagates.
    """
    measured = opponents(pool, program_id, agent)
    names = measured.names()
    seeds = rng.sample(config.FAST_SEED_RANGE, config.FAST_SEEDS)
    games = harness.play(agent, names, seeds, workers, days=True)
    rates = _rates(games, names)
    # Ties on the rate are broken by the margin, because before the first win
    # every rate is 0.0 and `min` would otherwise always name the first pool
    # key -- so every session would be shown the same opponent rather than the
    # one it came closest to beating.
    margins = harness.margins(games, names)
    hardest = min(rates, key=lambda name: (rates[name], margins[name].mean))
    shown = min(
        (game for game in games if game.opponent == hardest),
        key=lambda game: game.ours - game.theirs,
    )
    return FastResult(
        fitness=_mean(rates),
        field=vendored_field(rates),
        rates=rates,
        margins=margins,
        seeds=seeds,
        hardest=hardest,
        states=shown.days,
    )


def vendored_field(rates: dict[str, float]) -> float:
    """Mean win rate over the vendored incumbents in ``rates``.

    The pool grows as champions join it, so a mean over the pool moves for
    reasons that have nothing to do with a program improving. The vendored
    kernels are fixed, so this number is comparable across the whole
    campaign.

    Args:
        rates: Win rate per opponent name.

    Returns:
        The mean over the vendored names present.

    Raises:
        ValueError: None of them is present, so the field is undefined.
    """
    vendored = {name: rate for name, rate in rates.items() if name in VENDORED}
    if not vendored:
        raise ValueError("no vendored opponent was played; the field is undefined")
    return _mean(vendored)


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
        The mean score with its interval, the per-opponent rates, margins and
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
    rates, margins = field_gate.score_field(
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
        margins={name: margins[name] for name in names},
        intervals=intervals,
        field=sum(rates[name] for name in vendored) / len(vendored),
        held_out={name: rates[name] for name in HELD_OUT},
        games=games,
    )
