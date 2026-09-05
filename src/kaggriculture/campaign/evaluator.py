"""Two evaluations: a fast one for ranking, a deep one that decides.

Fast plays fresh non-exam seeds through the harness, weighted by the pool.
Deep plays the sealed exam block through the field gate, both seats, every
pool opponent and the held-out set, and reports Wilson intervals.
"""

import os
import random
import tempfile
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
        fitness: Pool-weighted win rate.
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
        score: Pool-weighted win rate over the pool opponents only.
        low: Wilson lower bound on ``score``.
        high: Wilson upper bound on ``score``.
        rates: Win rate per pool opponent.
        intervals: Wilson interval per pool opponent.
        field: Equal-weighted rate over the vendored opponents still in the pool.
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


def fast(agent: Path, pool: Pool, rng: random.Random, workers: int) -> FastResult:
    """Pool-weighted win rate over ``FAST_SEEDS`` fresh seeds, both seats.

    The seeds are drawn per call from ``FAST_SEED_RANGE``, which excludes the
    exam block, so ranking pressure never touches the seeds the gate decides
    on and no two candidates are ranked on a block that could be memorised.

    Args:
        agent: The candidate's ``main.py``.
        pool: The opponents to measure against, with their weights.
        rng: The generator the seeds are drawn from.
        workers: Processes to fan the games over.

    Returns:
        The weighted fitness, the per-opponent rates, and the seeds drawn.

    Raises:
        RuntimeError: A side raised during a game. A crashed candidate is a
            failed evaluation, never a zero score, so this propagates.
    """
    seeds = rng.sample(config.FAST_SEED_RANGE, config.FAST_SEEDS)
    games = harness.play(agent, pool.names(), seeds, workers)
    rates = _rates(games, pool.names())
    return FastResult(fitness=pool.weighted(rates), rates=rates, seeds=seeds)


def deep(agent: Path, program_id: str, pool: Pool, workers: int) -> DeepResult:
    """The gate: exam block x both seats x pool and held-out, with intervals.

    Playing a candidate executes it, and a candidate is evolved source that
    may write files, so the games run with the working directory moved into a
    scratch tree. The candidate is resolved to an absolute path first, because
    a relative one stops resolving the moment that move happens.

    Args:
        agent: The candidate's ``main.py``.
        program_id: The program this measurement belongs to.
        pool: The opponents that count towards the score, with their weights.
        workers: Processes to fan the games over.

    Returns:
        The weighted score with its interval, the per-opponent rates and
        intervals, the vendored-field rate, and the held-out rates.

    Raises:
        RuntimeError: A side raised during a game. A crashed candidate is a
            failed evaluation, never a zero score, so this propagates.
    """
    names = pool.names()
    opponents = names + [name for name in HELD_OUT if name not in names]
    candidate = agent.resolve()
    origin = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="campaign-deep-") as sandbox:
        os.chdir(sandbox)
        try:
            rates = field_gate.score_field(
                candidate, seeds=config.EXAM_SEEDS, workers=workers, opponents=opponents
            )
        finally:
            os.chdir(origin)
    games = 2 * len(config.EXAM_SEEDS)
    weighted = pool.weighted(rates)
    total = games * len(names)
    low, high = wilson_interval(weighted * total, total)
    vendored = [name for name in VENDORED if name in names]
    return DeepResult(
        program_id=program_id,
        score=weighted,
        low=low,
        high=high,
        rates={name: rates[name] for name in names},
        intervals={name: wilson_interval(rates[name] * games, games) for name in names},
        field=sum(rates[name] for name in vendored) / len(vendored),
        held_out={name: rates[name] for name in HELD_OUT},
        games=games,
    )
