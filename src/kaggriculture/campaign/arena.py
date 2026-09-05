"""Reference-engine games between two agent files.

The reference engine is the oracle and the Kaggle runner; the port under
``campaign.engine`` is the fast copy. Anything that must be true on Kaggle
is measured here.
"""

from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter

from kaggle_environments import make

from kaggriculture.constants import ENVIRONMENT, EPISODE_STEPS


class OutcomeScores(list[float]):
    """Win points with the paired bank margins, failures and duration behind them."""

    def __init__(self) -> None:
        super().__init__()
        self.margins: list[int] = []
        self.members: list[str] = []
        self.failures: list[str] = []
        self.runtime_seconds = 0.0


def run_banks(seat_zero: str, seat_one: str, seed: int) -> tuple[int, int]:
    """Play one reference-engine episode and return both final banks.

    Raises:
        RuntimeError: If either seat held a status other than ACTIVE, DONE or
            INACTIVE at any step. A crashed agent banks its untouched 3000 and
            would otherwise read as an ordinary loss.
    """
    environment = make(
        ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS, "seed": seed}
    )
    environment.run([seat_zero, seat_one])
    final = environment.steps[-1]
    statuses = (final[0].status, final[1].status)
    if (
        statuses != ("DONE", "DONE")
        or final[0].reward is None
        or final[1].reward is None
    ):
        raise RuntimeError(f"seed {seed} did not finish cleanly (statuses={statuses})")
    for seat in (0, 1):
        broken = {
            step[seat].status
            for step in environment.steps
            if step[seat].status not in ("ACTIVE", "DONE", "INACTIVE")
        }
        if broken:
            raise RuntimeError(
                f"seed {seed} seat {seat} held {sorted(broken)} during the episode"
            )
    return int(final[0].reward), int(final[1].reward)


def _one(work: tuple[str, str, int]) -> tuple[int, int]:
    return run_banks(*work)


def play(
    seat_zero: str, seat_one: str, seeds: Sequence[int], workers: int
) -> list[tuple[int, int]]:
    """Play every seed with the given seating, fanned over a process pool."""
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as pool:
        return list(pool.map(_one, [(seat_zero, seat_one, seed) for seed in seeds]))


_Work = tuple[str, int, int, str, str]
_Outcome = tuple[str, int, int, tuple[int, int] | None, str | None]


def _one_outcome(work: _Work) -> _Outcome:
    name, seed, seat, seat_zero, seat_one = work
    try:
        return name, seed, seat, run_banks(seat_zero, seat_one, seed), None
    except RuntimeError as error:
        return name, seed, seat, None, f"{name} seed {seed} seat {seat}: {error}"


def outcomes(
    candidate: str, league: Mapping[str, str], seeds: Sequence[int], workers: int
) -> OutcomeScores:
    """Score the candidate against every league member, both seats, ties as half.

    Games are played in a fixed order -- league member, then seed, then seat
    (candidate in seat zero, then seat one) -- but a game that raises is
    recorded in ``failures`` and contributes no entry to ``scores`` or
    ``scores.members``, so a failure shortens its member's own run rather than
    shifting every later member's games into the wrong slot. ``members[i]``
    names whose game ``scores[i]`` (and ``margins[i]``) belongs to; that
    parallel list, not position, is what ``summarize`` attributes by.
    """
    started = perf_counter()
    work: list[_Work] = [
        (name, seed, seat, *seating)
        for name, opponent in league.items()
        for seed in seeds
        for seat, seating in ((0, (candidate, opponent)), (1, (opponent, candidate)))
    ]
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as pool:
        results = list(pool.map(_one_outcome, work))
    scores = OutcomeScores()
    for name, _seed, seat, banks, failure in results:
        if failure is not None:
            scores.failures.append(failure)
            continue
        assert banks is not None
        ours, theirs = banks if seat == 0 else banks[::-1]
        scores.append(_win(ours, theirs))
        scores.margins.append(ours - theirs)
        scores.members.append(name)
    scores.runtime_seconds = perf_counter() - started
    return scores


def summarize(scores: OutcomeScores, league: Mapping[str, str]) -> dict[str, float]:
    """One win rate per league member, over the games actually scored for it.

    Attribution is by ``scores.members``, not by position, so a member that
    lost games to failures is averaged over the games it actually has rather
    than a fixed-width slice that would drift into a neighbour's scores. A
    member with zero scored games (every one of its games failed) gets 0.0.
    """
    summary = {}
    for name in league:
        games = [
            score
            for score, member in zip(scores, scores.members, strict=True)
            if member == name
        ]
        summary[name] = sum(games) / len(games) if games else 0.0
    return summary


def _win(ours: int, theirs: int) -> float:
    return 1.0 if ours > theirs else 0.5 if ours == theirs else 0.0
