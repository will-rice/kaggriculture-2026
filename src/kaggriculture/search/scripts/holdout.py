"""The exam a searched route must pass before it ships.

The hill-climb selects routes by playing them in its own arena, so its output
is a selection maximum: the accepted route is, by construction, whatever
looked best on the boards that chose it. This module is the exam the search
never sees -- a fixed block of reference-engine games, seat-swapped, against
the agent the competition currently fields, with a pass/fail rule fixed before
any result exists (design: docs/superpowers/specs/2026-08-16-route-search-design.md,
S5).

**A change ships when it beats the agent we currently serve, seat-swapped over
128 held-out reference-engine games, with the win rate's 95% interval excluding
0.5.** Both the game count and the pass rule are fixed here, before this module
has ever run against a real candidate -- they are not tunable after the fact
because a result came in low.

That single condition is not the whole exam. On 2026-08-20 a searched route
passed it at 0.734 [0.652, 0.803] over 128 seat-swapped games against the agent
we served, so it was served and submitted -- and converged roughly 130 rating
points *below* the agent it replaced, because rating is set at the frontier,
and above 1400 the route lost to the current meta about 4-to-1 (see
``main.py``). "Does it beat what we serve" and "is it worth fielding" are
different questions once the served agent is not itself at the frontier. The
second condition below adds the frontier to the exam: the candidate must also
be statistically not worse than a pool of routes harvested from the newest
top-band archive, or the PASS above is examining only history.

``GATE_SEEDS`` is disjoint from ``hillclimb.SEED_POOL`` on purpose: a seed the
hill-climb could have rotated its search block onto is a seed the candidate may
already be specialised for, and an exam drawn from seeds the candidate could
have seen during search is not a held-out exam -- it is one more round of the
same selection maximum this module exists to check. That disjointness is
checked below at import time, not only in a test, so widening the search's
seed pool later cannot silently void the exam.
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from kaggriculture.report import wilson_interval
from kaggriculture.search import arena
from kaggriculture.search.route import Route, load
from kaggriculture.search.scripts.hillclimb import SEED_POOL as _SEARCH_SEED_POOL

LOGGER = logging.getLogger(__name__)

# 64 seeds, each played in both seat orderings by arena.outcomes: 128 games.
GATE_SEEDS: tuple[int, ...] = tuple(range(700_000, 700_064))

if any(seed in _SEARCH_SEED_POOL for seed in GATE_SEEDS):
    raise RuntimeError(
        "GATE_SEEDS intersects hillclimb.SEED_POOL: seeds the search ever saw "
        "cannot serve as its exam"
    )

# The agent the competition currently fields. This must track what main.py
# imports -- today `from kaggriculture.boatlee_v14_policy import agent` --
# because the exam this module runs is "does the candidate beat what we would
# otherwise submit", not some other opponent. `test_holdout.py` parses
# main.py's own import to check this stays true.
SERVED = "src/kaggriculture/boatlee_v14_policy.py"

# Routes harvested from the newest published episode archive by
# `build_league.harvest`, refreshed by the operator, not by this module. This
# module only reads it -- it never writes here.
FRONTIER_DIR = Path("/data/kaggriculture/search/frontier-league")

# Older than this and the frontier pool is examining history, not the frontier
# -- the 2026-08-20 failure with extra steps. The gate raises rather than
# grading against a stale pool.
FRONTIER_STALE_DAYS = 7

# Derivation (fixed before any candidate is seen; do not retune after a
# result comes in low): the frontier pool this gate expects is >= 3 harvested
# tapes, each seat-swapped over GATE_SEEDS, i.e. >= 3 * 128 = 384 games. At
# n = 384, z = 1.96 (report.wilson_interval's default), a pooled win rate of
# exactly 0.500 (192/384) gives a Wilson lower bound of 0.4502 -- just clears
# 0.45. A rate of 0.499 gives ~0.4492 and fails. The bar is deliberately
# knife-edged at "statistically indistinguishable from a coin flip against the
# frontier", not "better than one": a candidate that is a true peer of the
# newest top-band tapes passes; the 2026-08-20 route, which pooled to
# 0.36-0.38 against them, fails by a wide margin.
FRONTIER_LOW_THRESHOLD = 0.45


@dataclass(frozen=True)
class Verdict:
    """The exam's result: two win rates, their bounds, and the ruling.

    Attributes:
        rate: Win rate vs. the served agent over ``games``, ties counted as
            half a win.
        games: Games played vs. the served agent -- ``2 * len(GATE_SEEDS)``.
        low: Wilson lower bound at 95% vs. the served agent.
        high: Wilson upper bound at 95% vs. the served agent.
        frontier_rate: Pooled win rate over the frontier league, ties counted
            as half a win.
        frontier_low: Pooled Wilson lower bound at 95% over the frontier
            league.
        passed: True iff ``low`` exceeds 0.5 vs. the served agent AND
            ``frontier_low`` is at least ``FRONTIER_LOW_THRESHOLD`` -- beating
            what we serve is necessary but not sufficient; the candidate must
            also hold up against the frontier.
    """

    rate: float
    games: int
    low: float
    high: float
    frontier_rate: float
    frontier_low: float
    passed: bool


def run(
    candidate: Route,
    against: str = SERVED,
    frontier: Path = FRONTIER_DIR,
    workers: int | None = None,
) -> Verdict:
    """Play the exam: ``candidate`` vs. ``against`` and vs. the frontier pool.

    Scoring is delegated entirely to ``arena.outcomes`` -- this module fixes
    which seeds and which opponents the exam uses, not how a game's outcome is
    scored. ``wins`` is summed straight from the per-game outcomes rather than
    reconstructed as ``rate * games``, which only happens to be exact here
    because every games count below is a multiple of two.

    The frontier pool is validated -- and can raise -- before either set of
    games is played, so a stale or empty frontier fails fast rather than after
    burning a served-agent exam's worth of compute.

    Args:
        candidate: The route under test.
        against: Path to the agent to gate against; the currently served
            agent by default.
        frontier: Directory of frontier league routes, harvested from the
            newest archive by ``build_league``.
        workers: Processes to spread games over, or None for the default.

    Returns:
        The ``Verdict``.

    Raises:
        RuntimeError: If ``frontier`` holds no files, or its newest file is
            older than ``FRONTIER_STALE_DAYS`` days.
    """
    frontier_league = _frontier_league(frontier)

    scores = arena.outcomes(candidate, {"served": against}, GATE_SEEDS, workers)
    games = len(scores)
    wins = sum(scores)
    rate = wins / games
    low, high = wilson_interval(wins, games)

    frontier_scores = arena.outcomes(candidate, frontier_league, GATE_SEEDS, workers)
    frontier_games = len(frontier_scores)
    frontier_wins = sum(frontier_scores)
    frontier_rate = frontier_wins / frontier_games
    frontier_low, _ = wilson_interval(frontier_wins, frontier_games)

    passed = low > 0.5 and frontier_low >= FRONTIER_LOW_THRESHOLD
    return Verdict(
        rate=rate,
        games=games,
        low=low,
        high=high,
        frontier_rate=frontier_rate,
        frontier_low=frontier_low,
        passed=passed,
    )


def _frontier_league(frontier: Path) -> dict[str, Route]:
    """Load every route file in ``frontier`` as a league member.

    Args:
        frontier: Directory of routes harvested by ``build_league``.

    Returns:
        One route per file, keyed by filename stem.

    Raises:
        RuntimeError: If ``frontier`` holds no files, or its newest file is
            older than ``FRONTIER_STALE_DAYS`` days -- a stale frontier is the
            2026-08-20 failure with extra steps, so this fails loudly rather
            than quietly grading a candidate against history.
    """
    files = sorted(path for path in frontier.iterdir() if path.is_file())
    if not files:
        raise RuntimeError(
            f"{frontier}: frontier league is empty -- refresh it with "
            "build_league before gating"
        )
    age_days = (time.time() - max(path.stat().st_mtime for path in files)) / 86400
    if age_days > FRONTIER_STALE_DAYS:
        raise RuntimeError(
            f"{frontier}: newest file is {age_days:.1f} days old "
            f"(> {FRONTIER_STALE_DAYS}) -- refresh it with build_league before gating"
        )
    return {path.stem: load(path) for path in files}


def main() -> None:
    """Gate one candidate route and exit non-zero on FAIL."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path, help="route (JSON) to gate")
    parser.add_argument(
        "--against", type=str, default=SERVED, help="agent path to gate against"
    )
    parser.add_argument(
        "--frontier",
        type=Path,
        default=FRONTIER_DIR,
        help="directory of frontier league routes",
    )
    parser.add_argument("--workers", type=int, default=None)
    arguments = parser.parse_args()

    candidate = load(arguments.candidate)
    verdict = run(candidate, arguments.against, arguments.frontier, arguments.workers)
    LOGGER.info(
        "rate=%.4f [%.4f, %.4f] over %d games vs %s; frontier rate=%.4f (low=%.4f): %s",
        verdict.rate,
        verdict.low,
        verdict.high,
        verdict.games,
        arguments.against,
        verdict.frontier_rate,
        verdict.frontier_low,
        "PASS" if verdict.passed else "FAIL",
    )
    if not verdict.passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
