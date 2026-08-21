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


@dataclass(frozen=True)
class Verdict:
    """The exam's result: a win rate, its Wilson interval, and the ruling.

    Attributes:
        rate: Win rate over ``games``, ties counted as half a win.
        games: Games played -- ``2 * len(GATE_SEEDS)``.
        low: Wilson lower bound at 95%.
        high: Wilson upper bound at 95%.
        passed: True iff ``low`` exceeds 0.5 -- the interval must clear 0.5
            entirely, not merely straddle it in the candidate's favour.
    """

    rate: float
    games: int
    low: float
    high: float
    passed: bool


def run(candidate: Route, against: str = SERVED, workers: int | None = None) -> Verdict:
    """Play the exam: ``candidate`` vs. ``against``, over ``GATE_SEEDS``, seat-swapped.

    Scoring is delegated entirely to ``arena.outcomes`` -- this module fixes
    which seeds and which opponent the exam uses, not how a game's outcome is
    scored. ``wins`` is summed straight from the per-game outcomes rather than
    reconstructed as ``rate * games``, which only happens to be exact here
    because 128 is a power of two.

    Args:
        candidate: The route under test.
        against: Path to the agent to gate against; the currently served
            agent by default.
        workers: Processes to spread games over, or None for the default.

    Returns:
        The ``Verdict``.
    """
    scores = arena.outcomes(candidate, {"served": against}, GATE_SEEDS, workers)
    games = len(scores)
    wins = sum(scores)
    rate = wins / games
    low, high = wilson_interval(wins, games)
    return Verdict(rate=rate, games=games, low=low, high=high, passed=low > 0.5)


def main() -> None:
    """Gate one candidate route and exit non-zero on FAIL."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path, help="route (JSON) to gate")
    parser.add_argument(
        "--against", type=str, default=SERVED, help="agent path to gate against"
    )
    parser.add_argument("--workers", type=int, default=None)
    arguments = parser.parse_args()

    candidate = load(arguments.candidate)
    verdict = run(candidate, arguments.against, arguments.workers)
    LOGGER.info(
        "rate=%.4f [%.4f, %.4f] over %d games vs %s: %s",
        verdict.rate,
        verdict.low,
        verdict.high,
        verdict.games,
        arguments.against,
        "PASS" if verdict.passed else "FAIL",
    )
    if not verdict.passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
