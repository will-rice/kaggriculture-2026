"""Hill-climb a route against a league, one edit at a time.

Accepts a candidate only when it beats the incumbent by more than the
standard error of a *paired* comparison, so a run does not walk uphill on
noise. Every accepted route is written out, because the interesting artefact
is the sequence of edits that paid, not only the final tape.

The comparison is paired, and its standard error is taken from the observed
data rather than assumed, to fix two flaws an earlier version of this script
had:

1. Scoring the candidate and the incumbent on two different seed blocks means
   the comparison also measures which block of boards each one happened to
   draw. That board-to-board offset is a single fixed draw (~2.8% at these
   sizes, against what was a ~3.95% threshold) shared by every later
   comparison in the run -- an unlucky block accepts nearly everything that
   follows, a lucky one rejects nearly everything. Playing the candidate and
   the incumbent on the *same* seed block each iteration, then differencing
   game by game, cancels that offset: whatever a given board is worth, it is
   worth the same amount to both routes, so only the difference between them
   survives the subtraction.
2. A win-rate standard error built from ``2 * sqrt(0.25 / games)`` assumes
   every game is an independent 0/1 draw. They are not: both seat orderings of
   one seed replay the same board, and every league member sees that same
   board too, so the effective number of independent draws is nearer to
   ``len(seeds)`` than to ``2 * len(seeds) * len(league)`` -- at these league
   sizes the assumed formula understates the true standard error by roughly a
   factor of three. Taking the standard error from the spread of the *observed*
   per-game differences (``stdev(diffs) / sqrt(len(diffs))``) instead measures
   whatever correlation is actually present, rather than assuming it away.

The seed block itself rotates every iteration -- fitting the search to one
fixed set of boards would let a candidate specialise on that set rather than
on the league it is meant to beat -- and is drawn from a stream seeded from
``--rng``, so a run started with the same ``--rng`` reproduces the same
sequence of blocks.
"""

import argparse
import json
import logging
import math
import random
import statistics
from collections.abc import Sequence
from pathlib import Path

from kaggriculture.search.arena import Opponent, outcomes, summarize
from kaggriculture.search.mutate import mutate
from kaggriculture.search.route import load, save

LOGGER = logging.getLogger(__name__)
# The pool a rotating seed block is drawn from, and how many seeds make up
# one block. 64 matches the block size the fixed-seed version of this script
# used.
SEED_POOL = range(500_000, 600_000)
BLOCK_SIZE = 64


def paired_difference(
    candidate_outcomes: Sequence[float], incumbent_outcomes: Sequence[float]
) -> tuple[float, float]:
    """Return the mean and standard error of a game-by-game paired difference.

    Both sequences must come from the same league and seed block, in the same
    order -- see ``arena.outcomes`` -- so that index ``i`` in each is the
    literal same board and seat ordering. That pairing is what lets a board's
    own variance cancel out of the difference instead of adding noise to it.

    Args:
        candidate_outcomes: The candidate's per-game outcomes.
        incumbent_outcomes: The incumbent's per-game outcomes, same order.

    Returns:
        ``(mean_difference, standard_error)``. The standard error comes from
        the observed spread of the per-game differences, not an assumed
        variance, and is ``0.0`` when fewer than two games were played (no
        spread to observe) -- which only arises with a degenerate, empty
        league or seed block.
    """
    diffs = [c - i for c, i in zip(candidate_outcomes, incumbent_outcomes, strict=True)]
    mean_difference = statistics.mean(diffs)
    standard_error = (
        statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
    )
    return mean_difference, standard_error


def load_league(directory: Path) -> dict[str, Opponent]:
    """Return every opponent in ``directory``: routes and policies alike.

    A ``.json`` file is a harvested route, loaded and replayed by our own
    closure; a ``.py`` file is an agent path, passed through unopened for the
    engine to run itself. ``arena.Opponent`` supports both, so restricting the
    league directory to ``*.json`` would make policy opponents unreachable
    from this script even though the arena can already play them -- a league
    of tapes alone cannot contain an opponent that reacts to the candidate.

    Args:
        directory: A directory of ``.json`` routes and/or ``.py`` agents.

    Returns:
        Opponents keyed by file stem, routes and agent paths together.

    Raises:
        ValueError: If a ``.json`` route and a ``.py`` agent share a stem --
            keying by stem means the second glob would otherwise silently
            overwrite the first opponent rather than adding a second one.
    """
    league: dict[str, Opponent] = {}
    for path in sorted(directory.glob("*.json")):
        league[path.stem] = load(path)
    for path in sorted(directory.glob("*.py")):
        if path.stem in league:
            raise ValueError(
                f"league stem collision: {path.stem!r} names both a .json "
                "route and a .py agent"
            )
        league[path.stem] = str(path)
    return league


def main() -> None:
    """Run the hill-climb until the candidate budget is exhausted."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed_route", type=Path, help="route to start from")
    parser.add_argument(
        "league",
        type=Path,
        help="directory of opponent routes (.json) and/or agents (.py)",
    )
    parser.add_argument("output", type=Path, help="directory for accepted routes")
    parser.add_argument("--candidates", type=int, default=200)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--rng", type=int, default=0)
    arguments = parser.parse_args()

    league = load_league(arguments.league)
    if not league:
        raise SystemExit(f"no opponents in {arguments.league}")
    incumbent = load(arguments.seed_route)
    arguments.output.mkdir(parents=True, exist_ok=True)

    # Two independent streams derived from one seed: which edit gets tried and
    # which seed block it is judged on must not share a stream, or the block
    # a candidate is judged on would depend on how many random draws its own
    # mutation happened to consume. Both streams are still fully determined by
    # `--rng`, so a run is reproducible end to end.
    master = random.Random(arguments.rng)
    rng = random.Random(master.getrandbits(64))
    block_rng = random.Random(master.getrandbits(64))

    for candidate in range(arguments.candidates):
        mutated, description = mutate(incumbent, rng)
        seeds = tuple(block_rng.sample(SEED_POOL, BLOCK_SIZE))

        candidate_outcomes = outcomes(mutated, league, seeds, arguments.workers)
        incumbent_outcomes = outcomes(incumbent, league, seeds, arguments.workers)
        mean_difference, standard_error = paired_difference(
            candidate_outcomes, incumbent_outcomes
        )

        if mean_difference > 2 * standard_error:
            incumbent = mutated
            save(incumbent, arguments.output / f"accepted-{candidate:04d}.json")
            LOGGER.info(
                "accepted diff=%.4f se=%.4f (%s) candidate=%s incumbent=%s",
                mean_difference,
                standard_error,
                description,
                json.dumps(summarize(candidate_outcomes, league, seeds)),
                json.dumps(summarize(incumbent_outcomes, league, seeds)),
            )
        else:
            LOGGER.info(
                "rejected diff=%.4f se=%.4f (%s)",
                mean_difference,
                standard_error,
                description,
            )


if __name__ == "__main__":
    main()
