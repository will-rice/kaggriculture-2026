"""Hill-climb a route against a league, one edit at a time.

Accepts a candidate only when its mean league win rate beats the incumbent's by
more than the standard error of the comparison, so a run does not walk uphill on
noise. Every accepted route is written out, because the interesting artefact is
the sequence of edits that paid, not only the final tape.
"""

import argparse
import json
import logging
import random
from pathlib import Path

from kaggriculture.search.arena import evaluate
from kaggriculture.search.mutate import mutate
from kaggriculture.search.route import load, save

LOGGER = logging.getLogger(__name__)
SEEDS = tuple(range(500_000, 500_064))


def main() -> None:
    """Run the hill-climb until the candidate budget is exhausted."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed_route", type=Path, help="route to start from")
    parser.add_argument("league", type=Path, help="directory of opponent routes")
    parser.add_argument("output", type=Path, help="directory for accepted routes")
    parser.add_argument("--candidates", type=int, default=200)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--rng", type=int, default=0)
    arguments = parser.parse_args()

    league = {path.stem: load(path) for path in sorted(arguments.league.glob("*.json"))}
    if not league:
        raise SystemExit(f"no opponent routes in {arguments.league}")
    incumbent = load(arguments.seed_route)
    arguments.output.mkdir(parents=True, exist_ok=True)

    scores = evaluate(incumbent, league, SEEDS, arguments.workers)
    best = sum(scores.values()) / len(scores)
    LOGGER.info("incumbent %.4f %s", best, json.dumps(scores))

    rng = random.Random(arguments.rng)
    for candidate in range(arguments.candidates):
        mutated, description = mutate(incumbent, rng)
        scores = evaluate(mutated, league, SEEDS, arguments.workers)
        mean = sum(scores.values()) / len(scores)
        # The standard error of a win rate over this many games, doubled because
        # incumbent and candidate are both estimates.
        games = 2 * len(SEEDS) * len(league)
        threshold = 2 * (0.25 / games) ** 0.5
        if mean > best + threshold:
            incumbent, best = mutated, mean
            save(incumbent, arguments.output / f"accepted-{candidate:04d}.json")
            LOGGER.info("accepted %.4f (%s) %s", mean, description, json.dumps(scores))
        else:
            LOGGER.info("rejected %.4f (%s)", mean, description)


if __name__ == "__main__":
    main()
