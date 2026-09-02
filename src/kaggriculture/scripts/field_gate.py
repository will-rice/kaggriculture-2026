"""Score a candidate against the lineages that actually populate the ladder.

Gating against the agent we serve measures the wrong thing. Two market
thresholds tuned that way won 127 of 128 mirror games and moved the ladder not
at all, because the mirror is one lineage out of a field of two dozen.

The public meta census keys a seat by the hash of its opening 24 actions, and
playing our own agents through that hash identified the lineage behind each of
the top signatures. This plays a candidate against the ones we hold.

Weights are deliberately equal by default. Occupancy is real -- the census
shows a kernel taking half the top band and collapsing within two days -- but
it is far too volatile to weight with: across three consecutive archives one
lineage read 38.3%, 7.1% and 15.2%, and ours read 7.2%, 28.9% and 14.5%.
Weighting by any single day is false precision, and the ranking it produces is
the same one equal weights produce, so equal weights are reported as the
number and occupancy is available only when a specific day's field is the
question being asked.
"""

import argparse
import json
import logging
import time
from pathlib import Path

LOGGER = logging.getLogger(__name__)

# Lineage -> the agent of ours that reproduces its opening signature, with the
# share it held on 2026-09-01. The shares are recorded for provenance; they are
# not used unless --occupancy is passed.
FIELD: dict[str, tuple[str, float]] = {
    "router2929": ("/data/kaggriculture/opponents/yhay81_router2929/main.py", 18.4),
    "v54": ("src/kaggriculture/kaito_v54_policy.py", 15.2),
    "v56": ("src/kaggriculture/kaito_v56_policy.py", 14.5),
    "shopforge": ("/data/kaggriculture/opponents/tetsutani_shopforge/main.py", 8.6),
    "indarkarhana": ("/data/kaggriculture/opponents/indarkarhana_top10/main.py", 5.1),
}


def main() -> None:
    """Play one candidate against every lineage and report both means."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path, help="agent file to score")
    # All 64 exam seeds, because 24 is not enough to act on. At 24 seeds (48
    # games) a per-lineage rate carries a Wilson interval about +/-0.13 wide,
    # and that cost us: this agent measured 0.562 against indarkarhana on 24
    # seeds and 0.719 on 64, so a gap twice the real one got chased. The seeds
    # are exam seeds and measurement is what they are for.
    #
    # Reusing the same 64 for every candidate does mean a screened winner is
    # somewhat seed-lucky. That is tolerable while we are choosing between a
    # handful of agents and would not be if we were sweeping hundreds.
    parser.add_argument("--seeds", type=int, default=64, help="exam seeds per opponent")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="lineage to leave out; pass the same set for every candidate so "
        "their means are comparable",
    )
    parser.add_argument(
        "--occupancy",
        action="store_true",
        help="also report the 2026-09-01 occupancy-weighted mean, which is a "
        "snapshot of one volatile day and not the headline number",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    report(args.candidate, args.seeds, args.workers, args.occupancy, set(args.exclude))


def report(
    candidate: Path,
    seeds: int,
    workers: int,
    occupancy: bool,
    exclude: set[str],
) -> None:
    """Score the candidate and log every per-lineage rate behind the mean.

    Args:
        candidate: The agent file to score.
        seeds: Exam seeds per opponent; each is played in both seat orderings.
        workers: Arena processes.
        occupancy: Whether to also report the occupancy-weighted mean.
        exclude: Lineages to leave out. A candidate is always excluded from
            its own field, which makes two candidates' means cover different
            opponents; pass the union of both candidates' lineages to compare
            them on the same set.
    """
    from kaggriculture.report import wilson_interval
    from kaggriculture.search import arena
    from kaggriculture.search.scripts import holdout

    exam = holdout.GATE_SEEDS[:seeds]
    rates: dict[str, float] = {}
    games = 0
    started = time.perf_counter()
    for lineage, (path, _share) in FIELD.items():
        if lineage in exclude or Path(path).resolve() == candidate.resolve():
            continue
        scores = arena.outcomes(str(candidate), {lineage: path}, exam, workers)
        rates[lineage] = sum(scores) / len(scores)
        games += len(scores)
        LOGGER.info(
            "   vs %-13s %.4f over %d games", lineage, rates[lineage], len(scores)
        )
    equal = sum(rates.values()) / len(rates)
    wins = sum(rates.values()) * (games / len(rates))
    low, high = wilson_interval(wins, games)
    LOGGER.info(
        "%s: FIELD %.4f over %d lineages, %d games, Wilson [%.4f, %.4f] (%.0fs)",
        candidate.name,
        equal,
        len(rates),
        games,
        low,
        high,
        time.perf_counter() - started,
    )
    if occupancy:
        weighted = sum(rates[k] * FIELD[k][1] for k in rates)
        total = sum(FIELD[k][1] for k in rates)
        LOGGER.info(
            "   occupancy-weighted on 2026-09-01: %.4f  (one day; shares moved "
            "3-5x across the three archives around it)",
            weighted / total,
        )
    summary = {
        "field": round(equal, 4),
        "per_lineage": {k: round(v, 4) for k, v in rates.items()},
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
