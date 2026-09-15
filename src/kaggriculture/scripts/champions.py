"""Play this campaign's champions against each other and rate the lineage.

The gate's pool is not a fixed yardstick. It grew from 50 opponents to 100 in a
week and then lost 36 in a night, and every one of those changes moves what a
"field win rate" means -- which is what froze promotions for eight hours on
2026-09-14, when candidates were compared against a champion measured on a pool
that no longer existed.

The lineage does not move. Fourteen files that never change, played against
each other, answer the question the pool cannot: is each champion actually
stronger than the one it replaced, or does the ratchet only hold against a
field that was drifting underneath it? The games are constants -- fixed files,
fixed seeds, no agent in the roster draws on randomness -- so this is worth
running once and keeping.

It is a diagnostic and not a gate. A campaign promoted on its own lineage
becomes that lineage playing itself, which this one already did once: 69
champions against 12 published agents. `harvest` exists to stop it.

    uv run champions --seeds 16
"""

import argparse
import json
import logging
import math
import random
import re
from pathlib import Path

from kaggriculture.campaign import arena, config, rating

LOGGER = logging.getLogger(__name__)

# `rating.standings` returns log-odds, negative for anything below the field's
# mean, and Elo is that same scale read in points: 400 per factor of ten. The
# conversion is a multiplication and not a logarithm -- taking one of a
# negative rating is what crashed the first two runs of this.
ELO_PER_LOG_ODDS = 400 / math.log(10)

# The draw the panel always uses, so two runs of this are comparable.
PANEL_SEED = 0


def main() -> None:
    """Round-robin the champions and print the lineage's ratings."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--champions",
        type=Path,
        default=config.RUN / "champions",
        help="where the champion files are (default: %(default)s)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=16,
        help="seasons per pairing (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=config.RUN / "champions-panel.json",
        help="where the pairings are written (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=config.CORE_BUDGET,
        help="worker processes (default: %(default)s)",
    )
    arguments = parser.parse_args()

    lineage = champions(arguments.champions)
    if len(lineage) < 2:
        raise SystemExit(f"need two champions to compare, found {len(lineage)}")
    # A fixed draw rather than the first N: seeds 1..16 are a corner of the
    # range and the gate samples from all of it. Fixed so the panel can be
    # re-run against a later champion and compared to this one.
    seeds = random.Random(PANEL_SEED).sample(config.GATE_SEED_RANGE, arguments.seeds)
    pairs = len(lineage) * (len(lineage) - 1) // 2
    LOGGER.info(
        "%d champions, %d pairings, %d seeds, both seats: %d games",
        len(lineage),
        pairs,
        len(seeds),
        pairs * len(seeds) * 2,
    )

    results = play(lineage, seeds, arguments.workers)
    # Written before anything is fitted. The first run of this played all 2,912
    # games and then lost them to a `math domain error` in the report.
    arguments.out.write_text(
        json.dumps({"seeds": seeds, "pairings": results}, indent=2) + "\n"
    )
    LOGGER.info("wrote %s", arguments.out)
    report(lineage, results)


def champions(home: Path) -> dict[str, str]:
    """Every champion file, in the order they were promoted."""
    found = {}
    for path in sorted(home.glob("champion_*.py")):
        number = re.search(r"champion_(\d+)", path.name)
        if number:
            found[int(number.group(1))] = str(path.resolve())
    return {f"champion_{n}": path for n, path in sorted(found.items())}


def play(
    lineage: dict[str, str], seeds: list[int], workers: int
) -> list[tuple[str, str, float, int]]:
    """Every unordered pairing once, both seats.

    `arena.outcomes` plays its candidate in both seats already, so a pairing
    asked for from both sides would be played twice and tell us nothing more.

    Args:
        lineage: Champion name to file.
        seeds: The seasons each pairing is played on.
        workers: Worker processes.

    Returns:
        One ``(one, two, rate, games)`` per pairing, for `rating.standings`.
    """
    results = []
    names = list(lineage)
    for index, name in enumerate(names[:-1]):
        later = {other: lineage[other] for other in names[index + 1 :]}
        scores = arena.outcomes(lineage[name], later, seeds, workers)
        summary = arena.summarize(scores, later)
        for other in later:
            games = sum(1 for member in scores.members if member == other)
            results.append((name, other, summary[other], games))
        LOGGER.info("%s: %d pairings played", name, len(later))
    return results


def report(lineage: dict[str, str], results: list[tuple[str, str, float, int]]) -> None:
    """Print each champion's rating, and how it fares against its predecessor."""
    fitted = rating.standings(results)
    against = {(one, two): rate for one, two, rate, _ in results}

    LOGGER.info(
        "\n%-14s%9s%9s%12s  %s",
        "champion",
        "elo",
        "vs prev",
        "vs first",
        "mean vs the rest",
    )
    names = list(lineage)
    for index, name in enumerate(names):
        elo = ELO_PER_LOG_ODDS * fitted[name]
        previous = _rate(against, name, names[index - 1]) if index else None
        first = _rate(against, name, names[0]) if index else None
        others = [_rate(against, name, other) for other in names if other != name]
        mean = sum(v for v in others if v is not None) / max(1, len(others))
        LOGGER.info(
            "%-14s%9.0f%9s%12s  %.3f",
            name,
            elo,
            "-" if previous is None else f"{previous:.3f}",
            "-" if first is None else f"{first:.3f}",
            mean,
        )

    ordered = sorted(fitted, key=lambda n: fitted[n], reverse=True)
    LOGGER.info("\nstrongest: %s", ", ".join(ordered[:3]))
    LOGGER.info("weakest:   %s", ", ".join(ordered[-3:]))


def _rate(against: dict[tuple[str, str], float], one: str, two: str) -> float | None:
    """``one``'s win rate against ``two``, whichever way it was recorded."""
    if (one, two) in against:
        return against[(one, two)]
    if (two, one) in against:
        return 1.0 - against[(two, one)]
    return None
