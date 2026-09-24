"""Scan every measure of our own games for the one that predicts a loss.

`evidence` does this for the ladder: one query per day over the recorded corpus,
asking which quantities separate the stronger side from the weaker. Nothing did
it for the games this campaign plays itself, so the champion's failures have been
read by eye -- and eye-reading produced a hypothesis on 2026-09-16 that two games
supported and seventy-two refuted, the shed being identically full at 19.8 in the
games we win and 19.0 in the ones we lose.

So this asks all of them at once. Every measure a `Day` carries, on every day of
the season, correlated with the final margin across a batch of real games, with
the count behind each number and no interpretation. What it returns is a list of
candidates ordered by how much they separate, which is a place to start looking
rather than an answer: a correlation over sixty games is worth exactly one
experiment, and the experiment is what settles it.

    uv run where-we-lose --seeds 24 --against ours_12 --against ours_16
"""

import argparse
import json
import logging
import statistics
from collections.abc import Sequence
from pathlib import Path

from kaggriculture.campaign import config, harness

LOGGER = logging.getLogger(__name__)

# The per-day measures a `Day` carries as plain numbers. The dict-valued ones --
# plants, shed, seeds, animals -- are summed, because "how much" is the question
# and which crop is a finer one than a first scan should ask.
SCALARS = ("ours_bank", "ours_quadrants", "ours_hands", "ours_weeds", "ours_fertilised")
SUMMED = ("ours_plants", "ours_shed", "ours_seeds", "ours_animals")
THEIRS = ("theirs_bank", "theirs_quadrants", "theirs_plants", "theirs_shed")


def main() -> None:
    """Play a batch and report which measure separates the wins from the losses."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--against",
        action="append",
        default=None,
        help="pool opponent to play; repeatable (default: the champion's worst four)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=24,
        help="seasons per opponent (default: %(default)s)",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=14,
        help="candidates to print (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=config.RUN / "where-we-lose.json",
        help="where the full scan is written (default: %(default)s)",
    )
    arguments = parser.parse_args()

    against = arguments.against or worst(4)
    seeds = list(range(1, arguments.seeds + 1))
    LOGGER.info("playing %d seasons against %s", len(seeds), ", ".join(against))
    games = harness.play(
        config.LIVE.floor / "main.py",
        against,
        seeds,
        config.CORE_BUDGET,
        days=True,
        pool=config.POOL,
    )
    scored = [game for game in games if game.error is None and game.days]
    won = sum(1 for game in scored if game.ours > game.theirs)
    LOGGER.info("%d games scored, %d won, %d lost", len(scored), won, len(scored) - won)

    found = scan(scored)
    arguments.out.write_text(json.dumps(found, indent=2) + "\n", encoding="utf-8")
    report(found, arguments.show)
    LOGGER.info("wrote %s", arguments.out)


def worst(count: int) -> list[str]:
    """The pool names the champion scored lowest against at its last gate."""
    champion = json.loads((config.RUN / "champion.json").read_text(encoding="utf-8"))
    rates = champion["result"]["rates"]
    return [name for name, _ in sorted(rates.items(), key=lambda kv: kv[1])[:count]]


def scan(games: Sequence[harness.Game]) -> list[dict]:
    """Correlate every (measure, day) with the final margin.

    Args:
        games: Played games carrying day tables.

    Returns:
        One record per measure and day that had enough spread to correlate,
        strongest separation first.
    """
    margins = [game.ours - game.theirs for game in games]
    if len(margins) < 4:
        return []
    found = []
    days = sorted({row.day for game in games for row in game.days})
    for day in days:
        for measure in (*SCALARS, *SUMMED, *THEIRS):
            values = [reading(game, day, measure) for game in games]
            if any(value is None for value in values):
                continue
            numbers = [value for value in values if value is not None]
            if len(set(numbers)) < 3:
                continue
            found.append(
                {
                    "day": day,
                    "measure": measure,
                    "correlation": statistics.correlation(numbers, margins),
                    "mean": statistics.fmean(numbers),
                    "games": len(numbers),
                }
            )
    return sorted(found, key=lambda row: -abs(row["correlation"]))


def reading(game: harness.Game, day: int, measure: str) -> float | None:
    """One measure of one day of one game, summed if it is a per-item dict."""
    for row in game.days:
        if row.day != day:
            continue
        value = getattr(row, measure, None)
        if isinstance(value, dict):
            return float(sum(value.values()))
        return None if value is None else float(value)
    return None


def report(found: Sequence[dict], show: int) -> None:
    """Print the strongest separations, and say what they are not."""
    if not found:
        LOGGER.info("nothing had enough spread to correlate")
        return
    LOGGER.info(
        "\n%-18s%5s%13s%12s%8s", "measure", "day", "correlation", "mean", "games"
    )
    for row in found[:show]:
        LOGGER.info(
            "%-18s%5d%13.3f%12.1f%8d",
            row["measure"],
            row["day"],
            row["correlation"],
            row["mean"],
            row["games"],
        )
    LOGGER.info(
        "\nA correlation here is a candidate, not a finding: bank on a late day "
        "correlates with the margin because it is most of the margin, and the "
        "shed looked like a mechanism over two games and was identical over "
        "seventy-two. What settles one is an edit and a paired measurement."
    )
