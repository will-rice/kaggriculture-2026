"""Per-opponent win rates and margins on a fixed seed block, both seats.

The exam block (``config.EXAM_SEEDS``) is the one this is meant for; the
harness refuses it, so this module plays through the engine directly.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

from kaggriculture.campaign import harness

LOGGER = logging.getLogger(__name__)


def score_field(
    candidate: Path, seeds: Sequence[int], workers: int, opponents: Sequence[str]
) -> tuple[dict[str, float], dict[str, harness.Margin]]:
    """Win rate and bank margin per opponent over ``2 * len(seeds)`` games each.

    Args:
        candidate: The program to play.
        seeds: The seed block to play it on.
        workers: Processes to fan the games over.
        opponents: Roster names to play.

    Returns:
        The win rate per opponent, ties as half, and the bank margin per
        opponent. The rate says how often it won; the margin says by how much,
        which is the difference between an opponent nearly beaten and one that
        is out of reach.
    """
    games = harness.play_unsealed(candidate, opponents, seeds, workers)
    names = list(opponents)
    rates: dict[str, float] = {}
    for name in names:
        mine = [g for g in games if g.opponent == name]
        rates[name] = sum(
            1.0 if g.ours > g.theirs else 0.5 if g.ours == g.theirs else 0.0
            for g in mine
        ) / len(mine)
        LOGGER.info("%s: %.4f over %d games", name, rates[name], len(mine))
    return rates, harness.margins(games, names)
