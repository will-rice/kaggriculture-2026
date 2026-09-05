"""Turning episode results into the number the ladder actually scores.

The competition ranks by wins, not by coins banked, so every acceptance
decision reads a win rate here. Rates carry a confidence interval because a
sixteen-seed result and a two-hundred-seed result are otherwise indistinguishable
on the page, and the difference decides whether a change ships.
"""

import math


def wilson_interval(wins: float, games: int, z: float = 1.96) -> tuple[float, float]:
    """Return a Wilson score interval for a win rate.

    The normal approximation collapses to zero width at zero wins, which is
    exactly where this agent has spent most of its life, so it would report
    certainty precisely when there is none. Wilson stays finite at both ends.

    Args:
        wins: Wins, counting a tie as half.
        games: Games played, excluding episodes that errored.
        z: Standard-normal quantile; 1.96 gives 95%.

    Returns:
        Lower and upper bounds, each within [0, 1].
    """
    if games <= 0:
        return 0.0, 1.0
    rate = wins / games
    denominator = 1 + z**2 / games
    centre = (rate + z**2 / (2 * games)) / denominator
    spread = z * math.sqrt(rate * (1 - rate) / games + z**2 / (4 * games**2))
    spread /= denominator
    return max(0.0, centre - spread), min(1.0, centre + spread)
