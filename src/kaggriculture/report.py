"""Turning episode results into the number the ladder actually scores.

The competition ranks by wins, not by coins banked, so every acceptance
decision reads a win rate here. Rates carry a confidence interval because a
sixteen-seed result and a two-hundred-seed result are otherwise indistinguishable
on the page, and the difference decides whether a change ships.
"""

import math
import statistics

from pydantic import BaseModel

from kaggriculture.result import Result


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


class Standing(BaseModel):
    """One agent's record against one opponent."""

    opponent: str
    games: int
    wins: int
    losses: int
    ties: int
    errors: int
    win_rate: float
    low: float
    high: float
    bank: float
    opponent_bank: float

    @property
    def half_width(self) -> float:
        """Return half the confidence interval, which is what gates read."""
        return (self.high - self.low) / 2


def standings(results: list[Result]) -> list[Standing]:
    """Return one ``Standing`` per opponent, in first-seen order.

    Episodes that errored are counted separately rather than as losses: a
    crashed agent and a beaten one need different responses, and folding them
    together hides the crash.
    """
    by_opponent: dict[str, list[Result]] = {}
    for result in results:
        by_opponent.setdefault(result.opponent, []).append(result)

    table: list[Standing] = []
    for opponent, played in by_opponent.items():
        scored = [result for result in played if result.error is None]
        wins = sum(result.score == 1.0 for result in scored)
        ties = sum(result.score == 0.5 for result in scored)
        losses = len(scored) - wins - ties
        points = wins + ties / 2
        low, high = wilson_interval(points, len(scored))
        banked = [result.scores for result in scored if result.scores]
        table.append(
            Standing(
                opponent=opponent,
                games=len(scored),
                wins=wins,
                losses=losses,
                ties=ties,
                errors=len(played) - len(scored),
                win_rate=points / len(scored) if scored else 0.0,
                low=low,
                high=high,
                bank=statistics.fmean(pair[0] for pair in banked) if banked else 0.0,
                opponent_bank=(
                    statistics.fmean(pair[1] for pair in banked) if banked else 0.0
                ),
            )
        )
    return table


def format_standing(standing: Standing) -> str:
    """Return one aligned line describing a standing."""
    return (
        f"vs {standing.opponent:<14s} "
        f"{standing.win_rate:.3f} [{standing.low:.3f}, {standing.high:.3f}]  "
        f"({standing.wins}W {standing.losses}L {standing.ties}T"
        f"{f' {standing.errors}E' if standing.errors else ''})"
        f"  bank {standing.bank:8.0f} vs {standing.opponent_bank:8.0f}"
    )


def league_win_rate(standings: list[Standing]) -> float:
    """Return the win rate across every opponent, weighting each equally.

    The acceptance rule in STRATEGY.md is stated against this number, so it has
    to exist somewhere a person running the evaluation can see. It previously
    lived only in the wandb payload, which meant the rule could not be checked
    without ``--track`` and an internet connection.

    Each opponent counts once regardless of how many games it played. That
    normally coincides with a games-weighted mean, because the harness gives
    every opponent the same seeds; the two diverge only when opponents error at
    different rates.

    Args:
        standings: Per-opponent records from one evaluation.

    Returns:
        The mean of the per-opponent win rates, or zero if there are none.
    """
    return statistics.fmean(s.win_rate for s in standings) if standings else 0.0


def format_league(standings: list[Standing]) -> str:
    """Return the one line the acceptance rule is actually read from."""
    played = sum(standing.games for standing in standings)
    return (
        f"   league {league_win_rate(standings):.3f} "
        f"over {len(standings)} opponents, {played} games"
    )
