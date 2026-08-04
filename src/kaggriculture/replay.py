"""Reading a finished episode back for diagnosis.

Every finding worth acting on so far came from a replay rather than from a
score: livestock starving because one carrier cannot walk fourteen pastures,
land bought six days later than the meta build buys it, a season ending with
fifty-two melons still in the shed. A score says a change was worse; a replay
says why.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

Step = list[dict[str, Any]]


class Season(BaseModel):
    """One player's season, reduced to the signals worth looking at.

    Attributes:
        player: Seat this season was summarised for.
        bank: Money at each recorded turn.
        crops: Crop mix keyed by day. Each day's value is the mix as of the
            *last* turn recorded for that day (a near-end-of-day snapshot),
            not a start-of-day reading or a day-long aggregate.
        animals: Herd size at each recorded turn.
        animals_lost: Total animals that vanished from the board over the
            season, counted positionally so a same-turn loss and gain
            elsewhere cannot cancel out.
        final_shed: Shed contents at the final recorded turn.
        final_prices: Market prices at the final recorded turn.
    """

    player: int
    bank: list[float]
    crops: dict[int, dict[str, int]]
    animals: list[int]
    animals_lost: int
    final_shed: dict[str, int]
    final_prices: dict[str, int]


def load(path: Path) -> list[Step]:
    """Return the steps of a replay written by ``EpisodeAgent``."""
    return json.loads(path.read_text())["steps"]


def summarise(steps: list[Step], player: int) -> Season:
    """Return one player's season summary.

    Args:
        steps: Replay steps, outermost index being the turn.
        player: Seat to report on.

    Returns:
        A ``Season`` holding the bank trajectory, the crop mix on each day
        (the mix as of the last turn recorded for that day, not a
        start-of-day or aggregate value), the herd size per turn, how many
        animals vanished, and the closing shed and prices.
    """
    bank: list[float] = []
    crops: dict[int, dict[str, int]] = {}
    animals: list[int] = []
    lost = 0
    previous_positions: set[tuple[int, int]] | None = None
    for step in steps:
        observation = step[0]["observation"]
        farm = observation["farms"][player]
        bank.append(farm["money"])
        crops[observation["day"]] = crop_mix(farm)
        positions = animal_positions(farm)
        if previous_positions is not None:
            lost += len(previous_positions - positions)
        animals.append(len(positions))
        previous_positions = positions

    closing = steps[-1][player]["observation"]
    return Season(
        player=player,
        bank=bank,
        crops=crops,
        animals=animals,
        animals_lost=lost,
        final_shed=dict(closing["private"]["shed"]),
        final_prices=dict(steps[-1][0]["observation"]["market"]["prices"]),
    )


def animal_positions(farm: dict[str, Any]) -> set[tuple[int, int]]:
    """Return the board positions currently holding an animal.

    Tracking positions rather than a headcount is what makes a loss
    detectable even when it is masked, in the same turn, by a placement
    elsewhere on the board: on starvation the engine replaces the tile with a
    bare structure of the same kind, so a position dropping out of this set
    is the signal of an animal leaving a tile, independent of the net herd
    count.

    Args:
        farm: One player's farm observation for a single turn.

    Returns:
        The ``(row, column)`` positions whose tile currently holds an animal.
    """
    return {
        (y, x)
        for y, row in enumerate(farm["tiles"])
        for x, tile in enumerate(row)
        if isinstance(tile, dict) and tile.get("animal")
    }


def crop_mix(farm: dict[str, Any]) -> dict[str, int]:
    """Return how many tiles of the farm hold each crop."""
    mix: dict[str, int] = {}
    for row in farm["tiles"]:
        for tile in row:
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                mix[tile["crop"]] = mix.get(tile["crop"], 0) + 1
    return mix
