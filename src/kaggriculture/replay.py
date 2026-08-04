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
    """One player's season, reduced to the signals worth looking at."""

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
        A ``Season`` holding the bank trajectory, the crop mix on each day, the
        herd size per turn, how many animals vanished, and the closing shed and
        prices.
    """
    bank: list[float] = []
    crops: dict[int, dict[str, int]] = {}
    animals: list[int] = []
    lost = 0
    for step in steps:
        observation = step[0]["observation"]
        farm = observation["farms"][player]
        bank.append(farm["money"])
        crops[observation["day"]] = crop_mix(farm)
        herd = sum(
            1
            for row in farm["tiles"]
            for tile in row
            if isinstance(tile, dict) and tile.get("animal")
        )
        if animals and herd < animals[-1]:
            lost += animals[-1] - herd
        animals.append(herd)

    closing = steps[-1][player]["observation"]
    return Season(
        player=player,
        bank=bank,
        crops=crops,
        animals=animals,
        animals_lost=lost,
        final_shed=dict(closing.get("private", {}).get("shed", {})),
        final_prices=dict(steps[-1][0]["observation"]["market"]["prices"]),
    )


def crop_mix(farm: dict[str, Any]) -> dict[str, int]:
    """Return how many tiles of the farm hold each crop."""
    mix: dict[str, int] = {}
    for row in farm["tiles"]:
        for tile in row:
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                mix[tile["crop"]] = mix.get(tile["crop"], 0) + 1
    return mix
