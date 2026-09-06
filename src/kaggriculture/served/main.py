"""The floor: our own agent, as far as the campaign has taken it.

This first version is deliberately minimal and entirely ours: on the first
turn it buys enough wheat seed for the whole season, then plants, waters and
harvests one tile in a loop and sells whatever lands in the shed. It never
moves the farmer, because the farmer starts each day on a shed-access tile, so
a single-tile loop never needs to walk back to drop or sell. The campaign
replaces this file; nothing here is a strategy.

One file ships, so nothing here is imported from the repository: the two crop
numbers it needs are written out below rather than read from
`kaggriculture.constants`, which does not travel with a submission.
"""

from collections.abc import Mapping
from typing import Any

# BUY_SEED WHEAT costs 10 each, and the whole season is bought on turn zero
# because a market order is resolved the turn it is issued.
SEED_STOCK = 30
# WHEAT's `max_yield_day`: a plant stops adding units four days after it was
# planted, so harvesting before then throws the rest of the crop away.
MAX_YIELD_DAY = 4


def agent(
    observation: Mapping[str, Any], configuration: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Take one turn: buy seed once, then run one tile through the crop cycle.

    Args:
        observation: The environment's raw observation for this player.
        configuration: The episode configuration, which this program ignores.

    Returns:
        The action dict the environment consumes: one farmer op, no hired
        hands, and the market orders for this turn.
    """
    farm = observation["farms"][observation["player"]]
    private = observation["private"]
    x, y = farm["farmer"]
    tile = farm["tiles"][y][x]
    growing = isinstance(tile, dict) and tile.get("kind") == "PLANT"

    market: list[list[Any]] = []
    if observation["step"] == 0:
        market.append(["BUY_SEED", "WHEAT", SEED_STOCK])
    shed_wheat = private["shed"].get("WHEAT", 0)
    if shed_wheat > 0:
        # SELL sells out of the shed, not out of the farmer's inventory.
        market.append(["SELL", "WHEAT", shed_wheat])

    if tile is None and private["seeds"].get("WHEAT", 0) > 0:
        # The tile is empty because WHEAT is not `ongoing`, so a HARVEST
        # clears it back to None and the cycle starts again here.
        farmer = ["PLANT", "WHEAT"]
    elif growing and not tile["watered_today"]:
        # A plant left unwatered two days running turns into a WEED, so it is
        # watered every day, not only inside the yield-bonus window.
        farmer = ["WATER"]
    elif (
        growing
        and tile["yield_units"] > 0
        and observation["day"] - tile["planted_day"] >= MAX_YIELD_DAY
    ):
        # HARVEST puts the yield in the farmer's inventory; the end of the day
        # drops that inventory into the shed and resets the farmer back to
        # this same shed-access tile, so no DROP and no walking is needed.
        farmer = ["HARVEST"]
    else:
        farmer = ["PASS"]
    return {"farmer": farmer, "hands": [], "market": market}
