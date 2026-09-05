"""The floor: our own agent, as far as the campaign has taken it.

This first version is deliberately minimal and entirely ours: on the first
turn it buys enough wheat seed for the whole season, then plants, waters and
harvests one tile in a loop and sells whatever lands in the shed. It never
moves the farmer: `_default_spawn` starts (and `_end_of_day` resets) the
farmer on a shed-access tile, so a single-tile loop never needs to walk back
to drop or sell. The campaign replaces this file; nothing here is a strategy.
"""

from collections.abc import Mapping
from typing import Any

from kaggriculture.actions import PASS, Turn
from kaggriculture.constants import CROPS
from kaggriculture.observation import Observation, is_plant

WHEAT = CROPS["WHEAT"]
SEED_STOCK = 30  # BUY_SEED WHEAT costs 10 each; enough for the whole season.


def agent(
    observation: Mapping[str, Any], configuration: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """One turn: buy seed once, then run one tile through plant/water/harvest."""
    view = Observation.parse(observation)
    farm = view.farm
    turn = Turn()
    x, y = farm["farmer"]
    tile = farm["tiles"][y][x]
    shed_wheat = view.shed.get("WHEAT", 0)

    if view.step == 0:
        # BUY_SEED is a market order resolved the same turn it is issued.
        turn.market.append(["BUY_SEED", "WHEAT", SEED_STOCK])
    if shed_wheat > 0:
        # SELL sells out of the shed, not the farmer's inventory.
        turn.market.append(["SELL", "WHEAT", shed_wheat])

    if tile is None and view.seeds.get("WHEAT", 0) > 0:
        # PLANT needs private["seeds"]["WHEAT"] > 0; the tile is empty because
        # WHEAT is not `ongoing`, so HARVEST clears it back to None.
        turn.farmer = ["PLANT", "WHEAT"]
    elif is_plant(tile) and not tile["watered_today"]:
        # A plant left unwatered two days running turns into a WEED, so it is
        # watered every day, not only inside the yield-bonus window.
        turn.farmer = ["WATER"]
    elif (
        is_plant(tile)
        and tile["yield_units"] > 0
        and view.day - tile["planted_day"] >= WHEAT["max_yield_day"]
    ):
        # HARVEST puts the yield in the farmer's inventory; `_end_of_day`
        # auto-drops that inventory into the shed and resets the farmer back
        # to this same shed-access tile, so no DROP is needed here.
        turn.farmer = ["HARVEST"]
    else:
        turn.farmer = list(PASS)
    return turn.to_action()
