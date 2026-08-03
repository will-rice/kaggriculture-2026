"""Typed view over the raw Kaggriculture observation dict.

The environment hands the agent a plain mapping every turn. This module wraps it
so policy code reads named fields instead of indexing nested dicts, without
copying the tile grid (a 720-turn episode re-reads it every step).
"""

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from kaggriculture.constants import CROPS

# A tile is None (empty), "LOCKED", or a plant / weed / structure dict.
Tile = Any
Position = tuple[int, int]


@dataclass(frozen=True)
class Observation:
    """One turn of state as seen by a single player."""

    player: int
    step: int
    day: int
    hour: int
    farms: Sequence[Mapping[str, Any]]
    market: Mapping[str, Any]
    town: Mapping[str, Any]
    private: Mapping[str, Any]

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "Observation":
        """Build an ``Observation`` from the environment's raw observation."""
        return cls(
            player=raw["player"],
            step=raw["step"],
            day=raw["day"],
            hour=raw["hour"],
            farms=raw["farms"],
            market=raw["market"],
            town=raw["town"],
            private=raw["private"],
        )

    @property
    def farm(self) -> Mapping[str, Any]:
        """Return this player's public farm state."""
        return self.farms[self.player]

    @property
    def opponent(self) -> Mapping[str, Any]:
        """Return the other player's public farm state."""
        return self.farms[1 - self.player]

    @property
    def money(self) -> float:
        """Return this player's bank balance."""
        return self.farm["money"]

    @property
    def tiles(self) -> Sequence[Sequence[Tile]]:
        """Return this player's tile grid, indexed ``tiles[y][x]``."""
        return self.farm["tiles"]

    @property
    def shed(self) -> Mapping[str, int]:
        """Return the shed contents (harvested produce, animals, fertilizer)."""
        return self.private["shed"]

    @property
    def seeds(self) -> Mapping[str, int]:
        """Return unplanted seed counts, which live outside the shed."""
        return self.private["seeds"]

    @property
    def prices(self) -> Mapping[str, int]:
        """Return the current per-unit market sale price of every product."""
        return self.market["prices"]

    @property
    def units(self) -> list[Position]:
        """Return unit positions, index 0 being the main farmer then each hand."""
        return [tuple(self.farm["farmer"]), *(tuple(p) for p in self.farm["hands"])]

    def inventory(self, unit: int) -> Mapping[str, int]:
        """Return the carried inventory of a unit (0 = main farmer)."""
        inventories = self.private["inventories"]
        return inventories[unit] if unit < len(inventories) else {}

    def tile(self, pos: Position) -> Tile:
        """Return the tile at ``pos``."""
        return self.tiles[pos[1]][pos[0]]

    def open_tiles(self) -> list[Position]:
        """Return unlocked, empty tiles that can be planted or built on."""
        return [
            (x, y)
            for y, row in enumerate(self.tiles)
            for x, tile in enumerate(row)
            if tile is None
        ]

    def unlocked_tile_count(self) -> int:
        """Return how many tiles this player owns."""
        return sum(tile != "LOCKED" for row in self.tiles for tile in row)

    def plant_count(self) -> int:
        """Return how many tiles hold a living crop."""
        return sum(is_plant(tile) for row in self.tiles for tile in row)


def is_plant(tile: Tile) -> bool:
    """Return whether a tile holds a growing crop."""
    return isinstance(tile, dict) and tile.get("kind") == "PLANT"


def is_weed(tile: Tile) -> bool:
    """Return whether a tile holds a weed that must be dug out."""
    return isinstance(tile, dict) and tile.get("kind") == "WEED"


def is_harvestable(tile: Tile, day: int) -> bool:
    """Return whether harvesting this plant now collects its best yield.

    One-time crops gain a unit for every watered day inside their bonus window,
    so they are held until that window's last watering has landed — unless the
    yield cap is already reached, which for melon happens two days before the
    documented window closes, or unless decay has started and waiting only loses
    units.
    """
    if not is_plant(tile) or tile["yield_units"] <= 0:
        return False
    crop = CROPS[tile["crop"]]
    age = day - tile["planted_day"]
    if age < crop["first_yield_day"]:
        return False
    if crop["ongoing"]:
        return True
    if age > crop["max_yield_day"] or tile["yield_units"] >= crop["max_yield"]:
        return True
    return age == crop["max_yield_day"] and tile["watered_today"]
