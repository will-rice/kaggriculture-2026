"""Game constants and derived geometry for Kaggriculture.

Every rules table is imported from the installed ``kaggle_environments`` package
rather than copied, so the agent can never drift from the rules it is scored
against. Kaggle runs submissions inside that same package, so the import is
available both locally and on the competition runner.
"""

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    ANIMALS,
    CROPS,
    LAND_ORDER,
    LAND_PRICES,
    MARKET_PARAMS,
    PRODUCTS,
    SHOPS,
    market_price,
)

__all__ = [
    "ENVIRONMENT",
    "BASELINE_AGENTS",
    "ANIMALS",
    "CROPS",
    "LAND_ORDER",
    "LAND_PRICES",
    "MARKET_PARAMS",
    "PRODUCTS",
    "SHOPS",
    "market_price",
    "BOARD_SIZE",
    "TURNS_PER_DAY",
    "EPISODE_STEPS",
    "SEASON_DAYS",
    "LAST_DAY",
    "STARTING_MONEY",
    "SHED_CAPACITY",
    "MAX_MARKET_ORDERS_PER_TURN",
    "ACT_TIMEOUT",
    "MOVES",
    "quadrant_of",
    "shed_access_tiles",
    "hire_cost",
    "water_bonus_window",
]

ENVIRONMENT = "kaggriculture"

# Reference opponents shipped with the environment, strongest first.
BASELINE_AGENTS = ("starter", "random", "pass")

# Environment configuration defaults (kaggriculture.json). The competition runs
# with these values; local evaluation overrides only `seed`.
BOARD_SIZE = 10
TURNS_PER_DAY = 24
EPISODE_STEPS = 720
SEASON_DAYS = EPISODE_STEPS // TURNS_PER_DAY
LAST_DAY = SEASON_DAYS - 1
STARTING_MONEY = 3000
SHED_CAPACITY = 100
MAX_MARKET_ORDERS_PER_TURN = 10
ACT_TIMEOUT = 1.0

# Movement op -> (dx, dy); y grows downward.
MOVES = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}


def quadrant_of(x: int, y: int, board_size: int = BOARD_SIZE) -> str:
    """Return the quadrant name ("NW", "NE", "SW", "SE") containing a tile."""
    half = board_size // 2
    return ("N" if y < half else "S") + ("W" if x < half else "E")


def shed_access_tiles(board_size: int = BOARD_SIZE) -> list[tuple[int, int]]:
    """Return the four inner-corner tiles that touch the shed, in NWSE order."""
    half = board_size // 2
    return [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]


def hire_cost(hires_today: int, mult: int = 1) -> int:
    """Return the coin cost of the next hire given how many were made today.

    Args:
        hires_today: Number of hands already hired during the current day.
        mult: The episode's ``farmHandCostMult`` (1 by default).

    Returns:
        Cost of the next ``HIRE`` order, ``mult * fib(hires_today)`` with fib
        indexed 1, 1, 2, 3, 5, ...
    """
    a, b = 1, 1
    for _ in range(hires_today):
        a, b = b, a + b
    return mult * a


def water_bonus_window(crop: str) -> tuple[int, int]:
    """Return the inclusive age-in-days window where watering adds yield.

    Only one-time crops (wheat, carrot, melon) have a bonus window; ongoing
    crops return ``(-1, -1)`` because their scheduled production ignores it.
    """
    data = CROPS[crop]
    if data["ongoing"]:
        return (-1, -1)
    return ((data["max_yield_day"] + 1) // 2, data["max_yield_day"])
