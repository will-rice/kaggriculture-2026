"""Guard the constants we re-derive against the environment's own definitions."""

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as env_module

from kaggriculture import constants


def test_board_defaults_match_environment() -> None:
    """Our assumed episode configuration is the environment's default."""
    configuration = make("kaggriculture").configuration

    assert configuration.boardSize == constants.BOARD_SIZE
    assert configuration.turnsPerDay == constants.TURNS_PER_DAY
    assert configuration.episodeSteps == constants.EPISODE_STEPS
    assert configuration.startingMoney == constants.STARTING_MONEY
    assert configuration.shedCapacity == constants.SHED_CAPACITY
    assert configuration.maxMarketOrdersPerTurn == constants.MAX_MARKET_ORDERS_PER_TURN


def test_movement_offsets_match_environment() -> None:
    """Our move table matches the offsets the interpreter applies."""
    assert constants.MOVES == env_module.FARMER_MOVES


def test_shed_access_and_quadrants_match_environment() -> None:
    """Shed adjacency and quadrant naming match the interpreter's private helpers."""
    assert constants.shed_access_tiles() == env_module._shed_access_tiles(constants.BOARD_SIZE)
    for x in range(constants.BOARD_SIZE):
        for y in range(constants.BOARD_SIZE):
            assert constants.quadrant_of(x, y) == env_module._quadrant_of(x, y, constants.BOARD_SIZE)


def test_hire_cost_matches_environment() -> None:
    """Hire pricing matches the fibonacci schedule the market applies."""
    for hires in range(12):
        assert constants.hire_cost(hires) == env_module._hire_cost(hires)


def test_water_bonus_window_matches_interpreter() -> None:
    """The bonus window matches the range the WATER op rewards."""
    for crop, data in constants.CROPS.items():
        start, end = constants.water_bonus_window(crop)
        if data["ongoing"]:
            assert (start, end) == (-1, -1)
        else:
            assert start == (data["max_yield_day"] + 1) // 2
            assert end == data["max_yield_day"]
