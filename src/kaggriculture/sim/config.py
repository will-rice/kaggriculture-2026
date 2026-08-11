"""Frozen competition configuration for the batched simulator."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """The one configuration supported by the competition simulator."""

    episode_steps: int = 720
    board_size: int = 10
    starting_money: int = 3000
    max_market_orders_per_turn: int = 10
    turns_per_day: int = 24
    shed_capacity: int = 100
    weed_spawn_chance: float = 0.005
    town_shop_unlock_interval: int = 3
    town_shop_sell_interval: int = 4
    town_center_sell_interval: int = 24
    farm_hand_cost_mult: int = 1
