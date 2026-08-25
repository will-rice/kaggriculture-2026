"""Action construction for a single Kaggriculture turn.

The environment expects ``{"farmer": op, "hands": [op, ...], "market": [order, ...]}``
where every op is a list of ``[name, *args]``. ``Turn`` keeps that shape in one
place so a policy cannot emit a malformed action dict.
"""

from dataclasses import dataclass, field
from typing import Any

from kaggriculture.constants import BOARD_SIZE, MAX_MARKET_ORDERS_PER_TURN, MOVES
from kaggriculture.observation import Position

Op = list[Any]

PASS: Op = ["PASS"]


@dataclass
class Turn:
    """One turn's orders for the farmer, every hired hand, and the market."""

    farmer: Op = field(default_factory=lambda: list(PASS))
    hands: list[Op] = field(default_factory=list)
    market: list[Op] = field(default_factory=list)

    def to_action(self) -> dict[str, Any]:
        """Return the action dict the environment consumes.

        Market orders past ``maxMarketOrdersPerTurn`` are dropped silently by the
        environment, so they are truncated here to make the loss visible in tests
        and replays rather than at scoring time.
        """
        return {
            "farmer": self.farmer,
            "hands": self.hands,
            "market": self.market[:MAX_MARKET_ORDERS_PER_TURN],
        }


def step_toward(src: Position, dst: Position) -> Op:
    """Return the move op that closes the gap between two tiles.

    Movement is one orthogonal step per turn and no tile blocks movement, so
    walking the x axis then the y axis is already a shortest path.
    """
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]
    if dx:
        return ["EAST"] if dx > 0 else ["WEST"]
    if dy:
        return ["SOUTH"] if dy > 0 else ["NORTH"]
    return list(PASS)


def ranked_steps_toward(src: Position, dst: Position) -> tuple[Op, ...]:
    """Return every in-bounds move ranked by resulting distance to ``dst``."""
    preferred = step_toward(src, dst)[0]
    candidates: list[tuple[int, bool, int, str]] = []
    for stable_order, (name, (dx, dy)) in enumerate(MOVES.items()):
        next_position = (src[0] + dx, src[1] + dy)
        if not all(0 <= coordinate < BOARD_SIZE for coordinate in next_position):
            continue
        candidates.append(
            (
                distance(next_position, dst),
                name != preferred,
                stable_order,
                name,
            )
        )
    return tuple([name] for _, _, _, name in sorted(candidates))


def distance(src: Position, dst: Position) -> int:
    """Return the number of turns needed to walk between two tiles."""
    return abs(dst[0] - src[0]) + abs(dst[1] - src[1])
