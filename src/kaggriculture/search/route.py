"""A route: one seat's whole season, in the engine's own action grammar.

The engine reads an action as ``{"farmer": [...], "hands": [[...], ...],
"market": [[...], ...]}``, and a route is 720 of them. Keeping that grammar
rather than inventing one means a route can be harvested from any replay, handed
straight back to the reference engine, and diffed against a competitor's tape
without a translation layer in between.
"""

import json
from pathlib import Path
from typing import Any

from kaggriculture.constants import EPISODE_STEPS

Route = list[dict[str, Any]]


def from_episode(episode: dict[str, Any], seat: int) -> Route:
    """Return one seat's actions from a decoded episode replay.

    Args:
        episode: A replay as published in the daily episode archives.
        seat: Which player to harvest, 0 or 1.

    Returns:
        One action per turn, normalised so every entry has all three keys.
    """
    if seat not in (0, 1):
        raise ValueError(f"seat must be 0 or 1, got {seat}")
    return [_normalise(step[seat].get("action")) for step in episode["steps"]]


def load(path: Path) -> Route:
    """Return the route stored at ``path``.

    Raises:
        ValueError: If the stored route is not a full ``EPISODE_STEPS``-turn
            season. Left unchecked, a short route surfaces later as an
            ``IndexError`` inside a worker process, far from the file that
            caused it.
    """
    route = [_normalise(turn) for turn in json.loads(Path(path).read_text())]
    if len(route) != EPISODE_STEPS:
        raise ValueError(f"{path}: expected {EPISODE_STEPS} turns, got {len(route)}")
    return route


def save(route: Route, path: Path) -> None:
    """Write ``route`` to ``path`` as JSON."""
    Path(path).write_text(json.dumps(route))


def _normalise(action: dict[str, Any] | None) -> dict[str, Any]:
    """Return one turn's action with every key present and every order a list."""
    action = action or {}
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(order or ["PASS"]) for order in (action.get("hands") or [])],
        "market": [list(order) for order in (action.get("market") or [])],
    }
