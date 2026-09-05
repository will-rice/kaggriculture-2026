"""Opponents by name. The paths are the only secret the campaign keeps.

A sandbox learns an opponent's name and its measured behaviour, never where
its source lives. ``path`` is called only by the harness and the gate.
"""

from pathlib import Path

from kaggriculture.campaign import config

AGENTS = Path("/data/kaggriculture/agents")

TRAINING: dict[str, Path] = {
    "router_v1": AGENTS / "yhay81_router_v1" / "main.py",
    "router2929": config.OPPONENTS / "yhay81_router2929" / "main.py",
    "v54": config.OPPONENTS / "kaito_v54" / "main.py",
    "v56": config.OPPONENTS / "kaito_v56" / "main.py",
    "shopforge": config.OPPONENTS / "tetsutani_shopforge" / "main.py",
    "indarkarhana": config.OPPONENTS / "indarkarhana_top10" / "main.py",
}
HELD_OUT: dict[str, Path] = {
    "salemali7_2900": config.OPPONENTS / "salemali7_2900" / "main.py",
    "lynnsakurai_v5": config.OPPONENTS / "lynnsakurai_v5" / "main.py",
}


def names() -> list[str]:
    """Every name the harness accepts."""
    return [*TRAINING, *HELD_OUT]


def path(name: str) -> Path:
    """Resolve a name; anything else is a KeyError, never a path lookup.

    A champion promoted into the opponent pool is an opponent the gate must
    play, and it is not vendored, so the pool file is the second and last
    place a name can resolve. It is read only on a roster miss, and the
    import is local because ``pool`` imports this module.

    Args:
        name: An opponent name.

    Returns:
        The opponent's ``main.py``.

    Raises:
        KeyError: The name is in neither the roster nor the pool.
    """
    if name in TRAINING:
        return TRAINING[name]
    if name in HELD_OUT:
        return HELD_OUT[name]
    from kaggriculture.campaign.pool import Pool

    if not config.POOL.exists():
        raise KeyError(name)
    return Path(Pool.load(config.POOL).opponents[name])
