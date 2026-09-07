"""Opponents by name. Paths resolve here and are printed nowhere.

A sandbox is handed an opponent's name and its measured behaviour, never
where its source lives -- but that is a convention this module keeps, not a
wall it builds: ``codex exec -s workspace-write`` restricts writes only, and
a session that walked up to the repository could read this file. What stops
a candidate from copying an opponent is the doctrine in the prompt and the
copy-check gate that rejects a candidate resembling one. ``path`` is called
only by the harness and the gate.

Measured overlap, so the held-out and field numbers are read with the right
discount (spec section 5.6): ``indarkarhana`` against ``lynnsakurai_v5``
scores 0.165, so that held-out opponent is close to independent of the pool;
``v54`` against ``v56`` scores 0.737, so the two Kaito lineages are close to
one opponent counted twice in the field average.

The published field clones itself heavily, so a name is not an opponent. Of
nine kernels pulled on 2026-09-06, three were dropped: ``flexonafft`` ships
byte-identical C++ to ``avioon``, ``yhay81/three-day-shop-router`` is the C++
build of the router ``lynnsakurai_threeday`` already is in pure Python, and
``kaitofukami``'s packing was not one this could read. Counting those as
opponents would have bought evaluation cost and no resolution.
"""

from pathlib import Path

from kaggriculture.campaign import config

TRAINING: dict[str, Path] = {
    # Harvested 2026-09-01 and before.
    "router_v1": config.AGENTS / "yhay81_router_v1" / "main.py",
    "router2929": config.OPPONENTS / "yhay81_router2929" / "main.py",
    "v54": config.OPPONENTS / "kaito_v54" / "main.py",
    "v56": config.OPPONENTS / "kaito_v56" / "main.py",
    "shopforge": config.OPPONENTS / "tetsutani_shopforge" / "main.py",
    "indarkarhana": config.OPPONENTS / "indarkarhana_top10" / "main.py",
    # Harvested 2026-09-06, from kernels last run 2026-09-02 to 09-05. The
    # pool the campaign was gating on had stopped being the ladder it is
    # aiming at: five of these are by authors the roster held nothing from.
    "boatlee_v29": config.OPPONENTS / "boatlee_v29" / "main.py",
    "lynnsakurai_threeday": config.OPPONENTS / "lynnsakurai_threeday" / "main.py",
    "pilkwang_economic": config.OPPONENTS / "pilkwang_economic" / "main.py",
    "tetsutani_shape0905": config.OPPONENTS / "tetsutani_shape0905" / "main.py",
    "thomastschinkel_router": config.OPPONENTS / "thomastschinkel_router" / "main.py",
    # C++: a policy plus a 108KB tape, built on first load and cached as
    # `agent.so` beside its `main.py`. That build takes about a hundred
    # seconds and games run one per process, many at once, so the binary is
    # built once by hand when the opponent is installed and never during an
    # evaluation. If it is ever missing, the first game to reach it pays the
    # compile and the rest race it.
    "avioon_apex_v7": config.OPPONENTS / "avioon_apex_v7" / "main.py",
}
# Opponents we hold that the pool does not include. They were the held-out
# set, played on the sealed block and never trained against, and that job is
# gone: seeds are drawn fresh for every evaluation now, so every rate the gate
# reads was already measured on maps the program was never selected on, and a
# frozen pair of agents adds nothing a fresh draw does not.
#
# They stay on the roster because the copy check walks it. A file we hold that
# no gate plays is still a file a session could reach, and a candidate that
# resembled one would be a copy whether or not it ever met it in a game.
UNPOOLED: dict[str, Path] = {
    "salemali7_2900": config.OPPONENTS / "salemali7_2900" / "main.py",
    "lynnsakurai_v5": config.OPPONENTS / "lynnsakurai_v5" / "main.py",
}


def names() -> list[str]:
    """Every name the harness accepts."""
    return [*TRAINING, *UNPOOLED]


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
    if name in UNPOOLED:
        return UNPOOLED[name]
    from kaggriculture.campaign.pool import Pool

    if not config.POOL.exists():
        raise KeyError(name)
    return Path(Pool.load(config.POOL).opponents[name])
