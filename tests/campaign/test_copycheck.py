"""Copied code cannot pass the gate."""

from pathlib import Path

from kaggriculture.campaign import copycheck, roster

SKELETON = Path("src/kaggriculture/served/main.py").read_text(encoding="utf-8")

# A plausible independent agent using the same op/item vocabulary as every
# opponent, but written fresh: this is the size and shape an evolved
# candidate will actually be, so its score is the one the threshold must
# clear.
INDEPENDENT_AGENT = '''
"""A small farming agent: water thirsty crops, harvest ready ones, buy seed."""

from collections.abc import Mapping
from typing import Any

ITEMS = ["wheat", "corn", "tomato", "chicken", "cow", "sheep"]


def _thirsty(tile: Mapping[str, Any]) -> bool:
    return tile.get("crop") is not None and tile.get("water", 0) < 2


def _ready(tile: Mapping[str, Any]) -> bool:
    growth = tile.get("growth", 0)
    return tile.get("crop") is not None and growth >= tile.get("maturity", 99)


def _farmer_action(observation: Mapping[str, Any], farmer: Mapping[str, Any]) -> str:
    tile = farmer.get("tile", {})
    if _ready(tile):
        return "HARVEST"
    if _thirsty(tile):
        return "WATER"
    if tile.get("crop") is None and farmer.get("holding") == "seed":
        return "PLANT"
    return "NORTH"


def _market_action(observation: Mapping[str, Any]) -> list[Any]:
    cash = observation.get("cash", 0)
    prices = observation.get("prices", {})
    for item in ITEMS:
        if cash > prices.get(item, 10_000) * 2:
            return ["BUY_SEED", item]
    return ["NONE"]


def agent(
    observation: Mapping[str, Any], configuration: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Water what is thirsty, harvest what is ready, buy seed when flush."""
    farmers = observation.get("farmers", [])
    hands = [_farmer_action(observation, farmer) for farmer in farmers]
    return {"farmer": hands, "hands": [], "market": _market_action(observation)}
'''


def test_a_verbatim_opponent_scores_one() -> None:
    """An opponent's own file matches itself, near-exactly."""
    source = roster.path("v54").read_text(encoding="utf-8")
    name, score = copycheck.against_opponents(source)
    assert name.startswith("v54") and score > 0.99


def test_the_skeleton_scores_near_zero() -> None:
    """The tiny PASS skeleton shares almost nothing with any opponent."""
    _, score = copycheck.against_opponents(SKELETON)
    assert score < 0.05


def test_a_realistic_independent_agent_scores_below_threshold() -> None:
    """The number that matters: evolved candidates will be this size."""
    _, score = copycheck.against_opponents(INDEPENDENT_AGENT)
    assert score < copycheck.THRESHOLD


def test_a_lifted_block_is_caught() -> None:
    """A 200-line block lifted from an opponent trips the gate."""
    source = roster.path("shopforge").read_text(encoding="utf-8")
    lines = source.splitlines()
    lifted = "\n".join(lines[len(lines) // 2 : len(lines) // 2 + 200])
    candidate = SKELETON + "\n" + lifted
    _, score = copycheck.against_opponents(candidate)
    assert score >= copycheck.THRESHOLD


def test_similarity_is_symmetric_and_bounded() -> None:
    """similarity(a, b) == similarity(b, a), and both are in [0, 1]."""
    a, b = "x = 1\ny = x + 2\n" * 20, "y = 2\nx = y + 1\n" * 20
    assert copycheck.similarity(a, b) == copycheck.similarity(b, a)
    assert 0.0 <= copycheck.similarity(a, b) <= 1.0
