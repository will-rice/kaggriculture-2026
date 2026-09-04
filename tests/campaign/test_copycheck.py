"""Copied code cannot pass the gate."""

import re
from pathlib import Path

from kaggriculture.campaign import copycheck, roster

SKELETON = Path("src/kaggriculture/served/main.py").read_text(encoding="utf-8")

# A plausible independent agent using the same op/item vocabulary as every
# opponent, but written fresh: this is the size and shape an evolved
# candidate will actually be, so its score is the one the threshold must
# clear.
INDEPENDENT_AGENT = '''
"""A small farming agent: water thirsty crops, harvest ready ones, buy seed,
hire a hand when cash is flush, and sell whatever is piling up in storage."""

from collections.abc import Mapping
from typing import Any

ITEMS = ["wheat", "corn", "tomato", "chicken", "cow", "sheep"]
HIRE_COST_CEILING = 5_000
STORAGE_SELL_FLOOR = 20


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
    if farmer.get("holding") in ITEMS:
        return "DROP"
    return "NORTH"


def _should_hire(observation: Mapping[str, Any]) -> bool:
    cash = observation.get("cash", 0)
    hire_price = observation.get("hire_price", HIRE_COST_CEILING + 1)
    farmers = observation.get("farmers", [])
    return cash > hire_price * 3 and len(farmers) < 4


def _oversupplied_item(observation: Mapping[str, Any]) -> str | None:
    storage = observation.get("storage", {})
    for item in ITEMS:
        if storage.get(item, 0) >= STORAGE_SELL_FLOOR:
            return item
    return None


def _market_action(observation: Mapping[str, Any]) -> list[Any]:
    if _should_hire(observation):
        return ["HIRE"]
    surplus = _oversupplied_item(observation)
    if surplus is not None:
        return ["SELL", surplus]
    cash = observation.get("cash", 0)
    prices = observation.get("prices", {})
    for item in ITEMS:
        if cash > prices.get(item, 10_000) * 2:
            return ["BUY_SEED", item]
    return ["NONE"]


def agent(
    observation: Mapping[str, Any], configuration: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Water, harvest, plant, hire when flush, and sell what piles up."""
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


def test_a_reformatted_lift_is_still_caught() -> None:
    """Collapsing whitespace and swapping quote style cannot launder a lift.

    This is what an LLM asked to "rewrite this" will do to a copied block:
    the tokens are unchanged, so the score must be too.
    """
    path = Path(
        "/data/kaggriculture/opponents/indarkarhana_top10/agents/"
        "e749a_niklita_consensus_network.py"
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    lifted = "\n".join(lines[90:200])  # lines 91-200, 1-indexed
    reformatted = re.sub(r"\s*(->|[:,=])\s*", r"\1", lifted).replace('"', "'")
    candidate = SKELETON + "\n" + reformatted
    _, score = copycheck.against_opponents(candidate)
    assert score >= copycheck.THRESHOLD


def test_similarity_is_symmetric_and_bounded() -> None:
    """similarity(a, b) == similarity(b, a), and both are in [0, 1]."""
    a, b = "x = 1\ny = x + 2\n" * 20, "y = 2\nx = y + 1\n" * 20
    assert copycheck.similarity(a, b) == copycheck.similarity(b, a)
    assert 0.0 <= copycheck.similarity(a, b) <= 1.0
