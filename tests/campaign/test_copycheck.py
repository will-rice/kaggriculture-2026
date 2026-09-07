"""Copied code cannot pass the gate."""

import re
from pathlib import Path

import pytest

from kaggriculture.campaign import config, copycheck, roster

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


@pytest.mark.local_data
def test_a_verbatim_opponent_scores_one() -> None:
    """An opponent's own file matches itself, near-exactly."""
    source = roster.path("v54").read_text(encoding="utf-8")
    name, score = copycheck.against_opponents(source)
    # Named for the directory it is vendored under, which is what the
    # corpus is built from now -- the roster's nickname is not a file.
    assert name.startswith(roster.path("v54").parent.name) and score > 0.99


def test_the_skeleton_scores_near_zero() -> None:
    """The tiny PASS skeleton shares almost nothing with any opponent."""
    _, score = copycheck.against_opponents(SKELETON)
    assert score < 0.05


def test_a_realistic_independent_agent_scores_below_threshold() -> None:
    """The number that matters: evolved candidates will be this size."""
    _, score = copycheck.against_opponents(INDEPENDENT_AGENT)
    assert score < copycheck.THRESHOLD


@pytest.mark.local_data
def test_a_lifted_block_is_caught() -> None:
    """A 200-line block lifted from an opponent trips the gate."""
    source = roster.path("shopforge").read_text(encoding="utf-8")
    lines = source.splitlines()
    lifted = "\n".join(lines[len(lines) // 2 : len(lines) // 2 + 200])
    candidate = SKELETON + "\n" + lifted
    _, score = copycheck.against_opponents(candidate)
    assert score >= copycheck.THRESHOLD


@pytest.mark.local_data
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


@pytest.mark.local_data
def test_an_agent_the_roster_never_named_is_still_in_the_corpus() -> None:
    """The corpus is what is on disk, not what the harness is allowed to play.

    Those were the same list while opponents arrived by hand-editing the
    roster. They stop being the same the moment the pool takes a harvested
    agent nobody registered -- and a gate keyed on the roster would let a
    candidate copy that one freely, silently, and precisely because it is
    new, which is to say precisely when it is the strongest thing in the pool.
    """
    covered = {key.split(":", 1)[0] for key in copycheck._corpus()}
    named = {roster.path(name).parent.name for name in roster.names()}

    assert covered - named, "the corpus is only the roster; the scan is not live"
    # And every registered opponent is still in there, the seed aside.
    assert named - covered == {config.SEED.parent.name}


def test_similarity_is_symmetric_and_bounded() -> None:
    """similarity(a, b) == similarity(b, a), and both are in [0, 1]."""
    a, b = "x = 1\ny = x + 2\n" * 20, "y = 2\nx = y + 1\n" * 20
    assert copycheck.similarity(a, b) == copycheck.similarity(b, a)
    assert 0.0 <= copycheck.similarity(a, b) <= 1.0


def test_a_machine_without_the_opponents_has_no_lineage_either(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The seed is under `/data`, and CI does not have `/data`.

    `_corpus` already degrades to empty there, because `rglob` over a
    directory that is not on the machine yields nothing. The lineage has to
    degrade the same way: reading the seed eagerly turns every unmarked test
    in this file into a `FileNotFoundError` everywhere but the box.
    """
    monkeypatch.setattr(config, "SEED_PROGRAM", Path("/nonexistent/seed.py"))
    copycheck._lineage.cache_clear()
    copycheck._corpus.cache_clear()
    try:
        assert copycheck._lineage() == frozenset()
    finally:
        copycheck._lineage.cache_clear()
        copycheck._corpus.cache_clear()


@pytest.mark.local_data
def test_the_seed_is_exempt_and_every_other_opponent_is_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The campaign must not fail its own gate, and nothing else may pass it.

    Every program descends from the stored seed, so without the exemption the
    gate rejects the whole campaign rather than a copy -- the seed scores
    1.000 against the file it was copied from, and a child that has not
    rewritten every line scores just behind it.

    The other half matters as much. The exemption is by similarity rather than
    by name, so a corpus that drifted -- another author republishing the
    seed's kernel, a threshold moved -- could quietly empty itself and leave
    nothing to catch a lift at all. Seeding from one opponent must not stand
    the gate down for a different one.

    Seeded here rather than read from the live campaign, so this measures the
    rule instead of whatever the box happens to be running today.
    """
    seeded = roster.path("shopforge").read_text(encoding="utf-8")
    seed = tmp_path / "seed.py"
    seed.write_text(seeded, encoding="utf-8")
    monkeypatch.setattr(config, "SEED_PROGRAM", seed)
    copycheck._lineage.cache_clear()
    copycheck._corpus.cache_clear()
    try:
        _, score = copycheck.against_opponents(seeded)
        assert score < copycheck.THRESHOLD

        other = roster.path("v54").read_text(encoding="utf-8")
        name, other_score = copycheck.against_opponents(other)
        assert other_score >= copycheck.THRESHOLD
        assert name.startswith(roster.path("v54").parent.name)
    finally:
        copycheck._lineage.cache_clear()
        copycheck._corpus.cache_clear()
