"""Routes are harvested from episodes and survive a round trip."""

import json
import zipfile
from pathlib import Path

import pytest

from kaggriculture.search import route as route_module

ARCHIVE = Path("/data/kaggriculture/episodes/kaggriculture-episodes-2026-08-15.zip")


@pytest.mark.skipif(not ARCHIVE.exists(), reason="replay corpus not on this machine")
def test_a_harvested_route_has_one_entry_per_turn() -> None:
    """An episode's seat is 720 turns of actions in the engine's own grammar."""
    with zipfile.ZipFile(ARCHIVE) as bundle:
        episode = json.loads(bundle.read("93454366.json"))

    harvested = route_module.from_episode(episode, seat=0)

    assert len(harvested) == 720
    assert all(set(turn) == {"farmer", "hands", "market"} for turn in harvested)
    assert any(turn["market"] for turn in harvested)


def test_a_route_survives_a_round_trip(tmp_path: Path) -> None:
    """Saving and loading must not alter a single order."""
    original = [
        {"farmer": ["PASS"], "hands": [["WATER"]], "market": [["SELL", "WHEAT", 3]]}
    ] * 720

    route_module.save(original, tmp_path / "route.json")

    assert route_module.load(tmp_path / "route.json") == original
