"""Play by retrieval: find the nearest recorded board and do what they did.

An opponent, not a submission. It reads an index off disk and is not bound by
the one-file rule or the one-second call limit that govern a Kaggle agent.

A tape replays one route regardless of the board. This holds thousands of
routes at once and picks per turn, so it answers a position it has never seen
with the closest position a strong player did see. Candidates are restricted to
the same turn of the season, because the same board on day 3 and day 27 calls
for opposite play and nothing in the feature vector says which is which.

Built for the pool. The campaign's opponents are 69 descendants of one program
and 14 harvested publics, so a candidate is largely being measured against its
own family; this is one opponent that plays like the aggregate of the ladder
and answers what the candidate actually does.
"""

import json
import pathlib

import numpy as np

INDEX = pathlib.Path(__file__).resolve().parent / "index.npz"
# Turns either side of the current one whose boards are also eligible. Zero
# would leave a turn with no near neighbour stranded on whatever it has; a
# small window borrows from adjacent turns, where play is nearly the same.
WINDOW = 1
_STORE = None


def _store() -> tuple:
    """Load the index once and keep it, bucketed by turn."""
    global _STORE
    if _STORE is None:
        raw = np.load(INDEX, allow_pickle=False)
        boards = raw["boards"]
        steps = raw["steps"]
        actions = raw["actions"]
        order = np.argsort(steps, kind="stable")
        boards, steps, actions = boards[order], steps[order], actions[order]
        # Where each turn's block starts, so a lookup is a slice.
        starts = np.searchsorted(steps, np.arange(steps.max() + 2))
        _STORE = (boards, actions, starts)
    return _STORE


def agent(observation: dict, configuration: object = None) -> dict:
    """The action a strong player took on the most similar board at this turn."""
    boards, actions, starts = _store()
    step = int(observation.get("day", 0)) * 24 + int(observation.get("hour", 0))
    low = starts[max(0, step - WINDOW)]
    high = starts[min(len(starts) - 1, step + WINDOW + 1)]
    if high <= low:
        seat = int(observation.get("player", 0))
        hands = observation["farms"][seat].get("hands") or []
        return {
            "farmer": ["PASS"],
            "hands": [["PASS"] for _ in hands],
            "market": [],
        }
    here = _features(observation)
    block = boards[low:high]
    gaps = block - here
    nearest = int(np.argmin(np.einsum("ij,ij->i", gaps, gaps)))
    chosen = json.loads(str(actions[low + nearest]))
    # The recorded seat had its own number of hands; ours is whatever we have.
    hands = observation["farms"][int(observation.get("player", 0))].get("hands") or []
    theirs = list(chosen.get("hands") or [])
    theirs = (theirs + [["PASS"]] * len(hands))[: len(hands)]
    return {
        "farmer": list(chosen.get("farmer") or ["PASS"]),
        "hands": theirs,
        "market": [list(order) for order in (chosen.get("market") or [])],
    }


PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL")
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
SCALE = np.array(
    [20_000.0, 20_000.0]
    + [50.0] * len(PRODUCTS)
    + [20.0] * len(CROPS)
    + [200.0] * len(PRODUCTS)
    + [200.0] * len(PRODUCTS)
    + [12.0, 4.0]
    + [25.0] * len(CROPS)
    + [10.0] * len(ANIMALS)
    + [25.0, 10.0],
    dtype=np.float32,
)


def _features(observation: dict) -> object:
    """The board as a vector, in the order the index was built with."""
    me = int(observation.get("player", 0))
    mine = observation["farms"][me]
    theirs = observation["farms"][1 - me]
    private = observation.get("private") or {}
    shed = private.get("shed") or {}
    seeds = private.get("seeds") or {}
    market = observation["market"]
    row = [float(mine.get("money", 0)), float(theirs.get("money", 0))]
    row += [float(shed.get(p, 0)) for p in PRODUCTS]
    row += [float(seeds.get(c, 0)) for c in CROPS]
    row += [float(market["prices"].get(p, 0)) for p in PRODUCTS]
    row += [float(market["inventory"].get(p, 10_000)) - 10_000.0 for p in PRODUCTS]
    row += [
        float(len(mine.get("hands") or [])),
        float(len(mine.get("unlocked_quadrants") or [])),
    ]
    row += _plants(mine) + _animals(mine)
    row += [float(sum(_plants(theirs))), float(sum(_animals(theirs)))]
    return np.array(row, dtype=np.float32) / SCALE


def _plants(farm: dict) -> list:
    """Growing tiles per crop on one farm."""
    counts = dict.fromkeys(CROPS, 0.0)
    for tile in _tiles(farm):
        crop = tile.get("crop")
        if crop in counts:
            counts[crop] += 1.0
    return [counts[c] for c in CROPS]


def _animals(farm: dict) -> list:
    """Placed animals per kind on one farm."""
    counts = dict.fromkeys(ANIMALS, 0.0)
    for tile in _tiles(farm):
        animal = tile.get("animal")
        if animal in counts:
            counts[animal] += 1.0
    return [counts[a] for a in ANIMALS]


def _tiles(farm: dict):  # noqa: ANN202 - yields tile dicts
    """Every tile on a farm that carries a description."""
    for row in farm.get("tiles") or []:
        for tile in row:
            if isinstance(tile, dict):
                yield tile
