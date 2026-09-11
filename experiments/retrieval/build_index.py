"""Index the ladder's recorded boards so a policy can look up what to do.

A tape is open loop: it plays the same 720 actions whatever the board does, so
it is strong on the seed it was recorded on and drifts everywhere else. The
same recordings hold much more than one route -- 17,679 qualifying episodes,
25.5 million state-action pairs -- and what makes them useful is that a policy
can *retrieve* from them instead of replaying them. Face a board, find the most
similar board a strong player faced, and play what they played there.

That is a closed-loop use of the same data: nothing about it commits to a route
ahead of time, and it answers a board it has never seen with the nearest one it
has. Our champion is already the degenerate case -- four routes and three
branch points -- and this is the same architecture with thousands of branches.

This builds the index offline, for an opponent rather than a submission, so
none of the one-file or one-second limits apply.

The pairing is the fiddly part. `tapes` records the action at ``steps[t]`` as
what the seat submitted while looking at ``steps[t-1]``'s observation, so an
example is that observation with that action, never the two at the same index.
"""

import argparse
import json
import logging
import pathlib

import numpy as np
from tqdm import tqdm

from kaggriculture.campaign import tapes

LOGGER = logging.getLogger(__name__)
HERE = pathlib.Path(__file__).resolve().parent
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL")
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
# Every feature is divided by one of these, so no single quantity dominates the
# distance simply by being counted in thousands.
SCALE = np.array(
    [20_000.0, 20_000.0]  # our bank, theirs
    + [50.0] * len(PRODUCTS)  # shed
    + [20.0] * len(CROPS)  # seeds held
    + [200.0] * len(PRODUCTS)  # market prices
    + [200.0] * len(PRODUCTS)  # market inventory, offset from baseline
    + [12.0, 4.0]  # hands, quadrants
    + [25.0] * len(CROPS)  # our plants by crop
    + [10.0] * len(ANIMALS)  # our animals by kind
    + [25.0, 10.0],  # their plants, their animals
    dtype=np.float32,
)


def features(observation: dict) -> np.ndarray:
    """The board as a vector, in the order `SCALE` scales.

    Public state for both farms, and the acting seat's own private state --
    which is what that seat could see when it chose, so the index never holds
    anything the retrieving agent will not have.
    """
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


def main() -> None:
    """Walk episodes and write one index of boards, actions and their steps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", type=int, help="how many episodes to index")
    parser.add_argument("--out", type=pathlib.Path, default=HERE / "index.npz")
    parser.add_argument(
        "--winners-only",
        action="store_true",
        help="index only the seat that won, so the policy imitates the winner",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    boards, actions, steps = [], [], []
    kept = 0
    for i in tqdm(range(args.episodes), desc="episodes"):
        try:
            episode = tapes.qualifying(i)
        except FileNotFoundError:
            LOGGER.info("corpus exhausted at %d episodes", i)
            break
        rewards = [record.get("reward") or 0.0 for record in episode.steps[-1]]
        seats = [int(np.argmax(rewards))] if args.winners_only else [0, 1]
        for seat in seats:
            for t in range(1, len(episode.steps)):
                action = episode.steps[t][seat].get("action")
                observation = episode.steps[t - 1][seat].get("observation")
                if not action or not observation or "farms" not in observation:
                    continue
                boards.append(features(observation))
                actions.append(json.dumps(action, separators=(",", ":")))
                steps.append(t - 1)
        kept += 1

    array = np.vstack(boards)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        boards=array,
        steps=np.array(steps, dtype=np.int32),
        actions=np.array(actions),
    )
    size = args.out.stat().st_size / 1e6
    LOGGER.info(
        "%d episodes -> %d boards of %d features, %.1f MB at %s",
        kept,
        len(boards),
        array.shape[1],
        size,
        args.out,
    )


if __name__ == "__main__":
    main()
