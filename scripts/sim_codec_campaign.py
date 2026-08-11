"""Verify codec round-trips over real 1.32.6 replay states."""

import argparse
import json
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from kaggle_environments.utils import Struct

from kaggriculture.sim.state import pack, unpack


def _agents(step: list[dict[str, Any]]) -> list[Struct]:
    agents = []
    for entry in step:
        raw = entry["observation"]
        observation = Struct(**raw)
        observation.market = Struct(**raw["market"])
        observation.town = Struct(**raw["town"])
        observation.private = Struct(**raw["private"])
        agents.append(
            Struct(
                observation=observation,
                status=entry["status"],
                reward=entry.get("reward"),
            )
        )
    return agents


def _states(archive: Path) -> Iterator[tuple[list[Struct], int]]:
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            if not name.endswith(".json"):
                continue
            episode = json.load(bundle.open(name))
            if int(episode["configuration"].get("townCenterSellInterval", 0)) != 24:
                continue
            configured_seed = episode["configuration"].get("seed")
            seed = int(
                episode.get("info", {}).get("seed")
                if configured_seed is None
                else configured_seed
            )
            for step in episode["steps"]:
                yield _agents(step), seed


def _check_batch(batch: list[tuple[list[Struct], int]], checked: int) -> None:
    encoded = pack(batch, materialize_rng=False)
    for batch_index, (agents, _seed) in enumerate(batch):
        state_index = checked + batch_index
        for seat in range(2):
            expected = dict(agents[seat]["observation"])
            expected.pop("remainingOverageTime", None)
            actual = unpack(encoded, batch_index, seat)
            if actual != expected:
                raise AssertionError(
                    f"codec divergence at state {state_index}, seat {seat}"
                )
            expected_inventories = expected["private"]["inventories"]
            actual_inventories = actual["private"]["inventories"]
            for unit_index, (expected_inventory, actual_inventory) in enumerate(
                zip(expected_inventories, actual_inventories, strict=True)
            ):
                if list(actual_inventory.items()) != list(expected_inventory.items()):
                    raise AssertionError(
                        "inventory order divergence at "
                        f"state {state_index}, seat {seat}, unit {unit_index}"
                    )


def run(archive: Path, limit: int, batch_size: int = 128) -> int:
    """Round-trip at most ``limit`` states and return the number checked."""
    checked = 0
    batch: list[tuple[list[Struct], int]] = []
    for state in _states(archive):
        batch.append(state)
        if len(batch) == min(batch_size, limit - checked):
            _check_batch(batch, checked)
            checked += len(batch)
            batch = []
            if checked >= limit:
                return checked
    if batch:
        _check_batch(batch, checked)
        checked += len(batch)
    return checked


def main() -> None:
    """Run the command-line codec campaign."""
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--limit", type=int, default=50_000)
    args = parser.parse_args()
    checked = run(args.archive, args.limit)
    if checked != args.limit:
        raise SystemExit(f"archive supplied only {checked} states")
    print(
        json.dumps({"archive": str(args.archive), "states": checked, "divergences": 0})
    )


if __name__ == "__main__":
    main()
