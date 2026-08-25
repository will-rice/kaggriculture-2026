"""Reproduce frozen observation tensor digests from the Task 3 base commit.

Run this file from a checkout of the Task 3 implementation with an exact,
detached checkout of ``BASE_COMMIT`` supplied via ``--legacy-root``.  The
result is printed to stdout for review and explicit check-in; tests only read
the checked-in JSON and never invoke this generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
import zipfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from torch import Tensor

BASE_COMMIT = "f0fcc55b1179e2389c75fb97a9a30298ee4fa0e8"
FIELDS = (
    "board",
    "scalars",
    "positions",
    "unit_mask",
    "quantity_mask",
    "market_mask",
)
CORPUS_LIMIT = 2048


def rich_observation() -> dict[str, Any]:
    """Return the deterministic high-coverage observation pinned by the goldens."""
    from kaggle_environments.envs.kaggriculture import kaggriculture as engine

    from kaggriculture.constants import MARKET_PARAMS, PRODUCTS, TURNS_PER_DAY

    def farm() -> dict[str, Any]:
        return {
            "tiles": [[None] * 10 for _ in range(10)],
            "money": 3_000.0,
            "farmer": [4, 4],
            "hands": [],
            "unlocked_quadrants": ["NW"],
            "hires_today": 0,
        }

    observation = {
        "player": 0,
        "day": 3,
        "hour": 7,
        "farms": [farm(), farm()],
        "market": {
            "prices": {name: int(MARKET_PARAMS[name]["base"]) for name in PRODUCTS},
            "inventory": {name: int(MARKET_PARAMS[name]["I0"]) for name in PRODUCTS},
        },
        "town": {"unlocked_shops": []},
        "private": engine._new_private(),
    }
    observation["farms"][0]["money"] = 4_250.0
    observation["farms"][1]["money"] = 1_750.0
    observation["farms"][0]["hands"] = [[2, 3]]
    observation["private"]["inventories"] = [
        {"WHEAT": 5},
        {"CARROT": 2},
    ]
    observation["private"]["shed"]["WHEAT"] = 11
    observation["private"]["seeds"]["MELON"] = 9
    crop = engine._new_plant("MELON", 1, TURNS_PER_DAY)
    crop["consecutive_unwatered"] = 1
    observation["farms"][0]["tiles"][2][3] = crop
    animal = engine._new_animal("GOOSE", 0)
    animal["consecutive_unfed"] = 2
    animal["yield_units"] = 3
    observation["farms"][1]["tiles"][6][7] = animal
    observation["market"]["prices"]["WHEAT"] += 17
    observation["market"]["inventory"]["WHEAT"] -= 23
    return observation


def corpus_observations(
    corpus: Path, limit: int = CORPUS_LIMIT
) -> Iterator[tuple[str, dict[str, Any], int]]:
    """Yield the stable archive/name/turn/seat sample used by the golden gate."""
    yielded = 0
    for archive in sorted(corpus.glob("*.zip")):
        with zipfile.ZipFile(archive) as bundle:
            names = sorted(name for name in bundle.namelist() if name.endswith(".json"))
            for name in names:
                with bundle.open(name) as member:
                    steps = json.load(member)["steps"]
                stride = max(1, len(steps) // 32)
                for index in range(0, len(steps), stride):
                    for seat in (0, 1):
                        yield archive.name, steps[index][seat]["observation"], seat
                        yielded += 1
                        if yielded >= limit:
                            return


def _tensor_bytes(tensor: Tensor) -> bytes:
    """Return a typed, shaped, contiguous byte record for one Torch tensor."""
    array = tensor.detach().cpu().contiguous().numpy()
    metadata = json.dumps(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    payload = array.tobytes(order="C")
    return (
        struct.pack(">Q", len(metadata))
        + metadata
        + struct.pack(">Q", len(payload))
        + payload
    )


def _digests(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Hash every bit of each named tensor over rows in their specified order."""
    hashes = {field: hashlib.sha256() for field in FIELDS}
    for row in rows:
        for field in FIELDS:
            hashes[field].update(_tensor_bytes(row[field]))
    return {field: digest.hexdigest() for field, digest in hashes.items()}


def tensor_digests(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Public test helper for hashing current tensors against frozen values."""
    return _digests(rows)


def _legacy_encoder() -> Callable[[Mapping[str, Any], int], Mapping[str, Any]]:
    """Return the six independent pre-refactor tensor producers."""
    from kaggriculture.learn.encoding import (
        encode_board,
        encode_positions,
        encode_scalars,
    )
    from kaggriculture.learn.mask import market_mask, unit_mask, unit_quantity_mask

    def encode(observation: Mapping[str, Any], seat: int) -> Mapping[str, Any]:
        return {
            "board": encode_board(observation, seat),
            "scalars": encode_scalars(observation, seat),
            "positions": encode_positions(observation, seat),
            "unit_mask": unit_mask(observation, seat),
            "quantity_mask": unit_quantity_mask(observation, seat),
            "market_mask": market_mask(observation, seat),
        }

    return encode


def _commit(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def generate(legacy_root: Path, corpus: Path) -> dict[str, Any]:
    """Generate the reviewed JSON payload using only the exact legacy checkout."""
    actual_commit = _commit(legacy_root)
    if actual_commit != BASE_COMMIT:
        raise ValueError(
            f"legacy root is {actual_commit}, expected exact base {BASE_COMMIT}"
        )
    sys.path.insert(0, str(legacy_root / "src"))
    encode = _legacy_encoder()
    rich = encode(rich_observation(), 0)
    corpus_rows: list[Mapping[str, Any]] = []
    archives: list[str] = []
    for archive, observation, seat in corpus_observations(corpus):
        if archive not in archives:
            archives.append(archive)
        corpus_rows.append(encode(observation, seat))
    if len(corpus_rows) != CORPUS_LIMIT:
        raise ValueError(f"sampled {len(corpus_rows)} rows, expected {CORPUS_LIMIT}")
    return {
        "provenance": {
            "base_commit": BASE_COMMIT,
            "generator": "tests/feature_golden_generator.py",
            "corpus_limit": CORPUS_LIMIT,
            "archives": archives,
        },
        "rich": _digests([rich]),
        "corpus": _digests(corpus_rows),
    }


def main() -> None:
    """Print reproducible golden JSON for explicit review and check-in."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument(
        "--corpus", type=Path, default=Path("/data/kaggriculture/episodes")
    )
    args = parser.parse_args()
    print(json.dumps(generate(args.legacy_root, args.corpus), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
