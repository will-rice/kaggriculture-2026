"""Record CPython and reference-engine values for the Rust golden tests.

Run from the repository root with the project environment::

    uv run python rust/tools/make_golden.py

The output lands in ``rust/tests/data/cpython_golden.json``.
"""

import json
import random
from pathlib import Path

from kaggle_environments.envs.kaggriculture import kaggriculture as engine

SEEDS = [
    0,
    1,
    12345,
    2**31 - 1,
    (2**31 - 1) * 1_000_003 ^ 29,
    2**32,
    2**40 + 7,
    123456789 * 1_000_003 ^ 17,
]

OFFSETS = [
    -20000, -5000, -1000, -500, -400, -332, -300, -200, -122, -105, -100, -50,
    -10, -3, -2, -1, 0, 1, 2, 3, 10, 50, 100, 105, 122, 200, 300, 332, 400, 450,
    500, 1000, 2000, 5000, 20000, 100000, 10**6,
]  # fmt: skip

SHAPE_POINTS = [0.0, 0.5, 1.0, 2.0, 100.0, 199.0, 200.0, 201.0, 250.0, 400.0, 1000.0]


def _rng_record(seed: int) -> dict[str, list[float | int]]:
    words = random.Random(seed)
    floats = random.Random(seed)
    choices = random.Random(seed)
    mixed = random.Random(seed)
    return {
        "words": [words.getrandbits(32) for _ in range(8)],
        "random": [floats.random() for _ in range(6)],
        "choice8": [choices.choice(list(range(8))) for _ in range(12)],
        "mixed": [
            mixed.random(),
            mixed.choice(list(range(8))),
            mixed.random(),
            mixed.choice(list(range(5))),
            mixed.getrandbits(32),
            mixed.random(),
        ],
    }


def main() -> None:
    """Write the golden file."""
    golden = {
        "rng": {str(seed): _rng_record(seed) for seed in SEEDS},
        "prices": {
            item: [
                [10_000 + offset, engine.market_price(item, 10_000 + offset)]
                for offset in OFFSETS
            ]
            for item in engine.PRODUCTS
        },
        "dense": {
            item: [
                engine.market_price(item, inventory) for inventory in range(9000, 11001)
            ]
            for item in engine.PRODUCTS
        },
        "fib": [engine._fib(n) for n in range(15)],
        "sorted_shops": sorted(engine.SHOPS),
        "shape": {
            func: [[x, engine._shape(func, x, 200)] for x in SHAPE_POINTS]
            for func in ["linear", "sq", "sqrt", "log", "log10", "hinge", "bogus"]
        },
    }
    target = (
        Path(__file__).resolve().parent.parent
        / "tests"
        / "data"
        / "cpython_golden.json"
    )
    target.write_text(json.dumps(golden) + "\n")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
