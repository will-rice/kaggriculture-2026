"""The whole loop on a fake mutator and a two-opponent pool, end to end."""

import random
from pathlib import Path

import pytest

from kaggriculture.campaign import config, loop, mutate, pool

PASS = (
    "def agent(observation, configuration=None):\n"
    "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
)
# Wheat costs 10 a seed, waters up to six units between its second and
# fourth day, and sells at 25 a unit: buy five, work one tile, and the
# episode ends four hundred coins above a PASS agent's untouched opening
# money, on every seed and in both seats.
SELLER = """
def agent(observation, configuration=None):
    day = observation["day"]
    farm = observation["farms"][observation["player"]]
    private = observation["private"]
    x, y = farm["farmer"]
    tile = farm["tiles"][y][x]
    market = [["BUY_SEED", "WHEAT", 5]] if observation["step"] == 0 else []
    wheat = private["shed"].get("WHEAT", 0)
    if wheat:
        market.append(["SELL", "WHEAT", wheat])
    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        age = day - tile["planted_day"]
        if age >= 4 and tile["yield_units"]:
            farmer = ["HARVEST"]
        elif not tile["watered_today"]:
            farmer = ["WATER"]
        else:
            farmer = ["PASS"]
    elif tile is None and private["seeds"].get("WHEAT", 0):
        farmer = ["PLANT", "WHEAT"]
    else:
        farmer = ["PASS"]
    return {"farmer": farmer, "hands": [], "market": market}
"""


def test_dry_run_promotes_a_better_child_over_a_pass_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One iteration end to end: mutate, insert, deep-evaluate, promote."""
    run = tmp_path / "run"
    for name in (
        "ARCHIVE",
        "PROGRAMS",
        "SANDBOXES",
        "POOL",
        "EPOCHS",
        "FLOOR",
        "CALLS",
    ):
        monkeypatch.setattr(
            config, name, run / getattr(config, name).relative_to(config.RUN)
        )
    monkeypatch.setattr(config, "EXAM_SEEDS", config.EXAM_SEEDS[:2])
    monkeypatch.setattr(config, "EPOCH_INTERVAL", 1)
    monkeypatch.setattr(config, "ISLANDS", 2)
    monkeypatch.setattr(config, "ISLAND_SIZE", 3)
    monkeypatch.setattr(config, "FAST_SEEDS", 1)
    monkeypatch.setattr(loop.gate, "SERVED", tmp_path / "served.py")
    # The one-opponent test pool stands in for the vendored field, and the
    # real held-out set is dropped: a dry run measures the pipeline, not a
    # field, and playing the held-out opponents would cost games nothing
    # here asserts on.
    monkeypatch.setattr(loop.evaluator, "VENDORED", ["pass"])
    monkeypatch.setattr(loop.evaluator, "HELD_OUT", [])
    pass_agent = tmp_path / "pass.py"
    pass_agent.write_text(PASS)
    seed = tmp_path / "seed.py"
    seed.write_text(PASS)
    p = pool.Pool(opponents={"pass": str(pass_agent)}, weights={"pass": 1.0})
    p.save(config.POOL)
    mutator = mutate.FakeMutator(edit=lambda _: SELLER)
    state = loop.run(
        iterations=1,
        mutator=mutator,
        workers=4,
        concurrency=2,
        seed_agent=seed,
        rng=random.Random(0),
    )
    assert state.champion is not None and state.champion.score > 0.5
    assert (config.FLOOR / "main.py").read_text() == SELLER
    assert "champion_1" in pool.Pool.load(config.POOL).names()
    assert config.EPOCHS.read_text().count("\n") >= 1
