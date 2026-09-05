"""A port of FAMOU's OpponentPool, checked on its own arithmetic."""

from pathlib import Path

from kaggriculture.campaign import config, pool, roster


def five() -> pool.Pool:
    """A five-opponent pool with equal weights, for tests that mutate it."""
    return pool.Pool(
        opponents={name: f"/x/{name}.py" for name in ["a", "b", "c", "d", "e"]},
        weights=dict.fromkeys(["a", "b", "c", "d", "e"], 0.2),
    )


def test_initial_pool_is_the_training_roster_with_equal_weights() -> None:
    """Pool.initial() mirrors roster.TRAINING with 1/n weights."""
    p = pool.Pool.initial()
    assert p.names() == list(roster.TRAINING)
    assert all(abs(w - 1 / len(roster.TRAINING)) < 1e-12 for w in p.weights.values())


def test_add_champion_gives_it_0_20_and_renormalises() -> None:
    """A champion joins a non-full pool at CHAMPION_WEIGHT; nobody retires."""
    p = five()
    retired = p.add_champion("champ", "/x/champ.py", rates={})
    assert retired is None
    assert abs(sum(p.weights.values()) - 1.0) < 1e-12
    assert abs(p.weights["champ"] - 0.20 / 1.20) < 1e-12
    assert all(abs(p.weights[n] - 0.20 / 1.20) < 1e-12 for n in "abcde")


def test_full_pool_retires_the_lowest_weight_opponent_the_champion_crushes() -> None:
    """A full pool retires the lowest-weight opponent beaten at >= RETIRE_THRESHOLD."""
    p = five()
    for i in range(5):
        p.add_champion(f"c{i}", f"/x/c{i}.py", rates={})
    assert len(p.names()) == config.POOL_CAP
    p.weights["b"] = 0.01
    p.weights["a"] = 0.05
    total = sum(p.weights.values())
    p.weights = {k: v / total for k, v in p.weights.items()}
    retired = p.add_champion("new", "/x/new.py", rates={"a": 0.99, "b": 0.97, "c": 0.5})
    assert retired == "b"
    assert "b" not in p.names() and "new" in p.names()


def test_weakness_pressure_doubles_the_weakest_and_caps_at_half() -> None:
    """Weakness pressure doubles one opponent's weight, capped at WEAKNESS_CAP."""
    p = five()
    p.apply_weakness_pressure("c")
    assert abs(p.weights["c"] - 0.4) < 1e-12
    assert all(abs(p.weights[n] - 0.15) < 1e-12 for n in "abde")
    p.apply_weakness_pressure("c")
    assert abs(p.weights["c"] - 0.5) < 1e-12
    assert abs(sum(p.weights.values()) - 1.0) < 1e-12


def test_weakest_and_weighted() -> None:
    """weakest() picks the lowest rate; weighted() is the weight-dot-rate sum."""
    p = five()
    rates = {"a": 0.9, "b": 0.2, "c": 0.5, "d": 0.7, "e": 0.6}
    assert p.weakest(rates) == "b"
    assert abs(p.weighted(rates) - 0.58) < 1e-12


def test_round_trips_through_json(tmp_path: Path) -> None:
    """save() then load() reproduces the same pool."""
    p = five()
    p.apply_weakness_pressure("a")
    p.save(tmp_path / "pool.json")
    assert pool.Pool.load(tmp_path / "pool.json") == p
