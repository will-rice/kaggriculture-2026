"""The Python bindings against the installed reference and the Rust CLI."""
# ruff: noqa: D103

import copy
import random

import pytest
from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as reference

from tests.rust.policies import biased_random, husbandry
from tests.rust.test_differential import VARIANT_CONFIGURATION, _compare, _expected

ke = pytest.importorskip("kaggriculture_engine", reason="build rust/python first")


def _play_both(configuration: dict, policy, turns: int) -> None:  # noqa: ANN001
    environment = make("kaggriculture", configuration=dict(configuration), debug=True)
    environment.reset(2)
    engine = ke.Engine(configuration)
    assert engine.seed == environment.info["seed"]
    for turn in range(turns):
        if environment.done:
            break
        actions = [policy(environment.state[seat].observation) for seat in range(2)]
        environment.step(actions)
        engine.step(actions)
        for seat in range(2):
            expected = _expected(environment, seat)
            actual = engine.observation(seat)
            if "step" not in expected:
                actual.pop("step")
            _compare(expected, actual, f"turn {turn} seat[{seat}]")
        assert engine.done == all(
            str(agent.status) == "DONE" for agent in environment.state
        )
    assert engine.rewards == [float(agent.reward) for agent in environment.state]


def test_bindings_follow_the_reference_under_a_biased_random_policy() -> None:
    rng = random.Random(11)
    _play_both({"seed": 401}, lambda obs: biased_random(rng, obs), 719)


def test_bindings_follow_the_reference_under_the_husbandry_script() -> None:
    _play_both({"seed": 402}, husbandry, 719)


def test_bindings_follow_the_reference_under_a_variant_configuration() -> None:
    rng = random.Random(12)
    _play_both(
        {"seed": 403, **VARIANT_CONFIGURATION}, lambda obs: biased_random(rng, obs), 200
    )


def test_state_round_trips_and_copies_are_independent() -> None:
    engine = ke.Engine(seed=7)
    for _ in range(30):
        engine.step([ke.starter_agent(engine, 0), None])
    snapshot = engine.state()
    resumed = ke.Engine.from_state(snapshot, seed=engine.seed)
    assert resumed.state() == snapshot
    assert resumed.observations() == engine.observations()

    clone = copy.deepcopy(engine)
    engine.step([ke.starter_agent(engine, 0), None])
    assert clone.step_count == 30
    assert engine.step_count == 31
    clone.step([ke.starter_agent(clone, 0), None])
    assert clone.observations() == engine.observations()

    engine.reset()
    assert engine.step_count == 0
    assert engine.money == [3000.0, 3000.0]


def test_random_stream_matches_cpython() -> None:
    for seed in (0, 1, 2**31 - 1, 12345 * 1_000_003 ^ 29):
        ours, theirs = ke.Random(seed), random.Random(seed)
        for _ in range(50):
            assert ours.random() == theirs.random()
            assert ours.getrandbits(32) == theirs.getrandbits(32)
            assert ours.choice(range(8)) == theirs.choice(range(8))
    day = ke.Random.for_day(5, 3)
    assert day.random() == random.Random((5 * 1_000_003) ^ 3).random()


def test_market_price_and_constants_match_the_reference() -> None:
    assert ke.PRODUCTS == list(reference.PRODUCTS)
    assert ke.CROPS == list(reference.CROPS)
    assert ke.ANIMALS == list(reference.ANIMALS)
    assert ke.SHOPS == sorted(reference.SHOPS)
    for item in reference.PRODUCTS:
        for inventory in (-500, 0, 9_000, 9_999, 10_000, 10_001, 12_000, 10**6):
            assert ke.market_price(item, inventory) == reference.market_price(
                item, inventory
            )
    with pytest.raises(ValueError, match="not a market product"):
        ke.market_price("GOOSE", 10_000)


def test_agents_match_the_reference_agents() -> None:
    engine = ke.Engine(seed=9)
    environment = make("kaggriculture", configuration={"seed": 9}, debug=True)
    environment.reset(2)
    for _ in range(100):
        observation = environment.state[0].observation
        assert ke.starter_agent(engine, 0) == reference.starter_agent(observation)
        assert ke.pass_agent(engine, 1) == reference.pass_agent(observation)
        action = reference.starter_agent(observation)
        environment.step([action, reference.pass_agent(observation)])
        engine.step([action, None])
    rng = ke.Random(1)
    action = ke.random_agent(engine, 1, rng)
    assert set(action) == {"farmer", "hands", "market"}


def test_malformed_input_is_tolerated_like_the_reference() -> None:
    engine = ke.Engine(seed=3)
    engine.step(None)
    engine.step([])
    engine.step(["nonsense", 42])
    engine.step(
        [{"farmer": "WATER", "hands": 3, "market": [["SELL", "WHEAT", "x"]]}, None]
    )
    assert engine.step_count == 4
    with pytest.raises(IndexError):
        engine.observation(2)
    with pytest.raises(ValueError, match="configuration must be a mapping"):
        ke.Engine([1, 2])
    with pytest.raises(ValueError):
        ke.Engine.from_state({"farms": []})
