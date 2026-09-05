"""The wrapper renders what the reference engine renders."""

from kaggle_environments import make

from kaggriculture.campaign.engine.wrapper import Engine, pack_action
from kaggriculture.constants import ENVIRONMENT, EPISODE_STEPS

PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def strip(observation: dict) -> dict:
    """The framework adds remainingOverageTime; the engine does not know it."""
    return {k: v for k, v in observation.items() if k != "remainingOverageTime"}


def reference_observations(seed: int, steps: int) -> list[list[dict]]:
    """Return the reference engine's per-player observation after each PASS step."""
    environment = make(
        ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS, "seed": seed}
    )
    environment.reset()
    out = []
    for _ in range(steps):
        environment.step([PASS, PASS])
        out.append([strip(dict(state.observation)) for state in environment.state])
    return out


def test_initial_observation_matches_the_reference_engine() -> None:
    """A fresh episode renders exactly what the framework hands each player."""
    engine = Engine(seed=7)
    environment = make(
        ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS, "seed": 7}
    )
    environment.reset()
    for player in (0, 1):
        assert engine.observation(player) == strip(
            dict(environment.state[player].observation)
        )


def test_thirty_pass_steps_match_including_an_end_of_day() -> None:
    """Thirty PASS steps cross a day boundary, so weeds and the daily reset show up."""
    engine = Engine(seed=7)
    expected = reference_observations(7, 30)
    for step in range(30):
        engine.step(PASS, PASS)
        for player in (0, 1):
            assert engine.observation(player) == expected[step][player], (
                f"step {step} player {player}"
            )


def test_pack_action_maps_names_to_the_port_enums() -> None:
    """Op and item names become the wire values sim.hpp's enums assign them."""
    packed = pack_action(
        {
            "farmer": ["PLANT", "MELON"],
            "hands": [["PICKUP", "WHEAT", 3], ["NORTH"], ["NONSENSE"]],
            "market": [
                ["HIRE"],
                ["BUY_SEED", "WHEAT", 5],
                ["SELL", "WOOL", 2],
                ["BUY_LAND"],
                ["BOGUS", "X", 1],
            ],
        }
    )
    assert packed.n_units == 4
    # PLANT, PICKUP, NORTH, PASS (an unknown op is a no-op, like PASS)
    assert list(packed.unit_ops[:4]) == [8, 5, 1, 0]
    assert list(packed.unit_args[:2]) == [4, 0]  # MELON, WHEAT
    assert list(packed.unit_ns[:2]) == [1, 3]
    assert packed.n_orders == 4  # the bogus order is dropped, as _parse_order drops it
    assert list(packed.order_ops[:4]) == [1, 3, 6, 2]
    assert list(packed.order_items[1:3]) == [0, 7]
    assert list(packed.order_ns[1:3]) == [5, 2]


def test_pack_action_keeps_the_lockstep_slot_of_a_dropped_order() -> None:
    """A malformed order is a hole, not a gap: later orders keep their index.

    ``_process_market`` walks both players' queues by index, so compacting the
    queue would pair a player's order against the wrong opponent order.
    """
    packed = pack_action(
        {"market": [["SELL"], ["SELL", "WOOL", 2]]}  # first order is malformed
    )
    assert packed.n_orders == 2
    assert list(packed.order_ops[:2]) == [0, 6]  # NONE, SELL
    assert packed.order_items[1] == 7 and packed.order_ns[1] == 2


def test_pack_action_makes_an_unknown_item_a_no_op() -> None:
    """The port no-ops an out-of-range item, as the reference no-ops an unknown name."""
    packed = pack_action({"farmer": ["PICKUP", "SPACESHIP", 2]})
    assert packed.unit_ops[0] == 5 and packed.unit_args[0] == 12


def test_bank_is_the_money_the_reference_engine_reports() -> None:
    """A PASS episode ends on the starting bank, at the final recorded step."""
    engine = Engine(seed=3)
    while not engine.done:
        engine.step(PASS, PASS)
    assert engine.bank(0) == 3000.0 and engine.bank(1) == 3000.0
    assert engine.step_index == EPISODE_STEPS - 1
