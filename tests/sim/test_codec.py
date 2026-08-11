"""Reference-state codec tests.

These tests protect the information that a tensor-only simulator is most
likely to lose: sparse inventory insertion order, duplicate shop draws, and
kind-specific tile attributes.
"""
# ruff: noqa: ANN001, ANN201, ANN202, D103

from copy import deepcopy

from kaggle_environments import make

from kaggriculture.sim.state import pack, unpack
from tests.sim import states


def _environment():
    environment = make("kaggriculture", configuration={"seed": 20260809}, debug=True)
    environment.reset(2)
    return environment


def _public_projection(observation):
    return {
        key: deepcopy(observation[key])
        for key in (
            "step",
            "player",
            "farms",
            "private",
            "market",
            "town",
            "day",
            "hour",
        )
        if key in observation
    }


def test_round_trip_preserves_a_fresh_reference_environment():
    environment = _environment()

    encoded = pack([environment])

    for seat in range(2):
        assert unpack(encoded, 0, seat) == _public_projection(
            environment.state[seat].observation
        )


def test_round_trip_preserves_inventory_order_shops_and_tile_attributes():
    environment = _environment()
    public = environment.state[0].observation
    private = environment.state[0].observation.private
    public.farms[0].tiles[1][2] = {
        "kind": "PLANT",
        "crop": "CARROT",
        "planted_day": 3,
        "watered_today": True,
        "consecutive_unwatered": 0,
        "yield_units": 4,
        "max_lifespan_step": 216,
        "fertilized_until_day": 7,
    }
    public.farms[1].tiles[3][4] = {
        "kind": "COOP",
        "animal": "GOOSE",
        "placed_day": 2,
        "yield_units": 5,
        "consecutive_unfed": 1,
        "fed_today": False,
        "cared_today": True,
        "fertilizer_available": True,
        "pending_care_bonus": 3,
    }
    public.farms[0].hands = [[3, 4]]
    private.inventories = [{"MILK": 2, "WHEAT": 1}, {"WOOL": 3, "EGG": 4}]
    public.town.unlocked_shops = ["BAKERY", "BAKERY", "YARN_STORE"]

    encoded = pack([environment])
    restored = unpack(encoded, 0, 0)

    assert restored == _public_projection(environment.state[0].observation)
    assert list(restored["private"]["inventories"][0].items()) == [
        ("MILK", 2),
        ("WHEAT", 1),
    ]
    assert restored["town"]["unlocked_shops"] == ["BAKERY", "BAKERY", "YARN_STORE"]


def test_pack_accepts_raw_two_seat_reference_state_with_an_explicit_seed():
    environment = _environment()

    encoded = pack([(environment.state, environment.info["seed"])])

    assert encoded.seed.tolist() == [environment.info["seed"]]
    assert unpack(encoded, 0, 1) == _public_projection(environment.state[1].observation)


def test_every_deliberate_state_builder_round_trips() -> None:
    builders = [
        states.shed_at_capacity(holding={"MILK": 2}),
        states.shed_one_below_capacity(),
        states.money_exactly(25, for_order=["BUY_SEED", "WHEAT", 1]),
        states.price_at_floor("WHEAT"),
        states.unit_on_locked_shed_tile(),
        states.fully_masked_unit(),
        states.occupied_structure("GOOSE", fed=True, cared=False),
        states.crop_at_age("WHEAT", 3, fertilized=True, watered=False),
        states.hire_ladder(hires_today=4, money=100),
        states.land_at_stage(2),
        states.town_with_shops(["BAKERY", "BAKERY"]),
        states.last_turn(),
        states.multi_item_inventory(["MILK", "WHEAT", "WOOL"]),
    ]
    for environment in builders:
        encoded = pack([environment])
        for seat in range(2):
            assert unpack(encoded, 0, seat) == _public_projection(
                environment.state[seat].observation
            )
