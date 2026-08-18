"""A mutation changes one thing, and leaves a route the engine can still read."""

import random

from kaggriculture.search.mutate import mutate


def _route() -> list[dict]:
    route = [
        {"farmer": ["PASS"], "hands": [["PASS"]], "market": []} for _ in range(720)
    ]
    for turn in range(0, 720, 7):
        route[turn]["market"] = [["SELL", "WHEAT", 3]]
    return route


def _route_with_multi_orders() -> list[dict]:
    """A route with turns carrying more than one order, exercising retime.

    Turn 0 holds two orders with *identical* content -- two `SELL WHEAT 3`
    queued back to back, which a real season plausibly produces. A naive
    index-based swap of that turn produces a route equal to its parent, since
    the two slots read the same either way. Turn 7 holds a genuine pair for
    contrast.
    """
    route = _route()
    route[0]["market"] = [["SELL", "WHEAT", 3], ["SELL", "WHEAT", 3]]
    route[7]["market"] = [["SELL", "WHEAT", 3], ["SELL", "CARROT", 5]]
    return route


def test_a_mutation_changes_exactly_one_turn_even_with_identical_orders() -> None:
    """A retime landing on two identical orders must not emit a no-op.

    Neither `_route` (at most one order per turn) nor the earlier version of
    this fixture exercised the retime swap at all, so a no-op retime on a
    turn of identical orders went uncaught. `mutate` must pick a differing
    pair to swap, or not retime that turn.
    """
    original = _route_with_multi_orders()

    for seed in range(2000):
        mutated, description = mutate(original, random.Random(seed))

        differing = [
            i for i, (a, b) in enumerate(zip(original, mutated, strict=True)) if a != b
        ]
        assert len(differing) == 1, description


def test_a_mutation_changes_exactly_one_turn() -> None:
    """Anything more and a fitness difference cannot be attributed."""
    original = _route()

    for seed in range(40):
        mutated, description = mutate(original, random.Random(seed))

        differing = [
            i for i, (a, b) in enumerate(zip(original, mutated, strict=True)) if a != b
        ]
        assert len(differing) == 1, description
        assert len(mutated) == 720
        assert description


def test_the_mutation_set_can_reach_the_hinge_products() -> None:
    """The search must be able to plant a tomato and buy a goose.

    The town drains 264 tomato a season into a market the whole field supplies
    with 15 units, and capturing that is the reason this search exists. A
    mutation set that cannot express `PLANT:TOMATO` or `BUY_SEED TOMATO` would
    hill-climb forever without reaching it and return a negative that looked
    honest. This asserts the reachability directly rather than trusting the
    constant lists to stay right.
    """
    route = _route()
    route[3]["market"] = [["BUY_SEED", "WHEAT", 2], ["BUY_ANIMAL", "COW", 1]]
    seen = set()

    for seed in range(400):
        mutated, _ = mutate(route, random.Random(seed))
        for turn in mutated:
            seen.add(tuple(turn["farmer"]))
            for order in turn["market"]:
                seen.add(tuple(order[:2]))

    # Engine grammar specifically: ["PLANT", "TOMATO"], not our ["PLANT:TOMATO"].
    assert ("PLANT", "TOMATO") in seen
    assert ("BUY_SEED", "TOMATO") in seen
    assert ("BUY_ANIMAL", "GOOSE") in seen


def test_a_mutation_does_not_alter_its_parent() -> None:
    """The search keeps the incumbent; an in-place edit would corrupt it."""
    original = _route()
    before = [dict(turn) for turn in original]

    mutate(original, random.Random(0))

    assert original == before
