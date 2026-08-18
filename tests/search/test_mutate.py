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

    Turn 14 holds two orders with *identical* content -- two `SELL WHEAT 3`
    queued back to back, which a real season plausibly produces. A naive
    index-based swap of that turn produces a route equal to its parent, since
    the two slots read the same either way. Turn 7 holds a genuine pair for
    contrast.

    Turn 0 is deliberately not used for either case: `mutate` never edits it,
    since `arena._replay` never submits it.
    """
    route = _route()
    route[14]["market"] = [["SELL", "WHEAT", 3], ["SELL", "WHEAT", 3]]
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


def test_buy_product_orders_are_reachable() -> None:
    """BUY_PRODUCT orders must be resizable and retargetable, not frozen.

    A sampled route carries 72 BUY_PRODUCT orders. Without a catalogue entry
    for it, ``resize`` and ``retarget`` can never select one -- it would sit
    in no catalogue at all.
    """
    route = _route()
    route[3]["market"] = [["BUY_PRODUCT", "WHEAT", 2]]
    seen_quantities = set()
    seen_items = set()

    for seed in range(400):
        mutated, _ = mutate(route, random.Random(seed))
        order = mutated[3]["market"][0]
        seen_quantities.add(order[2])
        seen_items.add(order[1])

    assert seen_quantities != {2}
    assert "FERTILIZER" in seen_items


def test_resize_is_stratified_by_verb_like_retarget() -> None:
    """A SELL-dominated route must not drown the lone BUY_SEED resize.

    ``_route`` carries roughly 100 SELL orders (turns 0, 7, 14, ... 714) and
    this test adds exactly one BUY_SEED order. Choosing uniformly over every
    order (the old behaviour) gives that order about a 1-in-100 chance of
    being resized; choosing a verb first, as ``retarget`` already does, gives
    it about a 1-in-2 chance among the verbs actually present. Growing a
    BUY_SEED quantity is the hinge edit the search exists to find, so it must
    be reached often, not almost never.
    """
    route = _route()
    route[3]["market"] = [["BUY_SEED", "TOMATO", 2]]

    buy_seed_resizes = 0
    for seed in range(2000):
        mutated, _ = mutate(route, random.Random(seed))
        if mutated[3]["market"][0][2] != 2:
            buy_seed_resizes += 1

    # Stratified: ~1/3 chance of "resize" * ~1/2 chance of the BUY_SEED verb
    # over 2000 seeds is ~333. Uniform-over-orders would give ~6.5. 200 sits
    # far above the uniform expectation and comfortably below the stratified
    # one.
    assert buy_seed_resizes > 200


def _route_with_malformed_quantity() -> list[dict]:
    """A minimal route with one valid and one malformed SELL order.

    Only two market orders exist in the whole route, so a resize that is not
    filtered against the malformed one has a real chance of landing on it
    inside a small number of seeds.
    """
    route = [
        {"farmer": ["PASS"], "hands": [["PASS"]], "market": []} for _ in range(720)
    ]
    route[2]["market"] = [["SELL", "WHEAT", 3]]
    route[5]["market"] = [["SELL", "WHEAT", None]]
    return route


def test_a_malformed_recorded_quantity_is_skipped_not_crashed_on() -> None:
    """A harvested tape can carry an order the engine recorded but never filled.

    The engine records actions verbatim and only aborts a malformed order at
    execution time, so a route can carry ``['SELL', 'WHEAT', None]``.
    ``kind_slots`` must skip it, or ``resize``'s ``int(order[2])`` crashes a
    multi-hour run the first time the sampler lands on that slot.
    """
    route = _route_with_malformed_quantity()

    for seed in range(500):
        mutate(route, random.Random(seed))  # must not raise


def test_a_mutation_does_not_alter_its_parent() -> None:
    """The search keeps the incumbent; an in-place edit would corrupt it."""
    original = _route()
    before = [dict(turn) for turn in original]

    mutate(original, random.Random(0))

    assert original == before


def test_a_mutation_never_touches_turn_zero() -> None:
    """Turn 0's action is never submitted, so editing it wastes an evaluation.

    ``arena._replay`` serves ``route[observation["step"] + 1]``, and the
    engine's first observation already reads ``step == 0``, so index 0 is
    never the index requested. A mutation set that can still land on turn 0
    would occasionally produce a candidate that scores identically to its
    parent while still costing a full evaluation to discover that.
    """
    route = _route_with_multi_orders()
    route[3]["market"] = [["BUY_SEED", "WHEAT", 2], ["BUY_ANIMAL", "COW", 1]]

    for seed in range(1000):
        mutated, description = mutate(route, random.Random(seed))
        assert mutated[0] == route[0], description


def test_unit_edits_reach_the_hands_not_only_the_farmer() -> None:
    """~90% of a route's labour lives in the hands, and the unit edit must reach it.

    Rewriting only ``action["farmer"]`` would leave every hand op permanently
    unreachable by the search. This asserts a hand op is actually changed
    across many seeds, not merely that hands are present in the route.
    """
    route = _route()
    for turn in route:
        turn["hands"] = [["PASS"], ["PASS"], ["PASS"]]

    hand_edits = 0
    for seed in range(500):
        mutated, description = mutate(route, random.Random(seed))
        for original_turn, mutated_turn in zip(route, mutated, strict=True):
            if original_turn["hands"] != mutated_turn["hands"]:
                hand_edits += 1
                assert original_turn["farmer"] == mutated_turn["farmer"], description

    assert hand_edits > 0
