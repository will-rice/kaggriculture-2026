"""The evaluator is what ships, so its behaviour is pinned exactly."""

from kaggriculture.rules.evaluate import apply, matches, read_field
from kaggriculture.rules.spec import Condition, Effect, RuleSet, RuleSpec

OBSERVATION = {
    "step": 1,
    "day": 0,
    "hour": 1,
    "player": 0,
    "farms": [
        {"money": 3000.0, "hands": [], "hires_today": 0},
        {"money": 2500.0, "hands": [[1, 1]], "hires_today": 1},
    ],
    "town": {"shops": ["YARN_STORE", "BAKERY"]},
}
ACTION = {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]]}


def test_reading_our_own_and_the_opponents_money() -> None:
    """`own` and `opponent` resolve by seat, not by index."""
    assert read_field(OBSERVATION, "own.money") == 3000.0
    assert read_field(OBSERVATION, "opponent.money") == 2500.0
    assert read_field(OBSERVATION, "town.shop_count") == 2.0


def test_seat_resolution_flips_with_the_player_index() -> None:
    """`own`/`opponent` must resolve by seat, not by a fixed farm index.

    An agent plays both seats over a match, so a regression to fixed
    indexing (``farms[0]``/``farms[1]``) would be invisible to any test that
    only ever sees ``"player": 0`` -- as every other fixture in this module
    does. This test is the only one that puts the reader in seat 1.
    """
    observation = {**OBSERVATION, "player": 1}

    assert read_field(observation, "own.money") == 2500.0
    assert read_field(observation, "opponent.money") == 3000.0


def test_a_condition_matches_only_when_it_holds() -> None:
    """The comparison is the whole condition language; it must be exact."""
    assert matches(Condition(field="step", op="eq", value=1), OBSERVATION)
    assert not matches(Condition(field="step", op="eq", value=2), OBSERVATION)
    assert matches(Condition(field="opponent.money", op="lt", value=3000), OBSERVATION)


def test_an_empty_ruleset_returns_the_action_unchanged() -> None:
    """No rule firing must be indistinguishable from not wrapping at all.

    This is the property the whole search rests on: a candidate that fires
    nowhere has to reproduce the base agent exactly, or every measured
    difference is confounded by the wrapper itself.
    """
    assert apply(RuleSet(), OBSERVATION, ACTION) == ACTION


def test_apply_never_aliases_the_input_action() -> None:
    """Every mutable field in the result must be a fresh object, always.

    Not just ``market`` (the only field any effect currently touches) and not
    only when a rule fires: a caller that mutated the result assuming it
    owns it would otherwise corrupt the base agent's own action on the path
    where nothing appears to have changed.
    """
    action = {"farmer": ["PASS"], "hands": [["PASS"]], "market": [["HIRE"]]}
    rules = RuleSet(
        rules=(
            RuleSpec(
                name="prebuy",
                condition=Condition(field="step", op="eq", value=1),
                effect=Effect(kind="market_insert", order=("HIRE",), at=0),
            ),
        )
    )

    fired = apply(rules, OBSERVATION, action)
    unfired = apply(RuleSet(), OBSERVATION, action)

    assert fired["farmer"] is not action["farmer"]
    assert fired["hands"] is not action["hands"]
    assert fired["hands"][0] is not action["hands"][0]
    assert fired["market"] is not action["market"]
    assert unfired["farmer"] is not action["farmer"]
    assert unfired["hands"] is not action["hands"]
    assert unfired["market"] is not action["market"]
    assert unfired == action


def test_a_market_insert_places_the_order_at_its_index() -> None:
    """Queue position sets execution order in the engine, so `at` is load-bearing."""
    rules = RuleSet(
        rules=(
            RuleSpec(
                name="prebuy",
                condition=Condition(field="step", op="eq", value=1),
                effect=Effect(
                    kind="market_insert", order=("BUY_PRODUCT", "WHEAT", 51), at=0
                ),
            ),
        )
    )

    result = apply(rules, OBSERVATION, ACTION)

    assert result["market"][0] == ["BUY_PRODUCT", "WHEAT", 51]
    assert result["market"][1] == ["HIRE"]
    assert ACTION["market"] == [["HIRE"]]


def test_the_market_queue_never_exceeds_ten_orders() -> None:
    """An insert past the cap truncates, distinct from never having inserted.

    The fixture starts already at the cap, so an insert genuinely pushes the
    queue to 11 before truncation drops it back to 10. Checking only the
    length is tautological here (10 orders in, 10 orders out, whether or not
    the insert ran); asserting the inserted order's content catches a
    ``market_insert`` that silently became a no-op, which a length-only
    check would not.
    """
    action = {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]] * 10}
    rules = RuleSet(
        rules=(
            RuleSpec(
                name="overflow",
                condition=Condition(field="step", op="eq", value=1),
                effect=Effect(kind="market_insert", order=("PREBUY",), at=0),
            ),
        )
    )

    result = apply(rules, action=action, observation=OBSERVATION)

    assert len(result["market"]) == 10
    assert result["market"][0] == ["PREBUY"]
    assert result["market"].count(["HIRE"]) == 9
