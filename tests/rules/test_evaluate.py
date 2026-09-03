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
    """The engine reads at most ten; a longer queue silently drops the tail."""
    action = {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]] * 10}
    rules = RuleSet(
        rules=(
            RuleSpec(
                name="overflow",
                condition=Condition(field="step", op="eq", value=1),
                effect=Effect(kind="market_insert", order=("HIRE",), at=0),
            ),
        )
    )

    assert len(apply(rules, action=action, observation=OBSERVATION)["market"]) == 10
