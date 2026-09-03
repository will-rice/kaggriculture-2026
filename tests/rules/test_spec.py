"""The schema is the contract the LLM writes against, so it rejects loudly."""

import typing

import pytest
from pydantic import ValidationError

from kaggriculture.rules.spec import (
    CONDITION_OPS,
    READABLE_FIELDS,
    Condition,
    Effect,
    RuleSet,
    RuleSpec,
)


def test_a_rule_round_trips_through_json() -> None:
    """The schema is the LLM's structured output, so it must parse its own dump."""
    rule = RuleSpec(
        name="yarn_prebuy",
        condition=Condition(field="town.shop_count", op="ge", value=2),
        effect=Effect(kind="market_insert", order=["BUY_PRODUCT", "WHEAT", 51], at=0),
    )

    assert RuleSpec.model_validate_json(rule.model_dump_json()) == rule


def test_an_unknown_field_is_rejected() -> None:
    """extra='forbid' turns an LLM typo into a parse error, not a silent no-op."""
    with pytest.raises(ValidationError):
        RuleSpec.model_validate(
            {
                "name": "typo",
                "condition": {"field": "town.shop_count", "op": "ge", "value": 2},
                "effect": {"kind": "market_insert", "order": ["HIRE"], "at": 0},
                "unexpected": 1,
            }
        )


def test_an_unreadable_condition_field_is_rejected() -> None:
    """A condition may only read state the evaluator can actually resolve.

    Goes through ``model_validate`` (untyped ``dict`` input) rather than the
    keyword constructor: the constructor's ``field`` parameter is typed as the
    ``Literal`` of readable fields, so a static type checker would flag
    ``"opponent.private.seeds"`` as a type error rather than let this test
    exercise the runtime ``ValidationError`` it is actually after.
    """
    with pytest.raises(ValidationError):
        Condition.model_validate(
            {"field": "opponent.private.seeds", "op": "ge", "value": 1}
        )


def test_a_ruleset_digest_is_stable_and_order_sensitive() -> None:
    """The digest stamps results, so it must distinguish two different rule sets."""
    first = RuleSpec(
        name="a",
        condition=Condition(field="step", op="eq", value=0),
        effect=Effect(kind="market_insert", order=["HIRE"], at=0),
    )
    second = RuleSpec(
        name="b",
        condition=Condition(field="step", op="eq", value=1),
        effect=Effect(kind="market_insert", order=["HIRE"], at=0),
    )

    assert (
        RuleSet(rules=(first, second)).digest()
        != RuleSet(rules=(second, first)).digest()
    )
    assert len(RuleSet(rules=(first,)).digest()) == 64


def test_readable_fields_matches_the_condition_field_literal() -> None:
    """The exported tuple constants must stay in sync with the spelled-out Literals.

    ``Condition.field`` and ``Condition.op`` are written as explicit ``Literal``
    members (see the module docstring for why), so nothing at the type level
    keeps them aligned with ``READABLE_FIELDS`` / ``CONDITION_OPS``. This test is
    that guarantee.
    """
    field_literal = typing.get_args(Condition.model_fields["field"].annotation)
    op_literal = typing.get_args(Condition.model_fields["op"].annotation)

    assert field_literal == READABLE_FIELDS
    assert op_literal == CONDITION_OPS


def test_every_condition_op_is_accepted() -> None:
    """Each declared op must actually construct a valid Condition.

    Goes through ``model_validate`` since ``op`` here is a loop variable typed
    as plain ``str``, not the narrower ``Literal`` the keyword constructor
    expects.
    """
    for op in CONDITION_OPS:
        Condition.model_validate({"field": "step", "op": op, "value": 0})


def test_every_readable_field_is_accepted() -> None:
    """Each declared field must actually construct a valid Condition.

    Goes through ``model_validate`` for the same reason as
    ``test_every_condition_op_is_accepted``.
    """
    for field in READABLE_FIELDS:
        Condition.model_validate({"field": field, "op": "eq", "value": 0})
