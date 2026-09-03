"""Apply a rule set to one turn's action. This is the part that ships.

Kept allocation-light, because it runs 719 times per episode inside a
one-second-per-step budget and travels into the submission archive. It does
import ``kaggriculture.rules.spec`` (and, through it, pydantic) for the
schema types -- there is no dependency free of that, only free of anything
heavier.
"""

import operator
from collections.abc import Mapping
from typing import Any

from kaggriculture.rules.spec import Condition, Effect, RuleSet

MAX_MARKET_ORDERS = 10

_OPS = {
    "eq": operator.eq,
    "ne": operator.ne,
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
}


def read_field(observation: Mapping[str, Any], field: str) -> float:
    """Resolve one readable field to a number.

    Args:
        observation: The observation the engine hands the agent.
        field: One of ``spec.READABLE_FIELDS``.

    Returns:
        The field's value as a float.
    """
    if field in ("step", "day", "hour"):
        return float(observation[field])
    if field == "town.shop_count":
        return float(len(observation.get("town", {}).get("shops", ())))
    side, _, leaf = field.partition(".")
    seat = int(observation.get("player", 0))
    farms = observation["farms"]
    farm = farms[seat] if side == "own" else farms[1 - seat]
    if leaf == "hands":
        return float(len(farm.get("hands", ())))
    return float(farm[leaf])


def matches(condition: Condition, observation: Mapping[str, Any]) -> bool:
    """Return whether the condition holds for this observation.

    Args:
        condition: The comparison to evaluate.
        observation: The observation the engine hands the agent.

    Returns:
        True when the comparison holds.
    """
    return bool(
        _OPS[condition.op](read_field(observation, condition.field), condition.value)
    )


def apply(
    rules: RuleSet, observation: Mapping[str, Any], action: dict[str, Any]
) -> dict[str, Any]:
    """Return a new action with every firing rule's effect applied, in order.

    Every mutable field -- ``farmer``, each order in ``hands``, and
    ``market`` -- is copied into a fresh list before anything runs, on every
    call, whether or not a rule fires. Nothing in the return value ever
    aliases the input: the base agent may hold a reference to what it
    returned, and a rule -- or a future effect kind not yet imagined --
    mutating a shared list in place would corrupt the agent's own state
    between turns. A shallower copy (e.g. only copying ``market``, or only
    copying when a rule fires) would be correct today, since only
    ``market_insert``/``market_drop`` exist and both already build fresh
    lists rather than mutate a shared one; it would silently stop being
    correct the day an effect kind that touches ``farmer`` or ``hands`` is
    added. Copying unconditionally removes that failure mode rather than
    documenting and re-checking it.

    Args:
        rules: The rule set to apply.
        observation: The observation the engine hands the agent.
        action: The base agent's action for this turn.

    Returns:
        A new action dict, equal to the input when no rule fires.
    """
    updated: dict[str, Any] = {
        "farmer": list(action["farmer"]),
        "hands": [list(order) for order in action["hands"]],
        "market": list(action["market"]),
    }
    for rule in rules.rules:
        if matches(rule.condition, observation):
            updated["market"] = _apply_effect(rule.effect, updated["market"])
    updated["market"] = updated["market"][:MAX_MARKET_ORDERS]
    return updated


def _apply_effect(effect: Effect, market: list[Any]) -> list[Any]:
    """Return the market queue with one effect applied.

    Args:
        effect: The effect to apply.
        market: The market queue so far, mutated in place and returned.

    Returns:
        The market queue after the effect.
    """
    if effect.kind == "market_insert":
        market.insert(effect.at, list(effect.order))
        return market
    if effect.at < len(market):
        del market[effect.at]
    return market
