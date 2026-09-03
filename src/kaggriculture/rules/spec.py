"""The rule schema: what an LLM may propose and what the evaluator may read.

The same model is the LLM's structured-output schema and the evaluator's input,
so a proposal that parses is a proposal the evaluator can run. ``extra="forbid"``
matters more here than usual: an LLM that invents a field would otherwise have it
silently ignored, and the rule would look applied while doing nothing.

``Condition.field`` and ``Condition.op`` are spelled out as explicit ``Literal``
members rather than ``Literal[READABLE_FIELDS]`` / ``Literal[CONDITION_OPS]``:
``ty`` rejects a ``Literal`` parameterized by a tuple variable
(``invalid-type-form``, since a type argument must be a literal value, not an
expression), so the two forms would drift if one were derived from the other by
the type checker alone. ``READABLE_FIELDS`` and ``CONDITION_OPS`` stay as plain
tuple constants -- kept in sync with the ``Literal`` members by
``test_readable_fields_matches_the_condition_field_literal`` -- for the
evaluator and the LLM prompt to import without touching pydantic internals.
"""

import hashlib
import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

LOGGER = logging.getLogger(__name__)

# Every path a condition may read, resolved against the observation the engine
# hands an agent. The opponent's board is here because it is visible at call
# time; their ``private`` is not, because it never is.
READABLE_FIELDS: tuple[str, ...] = (
    "step",
    "day",
    "hour",
    "own.money",
    "own.hands",
    "own.hires_today",
    "opponent.money",
    "opponent.hands",
    "opponent.hires_today",
    "town.shop_count",
)

CONDITION_OPS: tuple[str, ...] = ("eq", "ne", "lt", "le", "gt", "ge")


class Condition(BaseModel):
    """One comparison against a readable field of the observation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    field: Literal[
        "step",
        "day",
        "hour",
        "own.money",
        "own.hands",
        "own.hires_today",
        "opponent.money",
        "opponent.hands",
        "opponent.hires_today",
        "town.shop_count",
    ]
    op: Literal["eq", "ne", "lt", "le", "gt", "ge"]
    value: float


class Effect(BaseModel):
    """What firing the rule does to the base agent's action."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["market_insert", "market_drop"]
    order: tuple[str | int, ...] = ()
    at: int = Field(default=0, ge=0, le=9)


class RuleSpec(BaseModel):
    """A named condition and the effect it applies when every clause holds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    condition: Condition
    effect: Effect


class RuleSet(BaseModel):
    """An ordered set of rules, applied in order, stamped by its digest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rules: tuple[RuleSpec, ...] = ()

    def digest(self) -> str:
        """Return the sha256 of the canonical JSON, for stamping results.

        Returns:
            A 64-character lowercase hex digest.
        """
        return hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest()
