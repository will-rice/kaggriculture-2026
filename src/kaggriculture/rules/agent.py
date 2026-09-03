"""Write a self-contained agent that applies a rule set over a base agent.

Self-contained because a submission runs where ``/data`` does not exist: the
base agent's source and the evaluator are embedded rather than imported. The
same reason a packaged agent was verified against its gated original before
being submitted -- an import that resolves locally and not on Kaggle reads as
an ordinary loss, not as a crash.
"""

import inspect
from pathlib import Path

from kaggriculture.rules import evaluate, spec
from kaggriculture.rules.spec import RuleSet

TEMPLATE = '''"""Rule-wrapped agent. Generated; do not edit by hand."""

{spec_source}

{evaluate_source}

_BASE_SOURCE = {base_source!r}
_RULES = RuleSet.model_validate_json({rules_json!r})


def _load():
    from kaggle_environments.agent import get_last_callable

    return get_last_callable(_BASE_SOURCE, path="base.py")


_BASE = _load()


def agent(observation, configuration=None):
    """Play the base agent, then apply any rule that fires this turn."""
    try:
        action = _BASE(observation, configuration)
    except TypeError:
        action = _BASE(observation)
    if not isinstance(action, dict):
        return action
    return apply(_RULES, observation, action)
'''


def write_rule_agent(base: Path, rules: RuleSet, target: Path) -> Path:
    """Write a runnable agent applying ``rules`` over ``base``.

    Args:
        base: Path to the base agent's source.
        rules: The rule set to apply.
        target: Where to write the generated agent.

    Returns:
        ``target``, for chaining.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        TEMPLATE.format(
            spec_source=inspect.getsource(spec),
            evaluate_source=inspect.getsource(evaluate).replace(
                "from kaggriculture.rules.spec import Condition, Effect, RuleSet", ""
            ),
            base_source=base.read_text(encoding="utf-8"),
            rules_json=rules.model_dump_json(),
        ),
        encoding="utf-8",
    )
    return target
