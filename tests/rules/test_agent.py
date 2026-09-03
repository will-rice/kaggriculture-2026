"""A wrapped agent with no rules must play byte-identically to its base."""

import importlib.util
from pathlib import Path

from kaggle_environments import make

from kaggriculture.rules.agent import write_rule_agent
from kaggriculture.rules.spec import Condition, Effect, RuleSet, RuleSpec
from kaggriculture.search.arena import ENVIRONMENT, EPISODE_STEPS

BASE = Path("/data/kaggriculture/agents/tuned_v58_h7_ratio225.py")
OPPONENT = "/data/kaggriculture/opponents/tetsutani_shopforge/main.py"

FAKE_BASE_SOURCE = '''"""A stateful, single-argument fake base agent."""

_CALLS = []


def agent(observation):
    _CALLS.append(1)
    return {"farmer": [], "hands": [], "market": []}
'''


def _banks(agent: str, seed: int) -> tuple[float, float]:
    """Play one season and return both seats' final banks."""
    environment = make(
        ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS, "seed": seed}
    )
    environment.run([agent, OPPONENT])
    final = environment.steps[-1]
    return float(final[0].reward), float(final[1].reward)


def test_an_empty_ruleset_reproduces_the_base_agent_exactly(tmp_path: Path) -> None:
    """Any measured difference must come from the rules, not the wrapper."""
    wrapped = write_rule_agent(BASE, RuleSet(), tmp_path / "wrapped.py")

    assert _banks(str(wrapped), 700000) == _banks(str(BASE), 700000)


def test_a_firing_rule_changes_the_season(tmp_path: Path) -> None:
    """A rule that fires must demonstrably change the season.

    This is a causal control, not a strategy: the rule expresses only the
    day-0 pre-buy half of the known wheat squeeze, with no sellback, which
    is ruinous on its own (measured: 7,473 vs. the base's 72,067 at this
    seed). A ruinous one-sided rule is one token away from the winning
    two-sided one an LLM might propose, so the assertion here is only that
    the rule path is wired up and changes the outcome -- not that this
    particular rule is good.
    """
    rules = RuleSet(
        rules=(
            RuleSpec(
                name="prebuy",
                condition=Condition(field="step", op="eq", value=0),
                effect=Effect(
                    kind="market_insert", order=("BUY_PRODUCT", "WHEAT", 51), at=0
                ),
            ),
        )
    )
    wrapped = write_rule_agent(BASE, rules, tmp_path / "squeeze.py")

    assert _banks(str(wrapped), 700000) != _banks(str(BASE), 700000)


def test_a_single_argument_base_is_invoked_exactly_once_per_turn(
    tmp_path: Path,
) -> None:
    """A per-turn try/except would call a stateful single-argument base twice.

    The real base agents carry state across turns (module-level tallies
    mutated on every call), so a wrapper that retries a ``TypeError`` by
    calling the base again -- once raising partway through, once succeeding
    -- would corrupt that state on the very first turn. Arity must be
    resolved once, at load time, so no per-turn retry can exist to fire
    twice.
    """
    fake_base = tmp_path / "fake_base.py"
    fake_base.write_text(FAKE_BASE_SOURCE, encoding="utf-8")
    wrapped = write_rule_agent(fake_base, RuleSet(), tmp_path / "wrapped.py")

    module_spec = importlib.util.spec_from_file_location("wrapped_agent", wrapped)
    assert module_spec is not None
    loader = module_spec.loader
    assert loader is not None
    module = importlib.util.module_from_spec(module_spec)
    loader.exec_module(module)

    module.agent({"step": 0}, {"episodeSteps": EPISODE_STEPS})

    assert len(module._BASE.__globals__["_CALLS"]) == 1
