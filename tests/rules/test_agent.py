"""A wrapped agent with no rules must play byte-identically to its base."""

import functools
from pathlib import Path

from kaggle_environments import make

from kaggriculture.rules.agent import write_rule_agent
from kaggriculture.rules.arity import accepts_configuration
from kaggriculture.rules.spec import Condition, Effect, RuleSet, RuleSpec
from kaggriculture.search.arena import ENVIRONMENT, EPISODE_STEPS

BASE = Path("/data/kaggriculture/agents/tuned_v58_h7_ratio225.py")
OPPONENT = "/data/kaggriculture/opponents/tetsutani_shopforge/main.py"


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


def test_accepts_configuration_classifies_by_positional_arity() -> None:
    """Classification must key on what a two-positional-argument call needs.

    A downstream call-count test cannot distinguish this classifier from the
    per-turn mismatched-arity retry fallback it replaces: calling a strict
    one-parameter function with two positional arguments raises
    ``TypeError`` at argument-binding time, before the function body runs,
    so a counter inside that function never sees the failed call either way.
    Testing the classification directly is the only way to pin the two
    tricky shapes down: ``*args`` must count as accepting configuration
    (it accepts anything positionally), and a ``functools.partial`` that
    pre-binds ``configuration`` by keyword must not -- ``inspect.signature``
    reports that parameter as keyword-only once bound, and a keyword-only
    parameter cannot take a second positional argument at all.
    """

    def one_argument(observation: object) -> object:
        return observation

    def two_arguments(observation: object, configuration: object = None) -> object:
        return observation, configuration

    def var_positional(*args: object) -> object:
        return args

    def keyword_only_configuration(
        observation: object, *, configuration: object = None
    ) -> object:
        return observation, configuration

    partial_with_bound_configuration = functools.partial(
        two_arguments, configuration="bound"
    )

    assert accepts_configuration(one_argument) is False
    assert accepts_configuration(two_arguments) is True
    assert accepts_configuration(var_positional) is True
    assert accepts_configuration(keyword_only_configuration) is False
    assert accepts_configuration(partial_with_bound_configuration) is False


def test_the_generated_agent_never_retries_a_call_by_argument_count(
    tmp_path: Path,
) -> None:
    """A per-turn retry is the exact bug arity resolution replaced.

    A call-count test cannot tell a working classifier from a retry that
    happens to fail silently before the base's body runs (see the test
    above), so this checks the emitted source directly: nothing in the
    generated agent may catch ``TypeError`` to retry the base with a
    different number of arguments.
    """
    wrapped = write_rule_agent(BASE, RuleSet(), tmp_path / "wrapped.py")

    assert "except TypeError" not in wrapped.read_text(encoding="utf-8")
