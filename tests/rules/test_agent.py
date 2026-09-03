"""A wrapped agent with no rules must play byte-identically to its base."""

from pathlib import Path

from kaggle_environments import make

from kaggriculture.rules.agent import write_rule_agent
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
    """The known day-0 wheat squeeze must reproduce through the rule path.

    This is the positive control: the effect is measured at FIELD 0.6641 ->
    0.9115 when applied by hand, so a rule expressing it must change the game.
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
