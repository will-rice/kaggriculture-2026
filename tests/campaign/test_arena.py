"""Reference-engine games between two agent files."""

from pathlib import Path

from kaggriculture.campaign import arena

PASS_AGENT = """
def agent(observation, configuration=None):
    return {"farmer": ["PASS"], "hands": [], "market": []}
"""


def test_pass_versus_pass_banks_the_starting_money(tmp_path: Path) -> None:
    """Two agents that never trade end the season holding what they started with."""
    agent = tmp_path / "pass.py"
    agent.write_text(PASS_AGENT, encoding="utf-8")
    assert arena.run_banks(str(agent), str(agent), 1) == (3000, 3000)


def test_outcomes_plays_both_seats_and_scores_ties_as_half(tmp_path: Path) -> None:
    """Every seed is played with the candidate in each seat, ties worth 0.5."""
    agent = tmp_path / "pass.py"
    agent.write_text(PASS_AGENT, encoding="utf-8")
    scores = arena.outcomes(str(agent), {"mirror": str(agent)}, [1, 2], workers=2)
    assert list(scores) == [0.5, 0.5, 0.5, 0.5]
    assert arena.summarize(scores, {"mirror": str(agent)}) == {"mirror": 0.5}


def test_a_crashing_agent_is_a_failure_not_a_loss(tmp_path: Path) -> None:
    """A raising agent must not silently bank its untouched 3000 as a loss."""
    good = tmp_path / "pass.py"
    good.write_text(PASS_AGENT, encoding="utf-8")
    bad = tmp_path / "bad.py"
    bad.write_text(
        "def agent(o, c=None):\n    raise RuntimeError('boom')\n", encoding="utf-8"
    )
    scores = arena.outcomes(str(bad), {"pass": str(good)}, [1], workers=1)
    assert scores == [] or all(s is None for s in scores)
    assert scores.failures and "seat" in scores.failures[0]


def test_a_failing_league_member_does_not_shift_the_next_members_scores(
    tmp_path: Path,
) -> None:
    """A crashing member's failures must not steal a later member's slice."""
    good = tmp_path / "pass.py"
    good.write_text(PASS_AGENT, encoding="utf-8")
    bad = tmp_path / "bad.py"
    bad.write_text(
        "def agent(o, c=None):\n    raise RuntimeError('boom')\n", encoding="utf-8"
    )
    league = {"crashes": str(bad), "pass": str(good)}
    scores = arena.outcomes(str(good), league, [1], workers=1)
    summary = arena.summarize(scores, league)
    assert summary["pass"] == 0.5
    assert summary["crashes"] == 0.0
    assert len(scores.failures) == 2
