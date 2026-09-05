"""Per-opponent win rates on a fixed seed block, by roster name, both seats."""

from pathlib import Path

import pytest

from kaggriculture.campaign import field_gate

PASS_AGENT = (
    "def agent(observation, configuration=None):\n"
    "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
)


@pytest.mark.local_data
def test_score_field_returns_one_rate_per_opponent_on_the_given_seeds(
    tmp_path: Path,
) -> None:
    """One win rate per opponent over the given seeds, both seats, ties as half."""
    agent = tmp_path / "main.py"
    agent.write_text(PASS_AGENT, encoding="utf-8")
    rates = field_gate.score_field(
        agent, seeds=[1, 2], workers=4, opponents=["v54", "v56"]
    )
    assert set(rates) == {"v54", "v56"}
    assert all(
        rate == 0.0 for rate in rates.values()
    )  # PASS loses every game to a real economy
