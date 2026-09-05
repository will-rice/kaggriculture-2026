"""Per-opponent win rates and margins on a fixed seed block, by roster name."""

from pathlib import Path

import pytest

from kaggriculture.campaign import config, field_gate

# Fits the smallest box the suite runs on: CORE_BUDGET clamps to 1 on a
# 4-core CI runner.
WORKERS = min(4, config.CORE_BUDGET)

PASS_AGENT = (
    "def agent(observation, configuration=None):\n"
    "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
)


@pytest.mark.local_data
def test_score_field_returns_a_rate_and_a_margin_per_opponent(
    tmp_path: Path,
) -> None:
    """One win rate and one margin per opponent, both seats, ties as half.

    The margin is what a rate of zero cannot say: PASS loses every game, and
    the size of the gap is the difference between a near miss and a rout.
    """
    agent = tmp_path / "main.py"
    agent.write_text(PASS_AGENT, encoding="utf-8")
    rates, margins = field_gate.score_field(
        agent, seeds=[1, 2], workers=WORKERS, opponents=["v54", "v56"]
    )
    assert set(rates) == set(margins) == {"v54", "v56"}
    # PASS loses every game to a real economy, and by a long way.
    assert all(rate == 0.0 for rate in rates.values())
    assert all(margin.best < 0 for margin in margins.values())
    assert all(
        margin.worst <= margin.mean <= margin.best for margin in margins.values()
    )
