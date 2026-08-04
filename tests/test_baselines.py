"""Every league member must play a legal season, or the league lies."""

from pathlib import Path

import pytest
from kaggle_environments import make

from kaggriculture.config import LEAGUE
from kaggriculture.constants import ENVIRONMENT


@pytest.mark.parametrize("baseline", LEAGUE)
def test_league_member_plays_a_legal_season(baseline: str) -> None:
    """A baseline that errors would be scored as a free win for everyone."""
    path = Path(baseline)
    if path.is_absolute() and not path.exists():
        pytest.skip(f"local replay corpus not present at {baseline}")

    env = make(ENVIRONMENT, configuration={"episodeSteps": 96, "seed": 3})

    env.run([baseline, "pass"])

    assert [state.status for state in env.steps[-1]] == ["DONE", "DONE"]


def test_heuristic_v1_keeps_no_livestock() -> None:
    """v1 predates the herd; if it grew animals it would not be the v1 we shipped."""
    from baselines.heuristic_v1 import STRATEGY

    assert sum(STRATEGY.herd.values()) == 0
