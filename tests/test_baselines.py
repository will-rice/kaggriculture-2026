"""Every league member must play a legal season, or the league lies."""

import pytest
from kaggle_environments import make

from kaggriculture.config import LEAGUE
from kaggriculture.constants import ENVIRONMENT


@pytest.mark.parametrize("baseline", LEAGUE)
def test_league_member_plays_a_legal_season(baseline: str) -> None:
    """A baseline that errors would be scored as a free win for everyone."""
    env = make(ENVIRONMENT, configuration={"episodeSteps": 96, "seed": 3})

    env.run([baseline, "pass"])

    assert [state.status for state in env.steps[-1]] == ["DONE", "DONE"]


def test_heuristic_v1_keeps_no_livestock() -> None:
    """v1 predates the herd; if it grew animals it would not be the v1 we shipped."""
    from baselines.heuristic_v1 import STRATEGY

    assert sum(STRATEGY.herd.values()) == 0


def test_meta_build_matches_the_measured_ladder_build() -> None:
    """The reconstruction is only useful if it is the build 75% of the field plays."""
    from baselines.meta_build import STRATEGY

    assert dict(STRATEGY.crops) == {"WHEAT": 11, "MELON": 11, "STRAWBERRY": 40}
    assert dict(STRATEGY.herd) == {"COW": 8, "SHEEP": 6}
