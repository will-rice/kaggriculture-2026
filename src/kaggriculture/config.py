"""Configuration for the agent harness."""

from pydantic import BaseModel

from kaggriculture.constants import EPISODE_STEPS

# Frozen named opponents, strongest first. Paths rather than names because the
# environment loads them as files. The recorded tape plays the measured ladder
# build open-loop, so it cannot respond to anything we do; heuristic_v1 is a
# genuinely different, reactive shape of opponent (no livestock, one crop, a
# small crew); starter is the environment's own baseline. The tape entry is a
# local artifact — it is not checked into the repo, so it exists only on
# machines that have fetched the replay corpus to that path.
LEAGUE = (
    "/data/kaggriculture/baselines/meta_tape.py",
    "baselines/heuristic_v2.py",
    "baselines/heuristic_v1.py",
    "starter",
)


class HarnessConfig(BaseModel):
    """Configuration for the agent harness."""

    max_workers: int | None = None
    seed: int = 42
    games: int = 8
    opponents: tuple[str, ...] = LEAGUE
    episode_steps: int = EPISODE_STEPS
    debug: bool = False
