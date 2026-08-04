"""Configuration for the agent harness."""

from pydantic import BaseModel

from kaggriculture.constants import EPISODE_STEPS

# Frozen named opponents, strongest first. Paths rather than names because the
# environment loads them as files; the recorded tape is kept alongside the
# reactive reconstruction because the two fail differently — a recording can be
# beaten by exploiting its blindness, and a reactive opponent cannot.
LEAGUE = (
    "baselines/meta_build.py",
    "/data/kaggriculture/baselines/meta_tape.py",
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
