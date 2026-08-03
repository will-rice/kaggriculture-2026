"""Configuration for the agent harness."""

from pydantic import BaseModel

from kaggriculture.constants import BASELINE_AGENTS, EPISODE_STEPS


class HarnessConfig(BaseModel):
    """Configuration for the agent harness."""

    max_workers: int | None = None
    seed: int = 42
    games: int = 8
    opponents: tuple[str, ...] = BASELINE_AGENTS
    episode_steps: int = EPISODE_STEPS
    debug: bool = False
