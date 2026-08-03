"""Agent protocol and the episode-playing agent used for local evaluation."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence, runtime_checkable

from kaggle_environments import make

from kaggriculture.constants import ENVIRONMENT
from kaggriculture.task import Task


@runtime_checkable
class Agent(Protocol):
    """Protocol for agents to be evaluated."""

    def run(self, task: Task) -> Sequence[float]:
        """Run the agent on a task and return the output."""
        ...


@dataclass(frozen=True)
class EpisodeAgent:
    """An agent named the way the environment names one.

    ``spec`` is either a built-in agent name ("pass", "random", "starter") or a
    path to a Python file exposing an agent callable. Naming rather than
    embedding the policy keeps the agent picklable for parallel evaluation, and
    means local play exercises the exact ``main.py`` that gets submitted.
    """

    spec: str
    replay_dir: Optional[Path] = None

    def run(self, task: Task) -> list[float]:
        """Play one episode against the task's opponent and return final banks."""
        env = make(
            ENVIRONMENT,
            configuration={"episodeSteps": task.episode_steps, "seed": task.seed},
        )
        env.run([self.spec, task.opponent])
        if self.replay_dir is not None:
            self.replay_dir.mkdir(parents=True, exist_ok=True)
            (self.replay_dir / f"{task.id}.json").write_text(json.dumps(env.toJSON()))
        return [state.reward for state in env.steps[-1]]
