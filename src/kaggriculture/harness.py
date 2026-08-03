"""Main harness module."""

from concurrent.futures import ProcessPoolExecutor

from kaggriculture.agent import Agent
from kaggriculture.config import HarnessConfig
from kaggriculture.result import Result
from kaggriculture.task import MatchTask, Task


class Harness:
    """Harness for running and evaluating agents against tasks."""

    def __init__(self, config: HarnessConfig | None = None) -> None:
        """Initialize the harness with an optional config."""
        self.config = config or HarnessConfig()
        self.results: list[Result] = []

    def run(self, agent: Agent, tasks: list[Task]) -> list[Result]:
        """Run an agent against a list of tasks and return results.

        Episodes are CPU-bound pure Python, so they are fanned out across
        processes rather than threads.
        """
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as pool:
            results = list(pool.map(run_task, [agent] * len(tasks), tasks))
        self.results.extend(results)
        return results

    def matches(self) -> list[Task]:
        """Return one match per opponent per seeded game, from the config."""
        return [
            MatchTask(
                opponent=opponent, seed=seed, episode_steps=self.config.episode_steps
            )
            for opponent in self.config.opponents
            for seed in range(self.config.seed, self.config.seed + self.config.games)
        ]


def run_task(agent: Agent, task: Task) -> Result:
    """Run one agent-task pair and capture the outcome as a ``Result``."""
    try:
        output = agent.run(task)
        return Result(
            task_id=task.id,
            agent_id=repr(agent),
            score=task.evaluate(output),
            output=str(output),
            scores=list(output),
        )
    except Exception as error:  # noqa: BLE001 - one bad episode must not stop a sweep
        return Result(
            task_id=task.id, agent_id=repr(agent), score=0.0, error=str(error)
        )
