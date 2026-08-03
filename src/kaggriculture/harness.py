"""Main harness module."""

from kaggriculture.agent import Agent
from kaggriculture.config import HarnessConfig
from kaggriculture.result import Result
from kaggriculture.task import Task


class Harness:
    """Harness for running and evaluating agents against tasks."""

    def __init__(self, config: HarnessConfig | None = None) -> None:
        """Initialize the harness with an optional config."""
        self.config = config or HarnessConfig()
        self.results: list[Result] = []

    def run(self, agent: Agent, tasks: list[Task]) -> list[Result]:
        """Run an agent against a list of tasks and return results."""
        results = []
        for task in tasks:
            try:
                output = agent.run(task)
                score = task.evaluate(output)
                result = Result(
                    task_id=task.id,
                    agent_id=repr(agent),
                    score=score,
                    output=str(output),
                )
            except Exception as e:
                result = Result(
                    task_id=task.id,
                    agent_id=repr(agent),
                    score=0.0,
                    error=str(e),
                )
            results.append(result)
        self.results.extend(results)
        return results
