"""Result data models."""

from pydantic import BaseModel


class Result(BaseModel):
    """Result of running an agent on a task."""

    task_id: str
    agent_id: str
    score: float
    output: str | None = None
    error: str | None = None
    scores: list[float] | None = None

    @property
    def opponent(self) -> str:
        """Return the opponent named by the task this result came from."""
        return self.task_id.rsplit(":", 1)[0]
