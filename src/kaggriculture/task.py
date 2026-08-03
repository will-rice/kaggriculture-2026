"""Task protocol and the seeded head-to-head match used to score agents."""

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from kaggriculture.constants import EPISODE_STEPS


@runtime_checkable
class Task(Protocol):
    """Protocol for tasks to evaluate agents on.

    Here a task is one episode, so alongside identity and scoring it names the
    episode parameters an agent needs in order to play it.
    """

    @property
    def id(self) -> str:
        """Return the task identifier."""
        ...

    @property
    def opponent(self) -> str:
        """Return the agent to play against."""
        ...

    @property
    def seed(self) -> int:
        """Return the episode seed."""
        ...

    @property
    def episode_steps(self) -> int:
        """Return how many turns the episode runs for."""
        ...

    def evaluate(self, output: Sequence[float]) -> float:
        """Evaluate agent output and return a score between 0 and 1."""
        ...


@dataclass(frozen=True)
class MatchTask:
    """One seeded episode against a named opponent.

    The competition ladder scores wins, not margins, so ``evaluate`` mirrors it:
    the coin difference decides the result and nothing more. The seed fixes weed
    spawns and the town's shop unlock order, so replaying a seed against a
    different opponent compares like with like.
    """

    opponent: str
    seed: int
    episode_steps: int = EPISODE_STEPS

    @property
    def id(self) -> str:
        """Return the task identifier."""
        return f"{self.opponent}:{self.seed}"

    def evaluate(self, output: Sequence[float]) -> float:
        """Return 1.0 for a win, 0.5 for a tie and 0.0 for a loss."""
        ours, theirs = output
        if ours > theirs:
            return 1.0
        return 0.5 if ours == theirs else 0.0
