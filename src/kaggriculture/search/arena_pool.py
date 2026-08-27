"""One context-managed process pool for a whole hybrid evaluation arena."""

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Self

from kaggriculture.search.arena import GameResult, GameTask, run_game_task


class PersistentArena:
    """Own exactly one process pool across one or more batches of game tasks."""

    def __init__(self, workers: int = 32) -> None:
        if type(workers) is not int or not 1 <= workers <= 32:
            raise ValueError("workers must be between 1 and 32")
        self.workers = workers
        self._pool: ProcessPoolExecutor | None = None

    def __enter__(self) -> Self:
        """Create the sole executor owned by this arena instance."""
        if self._pool is not None:
            raise RuntimeError("persistent arena is already open")
        self._pool = ProcessPoolExecutor(max_workers=self.workers)
        return self

    def run(self, tasks: Sequence[GameTask]) -> tuple[GameResult, ...]:
        """Run a batch in input order after validating its provenance."""
        if self._pool is None:
            raise RuntimeError("persistent arena is not open")
        keys = tuple(task.key for task in tasks)
        if len(keys) != len(set(keys)):
            raise ValueError("game tasks contain duplicate provenance")
        rows = tuple(self._pool.map(run_game_task, tasks))
        if tuple(row.key for row in rows) != keys:
            raise RuntimeError("arena results differ from requested provenance")
        return rows

    def __exit__(self, *_: object) -> None:
        """Close the owned executor exactly once at context exit."""
        assert self._pool is not None
        self._pool.shutdown(wait=True, cancel_futures=True)
        self._pool = None
