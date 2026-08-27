"""Contracts for the typed, persistent hybrid evaluation arena."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from kaggriculture.hybrid.config import HybridConfig, to_runtime
from kaggriculture.search import arena
from kaggriculture.search.arena import (
    GameKey,
    GameResult,
    GameTask,
    HybridOpponent,
    _normalized_margin,
    _win,
    outcomes,
    run_game_task,
)
from kaggriculture.search.arena_pool import PersistentArena

RUNTIME = to_runtime(HybridConfig.default())
ECONOMIC_POLICY = "src/kaggriculture/economic_policy.py"
CANDIDATE = HybridOpponent(RUNTIME)
TASK_A = GameTask(GameKey("econ", 860_000, 0), CANDIDATE, ECONOMIC_POLICY)
TASK_B = GameTask(GameKey("econ", 860_000, 1), CANDIDATE, ECONOMIC_POLICY)


class RecordingExecutor:
    """In-process executor that makes ownership observable without forking."""

    def __init__(self, workers: int) -> None:
        self.workers = workers
        self.shutdown_calls = 0

    def map(
        self, function: Callable[[GameTask], GameResult], tasks: tuple[GameTask, ...]
    ) -> tuple[GameResult, ...]:
        """Synchronously mirror the small part of executor behavior under test."""
        return tuple(function(task) for task in tasks)

    def shutdown(self, **_: object) -> None:
        """Record the pool owner's shutdown request."""
        self.shutdown_calls += 1


class ReorderingExecutor(RecordingExecutor):
    """A broken executor that violates the requested game-key order."""

    def map(
        self, function: Callable[[GameTask], GameResult], tasks: tuple[GameTask, ...]
    ) -> tuple[GameResult, ...]:
        """Return valid rows in reverse provenance order without running games."""
        del function, tasks
        return (
            GameResult(TASK_B.key, 91, 17, 0.0),
            GameResult(TASK_A.key, 91, 17, 0.0),
        )


def recording_executor(
    created: list[RecordingExecutor],
) -> Callable[..., RecordingExecutor]:
    """Build a deterministic executor factory for ownership tests."""
    def factory(*, max_workers: int) -> RecordingExecutor:
        executor = RecordingExecutor(max_workers)
        created.append(executor)
        return executor

    return factory


def test_run_game_task_preserves_exact_provenance_and_candidate_seat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Swapping the candidate seat must preserve candidate-relative banks."""
    monkeypatch.setattr(arena, "_run_banks", lambda left, right, seed: (91, 17))

    result = run_game_task(TASK_A)

    assert result.key == TASK_A.key
    assert (result.ours, result.theirs, result.failure) == (91, 17, None)


def test_persistent_arena_constructs_one_executor_for_multiple_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new executor per call would defeat the arena's persistent contract."""
    from kaggriculture.search import arena_pool

    created: list[RecordingExecutor] = []
    monkeypatch.setattr(
        arena_pool, "ProcessPoolExecutor", recording_executor(created)
    )
    with PersistentArena(workers=3) as pool:
        first = pool.run((TASK_A,))
        second = pool.run((TASK_B,))

    assert [row.key for row in first + second] == [TASK_A.key, TASK_B.key]
    assert len(created) == 1
    assert created[0].workers == 3
    assert created[0].shutdown_calls == 1


@pytest.mark.parametrize(
    ("factory", "message"),
    (
        (lambda: GameKey("", 1, 0), "opponent"),
        (lambda: GameKey("econ", True, 0), "seed"),
        (lambda: GameKey("econ", 1, 2), "seat"),
        (lambda: GameTask("econ", CANDIDATE, ECONOMIC_POLICY), "key"),
        (lambda: GameTask(GameKey("econ", 1, 0), RUNTIME, ECONOMIC_POLICY), "candidate"),
        (lambda: GameTask(GameKey("econ", 1, 0), CANDIDATE, object()), "opponent"),
        (lambda: GameResult("econ", 1, 1, 0.0), "key"),
        (lambda: GameResult(GameKey("econ", 1, 0), None, 1, 0.0), "success"),
        (lambda: GameResult(GameKey("econ", 1, 0), 1, 1, -0.1), "runtime"),
    ),
)
def test_typed_game_provenance_rejects_malformed_rows(
    factory: Callable[[], object], message: str
) -> None:
    """Malformed provenance must fail before it reaches persistent evidence."""
    with pytest.raises(ValueError, match=message):
        factory()


@pytest.mark.parametrize("workers", (0, 33, True, 1.0))
def test_persistent_arena_rejects_workers_outside_the_operational_range(
    workers: object,
) -> None:
    """Only the documented one-through-32 CPU-worker range is accepted."""
    with pytest.raises(ValueError, match="1 and 32"):
        PersistentArena(workers=workers)  # type: ignore[arg-type]


def test_run_game_task_keeps_provenance_when_the_engine_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A terminal-status failure remains attributable to its exact game cell."""

    class Seat:
        def __init__(self, reward: int, status: str) -> None:
            self.reward = reward
            self.status = status

    class Environment:
        steps = [[Seat(1, "ERROR"), Seat(0, "DONE")]]

        def run(self, agents: object) -> None:
            return None

    monkeypatch.setattr(arena, "make", lambda *args, **kwargs: Environment())

    result = run_game_task(TASK_B)

    assert result.key == TASK_B.key
    assert result.ours is None and result.theirs is None
    assert result.failure is not None and "statuses" in result.failure


def test_persistent_arena_matches_legacy_outcomes_for_two_seeds() -> None:
    """The new arena must reproduce legacy candidate scores and margins exactly."""
    seeds = (860_000, 860_001)
    tasks = tuple(
        GameTask(GameKey("econ", seed, seat), CANDIDATE, ECONOMIC_POLICY)
        for seed in seeds
        for seat in (0, 1)
    )

    with PersistentArena(workers=2) as pool:
        rows = pool.run(tasks)
    legacy = outcomes(CANDIDATE, {"econ": ECONOMIC_POLICY}, seeds, workers=2)

    assert not [row for row in rows if row.failure is not None]
    assert [row.key for row in rows] == [task.key for task in tasks]
    assert [_win(row.ours, row.theirs) for row in rows] == legacy
    assert [row.ours - row.theirs for row in rows] == legacy.margins
    assert [_normalized_margin(row.ours, row.theirs) for row in rows] == pytest.approx(
        legacy.normalized_margins
    )


def test_persistent_arena_rejects_duplicate_or_reordered_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input/result provenance is a hard boundary, not an inferred ordering."""
    from kaggriculture.search import arena_pool

    created: list[RecordingExecutor] = []
    monkeypatch.setattr(
        arena_pool, "ProcessPoolExecutor", recording_executor(created)
    )
    with PersistentArena() as pool:
        with pytest.raises(ValueError, match="duplicate provenance"):
            pool.run((TASK_A, TASK_A))


def test_persistent_arena_rejects_reordered_executor_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid but reordered result batch cannot be associated by position."""
    from kaggriculture.search import arena_pool

    created: list[RecordingExecutor] = []

    def factory(*, max_workers: int) -> ReorderingExecutor:
        executor = ReorderingExecutor(max_workers)
        created.append(executor)
        return executor

    monkeypatch.setattr(arena_pool, "ProcessPoolExecutor", factory)
    with PersistentArena() as pool:
        with pytest.raises(RuntimeError, match="results differ"):
            pool.run((TASK_A, TASK_B))

    assert created[0].shutdown_calls == 1
