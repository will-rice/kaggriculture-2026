"""The campaign's one shared database of programs.

An append-only JSON-lines log, replayed on start and shared by every worker.
Nothing is evicted and there is no population structure: every program a
session produced stays, ranked only by the fitness it was added with, and the
failures are kept beside them so a prompt can show a lineage what has already
been tried and did not work.
"""

import json
from pathlib import Path

from pydantic import BaseModel

from kaggriculture.campaign.evaluator import DeepResult


class Program(BaseModel):
    """One program the campaign kept.

    Attributes:
        id: Unique identifier, also the stored file's stem.
        source_path: Where the source is stored.
        started_from: The id this program was edited from; "" for the seed.
        instruction: The mutation instruction the session was given.
        fitness: Weighted pool win rate from the fast evaluation.
        rates: Fast-evaluation win rate per pool opponent.
        created: Unix timestamp.
        deep: The sealed-block result, once it has one.
    """

    id: str
    source_path: str
    started_from: str
    instruction: str
    fitness: float
    rates: dict[str, float] = {}
    created: float
    deep: DeepResult | None = None


class Failure(BaseModel):
    """One attempt that never became a program.

    Attributes:
        started_from: The id the session was editing.
        instruction: The mutation instruction it was given.
        reason: Why the attempt was rejected.
        created: Unix timestamp.
    """

    started_from: str
    instruction: str
    reason: str
    created: float


class Database:
    """Every program and every failure, backed by an append-only log.

    Each change appends one event to `path` and then folds that same event
    into memory, so the log is the source of truth: constructing a
    `Database` over an existing log replays it into identical state.
    """

    def __init__(self, path: Path, programs_dir: Path) -> None:
        """Open the database at `path`, replaying it if it exists.

        Args:
            path: The JSON-lines log.
            programs_dir: Where `store` writes program sources.
        """
        self.path = path
        self.programs_dir = programs_dir
        self._programs: dict[str, Program] = {}
        self._failures: list[Failure] = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                self._apply(json.loads(line), persist=False)

    def _apply(self, event: dict, persist: bool = True) -> None:
        """Persist one event (unless replaying) and fold it into state.

        Args:
            event: The event to apply.
            persist: Append `event` to the log first. False during replay,
                since the event already exists in the log being read.

        Raises:
            ValueError: The event names a type the database does not know.
        """
        if persist:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
        kind = event["event"]
        if kind == "program":
            program = Program.model_validate(event["program"])
            self._programs[program.id] = program
        elif kind == "deep":
            self.get(event["program_id"]).deep = DeepResult.model_validate(
                event["deep"]
            )
        elif kind == "failure":
            self._failures.append(Failure.model_validate(event["failure"]))
        else:
            raise ValueError(f"unknown database event: {kind!r}")

    @property
    def programs(self) -> list[Program]:
        """Every program, in the order it was added."""
        return list(self._programs.values())

    def store(self, source: str, program_id: str) -> Path:
        """Write `source` to `programs_dir/<program_id>.py` and return its path."""
        self.programs_dir.mkdir(parents=True, exist_ok=True)
        target = self.programs_dir / f"{program_id}.py"
        target.write_text(source, encoding="utf-8")
        return target

    def add(self, program: Program) -> None:
        """Add `program` to the database."""
        self._apply({"event": "program", "program": program.model_dump()})

    def record_failure(self, failure: Failure) -> None:
        """Record an attempt that produced no program."""
        self._apply({"event": "failure", "failure": failure.model_dump()})

    def record_deep(self, program_id: str, result: DeepResult) -> None:
        """Attach the sealed-block result `result` to the program `program_id`."""
        self._apply(
            {"event": "deep", "program_id": program_id, "deep": result.model_dump()}
        )

    def top(self, k: int) -> list[Program]:
        """Return the `k` best programs by fitness, best first."""
        return sorted(self._programs.values(), key=lambda p: p.fitness, reverse=True)[
            :k
        ]

    def get(self, program_id: str) -> Program:
        """Return the program `program_id`.

        Raises:
            KeyError: No program has that id.
        """
        return self._programs[program_id]

    def failures(self, started_from: str) -> list[Failure]:
        """Return the failures recorded against `started_from`, oldest first."""
        return [f for f in self._failures if f.started_from == started_from]
