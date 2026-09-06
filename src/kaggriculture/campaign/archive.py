"""The campaign's one shared database of programs.

An append-only JSON-lines log, replayed on start and shared by every worker.
Nothing is evicted and there is no population structure: every program a
round produced stays, ranked only by the fitness it was added with, and the
failures are kept beside them: a round that produced no program is not in the
database, so the ledger is the only place it is written down.
"""

import json
from pathlib import Path

from pydantic import BaseModel

from kaggriculture.campaign.evaluator import DeepResult
from kaggriculture.campaign.harness import Margin


def _mean_margin(program: "Program") -> float:
    """Mean bank margin across the opponents a program was measured on.

    The tie-break behind `Database.top`. A program with no margins scores
    zero, which is neither the best nor the worst: a margin is a bank
    difference and runs either side of zero. Only a record written before
    margins existed has none, and this campaign started after they did.
    """
    if not program.margins:
        return 0.0
    return sum(m.mean for m in program.margins.values()) / len(program.margins)


class Program(BaseModel):
    """One program the campaign kept.

    Attributes:
        id: Unique identifier, also the stored file's stem.
        source_path: Where the source is stored.
        started_from: The id this program was edited from; "" for the seed.
        instruction: The mutation instruction the round was given.
        model: The model that wrote it, so a block of quota can be judged
            after the fact. Empty for the seed, which no model wrote.
        fitness: Mean pool win rate from the fast evaluation.
        rates: Fast-evaluation win rate per pool opponent.
        field: Mean win rate over the vendored incumbents alone, which never
            change, so it is comparable across the whole campaign where
            ``fitness`` is not.
        margins: Fast-evaluation bank margin per pool opponent. Defaulted,
            so a program written before margins existed still loads.
        created: Unix timestamp.
        deep: The sealed-block result, once it has one.
    """

    id: str
    source_path: str
    started_from: str
    instruction: str
    model: str
    fitness: float
    field: float
    rates: dict[str, float] = {}
    margins: dict[str, Margin] = {}
    created: float
    deep: DeepResult | None = None


class Failure(BaseModel):
    """One attempt that never became a program.

    Attributes:
        started_from: The id or name the round was editing.
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
        """Fold one event into state and, unless replaying, append it to the log.

        The fold comes first and the line is written only once it has
        succeeded, so an event that cannot be applied can never reach the
        log. That is what makes a restart total: every line on disk is a
        line this same code has already applied, so replaying them all
        rebuilds exactly the state the writer had.

        Args:
            event: The event to apply.
            persist: Append `event` to the log after folding it. False during
                replay, since the event already exists in the log being read.

        Raises:
            ValueError: The event names a type the database does not know, or
                adds a program id the database already holds.
            KeyError: A deep result names a program the database does not hold.
        """
        kind = event["event"]
        if kind == "program":
            program = Program.model_validate(event["program"])
            if program.id in self._programs:
                raise ValueError(f"duplicate program id: {program.id!r}")
            self._programs[program.id] = program
        elif kind == "deep":
            self.get(event["program_id"]).deep = DeepResult.model_validate(
                event["deep"]
            )
        elif kind == "failure":
            self._failures.append(Failure.model_validate(event["failure"]))
        else:
            raise ValueError(f"unknown database event: {kind!r}")
        if persist:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")

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
        """Add `program` to the database.

        Raises:
            ValueError: The database already holds a program with that id.
        """
        self._apply({"event": "program", "program": program.model_dump()})

    def record_failure(self, failure: Failure) -> None:
        """Record an attempt that produced no program."""
        self._apply({"event": "failure", "failure": failure.model_dump()})

    def record_deep(self, program_id: str, result: DeepResult) -> None:
        """Attach the sealed-block result `result` to the program `program_id`.

        Raises:
            KeyError: No program has that id; nothing is written.
        """
        self._apply(
            {"event": "deep", "program_id": program_id, "deep": result.model_dump()}
        )

    def top(self, k: int) -> list[Program]:
        """Return the `k` best programs, best first.

        Ranked on fitness -- the mean over the pool it was measured against
        -- and not on ``field``, which is the mean over the vendored
        incumbents alone and so is the one number comparable across the whole
        campaign. That comparability is what this ranking does not need:
        ``top`` picks who the exam block is spent on, and the exam block asks
        whether a program beats every *current* pool opponent, champions
        included. ``field`` cannot see the champions, which are the hardest
        opponents in the pool. Fitness can, and its one flaw -- an old
        program was measured against a smaller pool -- costs at most one exam
        block per program, because nothing is ever measured twice.

        Then on the mean bank margin across opponents.
        The tie-break is what makes the opening hours a search rather than a
        random walk: until some program wins a game every fitness is 0.0,
        and sorting on fitness alone leaves ties in insertion order, so the
        gate would confirm the first three programs ever written and the
        session draw would pick uniformly from the first ten. The margins
        say which of those zeroes came closest, which is the only signal
        there is before the first win.
        """
        return sorted(
            self._programs.values(),
            key=lambda p: (p.fitness, _mean_margin(p)),
            reverse=True,
        )[:k]

    def children(self, program_id: str) -> list[Program]:
        """Every program written from `program_id`, best first.

        What the next round is told has already been tried from where it
        stands. Ordered like `top`, on fitness and then on the mean bank
        margin, because before the first win every fitness is 0.0 and
        insertion order says nothing about which attempt came closest.
        """
        return sorted(
            (p for p in self._programs.values() if p.started_from == program_id),
            key=lambda p: (p.fitness, _mean_margin(p)),
            reverse=True,
        )

    def get(self, program_id: str) -> Program:
        """Return the program `program_id`.

        Raises:
            KeyError: No program has that id.
        """
        return self._programs[program_id]

    def failures(self, started_from: str) -> list[Failure]:
        """Return the failures recorded against `started_from`, oldest first."""
        return [f for f in self._failures if f.started_from == started_from]
