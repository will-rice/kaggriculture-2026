"""The population: islands of agent programs and the log that is their memory.

Every state change is one appended JSON line; loading replays the log. UCB
picks parents, a full island replaces its worst, migration copies the top
programs around a ring, and a reset reseeds the worst island from the
champion — FAMOU's island model at our scale.
"""

import json
import math
import random
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from kaggriculture.campaign import config

Kind = Literal["seed", "full", "cross", "migrant", "reset"]


class Program(BaseModel):
    """One agent program tracked by the archive.

    Attributes:
        id: Unique identifier.
        island: Index of the island this program lives on.
        source_path: Path to the stored source file.
        parents: Ids of the program(s) this one was derived from.
        kind: How this program was produced.
        fitness_sum: Sum of fitness across all deep evaluations so far.
        n_evals: Number of deep evaluations contributing to `fitness_sum`.
        status: Free-form status ("ok", "failed", ...).
        reason: Free-form explanation, e.g. a failure reason.
        created: Unix timestamp of creation.
        rates: Per-opponent fast-eval win rates, written by callers and
            simply persisted through the log.
    """

    id: str
    island: int
    source_path: str
    parents: list[str]
    kind: Kind
    fitness_sum: float
    n_evals: int
    status: str
    reason: str
    created: float
    rates: dict[str, float] = {}

    @property
    def mean(self) -> float:
        """Mean fitness across deep evaluations, or 0.0 with none yet."""
        return self.fitness_sum / self.n_evals if self.n_evals else 0.0


class Archive:
    """The population of islands, backed by an append-only JSON-lines log.

    Every mutation is applied by appending one event to `path` and then
    replaying that same event onto in-memory state, so the log is always
    the source of truth: `Archive(path=...)` rebuilds identical state by
    replaying it from empty.
    """

    def __init__(
        self, path: Path = config.ARCHIVE, programs_dir: Path = config.PROGRAMS
    ) -> None:
        self.path = path
        self.programs_dir = programs_dir
        self._islands: list[dict[str, Program]] = [{} for _ in range(config.ISLANDS)]
        self._failures: list[Program] = []
        # UCB selection counts are per-process exploration state, not part
        # of the archive's persisted identity, so they are never logged.
        self._selections = 0
        self._picks: dict[str, int] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                self._apply(json.loads(line), persist=False)

    def _apply(self, event: dict, persist: bool = True) -> None:
        """Persist one event (unless replaying) and fold it into state.

        Args:
            event: The event to apply.
            persist: Append `event` to the log first. False during replay,
                since the event already exists in the log being read.
        """
        if persist:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
        kind = event["event"]
        if kind == "insert":
            program = Program.model_validate(event["program"])
            self._islands[program.island][program.id] = program
        elif kind == "remove":
            self._islands[event["island"]].pop(event["id"], None)
        elif kind == "eval":
            program = self._find(event["id"])
            program.fitness_sum += event["fitness"]
            program.n_evals += 1
        elif kind == "failure":
            self._failures.append(Program.model_validate(event["program"]))
        else:
            raise ValueError(f"unknown archive event: {kind!r}")

    def _find(self, program_id: str) -> Program:
        for island in self._islands:
            if program_id in island:
                return island[program_id]
        raise KeyError(program_id)

    def island(self, i: int) -> list[Program]:
        """Return every program on island `i`."""
        return list(self._islands[i].values())

    def failures(self) -> list[Program]:
        """Return every recorded failure, in the order they were logged."""
        return list(self._failures)

    def top(self, k: int) -> list[Program]:
        """Return the `k` best programs across all islands by mean fitness.

        Programs with `n_evals == 0` (never deep-evaluated) are excluded.
        """
        everyone = [p for island in self._islands for p in island.values() if p.n_evals]
        return sorted(everyone, key=lambda p: p.mean, reverse=True)[:k]

    def store(self, source: str, program_id: str) -> Path:
        """Write `source` to `PROGRAMS/<program_id>.py` and return its path."""
        self.programs_dir.mkdir(parents=True, exist_ok=True)
        target = self.programs_dir / f"{program_id}.py"
        target.write_text(source, encoding="utf-8")
        return target

    def seed(self, source: Path, fitness: float) -> Program:
        """Place one copy of `source` on every island as kind "seed"."""
        text = source.read_text(encoding="utf-8")
        first: Program | None = None
        for island in range(config.ISLANDS):
            program_id = f"seed-{island}"
            path = self.store(text, program_id)
            program = Program(
                id=program_id,
                island=island,
                source_path=str(path),
                parents=[],
                kind="seed",
                fitness_sum=fitness,
                n_evals=1,
                status="ok",
                reason="",
                created=time.time(),
            )
            self.insert(program)
            first = first or program
        assert first is not None
        return first

    def insert(self, program: Program) -> Program | None:
        """Add `program` to its island.

        On a full island, this replaces the worst program by mean fitness
        if `program` is better, returning the program it replaced. If the
        island is full and `program` is not better than the worst, `program`
        itself is dropped and returned unchanged — callers read "the
        returned program is the child" as "the child was dropped".
        """
        island = self._islands[program.island]
        if len(island) >= config.ISLAND_SIZE:
            worst = min(island.values(), key=lambda p: p.mean)
            if program.mean <= worst.mean:
                return program
            self._apply({"event": "remove", "island": program.island, "id": worst.id})
            self._apply({"event": "insert", "program": program.model_dump()})
            return worst
        self._apply({"event": "insert", "program": program.model_dump()})
        return None

    def append_eval(self, program_id: str, fitness: float) -> None:
        """Log one more deep-evaluation fitness for `program_id`."""
        self._apply({"event": "eval", "id": program_id, "fitness": fitness})

    def record_failure(
        self, island: int, parents: list[str], kind: Kind, reason: str
    ) -> Program:
        """Log a failed candidate that never joined an island."""
        program = Program(
            id=f"fail-{int(time.time() * 1000)}-{random.randrange(1 << 20):05x}",
            island=island,
            source_path="",
            parents=parents,
            kind=kind,
            fitness_sum=0.0,
            n_evals=0,
            status="failed",
            reason=reason,
            created=time.time(),
        )
        self._apply({"event": "failure", "program": program.model_dump()})
        return program

    def ucb_parent(self, island: int, rng: random.Random) -> Program:
        """Pick a parent from `island` by UCB over mean fitness.

        A program's visit count is its `n_evals` (real deep evaluations)
        plus any picks this process has already given it, so two programs
        with a tied mean but different evaluation histories still diverge:
        the less-sampled one carries the bigger exploration bonus. Selection
        counts (`_picks`, `_selections`) live only for this process's
        lifetime — they are exploration bookkeeping, not part of the
        archive's persisted state, so a fresh process starts them cold.
        """
        programs = self.island(island)
        self._selections += 1

        def score(p: Program) -> float:
            visits = p.n_evals + self._picks.get(p.id, 0)
            bonus = config.UCB_C * math.sqrt(
                2 * math.log(self._selections + 1) / (visits + 1)
            )
            return p.mean + bonus + rng.random() * 1e-9

        parent = max(programs, key=score)
        self._picks[parent.id] = self._picks.get(parent.id, 0) + 1
        return parent

    def migrate(self) -> None:
        """Copy each island's top `MIGRANTS` programs to the next island in a ring.

        Every island's top-`MIGRANTS` snapshot is taken before any copy is
        inserted, so an earlier island's migrants never contaminate a later
        island's "top" computation within the same call.
        """
        tops = [
            sorted(self.island(i), key=lambda p: p.mean, reverse=True)[
                : config.MIGRANTS
            ]
            for i in range(config.ISLANDS)
        ]
        for island in range(config.ISLANDS):
            target = (island + 1) % config.ISLANDS
            for program in tops[island]:
                migrant = program.model_copy(
                    update={
                        "id": f"{program.id}-m{target}",
                        "island": target,
                        "parents": [program.id],
                        "kind": "migrant",
                        "created": time.time(),
                    }
                )
                if migrant.id not in self._islands[target]:
                    self.insert(migrant)

    def reset_worst_island(self, champion: Program) -> int:
        """Clear the island with the lowest best-program mean, reseed from `champion`.

        Returns:
            The index of the island that was reset.
        """
        worst = min(
            range(config.ISLANDS),
            key=lambda i: max((p.mean for p in self.island(i)), default=-1.0),
        )
        for program in self.island(worst):
            self._apply({"event": "remove", "island": worst, "id": program.id})
        fresh = champion.model_copy(
            update={
                "id": f"{champion.id}-r{worst}-{int(time.time())}",
                "island": worst,
                "parents": [champion.id],
                "kind": "reset",
                "created": time.time(),
            }
        )
        self.insert(fresh)
        return worst
