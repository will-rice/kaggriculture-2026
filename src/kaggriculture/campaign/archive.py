"""The campaign's one shared database of programs.

An append-only JSON-lines log, replayed on start and shared by every worker.
Nothing is evicted and there is no population structure: every program a
round produced stays, ranked only by the fitness it was added with, and the
failures are kept beside them: a round that produced no program is not in the
database, so the ledger is the only place it is written down.
"""

import json
import time
from pathlib import Path

from pydantic import BaseModel

from kaggriculture.campaign import plan
from kaggriculture.campaign.evaluator import Result
from kaggriculture.campaign.harness import Margin


def mean_margin(program: "Program") -> float:
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
        fitness: Mean win rate over the pool it was measured against.
        rates: Win rate per pool opponent.
        margins: Bank margin per pool opponent. Defaulted, so a program
            written before margins existed still loads.
        changed: What this edit did to the plan, as `plan.described` renders
            it, or "" when nothing computed it -- the seed, a program from
            before this field, or a child whose parent carries no packed plan.

            The score was the only thing kept about an attempt, and a score
            says an edit was worth keeping without saying what it was. Every
            promotion's change was rendered for the prompt by reading the
            champion files back and diffing them, so the thirty-nine that
            worked were described and the four hundred and fifty that did not
            were described nowhere. Recorded here because the moment a program
            is stored is the only one where both plans are already in hand.
        created: Unix timestamp.
    """

    id: str
    source_path: str
    started_from: str
    instruction: str
    model: str
    fitness: float
    rates: dict[str, float] = {}
    margins: dict[str, Margin] = {}
    changed: str = ""
    created: float


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
        self._promoted: dict[str, str] = {}
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
        """
        kind = event["event"]
        if kind == "program":
            program = Program.model_validate(event["program"])
            if program.id in self._programs:
                raise ValueError(f"duplicate program id: {program.id!r}")
            self._programs[program.id] = program
        elif kind == "failure":
            self._failures.append(Failure.model_validate(event["failure"]))
        elif kind == "promotion":
            if event["id"] not in self._programs:
                raise ValueError(f"promotion of an unknown program: {event['id']!r}")
            self._promoted[event["id"]] = event["name"]
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

    def attempts(self, path: Path) -> int:
        """Write every attempt and what it did, one JSON object per line.

        The whole campaign as something a round can grep. It has been told what
        its own siblings scored and what the promotions changed, which is a few
        dozen edits out of four hundred and eighty-nine; the rest -- every
        change measured and refused -- was on the record and reachable by
        nothing.

        A file rather than more of the message, and that is the point. Four
        hundred attempts would not fit in a round's budget, and a round pays for
        what it reads: this costs nothing until a question calls for it. It is
        also not an example. Worked examples in the prompt become the subject --
        forty-six of forty-six queries went where the one worked query pointed
        -- where a file answers the question the round brought to it.

        The opponent names come too. They are already in `games.days` and
        `games.episodes` by roster name, which a round queries, so withholding
        them here would buy nothing and cost the one question worth asking of
        this file: whether an edit helped against the agent that beats us.

        Args:
            path: Where to write it.

        Returns:
            How many attempts were written.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as out:
            for program in self._programs.values():
                out.write(
                    json.dumps(
                        {
                            "id": program.id,
                            "from": program.started_from,
                            "instruction": program.instruction,
                            "changed": program.changed,
                            "wins": round(program.fitness, 4),
                            "margin": round(mean_margin(program)),
                            "promoted": program.id in self._promoted,
                            "rates": {
                                name: round(rate, 3)
                                for name, rate in program.rates.items()
                            },
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        return len(self._programs)

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

    def promoted_as(self, program_id: str, name: str) -> None:
        """Record that `program_id` became the champion the pool knows as `name`.

        The archive is the record of every attempt, and which of them survived
        a gate is the one fact about an attempt worth more than its score. It
        was recoverable only by diffing the champion files against every stored
        source, because `gate.promote` copies a program to `champion_N.py` and
        keeps no id. An event, like the rest, so a replay knows it too.

        Raises:
            ValueError: The archive holds no such program.
        """
        self._apply({"event": "promotion", "id": program_id, "name": name})

    @property
    def promoted(self) -> dict[str, str]:
        """Every program that became a champion, by id, to the name it took."""
        return dict(self._promoted)

    def top(self, k: int) -> list[Program]:
        """Return the `k` best programs, best first: most games won.

        On the win rate, because every candidate plays the whole pool on the
        same seeds in both seats, which makes the rate directly comparable.
        Then on the mean bank margin across opponents. That tie-break is what
        makes the opening hours a search rather than a random walk: before the
        first win every rate is 0.0, sorting on it alone leaves the ties in
        insertion order, and the margins say which of those came closest.
        """
        return sorted(
            self._programs.values(),
            key=lambda p: (p.fitness, mean_margin(p)),
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
            key=lambda p: (p.fitness, mean_margin(p)),
            reverse=True,
        )

    def descendants(self, root: str) -> list[Program]:
        """Every program grown from ``root``, however many generations down.

        One forward pass is enough because programs are added in the order
        they were written and a child is always added after its parent, so a
        parent is already in the family by the time its children are read.

        Args:
            root: The id or name the lineage starts from.

        Returns:
            The lineage, oldest first, excluding ``root`` itself.
        """
        family = {root}
        out = []
        for program in self._programs.values():
            if program.started_from in family:
                family.add(program.id)
                out.append(program)
        return out

    def get(self, program_id: str) -> Program:
        """Return the program `program_id`.

        Raises:
            KeyError: No program has that id.
        """
        return self._programs[program_id]

    def failures(self, started_from: str) -> list[Failure]:
        """Return the failures recorded against `started_from`, oldest first."""
        return [f for f in self._failures if f.started_from == started_from]


def measured(
    program_id: str,
    source: Path,
    started_from: str,
    instruction: str,
    model: str,
    result: Result,
    parent: Path | None = None,
) -> Program:
    """One database entry for a measured program, stamped now.

    ``parent`` is the source this was edited from, and it is here for one
    reason: this is the only moment both plans exist as files, so describing
    the edit costs a read of two programs already on the disk. Asked later it
    costs the whole archive -- 489 programs at 869KB each.
    """
    return Program(
        changed=changed(source, parent),
        id=program_id,
        source_path=str(source),
        started_from=started_from,
        instruction=instruction,
        model=model,
        fitness=result.fitness,
        rates=result.rates,
        margins=result.margins,
        created=time.time(),
    )


def changed(source: Path, parent: Path | None) -> str:
    """What this program did to its parent's plan, in a round's own terms.

    Empty rather than raising, for every reason a pair of programs might not be
    comparable: the seed has no parent, a program from before champion_17
    carries no packed plan, and a round can rewrite the controller into
    something `split` refuses. None of those is a failed evaluation, and a
    campaign must not lose a scored program because the sentence describing it
    could not be written.
    """
    if parent is None:
        return ""
    try:
        _, before = plan.split(parent.read_text(encoding="utf-8"))
        _, after = plan.split(source.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return ""
    return plan.described(before, after)
