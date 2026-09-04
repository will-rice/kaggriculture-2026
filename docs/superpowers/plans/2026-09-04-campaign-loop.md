# Campaign Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The FAMOU loop over our own agent: an island archive of agent files, `codex exec` as the mutation operator, fast evaluation on fresh seeds for ranking, the sealed exam block as the deep evaluation, a co-evolving opponent pool with weakness pressure, promotion of champions to the floor, and a dry run that proves the whole pipeline without codex.

**Architecture:** Every module is a pure function or a small persisted record under `run/campaign/`; `loop.py` composes them into Algorithm 1 of the spec. Codex is called only by `mutate.py`, always in a sandbox `prompt.py` built, never with a path under `/data/kaggriculture/opponents/` in it. The gate is `field_gate.score_field` on `config.EXAM_SEEDS` (Plan 1, Task 10). A `FakeMutator` with the same interface as the codex one lets every loop test run without a model.

**Tech Stack:** Python 3.11, pydantic, `ThreadPoolExecutor`, `codex exec` 0.147 (`--json`), pytest. Spec: `docs/superpowers/specs/2026-09-04-codex-campaign-design.md` §2–§6. Foundation: `docs/superpowers/plans/2026-09-04-campaign-foundation.md`.

## Global Constraints

- Exam seeds `config.EXAM_SEEDS` (700000–700063) are played only by `evaluator.deep`; fast evaluation draws seeds from `range(1, 600_000)`.
- No sandbox may contain a path under `/data/kaggriculture/opponents/` or `/data/kaggriculture/agents/`, or any opponent source. Opponents appear only as names.
- Fitness is win rate only, pool-weighted. No margin term.
- Hyperparameters live in `config.py` as constants (spec §3): `ISLANDS = 4`, `ISLAND_SIZE = 12`, `MIGRATION_INTERVAL = 10`, `MIGRANTS = 2`, `RESET_INTERVAL = 40`, `UCB_C = 0.5`, `CROSS_PROBABILITY = 0.3`, `FAST_SEEDS = 4`, `EPOCH_INTERVAL = 25`, `DEEP_TOP_K = 3`, `CHAMPION_WEIGHT = 0.20`, `POOL_CAP = 10`, `RETIRE_THRESHOLD = 0.95`, `WEAKNESS_CAP = 0.5`, `CODEX_CONCURRENCY = 8`, `MUTATION_TIMEOUT_SECONDS = 600`, `DAILY_CALL_BUDGET = 200`.
- The floor is written only by `gate.promote`; the archive only by `archive.py`; the pool only by `pool.py`.
- Every record is one JSON line appended to a file under `run/campaign/`; nothing is rewritten except `pool.json` and the archive's index.
- Style per `~/.claude/CLAUDE.md`; `git rm`/`git add` by explicit path; conventional commits that say why.

---

## File structure

| path                                                                                                                                      | responsibility                                                                                                                                                                                                                                                                           |
| ----------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `src/kaggriculture/campaign/config.py`                                                                                                    | add the hyperparameters above and `RUN`-relative paths: `ARCHIVE = RUN / "archive.jsonl"`, `PROGRAMS = RUN / "programs"`, `SANDBOXES = RUN / "sandboxes"`, `POOL = RUN / "pool.json"`, `EPOCHS = RUN / "epochs.jsonl"`, `FLOOR = RUN / "floor" / "agent"`, `CALLS = RUN / "calls.jsonl"` |
| `src/kaggriculture/campaign/pool.py`                                                                                                      | `Pool`: opponents and weights; `add_champion`, `apply_weakness_pressure`, `weakest`; JSON persistence — a port of FAMOU's `opponent_pool.py`                                                                                                                                             |
| `src/kaggriculture/campaign/archive.py`                                                                                                   | `Program`, `Archive`: islands, UCB parent, insert/replace, migration, reset; JSONL persistence                                                                                                                                                                                           |
| `src/kaggriculture/campaign/evaluator.py`                                                                                                 | `fast(program) -> FastResult`, `deep(program) -> DeepResult` (exam block, Wilson, held-out)                                                                                                                                                                                              |
| `src/kaggriculture/campaign/prompt.py`                                                                                                    | builds one mutation sandbox: `AGENTS.md`, `parent.py`, `inspiration.py`, `feedback.md`, `engine/`, `PROMPT.md`                                                                                                                                                                           |
| `src/kaggriculture/campaign/mutate.py`                                                                                                    | `Mutator` protocol; `CodexMutator` (one `codex exec` call, JSONL parse, tokens); `FakeMutator`                                                                                                                                                                                           |
| `src/kaggriculture/campaign/gate.py`                                                                                                      | `promotion(candidate: DeepResult, champion: DeepResult                                                                                                                                                                                                                                   | None) -> bool`; `promote(program)`: floor write, `served/main.py`, pool add, epoch line, git commit |
| `src/kaggriculture/campaign/loop.py`                                                                                                      | Algorithm 1; `campaign loop` and `campaign dry-run` CLI                                                                                                                                                                                                                                  |
| `tests/campaign/test_pool.py`, `test_archive.py`, `test_evaluator.py`, `test_prompt.py`, `test_mutate.py`, `test_gate.py`, `test_loop.py` | one per module                                                                                                                                                                                                                                                                           |

---

### Task 1: Hyperparameters and runtime paths in `config.py`

**Files:**

- Modify: `src/kaggriculture/campaign/config.py`
- Test: `tests/campaign/test_config.py`

**Interfaces:**

- Produces: the constants in Global Constraints, plus `config.FAST_SEED_RANGE = range(1, 600_000)`, and the paths listed in File structure.

- [ ] **Step 1: Add the failing test**

Append to `tests/campaign/test_config.py`:

```python
def test_fast_seed_range_never_touches_the_exam_block() -> None:
    assert not set(config.FAST_SEED_RANGE) & set(config.EXAM_SEEDS)


def test_hyperparameters_match_the_spec_table() -> None:
    assert (config.ISLANDS, config.ISLAND_SIZE) == (4, 12)
    assert (config.MIGRATION_INTERVAL, config.MIGRANTS, config.RESET_INTERVAL) == (10, 2, 40)
    assert config.UCB_C == 0.5 and config.CROSS_PROBABILITY == 0.3
    assert (config.FAST_SEEDS, config.EPOCH_INTERVAL, config.DEEP_TOP_K) == (4, 25, 3)
    assert (config.CHAMPION_WEIGHT, config.POOL_CAP, config.RETIRE_THRESHOLD, config.WEAKNESS_CAP) == (0.20, 10, 0.95, 0.5)
    assert (config.CODEX_CONCURRENCY, config.MUTATION_TIMEOUT_SECONDS, config.DAILY_CALL_BUDGET) == (8, 600, 200)


def test_runtime_paths_live_under_run_campaign() -> None:
    for path in (config.ARCHIVE, config.PROGRAMS, config.SANDBOXES, config.POOL, config.EPOCHS, config.FLOOR, config.CALLS):
        assert config.RUN in path.parents
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/campaign/test_config.py -v` → `AttributeError`.

- [ ] **Step 3: Add the constants**

```python
# Spec §3. One iteration is one mutation per island.
ISLANDS = 4
ISLAND_SIZE = 12
MIGRATION_INTERVAL = 10
MIGRANTS = 2
RESET_INTERVAL = 40
UCB_C = 0.5
CROSS_PROBABILITY = 0.3
FAST_SEEDS = 4                      # x both seats x every pool opponent
FAST_SEED_RANGE = range(1, 600_000)  # never the exam block
EPOCH_INTERVAL = 25
DEEP_TOP_K = 3
CHAMPION_WEIGHT = 0.20
POOL_CAP = 10
RETIRE_THRESHOLD = 0.95
WEAKNESS_CAP = 0.5
CODEX_CONCURRENCY = 8
MUTATION_TIMEOUT_SECONDS = 600
DAILY_CALL_BUDGET = 200

ARCHIVE = RUN / "archive.jsonl"
PROGRAMS = RUN / "programs"
SANDBOXES = RUN / "sandboxes"
POOL = RUN / "pool.json"
EPOCHS = RUN / "epochs.jsonl"
FLOOR = RUN / "floor" / "agent"
CALLS = RUN / "calls.jsonl"
```

- [ ] **Step 4: Run to verify it passes**, then commit: `git add src/kaggriculture/campaign/config.py tests/campaign/test_config.py && git commit -m "feat: pin the loop's hyperparameters and runtime paths"`.

---

### Task 2: The opponent pool

**Files:**

- Create: `src/kaggriculture/campaign/pool.py`
- Test: `tests/campaign/test_pool.py`

**Interfaces:**

- Consumes: `roster.TRAINING`, `config.CHAMPION_WEIGHT`, `POOL_CAP`, `RETIRE_THRESHOLD`, `WEAKNESS_CAP`.
- Produces: `class Pool(BaseModel)` with `opponents: dict[str, str]` (name → path), `weights: dict[str, float]`, `history: list[dict]`; `Pool.initial() -> Pool` (from `roster.TRAINING`, equal weights); `Pool.load(path) -> Pool`, `.save(path)`; `.add_champion(name, path, rates: dict[str, float]) -> str | None` (returns the retired name); `.apply_weakness_pressure(name)`; `.weakest(rates) -> str`; `.names() -> list[str]`; `.weighted(rates: dict[str, float]) -> float`.

- [ ] **Step 1: Write the failing tests**

```python
"""A port of FAMOU's OpponentPool, checked on its own arithmetic."""

from pathlib import Path

import pytest

from kaggriculture.campaign import config, pool, roster


def five() -> pool.Pool:
    return pool.Pool(
        opponents={name: f"/x/{name}.py" for name in ["a", "b", "c", "d", "e"]},
        weights={name: 0.2 for name in ["a", "b", "c", "d", "e"]},
    )


def test_initial_pool_is_the_training_roster_with_equal_weights() -> None:
    p = pool.Pool.initial()
    assert p.names() == list(roster.TRAINING)
    assert all(abs(w - 1 / len(roster.TRAINING)) < 1e-12 for w in p.weights.values())


def test_add_champion_gives_it_0_20_and_renormalises() -> None:
    p = five()
    retired = p.add_champion("champ", "/x/champ.py", rates={})
    assert retired is None
    assert abs(sum(p.weights.values()) - 1.0) < 1e-12
    assert abs(p.weights["champ"] - 0.20 / 1.20) < 1e-12
    assert all(abs(p.weights[n] - 0.20 / 1.20) < 1e-12 for n in "abcde")


def test_full_pool_retires_the_lowest_weight_opponent_the_champion_crushes() -> None:
    p = five()
    for i in range(5):
        p.add_champion(f"c{i}", f"/x/c{i}.py", rates={})
    assert len(p.names()) == config.POOL_CAP
    p.weights["b"] = 0.01
    p.weights["a"] = 0.05
    total = sum(p.weights.values())
    p.weights = {k: v / total for k, v in p.weights.items()}
    retired = p.add_champion("new", "/x/new.py", rates={"a": 0.99, "b": 0.97, "c": 0.5})
    assert retired == "b"
    assert "b" not in p.names() and "new" in p.names()


def test_weakness_pressure_doubles_the_weakest_and_caps_at_half() -> None:
    p = five()
    p.apply_weakness_pressure("c")
    assert abs(p.weights["c"] - 0.4) < 1e-12
    assert all(abs(p.weights[n] - 0.15) < 1e-12 for n in "abde")
    p.apply_weakness_pressure("c")
    assert abs(p.weights["c"] - 0.5) < 1e-12
    assert abs(sum(p.weights.values()) - 1.0) < 1e-12


def test_weakest_and_weighted() -> None:
    p = five()
    rates = {"a": 0.9, "b": 0.2, "c": 0.5, "d": 0.7, "e": 0.6}
    assert p.weakest(rates) == "b"
    assert abs(p.weighted(rates) - 0.58) < 1e-12


def test_round_trips_through_json(tmp_path: Path) -> None:
    p = five()
    p.apply_weakness_pressure("a")
    p.save(tmp_path / "pool.json")
    assert pool.Pool.load(tmp_path / "pool.json") == p
```

- [ ] **Step 2: Run to verify they fail** — `ImportError`.

- [ ] **Step 3: Write `pool.py`**

```python
"""The opponent pool: who a candidate is measured against, and how much each counts.

A port of the released FAMOU ``opponent_pool.py``: champions join at a fixed
weight, weights renormalise, a full pool retires the lowest-weight opponent
the new champion already beats, and weakness pressure doubles the weakest
opponent's weight up to a cap. Paths are stored here and shown nowhere.
"""

import json
import time
from pathlib import Path

from pydantic import BaseModel

from kaggriculture.campaign import config, roster


class Pool(BaseModel):
    opponents: dict[str, str]
    weights: dict[str, float]
    history: list[dict] = []

    @classmethod
    def initial(cls) -> "Pool":
        names = list(roster.TRAINING)
        return cls(
            opponents={n: str(roster.TRAINING[n]) for n in names},
            weights={n: 1.0 / len(names) for n in names},
        )

    @classmethod
    def load(cls, path: Path) -> "Pool":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    def names(self) -> list[str]:
        return list(self.opponents)

    def weighted(self, rates: dict[str, float]) -> float:
        """Pool-weighted win rate over the opponents present in ``rates``."""
        return sum(self.weights[n] * rates[n] for n in self.opponents if n in rates)

    def weakest(self, rates: dict[str, float]) -> str:
        return min((n for n in self.opponents if n in rates), key=lambda n: rates[n])

    def add_champion(self, name: str, path: str, rates: dict[str, float]) -> str | None:
        """Add at ``CHAMPION_WEIGHT``; retire one crushed opponent if the pool is full."""
        retired = None
        if len(self.opponents) >= config.POOL_CAP:
            crushed = [(self.weights[n], n) for n, r in rates.items() if n in self.opponents and r >= config.RETIRE_THRESHOLD]
            if crushed:
                retired = min(crushed)[1]
                del self.opponents[retired]
                del self.weights[retired]
        self.opponents[name] = path
        self.weights[name] = config.CHAMPION_WEIGHT
        self._rebalance()
        self.history.append({"action": "add_champion", "name": name, "retired": retired, "ts": time.time()})
        return retired

    def apply_weakness_pressure(self, name: str) -> None:
        old = self.weights[name]
        new = min(old * 2, config.WEAKNESS_CAP)
        scale = (1.0 - new) / (1.0 - old)
        for n in self.weights:
            self.weights[n] = new if n == name else self.weights[n] * scale
        self.history.append({"action": "weakness_pressure", "name": name, "old": old, "new": new, "ts": time.time()})

    def _rebalance(self) -> None:
        total = sum(self.weights.values())
        for n in self.weights:
            self.weights[n] /= total
```

- [ ] **Step 4: Run to verify they pass**, commit: `git add src/kaggriculture/campaign/pool.py tests/campaign/test_pool.py && git commit -m "feat: port FAMOU's opponent pool with retirement and weakness pressure"`.

---

### Task 3: The archive

**Files:**

- Create: `src/kaggriculture/campaign/archive.py`
- Test: `tests/campaign/test_archive.py`

**Interfaces:**

- Consumes: `config.ISLANDS`, `ISLAND_SIZE`, `MIGRANTS`, `UCB_C`, `PROGRAMS`, `ARCHIVE`.
- Produces: `class Program(BaseModel)`: `id: str`, `island: int`, `source_path: str`, `parents: list[str]`, `kind: Literal["seed", "full", "cross", "migrant", "reset"]`, `fitness_sum: float`, `n_evals: int`, `status: str`, `reason: str`, `created: float`; `.mean -> float`. `class Archive`: `Archive(path: Path = config.ARCHIVE, programs_dir: Path = config.PROGRAMS)`; `.seed(source: Path, fitness: float) -> Program` (one copy per island, kind `seed`); `.ucb_parent(island: int, rng: random.Random) -> Program`; `.insert(program: Program) -> Program | None` (returns the program it replaced, if the island was full); `.record_failure(island, parents, kind, reason) -> Program`; `.migrate() -> None` (ring, top `MIGRANTS` copied to the next island as kind `migrant`); `.reset_worst_island(champion: Program) -> int`; `.top(k) -> list[Program]` (by mean over all islands, `n_evals ≥ 1`); `.island(i) -> list[Program]`; `.store(source: str, program_id: str) -> Path` (writes `PROGRAMS/<id>.py`); `.append_eval(program_id, fitness)`. Persistence: every mutation of state appends one line to `ARCHIVE`; `Archive.load()` replays the log.

- [ ] **Step 1: Write the failing tests**

```python
"""Islands, UCB, replacement, migration, reset — each on a hand-built archive."""

import random
from pathlib import Path

from kaggriculture.campaign import archive, config


def make(tmp_path: Path) -> archive.Archive:
    return archive.Archive(path=tmp_path / "archive.jsonl", programs_dir=tmp_path / "programs")


def test_seed_places_one_copy_on_every_island(tmp_path: Path) -> None:
    a = make(tmp_path)
    seed = tmp_path / "seed.py"
    seed.write_text("def agent(o, c=None):\n    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n")
    a.seed(seed, fitness=0.1)
    assert [len(a.island(i)) for i in range(config.ISLANDS)] == [1] * config.ISLANDS
    assert all(p.kind == "seed" and p.mean == 0.1 for i in range(config.ISLANDS) for p in a.island(i))


def test_ucb_prefers_the_under_sampled_program_when_means_tie(tmp_path: Path) -> None:
    a = make(tmp_path)
    many = a.insert(archive.Program(id="many", island=0, source_path="x", parents=[], kind="full", fitness_sum=5.0, n_evals=10, status="ok", reason="", created=0.0))
    few = a.insert(archive.Program(id="few", island=0, source_path="y", parents=[], kind="full", fitness_sum=0.5, n_evals=1, status="ok", reason="", created=0.0))
    assert many is None and few is None
    assert a.ucb_parent(0, random.Random(0)).id == "few"


def test_insert_replaces_the_worst_when_the_island_is_full(tmp_path: Path) -> None:
    a = make(tmp_path)
    for i in range(config.ISLAND_SIZE):
        a.insert(archive.Program(id=f"p{i}", island=1, source_path="x", parents=[], kind="full", fitness_sum=i / 100, n_evals=1, status="ok", reason="", created=0.0))
    replaced = a.insert(archive.Program(id="new", island=1, source_path="x", parents=[], kind="full", fitness_sum=0.5, n_evals=1, status="ok", reason="", created=0.0))
    assert replaced is not None and replaced.id == "p0"
    assert len(a.island(1)) == config.ISLAND_SIZE and "new" in {p.id for p in a.island(1)}


def test_a_weaker_child_does_not_evict_anyone_from_a_full_island(tmp_path: Path) -> None:
    a = make(tmp_path)
    for i in range(config.ISLAND_SIZE):
        a.insert(archive.Program(id=f"p{i}", island=2, source_path="x", parents=[], kind="full", fitness_sum=0.5, n_evals=1, status="ok", reason="", created=0.0))
    replaced = a.insert(archive.Program(id="weak", island=2, source_path="x", parents=[], kind="full", fitness_sum=0.1, n_evals=1, status="ok", reason="", created=0.0))
    assert replaced is not None and replaced.id == "weak"     # the child itself is what was dropped
    assert "weak" not in {p.id for p in a.island(2)}


def test_migration_copies_the_top_two_to_the_next_island_in_a_ring(tmp_path: Path) -> None:
    a = make(tmp_path)
    for island in range(config.ISLANDS):
        for j in range(3):
            a.insert(archive.Program(id=f"i{island}p{j}", island=island, source_path="x", parents=[], kind="full", fitness_sum=(island + 1) * (j + 1) / 20, n_evals=1, status="ok", reason="", created=0.0))
    a.migrate()
    last = config.ISLANDS - 1
    ids_on_0 = {p.id for p in a.island(0)}
    assert {f"i{last}p2", f"i{last}p1"} <= {p.parents[0] for p in a.island(0) if p.kind == "migrant"}
    assert len(ids_on_0) == 5


def test_reset_reseeds_the_worst_island_from_the_champion(tmp_path: Path) -> None:
    a = make(tmp_path)
    for island in range(config.ISLANDS):
        a.insert(archive.Program(id=f"i{island}", island=island, source_path="x", parents=[], kind="full", fitness_sum=island / 10, n_evals=1, status="ok", reason="", created=0.0))
    champion = a.top(1)[0]
    reset = a.reset_worst_island(champion)
    assert reset == 0
    assert [p.kind for p in a.island(0)] == ["reset"] and a.island(0)[0].parents == [champion.id]


def test_the_log_replays_to_the_same_state(tmp_path: Path) -> None:
    a = make(tmp_path)
    a.insert(archive.Program(id="p", island=0, source_path="x", parents=[], kind="full", fitness_sum=0.3, n_evals=1, status="ok", reason="", created=0.0))
    a.append_eval("p", 0.7)
    a.record_failure(0, ["p"], "full", "syntax")
    b = archive.Archive(path=tmp_path / "archive.jsonl", programs_dir=tmp_path / "programs")
    assert [p.model_dump() for p in b.island(0)] == [p.model_dump() for p in a.island(0)]
    assert a.island(0)[0].mean == 0.5 and b.failures()[0].reason == "syntax"
```

- [ ] **Step 2: Run to verify they fail** — `ImportError`.

- [ ] **Step 3: Write `archive.py`**

```python
"""The population: islands of agent programs and the log that is their memory.

Every state change is one appended JSON line; loading replays the log. UCB
picks parents, a full island replaces its worst, migration copies the top
programs around a ring, and a reset reseeds the worst island from the
champion — FAMOU's island model at our scale.
"""

import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from kaggriculture.campaign import config

Kind = Literal["seed", "full", "cross", "migrant", "reset"]


class Program(BaseModel):
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

    @property
    def mean(self) -> float:
        return self.fitness_sum / self.n_evals if self.n_evals else 0.0


class Archive:
    def __init__(self, path: Path = config.ARCHIVE, programs_dir: Path = config.PROGRAMS) -> None:
        self.path = path
        self.programs_dir = programs_dir
        self._islands: list[dict[str, Program]] = [{} for _ in range(config.ISLANDS)]
        self._failures: list[Program] = []
        self._selections = 0
        self._picks: dict[str, int] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                self._apply(json.loads(line), persist=False)

    def _apply(self, event: dict, persist: bool = True) -> None:
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

    def _find(self, program_id: str) -> Program:
        for island in self._islands:
            if program_id in island:
                return island[program_id]
        raise KeyError(program_id)

    def island(self, i: int) -> list[Program]:
        return list(self._islands[i].values())

    def failures(self) -> list[Program]:
        return list(self._failures)

    def top(self, k: int) -> list[Program]:
        everyone = [p for island in self._islands for p in island.values() if p.n_evals]
        return sorted(everyone, key=lambda p: p.mean, reverse=True)[:k]

    def store(self, source: str, program_id: str) -> Path:
        self.programs_dir.mkdir(parents=True, exist_ok=True)
        target = self.programs_dir / f"{program_id}.py"
        target.write_text(source, encoding="utf-8")
        return target

    def seed(self, source: Path, fitness: float) -> Program:
        text = source.read_text(encoding="utf-8")
        first = None
        for island in range(config.ISLANDS):
            program_id = f"seed-{island}"
            path = self.store(text, program_id)
            program = Program(id=program_id, island=island, source_path=str(path), parents=[], kind="seed",
                              fitness_sum=fitness, n_evals=1, status="ok", reason="", created=time.time())
            self.insert(program)
            first = first or program
        return first

    def insert(self, program: Program) -> Program | None:
        """Add to its island; on a full island replace the worst, or drop the child if it is the worst."""
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
        self._apply({"event": "eval", "id": program_id, "fitness": fitness})

    def record_failure(self, island: int, parents: list[str], kind: Kind, reason: str) -> Program:
        program = Program(id=f"fail-{int(time.time() * 1000)}-{random.randrange(1 << 20):05x}", island=island,
                          source_path="", parents=parents, kind=kind, fitness_sum=0.0, n_evals=0,
                          status="failed", reason=reason, created=time.time())
        self._apply({"event": "failure", "program": program.model_dump()})
        return program

    def ucb_parent(self, island: int, rng: random.Random) -> Program:
        """UCB over mean fitness with an exploration bonus for rarely selected programs."""
        programs = self.island(island)
        self._selections += 1
        def score(p: Program) -> float:
            picks = self._picks.get(p.id, 0)
            bonus = config.UCB_C * math.sqrt(2 * math.log(self._selections + 1) / (picks + 1))
            return p.mean + bonus + rng.random() * 1e-9
        parent = max(programs, key=score)
        self._picks[parent.id] = self._picks.get(parent.id, 0) + 1
        return parent

    def migrate(self) -> None:
        for island in range(config.ISLANDS):
            target = (island + 1) % config.ISLANDS
            for program in sorted(self.island(island), key=lambda p: p.mean, reverse=True)[: config.MIGRANTS]:
                migrant = program.model_copy(update={
                    "id": f"{program.id}-m{target}", "island": target, "parents": [program.id], "kind": "migrant",
                    "created": time.time(),
                })
                if migrant.id not in self._islands[target]:
                    self.insert(migrant)

    def reset_worst_island(self, champion: Program) -> int:
        worst = min(range(config.ISLANDS), key=lambda i: max((p.mean for p in self.island(i)), default=-1.0))
        for program in self.island(worst):
            self._apply({"event": "remove", "island": worst, "id": program.id})
        fresh = champion.model_copy(update={
            "id": f"{champion.id}-r{worst}-{int(time.time())}", "island": worst, "parents": [champion.id],
            "kind": "reset", "created": time.time(),
        })
        self.insert(fresh)
        return worst
```

`insert` on a full island where the child is the worst returns the child itself (the test asserts this); callers read "replaced is the child" as "dropped".

- [ ] **Step 4: Run to verify they pass**, commit: `git add src/kaggriculture/campaign/archive.py tests/campaign/test_archive.py && git commit -m "feat: an island archive with UCB parents, replacement, migration and reset"`.

---

### Task 4: Evaluators

**Files:**

- Create: `src/kaggriculture/campaign/evaluator.py`
- Test: `tests/campaign/test_evaluator.py`

**Interfaces:**

- Consumes: `harness.play` (fast; refuses exam seeds), `field_gate.score_field` (deep; exam seeds), `pool.Pool`, `roster.HELD_OUT`, `report.wilson_interval`, `config.FAST_SEEDS`, `FAST_SEED_RANGE`, `EXAM_SEEDS`, `CORE_BUDGET`.
- Produces: `class FastResult(BaseModel)`: `fitness: float`, `rates: dict[str, float]`, `seeds: list[int]`; `class DeepResult(BaseModel)`: `program_id: str`, `score: float`, `low: float`, `high: float`, `rates: dict[str, float]`, `intervals: dict[str, tuple[float, float]]`, `field: float`, `held_out: dict[str, float]`, `games: int`; `fast(agent: Path, pool: Pool, rng: random.Random, workers: int) -> FastResult`; `deep(agent: Path, program_id: str, pool: Pool, workers: int) -> DeepResult`; `VENDORED = list(roster.TRAINING)` (the `field` set).

- [ ] **Step 1: Write the failing tests**

```python
"""Fast eval on fresh seeds; deep eval on the exam block. Both on the port."""

import random
from pathlib import Path

from kaggriculture.campaign import config, evaluator, pool

PASS = "def agent(observation, configuration=None):\n    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"


def test_fast_draws_fresh_non_exam_seeds_and_weights_by_the_pool(tmp_path: Path) -> None:
    agent = tmp_path / "main.py"
    agent.write_text(PASS, encoding="utf-8")
    p = pool.Pool(opponents={"v54": str(config.OPPONENTS / "kaito_v54" / "main.py")}, weights={"v54": 1.0})
    result = evaluator.fast(agent, p, random.Random(1), workers=4)
    assert len(result.seeds) == config.FAST_SEEDS and not set(result.seeds) & set(config.EXAM_SEEDS)
    assert result.rates == {"v54": 0.0} and result.fitness == 0.0


def test_two_fast_calls_draw_different_seeds(tmp_path: Path) -> None:
    agent = tmp_path / "main.py"
    agent.write_text(PASS, encoding="utf-8")
    p = pool.Pool(opponents={"v54": str(config.OPPONENTS / "kaito_v54" / "main.py")}, weights={"v54": 1.0})
    rng = random.Random(2)
    assert evaluator.fast(agent, p, rng, workers=4).seeds != evaluator.fast(agent, p, rng, workers=4).seeds


def test_deep_scores_the_exam_block_with_intervals_and_held_out(tmp_path: Path, monkeypatch) -> None:
    agent = tmp_path / "main.py"
    agent.write_text(PASS, encoding="utf-8")
    monkeypatch.setattr(config, "EXAM_SEEDS", config.EXAM_SEEDS[:2])   # 4 games per opponent for the test
    p = pool.Pool(opponents={"v54": str(config.OPPONENTS / "kaito_v54" / "main.py")}, weights={"v54": 1.0})
    result = evaluator.deep(agent, "prog", p, workers=4)
    assert result.program_id == "prog" and result.score == 0.0 and result.games == 4
    assert result.intervals["v54"][0] == 0.0 and result.intervals["v54"][1] < 1.0
    assert set(result.held_out) == set(evaluator.HELD_OUT)
```

- [ ] **Step 2: Run to verify they fail** — `ImportError`.

- [ ] **Step 3: Write `evaluator.py`**

```python
"""Two evaluations: a fast one for ranking, a deep one that decides.

Fast plays fresh non-exam seeds through the harness, weighted by the pool.
Deep plays the sealed exam block through the field gate, both seats, every
pool opponent and the held-out set, and reports Wilson intervals.
"""

import random
from pathlib import Path

from pydantic import BaseModel

from kaggriculture.campaign import config, field_gate, harness, roster
from kaggriculture.campaign.pool import Pool
from kaggriculture.report import wilson_interval

VENDORED = list(roster.TRAINING)
HELD_OUT = list(roster.HELD_OUT)


class FastResult(BaseModel):
    fitness: float
    rates: dict[str, float]
    seeds: list[int]


class DeepResult(BaseModel):
    program_id: str
    score: float
    low: float
    high: float
    rates: dict[str, float]
    intervals: dict[str, tuple[float, float]]
    field: float
    held_out: dict[str, float]
    games: int


def _rates(games: list[harness.Game], names: list[str]) -> dict[str, float]:
    out = {}
    for name in names:
        mine = [g for g in games if g.opponent == name]
        out[name] = sum(1.0 if g.ours > g.theirs else 0.5 if g.ours == g.theirs else 0.0 for g in mine) / len(mine)
    return out


def fast(agent: Path, pool: Pool, rng: random.Random, workers: int) -> FastResult:
    """Pool-weighted win rate over ``FAST_SEEDS`` fresh seeds, both seats."""
    seeds = rng.sample(config.FAST_SEED_RANGE, config.FAST_SEEDS)
    games = harness.play(agent, pool.names(), seeds, workers)
    rates = _rates(games, pool.names())
    return FastResult(fitness=pool.weighted(rates), rates=rates, seeds=seeds)


def deep(agent: Path, program_id: str, pool: Pool, workers: int) -> DeepResult:
    """The gate: exam block × both seats × pool and held-out, with intervals."""
    names = pool.names()
    with_paths = {**pool.opponents, **{n: str(roster.HELD_OUT[n]) for n in HELD_OUT}}
    rates = field_gate.score_field(agent, seeds=config.EXAM_SEEDS, workers=workers, opponents=list(with_paths))
    games = 2 * len(config.EXAM_SEEDS)
    intervals = {n: wilson_interval(rates[n] * games, games) for n in rates}
    weighted = pool.weighted(rates)
    total = games * len(names)
    low, high = wilson_interval(weighted * total, total)
    vendored = [n for n in VENDORED if n in rates]
    return DeepResult(
        program_id=program_id, score=weighted, low=low, high=high,
        rates={n: rates[n] for n in names}, intervals={n: intervals[n] for n in names},
        field=sum(rates[n] for n in vendored) / len(vendored),
        held_out={n: rates[n] for n in HELD_OUT if n in rates}, games=games,
    )
```

`field_gate.score_field` resolves names through `roster.path`; a champion in the pool is not in the roster. Extend `roster.path` to consult `config.POOL` (`Pool.load(config.POOL).opponents`) when a name is not in `TRAINING`/`HELD_OUT`, keeping the `KeyError` for unknown names — and add a test in `tests/campaign/test_roster.py` that a champion name registered in a temporary `pool.json` resolves while a random string still raises.

- [ ] **Step 4: Run to verify they pass**, commit: `git add src/kaggriculture/campaign/evaluator.py src/kaggriculture/campaign/roster.py tests/campaign/test_evaluator.py tests/campaign/test_roster.py && git commit -m "feat: fast evaluation on fresh seeds, deep evaluation on the exam block"`.

---

### Task 5: The mutation sandbox

**Files:**

- Create: `src/kaggriculture/campaign/prompt.py`
- Test: `tests/campaign/test_prompt.py`

**Interfaces:**

- Consumes: `campaign/task_prompt.md`, `archive.Program`, `evaluator.FastResult`, `pool.Pool`, `config.SANDBOXES`.
- Produces: `build_sandbox(program_id: str, kind: Literal["full", "cross"], parent: Path, inspiration: Path | None, feedback: dict[str, float], weakest: str, weights: dict[str, float], failures: list[str]) -> Path` (the sandbox directory, containing `AGENTS.md`, `parent.py`, optional `inspiration.py`, `feedback.md`, `engine/kaggriculture.py`, `PROMPT.md`); `HARNESS_SECTION: str`; `DOCTRINE: str`.

- [ ] **Step 1: Write the failing tests**

```python
"""What a sandbox contains, and what it must never contain."""

from pathlib import Path

from kaggriculture.campaign import config, prompt


def test_sandbox_has_every_required_file_and_no_opponent_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "SANDBOXES", tmp_path)
    parent = tmp_path / "p.py"
    parent.write_text("def agent(o, c=None):\n    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n")
    box = prompt.build_sandbox("abc", "full", parent, None, {"v54": 0.1, "router_v1": 0.0}, "router_v1", {"v54": 0.5, "router_v1": 0.5}, ["syntax: bad"])
    for name in ("AGENTS.md", "parent.py", "feedback.md", "engine/kaggriculture.py", "PROMPT.md"):
        assert (box / name).exists(), name
    everything = "".join(p.read_text(encoding="utf-8", errors="replace") for p in box.rglob("*") if p.is_file())
    assert "/data/kaggriculture" not in everything
    assert "router_v1" in (box / "feedback.md").read_text() and "weakest" in (box / "feedback.md").read_text().lower()
    assert "syntax: bad" in (box / "feedback.md").read_text()


def test_cross_sandbox_carries_the_inspiration(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "SANDBOXES", tmp_path)
    parent = tmp_path / "p.py"; parent.write_text("# parent\n")
    other = tmp_path / "q.py"; other.write_text("# inspiration\n")
    box = prompt.build_sandbox("xyz", "cross", parent, other, {}, "", {}, [])
    assert (box / "inspiration.py").read_text() == "# inspiration\n"
    assert "inspiration.py" in (box / "PROMPT.md").read_text()


def test_agents_md_is_the_task_prompt_plus_harness_and_doctrine(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "SANDBOXES", tmp_path)
    parent = tmp_path / "p.py"; parent.write_text("# parent\n")
    box = prompt.build_sandbox("q", "full", parent, None, {}, "", {}, [])
    text = (box / "AGENTS.md").read_text()
    assert "campaign play" in text and "never read their source" in text.lower()
    assert "700000" not in text
```

- [ ] **Step 2: Run to verify they fail** — `ImportError`.

- [ ] **Step 3: Write `prompt.py`**

```python
"""Build the sandbox one codex call works in.

``AGENTS.md`` is codex's standing context: the task prompt phase 1 wrote,
how to use the harness, and the doctrine. ``PROMPT.md`` is FAMOU's mutation
instruction. ``feedback.md`` is what the evaluator said about the parent.
Nothing in the sandbox names where an opponent lives.
"""

import shutil
from pathlib import Path
from typing import Literal

from kaggle_environments.envs.kaggriculture import kaggriculture as engine_module

from kaggriculture.campaign import config

TASK_PROMPT = Path(__file__).with_name("task_prompt.md")

HARNESS_SECTION = """
## Testing what you write

- `campaign check child.py` — loads the file as Kaggle does and plays one
  episode against itself; reports the worst per-step latency (budget 0.5 s).
- `campaign play child.py --vs NAME... --seeds A-B --workers N` — plays the
  named opponents on the engine, both seats, at most 16 games. Opponent
  names are those in feedback.md. Seeds are your choice; the evaluator uses
  others.
The evaluator measures for real after you finish; use these only to make
sure the file runs and does what you intended.
"""

DOCTRINE = """
## Doctrine

Measure opponents through the harness. Never read their source, never ask
for it, never reconstruct it: the gate rejects code that resembles any
opponent's. Your agent is ours.
"""

FULL = """Read AGENTS.md, then parent.py and feedback.md.

Write child.py: a complete agent file implementing the interface in
AGENTS.md. Start from parent.py. Keep what the feedback says is winning;
change what is losing, and say in a docstring at the top what you changed
and why. The weakest opponent is where the score moves most.

Run `campaign check child.py` before you finish. Do not modify parent.py.
"""

CROSS = """Read AGENTS.md, then parent.py, inspiration.py and feedback.md.

Write child.py: a complete agent file implementing the interface in
AGENTS.md, combining the strongest ideas of parent.py and inspiration.py.
Say in a docstring at the top which idea came from which and why the
combination should beat both. Run `campaign check child.py` before you
finish. Do not modify parent.py or inspiration.py.
"""


def build_sandbox(
    program_id: str, kind: Literal["full", "cross"], parent: Path, inspiration: Path | None,
    feedback: dict[str, float], weakest: str, weights: dict[str, float], failures: list[str],
) -> Path:
    box = config.SANDBOXES / program_id
    if box.exists():
        shutil.rmtree(box)
    (box / "engine").mkdir(parents=True)
    (box / "AGENTS.md").write_text(TASK_PROMPT.read_text(encoding="utf-8") + HARNESS_SECTION + DOCTRINE, encoding="utf-8")
    shutil.copy(parent, box / "parent.py")
    if inspiration is not None:
        shutil.copy(inspiration, box / "inspiration.py")
    shutil.copy(engine_module.__file__, box / "engine" / "kaggriculture.py")
    lines = ["# Feedback on parent.py", "", "| opponent | parent win rate | weight |", "| --- | --- | --- |"]
    lines += [f"| {name} | {feedback.get(name, float('nan')):.3f} | {weights.get(name, 0.0):.2f} |" for name in weights]
    lines += ["", f"Weakest opponent: **{weakest}**." if weakest else ""]
    if failures:
        lines += ["", "Recent failures in this lineage (do not repeat):", *[f"- {f}" for f in failures]]
    (box / "feedback.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (box / "PROMPT.md").write_text(CROSS if kind == "cross" else FULL, encoding="utf-8")
    return box
```

The exam-seed constant is not in `AGENTS.md` by construction (the test asserts `700000` is absent); the harness refuses the block regardless.

- [ ] **Step 4: Run to verify they pass**, commit: `git add src/kaggriculture/campaign/prompt.py tests/campaign/test_prompt.py && git commit -m "feat: build the sandbox one mutation call works in"`.

---

### Task 6: Mutators — codex and fake

**Files:**

- Create: `src/kaggriculture/campaign/mutate.py`
- Test: `tests/campaign/test_mutate.py`

**Interfaces:**

- Consumes: `prompt.build_sandbox`, `config.MUTATION_TIMEOUT_SECONDS`, `CALLS`.
- Produces: `class Mutation(BaseModel)`: `program_id: str`, `child: Path | None`, `status: Literal["ok", "no_output", "timeout", "exec_error"]`, `reason: str`, `seconds: float`, `input_tokens: int`, `output_tokens: int`; `class Mutator(Protocol)`: `__call__(self, sandbox: Path, program_id: str) -> Mutation`; `class CodexMutator` (`model: str = "gpt-5.6-sol"`); `class FakeMutator` (`__call__` copies `parent.py` to `child.py` with one constant changed via a caller-supplied `edit: Callable[[str], str]`); `record(mutation: Mutation)` appends to `CALLS`.

- [ ] **Step 1: Write the failing tests**

```python
"""The mutation call, with a fake for every test that is not about codex itself."""

import os
from pathlib import Path

import pytest

from kaggriculture.campaign import mutate


def sandbox(tmp_path: Path) -> Path:
    (tmp_path / "parent.py").write_text("LIMIT = 1\ndef agent(o, c=None):\n    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n")
    (tmp_path / "PROMPT.md").write_text("Write child.py.\n")
    (tmp_path / "AGENTS.md").write_text("rules\n")
    return tmp_path


def test_fake_mutator_writes_a_child_with_the_edit_applied(tmp_path: Path) -> None:
    box = sandbox(tmp_path)
    mutator = mutate.FakeMutator(edit=lambda s: s.replace("LIMIT = 1", "LIMIT = 2"))
    result = mutator(box, "p1")
    assert result.status == "ok" and result.child == box / "child.py"
    assert "LIMIT = 2" in result.child.read_text()


def test_record_appends_one_json_line(tmp_path: Path, monkeypatch) -> None:
    from kaggriculture.campaign import config
    monkeypatch.setattr(config, "CALLS", tmp_path / "calls.jsonl")
    m = mutate.Mutation(program_id="p", child=None, status="timeout", reason="", seconds=1.0, input_tokens=0, output_tokens=0)
    mutate.record(m)
    mutate.record(m)
    assert len((tmp_path / "calls.jsonl").read_text().splitlines()) == 2


def test_codex_mutator_reports_no_output_when_nothing_is_written(tmp_path: Path, monkeypatch) -> None:
    box = sandbox(tmp_path)
    monkeypatch.setattr(mutate.CodexMutator, "COMMAND", ["true"])   # a codex that says nothing
    result = mutate.CodexMutator()(box, "p2")
    assert result.status == "no_output"


def test_codex_mutator_reports_timeout(tmp_path: Path, monkeypatch) -> None:
    box = sandbox(tmp_path)
    monkeypatch.setattr(mutate.CodexMutator, "COMMAND", ["sleep", "5"])
    result = mutate.CodexMutator(timeout=1)(box, "p3")
    assert result.status == "timeout"


@pytest.mark.skipif(not os.environ.get("CAMPAIGN_CODEX"), reason="set CAMPAIGN_CODEX=1 to spend a real codex call")
def test_real_codex_writes_a_child(tmp_path: Path) -> None:
    box = sandbox(tmp_path)
    result = mutate.CodexMutator()(box, "p4")
    assert result.status == "ok" and "def agent" in result.child.read_text()
```

- [ ] **Step 2: Run to verify they fail** — `ImportError`.

- [ ] **Step 3: Write `mutate.py`**

```python
"""One mutation: a codex session in a sandbox, or a fake that edits a constant.

The codex command is a class attribute so a test can replace it with ``true``
or ``sleep``; the rest of the module never changes between the fake and the
real thing.
"""

import json
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel

from kaggriculture.campaign import config


class Mutation(BaseModel):
    program_id: str
    child: Path | None
    status: Literal["ok", "no_output", "timeout", "exec_error"]
    reason: str
    seconds: float
    input_tokens: int
    output_tokens: int


class Mutator(Protocol):
    def __call__(self, sandbox: Path, program_id: str) -> Mutation: ...


class CodexMutator:
    COMMAND = ["codex", "exec", "-s", "workspace-write", "-c", "approval_policy=never", "--json", "-"]

    def __init__(self, model: str = "gpt-5.6-sol", timeout: float = config.MUTATION_TIMEOUT_SECONDS) -> None:
        self.model = model
        self.timeout = timeout

    def __call__(self, sandbox: Path, program_id: str) -> Mutation:
        started = time.perf_counter()
        prompt = (sandbox / "PROMPT.md").read_text(encoding="utf-8")
        command = [*self.COMMAND]
        if command[0] == "codex":
            command += ["-m", self.model, "-C", str(sandbox)]
        log = sandbox / "codex.jsonl"
        try:
            with log.open("w", encoding="utf-8") as handle:
                subprocess.run(command, input=prompt, stdout=handle, stderr=subprocess.PIPE, text=True,
                               cwd=sandbox, timeout=self.timeout, check=True)
        except subprocess.TimeoutExpired:
            return Mutation(program_id=program_id, child=None, status="timeout", reason=f"{self.timeout}s",
                            seconds=time.perf_counter() - started, input_tokens=0, output_tokens=0)
        except subprocess.CalledProcessError as error:
            return Mutation(program_id=program_id, child=None, status="exec_error", reason=(error.stderr or "")[-500:],
                            seconds=time.perf_counter() - started, input_tokens=0, output_tokens=0)
        tokens_in, tokens_out = _tokens(log)
        child = sandbox / "child.py"
        if not child.exists() or not child.read_text(encoding="utf-8").strip():
            return Mutation(program_id=program_id, child=None, status="no_output", reason="child.py missing or empty",
                            seconds=time.perf_counter() - started, input_tokens=tokens_in, output_tokens=tokens_out)
        return Mutation(program_id=program_id, child=child, status="ok", reason="",
                        seconds=time.perf_counter() - started, input_tokens=tokens_in, output_tokens=tokens_out)


def _tokens(log: Path) -> tuple[int, int]:
    """Sum token counts from codex's JSONL; zero when the format carries none."""
    tokens_in = tokens_out = 0
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        usage = event.get("usage") or event.get("token_usage") or {}
        tokens_in += int(usage.get("input_tokens", 0))
        tokens_out += int(usage.get("output_tokens", 0))
    return tokens_in, tokens_out


class FakeMutator:
    """Copies parent.py to child.py through ``edit``. For dry runs and tests."""

    def __init__(self, edit: Callable[[str], str]) -> None:
        self.edit = edit

    def __call__(self, sandbox: Path, program_id: str) -> Mutation:
        child = sandbox / "child.py"
        child.write_text(self.edit((sandbox / "parent.py").read_text(encoding="utf-8")), encoding="utf-8")
        return Mutation(program_id=program_id, child=child, status="ok", reason="", seconds=0.0, input_tokens=0, output_tokens=0)


def record(mutation: Mutation) -> None:
    config.CALLS.parent.mkdir(parents=True, exist_ok=True)
    with config.CALLS.open("a", encoding="utf-8") as handle:
        handle.write(mutation.model_dump_json() + "\n")
```

Run one real call (`CAMPAIGN_CODEX=1 uv run pytest tests/campaign/test_mutate.py -k real -v`) and read `codex.jsonl` to confirm which event carries token usage in codex 0.147's `--json` output; fix `_tokens` to that shape and note it in the docstring.

- [ ] **Step 4: Run to verify they pass**, commit: `git add src/kaggriculture/campaign/mutate.py tests/campaign/test_mutate.py && git commit -m "feat: the mutation call, with a fake for everything that is not codex"`.

---

### Task 7: The gate — promotion and the floor

**Files:**

- Create: `src/kaggriculture/campaign/gate.py`
- Test: `tests/campaign/test_gate.py`

**Interfaces:**

- Consumes: `evaluator.DeepResult`, `pool.Pool`, `config.FLOOR`, `EPOCHS`, `POOL`.
- Produces: `promotion(candidate: DeepResult, champion: DeepResult | None) -> tuple[bool, str]`; `promote(program: archive.Program, result: DeepResult, pool: Pool, commit: bool = True) -> str` (returns the champion's pool name; writes `FLOOR/main.py` at 0o444, copies to `src/kaggriculture/served/main.py`, `pool.add_champion`, `pool.apply_weakness_pressure(pool.weakest(result.rates))`, saves `POOL`, appends an epoch line, `git commit` of `served/main.py` when `commit`); `epoch_line(iteration: int, results: list[DeepResult], promoted: str | None, rho: float | None) -> None`.

- [ ] **Step 1: Write the failing tests**

```python
"""The promotion rule on hand-built numbers, and the floor write."""

import os
from pathlib import Path

from kaggriculture.campaign import archive, config, evaluator, gate, pool


def result(pid: str, score: float, low: float, rates: dict[str, float], width: float = 0.05) -> evaluator.DeepResult:
    return evaluator.DeepResult(
        program_id=pid, score=score, low=low, high=min(1.0, score + (score - low)), rates=rates,
        intervals={n: (max(0.0, r - width), min(1.0, r + width)) for n, r in rates.items()},
        field=score, held_out={}, games=128,
    )


def test_no_champion_promotes_anything_with_a_score() -> None:
    ok, why = gate.promotion(result("c", 0.3, 0.25, {"a": 0.3}), None)
    assert ok


def test_lower_bound_must_beat_the_champion_point_estimate() -> None:
    champ = result("k", 0.60, 0.55, {"a": 0.6})
    assert not gate.promotion(result("c", 0.62, 0.58, {"a": 0.62}), champ)[0]
    assert gate.promotion(result("c", 0.70, 0.65, {"a": 0.70}), champ)[0]


def test_field_may_not_drop_more_than_two_points() -> None:
    champ = result("k", 0.60, 0.55, {"a": 0.6})
    candidate = result("c", 0.70, 0.65, {"a": 0.70})
    candidate = candidate.model_copy(update={"field": 0.57})
    assert not gate.promotion(candidate, champ)[0]


def test_no_opponent_may_regress_beyond_noise() -> None:
    champ = result("k", 0.60, 0.55, {"a": 0.9, "b": 0.3})
    candidate = result("c", 0.70, 0.65, {"a": 0.6, "b": 0.8})
    ok, why = gate.promotion(candidate, champ)
    assert not ok and "a" in why


def test_promote_writes_a_read_only_floor_and_updates_the_pool(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "FLOOR", tmp_path / "floor" / "agent")
    monkeypatch.setattr(config, "POOL", tmp_path / "pool.json")
    monkeypatch.setattr(config, "EPOCHS", tmp_path / "epochs.jsonl")
    monkeypatch.setattr(gate, "SERVED", tmp_path / "served" / "main.py")
    source = tmp_path / "prog.py"
    source.write_text("def agent(o, c=None):\n    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n")
    program = archive.Program(id="p9", island=0, source_path=str(source), parents=[], kind="full", fitness_sum=0.5, n_evals=1, status="ok", reason="", created=0.0)
    p = pool.Pool(opponents={"a": "/x/a.py", "b": "/x/b.py"}, weights={"a": 0.5, "b": 0.5})
    name = gate.promote(program, result("p9", 0.7, 0.65, {"a": 0.9, "b": 0.5}), p, commit=False)
    assert name == "champion_1"
    floor = config.FLOOR / "main.py"
    assert floor.read_text() == source.read_text() and (floor.stat().st_mode & 0o777) == 0o444
    assert gate.SERVED.read_text() == source.read_text()
    saved = pool.Pool.load(config.POOL)
    assert "champion_1" in saved.names() and saved.weights["b"] > saved.weights["a"]
    assert "champion_1" in (tmp_path / "epochs.jsonl").read_text()
```

- [ ] **Step 2: Run to verify they fail** — `ImportError`.

- [ ] **Step 3: Write `gate.py`**

```python
"""The one place a candidate becomes the floor.

The promotion rule is the spec's §5.5: the candidate's lower Wilson bound
above the champion's point estimate, the vendored field not down more than
two points, no opponent regressed beyond the wider of the two intervals.
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from kaggriculture.campaign import config
from kaggriculture.campaign.archive import Program
from kaggriculture.campaign.evaluator import DeepResult
from kaggriculture.campaign.pool import Pool

SERVED = config.ROOT / "src" / "kaggriculture" / "served" / "main.py"
FIELD_TOLERANCE = 0.02


def promotion(candidate: DeepResult, champion: DeepResult | None) -> tuple[bool, str]:
    """Whether ``candidate`` replaces ``champion``, and why not if not."""
    if champion is None:
        return True, "no champion yet"
    if candidate.low <= champion.score:
        return False, f"lower bound {candidate.low:.3f} <= champion {champion.score:.3f}"
    if candidate.field < champion.field - FIELD_TOLERANCE:
        return False, f"field {candidate.field:.3f} < champion {champion.field:.3f} - {FIELD_TOLERANCE}"
    for name, rate in champion.rates.items():
        if name not in candidate.rates:
            continue
        width = max(champion.intervals[name][1] - champion.intervals[name][0], candidate.intervals[name][1] - candidate.intervals[name][0])
        if candidate.rates[name] < rate - width:
            return False, f"regressed against {name}: {candidate.rates[name]:.3f} < {rate:.3f} - {width:.3f}"
    return True, "promoted"


def promote(program: Program, result: DeepResult, pool: Pool, commit: bool = True) -> str:
    """Write the floor, register the champion in the pool, record the epoch, commit."""
    number = 1 + sum(1 for n in pool.names() if n.startswith("champion_"))
    name = f"champion_{number}"
    source = Path(program.source_path).read_text(encoding="utf-8")
    config.FLOOR.mkdir(parents=True, exist_ok=True)
    floor = config.FLOOR / "main.py"
    if floor.exists():
        os.chmod(floor, 0o644)
    floor.write_text(source, encoding="utf-8")
    os.chmod(floor, 0o444)
    SERVED.parent.mkdir(parents=True, exist_ok=True)
    SERVED.write_text(source, encoding="utf-8")
    pool.add_champion(name, str(floor), result.rates)
    pool.apply_weakness_pressure(pool.weakest(result.rates))
    pool.save(config.POOL)
    epoch_line(iteration=-1, results=[result], promoted=name, rho=None)
    if commit:
        subprocess.run(["git", "add", str(SERVED)], check=True, cwd=config.ROOT)
        subprocess.run(["git", "commit", "-q", "-m", f"feat: promote {program.id} to the floor as {name}\n\ndeep {result.score:.4f} [{result.low:.4f}, {result.high:.4f}], field {result.field:.4f}"],
                       check=True, cwd=config.ROOT)
    return name


def epoch_line(iteration: int, results: list[DeepResult], promoted: str | None, rho: float | None) -> None:
    config.EPOCHS.parent.mkdir(parents=True, exist_ok=True)
    with config.EPOCHS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "ts": time.time(), "iteration": iteration, "promoted": promoted, "rho_fast_deep": rho,
            "results": [r.model_dump() for r in results],
        }) + "\n")
```

- [ ] **Step 4: Run to verify they pass**, commit: `git add src/kaggriculture/campaign/gate.py tests/campaign/test_gate.py && git commit -m "feat: the promotion rule and the only writer of the floor"`.

---

### Task 8: The loop and the dry run

**Files:**

- Create: `src/kaggriculture/campaign/loop.py`
- Modify: `src/kaggriculture/campaign/harness.py` (add `loop` and `dry-run` subcommands delegating to `loop.main`)
- Test: `tests/campaign/test_loop.py`

**Interfaces:**

- Consumes: everything above.
- Produces: `class State(BaseModel)`: `iteration: int`, `champion: DeepResult | None`, `calls_today: int`, `day: str`; `run(iterations: int, mutator: Mutator, workers: int, concurrency: int, seed_agent: Path, rng: random.Random) -> State`; `iterate(state, archive, pool, mutator, workers, concurrency, rng) -> State` (one iteration: one mutation per island in parallel, validate, fast eval, insert; then migration / epoch / reset on schedule); `epoch(state, archive, pool, workers) -> State`; `main()`.

- [ ] **Step 1: Write the failing test**

```python
"""The whole loop on a fake mutator and a two-opponent pool, end to end."""

import random
from pathlib import Path

from kaggriculture.campaign import config, loop, mutate, pool

PASS = "def agent(observation, configuration=None):\n    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
SELLER = '''
def agent(observation, configuration=None):
    step = observation["step"]
    farm = observation["farms"][observation["player"]]
    shed = observation["private"]["shed"]
    market = [["BUY_SEED", "WHEAT", 5]] if step == 0 else []
    if shed.get("WHEAT", 0):
        market.append(["SELL", "WHEAT", shed["WHEAT"]])
    x, y = farm["farmer"]
    tile = farm["tiles"][y][x]
    seeds = observation["private"]["seeds"]
    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        farmer = ["HARVEST"] if tile.get("yield_units", 0) else (["WATER"] if not tile.get("watered_today") else ["EAST"])
    elif tile is None and seeds.get("WHEAT", 0):
        farmer = ["PLANT", "WHEAT"]
    else:
        farmer = ["EAST"] if x < 4 else ["PASS"]
    return {"farmer": farmer, "hands": [], "market": market}
'''


def test_dry_run_promotes_a_better_child_over_a_pass_floor(tmp_path: Path, monkeypatch) -> None:
    run = tmp_path / "run"
    for name in ("ARCHIVE", "PROGRAMS", "SANDBOXES", "POOL", "EPOCHS", "FLOOR", "CALLS"):
        monkeypatch.setattr(config, name, run / getattr(config, name).relative_to(config.RUN))
    monkeypatch.setattr(config, "EXAM_SEEDS", config.EXAM_SEEDS[:2])
    monkeypatch.setattr(config, "EPOCH_INTERVAL", 1)
    monkeypatch.setattr(config, "ISLANDS", 2)
    monkeypatch.setattr(config, "ISLAND_SIZE", 3)
    monkeypatch.setattr(config, "FAST_SEEDS", 1)
    monkeypatch.setattr(loop.gate, "SERVED", tmp_path / "served.py")
    pass_agent = tmp_path / "pass.py"
    pass_agent.write_text(PASS)
    seed = tmp_path / "seed.py"
    seed.write_text(PASS)
    p = pool.Pool(opponents={"pass": str(pass_agent)}, weights={"pass": 1.0})
    p.save(config.POOL)
    monkeypatch.setattr(loop.roster, "path", lambda name: Path(pool.Pool.load(config.POOL).opponents[name]))
    monkeypatch.setattr(loop.evaluator, "HELD_OUT", [])
    mutator = mutate.FakeMutator(edit=lambda _: SELLER)
    state = loop.run(iterations=1, mutator=mutator, workers=4, concurrency=2, seed_agent=seed, rng=random.Random(0))
    assert state.champion is not None and state.champion.score > 0.5
    assert (config.FLOOR / "main.py").read_text() == SELLER
    assert "champion_1" in pool.Pool.load(config.POOL).names()
    assert config.EPOCHS.read_text().count("\n") >= 1
```

- [ ] **Step 2: Run to verify it fails** — `ImportError`.

- [ ] **Step 3: Write `loop.py`**

```python
"""Algorithm 1: mutate, validate, fast-evaluate, insert; migrate, deep-evaluate, reset on schedule."""

import argparse
import datetime
import logging
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel

from kaggriculture.campaign import archive as archive_module
from kaggriculture.campaign import config, evaluator, gate, mutate, prompt, roster, validate
from kaggriculture.campaign.evaluator import DeepResult
from kaggriculture.campaign.mutate import Mutator
from kaggriculture.campaign.pool import Pool

LOGGER = logging.getLogger(__name__)
STATE = config.RUN / "state.json"


class State(BaseModel):
    iteration: int = 0
    champion: DeepResult | None = None
    calls_today: int = 0
    day: str = ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10**9)
    parser.add_argument("--workers", type=int, default=config.CORE_BUDGET // 2)
    parser.add_argument("--concurrency", type=int, default=config.CODEX_CONCURRENCY)
    parser.add_argument("--seed-agent", type=Path, default=gate.SERVED)
    parser.add_argument("--dry-run", action="store_true", help="fake mutator: copies the parent and bumps a constant")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    mutator: Mutator = mutate.FakeMutator(edit=lambda s: s) if args.dry_run else mutate.CodexMutator()
    run(args.iterations, mutator, args.workers, args.concurrency, args.seed_agent, random.Random())


def run(iterations: int, mutator: Mutator, workers: int, concurrency: int, seed_agent: Path, rng: random.Random) -> State:
    state = State.model_validate_json(STATE.read_text()) if STATE.exists() else State()
    pool = Pool.load(config.POOL) if config.POOL.exists() else Pool.initial()
    pool.save(config.POOL)
    archive = archive_module.Archive()
    if not any(archive.island(i) for i in range(config.ISLANDS)):
        fitness = evaluator.fast(seed_agent, pool, rng, workers).fitness
        archive.seed(seed_agent, fitness)
        LOGGER.info("seeded every island from %s at fast fitness %.3f", seed_agent, fitness)
    for _ in range(iterations):
        state = iterate(state, archive, pool, mutator, workers, concurrency, rng)
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    return state


def iterate(state: State, archive: archive_module.Archive, pool: Pool, mutator: Mutator, workers: int, concurrency: int, rng: random.Random) -> State:
    today = datetime.date.today().isoformat()
    if state.day != today:
        state = state.model_copy(update={"day": today, "calls_today": 0})
    if state.calls_today >= config.DAILY_CALL_BUDGET:
        LOGGER.warning("daily call budget spent; evaluating only")
    else:
        jobs = [(island, _plan(archive, island, rng)) for island in range(config.ISLANDS)]
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            results = list(executor.map(lambda job: _mutate(job, archive, pool, mutator, workers, rng), jobs))
        state = state.model_copy(update={"calls_today": state.calls_today + len(results)})
    iteration = state.iteration + 1
    state = state.model_copy(update={"iteration": iteration})
    if iteration % config.MIGRATION_INTERVAL == 0:
        archive.migrate()
    if iteration % config.EPOCH_INTERVAL == 0:
        state = epoch(state, archive, pool, workers)
    if iteration % config.RESET_INTERVAL == 0 and archive.top(1):
        archive.reset_worst_island(archive.top(1)[0])
    return state


def _plan(archive: archive_module.Archive, island: int, rng: random.Random) -> tuple[str, archive_module.Program, archive_module.Program | None]:
    parent = archive.ucb_parent(island, rng)
    if rng.random() < config.CROSS_PROBABILITY and len(archive.island(island)) > 1:
        inspiration = archive.ucb_parent(island, rng)
        if inspiration.id != parent.id:
            return "cross", parent, inspiration
    return "full", parent, None


def _mutate(job, archive: archive_module.Archive, pool: Pool, mutator: Mutator, workers: int, rng: random.Random) -> None:
    island, (kind, parent, inspiration) = job
    program_id = f"i{island}-{int(rng.random() * 1e9):09d}"
    failures = [f.reason for f in archive.failures() if parent.id in f.parents][-3:]
    box = prompt.build_sandbox(program_id, kind, Path(parent.source_path), Path(inspiration.source_path) if inspiration else None,
                               feedback=_last_rates(parent), weakest=pool.weakest(_last_rates(parent)) if _last_rates(parent) else "",
                               weights=pool.weights, failures=failures)
    mutation = mutator(box, program_id)
    mutate.record(mutation)
    parents = [parent.id] + ([inspiration.id] if inspiration else [])
    if mutation.status != "ok" or mutation.child is None:
        archive.record_failure(island, parents, kind, f"{mutation.status}: {mutation.reason}")
        return
    verdict = validate.validate(mutation.child)
    if verdict.status != "ok":
        archive.record_failure(island, parents, kind, f"{verdict.status}: {verdict.reason}")
        return
    stored = archive.store(mutation.child.read_text(encoding="utf-8"), program_id)
    result = evaluator.fast(stored, pool, rng, workers)
    program = archive_module.Program(id=program_id, island=island, source_path=str(stored), parents=parents, kind=kind,
                                     fitness_sum=result.fitness, n_evals=1, status="ok", reason="", created=mutation.seconds)
    _RATES[program_id] = result.rates
    replaced = archive.insert(program)
    LOGGER.info("%s %s fast %.3f (%s)", program_id, kind, result.fitness, "dropped" if replaced is program else "inserted")
    shutil.rmtree(box, ignore_errors=True)


_RATES: dict[str, dict[str, float]] = {}


def _last_rates(program: archive_module.Program) -> dict[str, float]:
    return _RATES.get(program.id, {})


def epoch(state: State, archive: archive_module.Archive, pool: Pool, workers: int) -> State:
    """Deep-evaluate the top K on the exam block; promote if the rule says so."""
    candidates = archive.top(config.DEEP_TOP_K)
    results = [evaluator.deep(Path(p.source_path), p.id, pool, workers) for p in candidates]
    rho = _spearman([p.mean for p in candidates], [r.score for r in results]) if len(results) > 2 else None
    best = max(results, key=lambda r: r.score, default=None)
    promoted = None
    if best is not None:
        ok, why = gate.promotion(best, state.champion)
        LOGGER.info("epoch %d: best deep %.4f [%.4f, %.4f] — %s", state.iteration, best.score, best.low, best.high, why)
        if ok:
            program = next(p for p in candidates if p.id == best.program_id)
            promoted = gate.promote(program, best, pool)
            state = state.model_copy(update={"champion": best})
    gate.epoch_line(state.iteration, results, promoted, rho)
    return state


def _spearman(a: list[float], b: list[float]) -> float:
    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        for rank, index in enumerate(order):
            out[index] = float(rank)
        return out
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    return 1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb)) / (n * (n * n - 1))


if __name__ == "__main__":
    main()
```

Per-program fast rates (`_RATES`) are process memory; persist them by adding a `rates: dict[str, float]` field to `Program` (Task 3's model) written at insert, and delete `_RATES` — the dry-run test does not care which, the ledger does. Do the field.

- [ ] **Step 4: Run the dry run test until it passes**

Run: `uv run pytest tests/campaign/test_loop.py -v`
Expected: PASS in a few minutes (fast eval + a 4-game deep eval on the port).

- [ ] **Step 5: Wire the CLI and the runbook**

Add to `harness.main`: subcommands `loop` and `dry-run` that call `loop.main` (with `--dry-run` set for the latter). Append to `README.md` a "Running the campaign" section:

```
uv run campaign dry-run --iterations 2         # fake mutator, proves the pipeline
nohup uv run campaign loop > run/campaign/loop.log 2>&1 &
tail -f run/campaign/loop.log; tail -1 run/campaign/epochs.jsonl | python -m json.tool
```

- [ ] **Step 6: Commit**

```bash
git add src/kaggriculture/campaign/loop.py src/kaggriculture/campaign/harness.py src/kaggriculture/campaign/archive.py tests/campaign/test_loop.py README.md
git commit -m "feat: the evolution loop, proven end to end on a fake mutator"
```

---

## Self-review

**Spec coverage.** §2 loop: Task 8. §3 hyperparameters: Task 1. §5.2 mutation call: Tasks 5–6. §5.4 validation: Plan 1 (`validate.validate`, consumed in Task 8). §5.5 evaluators + promotion: Tasks 4, 7. §5.6 pool: Task 2. §5.7 shipping stays manual (`uv run submit`). §6 phase-0 dry run: Task 8's test and `campaign dry-run`. Held-out set reported every epoch: `DeepResult.held_out` in the epoch line. ρ(fast, deep): `epoch`. Quota: `DAILY_CALL_BUDGET` in `iterate`. Kernel-watch feeding the held-out set is an operational step (its output names go into `roster.HELD_OUT` by hand, then into the pool by a recorded decision) — no code beyond Plan 1's move.

**Placeholders.** The codex JSONL token shape is verified by one real call in Task 6 rather than assumed.

**Type consistency.** `Program` gains `rates` in Task 8 (noted); `DeepResult` fields used by `gate.promotion` are those defined in Task 4; `Pool.weighted`/`weakest`/`names` used by evaluator, gate and loop match Task 2; `Mutation.child: Path | None` checked in `_mutate`.
