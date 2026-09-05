# The Campaign Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the campaign loop with the system in
`docs/superpowers/specs/2026-09-05-campaign-script-design.md`: eight coding
agents run at once, each handed the champion and told to beat it, and what
beats it on the sealed exam block becomes the new champion.

**Architecture:** One asyncio event loop owns one shared database, one
pool and one champion. Eight worker coroutines each run mutate → validate
→ fast-evaluate → insert forever; a program entering the top three fires a
deep evaluation that may promote. Every blocking call goes to a thread or
a subprocess.

**Tech Stack:** Python 3.11, asyncio, pydantic, wandb, GitPython, a C++
engine port through ctypes, `codex exec` as the mutation operator.

## Global Constraints

Copied from the spec; every task's requirements include these.

- `loop.py` is under 300 lines including docstrings, `main()` at the top,
  then functions in call order.
- The four steps of spec §1 are visible as four steps in `main()`.
- Nothing blocks the event-loop thread: the session is an asyncio
  subprocess; validate, fast, deep, packaging, the image load test and any
  file tree removal go through `asyncio.to_thread`.
- The database, the pool, the state and the wandb run are touched only
  from coroutines on the loop thread. No locks. An evaluation receives a
  deep copy of the pool taken on the loop thread.
- **Deliberately not built** (spec §9): islands, migration, island resets,
  MAP-Elites, UCB parent selection, a call cap, meta-prompt evolution,
  LLM-graded feedback, diffs, multiple scores, sessions surviving restart.
  Deleting the code for these is part of the work, not a follow-up.
- The loop never runs git. Every write is under `run/campaign`.
- No latency measurement, threshold, or report anywhere.
- Exactly two time limits, both liveness: `SESSION_LIMIT_SECONDS = 1500`
  and `GAME_LIMIT_SECONDS = 120`.
- Tests use real code. Only the codex process may be substituted
  (`FakeMutator`, or `CodexMutator` with its `COMMAND` replaced by a real
  shell command). Real validate, real games through the real harness and
  process pools, real database, pool, gate and files under `tmp_path`, and
  a real `wandb.init(mode="disabled")` run. Never stub `evaluator`,
  `validate`, `harness`, the gate, or a process object.
- Every new test is shown red by a named mutation of the new code; state
  the mutation in the task's report.
- Style: no defensive fallbacks, raise concise errors for unsupported
  cases, Google docstrings, `logging` not `print`, no
  `from __future__ import annotations`, minimise helpers, absolute
  imports, fix type errors rather than ignoring them.
- Commits: explicit paths, never `git add -A`, never `--no-verify` or
  `SKIP=`. Conventional messages that say why. Nothing about Claude.
- **The live loop is running from this worktree on the old code.** Do not
  kill it, do not touch `run/campaign/` or
  `/data/kaggriculture/campaign/pool.json`. Never run `pkill -f` with a
  pattern that also appears elsewhere in the same command.
- Foreground only: run tests and commits as ordinary foreground commands
  with a 600000 ms timeout. `uv run pytest tests/campaign -q` is about two
  minutes; `uv run pre-commit run -a` about three.

---

## File Structure

| file                                                                                                                   | change                                                                                                                                                       |
| ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `src/kaggriculture/campaign/config.py`                                                                                 | constants: delete the island and cap set, add `SESSIONS`, `DEEP_CONCURRENCY`, `STAGNATION_SESSIONS`, `GAME_LIMIT_SECONDS`, rename `MUTATION_TIMEOUT_SECONDS` |
| `src/kaggriculture/campaign/archive.py`                                                                                | becomes the shared database: no islands, no UCB, no migration, no reset; gains `deep` records                                                                |
| `src/kaggriculture/campaign/prompt.py`                                                                                 | sandbox holds `child.py` (the champion) not `parent.py`; five `full` instructions; the bar in `feedback.md`                                                  |
| `src/kaggriculture/campaign/gate.py`                                                                                   | `promote` packages a tarball and image-tests it; `commit_floor` deleted                                                                                      |
| `src/kaggriculture/campaign/loop.py`                                                                                   | rewritten: `main`, `campaign`, `worker`, `gate`, under 300 lines                                                                                             |
| `src/kaggriculture/campaign/mutate.py`                                                                                 | unchanged except the constant rename                                                                                                                         |
| `src/kaggriculture/campaign/harness.py`, `evaluator.py`, `pool.py`, `validate.py`, `copycheck.py`, `kaggle_image.py`   | unchanged                                                                                                                                                    |
| `tests/campaign/test_archive.py`, `test_prompt.py`, `test_gate.py`, `test_loop.py`, `test_config.py`, `test_mutate.py` | follow their modules                                                                                                                                         |

Tasks are ordered so each leaves the tree green: constants, then the
database, then the prompt, then the gate, then the loop that uses them.

---

### Task 1: Constants

**Files:**

- Modify: `src/kaggriculture/campaign/config.py`
- Modify: `src/kaggriculture/campaign/mutate.py` (the rename only)
- Test: `tests/campaign/test_config.py`

**Interfaces:**

- Produces: the constant names every later task uses.

- [ ] **Step 1: Write the failing test**

Replace the spec-table test in `tests/campaign/test_config.py` with:

```python
def test_constants_match_the_spec_table() -> None:
    """Spec section 8: the campaign's constants, and only these."""
    assert config.SESSIONS == 8
    assert config.SESSION_LIMIT_SECONDS == 1500
    assert config.GAME_LIMIT_SECONDS == 120
    assert config.FAST_SEEDS == 4
    assert len(config.EXAM_SEEDS) == 64
    assert (config.DEEP_TOP_K, config.DEEP_CONCURRENCY) == (3, 2)
    assert config.CHAMPION_WEIGHT == 0.20
    assert (config.POOL_CAP, config.RETIRE_THRESHOLD) == (10, 0.95)
    assert config.WEAKNESS_CAP == 0.5
    assert config.STAGNATION_SESSIONS == 40
    assert config.CODEX_MODEL == "gpt-6-astra"
    assert config.CODEX_FALLBACK_MODEL == "gpt-5.6-sol"


def test_the_deleted_constants_are_gone() -> None:
    """Islands, epochs and the call cap are not part of this system."""
    for name in (
        "ISLANDS",
        "ISLAND_SIZE",
        "MIGRANTS",
        "MIGRATION_INTERVAL",
        "RESET_INTERVAL",
        "UCB_C",
        "EPOCH_INTERVAL",
        "DAILY_CALL_BUDGET",
        "CODEX_CONCURRENCY",
        "MUTATION_TIMEOUT_SECONDS",
        "CHECK_TIMEOUT_SECONDS",
        "WANDB_RUN_ID",
    ):
        assert not hasattr(config, name), name
```

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run pytest tests/campaign/test_config.py -q`
Expected: FAIL, `AttributeError` on `config.SESSIONS`.

- [ ] **Step 3: Edit `config.py`**

Delete `ISLANDS`, `ISLAND_SIZE`, `MIGRANTS`, `MIGRATION_INTERVAL`,
`RESET_INTERVAL`, `UCB_C`, `EPOCH_INTERVAL`, `DAILY_CALL_BUDGET`,
`CODEX_CONCURRENCY`, `CHECK_TIMEOUT_SECONDS`, `WANDB_RUN_ID` and their
comments. Rename `MUTATION_TIMEOUT_SECONDS` to `SESSION_LIMIT_SECONDS`.
Add, with the comments shown:

```python
# One codex session per worker; eight fit the machine beside their
# evaluations. Spec section 8.
SESSIONS = 8
# Deep evaluations in flight at once. Each is about ten minutes of games.
DEEP_CONCURRENCY = 2
# Sessions without a promotion before a session starts from a program
# drawn from the database's top ten instead of the champion.
STAGNATION_SESSIONS = 40
# Liveness only, never speed: a 720-turn game of rule-based policies takes
# well under a second, so one still running after two minutes is stuck.
GAME_LIMIT_SECONDS = 120
```

`CROSS_PROBABILITY` stays. In `mutate.py`, the `CodexMutator.__init__`
default becomes `config.SESSION_LIMIT_SECONDS`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/campaign/test_config.py tests/campaign/test_mutate.py -q`
Expected: PASS. Other modules still import deleted names and will fail;
that is Tasks 2–5.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/campaign/config.py src/kaggriculture/campaign/mutate.py tests/campaign/test_config.py
git commit -m "refactor: the campaign's constants are the eight in the spec"
```

---

### Task 2: The shared database

**Files:**

- Modify: `src/kaggriculture/campaign/archive.py`
- Test: `tests/campaign/test_archive.py`

**Interfaces:**

- Consumes: `config` from Task 1.
- Produces:
  - `class Program(BaseModel)`: `id: str`, `source_path: str`,
    `started_from: str` (the id this was edited from, `""` for the seed),
    `instruction: str`, `fitness: float`, `rates: dict[str, float]`,
    `created: float`, `deep: DeepResult | None = None`.
  - `class Failure(BaseModel)`: `started_from: str`, `instruction: str`,
    `reason: str`, `created: float`.
  - `class Database`: `__init__(path: Path, programs_dir: Path)`,
    `store(source: str, program_id: str) -> Path`,
    `add(program: Program) -> None`,
    `record_failure(failure: Failure) -> None`,
    `record_deep(program_id: str, result: DeepResult) -> None`,
    `top(k: int) -> list[Program]` (by fitness, descending),
    `get(program_id: str) -> Program`,
    `failures(started_from: str) -> list[Failure]`,
    `programs -> list[Program]` (property).

- [ ] **Step 1: Write the failing tests**

Replace `tests/campaign/test_archive.py` wholesale:

```python
"""The shared database: every program, its scores, and what failed."""

import time
from pathlib import Path

import pytest

from kaggriculture.campaign import archive
from kaggriculture.campaign.evaluator import DeepResult

AGENT = "def agent(o, c=None):\n    return {}\n"


def make(tmp_path: Path) -> archive.Database:
    """A fresh database backed by files under `tmp_path`."""
    return archive.Database(tmp_path / "db.jsonl", tmp_path / "programs")


def program(db: archive.Database, name: str, fitness: float) -> archive.Program:
    """Store a source and add a program with `fitness`."""
    p = archive.Program(
        id=name,
        source_path=str(db.store(AGENT, name)),
        started_from="",
        instruction="improve",
        fitness=fitness,
        rates={"v54": fitness},
        created=time.time(),
    )
    db.add(p)
    return p


def test_top_ranks_by_fitness(tmp_path: Path) -> None:
    """`top` is the best programs by fitness, best first."""
    db = make(tmp_path)
    program(db, "a", 0.1)
    program(db, "b", 0.7)
    program(db, "c", 0.4)
    assert [p.id for p in db.top(2)] == ["b", "c"]


def test_a_deep_result_is_stored_on_the_program(tmp_path: Path) -> None:
    """A confirmed program carries its deep result."""
    db = make(tmp_path)
    program(db, "a", 0.5)
    result = DeepResult(
        program_id="a",
        score=0.6,
        low=0.5,
        high=0.7,
        rates={"v54": 0.6},
        intervals={"v54": (0.5, 0.7)},
        field=0.6,
        held_out={},
        games=128,
    )
    db.record_deep("a", result)
    assert db.get("a").deep == result


def test_the_log_survives_a_restart(tmp_path: Path) -> None:
    """Everything replays: programs, deep results and failures."""
    db = make(tmp_path)
    program(db, "a", 0.5)
    db.record_deep("a", DeepResult(
        program_id="a", score=0.6, low=0.5, high=0.7, rates={}, intervals={},
        field=0.6, held_out={}, games=128,
    ))
    db.record_failure(archive.Failure(
        started_from="a", instruction="improve", reason="syntax: bad",
        created=time.time(),
    ))

    again = archive.Database(tmp_path / "db.jsonl", tmp_path / "programs")

    assert [p.id for p in again.programs] == ["a"]
    assert again.get("a").deep is not None
    assert [f.reason for f in again.failures("a")] == ["syntax: bad"]


def test_failures_are_looked_up_by_what_they_started_from(tmp_path: Path) -> None:
    """The prompt shows a lineage only its own failures."""
    db = make(tmp_path)
    for started, reason in (("a", "one"), ("b", "two"), ("a", "three")):
        db.record_failure(archive.Failure(
            started_from=started, instruction="improve", reason=reason,
            created=time.time(),
        ))
    assert [f.reason for f in db.failures("a")] == ["one", "three"]


def test_get_raises_for_an_unknown_program(tmp_path: Path) -> None:
    """An id the database does not hold is a programming error, not a None."""
    with pytest.raises(KeyError, match="nope"):
        make(tmp_path).get("nope")
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/campaign/test_archive.py -q`
Expected: FAIL, `AttributeError: module ... has no attribute 'Database'`.

- [ ] **Step 3: Rewrite `archive.py`**

Delete `Kind`, the island field, `island()`, `ucb_parent`, `migrate`,
`reset_worst_island`, `seed`, `insert`, `mean`, `fitness_sum`, `n_evals`,
`status`, `parents`, `kind`. The module docstring says what it is: an
append-only log of every program, replayed on start, shared by every
worker, with no eviction and no population structure.

```python
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
```

`Database` keeps `self._programs: dict[str, Program]` in insertion order
and `self._failures: list[Failure]`. `_apply` handles three event types,
`program`, `deep`, `failure`; an unknown type raises `ValueError` naming
it. `add`, `record_failure` and `record_deep` each append one JSON line
and update memory. `top(k)` sorts by `fitness` descending. `store` writes
`programs_dir / f"{program_id}.py"` and returns it.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/campaign/test_archive.py -q`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/campaign/archive.py tests/campaign/test_archive.py
git commit -m "refactor: one shared database of programs, with no population structure"
```

---

### Task 3: The sandbox and the five instructions

**Files:**

- Modify: `src/kaggriculture/campaign/prompt.py`
- Test: `tests/campaign/test_prompt.py`

**Interfaces:**

- Consumes: `config` from Task 1.
- Produces:
  - `INSTRUCTIONS: tuple[tuple[str, str], ...]` — five `(name, text)`
    pairs for `full`, plus `CROSS`.
  - `build_sandbox(program_id: str, champion: Path, instruction: str,
inspiration: Path | None, bar: dict[str, float], rates: dict[str,
float], weights: dict[str, float], weakest: str, failures: list[str],
started_from: str) -> Path`

- [ ] **Step 1: Write the failing tests**

Add to `tests/campaign/test_prompt.py`, keeping the existing path-leak
test:

```python
def test_the_sandbox_holds_the_champion_as_child_py(tmp_path: Path) -> None:
    """A session edits the champion in place; there is no parent.py."""
    monkeypatched_sandbox(tmp_path)
    champion = tmp_path / "champion.py"
    champion.write_text("def agent(o, c=None):\n    return {'x': 1}\n")

    box = prompt.build_sandbox(
        "p1", champion, prompt.INSTRUCTIONS[0][1], None,
        bar={"fitness": 0.42}, rates={"v54": 0.3}, weights={"v54": 1.0},
        weakest="v54", failures=[], started_from="champion_1",
    )

    assert (box / "child.py").read_text() == champion.read_text()
    assert not (box / "parent.py").exists()


def test_the_feedback_states_the_bar(tmp_path: Path) -> None:
    """The session is told the number it has to beat."""
    monkeypatched_sandbox(tmp_path)
    champion = tmp_path / "champion.py"
    champion.write_text("def agent(o, c=None):\n    return {}\n")

    box = prompt.build_sandbox(
        "p2", champion, prompt.INSTRUCTIONS[0][1], None,
        bar={"fitness": 0.42}, rates={"v54": 0.3}, weights={"v54": 1.0},
        weakest="v54", failures=[], started_from="champion_1",
    )

    feedback = (box / "feedback.md").read_text()
    assert "0.42" in feedback and "v54" in feedback


def test_there_are_five_full_instructions_and_they_differ() -> None:
    """FAMOU C.2's variants, so eight sessions on one champion diverge."""
    names = [name for name, _ in prompt.INSTRUCTIONS]
    texts = [text for _, text in prompt.INSTRUCTIONS]
    assert len(prompt.INSTRUCTIONS) == 5
    assert len(set(names)) == 5 and len(set(texts)) == 5
    assert all("child.py" in text for text in texts)
```

`monkeypatched_sandbox` is a helper the existing module already needs:
point `config.SANDBOXES` at `tmp_path / "sandboxes"`.

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/campaign/test_prompt.py -q`
Expected: FAIL, `AttributeError: ... 'INSTRUCTIONS'`.

- [ ] **Step 3: Edit `prompt.py`**

Replace `FULL` with:

```python
# FAMOU appendix C.2's five rewrite instructions. One is drawn per
# session, so eight sessions starting from the same champion are pushed
# eight different ways.
INSTRUCTIONS: tuple[tuple[str, str], ...] = (
    ("improve", "Improve child.py's performance against the pool."),
    ("different", "Design a completely different algorithm for the same game."),
    ("inspired", "Create a novel approach inspired by child.py that works fundamentally differently."),
    ("restructure", "Redesign child.py's core components, keeping what the feedback says wins."),
    ("tune", "Tune constants and thresholds only; keep the structure."),
)
```

Each text is wrapped by a common preamble and postamble at build time:
read `AGENTS.md` and `feedback.md`; keep editing `child.py` until it
clears the bar, testing with the harness as you go; stop when it clears
both parts; keep what the feedback says is winning; never read or
reconstruct an opponent's source; the budget is N minutes and whatever
`child.py` holds then is what is evaluated.

`build_sandbox` copies `champion` to `child.py` rather than `parent.py`,
writes the chosen instruction and the budget to `PROMPT.md`, and
`_feedback_lines` gains a first section stating the bar: the champion's
fitness to beat, and that a winning record against the champion by name
is the second half.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/campaign/test_prompt.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/campaign/prompt.py tests/campaign/test_prompt.py
git commit -m "feat: a session is handed the champion and the bar to beat it"
```

---

### Task 4: Promotion ships a tarball

**Files:**

- Modify: `src/kaggriculture/campaign/gate.py`
- Test: `tests/campaign/test_gate.py`

**Interfaces:**

- Consumes: `Program` from Task 2.
- Produces:
  - `Champion` gains `tarball: str`.
  - `promote(program: Program, result: DeepResult, pool: Pool) -> Champion`
    — packages, image-tests, writes, saves, returns.
  - `class NotShippable(RuntimeError)` — raised when the tarball will not
    run in the Kaggle image.
  - `commit_floor` is deleted.

- [ ] **Step 1: Write the failing tests**

```python
def test_promote_leaves_a_tarball_the_champion_record_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cut is uploading this file; nothing is built at cut time."""
    program = _program(tmp_path, monkeypatch)
    p = pool.Pool(opponents={"a": "/x/a.py"}, weights={"a": 1.0})
    monkeypatch.setattr(gate.kaggle_image, "load_test", lambda tar: "EPISODE_OK")

    champion = gate.promote(program, _result(), p)

    tarball = Path(champion.tarball)
    assert tarball.exists() and tarball.parent == config.CHAMPIONS
    with tarfile.open(tarball) as tar:
        names = tar.getnames()
    assert "main.py" in names and "kaggriculture_engine.so" in names


def test_a_tarball_that_will_not_run_is_not_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The floor only ever holds a program Kaggle's own image played."""
    program = _program(tmp_path, monkeypatch)
    p = pool.Pool(opponents={"a": "/x/a.py"}, weights={"a": 1.0})

    def refuses(tarball: Path) -> str:
        raise RuntimeError("ENGINE_MISSING")

    monkeypatch.setattr(gate.kaggle_image, "load_test", refuses)

    with pytest.raises(gate.NotShippable, match="ENGINE_MISSING"):
        gate.promote(program, _result(), p)
    assert not (config.FLOOR / "main.py").exists()
    assert not config.CHAMPION.exists()


def test_gate_runs_no_version_control() -> None:
    """The loop never touches git; provenance is champion.json."""
    assert not hasattr(gate, "commit_floor")
    assert "subprocess" not in gate.__dict__
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/campaign/test_gate.py -q`
Expected: FAIL, `Champion` has no `tarball`.

- [ ] **Step 3: Edit `gate.py`**

Delete `commit_floor`, the `subprocess` and `time` imports and the
`commit` parameter. Add `NotShippable`. `promote` now, in order:
`harness.package(source, config.CHAMPIONS / f"{name}.tar.gz")`;
`kaggle_image.load_test(tarball)`, wrapping any exception in
`NotShippable` and deleting the tarball before raising; then the champion
file, the floor, `SERVED`, the pool save and `champion.json` exactly as
now, with `tarball` in the record.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/campaign/test_gate.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/campaign/gate.py tests/campaign/test_gate.py
git commit -m "feat: a promotion produces the tarball a cut uploads"
```

---

### Task 5: The script

**Files:**

- Rewrite: `src/kaggriculture/campaign/loop.py`
- Test: `tests/campaign/test_loop.py`

**Interfaces:**

- Consumes: everything from Tasks 1–4.
- Produces:
  - `State`: `sessions: int = 0`, `champion: DeepResult | None = None`,
    `champion_path: str | None = None`,
    `sessions_since_promotion: int = 0`.
  - `run(sessions: int, mutator: Mutator, workers: int, seed_agent: Path,
rng: random.Random, log: wandb.Run) -> State`
  - `main(argv: list[str] | None = None) -> None` with `--sessions`,
    `--workers`, `--seed-agent`, `--dry-run`.

- [ ] **Step 1: Write the failing tests**

Eleven tests, spec §11. The end-to-end one:

```python
def test_a_better_child_is_deep_scored_and_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: wandb.Run,
    records: list[tuple[float, dict]],
) -> None:
    """Two sessions, no schedule: the better child is confirmed and shipped."""
    tiny_run(tmp_path, monkeypatch)
    pass_agent = _write(tmp_path / "pass.py", PASS)
    p = pool.Pool(opponents={"pass": str(pass_agent)}, weights={"pass": 1.0})
    p.save(config.POOL)

    state = loop.run(
        sessions=2,
        mutator=mutate.FakeMutator(edit=lambda _: SELLER),
        workers=WORKERS,
        seed_agent=_write(tmp_path / "seed.py", PASS),
        rng=random.Random(0),
        log=log,
    )

    assert state.champion is not None and state.champion.score > 0.5
    assert Path(state.champion_path).read_text() == SELLER
    assert (config.FLOOR / "main.py").read_text() == SELLER
    assert "champion_1" in pool.Pool.load(config.POOL).names()
    deep = [r for _, r in records if "deep/promoted" in r]
    assert len(deep) == 1 and deep[0]["deep/promoted"] == 1
```

And, each with the mutation that reddens it named in the report:

2. `test_eight_workers_run_at_once` — a `FakeMutator` whose `edit` sleeps
   and records timestamps; assert the intervals overlap. Red under
   `SESSIONS = 1`.
3. `test_a_promotion_changes_what_the_next_session_starts_from` — after a
   promotion, the next sandbox's `child.py` is the champion's source. Red
   if the seed is used instead.
4. `test_a_program_is_deep_scored_once` — a spy counting `evaluator.deep`
   calls per program id; run enough sessions that the same program stays
   in the top three. Red if the `deep is None` check is dropped.
5. `test_the_champion_is_rescored_after_a_promotion` — the pool the second
   comparison used contains `champion_1`. Red if the re-score is dropped.
6. `test_an_unshippable_candidate_is_not_promoted` — `NotShippable` from
   the gate leaves the floor unchanged and the run going.
7. `test_a_provider_failure_is_not_the_lineages_failure` — as the existing
   test, against the new `Failure` record.
8. `test_cancellation_kills_the_session_process_group` and
   `test_a_broken_pool_opponent_stops_the_run` — as the existing ones.
9. `test_stagnation_switches_the_starting_program` — with
   `STAGNATION_SESSIONS` patched to 1 and no promotion, the second
   sandbox's `child.py` is a top-ten program, and `PROMPT.md` says so.
10. `test_a_dirty_src_refuses_to_start` — `main` raises naming the file.
11. `test_resume` — `state.json` and `champion.json` are read back.

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/campaign/test_loop.py -q`
Expected: FAIL, `run() got an unexpected keyword argument 'sessions'`.

- [ ] **Step 3: Rewrite `loop.py`**

`main` is the four steps:

```python
def main(argv: list[str] | None = None) -> None:
    """``campaign loop``: eight agents, each told to beat the champion."""
    args = _arguments(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    # 1. wandb, named for the model and the code that produced the run.
    log = _open_run(dry_run=args.dry_run)
    try:
        # 2-4. the event loop, the workers, and the gate they fire.
        run(args.sessions, _mutator(args), args.workers, args.seed_agent,
            random.Random(), log)
    finally:
        log.finish()
```

`_open_run` reads the revision through GitPython, raises naming any
uncommitted file under `src/`, and passes
`id=name=f"{config.CODEX_MODEL}-{revision}"`, `resume="allow"`.

`run` loads the state, the champion, the pool and the database, seeds the
database if empty, then `asyncio.run(campaign.drive())`. `Campaign` holds
`state`, `database`, `pool`, `mutator`, `workers`, `rng`, `log`, a
`asyncio.Semaphore(config.DEEP_CONCURRENCY)`, and a `TaskGroup`. Its
methods are `drive` (signal handlers, the group, `SESSIONS` workers),
`worker` (spec §3, verbatim), `session` (build the sandbox, await the
mutator, validate, fast, add, fire the gate), `gate` (semaphore, deep,
record, promotion, promote in a thread, re-score), and `record` (the wandb
dict). `_start` returns the champion path, or a draw from `top(10)` when
`state.sessions_since_promotion >= config.STAGNATION_SESSIONS`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/campaign/test_loop.py -q`, then
`uv run pytest tests/campaign -q`, then `uv run pre-commit run -a`.
Expected: PASS, and `wc -l src/kaggriculture/campaign/loop.py` under 300.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/campaign/loop.py tests/campaign/test_loop.py
git commit -m "feat: eight agents, one champion, and a gate that fires on a result"
```

---

### Task 6: Ship from the floor, not from src

**Files:**

- Modify: `src/kaggriculture/scripts/package.py`
- Modify: `.pre-commit-config.yaml`
- Test: `tests/test_package.py`, `tests/test_submission.py`

**Interfaces:**

- Consumes: `config.FLOOR` from Task 4.

- [ ] **Step 1: Write the failing test**

```python
def test_the_packager_ships_the_floor(tmp_path: Path) -> None:
    """What we upload is what the gate promoted, not what is in src."""
    assert package.ENTRYPOINT == config.FLOOR / "main.py"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run pytest tests/test_package.py -q`
Expected: FAIL, `ENTRYPOINT` is `config.SERVED`.

- [ ] **Step 3: Make the change**

`package.ENTRYPOINT = config.FLOOR / "main.py"`. Since the loop no longer
writes into `src`, remove the `served/` exclusions from the `ruff-format`,
`ruff` and `ty` hooks in `.pre-commit-config.yaml`; `served/main.py` is
now an ordinary committed file, the campaign's seed program.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/ -q` and `uv run pre-commit run -a`.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/scripts/package.py .pre-commit-config.yaml tests/test_package.py tests/test_submission.py
git commit -m "refactor: ship the floor the gate wrote, not the seed in src"
```

---

### Task 7: Documentation

**Files:**

- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-09-04-codex-campaign-design.md`

- [ ] **Step 1: Rewrite the README's campaign section**

The four steps, `uv run campaign loop --sessions N`, what to watch on
wandb (the deep field rate and the held-out rates), where the champion
tarball lands, and that a cut is uploading it. Delete every mention of
islands, epochs, iterations and the call budget.

- [ ] **Step 2: Mark the superseded spec**

One line at the top of the 2026-09-04 spec: its §3 schedule is superseded
by `2026-09-05-campaign-script-design.md`; its engine, harness, validation
and opponent sections still stand.

- [ ] **Step 3: Commit**

```bash
git add README.md docs/superpowers/specs/2026-09-04-codex-campaign-design.md
git commit -m "docs: describe the campaign as it now runs"
```

---

## Self-Review

**Spec coverage.** §1 four steps → Task 5. §2 definitions → Tasks 1, 2.
§3 worker loop → Task 5. §4 sandbox, bar, instructions → Task 3. §5.1
cascade → unchanged code, exercised in Task 5. §5.2 promotion and tarball
→ Task 4. §5.3 pool co-evolution → unchanged `pool.py`, wired in Task 5.
§6 session → unchanged `mutate.py` plus Task 1's rename. §7 machine →
`GAME_LIMIT_SECONDS` in Task 1, validation unchanged, state in Tasks 2 and
5, stagnation in Task 5, run identity in Task 5. §8 constants → Task 1.
§9 not-built → deletions in Tasks 1, 2, 4. §10 telemetry → Task 5. §11
tests → Task 5. §12 size → Task 5 step 4.

**Gap found and closed:** the spec's `GAME_LIMIT_SECONDS` needs a caller.
`validate.validate` plays the check game; Task 1 adds the constant and
Task 5's suite covers a hung program through the loop. If the implementer
finds validation does not enforce it, that is a defect to report, not to
route around.

**Type consistency.** `Program` in Task 2 is used by Task 4's `promote`
and Task 5's database calls with the same field names. `Champion.tarball`
in Task 4 is read by Task 5's artifact logging. `INSTRUCTIONS` in Task 3
is indexed by Task 5's `_start`.
