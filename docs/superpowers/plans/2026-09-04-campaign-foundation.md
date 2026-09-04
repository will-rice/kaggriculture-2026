# Campaign Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the repository to what the codex campaign uses, adopt the C++ engine port behind a ctypes bridge proven bit-identical to the reference engine on archived tapes, and ship the harness (`play` / `check` / `package`) and validation (`copy` / `load` / `latency`) that both the mutation sandbox and the evolution loop are built on.

**Architecture:** One package `kaggriculture.campaign` holds everything. Games run inside the adopted engine (`kag::Sim` in `sim.hpp`), driven from Python through a packed-struct C ABI; Python renders the engine's state into the exact observation dict the reference engine produces, so any Python agent (ours or a vendored opponent) plays on it unchanged. The reference engine remains the oracle: the differential test replays archived episodes (which carry their seed in `info.seed`) through the port and compares every step's observation and both final banks. Plan 2 (the evolution loop) consumes the interfaces named in each task's **Produces** block.

**Tech Stack:** Python 3.11, `kaggle-environments` 1.32.7, `ctypes`, g++ (C++17), pytest, uv. Spec: `docs/superpowers/specs/2026-09-04-codex-campaign-design.md`.

## Global Constraints

- Exam seeds are `range(700_000, 700_064)`; nothing outside the deep evaluation may play them.
- Opponent sources live under `/data/kaggriculture/opponents/`; no path under it is ever written into a sandbox or printed by the harness.
- The engine library is named `kaggriculture_engine.so` and is loaded by absolute path — never `agent.so` (two agents loading a library of that name into one process collide and bank 0 silently).
- Deletions are explicit `git rm` paths. Never `git add -A`.
- `uv run pre-commit run -a` passes before every commit; the pytest hook runs the whole suite (~minutes after the cut).
- Commit messages: conventional style, say why, no mention of Claude or of hardware.
- Python style per `~/.claude/CLAUDE.md`: `logging` not `print`, Google docstrings, absolute imports, no `from __future__ import annotations`, constants over CLI flags, `ThreadPoolExecutor` / `ProcessPoolExecutor` for concurrency, `tqdm` over iterables, `pytest` functional style, minimal mocks.

---

## File structure

| path                                                                  | responsibility                                                                                                                                                  |
| --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/kaggriculture/campaign/__init__.py`                              | package marker                                                                                                                                                  |
| `src/kaggriculture/campaign/config.py`                                | every path and constant: `EXAM_SEEDS`, `ROOT`, `RUN`, `OPPONENTS`, `EPISODES`, `ENGINE_LIBRARY`, `CORE_BUDGET`, `ITEMS`, `SHOP_NAMES`, `UNIT_OPS`, `MARKET_OPS` |
| `src/kaggriculture/campaign/arena.py`                                 | reference-engine games between two agent files, both seats, over a process pool (from `search/arena.py`, stripped to file-path opponents)                       |
| `src/kaggriculture/campaign/engine/sim.hpp`, `pyrandom.hpp`, `NOTICE` | the adopted port, verbatim, with license                                                                                                                        |
| `src/kaggriculture/campaign/engine/bridge.cpp`                        | `extern "C"` surface: create, step, export packed state, free                                                                                                   |
| `src/kaggriculture/campaign/engine/build.py`                          | compiles `kaggriculture_engine.so` beside itself                                                                                                                |
| `src/kaggriculture/campaign/engine/wrapper.py`                        | `Engine` class: `reset`, `step`, `observation(player)`, `bank(player)`; `pack_action`, `render`                                                                 |
| `src/kaggriculture/campaign/tapes.py`                                 | iterate archived episodes: seed, actions per step, observations per step                                                                                        |
| `src/kaggriculture/campaign/roster.py`                                | opponent names → paths (never exposed), training pool and held-out set                                                                                          |
| `src/kaggriculture/campaign/harness.py`                               | `play`, `check`, `package`; the `campaign` CLI                                                                                                                  |
| `src/kaggriculture/campaign/copycheck.py`                             | token-shingle similarity against opponent sources                                                                                                               |
| `src/kaggriculture/campaign/validate.py`                              | `validate(agent) -> Verdict`: syntax, contract, imports, copy, load, latency                                                                                    |
| `src/kaggriculture/campaign/kaggle_image.py`                          | load test of a tarball inside `gcr.io/kaggle-gpu-images/python:latest`                                                                                          |
| `src/kaggriculture/campaign/field_gate.py`, `kernel_watch.py`         | moved from `scripts/`, imports fixed; Plan 2 folds `field_gate` into the evaluator                                                                              |
| `src/kaggriculture/scripts/package.py`, `submit.py`                   | shipping, without the routes store                                                                                                                              |
| `src/kaggriculture/served/main.py`                                    | the floor: starts as the skeleton                                                                                                                               |
| `src/kaggriculture/campaign/task_prompt.md`                           | phase 1's product; a first draft is written by hand in Task 12                                                                                                  |
| `tests/campaign/*.py`                                                 | one test module per source module                                                                                                                               |

---

### Task 1: Branch, `config.py`, and the stripped arena

**Files:**

- Create: `src/kaggriculture/campaign/__init__.py`, `src/kaggriculture/campaign/config.py`, `src/kaggriculture/campaign/arena.py`
- Test: `tests/campaign/__init__.py`, `tests/campaign/test_config.py`, `tests/campaign/test_arena.py`

**Interfaces:**

- Produces: `config.EXAM_SEEDS: tuple[int, ...]`, `config.ROOT: Path`, `config.RUN: Path`, `config.OPPONENTS: Path`, `config.EPISODES: Path`, `config.ENGINE_LIBRARY: Path`, `config.CORE_BUDGET: int`, `config.ITEMS: list[str]`, `config.SHOP_NAMES: list[str]`, `config.UNIT_OPS: list[str]`, `config.MARKET_OPS: list[str]`; `arena.run_banks(seat_zero: str, seat_one: str, seed: int) -> tuple[int, int]`, `arena.play(seat_zero: str, seat_one: str, seeds: Sequence[int], workers: int) -> list[tuple[int, int]]`, `arena.outcomes(candidate: str, league: Mapping[str, str], seeds: Sequence[int], workers: int) -> OutcomeScores`, `arena.summarize(scores, league, seeds) -> dict[str, float]`.

- [ ] **Step 1: Create the branch**

```bash
git checkout -b campaign
git status --short   # the staged deletions of src/kaggriculture/rules and tests/rules from the previous branch are fine; they are part of the cut
```

- [ ] **Step 2: Write the failing tests**

`tests/campaign/__init__.py` is empty. `tests/campaign/test_config.py`:

```python
"""The constants every other campaign module trusts."""

from kaggle_environments.envs.kaggriculture.kaggriculture import ANIMALS, CROPS, PRODUCTS, SHOPS

from kaggriculture.campaign import config


def test_exam_seeds_are_the_sealed_block() -> None:
    assert config.EXAM_SEEDS == tuple(range(700_000, 700_064))


def test_item_order_matches_the_engine_port_enum() -> None:
    """sim.hpp's Item enum is WHEAT..FERTILIZER then GOOSE, COW, SHEEP."""
    assert config.ITEMS == [
        "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
        "EGG", "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP",
    ]
    assert config.ITEMS[: len(PRODUCTS)] == PRODUCTS
    assert config.ITEMS[len(PRODUCTS):] == list(ANIMALS)
    assert list(CROPS) == config.ITEMS[: len(CROPS)]


def test_shop_names_are_sorted_like_the_engine_unlocks_them() -> None:
    assert config.SHOP_NAMES == sorted(SHOPS)


def test_unit_ops_match_the_port_enum_order() -> None:
    assert config.UNIT_OPS == [
        "PASS", "NORTH", "SOUTH", "EAST", "WEST",
        "PICKUP", "DROP", "PLACE",
        "PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG",
        "BUILD_COOP", "BUILD_PASTURE",
        "FEED", "COLLECT_FERTILIZER", "CARE",
    ]
    assert config.MARKET_OPS == [
        "NONE", "HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"
    ]


def test_core_budget_leaves_headroom() -> None:
    import os

    assert 1 <= config.CORE_BUDGET <= max(1, os.cpu_count() - 8)
```

`tests/campaign/test_arena.py`:

```python
"""Reference-engine games between two agent files."""

from pathlib import Path

from kaggriculture.campaign import arena

PASS_AGENT = '''
def agent(observation, configuration=None):
    return {"farmer": ["PASS"], "hands": [], "market": []}
'''


def test_pass_versus_pass_banks_the_starting_money(tmp_path: Path) -> None:
    agent = tmp_path / "pass.py"
    agent.write_text(PASS_AGENT, encoding="utf-8")
    assert arena.run_banks(str(agent), str(agent), 1) == (3000, 3000)


def test_outcomes_plays_both_seats_and_scores_ties_as_half(tmp_path: Path) -> None:
    agent = tmp_path / "pass.py"
    agent.write_text(PASS_AGENT, encoding="utf-8")
    scores = arena.outcomes(str(agent), {"mirror": str(agent)}, [1, 2], workers=2)
    assert list(scores) == [0.5, 0.5, 0.5, 0.5]
    assert arena.summarize(scores, {"mirror": str(agent)}, [1, 2]) == {"mirror": 0.5}


def test_a_crashing_agent_is_a_failure_not_a_loss(tmp_path: Path) -> None:
    good = tmp_path / "pass.py"
    good.write_text(PASS_AGENT, encoding="utf-8")
    bad = tmp_path / "bad.py"
    bad.write_text("def agent(o, c=None):\n    raise RuntimeError('boom')\n", encoding="utf-8")
    scores = arena.outcomes(str(bad), {"pass": str(good)}, [1], workers=1)
    assert scores == [] or all(s is None for s in scores)
    assert scores.failures and "seat" in scores.failures[0]
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/campaign -v`
Expected: FAIL with `ModuleNotFoundError: kaggriculture.campaign`

- [ ] **Step 4: Write `config.py`**

```python
"""Every path and constant the campaign shares.

Opponent paths and the exam seeds live here and nowhere else, so the one
place a sandbox could learn either is a module it never imports.
"""

import os
from pathlib import Path

from kaggle_environments.envs.kaggriculture.kaggriculture import ANIMALS, PRODUCTS, SHOPS

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "run" / "campaign"
OPPONENTS = Path("/data/kaggriculture/opponents")
EPISODES = Path("/data/kaggriculture/episodes")
ENGINE_LIBRARY = Path(__file__).parent / "engine" / "kaggriculture_engine.so"

# Measurement only. The deep evaluation plays these; nothing else may.
EXAM_SEEDS: tuple[int, ...] = tuple(range(700_000, 700_064))

# Cores the arena and the engine may use between them. Eight are left for
# codex sessions, the loop, and the box's own tenants.
CORE_BUDGET = max(1, (os.cpu_count() or 1) - 8)

# Item order is sim.hpp's `Item` enum: the nine products, then the animals.
ITEMS: list[str] = list(PRODUCTS) + list(ANIMALS)
# The engine unlocks shops with `rng.choice(sorted(SHOPS))`.
SHOP_NAMES: list[str] = sorted(SHOPS)
# sim.hpp's `Op` and `MOp` enums, in order; the index is the wire value.
UNIT_OPS: list[str] = [
    "PASS", "NORTH", "SOUTH", "EAST", "WEST",
    "PICKUP", "DROP", "PLACE",
    "PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG",
    "BUILD_COOP", "BUILD_PASTURE",
    "FEED", "COLLECT_FERTILIZER", "CARE",
]
MARKET_OPS: list[str] = ["NONE", "HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"]
```

- [ ] **Step 5: Write `arena.py`**

Copy `src/kaggriculture/search/arena.py`'s `OutcomeScores`, `_win`, `_normalized_margin`, `outcomes`, `summarize` and `_run_banks` (renamed `run_banks`), and delete `HybridOpponent`, `GameKey`, `GameTask`, `GameResult`, `run_game_task`, `action_traces`, `_one_trace`, `_replay`, `_side`, `evaluate`. Opponents are file paths only:

```python
"""Reference-engine games between two agent files.

The reference engine is the oracle and the Kaggle runner; the port under
``campaign.engine`` is the fast copy. Anything that must be true on Kaggle
is measured here.
"""

import math
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter

from kaggle_environments import make

from kaggriculture.constants import ENVIRONMENT, EPISODE_STEPS


class OutcomeScores(list[float]):
    """Win points with the paired bank margins, failures and duration behind them."""

    def __init__(self) -> None:
        super().__init__()
        self.margins: list[int] = []
        self.failures: list[str] = []
        self.runtime_seconds = 0.0


def run_banks(seat_zero: str, seat_one: str, seed: int) -> tuple[int, int]:
    """Play one reference-engine episode and return both final banks.

    Raises:
        RuntimeError: If either seat held a status other than ACTIVE, DONE or
            INACTIVE at any step. A crashed agent banks its untouched 3000 and
            would otherwise read as an ordinary loss.
    """
    environment = make(ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS, "seed": seed})
    environment.run([seat_zero, seat_one])
    final = environment.steps[-1]
    statuses = (final[0].status, final[1].status)
    if statuses != ("DONE", "DONE") or final[0].reward is None or final[1].reward is None:
        raise RuntimeError(f"seed {seed} did not finish cleanly (statuses={statuses})")
    for seat in (0, 1):
        broken = {
            step[seat].status
            for step in environment.steps
            if step[seat].status not in ("ACTIVE", "DONE", "INACTIVE")
        }
        if broken:
            raise RuntimeError(f"seed {seed} seat {seat} held {sorted(broken)} during the episode")
    return int(final[0].reward), int(final[1].reward)


def _one(work: tuple[str, str, int]) -> tuple[int, int]:
    return run_banks(*work)


def play(seat_zero: str, seat_one: str, seeds: Sequence[int], workers: int) -> list[tuple[int, int]]:
    """Play every seed with the given seating, fanned over a process pool."""
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as pool:
        return list(pool.map(_one, [(seat_zero, seat_one, seed) for seed in seeds]))


def outcomes(candidate: str, league: Mapping[str, str], seeds: Sequence[int], workers: int) -> OutcomeScores:
    """Score the candidate against every league member, both seats, ties as half.

    Order is fixed: league member, then seed, then seat (candidate in seat
    zero, then seat one), ``2 * len(seeds)`` entries per member. A game that
    raises is recorded in ``failures`` and contributes no score.
    """
    started = perf_counter()
    scores = OutcomeScores()
    for name, opponent in league.items():
        for seed in seeds:
            for seat, seating in ((0, (candidate, opponent)), (1, (opponent, candidate))):
                try:
                    banks = run_banks(*seating, seed) if workers == 1 else play(*seating, [seed], 1)[0]
                except RuntimeError as error:
                    scores.failures.append(f"{name} seed {seed} seat {seat}: {error}")
                    continue
                ours, theirs = banks if seat == 0 else banks[::-1]
                scores.append(_win(ours, theirs))
                scores.margins.append(ours - theirs)
    scores.runtime_seconds = perf_counter() - started
    return scores
```

Then replace the per-game loop with a pooled version: build the full list of `(name, seed, seat, seating)` work items first, run them through one `ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1)`, and assemble `scores` in the fixed order afterwards. `max_tasks_per_child=1` is load-bearing: a worker that played a compiled opponent has that opponent's library loaded, and a second compiled opponent in the same process collides.

```python
def summarize(scores: Sequence[float], league: Mapping[str, str], seeds: Sequence[int]) -> dict[str, float]:
    """One win rate per league member, over ``2 * len(seeds)`` games each."""
    games = 2 * len(seeds)
    return {name: sum(scores[i * games : (i + 1) * games]) / games for i, name in enumerate(league)}


def _win(ours: int, theirs: int) -> float:
    return 1.0 if ours > theirs else 0.5 if ours == theirs else 0.0
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/campaign -v`
Expected: PASS (the arena tests take ~1 minute: reference-engine games are ~17 s each).

- [ ] **Step 7: Commit**

```bash
git add src/kaggriculture/campaign/__init__.py src/kaggriculture/campaign/config.py src/kaggriculture/campaign/arena.py tests/campaign
git commit -m "feat: seed the campaign package with its constants and a file-only arena"
```

---

### Task 2: The cut

**Files:**

- Move: `src/kaggriculture/kaito_v54_policy.py`, `kaito_v56_policy.py` → `/data/kaggriculture/opponents/kaito_v54/main.py`, `/data/kaggriculture/opponents/kaito_v56/main.py`
- Move: `src/kaggriculture/scripts/field_gate.py` → `src/kaggriculture/campaign/field_gate.py`; `src/kaggriculture/scripts/kernel_watch.py` → `src/kaggriculture/campaign/kernel_watch.py`; `tests/test_kernel_watch.py` → `tests/campaign/test_kernel_watch.py`
- Delete: everything in §9 of the spec
- Modify: `pyproject.toml`, `.pre-commit-config.yaml`, `src/kaggriculture/scripts/package.py`, `README.md`, `main.py`

**Interfaces:**

- Consumes: `campaign.arena`, `campaign.config`
- Produces: `campaign.field_gate.FIELD: dict[str, tuple[str, float]]` with all six lineage paths under `/data/kaggriculture/opponents/` or `/data/kaggriculture/agents/`; `campaign.field_gate.score_field(candidate: Path, seeds: int, workers: int, exclude: set[str]) -> tuple[dict[str, float], int]`; `campaign.kernel_watch` unchanged in behaviour.

- [ ] **Step 1: Move the two vendored opponents out of the package**

```bash
mkdir -p /data/kaggriculture/opponents/kaito_v54 /data/kaggriculture/opponents/kaito_v56
git mv src/kaggriculture/kaito_v54_policy.py /tmp/claude-1000/kaito_v54_main.py 2>/dev/null || cp src/kaggriculture/kaito_v54_policy.py /data/kaggriculture/opponents/kaito_v54/main.py
cp src/kaggriculture/kaito_v56_policy.py /data/kaggriculture/opponents/kaito_v56/main.py
sha256sum src/kaggriculture/kaito_v54_policy.py /data/kaggriculture/opponents/kaito_v54/main.py   # must match
sha256sum src/kaggriculture/kaito_v56_policy.py /data/kaggriculture/opponents/kaito_v56/main.py   # must match
```

- [ ] **Step 2: Move the two kept scripts and fix their imports**

```bash
git mv src/kaggriculture/scripts/field_gate.py src/kaggriculture/campaign/field_gate.py
git mv src/kaggriculture/scripts/kernel_watch.py src/kaggriculture/campaign/kernel_watch.py
git mv tests/test_kernel_watch.py tests/campaign/test_kernel_watch.py
```

In `campaign/field_gate.py`: replace `from kaggriculture.search import arena` with `from kaggriculture.campaign import arena`; replace `from kaggriculture.search.scripts import holdout` with `from kaggriculture.campaign import config` and `holdout.GATE_SEEDS` with `config.EXAM_SEEDS`; in `FIELD`, change `"src/kaggriculture/kaito_v54_policy.py"` to `"/data/kaggriculture/opponents/kaito_v54/main.py"` and likewise for v56. In `campaign/kernel_watch.py`: replace `from kaggriculture.scripts.field_gate import score_field` with `from kaggriculture.campaign.field_gate import score_field`. In `tests/campaign/test_kernel_watch.py`: `from kaggriculture.campaign import kernel_watch`.

- [ ] **Step 3: Delete the closed lines**

```bash
git rm -r -q src/kaggriculture/learn src/kaggriculture/market_residual src/kaggriculture/hybrid \
  src/kaggriculture/routes src/kaggriculture/sim src/kaggriculture/search src/kaggriculture/hooks
git rm -q src/kaggriculture/scripts/autonomous.py src/kaggriculture/scripts/budget.py \
  src/kaggriculture/scripts/market_counterfactuals.py src/kaggriculture/scripts/market_train.py \
  src/kaggriculture/scripts/market_pretrain.py src/kaggriculture/scripts/meta.py \
  src/kaggriculture/scripts/tracking.py src/kaggriculture/scripts/rule_search.py src/kaggriculture/scripts/run.py
git rm -q src/kaggriculture/features.py src/kaggriculture/action_codec.py src/kaggriculture/harness.py \
  src/kaggriculture/task.py src/kaggriculture/result.py src/kaggriculture/config.py src/kaggriculture/agent.py \
  src/kaggriculture/policy.py
git rm -q src/kaggriculture/kaito_policy.py src/kaggriculture/kaito_v22_policy.py src/kaggriculture/kaito_v23_policy.py \
  src/kaggriculture/kaito_v54_policy.py src/kaggriculture/kaito_v56_policy.py src/kaggriculture/kaito_v56_tuned_policy.py \
  src/kaggriculture/boatlee_v14_policy.py src/kaggriculture/economic_policy.py src/kaggriculture/searched_route_policy.py
git rm -r -q tests/hooks tests/hybrid tests/learn tests/market_residual tests/routes tests/search tests/sim tests/rules
git rm -q tests/feature_golden_generator.py tests/test_action_codec.py tests/test_autonomous.py tests/test_baselines.py \
  tests/test_benchmark_simulator.py tests/test_economic_model.py tests/test_economic_policy.py tests/test_features.py \
  tests/test_fetch_episodes.py tests/test_harness.py tests/test_kaito_policy.py tests/test_kaito_v56_tuned.py \
  tests/test_meta.py tests/test_policy.py tests/test_tracking.py tests/test_vendored_policies.py \
  tests/fixtures/economic_policy_decisions.json tests/fixtures/feature_goldens.json
git rm -r -q baselines 2>/dev/null || true
```

Then `ls src/kaggriculture tests` and confirm only these remain: `src/kaggriculture/{__init__.py, constants.py, observation.py, actions.py, report.py, replay.py, campaign/, scripts/{__init__.py, package.py, submit.py}}` and `tests/{__init__.py, campaign/, test_constants.py, test_report.py, test_package.py, test_submission.py, test_replay.py}`. If `tests/test_submission.py` or `tests/test_package.py` import removed modules, fix the import or delete the affected test function — do not keep a test that references deleted code.

- [ ] **Step 4: Prune `pyproject.toml`**

Replace the `dependencies` list with:

```toml
dependencies = [
  "kaggle-environments>=1.32.7",
  "kaggle>=1.7.4",
  "pre-commit>=4.1.0",
  "pydantic>=2.12.2",
  "pytest>=8.3.4",
  "python-dotenv>=1.1.1",
  "ruff>=0.9.7",
  "tqdm>=4.70.0",
  "ty>=0.0.18",
]
```

and the scripts with:

```toml
[project.scripts]
package = "kaggriculture.scripts.package:main"
submit = "kaggriculture.scripts.submit:main"
campaign = "kaggriculture.campaign.harness:main"
```

Run `uv sync` and confirm `uv run python -c "import torch"` fails.

- [ ] **Step 5: Fix `.pre-commit-config.yaml`**

Delete the top-level `exclude:` block (the Toad tree is gone) and the `exclude:` line under `ruff-format` (the vendored kernels are gone).

- [ ] **Step 6: Fix `package.py`**

Delete `from kaggriculture.routes import STORE`, the `REQUIRED` dict and `_validate_required`, and the `required` parameter of `build`. `EXCLUDED` becomes `shutil.ignore_patterns("__pycache__", "scripts", "campaign")`. Task 6 replaces this module's role for candidates; the project-level `package`/`submit` keep working for the floor.

- [ ] **Step 7: Rewrite `main.py` and `README.md`**

`main.py` becomes the skeleton served agent (Task 11 writes the real one); for now:

```python
"""Competition entrypoint: Kaggle loads this file and calls the last callable."""

from kaggriculture.served.main import agent  # noqa: F401
```

and `src/kaggriculture/served/__init__.py` (empty) plus `src/kaggriculture/served/main.py`:

```python
"""The floor: what the campaign has promoted, or the skeleton before it has."""


def agent(observation, configuration=None):
    """Play a turn. The skeleton passes; the campaign replaces this file."""
    return {"farmer": ["PASS"], "hands": [], "market": []}
```

`README.md`: replace the Features, Project Structure, Usage and "Changing the Strategy"/"Implementing" sections with a short description of the campaign (link the spec), the layout table from this plan's File structure, and the commands `uv sync`, `uv run campaign --help`, `uv run pytest`, `uv run pre-commit run -a`, `uv run package`, `uv run submit`.

- [ ] **Step 8: Run everything**

Run: `uv run pytest -q` then `uv run pre-commit run -a`
Expected: all remaining tests pass; pre-commit clean. Fix any import the cut broke in a kept file.

- [ ] **Step 9: Commit**

```bash
git add -u
git add src/kaggriculture/served src/kaggriculture/campaign tests/campaign pyproject.toml uv.lock .pre-commit-config.yaml README.md main.py
git commit -m "refactor: cut the repository to what the codex campaign uses

Every removed line is a measured-closed direction; git history keeps them.
The two vendored kernels the gate plays move to /data with the other
opponents, because other people's code does not live in our package."
```

(`git add -u` stages the deletions and modifications already known to git — it is not `add -A`; untracked paths are added by name.)

---

### Task 3: Adopt the engine and build the bridge

**Files:**

- Create: `src/kaggriculture/campaign/engine/sim.hpp`, `pyrandom.hpp`, `NOTICE`, `bridge.cpp`, `build.py`, `__init__.py`
- Test: `tests/campaign/test_engine_build.py`

**Interfaces:**

- Produces: `campaign.engine.build.build() -> Path` (compiles and returns `config.ENGINE_LIBRARY`); the C ABI below.

- [ ] **Step 1: Copy the port with its license**

```bash
S=/data/kaggriculture/agents/yhay81_router_v1/source/include
mkdir -p src/kaggriculture/campaign/engine
cp $S/sim.hpp $S/pyrandom.hpp src/kaggriculture/campaign/engine/
head -3 src/kaggriculture/campaign/engine/sim.hpp   # confirm the Apache-2.0 SPDX line and the pinned engine SHA
```

`NOTICE`:

```
sim.hpp and pyrandom.hpp are a C++ port of kaggriculture.py from
kaggle-environments 1.32.7 (official source SHA256
bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e),
published under the Apache License 2.0 in the Kaggle kernel
yhay81/three-day-shop-router. They are used here unmodified as a
simulator. No policy, tape, or guard from that kernel is included.
```

- [ ] **Step 2: Write the failing build test**

```python
"""The engine library builds, exports the ABI, and reports the pinned engine."""

import ctypes

from kaggriculture.campaign import config
from kaggriculture.campaign.engine import build


def test_build_produces_a_loadable_library_with_abi_version_1() -> None:
    path = build.build()
    assert path == config.ENGINE_LIBRARY and path.exists()
    library = ctypes.CDLL(str(path))
    library.kag_abi_version.restype = ctypes.c_uint32
    assert library.kag_abi_version() == 1
    library.kag_engine_version.restype = ctypes.c_char_p
    assert library.kag_engine_version() == b"1.32.7"
```

- [ ] **Step 3: Run it to verify it fails**

Run: `uv run pytest tests/campaign/test_engine_build.py -v`
Expected: FAIL with `ModuleNotFoundError` or missing `bridge.cpp`.

- [ ] **Step 4: Write `bridge.cpp`**

The packed layouts mirror the router's `submission_bridge.cpp` (which is how a ctypes caller is known to talk to this engine on Kaggle) plus the private fields a full state export needs.

```cpp
// SPDX-License-Identifier: Apache-2.0
// C ABI over the adopted engine port. Python owns observation rendering and
// action parsing; this file only moves packed structs across the boundary.
#include "sim.hpp"

#include <cstdint>
#include <cstring>

namespace {

#pragma pack(push, 1)
struct PackedTile {
    std::uint8_t kind = 0;
    std::uint8_t what = 0;
    std::uint8_t flags = 0;               // 1 has_animal, 2 watered, 4 fed, 8 cared, 16 fertilizer_available
    std::int8_t consecutive_dry = 0;
    std::int8_t yield_units = 0;
    std::int8_t pending_care_bonus = 0;
    std::int16_t planted_day = 0;
    std::int32_t max_lifespan_step = -1;
    std::int16_t fertilized_until_day = -1;
};

struct PackedFarm {
    double money = 0;
    PackedTile tiles[kag::BOARD][kag::BOARD]{};
    std::int8_t pos_x[kag::MAX_UNITS]{};
    std::int8_t pos_y[kag::MAX_UNITS]{};
    std::int32_t n_units = 1;
    std::int32_t n_quadrants = 1;
    std::int32_t hires_today = 0;
    std::int16_t shed[kag::N_ITEMS]{};
    std::int16_t seeds[kag::N_CROPS]{};
    std::int16_t inv[kag::MAX_UNITS][kag::N_ITEMS]{};
    std::uint8_t inv_keys[kag::MAX_UNITS][kag::N_ITEMS]{};   // insertion order of each unit's inventory
    std::uint8_t inv_nkeys[kag::MAX_UNITS]{};
};

struct PackedState {
    std::int32_t step = 0;
    std::int32_t day = 0;
    std::int32_t hour = 0;
    std::int32_t done = 0;
    std::int32_t n_shops = 0;
    std::int32_t market_inventory[kag::N_PRODUCTS]{};
    std::int32_t market_prices[kag::N_PRODUCTS]{};
    std::uint8_t shops[kag::MAX_SHOP_INSTANCES]{};
    PackedFarm farms[2]{};
};

struct PackedAction {
    std::uint8_t unit_ops[kag::MAX_UNITS]{};
    std::uint8_t unit_args[kag::MAX_UNITS]{};
    std::int16_t unit_ns[kag::MAX_UNITS]{};
    std::int32_t n_units = 1;
    std::uint8_t order_ops[16]{};
    std::uint8_t order_items[16]{};
    std::int32_t order_ns[16]{};
    std::int32_t n_orders = 0;
};
#pragma pack(pop)

kag::Action unpack(const PackedAction& packed) {
    kag::Action action{};
    action.n_units = std::max(1, std::min(packed.n_units, kag::MAX_UNITS));
    for (int i = 0; i < action.n_units; ++i)
        action.units[i] = {packed.unit_ops[i], packed.unit_args[i], packed.unit_ns[i]};
    action.n_orders = std::max(0, std::min(packed.n_orders, 16));
    for (int i = 0; i < action.n_orders; ++i)
        action.orders[i] = {packed.order_ops[i], packed.order_items[i], packed.order_ns[i]};
    return action;
}

void pack(const kag::State& state, PackedState& out) {
    out = PackedState{};
    out.step = state.step; out.day = state.day; out.hour = state.hour; out.done = state.done;
    out.n_shops = state.n_shops;
    for (int i = 0; i < state.n_shops; ++i) out.shops[i] = state.shops[i];
    for (int i = 0; i < kag::N_PRODUCTS; ++i) {
        out.market_inventory[i] = state.market.inventory[i];
        out.market_prices[i] = state.market.prices[i];
    }
    for (int p = 0; p < 2; ++p) {
        const kag::Farm& farm = state.farms[p];
        PackedFarm& dst = out.farms[p];
        dst.money = farm.money; dst.n_units = farm.n_units; dst.n_quadrants = farm.n_quadrants;
        dst.hires_today = farm.hires_today;
        for (int u = 0; u < farm.n_units; ++u) { dst.pos_x[u] = farm.pos_x[u]; dst.pos_y[u] = farm.pos_y[u]; }
        for (int y = 0; y < kag::BOARD; ++y) for (int x = 0; x < kag::BOARD; ++x) {
            const kag::Tile& t = farm.tiles[y][x];
            PackedTile& d = dst.tiles[y][x];
            d.kind = t.kind; d.what = t.what;
            d.flags = (t.has_animal ? 1 : 0) | (t.watered_today ? 2 : 0) | (t.fed_today ? 4 : 0)
                    | (t.cared_today ? 8 : 0) | (t.fertilizer_available ? 16 : 0);
            d.consecutive_dry = t.consecutive_dry; d.yield_units = t.yield_units;
            d.pending_care_bonus = t.pending_care_bonus; d.planted_day = t.planted_day;
            d.max_lifespan_step = t.max_lifespan_step; d.fertilized_until_day = t.fertilized_until_day;
        }
        std::memcpy(dst.shed, farm.shed, sizeof dst.shed);
        std::memcpy(dst.seeds, farm.seeds, sizeof dst.seeds);
        std::memcpy(dst.inv, farm.inv, sizeof dst.inv);
        std::memcpy(dst.inv_keys, farm.inv_keys, sizeof dst.inv_keys);
        std::memcpy(dst.inv_nkeys, farm.inv_nkeys, sizeof dst.inv_nkeys);
    }
}

}  // namespace

extern "C" {

std::uint32_t kag_abi_version() { return 1; }
const char* kag_engine_version() { return kag::ENGINE_VERSION; }

void* kag_new(std::uint64_t seed, std::int32_t episode_steps) {
    kag::Config config{};
    config.seed = seed;
    config.episode_steps = episode_steps;
    return new kag::Sim(config);
}

void kag_free(void* sim) { delete static_cast<kag::Sim*>(sim); }

void kag_step(void* sim, const PackedAction* a0, const PackedAction* a1) {
    static_cast<kag::Sim*>(sim)->step(unpack(*a0), unpack(*a1));
}

void kag_export(const void* sim, PackedState* out) { pack(static_cast<const kag::Sim*>(sim)->st, *out); }

}  // extern "C"
```

If `kag::Tile` has no `fertilized_until_day` member (check `sim.hpp` lines 171–186), drop that field from `PackedTile` here and from the ctypes mirror in Task 4; the differential test decides whether the rendered plant dict still matches.

- [ ] **Step 5: Write `build.py`**

```python
"""Compile the engine bridge into the library the wrapper loads.

The command is the one the port's own kernel used, with our bridge in place
of its policy.
"""

import logging
import subprocess
from pathlib import Path

from kaggriculture.campaign import config

LOGGER = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent


def build() -> Path:
    """Compile ``bridge.cpp`` and return the library path; skip if up to date."""
    sources = [HERE / "bridge.cpp", HERE / "sim.hpp", HERE / "pyrandom.hpp"]
    target = config.ENGINE_LIBRARY
    if target.exists() and target.stat().st_mtime >= max(s.stat().st_mtime for s in sources):
        return target
    command = [
        "g++", "-O3", "-std=c++17", "-shared", "-fPIC", "-I", str(HERE),
        "-o", str(target), str(HERE / "bridge.cpp"),
    ]
    LOGGER.info("building %s", target.name)
    subprocess.run(command, check=True)
    return target
```

`engine/__init__.py` is empty.

- [ ] **Step 6: Run the test to verify it passes**

Run: `uv run pytest tests/campaign/test_engine_build.py -v`
Expected: PASS. Add `src/kaggriculture/campaign/engine/*.so` to `.gitignore`.

- [ ] **Step 7: Commit**

```bash
git add src/kaggriculture/campaign/engine tests/campaign/test_engine_build.py .gitignore
git commit -m "feat: adopt the C++ engine port behind a packed-struct C ABI

The port is a tool with no strategy in it, taken with its Apache-2.0
headers; the bridge exports full state so Python can render the exact
observation the reference engine produces."
```

---

### Task 4: The Python wrapper — action packing and observation rendering

**Files:**

- Create: `src/kaggriculture/campaign/engine/wrapper.py`
- Test: `tests/campaign/test_wrapper.py`

**Interfaces:**

- Consumes: the C ABI from Task 3; `config.ITEMS`, `config.SHOP_NAMES`, `config.UNIT_OPS`, `config.MARKET_OPS`.
- Produces: `class Engine` with `Engine(seed: int, episode_steps: int = 720)`, `.step(action0: dict, action1: dict) -> None`, `.observation(player: int) -> dict`, `.bank(player: int) -> float`, `.done: bool`, `.step_index: int`; `pack_action(action: dict) -> PackedAction`; `render(state: PackedState, player: int) -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
"""The wrapper renders what the reference engine renders."""

from kaggle_environments import make

from kaggriculture.campaign.engine.wrapper import Engine, pack_action
from kaggriculture.constants import ENVIRONMENT, EPISODE_STEPS

PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def strip(observation: dict) -> dict:
    """The framework adds remainingOverageTime; the engine does not know it."""
    return {k: v for k, v in observation.items() if k != "remainingOverageTime"}


def reference_observations(seed: int, steps: int) -> list[list[dict]]:
    environment = make(ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS, "seed": seed})
    environment.reset()
    out = []
    for _ in range(steps):
        environment.step([PASS, PASS])
        out.append([strip(dict(state.observation)) for state in environment.state])
    return out


def test_initial_observation_matches_the_reference_engine() -> None:
    engine = Engine(seed=7)
    environment = make(ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS, "seed": 7})
    environment.reset()
    for player in (0, 1):
        assert engine.observation(player) == strip(dict(environment.state[player].observation))


def test_thirty_pass_steps_match_including_an_end_of_day() -> None:
    engine = Engine(seed=7)
    expected = reference_observations(7, 30)
    for step in range(30):
        engine.step(PASS, PASS)
        for player in (0, 1):
            assert engine.observation(player) == expected[step][player], f"step {step} player {player}"


def test_pack_action_maps_names_to_the_port_enums() -> None:
    packed = pack_action({
        "farmer": ["PLANT", "MELON"],
        "hands": [["PICKUP", "WHEAT", 3], ["NORTH"], ["NONSENSE"]],
        "market": [["HIRE"], ["BUY_SEED", "WHEAT", 5], ["SELL", "WOOL", 2], ["BUY_LAND"], ["BOGUS", "X", 1]],
    })
    assert packed.n_units == 4
    assert list(packed.unit_ops[:4]) == [8, 5, 1, 0]        # PLANT, PICKUP, NORTH, PASS (unknown op is a no-op)
    assert list(packed.unit_args[:2]) == [4, 0]            # MELON, WHEAT
    assert list(packed.unit_ns[:2]) == [1, 3]
    assert packed.n_orders == 4                            # the bogus order is dropped, as _parse_order drops it
    assert list(packed.order_ops[:4]) == [1, 3, 6, 2]
    assert list(packed.order_items[1:3]) == [0, 7]
    assert list(packed.order_ns[1:3]) == [5, 2]


def test_bank_is_the_money_the_reference_engine_reports() -> None:
    engine = Engine(seed=3)
    while not engine.done:
        engine.step(PASS, PASS)
    assert engine.bank(0) == 3000.0 and engine.bank(1) == 3000.0
    assert engine.step_index == EPISODE_STEPS - 1
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/campaign/test_wrapper.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write `wrapper.py`**

```python
"""Drive the engine port from Python and speak the reference engine's dialect.

Rendering lives here rather than in C++ because the reference engine's
observation is a Python dict with a specific shape, and the differential
test compares dicts. Read ``kaggriculture.py`` for every shape below:
``_new_farm`` (farm), ``_new_plant`` and ``_new_animal`` and the
BUILD_COOP/BUILD_PASTURE branches of ``_apply_unit_action`` (tiles),
``_spawn_weeds`` (weeds), ``_new_private`` (private), ``_new_market``,
``_new_town``.
"""

import ctypes
from typing import Any

from kaggle_environments.envs.kaggriculture.kaggriculture import ANIMALS, CROPS, LAND_ORDER, PRODUCTS

from kaggriculture.campaign import config
from kaggriculture.campaign.engine import build

BOARD = 10
MAX_UNITS = 40
MAX_SHOPS = 8
N_ITEMS = len(config.ITEMS)
N_CROPS = len(CROPS)
N_PRODUCTS = len(PRODUCTS)
ITEM_INDEX = {name: i for i, name in enumerate(config.ITEMS)}
UNIT_OP_INDEX = {name: i for i, name in enumerate(config.UNIT_OPS)}
MARKET_OP_INDEX = {name: i for i, name in enumerate(config.MARKET_OPS)}
# sim.hpp TileKind: T_EMPTY, T_LOCKED, T_WEED, T_COOP, T_PASTURE, T_PLANT
T_EMPTY, T_LOCKED, T_WEED, T_COOP, T_PASTURE, T_PLANT = range(6)
QUANTIFIED_OPS = {"PICKUP", "DROP", "PLACE"}
ITEM_OPS = {"PICKUP", "DROP", "PLACE", "PLANT"}
QUANTIFIED_ORDERS = {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"}


class PackedTile(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("kind", ctypes.c_uint8), ("what", ctypes.c_uint8), ("flags", ctypes.c_uint8),
        ("consecutive_dry", ctypes.c_int8), ("yield_units", ctypes.c_int8),
        ("pending_care_bonus", ctypes.c_int8), ("planted_day", ctypes.c_int16),
        ("max_lifespan_step", ctypes.c_int32), ("fertilized_until_day", ctypes.c_int16),
    ]


class PackedFarm(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("money", ctypes.c_double), ("tiles", PackedTile * BOARD * BOARD),
        ("pos_x", ctypes.c_int8 * MAX_UNITS), ("pos_y", ctypes.c_int8 * MAX_UNITS),
        ("n_units", ctypes.c_int32), ("n_quadrants", ctypes.c_int32), ("hires_today", ctypes.c_int32),
        ("shed", ctypes.c_int16 * N_ITEMS), ("seeds", ctypes.c_int16 * N_CROPS),
        ("inv", ctypes.c_int16 * N_ITEMS * MAX_UNITS),
        ("inv_keys", ctypes.c_uint8 * N_ITEMS * MAX_UNITS), ("inv_nkeys", ctypes.c_uint8 * MAX_UNITS),
    ]


class PackedState(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("step", ctypes.c_int32), ("day", ctypes.c_int32), ("hour", ctypes.c_int32), ("done", ctypes.c_int32),
        ("n_shops", ctypes.c_int32), ("market_inventory", ctypes.c_int32 * N_PRODUCTS),
        ("market_prices", ctypes.c_int32 * N_PRODUCTS), ("shops", ctypes.c_uint8 * MAX_SHOPS),
        ("farms", PackedFarm * 2),
    ]


class PackedAction(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("unit_ops", ctypes.c_uint8 * MAX_UNITS), ("unit_args", ctypes.c_uint8 * MAX_UNITS),
        ("unit_ns", ctypes.c_int16 * MAX_UNITS), ("n_units", ctypes.c_int32),
        ("order_ops", ctypes.c_uint8 * 16), ("order_items", ctypes.c_uint8 * 16),
        ("order_ns", ctypes.c_int32 * 16), ("n_orders", ctypes.c_int32),
    ]


def library() -> ctypes.CDLL:
    """Load the engine library once, by absolute path, under its unique name."""
    path = build.build()
    lib = ctypes.CDLL(str(path))
    lib.kag_new.restype = ctypes.c_void_p
    lib.kag_new.argtypes = [ctypes.c_uint64, ctypes.c_int32]
    lib.kag_free.argtypes = [ctypes.c_void_p]
    lib.kag_step.argtypes = [ctypes.c_void_p, ctypes.POINTER(PackedAction), ctypes.POINTER(PackedAction)]
    lib.kag_export.argtypes = [ctypes.c_void_p, ctypes.POINTER(PackedState)]
    return lib


def pack_action(action: Any) -> PackedAction:
    """Encode an action dict the way ``_apply_unit_action`` and ``_parse_order`` read it.

    Unknown unit ops become PASS (the engine no-ops them); malformed orders are
    dropped (``_parse_order`` returns None for them).
    """
    packed = PackedAction()
    units = [action.get("farmer", ["PASS"])] if isinstance(action, dict) else [["PASS"]]
    hands = action.get("hands", []) if isinstance(action, dict) else []
    units.extend(hands if isinstance(hands, list) else [])
    units = units[:MAX_UNITS]
    packed.n_units = max(1, len(units))
    for i, unit in enumerate(units):
        op = unit[0] if isinstance(unit, list) and unit else "PASS"
        packed.unit_ops[i] = UNIT_OP_INDEX.get(op, 0)
        packed.unit_args[i] = ITEM_INDEX.get(unit[1], 0) if op in ITEM_OPS and len(unit) > 1 else 0
        packed.unit_ns[i] = int(unit[2]) if op in QUANTIFIED_OPS and len(unit) > 2 else 1
    orders = action.get("market", []) if isinstance(action, dict) else []
    kept = 0
    for order in orders if isinstance(orders, list) else []:
        if not isinstance(order, list) or not order or order[0] not in MARKET_OP_INDEX or order[0] == "NONE":
            continue
        op = order[0]
        if op in QUANTIFIED_ORDERS:
            if len(order) < 3 or order[1] not in ITEM_INDEX:
                continue
            try:
                n = int(order[2])
            except (TypeError, ValueError):
                continue
            if n <= 0:
                continue
            packed.order_items[kept] = ITEM_INDEX[order[1]]
            packed.order_ns[kept] = n
        packed.order_ops[kept] = MARKET_OP_INDEX[op]
        kept += 1
        if kept == 16:
            break
    packed.n_orders = kept
    return packed


def render_tile(tile: PackedTile) -> Any:
    """Return the tile as the reference engine stores it."""
    if tile.kind == T_EMPTY:
        return None
    if tile.kind == T_LOCKED:
        return "LOCKED"
    if tile.kind == T_WEED:
        return {"kind": "WEED"}
    if tile.kind == T_PLANT:
        return {
            "kind": "PLANT",
            "crop": config.ITEMS[tile.what],
            "planted_day": tile.planted_day,
            "watered_today": bool(tile.flags & 2),
            "consecutive_unwatered": tile.consecutive_dry,
            "yield_units": tile.yield_units,
            "max_lifespan_step": tile.max_lifespan_step,
            "fertilized_until_day": tile.fertilized_until_day,
        }
    structure = "COOP" if tile.kind == T_COOP else "PASTURE"
    if not tile.flags & 1:
        return {"kind": structure, "animal": None}   # confirm against BUILD_COOP in kaggriculture.py
    return {
        "kind": structure,
        "animal": config.ITEMS[tile.what],
        "placed_day": tile.planted_day,
        "yield_units": tile.yield_units,
        "consecutive_unfed": tile.consecutive_dry,
        "fed_today": bool(tile.flags & 4),
        "cared_today": bool(tile.flags & 8),
        "fertilizer_available": bool(tile.flags & 16),
        "pending_care_bonus": tile.pending_care_bonus,
    }


def render_farm(farm: PackedFarm) -> dict[str, Any]:
    return {
        "money": float(farm.money),
        "tiles": [[render_tile(farm.tiles[y][x]) for x in range(BOARD)] for y in range(BOARD)],
        "farmer": [int(farm.pos_x[0]), int(farm.pos_y[0])],
        "hands": [[int(farm.pos_x[u]), int(farm.pos_y[u])] for u in range(1, farm.n_units)],
        "unlocked_quadrants": ["NW", *LAND_ORDER[: farm.n_quadrants - 1]],
        "hires_today": int(farm.hires_today),
    }


def render_private(farm: PackedFarm) -> dict[str, Any]:
    inventories = []
    for u in range(farm.n_units):
        keys = [config.ITEMS[farm.inv_keys[u][k]] for k in range(farm.inv_nkeys[u])]
        inventories.append({name: int(farm.inv[u][ITEM_INDEX[name]]) for name in keys})
    return {
        "shed": {name: int(farm.shed[i]) for i, name in enumerate(config.ITEMS)},
        "seeds": {crop: int(farm.seeds[i]) for i, crop in enumerate(CROPS)},
        "inventories": inventories,
    }


def render(state: PackedState, player: int) -> dict[str, Any]:
    """Return the observation the reference engine hands ``player`` at this step."""
    return {
        "player": player,
        "step": int(state.step),
        "day": int(state.day),
        "hour": int(state.hour),
        "farms": [render_farm(state.farms[0]), render_farm(state.farms[1])],
        "market": {
            "inventory": {name: int(state.market_inventory[i]) for i, name in enumerate(PRODUCTS)},
            "prices": {name: int(state.market_prices[i]) for i, name in enumerate(PRODUCTS)},
        },
        "town": {"unlocked_shops": [config.SHOP_NAMES[state.shops[i]] for i in range(state.n_shops)]},
        "private": render_private(state.farms[player]),
    }


class Engine:
    """One episode of the port, stepped by both players' action dicts."""

    def __init__(self, seed: int, episode_steps: int = 720) -> None:
        self._lib = library()
        self._sim = self._lib.kag_new(seed, episode_steps)
        self._state = PackedState()
        self._export()

    def _export(self) -> None:
        self._lib.kag_export(self._sim, ctypes.byref(self._state))

    def step(self, action0: Any, action1: Any) -> None:
        a0, a1 = pack_action(action0), pack_action(action1)
        self._lib.kag_step(self._sim, ctypes.byref(a0), ctypes.byref(a1))
        self._export()

    def observation(self, player: int) -> dict[str, Any]:
        return render(self._state, player)

    def bank(self, player: int) -> float:
        return float(self._state.farms[player].money)

    @property
    def done(self) -> bool:
        return bool(self._state.done)

    @property
    def step_index(self) -> int:
        return int(self._state.step)

    def __del__(self) -> None:
        if getattr(self, "_sim", None):
            self._lib.kag_free(self._sim)
            self._sim = None
```

Where a rendered shape is uncertain (the empty coop/pasture dict, the `seeds` dict order, whether the `shed` dict includes animals — `_new_private` builds it over `PRODUCTS + list(ANIMALS)`), read the named function in `kaggriculture.py` and match it; the tests in Steps 1 and the differential test in Task 5 are the arbiter.

- [ ] **Step 4: Run the tests until they pass**

Run: `uv run pytest tests/campaign/test_wrapper.py -v`
Expected: PASS. On a mismatch the assertion shows the two dicts; fix the renderer (or the packing), never the test.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/campaign/engine/wrapper.py tests/campaign/test_wrapper.py
git commit -m "feat: render the engine port's state as the reference observation"
```

---

### Task 5: Tapes and the differential test

**Files:**

- Create: `src/kaggriculture/campaign/tapes.py`
- Test: `tests/campaign/test_differential.py`

**Interfaces:**

- Consumes: `Engine`, `config.EPISODES`.
- Produces: `tapes.Episode` (pydantic: `seed: int`, `steps: list[list[dict]]`, `engine_version: str`), `tapes.iter_archive(path: Path) -> Iterator[Episode]`, `tapes.sample(n: int, since: str = "2026-08-24") -> list[Episode]` (episodes with `module_version == "1.32.7"` from archives dated `since` or later, deterministic order).

- [ ] **Step 1: Write the failing test**

```python
"""The port must agree with the reference engine on every recorded step.

An archived episode carries its seed in ``info.seed`` and the reference
engine's own observation at every step, so replaying its actions through the
port and comparing is a differential test against the oracle with no
reference-engine time at all.
"""

import pytest

from kaggriculture.campaign import tapes
from kaggriculture.campaign.engine.wrapper import Engine


def strip(observation: dict) -> dict:
    return {k: v for k, v in observation.items() if k != "remainingOverageTime"}


@pytest.mark.parametrize("episode", tapes.sample(8), ids=lambda e: str(e.seed))
def test_replaying_an_archived_episode_reproduces_every_observation(episode: tapes.Episode) -> None:
    engine = Engine(seed=episode.seed)
    for index in range(1, len(episode.steps)):
        recorded = episode.steps[index]
        engine.step(episode.steps[index][0]["action"], episode.steps[index][1]["action"])
        for player in (0, 1):
            assert engine.observation(player) == strip(recorded[player]["observation"]), (
                f"seed {episode.seed} step {index} player {player}"
            )
    assert (engine.bank(0), engine.bank(1)) == (
        float(episode.steps[-1][0]["reward"]), float(episode.steps[-1][1]["reward"])
    )


@pytest.mark.slow
@pytest.mark.parametrize("episode", tapes.sample(200), ids=lambda e: str(e.seed))
def test_two_hundred_episodes(episode: tapes.Episode) -> None:
    test_replaying_an_archived_episode_reproduces_every_observation(episode)
```

Register the marker in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
  "slow: hundreds of tapes; run with -m slow before trusting the engine",
]
addopts = "-m 'not slow'"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/campaign/test_differential.py -v`
Expected: FAIL with `ImportError: tapes`.

- [ ] **Step 3: Write `tapes.py`**

```python
"""Archived episodes as replayable tapes.

Each daily archive under ``config.EPISODES`` is a zip of episode JSONs from
the ladder. ``info.seed`` is the seed the engine was created with; the
action recorded at ``steps[t]`` is what each seat submitted while looking at
``steps[t-1]``'s observation, and ``steps[t]``'s observation is the state it
produced.
"""

import json
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from kaggriculture.campaign import config


class Episode(BaseModel):
    """One recorded episode."""

    seed: int
    engine_version: str
    steps: list[list[dict[str, Any]]]


def iter_archive(path: Path) -> Iterator[Episode]:
    """Yield every episode in one daily archive, in name order."""
    with zipfile.ZipFile(path) as archive:
        for name in sorted(n for n in archive.namelist() if n.endswith(".json")):
            raw = json.loads(archive.read(name))
            yield Episode(seed=int(raw["info"]["seed"]), engine_version=str(raw["module_version"]), steps=raw["steps"])


def sample(n: int, since: str = "2026-08-24") -> list[Episode]:
    """Return the first ``n`` 1.32.7 episodes across archives dated ``since`` or later."""
    archives = sorted(p for p in config.EPISODES.glob("kaggriculture-episodes-*.zip") if p.stem[-10:] >= since)
    out: list[Episode] = []
    for archive in archives:
        for episode in iter_archive(archive):
            if episode.engine_version == "1.32.7":
                out.append(episode)
                if len(out) == n:
                    return out
    return out
```

- [ ] **Step 4: Run the eight-tape test until it passes**

Run: `uv run pytest tests/campaign/test_differential.py -v`
Expected: PASS. A mismatch names the seed, step and player; the offending dict key tells you which renderer branch or engine path is wrong. The port claims bit-identity; if a genuine engine divergence appears, record it in the test as an xfail with the seed and open it as a finding — do not patch `sim.hpp` in this task.

- [ ] **Step 5: Run the two-hundred-tape test once**

Run: `uv run pytest tests/campaign/test_differential.py -m slow -q`
Expected: 200 passed in a few minutes. This is the acceptance test named in the spec; paste its summary line into the commit message.

- [ ] **Step 6: Commit**

```bash
git add src/kaggriculture/campaign/tapes.py tests/campaign/test_differential.py pyproject.toml
git commit -m "test: prove the engine port bit-identical on archived tapes

<paste the -m slow summary line>"
```

---

### Task 6: Roster and harness (`play`, `check`, `package`, the CLI)

**Files:**

- Create: `src/kaggriculture/campaign/roster.py`, `src/kaggriculture/campaign/harness.py`
- Test: `tests/campaign/test_roster.py`, `tests/campaign/test_harness.py`

**Interfaces:**

- Consumes: `Engine`, `arena.run_banks`, `config`.
- Produces: `roster.TRAINING: dict[str, Path]`, `roster.HELD_OUT: dict[str, Path]`, `roster.path(name: str) -> Path`, `roster.names() -> list[str]`; `harness.Game` (pydantic: `opponent: str`, `seed: int`, `seat: int`, `ours: float`, `theirs: float`, `worst_step_seconds: float`), `harness.play(agent: Path, opponents: Sequence[str], seeds: Sequence[int], workers: int) -> list[Game]`, `harness.check(agent: Path) -> CheckReport` (pydantic: `loaded: bool`, `bank: float`, `worst_step_seconds: float`, `error: str | None`), `harness.package(agent: Path, output: Path) -> Path`, `harness.main()`.

- [ ] **Step 1: Write the failing tests**

`tests/campaign/test_roster.py`:

```python
from pathlib import Path

import pytest

from kaggriculture.campaign import roster


def test_every_roster_entry_exists_on_disk() -> None:
    for name in roster.names():
        assert roster.path(name).exists(), name


def test_training_and_held_out_do_not_overlap() -> None:
    assert not set(roster.TRAINING) & set(roster.HELD_OUT)


def test_an_unknown_name_is_an_error_not_a_path() -> None:
    with pytest.raises(KeyError):
        roster.path("../../../etc/passwd")


def test_no_roster_path_is_ever_relative_or_inside_the_repo() -> None:
    for name in roster.names():
        assert roster.path(name).is_absolute()
        assert not str(roster.path(name)).startswith(str(Path.cwd()))
```

`tests/campaign/test_harness.py`:

```python
"""What a sandbox may run, and what it may not."""

from pathlib import Path

import pytest

from kaggriculture.campaign import config, harness

PASS_AGENT = '''
def agent(observation, configuration=None):
    return {"farmer": ["PASS"], "hands": [], "market": []}
'''

SLOW_AGENT = '''
import time
def agent(observation, configuration=None):
    time.sleep(0.6)
    return {"farmer": ["PASS"], "hands": [], "market": []}
'''


@pytest.fixture
def pass_agent(tmp_path: Path) -> Path:
    path = tmp_path / "main.py"
    path.write_text(PASS_AGENT, encoding="utf-8")
    return path


def test_play_refuses_exam_seeds(pass_agent: Path) -> None:
    with pytest.raises(ValueError, match="exam"):
        harness.play(pass_agent, ["v54"], [config.EXAM_SEEDS[0]], workers=1)


def test_play_refuses_a_path_where_a_name_belongs(pass_agent: Path) -> None:
    with pytest.raises(KeyError):
        harness.play(pass_agent, ["/data/kaggriculture/opponents/kaito_v54/main.py"], [1], workers=1)


def test_play_reports_both_seats_and_latency(pass_agent: Path) -> None:
    games = harness.play(pass_agent, ["v54"], [1, 2], workers=2)
    assert [(g.seed, g.seat) for g in games] == [(1, 0), (1, 1), (2, 0), (2, 1)]
    assert all(g.ours == 3000.0 for g in games)
    assert all(g.worst_step_seconds < 0.5 for g in games)


def test_play_caps_workers_at_the_core_budget(pass_agent: Path) -> None:
    with pytest.raises(ValueError, match="CORE_BUDGET"):
        harness.play(pass_agent, ["v54"], [1], workers=config.CORE_BUDGET + 1)


def test_check_loads_as_kaggle_does_and_times_steps(pass_agent: Path) -> None:
    report = harness.check(pass_agent)
    assert report.loaded and report.error is None and report.bank == 3000.0
    assert report.worst_step_seconds < 0.5


def test_check_flags_a_slow_agent(tmp_path: Path) -> None:
    slow = tmp_path / "main.py"
    slow.write_text(SLOW_AGENT, encoding="utf-8")
    report = harness.check(slow, steps=5)
    assert report.worst_step_seconds >= 0.5


def test_package_places_main_and_the_engine_library_at_the_root(pass_agent: Path, tmp_path: Path) -> None:
    import tarfile

    archive = harness.package(pass_agent, tmp_path / "submission.tar.gz")
    with tarfile.open(archive) as tar:
        names = set(tar.getnames())
    assert "main.py" in names and "kaggriculture_engine.so" in names
    assert "kaggriculture/constants.py" in names
    assert not any(name.startswith("kaggriculture/campaign") for name in names)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/campaign/test_roster.py tests/campaign/test_harness.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write `roster.py`**

```python
"""Opponents by name. The paths are the only secret the campaign keeps.

A sandbox learns an opponent's name and its measured behaviour, never where
its source lives. ``path`` is called only by the harness and the gate.
"""

from pathlib import Path

from kaggriculture.campaign import config

AGENTS = Path("/data/kaggriculture/agents")

TRAINING: dict[str, Path] = {
    "router_v1": AGENTS / "yhay81_router_v1" / "main.py",
    "router2929": config.OPPONENTS / "yhay81_router2929" / "main.py",
    "v54": config.OPPONENTS / "kaito_v54" / "main.py",
    "v56": config.OPPONENTS / "kaito_v56" / "main.py",
    "shopforge": config.OPPONENTS / "tetsutani_shopforge" / "main.py",
    "indarkarhana": config.OPPONENTS / "indarkarhana_top10" / "main.py",
}
HELD_OUT: dict[str, Path] = {
    "salemali7_2900": config.OPPONENTS / "salemali7_2900" / "main.py",
    "lynnsakurai_v5": config.OPPONENTS / "lynnsakurai_v5" / "main.py",
}


def names() -> list[str]:
    """Every name the harness accepts."""
    return [*TRAINING, *HELD_OUT]


def path(name: str) -> Path:
    """Resolve a name; anything else is a KeyError, never a path lookup."""
    if name in TRAINING:
        return TRAINING[name]
    return HELD_OUT[name]
```

Check `ls /data/kaggriculture/opponents/salemali7_2900 /data/kaggriculture/opponents/lynnsakurai_v5` for the entrypoint file name and adjust if it is not `main.py`.

- [ ] **Step 4: Write `harness.py`**

```python
"""What a mutation sandbox may run: play by name, check as Kaggle loads, package.

Games run on the engine port; a 2% sample is replayed on the reference engine
and any bank disagreement fails the call, so drift between the two engines is
a harness error rather than a quietly wrong fitness.
"""

import argparse
import logging
import random
import shutil
import tarfile
import tempfile
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

from kaggle_environments import make
from kaggle_environments.agent import get_last_callable
from kaggle_environments.utils import structify
from pydantic import BaseModel

from kaggriculture.campaign import arena, config, roster
from kaggriculture.campaign.engine.wrapper import Engine
from kaggriculture.constants import ENVIRONMENT, EPISODE_STEPS

LOGGER = logging.getLogger(__name__)
REFERENCE_SAMPLE = 0.02
LATENCY_BUDGET = 0.5   # half of actTimeout
PACKAGE_MODULES = ("__init__.py", "constants.py", "observation.py", "actions.py")


class Game(BaseModel):
    opponent: str
    seed: int
    seat: int
    ours: float
    theirs: float
    worst_step_seconds: float


class CheckReport(BaseModel):
    loaded: bool
    bank: float
    worst_step_seconds: float
    error: str | None


def main() -> None:
    """``campaign play|check|package``."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    play_parser = commands.add_parser("play")
    play_parser.add_argument("agent", type=Path)
    play_parser.add_argument("--vs", nargs="+", required=True, help="opponent names")
    play_parser.add_argument("--seeds", required=True, help="A-B inclusive")
    play_parser.add_argument("--workers", type=int, default=4)
    check_parser = commands.add_parser("check")
    check_parser.add_argument("agent", type=Path)
    package_parser = commands.add_parser("package")
    package_parser.add_argument("agent", type=Path)
    package_parser.add_argument("--output", type=Path, default=Path("submission.tar.gz"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.command == "play":
        first, last = (int(part) for part in args.seeds.split("-"))
        for game in play(args.agent, args.vs, range(first, last + 1), args.workers):
            LOGGER.info("%s", game.model_dump_json())
    elif args.command == "check":
        LOGGER.info("%s", check(args.agent).model_dump_json())
    else:
        LOGGER.info("wrote %s", package(args.agent, args.output))


def load_agent(path: Path) -> Any:
    """Load an agent file the way the Kaggle runner does: the last callable."""
    return get_last_callable(path.read_text(encoding="utf-8"), path=str(path))


def configuration() -> Any:
    return make(ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS}).configuration


def _one(work: tuple[str, str, str, int, int]) -> Game:
    """Play one game on the engine port. Runs in a fresh process per game."""
    agent_path, opponent_name, opponent_path, seed, seat = work
    agents = [load_agent(Path(agent_path)), load_agent(Path(opponent_path))]
    if seat == 1:
        agents.reverse()
    engine = Engine(seed=seed)
    conf = configuration()
    worst = 0.0
    while not engine.done:
        actions = []
        for player, agent in enumerate(agents):
            observation = structify({**engine.observation(player), "remainingOverageTime": 60})
            started = perf_counter()
            actions.append(agent(observation, conf))
            elapsed = perf_counter() - started
            if player == seat:
                worst = max(worst, elapsed)
        engine.step(actions[0], actions[1])
    ours, theirs = engine.bank(seat), engine.bank(1 - seat)
    return Game(opponent=opponent_name, seed=seed, seat=seat, ours=ours, theirs=theirs, worst_step_seconds=worst)


def play(agent: Path, opponents: Sequence[str], seeds: Sequence[int], workers: int) -> list[Game]:
    """Play every (opponent, seed, seat) on the port; verify a sample on the reference engine.

    Raises:
        ValueError: An exam seed, or more workers than the core budget.
        KeyError: An opponent name not in the roster.
        RuntimeError: The reference-engine sample disagreed with the port.
    """
    if any(seed in config.EXAM_SEEDS for seed in seeds):
        raise ValueError("exam seeds are sealed; the harness will not play them")
    if workers > config.CORE_BUDGET:
        raise ValueError(f"workers exceeds CORE_BUDGET ({config.CORE_BUDGET})")
    paths = {name: roster.path(name) for name in opponents}
    work = [(str(agent), name, str(paths[name]), seed, seat) for name in opponents for seed in seeds for seat in (0, 1)]
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as pool:
        games = list(pool.map(_one, work))
    _verify_sample(agent, paths, games)
    return games


def _verify_sample(agent: Path, paths: dict[str, Path], games: list[Game]) -> None:
    rng = random.Random(len(games))
    for game in games:
        if rng.random() >= REFERENCE_SAMPLE:
            continue
        seating = (str(agent), str(paths[game.opponent])) if game.seat == 0 else (str(paths[game.opponent]), str(agent))
        banks = arena.run_banks(*seating, game.seed)
        ours, theirs = banks if game.seat == 0 else banks[::-1]
        if (float(ours), float(theirs)) != (game.ours, game.theirs):
            raise RuntimeError(
                f"engine drift: seed {game.seed} seat {game.seat} vs {game.opponent}: "
                f"port {game.ours}/{game.theirs}, reference {ours}/{theirs}"
            )


def check(agent: Path, steps: int = EPISODE_STEPS) -> CheckReport:
    """Load as Kaggle does and play up to ``steps`` turns vs itself on the reference engine."""
    try:
        policy = load_agent(agent)
    except Exception as error:  # noqa: BLE001 - the report is the point
        return CheckReport(loaded=False, bank=0.0, worst_step_seconds=0.0, error=f"{type(error).__name__}: {error}")
    environment = make(ENVIRONMENT, configuration={"episodeSteps": EPISODE_STEPS})
    environment.reset()
    worst = 0.0
    try:
        for _ in range(steps):
            if environment.done:
                break
            actions = []
            for state in environment.state:
                started = perf_counter()
                actions.append(policy(state.observation, environment.configuration))
                worst = max(worst, perf_counter() - started)
            environment.step(actions)
    except Exception as error:  # noqa: BLE001
        return CheckReport(loaded=True, bank=0.0, worst_step_seconds=worst, error=f"{type(error).__name__}: {error}")
    return CheckReport(loaded=True, bank=float(environment.state[0].observation["farms"][0]["money"]), worst_step_seconds=worst, error=None)


def package(agent: Path, output: Path) -> Path:
    """Write ``main.py`` + the engine library + the plumbing package into a tarball."""
    from kaggriculture.campaign.engine import build

    library = build.build()
    package_root = Path(config.ROOT / "src" / "kaggriculture")
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        shutil.copy(agent, root / "main.py")
        shutil.copy(library, root / library.name)
        (root / "kaggriculture").mkdir()
        for module in PACKAGE_MODULES:
            shutil.copy(package_root / module, root / "kaggriculture" / module)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output, "w:gz") as tar:
            for item in sorted(root.rglob("*")):
                tar.add(item, arcname=str(item.relative_to(root)))
    return output


if __name__ == "__main__":
    main()
```

The `_one` worker copies `remainingOverageTime: 60` into the observation because the framework adds it and vendored agents may read it.

- [ ] **Step 5: Run the tests until they pass**

Run: `uv run pytest tests/campaign/test_roster.py tests/campaign/test_harness.py -v`
Expected: PASS. The `play` test may trip `_verify_sample` (2% of 4 games ≈ never); if the check agent's per-step latency on the reference engine exceeds 0.5 s for the PASS agent, the box is overloaded — rerun.

- [ ] **Step 6: Commit**

```bash
git add src/kaggriculture/campaign/roster.py src/kaggriculture/campaign/harness.py tests/campaign/test_roster.py tests/campaign/test_harness.py
git commit -m "feat: give sandboxes a harness that plays opponents by name on the engine port"
```

---

### Task 7: Copy check

**Files:**

- Create: `src/kaggriculture/campaign/copycheck.py`
- Test: `tests/campaign/test_copycheck.py`

**Interfaces:**

- Produces: `copycheck.similarity(a: str, b: str, k: int = 8) -> float` (Jaccard over token k-shingles), `copycheck.against_opponents(source: str) -> tuple[str, float]` (worst offender name, score), `copycheck.THRESHOLD: float`.

- [ ] **Step 1: Write the failing tests**

```python
"""Copied code cannot pass the gate."""

from pathlib import Path

from kaggriculture.campaign import copycheck, roster

SKELETON = Path("src/kaggriculture/served/main.py").read_text(encoding="utf-8")


def test_a_verbatim_opponent_scores_one() -> None:
    source = roster.path("v54").read_text(encoding="utf-8")
    name, score = copycheck.against_opponents(source)
    assert name == "v54" and score > 0.99


def test_the_skeleton_scores_near_zero() -> None:
    _, score = copycheck.against_opponents(SKELETON)
    assert score < 0.05


def test_a_lifted_block_is_caught() -> None:
    source = roster.path("shopforge").read_text(encoding="utf-8")
    lines = source.splitlines()
    lifted = "\n".join(lines[len(lines) // 2 : len(lines) // 2 + 200])
    candidate = SKELETON + "\n" + lifted
    _, score = copycheck.against_opponents(candidate)
    assert score >= copycheck.THRESHOLD


def test_similarity_is_symmetric_and_bounded() -> None:
    a, b = "x = 1\ny = x + 2\n" * 20, "y = 2\nx = y + 1\n" * 20
    assert copycheck.similarity(a, b) == copycheck.similarity(b, a)
    assert 0.0 <= copycheck.similarity(a, b) <= 1.0
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/campaign/test_copycheck.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write `copycheck.py`**

```python
"""Token-shingle similarity between a candidate and every opponent source.

Independent programs that solve the same game share vocabulary (the op
names, the item names) but not eight-token sequences; a lifted block
shares hundreds. THRESHOLD is set from the skeleton and a lifted block, not
from theory, and the tests pin both sides.
"""

import tokenize
from io import StringIO
from pathlib import Path

from kaggriculture.campaign import roster

ROUTER_POLICY = Path("/data/kaggriculture/agents/yhay81_router_v1/source/policy.cpp")
THRESHOLD = 0.02


def tokens(source: str) -> list[str]:
    """Python tokens with names and numbers kept literal, comments and strings dropped."""
    out: list[str] = []
    try:
        for token in tokenize.generate_tokens(StringIO(source).readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
                continue
            out.append(token.string)
    except tokenize.TokenizeError:
        out = source.split()
    return out


def shingles(source: str, k: int) -> set[tuple[str, ...]]:
    stream = tokens(source)
    return {tuple(stream[i : i + k]) for i in range(max(0, len(stream) - k + 1))}


def similarity(a: str, b: str, k: int = 8) -> float:
    """Jaccard similarity of the two sources' k-shingle sets."""
    sa, sb = shingles(a, k), shingles(b, k)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def against_opponents(source: str) -> tuple[str, float]:
    """The most similar opponent source and its score."""
    worst_name, worst = "", 0.0
    corpus = {name: roster.path(name).read_text(encoding="utf-8", errors="replace") for name in roster.names()}
    corpus["router_policy_cpp"] = ROUTER_POLICY.read_text(encoding="utf-8", errors="replace")
    for name, text in corpus.items():
        score = similarity(source, text)
        if score > worst:
            worst_name, worst = name, score
    return worst_name, worst
```

Compiled opponents' `main.py` files are loaders; their strategy is in `.cpp`/`.inc`, which is why `policy.cpp` is in the corpus. For `router2929` and `indarkarhana`, add every `*.py`, `*.cpp`, `*.inc` under their directories to the corpus (walk `roster.path(name).parent`).

- [ ] **Step 4: Calibrate and run**

Run: `uv run pytest tests/campaign/test_copycheck.py -v`
Expected: PASS. If the skeleton scores above 0.05 or the lifted block below `THRESHOLD`, print both numbers, set `THRESHOLD` to the midpoint on a log scale, and record the two numbers in the module docstring.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/campaign/copycheck.py tests/campaign/test_copycheck.py
git commit -m "feat: reject candidates that carry an opponent's code"
```

---

### Task 8: Validation

**Files:**

- Create: `src/kaggriculture/campaign/validate.py`
- Test: `tests/campaign/test_validate.py`

**Interfaces:**

- Consumes: `copycheck.against_opponents`, `harness.check`, `harness.package`.
- Produces: `validate.Verdict` (pydantic: `status: Literal["ok", "syntax", "contract", "imports", "copy", "crashed", "too_slow"]`, `reason: str`, `worst_step_seconds: float`), `validate.validate(agent: Path) -> Verdict`, `validate.ALLOWED_IMPORTS: frozenset[str]`.

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

from kaggriculture.campaign import roster, validate

GOOD = '''
import math
from kaggriculture.constants import CROPS

def agent(observation, configuration=None):
    return {"farmer": ["PASS"], "hands": [], "market": []}
'''


def write(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "main.py"
    path.write_text(source, encoding="utf-8")
    return path


def test_a_good_agent_is_ok(tmp_path: Path) -> None:
    verdict = validate.validate(write(tmp_path, GOOD))
    assert verdict.status == "ok", verdict.reason


def test_syntax_error(tmp_path: Path) -> None:
    assert validate.validate(write(tmp_path, "def agent(:\n")).status == "syntax"


def test_missing_entrypoint_is_a_contract_failure(tmp_path: Path) -> None:
    assert validate.validate(write(tmp_path, "x = 1\n")).status == "contract"


def test_forbidden_import(tmp_path: Path) -> None:
    verdict = validate.validate(write(tmp_path, "import socket\n" + GOOD))
    assert verdict.status == "imports" and "socket" in verdict.reason


def test_copied_opponent(tmp_path: Path) -> None:
    verdict = validate.validate(write(tmp_path, roster.path("v56").read_text(encoding="utf-8")))
    assert verdict.status == "copy" and "v56" in verdict.reason


def test_crash(tmp_path: Path) -> None:
    verdict = validate.validate(write(tmp_path, "def agent(o, c=None):\n    raise ValueError('x')\n"))
    assert verdict.status == "crashed"


def test_too_slow(tmp_path: Path) -> None:
    slow = "import time\ndef agent(o, c=None):\n    time.sleep(0.6)\n    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
    verdict = validate.validate(write(tmp_path, slow), steps=3)
    assert verdict.status == "too_slow"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/campaign/test_validate.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write `validate.py`**

```python
"""Everything that rejects a candidate before a game is scored.

Cheap checks first, in the order they were each found necessary.
"""

import ast
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from kaggriculture.campaign import copycheck, harness

ALLOWED_IMPORTS: frozenset[str] = frozenset({
    "math", "statistics", "itertools", "collections", "functools", "operator", "random",
    "heapq", "bisect", "dataclasses", "typing", "enum", "copy", "json", "ctypes", "pathlib",
    "kaggriculture", "kaggle_environments",
})


class Verdict(BaseModel):
    status: Literal["ok", "syntax", "contract", "imports", "copy", "crashed", "too_slow"]
    reason: str
    worst_step_seconds: float = 0.0


def validate(agent: Path, steps: int = 720) -> Verdict:
    """Return the first failing check, or ``ok``."""
    source = agent.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return Verdict(status="syntax", reason=str(error))
    if not any(isinstance(node, ast.FunctionDef) and node.name == "agent" for node in tree.body):
        return Verdict(status="contract", reason="no top-level function named agent")
    for node in ast.walk(tree):
        names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else (
            [node.module or ""] if isinstance(node, ast.ImportFrom) else []
        )
        for name in names:
            if name.split(".", 1)[0] not in ALLOWED_IMPORTS:
                return Verdict(status="imports", reason=f"import of {name!r} is not allowed")
    offender, score = copycheck.against_opponents(source)
    if score >= copycheck.THRESHOLD:
        return Verdict(status="copy", reason=f"{score:.3f} similar to {offender}")
    report = harness.check(agent, steps=steps)
    if not report.loaded or report.error is not None:
        return Verdict(status="crashed", reason=report.error or "did not load", worst_step_seconds=report.worst_step_seconds)
    if report.worst_step_seconds > harness.LATENCY_BUDGET:
        return Verdict(status="too_slow", reason=f"worst step {report.worst_step_seconds:.3f}s", worst_step_seconds=report.worst_step_seconds)
    return Verdict(status="ok", reason="", worst_step_seconds=report.worst_step_seconds)
```

- [ ] **Step 4: Run the tests until they pass**

Run: `uv run pytest tests/campaign/test_validate.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/campaign/validate.py tests/campaign/test_validate.py
git commit -m "feat: validate a candidate before it costs a game"
```

---

### Task 9: Kaggle-image load test

**Files:**

- Create: `src/kaggriculture/campaign/kaggle_image.py`
- Test: `tests/campaign/test_kaggle_image.py` (marked `slow`; requires docker)

**Interfaces:**

- Consumes: `harness.package`.
- Produces: `kaggle_image.load_test(tarball: Path) -> str` (the container's stdout; raises `RuntimeError` on a nonzero exit).

- [ ] **Step 1: Write the failing test**

```python
import shutil
from pathlib import Path

import pytest

from kaggriculture.campaign import harness, kaggle_image

pytestmark = pytest.mark.slow

PASS_AGENT = "def agent(observation, configuration=None):\n    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available")
def test_the_packaged_skeleton_loads_and_plays_in_the_kaggle_image(tmp_path: Path) -> None:
    agent = tmp_path / "main.py"
    agent.write_text(PASS_AGENT, encoding="utf-8")
    tarball = harness.package(agent, tmp_path / "submission.tar.gz")
    output = kaggle_image.load_test(tarball)
    assert "ENGINE_OK 1.32.7" in output and "EPISODE_OK" in output
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/campaign/test_kaggle_image.py -m slow -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write `kaggle_image.py`**

```python
"""Prove a tarball loads on the Kaggle image before a slot is spent on it.

The image is the notebook image already pulled on this box; the runner's
glibc and architecture are the same, which is what the library depends on.
"""

import subprocess
from pathlib import Path

IMAGE = "gcr.io/kaggle-gpu-images/python:latest"
SCRIPT = r"""
import ctypes, tarfile, subprocess, sys
tarfile.open('/work/submission.tar.gz').extractall('/work/agent')
lib = ctypes.CDLL('/work/agent/kaggriculture_engine.so')
lib.kag_engine_version.restype = ctypes.c_char_p
print('ENGINE_OK', lib.kag_engine_version().decode())
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'kaggle-environments==1.32.7'], check=True)
from kaggle_environments import make
env = make('kaggriculture', configuration={'episodeSteps': 720})
env.run(['/work/agent/main.py', '/work/agent/main.py'])
print('EPISODE_OK', env.steps[-1][0].reward, env.steps[-1][1].reward)
"""


def load_test(tarball: Path) -> str:
    """Run the load script against ``tarball`` inside the Kaggle image."""
    command = [
        "docker", "run", "--rm", "-v", f"{tarball.resolve()}:/work/submission.tar.gz:ro",
        "-v", f"{tarball.resolve().parent}:/work", IMAGE, "python", "-c", SCRIPT,
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:])
    return result.stdout
```

- [ ] **Step 4: Run it until it passes**

Run: `uv run pytest tests/campaign/test_kaggle_image.py -m slow -v`
Expected: PASS (first run pulls nothing; the image is local). If the container cannot install `kaggle-environments`, the image has no network — replace the `pip install` with a bind-mount of this repo's `.venv/lib/python3.11/site-packages/kaggle_environments` into the container and add it to `sys.path`.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/campaign/kaggle_image.py tests/campaign/test_kaggle_image.py
git commit -m "test: load a packaged agent inside the Kaggle image before shipping it"
```

---

### Task 10: Fold the field gate onto the port

**Files:**

- Modify: `src/kaggriculture/campaign/field_gate.py`
- Test: `tests/campaign/test_field_gate.py`

**Interfaces:**

- Consumes: `harness.play`, `roster.TRAINING`, `config.EXAM_SEEDS`, `report.wilson_interval`.
- Produces: `field_gate.score_field(candidate: Path, seeds: Sequence[int], workers: int, opponents: Sequence[str]) -> dict[str, float]` (per-opponent win rate, ties as half, both seats) — the deep evaluator of Plan 2 calls this with `config.EXAM_SEEDS`; `field_gate.FIELD` is deleted in favour of `roster.TRAINING`.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from kaggriculture.campaign import field_gate

PASS_AGENT = "def agent(observation, configuration=None):\n    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"


def test_score_field_returns_one_rate_per_opponent_on_the_given_seeds(tmp_path: Path) -> None:
    agent = tmp_path / "main.py"
    agent.write_text(PASS_AGENT, encoding="utf-8")
    rates = field_gate.score_field(agent, seeds=[1, 2], workers=4, opponents=["v54", "v56"])
    assert set(rates) == {"v54", "v56"}
    assert all(rate == 0.0 for rate in rates.values())   # PASS loses every game to a real economy
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/campaign/test_field_gate.py -v`
Expected: FAIL (signature mismatch).

- [ ] **Step 3: Rewrite `field_gate.py`**

Replace the module body with:

```python
"""Per-opponent win rates on a fixed seed block, both seats, ties as half.

The exam block (``config.EXAM_SEEDS``) is the one this is meant for; the
harness refuses it, so this module plays through the engine directly.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

from kaggriculture.campaign import config, harness, roster

LOGGER = logging.getLogger(__name__)


def score_field(candidate: Path, seeds: Sequence[int], workers: int, opponents: Sequence[str]) -> dict[str, float]:
    """Win rate per opponent over ``2 * len(seeds)`` games each."""
    games = harness.play_unsealed(candidate, opponents, seeds, workers)
    rates: dict[str, float] = {}
    for name in opponents:
        mine = [g for g in games if g.opponent == name]
        rates[name] = sum(1.0 if g.ours > g.theirs else 0.5 if g.ours == g.theirs else 0.0 for g in mine) / len(mine)
        LOGGER.info("%s: %.4f over %d games", name, rates[name], len(mine))
    return rates
```

and add to `harness.py` a `play_unsealed` that is `play` without the exam-seed refusal, documented as callable only by the gate:

```python
def play_unsealed(agent: Path, opponents: Sequence[str], seeds: Sequence[int], workers: int) -> list[Game]:
    """``play`` for the gate: the exam seeds are allowed. Never exposed on the CLI."""
    ...  # identical body to play() minus the EXAM_SEEDS check
```

Refactor so both share one private `_play` with a `sealed: bool` flag rather than duplicating the body. Delete the old `FIELD` dict, `main`, `report`, and the `--occupancy` machinery; `kernel_watch.gate` now calls `score_field(path, config.EXAM_SEEDS[:gate_seeds], workers, list(roster.TRAINING))`.

- [ ] **Step 4: Run the tests until they pass**

Run: `uv run pytest tests/campaign/test_field_gate.py tests/campaign/test_kernel_watch.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/campaign/field_gate.py src/kaggriculture/campaign/harness.py src/kaggriculture/campaign/kernel_watch.py tests/campaign/test_field_gate.py
git commit -m "refactor: score the field on the engine port, by opponent name"
```

---

### Task 11: The skeleton floor

**Files:**

- Modify: `src/kaggriculture/served/main.py`
- Test: `tests/campaign/test_served.py`

**Interfaces:**

- Produces: a served agent that passes `validate.validate` with status `ok`, uses `kaggriculture.observation.Observation` and `kaggriculture.actions.Turn`, and does one legal, measurable thing beyond PASS so the differential between "skeleton" and "nothing" is visible to the first mutations.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from kaggriculture.campaign import harness, validate

SERVED = Path("src/kaggriculture/served/main.py")


def test_the_skeleton_validates() -> None:
    assert validate.validate(SERVED).status == "ok"


def test_the_skeleton_plants_something(tmp_path: Path) -> None:
    games = harness.play(SERVED, ["v54"], [1], workers=2)
    assert all(game.ours >= 0 for game in games)
    report = harness.check(SERVED, steps=60)
    assert report.error is None
```

- [ ] **Step 2: Write the skeleton**

```python
"""The floor: our own agent, as far as the campaign has taken it.

This first version is deliberately minimal and entirely ours: the farmer
buys wheat seed on the first turn, plants a tile per turn while it has
seed, waters what it planted, harvests what is ready, and sells at the
market. The campaign replaces this file; nothing here is a strategy.
"""

from kaggriculture.actions import PASS, Turn, step_toward
from kaggriculture.observation import Observation


def agent(observation, configuration=None):
    """One turn: buy seed once, then plant, water, harvest, sell."""
    view = Observation.parse(observation)
    farm = view.farm
    turn = Turn()
    x, y = farm["farmer"]
    tile = farm["tiles"][y][x]
    seeds = view.private["seeds"]
    shed = view.private["shed"]
    if view.step == 0:
        turn.market.append(["BUY_SEED", "WHEAT", 20])
        return turn.to_action()
    if shed.get("WHEAT", 0) >= 10:
        turn.market.append(["SELL", "WHEAT", shed["WHEAT"]])
    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        if tile.get("yield_units", 0) > 0:
            turn.farmer = ["HARVEST"]
        elif not tile.get("watered_today"):
            turn.farmer = ["WATER"]
        else:
            turn.farmer = step_toward((x, y), _next_empty(farm, x, y))
    elif tile is None and seeds.get("WHEAT", 0) > 0:
        turn.farmer = ["PLANT", "WHEAT"]
    else:
        turn.farmer = step_toward((x, y), _next_empty(farm, x, y)) if seeds.get("WHEAT", 0) else list(PASS)
    return turn.to_action()


def _next_empty(farm, x, y):
    """Nearest unlocked empty tile to (x, y), or (x, y) itself."""
    best, best_distance = (x, y), 10**9
    for ty, row in enumerate(farm["tiles"]):
        for tx, tile in enumerate(row):
            if tile is None:
                distance = abs(tx - x) + abs(ty - y)
                if distance < best_distance:
                    best, best_distance = (tx, ty), distance
    return best
```

Check `PICKUP`/`HARVEST` semantics in `kaggriculture.py` lines 358–475: if harvest yields go to the unit's inventory rather than the shed, add a `DROP`-at-shed step (the shed-access tiles are `_shed_access_tiles`); the point is a legal loop that banks more than 3000 on the reference engine, verified by `harness.check`.

- [ ] **Step 3: Run the tests until they pass**

Run: `uv run pytest tests/campaign/test_served.py -v`
Expected: PASS, and `harness.check(SERVED).bank > 3000`.

- [ ] **Step 4: Commit**

```bash
git add src/kaggriculture/served/main.py tests/campaign/test_served.py
git commit -m "feat: a skeleton floor that is ours and plays a legal loop"
```

---

### Task 12: Phase 1 brief and the first task prompt draft

**Files:**

- Create: `docs/campaign/phase1-brief.md`, `src/kaggriculture/campaign/task_prompt.md`
- Create: `run/campaign/phase1/` (runtime; gitignored)

**Interfaces:**

- Produces: `task_prompt.md`, the file Plan 2's `prompt.py` copies into every sandbox as `AGENTS.md`.

- [ ] **Step 1: Write the brief**

`docs/campaign/phase1-brief.md`:

```markdown
# Phase 1: how Kaggriculture works

You are in a workspace containing `engine/kaggriculture.py` (the exact
engine, version 1.32.7, read it as ground truth), `docs/competition.md`
(notes, some stale — the engine wins any disagreement), and the harness
`campaign play/check` (opponents by name: router_v1, router2929, v54, v56,
shopforge, indarkarhana). You may not read opponent source; you may measure
them through the harness.

Write `notes/game-model.md`: how a season works (turns, days, hours, the
end-of-day transition), every unit action and what it costs and yields,
crops (seed cost, first yield day, interval, max yield, ongoing or not),
animals, structures, land, hiring (Fibonacci pricing), the market (base
prices, I0, the price curve shapes including the hinge, price impact of a
sale, town-centre and shop consumption, shop unlock schedule and the seed's
role), the shed cap and what it silently destroys, and what the win
condition rewards (relative bank at turn 720; win/loss only).

Every number must come from a script under `experiments/` that you ran
against the engine, and the note must cite the script. Then write
`task_prompt.md`: the same content compressed to what a policy author needs
in one read — objective, rules, interface, verified economics, the
doctrine (measure opponents through the harness; never read their source).
Mirror the shape of a competition task prompt: objective first, key
insights second, rules third, interface last.
```

- [ ] **Step 2: Set up the workspace and run codex**

```bash
mkdir -p run/campaign/phase1/engine run/campaign/phase1/notes run/campaign/phase1/experiments
cp "$(uv run python -c 'import kaggle_environments.envs.kaggriculture.kaggriculture as m; print(m.__file__)')" run/campaign/phase1/engine/
cp docs/competition.md run/campaign/phase1/docs-competition.md
cp docs/campaign/phase1-brief.md run/campaign/phase1/AGENTS.md
cd run/campaign/phase1 && codex exec -s workspace-write -c approval_policy=never -m gpt-5.6-sol --json - < AGENTS.md > session-1.jsonl 2> session-1.err
```

Run two more sessions with `codex exec resume --last "Continue; verify anything unverified; then write task_prompt.md"`.

- [ ] **Step 3: Spot-check five numbers**

Pick five numbers from `notes/game-model.md` (a seed cost, a first-yield day, a base price, the hire price of the third hand, the shop unlock interval) and confirm each against `kaggriculture.py` constants. Any wrong number: send it back as the next resume prompt.

- [ ] **Step 4: Adopt the prompt**

```bash
cp run/campaign/phase1/task_prompt.md src/kaggriculture/campaign/task_prompt.md
mkdir -p docs/research/campaign && cp -r run/campaign/phase1/notes run/campaign/phase1/experiments docs/research/campaign/
git add src/kaggriculture/campaign/task_prompt.md docs/campaign/phase1-brief.md docs/research/campaign
git commit -m "docs: the task prompt every mutation reads, and the model it came from"
```

---

## Self-review

**Spec coverage.** §1 decisions 1–2, 5–7: Tasks 2, 3, 12. §5.1 engine + differential + Kaggle load: Tasks 3, 4, 5, 9. §5.2 sandbox harness surface: Task 6 (`play`, `check`, `package`). §5.4 validation: Tasks 7, 8. §9 removal: Task 2. §6 phase 0 dry run and §2–3, §5.5–5.7 (archive, evaluators, pool, gate, loop, promotion) are Plan 2. `kag_from_observation` is deferred by the spec itself.

**Placeholders.** Two places defer to the reference source by function name (empty-structure tile dict in Task 4; harvest/shed semantics in Task 11); each names the exact function and the test that arbitrates.

**Type consistency.** `harness.play` returns `list[Game]` and is consumed by `field_gate.score_field`; `harness.check` returns `CheckReport`, consumed by `validate`; `arena.run_banks` returns `tuple[int, int]`, consumed by `harness._verify_sample`; `roster.path` raises `KeyError`, asserted in both roster and harness tests; `copycheck.THRESHOLD` used by `validate`.
