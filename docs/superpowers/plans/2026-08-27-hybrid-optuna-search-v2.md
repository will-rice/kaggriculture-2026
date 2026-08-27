# Hybrid Optuna Search v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the ineffective one-hot hybrid evolution with a certified, resumable Optuna TPE search over typed semantic policy parameters, using progressive league evidence, one persistent CPU arena, and one failure-isolated W&B run.

**Architecture:** A semantic-space module maps Optuna trials directly to strict `HybridConfig` values. One sequential Optuna coordinator owns SQLite, pruning, evidence, and callbacks while a persistent 32-process arena executes canonical seat-swapped games. A new certified finalist schema binds the completed study to the existing single-use promotion boundary without modifying the old generation-8 artifact or touching promotion seeds during search.

**Tech Stack:** Python 3.11, Pydantic 2.12, Optuna 4.9.0, optuna-integration 4.9.0, W&B 0.28, SQLite, `ProcessPoolExecutor`, `kaggle-environments` 1.32.7, pytest, Ruff, ty.

**Spec:** `docs/superpowers/specs/2026-08-27-hybrid-optuna-search-v2-design.md`

## Global Constraints

- Preserve the old generation-8 state byte-for-byte at `run/hybrid/search-state.json`, SHA-256 `c74ae9ff99fc6bd1d39c93f2edda8ac3a9ee1894d1422c2e02e9c6d02aab5c5b`.
- Create a new study named `hybrid-optuna-v2` under `run/hybrid/optuna-v2/`; never migrate the legacy evolution state into SQLite.
- Use `TPESampler(seed=20260827, multivariate=True, group=True, n_startup_trials=16)` and `SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)`.
- Call `study.optimize(objective, n_jobs=1, callbacks=callbacks, catch=(TrialEvaluationError,))`; parallelize episodes only through one persistent executor.
- Default to 32 CPU workers, accept 1–32, perform no internal `os.nice`, and perform no GPU work.
- Use nested search seeds `860000..860003`, `860000..860015`, and `860000..860031`; they must remain disjoint from frontier, legacy search, determinism, and promotion seeds.
- Treat win points as primary and add only `1e-6 * dense_margin_0_to_1`; a real win-point increment must always outrank any margin difference.
- Fail an entire trial on an illegal action, exception, timeout, invalid status, missing provenance, or malformed result.
- The immutable study maximum is 512 terminal trials; `--stop-after 32` is an operational pilot boundary, not a second study identity.
- Stop at 128 terminal trials with no finalists if every clean candidate has zero `economic_policy` win points.
- W&B is never authoritative. Import, authentication, initialization, log, or finish failure must not change SQLite, trial order, trial state, evidence, or finalists.
- Search never consumes `PROMOTION_SEEDS`, edits `main.py`, packages an agent, or submits to Kaggle.
- A qualifying search writes finalists only; the existing Task 9 holdout remains a separate single-use boundary.

---

## File Structure

### New search-v2 modules

- `src/kaggriculture/search/optuna_space.py`: stable semantic parameter manifest, trial-to-`HybridConfig` conversion, config-to-parameter conversion, and space digest.
- `src/kaggriculture/search/optuna_seeds.py`: three complete hand-authored configs plus validated import of default and four immutable generation-8 elites.
- `src/kaggriculture/search/arena_pool.py`: one context-managed persistent process pool and canonical game provenance.
- `src/kaggriculture/search/optuna_protocol.py`: fixed seed banks, deterministic opponent panels, rung evidence, score calculation, incremental-game selection, and atomic evidence serialization.
- `src/kaggriculture/search/optuna_state.py`: Pydantic study identity, path ownership, SQLite creation/resume validation, stale-running reconciliation, and terminal database hashing.
- `src/kaggriculture/search/optuna_search.py`: sequential coordinator, objective, pruning, stop gates, warm-start enqueue, terminal counts, and finalist trigger.
- `src/kaggriculture/search/optuna_wandb.py`: one persistent official Optuna W&B callback plus rich metrics and failure isolation.
- `src/kaggriculture/search/optuna_finalists.py`: certified finalist schema, database/evidence validation, atomic writer, and promotion-normalization adapter.
- `src/kaggriculture/search/scripts/hybrid_optuna.py`: production CLI and preflight ordering.

### Existing files modified narrowly

- `pyproject.toml`, `uv.lock`: pin Optuna and Optuna Integration 4.9.0.
- `src/kaggriculture/search/arena.py`: expose one typed, picklable single-game execution boundary while preserving all legacy APIs.
- `src/kaggriculture/search/scripts/hybrid_holdout.py`: accept either certified legacy finalists or certified Optuna finalists through one normalized adapter.
- `tests/search/`: add focused v2 contract, persistence, W&B, finalist, and integration tests; retain all v1 tests unchanged.

---

### Task 1: Pinned dependencies and semantic parameter space

**Files:**

- Modify: `pyproject.toml:7-23`
- Modify: `uv.lock`
- Create: `src/kaggriculture/search/optuna_space.py`
- Create: `src/kaggriculture/search/optuna_seeds.py`
- Create: `tests/search/test_optuna_space.py`
- Create: `tests/search/test_optuna_seeds.py`

**Interfaces:**

- Consumes: `HybridConfig`, `OpeningPhaseConfig`, `PHASE_START_DAYS`, `CROP_NAMES`, `ANIMAL_NAMES`, and `SearchState.load(path)`.
- Produces: `ParameterSpec`, `SpaceManifest`, `SPACE_MANIFEST`, `SPACE_SHA256`, `suggest_config(trial: TrialLike) -> HybridConfig`, `parameters_for_config(config: HybridConfig) -> dict[str, int | float | str]`, `config_sha256(config: HybridConfig) -> str`, `deliberate_seed_configs() -> tuple[HybridConfig, HybridConfig, HybridConfig]`, and `warm_start_configs(path: Path, expected_sha256: str = LEGACY_STATE_SHA256) -> tuple[HybridConfig, ...]`.

- [ ] **Step 1: Pin and install the reviewed Optuna versions**

Add these exact dependencies:

```toml
  "optuna==4.9.0",
  "optuna-integration[wandb]==4.9.0",
```

Run:

```bash
UV_CACHE_DIR=/tmp/kaggriculture-optuna-uv-cache uv lock
UV_CACHE_DIR=/tmp/kaggriculture-optuna-uv-cache uv sync
.venv/bin/python -c "import optuna, optuna_integration; assert optuna.__version__ == '4.9.0'"
```

Expected: dependency resolution succeeds and the assertion exits 0.

- [ ] **Step 2: Write RED semantic-space tests**

```python
def test_space_manifest_names_domains_and_hash_are_stable() -> None:
    assert SPACE_MANIFEST.parameters[0] == ParameterSpec(
        name="opening.phase_0.target_hands", kind="int", low=0, high=19
    )
    assert "opening.phase_0.start_day" not in {
        spec.name for spec in SPACE_MANIFEST.parameters
    }
    assert len(SPACE_SHA256) == 64


def test_fixed_trial_round_trips_every_semantic_parameter() -> None:
    expected = HybridConfig.default()
    trial = optuna.trial.FixedTrial(parameters_for_config(expected))
    actual = suggest_config(trial)
    assert actual == expected
    assert parameters_for_config(actual) == parameters_for_config(expected)


def test_margin_genome_is_not_part_of_optuna_space() -> None:
    names = {spec.name for spec in SPACE_MANIFEST.parameters}
    assert all("logit" not in name and "genome" not in name for name in names)
    assert len(names) < 64
```

Run: `.venv/bin/python -m pytest tests/search/test_optuna_space.py -q`

Expected: FAIL during collection because `optuna_space` does not exist.

- [ ] **Step 3: Implement the stable manifest and bidirectional conversion**

Use these exact public shapes:

```python
ParameterValue = int | float | str
ParameterKind = Literal["int", "float", "categorical"]


class TrialLike(Protocol):
    def suggest_int(self, name: str, low: int, high: int, *, step: int = 1) -> int: ...
    def suggest_float(self, name: str, low: float, high: float) -> float: ...
    def suggest_categorical(
        self, name: str, choices: Sequence[str]
    ) -> str: ...


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    kind: ParameterKind
    low: int | float | None = None
    high: int | float | None = None
    step: int | None = None
    choices: tuple[str, ...] = ()


@dataclass(frozen=True)
class SpaceManifest:
    parameters: tuple[ParameterSpec, ...]


def suggest_config(trial: TrialLike) -> HybridConfig:
    phases = []
    for index, allowed_days in enumerate(PHASE_START_DAYS):
        prefix = f"opening.phase_{index}"
        start_day = (
            allowed_days[0]
            if len(allowed_days) == 1
            else trial.suggest_int(f"{prefix}.start_day", allowed_days[0], allowed_days[-1])
        )
        phases.append(
            OpeningPhaseConfig(
                start_day=start_day,
                target_hands=trial.suggest_int(f"{prefix}.target_hands", 0, 19),
                target_quadrants=trial.suggest_int(f"{prefix}.target_quadrants", 1, 5),
                primary_crop=trial.suggest_categorical(
                    f"{prefix}.primary_crop", tuple(CROP_NAMES)
                ),
                crop_targets=tuple(
                    trial.suggest_int(f"{prefix}.crop.{name}", 0, 100)
                    for name in CROP_NAMES
                ),
                animal_targets=tuple(
                    trial.suggest_int(f"{prefix}.animal.{name}", 0, 100)
                    for name in ANIMAL_NAMES
                ),
                structure_targets=(
                    trial.suggest_int(f"{prefix}.structure.coop", 0, 100),
                    trial.suggest_int(f"{prefix}.structure.pasture", 0, 100),
                ),
                cash_reserve=trial.suggest_int(
                    f"{prefix}.cash_reserve", 0, 5_000, step=100
                ),
                inventory_reserve=trial.suggest_int(
                    f"{prefix}.inventory_reserve", 0, 100
                ),
            )
        )
    return HybridConfig(
        opening=OpeningConfig(phases=tuple(phases)),
        jobs=JobWeightsConfig(**_suggest_weights(trial, "jobs", JOB_WEIGHT_FIELDS)),
        market=MarketConfig(
            **_suggest_weights(trial, "market", MARKET_WEIGHT_FIELDS)
        ),
        liquidation_start_day=trial.suggest_int("liquidation_start_day", 20, 29),
    )
```

Generate `SPACE_MANIFEST` from the same named loops, serialize it with sorted
keys and compact separators, and define `SPACE_SHA256` as that canonical JSON's
SHA-256. Implement `parameters_for_config` with the exact inverse names and
values. Tests must compare the manifest to an independently written expected
name/domain table so a shared-loop bug cannot bless itself.

- [ ] **Step 4: Write RED warm-start tests**

```python
def test_deliberate_seeds_are_three_distinct_complete_configs() -> None:
    seeds = deliberate_seed_configs()
    assert len(seeds) == 3
    assert len({config_sha256(config) for config in seeds}) == 3
    assert {config.liquidation_start_day for config in seeds} == {23, 25, 27}


def test_warm_start_is_default_four_elites_then_three_deliberate(
    legacy_state_path: Path,
) -> None:
    configs = warm_start_configs(
        legacy_state_path,
        expected_sha256=hashlib.sha256(legacy_state_path.read_bytes()).hexdigest(),
    )
    assert len(configs) == 8
    assert configs[0] == HybridConfig.default()
    assert configs[1:5] == tuple(row.config for row in SearchState.load(legacy_state_path).elites)
    assert configs[5:] == deliberate_seed_configs()
    assert len({config_sha256(config) for config in configs}) == 8
```

Run: `.venv/bin/python -m pytest tests/search/test_optuna_seeds.py -q`

Expected: FAIL because the seed functions do not exist.

- [ ] **Step 5: Implement exact deliberate seeds and legacy import validation**

Define three complete literal configs, with crop tuple order
`(CARROT, MELON, STRAWBERRY, TOMATO, WHEAT)` and animal tuple order
`(COW, GOOSE, SHEEP)`:

```python
DELIBERATE_PAYLOADS: tuple[dict[str, object], ...] = (
    {
        "opening": {"phases": (
            {"start_day": 0, "target_hands": 3, "target_quadrants": 1,
             "primary_crop": "WHEAT", "crop_targets": (0, 0, 0, 0, 8),
             "animal_targets": (0, 0, 0), "structure_targets": (0, 0),
             "cash_reserve": 1000, "inventory_reserve": 4},
            {"start_day": 12, "target_hands": 6, "target_quadrants": 2,
             "primary_crop": "STRAWBERRY", "crop_targets": (0, 0, 10, 0, 8),
             "animal_targets": (0, 0, 0), "structure_targets": (0, 0),
             "cash_reserve": 1200, "inventory_reserve": 6},
            {"start_day": 22, "target_hands": 8, "target_quadrants": 3,
             "primary_crop": "STRAWBERRY", "crop_targets": (0, 0, 18, 0, 8),
             "animal_targets": (0, 0, 0), "structure_targets": (0, 0),
             "cash_reserve": 1500, "inventory_reserve": 8},
        )},
        "jobs": {"recovery": 9.0, "urgency": 7.0, "distance_penalty": 1.5,
                 "production": 7.0, "transport": 6.0, "structure": 2.0},
        "market": {"live_price": 7.0, "town_demand": 6.0,
                   "opponent_supply": 4.0, "reserve_penalty": 9.0,
                   "liquidation_urgency": 8.0},
        "liquidation_start_day": 25,
    },
    {
        "opening": {"phases": (
            {"start_day": 0, "target_hands": 5, "target_quadrants": 1,
             "primary_crop": "WHEAT", "crop_targets": (0, 0, 0, 0, 12),
             "animal_targets": (0, 0, 0), "structure_targets": (0, 0),
             "cash_reserve": 200, "inventory_reserve": 2},
            {"start_day": 10, "target_hands": 10, "target_quadrants": 3,
             "primary_crop": "STRAWBERRY", "crop_targets": (0, 0, 18, 0, 12),
             "animal_targets": (0, 0, 0), "structure_targets": (0, 0),
             "cash_reserve": 300, "inventory_reserve": 3},
            {"start_day": 20, "target_hands": 15, "target_quadrants": 5,
             "primary_crop": "STRAWBERRY", "crop_targets": (0, 0, 30, 0, 12),
             "animal_targets": (0, 0, 0), "structure_targets": (0, 0),
             "cash_reserve": 500, "inventory_reserve": 4},
        )},
        "jobs": {"recovery": 7.0, "urgency": 8.0, "distance_penalty": 0.5,
                 "production": 10.0, "transport": 7.0, "structure": 1.0},
        "market": {"live_price": 5.0, "town_demand": 5.0,
                   "opponent_supply": 3.0, "reserve_penalty": 3.0,
                   "liquidation_urgency": 10.0},
        "liquidation_start_day": 27,
    },
    {
        "opening": {"phases": (
            {"start_day": 0, "target_hands": 3, "target_quadrants": 1,
             "primary_crop": "WHEAT", "crop_targets": (0, 0, 0, 0, 6),
             "animal_targets": (0, 0, 0), "structure_targets": (0, 0),
             "cash_reserve": 1200, "inventory_reserve": 5},
            {"start_day": 14, "target_hands": 5, "target_quadrants": 2,
             "primary_crop": "STRAWBERRY", "crop_targets": (0, 0, 8, 0, 6),
             "animal_targets": (0, 0, 0), "structure_targets": (0, 0),
             "cash_reserve": 1800, "inventory_reserve": 8},
            {"start_day": 24, "target_hands": 6, "target_quadrants": 2,
             "primary_crop": "STRAWBERRY", "crop_targets": (0, 0, 10, 0, 6),
             "animal_targets": (0, 0, 0), "structure_targets": (0, 0),
             "cash_reserve": 2400, "inventory_reserve": 10},
        )},
        "jobs": {"recovery": 10.0, "urgency": 9.0, "distance_penalty": 2.0,
                 "production": 5.0, "transport": 8.0, "structure": 1.0},
        "market": {"live_price": 9.0, "town_demand": 8.0,
                   "opponent_supply": 6.0, "reserve_penalty": 10.0,
                   "liquidation_urgency": 10.0},
        "liquidation_start_day": 23,
    },
)
```

`warm_start_configs` must hash the source before parsing, require the expected
digest, load with `SearchState.load`, require `generation == 8`, require four
elites, concatenate default/elites/deliberate seeds, and reject any duplicate
canonical config digest.

- [ ] **Step 6: Run focused and adjacent gates**

Run:

```bash
.venv/bin/python -m pytest tests/search/test_optuna_space.py tests/search/test_optuna_seeds.py tests/search/test_genome.py tests/hybrid/test_config.py -q
.venv/bin/ruff check src/kaggriculture/search/optuna_space.py src/kaggriculture/search/optuna_seeds.py tests/search/test_optuna_space.py tests/search/test_optuna_seeds.py
.venv/bin/ty check src/kaggriculture/search/optuna_space.py src/kaggriculture/search/optuna_seeds.py
```

Expected: all tests and static checks pass.

- [ ] **Step 7: Commit Task 1**

```bash
git add pyproject.toml uv.lock src/kaggriculture/search/optuna_space.py src/kaggriculture/search/optuna_seeds.py tests/search/test_optuna_space.py tests/search/test_optuna_seeds.py
git commit -m "feat: define the semantic hybrid search space"
```

---

### Task 2: Persistent provenance-preserving arena

**Files:**

- Modify: `src/kaggriculture/search/arena.py:1-280`
- Create: `src/kaggriculture/search/arena_pool.py`
- Modify: `tests/search/test_arena.py`
- Create: `tests/search/test_arena_pool.py`

**Interfaces:**

- Consumes: `arena.Opponent`, `HybridOpponent`, and reference-engine episode execution.
- Produces: `GameKey(opponent: str, seed: int, seat: int)`, `GameTask(key: GameKey, candidate: HybridOpponent, opponent: Opponent)`, `GameResult(key: GameKey, ours: int | None, theirs: int | None, runtime_seconds: float, failure: str | None)`, `run_game_task(task: GameTask) -> GameResult`, and `PersistentArena.run(tasks: Sequence[GameTask]) -> tuple[GameResult, ...]`.

- [ ] **Step 1: Write RED single-game and persistent-pool tests**

```python
def test_run_game_task_preserves_exact_provenance_and_candidate_seat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(arena, "_run_banks", lambda left, right, seed: (91, 17))
    result = run_game_task(GameTask(
        GameKey("econ", 860000, 0), HybridOpponent(RUNTIME), "econ.py"
    ))
    assert result.key == GameKey("econ", 860000, 0)
    assert (result.ours, result.theirs, result.failure) == (91, 17, None)


def test_persistent_arena_constructs_one_executor_for_multiple_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = []
    monkeypatch.setattr(arena_pool, "ProcessPoolExecutor", recording_executor(created))
    with PersistentArena(workers=3) as pool:
        first = pool.run((TASK_A,))
        second = pool.run((TASK_B,))
    assert [row.key for row in first + second] == [TASK_A.key, TASK_B.key]
    assert len(created) == 1
    assert created[0].shutdown_calls == 1
```

Run: `.venv/bin/python -m pytest tests/search/test_arena_pool.py -q`

Expected: FAIL because the typed game boundary and persistent arena are absent.

- [ ] **Step 2: Refactor one episode behind a typed task without changing legacy APIs**

Add to `arena.py`:

```python
@dataclass(frozen=True, order=True)
class GameKey:
    opponent: str
    seed: int
    seat: int


@dataclass(frozen=True)
class GameTask:
    key: GameKey
    candidate: HybridOpponent
    opponent: Opponent


@dataclass(frozen=True)
class GameResult:
    key: GameKey
    ours: int | None
    theirs: int | None
    runtime_seconds: float
    failure: str | None = None


def run_game_task(task: GameTask) -> GameResult:
    started = perf_counter()
    try:
        seats = (
            (task.candidate, task.opponent)
            if task.key.seat == 0
            else (task.opponent, task.candidate)
        )
        left, right = _run_banks(*seats, task.key.seed)
        ours, theirs = (left, right) if task.key.seat == 0 else (right, left)
        return GameResult(task.key, ours, theirs, perf_counter() - started)
    except Exception as error:
        return GameResult(
            task.key,
            None,
            None,
            perf_counter() - started,
            f"{type(error).__name__}: {error}",
        )
```

Extract the current `_one` engine work into `_run_banks`; keep `play`,
`outcomes`, `action_traces`, and their order/error behavior unchanged by having
their adapters raise `RuntimeError(result.failure)` for a failed `GameResult`.
Reject non-integer seeds, seats outside `{0, 1}`, empty opponent names, negative
runtimes, and success rows missing banks in dataclass validation.

- [ ] **Step 3: Implement the one-owner persistent executor**

```python
class PersistentArena:
    def __init__(self, workers: int = 32) -> None:
        if type(workers) is not int or not 1 <= workers <= 32:
            raise ValueError("workers must be between 1 and 32")
        self.workers = workers
        self._pool: ProcessPoolExecutor | None = None

    def __enter__(self) -> Self:
        if self._pool is not None:
            raise RuntimeError("persistent arena is already open")
        self._pool = ProcessPoolExecutor(max_workers=self.workers)
        return self

    def run(self, tasks: Sequence[GameTask]) -> tuple[GameResult, ...]:
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
        assert self._pool is not None
        self._pool.shutdown(wait=True, cancel_futures=True)
        self._pool = None
```

- [ ] **Step 4: Differential-test persistent and legacy reference results**

Add a real two-seed test using `HybridConfig.default()` against
`economic_policy`: build four `GameTask`s, run them through
`PersistentArena(workers=2)`, call legacy `arena.outcomes` on the same seeds,
and assert identical win points, normalized margins, ordering, and zero
failures. Add failure injection for invalid final status and prove the row has
the exact key and a nonempty failure instead of losing provenance.

Run:

```bash
.venv/bin/python -m pytest tests/search/test_arena.py tests/search/test_arena_pool.py -q
```

Expected: all pass.

- [ ] **Step 5: Static checks and commit Task 2**

```bash
.venv/bin/ruff check src/kaggriculture/search/arena.py src/kaggriculture/search/arena_pool.py tests/search/test_arena.py tests/search/test_arena_pool.py
.venv/bin/ty check src/kaggriculture/search/arena.py src/kaggriculture/search/arena_pool.py
git add src/kaggriculture/search/arena.py src/kaggriculture/search/arena_pool.py tests/search/test_arena.py tests/search/test_arena_pool.py
git commit -m "feat: keep one hybrid evaluation arena alive"
```

---

### Task 3: Progressive panels, evidence, and lexicographic objective

**Files:**

- Create: `src/kaggriculture/search/optuna_protocol.py`
- Create: `tests/search/test_optuna_protocol.py`

**Interfaces:**

- Consumes: `FrontierReport`, `StrengthWeights`, `GameKey`, `GameTask`, `GameResult`, and immutable league paths.
- Produces: `RUNG_1_SEEDS`, `RUNG_2_SEEDS`, `RUNG_3_SEEDS`, `RungSpec`, `PanelSet`, `derive_panels(report: FrontierReport) -> PanelSet`, `rung_specs(panels: PanelSet) -> tuple[RungSpec, RungSpec, RungSpec]`, `missing_game_tasks(candidate: HybridOpponent, league: Mapping[str, Opponent], opponents: Sequence[str], seeds: Sequence[int], completed: Mapping[GameKey, GameResult]) -> tuple[GameTask, ...]`, `GameEvidence`, `MatchupEvidence`, `RungScore`, `RungEvidence`, `IneligibleEvidenceError`, `build_rung_evidence(trial_number: int, config: HybridConfig, spec: RungSpec, completed: Mapping[GameKey, GameResult], weights: StrengthWeights) -> RungEvidence`, `score_evidence(evidence: RungEvidence, weights: StrengthWeights) -> RungScore`, `minimum_primary_increment(spec: RungSpec, weights: StrengthWeights) -> float`, `objective_value(primary: float, dense_margin: float) -> float`, `evidence_path(root: Path, trial_number: int, rung: int) -> Path`, and `write_rung_evidence_atomic(root: Path, evidence: RungEvidence) -> tuple[Path, str]`.

- [ ] **Step 1: Write RED seed, panel, and incremental-task tests**

```python
def test_nested_seed_banks_are_fixed_and_protected() -> None:
    assert RUNG_1_SEEDS == tuple(range(860_000, 860_004))
    assert RUNG_2_SEEDS == tuple(range(860_000, 860_016))
    assert RUNG_3_SEEDS == tuple(range(860_000, 860_032))
    assert set(RUNG_1_SEEDS) < set(RUNG_2_SEEDS) < set(RUNG_3_SEEDS)
    for protected in (FRONTIER_SEEDS, SCREENING_SEEDS, DEVELOPMENT_SEEDS,
                      DETERMINISM_SEEDS, PROMOTION_SEEDS):
        assert set(RUNG_3_SEEDS).isdisjoint(protected)


def test_panels_are_exact_three_six_eleven_from_ranked_report(
    frontier_report: FrontierReport,
) -> None:
    panels = derive_panels(frontier_report)
    assert len(panels.rung_1) == 3
    assert panels.rung_1[0] == "economic_policy"
    assert len(panels.rung_2) == 6
    assert {"boatlee_v14_current", frontier_report.frontier_name} <= set(panels.rung_2)
    assert panels.rung_3 == tuple(row.name for row in frontier_report.rows)


def test_later_rung_runs_only_missing_cells() -> None:
    prior = completed_keys(panel=PANEL_1, seeds=RUNG_1_SEEDS)
    tasks = missing_game_tasks(CANDIDATE, LEAGUE, PANEL_2, RUNG_2_SEEDS, prior)
    assert len(tasks) == 168
    assert not prior & {task.key for task in tasks}
```

Run: `.venv/bin/python -m pytest tests/search/test_optuna_protocol.py -k 'seed or panel or missing' -q`

Expected: FAIL because `optuna_protocol` does not exist.

- [ ] **Step 2: Implement fixed rungs and deterministic panel derivation**

```python
@dataclass(frozen=True)
class RungSpec:
    rung: int
    resource_step: int
    opponents: tuple[str, ...]
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class PanelSet:
    rung_1: tuple[str, ...]
    rung_2: tuple[str, ...]
    rung_3: tuple[str, ...]


def derive_panels(report: FrontierReport) -> PanelSet:
    ranked = tuple(row.name for row in report.rows)
    weakest = tuple(reversed(ranked))
    rung_1 = _fill_distinct(("economic_policy", *weakest), 3)
    median = ranked[len(ranked) // 2]
    rung_2 = _fill_distinct(
        (*rung_1, "boatlee_v14_current", report.frontier_name, median, *weakest), 6
    )
    return PanelSet(rung_1, rung_2, ranked)
```

Require exactly 11 ranked unique opponents and required economic/Boatlee names;
reject any ambiguity instead of selecting a shorter panel.

- [ ] **Step 3: Write RED aggregation, failure, and epsilon-order tests**

```python
def test_rung_evidence_aggregates_counts_margin_runtime_and_failures() -> None:
    evidence = build_rung_evidence(TRIAL, RUNG, synthetic_game_rows())
    econ = evidence.matchups["economic_policy"]
    assert (econ.wins, econ.draws, econ.losses, econ.games) == (2, 1, 1, 4)
    assert econ.win_points == pytest.approx(0.625)
    assert evidence.failures == ()


@pytest.mark.parametrize("rung", (1, 2, 3))
def test_one_primary_increment_beats_maximum_margin_difference(rung: int) -> None:
    minimum_increment = minimum_primary_increment(SPECS[rung - 1], WEIGHTS)
    assert minimum_increment > 1e-6
    assert objective_value(PRIMARY + minimum_increment, -1.0) > objective_value(PRIMARY, 1.0)


def test_any_failed_game_makes_rung_ineligible() -> None:
    evidence = build_rung_evidence(TRIAL, RUNG, (*clean_rows(), failed_row()))
    assert evidence.failures == ("economic_policy/860000/seat0: boom",)
    with pytest.raises(IneligibleEvidenceError, match="boom"):
        score_evidence(evidence, WEIGHTS)
```

Run: `.venv/bin/python -m pytest tests/search/test_optuna_protocol.py -k 'evidence or primary or failed' -q`

Expected: FAIL because evidence/scoring functions are missing.

- [ ] **Step 4: Implement strict Pydantic evidence and scoring**

Use frozen, extra-forbid, strict Pydantic models:

```python
class MatchupEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)
    opponent: str
    wins: int = Field(ge=0)
    draws: int = Field(ge=0)
    losses: int = Field(ge=0)
    games: int = Field(gt=0)
    win_points: float = Field(ge=0.0, le=1.0)
    paired_normalized_margin: float = Field(ge=-1.0, le=1.0)
    runtime_seconds: float = Field(ge=0.0)


class GameEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)
    opponent: str
    seed: int
    seat: Literal[0, 1]
    ours: int | None
    theirs: int | None
    runtime_seconds: float = Field(ge=0.0)
    failure: str | None = None


class RungScore(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)
    primary: float = Field(ge=0.0, le=1.0)
    dense_margin: float = Field(ge=-1.0, le=1.0)
    objective: float = Field(ge=0.0, le=1.000001)


class IneligibleEvidenceError(ValueError):
    """A rung contains execution failures and cannot receive a score."""


class RungEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)
    schema_version: Literal[1] = 1
    trial_number: int = Field(ge=0)
    rung: Literal[1, 2, 3]
    resource_step: Literal[1, 4, 16]
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    opponents: tuple[str, ...]
    seeds: tuple[int, ...]
    games: tuple[GameEvidence, ...]
    matchups: dict[str, MatchupEvidence]
    primary: float | None
    dense_margin: float | None
    objective: float | None
    failures: tuple[str, ...]
```

Compute `objective = primary + 1e-6 * ((dense_margin + 1.0) / 2.0)`. Derive
every aggregate from raw game rows, validate exact expected keys/counts, and
write canonical sorted compact JSON to a same-directory temporary file followed
by file fsync, `os.replace`, and parent-directory fsync.

For each rung, construct a temporary `StrengthWeights` containing exactly that
panel's names and original integer weights. Define the conservative minimum
primary increment as
`0.70 * min_weight / sum_weights * 0.5 / (2 * len(spec.seeds))`; enumerate
single-game win/draw changes in tests and require each observed positive
primary delta to be at least that bound.

- [ ] **Step 5: Run protocol gates and commit Task 3**

```bash
.venv/bin/python -m pytest tests/search/test_optuna_protocol.py tests/search/test_fitness.py tests/search/test_promotion.py -q
.venv/bin/ruff check src/kaggriculture/search/optuna_protocol.py tests/search/test_optuna_protocol.py
.venv/bin/ty check src/kaggriculture/search/optuna_protocol.py
git add src/kaggriculture/search/optuna_protocol.py tests/search/test_optuna_protocol.py
git commit -m "feat: define progressive hybrid trial evidence"
```

---

### Task 4: Durable study identity and SQLite resume

**Files:**

- Create: `src/kaggriculture/search/optuna_state.py`
- Create: `tests/search/test_optuna_state.py`

**Interfaces:**

- Consumes: verified frontier/snapshot facts, `SPACE_SHA256`, panels, weights, warm-start hashes, Optuna 4.9.0.
- Produces: `StudyPaths.from_root(root: Path) -> StudyPaths`, `SamplerIdentity`, `PrunerIdentity`, `SeedBankIdentity`, `PanelIdentity`, `StudyIdentity`, `StudyIdentityError`, `StudyEvidenceError`, `build_identity(frontier, snapshot, report, panels, weights, warm_starts, legacy_state_sha256) -> StudyIdentity`, `open_study(paths: StudyPaths, identity: StudyIdentity) -> optuna.Study`, `terminal_counts(study: optuna.Study) -> dict[str, int]`, `reconcile_running_trials(study: optuna.Study) -> tuple[int, ...]`, `validate_study_evidence(study, paths, identity) -> None`, and `close_and_hash_storage(study, paths) -> str`.

- [ ] **Step 1: Write RED identity and path-ownership tests**

```python
def test_identity_binds_every_semantic_input() -> None:
    identity = build_identity(FRONTIER, SNAPSHOT, REPORT, PANELS, WEIGHTS, WARM_STARTS)
    assert identity.study_name == "hybrid-optuna-v2"
    assert identity.optuna_version == "4.9.0"
    assert identity.maximum_trials == 512
    assert identity.space_sha256 == SPACE_SHA256
    assert identity.sampler.model_dump() == {
        "seed": 20260827, "multivariate": True,
        "group": True, "n_startup_trials": 16,
    }
    assert identity.pruner.model_dump() == {
        "min_resource": 1, "reduction_factor": 4,
        "min_early_stopping_rate": 0,
    }


def test_resume_rejects_semantic_drift_before_sqlite_mutation(tmp_path: Path) -> None:
    paths = StudyPaths.from_root(tmp_path / "study")
    first = open_study(paths, IDENTITY)
    original = paths.sqlite.read_bytes()
    with pytest.raises(StudyIdentityError, match="space_sha256"):
        open_study(paths, IDENTITY.model_copy(update={"space_sha256": "f" * 64}))
    assert paths.sqlite.read_bytes() == original


def test_paths_reject_snapshot_or_legacy_state_aliases(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="protected"):
        validate_study_paths(PATHS_INSIDE_SNAPSHOT, SNAPSHOT, LEGACY_STATE)
```

Run: `.venv/bin/python -m pytest tests/search/test_optuna_state.py -k 'identity or path' -q`

Expected: FAIL because `optuna_state` does not exist.

- [ ] **Step 2: Implement canonical identity and preflight-first study opening**

```python
class StudyIdentityError(ValueError):
    """Stored and requested semantic identities differ."""


class StudyEvidenceError(ValueError):
    """SQLite trial state and canonical evidence disagree."""


@dataclass(frozen=True)
class StudyPaths:
    root: Path
    sqlite: Path
    identity: Path
    evidence: Path
    diagnostic: Path
    finalists: Path

    @classmethod
    def from_root(cls, root: Path) -> Self:
        return cls(
            root=root,
            sqlite=root / "study.sqlite3",
            identity=root / "identity.json",
            evidence=root / "evidence",
            diagnostic=root / "diagnostic.json",
            finalists=root / "finalists.json",
        )


class SamplerIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    seed: Literal[20260827] = 20260827
    multivariate: Literal[True] = True
    group: Literal[True] = True
    n_startup_trials: Literal[16] = 16


class PrunerIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    min_resource: Literal[1] = 1
    reduction_factor: Literal[4] = 4
    min_early_stopping_rate: Literal[0] = 0


class SeedBankIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    rung_1: tuple[int, ...]
    rung_2: tuple[int, ...]
    rung_3: tuple[int, ...]
    protected_promotion_sha256: str


class PanelIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    rung_1: tuple[str, str, str]
    rung_2: tuple[str, str, str, str, str, str]
    rung_3: tuple[str, ...]


class StudyIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    schema_version: Literal[1] = 1
    study_name: Literal["hybrid-optuna-v2"] = "hybrid-optuna-v2"
    optuna_version: Literal["4.9.0"] = "4.9.0"
    engine: str
    manifest_sha256: str
    frontier_report_sha256: str
    source_sha256: tuple[tuple[str, str], ...]
    league_snapshots: tuple[tuple[str, str, str, str], ...]
    space_sha256: str
    sampler: SamplerIdentity
    pruner: PrunerIdentity
    seed_banks: SeedBankIdentity
    panels: PanelIdentity
    strength_weights: tuple[tuple[str, int], ...]
    objective_schema: Literal["win-primary-margin-epsilon-v1"]
    warm_start_sha256: tuple[str, ...]
    legacy_state_sha256: str
    maximum_trials: Literal[512] = 512

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        return hashlib.sha256(payload).hexdigest()
```

Canonicalize with `model_dump(mode="json")`, sorted compact JSON, and SHA-256.
Validate all external files and path alias/inode/symlink boundaries before
creating the run root or SQLite. On first open, write/fsync `identity.json`,
create `RDBStorage(f"sqlite:///{paths.sqlite}")`, create the study with the exact sampler,
pruner, and `direction="maximize"`, then set `study_identity_sha256`. On resume,
load and compare identity JSON and study user attribute before any mutation.

- [ ] **Step 3: Write RED terminal-state and stale-running tests**

```python
def test_resume_marks_only_stale_running_trial_failed(tmp_path: Path) -> None:
    study = prepared_study(tmp_path, states=(COMPLETE, PRUNED, RUNNING))
    before = frozen_trial_payload(study.trials[:2])
    reconciled = reconcile_running_trials(study)
    assert reconciled == (2,)
    assert study.trials[2].state is TrialState.FAIL
    assert study.trials[2].system_attrs["interruption_reason"] == "process_restarted"
    assert frozen_trial_payload(study.trials[:2]) == before


def test_completed_evidence_must_match_sqlite_user_attributes(tmp_path: Path) -> None:
    study, paths = completed_study(tmp_path)
    tamper_json(paths.evidence / "trial-0-rung-1.json", {"objective": 0.99})
    with pytest.raises(StudyEvidenceError, match="trial 0 rung 1"):
        validate_study_evidence(study, paths, IDENTITY)
```

Run: `.venv/bin/python -m pytest tests/search/test_optuna_state.py -k 'running or evidence' -q`

Expected: FAIL because reconciliation/validation is absent.

- [ ] **Step 4: Implement reconciliation, evidence cross-check, and terminal hash**

Use public Optuna APIs only: enumerate `study.get_trials(deepcopy=False,
states=(TrialState.RUNNING,))`, set the interruption reason through storage
before `study.tell(number, state=TrialState.FAIL)`, and never convert an
existing COMPLETE/PRUNED/FAIL trial. Cross-check each trial's config digest,
rung, resource step, objective, and evidence digest user attributes against the
canonical evidence file. Reject orphan completed evidence and missing evidence.

For terminal hashing, require no RUNNING trials, execute SQLite
`PRAGMA wal_checkpoint(TRUNCATE)` through a short direct connection, dispose
Optuna's SQLAlchemy engine, fsync the database, and return its SHA-256. Tests
must prove no `-wal` or `-shm` file remains and a reopened read-only study has
the same terminal states.

- [ ] **Step 5: Run state gates and commit Task 4**

```bash
.venv/bin/python -m pytest tests/search/test_optuna_state.py tests/search/test_evolution.py -q
.venv/bin/ruff check src/kaggriculture/search/optuna_state.py tests/search/test_optuna_state.py
.venv/bin/ty check src/kaggriculture/search/optuna_state.py
git add src/kaggriculture/search/optuna_state.py tests/search/test_optuna_state.py
git commit -m "feat: persist the certified Optuna study"
```

---

### Task 5: Sequential coordinator, CLI, pruning, and stop gates

**Files:**

- Create: `src/kaggriculture/search/optuna_search.py`
- Create: `src/kaggriculture/search/scripts/hybrid_optuna.py`
- Create: `tests/search/test_optuna_search.py`
- Create: `tests/search/test_hybrid_optuna_cli.py`

**Interfaces:**

- Consumes: Tasks 1–4, verified frontier/report/snapshot APIs, and callback sequence supplied by Task 6.
- Produces: `SearchInputs`, `SearchRunConfig`, `SearchSummary`, `TrialEvaluationError`, internal `SearchInterrupted`, `OptunaCoordinator.objective(trial: optuna.Trial) -> float`, `run_search(inputs: SearchInputs, config: SearchRunConfig, callbacks: Sequence[Callable], arena_factory: Callable[[int], PersistentArena] = PersistentArena) -> SearchSummary`, `pilot_gate(study: optuna.Study, paths: StudyPaths) -> PilotVerdict`, `economic_stop_gate(study: optuna.Study, paths: StudyPaths) -> bool`, and CLI `python -m kaggriculture.search.scripts.hybrid_optuna`.

- [ ] **Step 1: Write RED objective/pruning/failure tests**

```python
def test_objective_writes_each_rung_before_reporting_and_pruning(tmp_path: Path) -> None:
    trial = RecordingTrial(prune_after_step=4)
    coordinator = coordinator_fixture(tmp_path)
    with coordinator.arena:
        with pytest.raises(optuna.TrialPruned):
            coordinator.objective(trial)
    assert trial.events == [
        "write:rung1", "report:1", "prune:1",
        "write:rung2", "report:4", "prune:4",
    ]
    assert len(coordinator.arena.requested_keys) == 24 + 168


def test_game_failure_fails_trial_instead_of_returning_bad_score(tmp_path: Path) -> None:
    coordinator = coordinator_fixture(tmp_path, failure_at=GameKey("economic_policy", 860000, 0))
    with coordinator.arena:
        with pytest.raises(TrialEvaluationError, match="economic_policy/860000/seat0"):
            coordinator.objective(RecordingTrial())
    assert load_evidence(tmp_path, trial=0, rung=1).failures


def test_completed_rung_three_returns_exact_stored_objective(tmp_path: Path) -> None:
    coordinator = coordinator_fixture(tmp_path)
    with coordinator.arena:
        value = coordinator.objective(RecordingTrial())
    assert value == load_evidence(tmp_path, trial=0, rung=3).objective
    assert len(coordinator.arena.requested_keys) == 24 + 168 + 512
```

Run: `.venv/bin/python -m pytest tests/search/test_optuna_search.py -k 'objective or failure' -q`

Expected: FAIL because the coordinator does not exist.

- [ ] **Step 2: Implement the objective and trial user attributes**

```python
class TrialEvaluationError(RuntimeError):
    """A complete trial that cannot be eligible because execution failed."""


class SearchInterrupted(RuntimeError):
    """SIGINT or SIGTERM requested a bounded coordinator shutdown."""


@dataclass(frozen=True)
class SearchInputs:
    study: optuna.Study
    paths: StudyPaths
    identity: StudyIdentity
    league: Mapping[str, Opponent]
    weights: StrengthWeights
    rungs: tuple[RungSpec, RungSpec, RungSpec]
    warm_starts: tuple[HybridConfig, ...]


class SearchRunConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    workers: int = Field(default=32, ge=1, le=32)
    stop_after: int = Field(default=512, ge=1, le=512)


class SearchSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    started_trials: int = Field(ge=0)
    terminal_trials: int = Field(ge=0, le=512)
    complete_trials: int = Field(ge=0)
    pruned_trials: int = Field(ge=0)
    failed_trials: int = Field(ge=0)
    stopped_reason: Literal[
        "stop_after", "no_economic_points_at_128", "complete", "interrupted"
    ]
    best_trial: int | None


@dataclass(frozen=True)
class OptunaCoordinator:
    paths: StudyPaths
    league: Mapping[str, Opponent]
    weights: StrengthWeights
    rungs: tuple[RungSpec, RungSpec, RungSpec]
    arena: PersistentArena


def objective(self, trial: optuna.Trial) -> float:
    config = suggest_config(trial)
    candidate = HybridOpponent(to_runtime(config))
    completed: dict[GameKey, GameResult] = {}
    for spec in self.rungs:
        tasks = missing_game_tasks(
            candidate, self.league, spec.opponents, spec.seeds, completed
        )
        rows = self.arena.run(tasks)
        completed.update((row.key, row) for row in rows)
        evidence = build_rung_evidence(trial.number, config, spec, completed, self.weights)
        path, digest = write_rung_evidence_atomic(self.paths, evidence)
        _set_trial_summary(trial, evidence, path, digest)
        if evidence.failures:
            raise TrialEvaluationError("; ".join(evidence.failures))
        if evidence.objective is None:
            raise RuntimeError("clean rung evidence has no objective")
        trial.report(evidence.objective, step=spec.resource_step)
        if trial.should_prune():
            raise optuna.TrialPruned(f"pruned after rung {spec.rung}")
    if evidence.objective is None:
        raise RuntimeError("completed rung-three evidence has no objective")
    return evidence.objective
```

`_set_trial_summary` records only canonical scalar/count/path/digest attributes;
raw game rows remain in evidence files. Pass
`catch=(TrialEvaluationError,)` to `study.optimize` so one failed trial does not
abort later trials. Any storage/evidence/pool exception outside that exact type
must stop the study.

- [ ] **Step 3: Write RED warm enqueue, pilot, 128-trial, and resume-budget tests**

```python
def test_fresh_study_enqueues_eight_warm_starts_once(tmp_path: Path) -> None:
    study = fresh_study(tmp_path)
    enqueue_warm_starts(study, WARM_CONFIGS)
    enqueue_warm_starts(study, WARM_CONFIGS)
    assert [trial.params for trial in study.trials] == [
        parameters_for_config(config) for config in WARM_CONFIGS
    ]


def test_stop_after_counts_all_terminal_states_on_resume(tmp_path: Path) -> None:
    study = study_with_states(tmp_path, complete=10, pruned=18, failed=4)
    summary = run_search(
        STUDY_INPUTS, SearchRunConfig(stop_after=32), callbacks=()
    )
    assert summary.started_trials == 0
    assert summary.terminal_trials == 32


def test_no_economic_points_at_128_writes_diagnostic_not_finalists(tmp_path: Path) -> None:
    study = study_with_zero_economic_points(tmp_path, terminal=128)
    summary = run_search(
        STUDY_INPUTS, SearchRunConfig(stop_after=512), callbacks=()
    )
    assert summary.stopped_reason == "no_economic_points_at_128"
    assert PATHS.diagnostic.exists()
    assert not PATHS.finalists.exists()


def test_interrupt_closes_arena_and_leaves_no_false_finalist(tmp_path: Path) -> None:
    arena = InterruptingArena(after_calls=1)
    summary = run_search(
        STUDY_INPUTS,
        SearchRunConfig(stop_after=32),
        callbacks=(),
        arena_factory=lambda _workers: arena,
    )
    assert summary.stopped_reason == "interrupted"
    assert arena.closed
    assert not PATHS.finalists.exists()
    reconcile_running_trials(STUDY)
    validate_study_evidence(STUDY, PATHS, IDENTITY)
```

Run: `.venv/bin/python -m pytest tests/search/test_optuna_search.py -k 'enqueue or stop_after or economic' -q`

Expected: FAIL because run control and gates are absent.

- [ ] **Step 4: Implement terminal-budget control and pilot verdict**

Count COMPLETE, PRUNED, and FAIL as terminal. Enqueue warm starts only when
there are no trials; otherwise validate that trials 0–7 exactly match the warm
parameter/config digests. Call `study.optimize` with `n_jobs=1` and
`n_trials=max(0, stop_after - terminal_count)`.

Define:

```python
class PilotVerdict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    passed: bool
    reasons: tuple[str, ...]
    terminal_trials: int
    best_trial: int | None
    default_trial: int
    best_same_rung_delta: float | None
    failures: int


def pilot_gate(study: optuna.Study, paths: StudyPaths) -> PilotVerdict:
    # Require exactly 32 terminal trials, zero infrastructure/evidence failures,
    # a non-default clean trial above trial 0 at the same rung, complete resume
    # validation, and no finalists. Return every failed reason independently.
```

At terminal count 128, scan validated evidence, not W&B or trial display names,
for any clean `economic_policy.win_points > 0.0`. If absent, atomically write a
canonical diagnostic containing the best margin trial and stop the study.

Install SIGINT/SIGTERM handlers only after all read-only preflight succeeds and
restore the previous handlers in `finally`. The handlers request study stop and
raise one internal `SearchInterrupted` exception; the outer run boundary closes
the arena and W&B session, leaves the active trial RUNNING for the normal next-
startup reconciliation rule, writes no finalist, and returns an interrupted
summary. Pool/storage exceptions other than `TrialEvaluationError` still
propagate after context cleanup rather than being mislabeled as an interrupt.

- [ ] **Step 5: Write RED CLI preflight tests**

```python
def test_cli_preflights_every_identity_input_before_pool_or_wandb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []
    monkeypatch.setattr(cli, "verify_frontier", lambda *args: calls.append("verify") or FRONTIER)
    monkeypatch.setattr(cli, "PersistentArena", forbidden("arena"))
    args = valid_args(tmp_path, frontier_report=tmp_path / "missing.json")
    with pytest.raises(SystemExit, match="frontier report"):
        cli.run(args)
    assert calls == ["verify"]


def test_cli_worker_range_is_one_through_32() -> None:
    assert parse_args([*BASE_ARGS, "--workers", "32"]).workers == 32
    with pytest.raises(SystemExit):
        parse_args([*BASE_ARGS, "--workers", "33"])
```

Run: `.venv/bin/python -m pytest tests/search/test_hybrid_optuna_cli.py -q`

Expected: FAIL because the CLI does not exist.

- [ ] **Step 6: Implement production CLI with side-effect ordering**

Arguments:

```text
--manifest             default existing frontier manifest
--artifact-root        default /data/kaggriculture/search/public-frontier
--frontier-report      required
--legacy-state         default run/hybrid/search-state.json
--root                 default run/hybrid/optuna-v2
--workers              default 32, choices 1..32
--stop-after           default 512, choices 1..512
--validate-only        read and verify identity/SQLite/evidence without workers or W&B
--wandb/--no-wandb     default enabled
--wandb-entity         default will-rice
--wandb-project        default kaggriculture-2026
```

Preflight order is: parse/validate arguments; verify legacy digest; verify
frontier and report; validate all seed disjointness; validate output/path
ownership; create or reopen immutable snapshot/identity/study; reconcile stale
RUNNING trials; validate existing evidence; then open W&B and arena. Print the
resolved study digest, terminal counts, workers, stop boundary, and CPU-only
status before starting games. Do not call `os.nice`.

- [ ] **Step 7: Run coordinator/CLI gates and commit Task 5**

```bash
.venv/bin/python -m pytest tests/search/test_optuna_search.py tests/search/test_hybrid_optuna_cli.py tests/search/test_optuna_state.py tests/search/test_optuna_protocol.py -q
.venv/bin/ruff check src/kaggriculture/search/optuna_search.py src/kaggriculture/search/scripts/hybrid_optuna.py tests/search/test_optuna_search.py tests/search/test_hybrid_optuna_cli.py
.venv/bin/ty check src/kaggriculture/search/optuna_search.py src/kaggriculture/search/scripts/hybrid_optuna.py
git add src/kaggriculture/search/optuna_search.py src/kaggriculture/search/scripts/hybrid_optuna.py tests/search/test_optuna_search.py tests/search/test_hybrid_optuna_cli.py
git commit -m "feat: coordinate progressive Optuna hybrid trials"
```

---

### Task 6: One persistent failure-isolated W&B run

**Files:**

- Create: `src/kaggriculture/search/optuna_wandb.py`
- Create: `tests/search/test_optuna_wandb.py`
- Modify: `src/kaggriculture/search/scripts/hybrid_optuna.py`
- Modify: `tests/search/test_hybrid_optuna_cli.py`

**Interfaces:**

- Consumes: `StudyIdentity`, Optuna `Study`/`FrozenTrial`, trial user attributes written by Task 5, `optuna_integration.WeightsAndBiasesCallback`, and `wandb`.
- Produces: `WandbSettings`, `WandbApi` protocol, `WandbDependencies`, lazy `load_wandb_dependencies()`, `WandbSession.open(settings: WandbSettings, identity: StudyIdentity, revision: str, dependencies: WandbDependencies | None = None) -> WandbSession`, `WandbSession.callbacks -> tuple[Callable[[optuna.Study, FrozenTrial], None], ...]`, `trial_metrics(trial: FrozenTrial) -> dict[str, float]`, and bounded `close() -> None`.

- [ ] **Step 1: Write RED one-run, unique-step, and failure-isolation tests**

```python
def test_session_initializes_once_and_combines_rich_metrics_with_official_callback() -> None:
    api = FakeWandb()
    session = WandbSession.open(
        SETTINGS,
        IDENTITY,
        "abc123",
        dependencies=WandbDependencies(api, fake_official_factory(api)),
    )
    for trial in (TRIAL_0, TRIAL_1):
        session.callbacks[0](STUDY, trial)
    session.close()
    assert len(api.init_calls) == 1
    assert api.init_calls[0]["id"] == f"hybrid-optuna-{IDENTITY.digest[:20]}"
    assert [row.step for row in api.history] == [0, 1]
    assert api.history[1].metrics["best/economic/win_points"] == 0.125
    assert api.finish_calls == 1


@pytest.mark.parametrize("failure", ("import", "init", "log", "finish"))
def test_wandb_failure_never_changes_study_or_raises(failure: str) -> None:
    before = frozen_trial_payload(STUDY.trials)
    session = session_with_failure(failure)
    for trial in STUDY.trials:
        session.callback(STUDY, trial)
    session.close()
    assert frozen_trial_payload(STUDY.trials) == before
    assert session.disabled_reason is not None
```

Run: `.venv/bin/python -m pytest tests/search/test_optuna_wandb.py -q`

Expected: FAIL because `optuna_wandb` does not exist.

- [ ] **Step 2: Implement one preinitialized official callback and rich metric merge**

```python
class WandbSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    enabled: bool = True
    entity: str = "will-rice"
    project: str = "kaggriculture-2026"


class WandbRun(Protocol):
    def log(
        self, metrics: Mapping[str, float], *, step: int, commit: bool
    ) -> None: ...

    def finish(self, exit_code: int = 0, quiet: bool = True) -> None: ...


class WandbApi(Protocol):
    def init(self, **kwargs: object) -> WandbRun: ...


@dataclass(frozen=True)
class WandbDependencies:
    api: WandbApi
    callback_factory: Callable[..., object]


def load_wandb_dependencies() -> WandbDependencies:
    import wandb
    from optuna_integration import WeightsAndBiasesCallback

    return WandbDependencies(wandb, WeightsAndBiasesCallback)


@dataclass
class WandbSession:
    run: WandbRun | None
    official: Callable[[optuna.Study, FrozenTrial], None] | None
    disabled_reason: str | None = None

    @classmethod
    def disabled(cls, reason: str) -> Self:
        return cls(run=None, official=None, disabled_reason=reason)

    @classmethod
    def open(
        cls,
        settings: WandbSettings,
        identity: StudyIdentity,
        revision: str,
        *,
        dependencies: WandbDependencies | None = None,
    ) -> Self:
        if not settings.enabled:
            return cls.disabled("disabled_by_config")
        try:
            resolved = dependencies or load_wandb_dependencies()
            run = resolved.api.init(
                entity=settings.entity,
                project=settings.project,
                id=f"hybrid-optuna-{identity.digest[:20]}",
                name="hybrid-optuna-v2",
                resume="allow",
                config=wandb_config(identity, revision),
            )
            official = resolved.callback_factory(as_multirun=False)
            return cls(run=run, official=official)
        except Exception as error:
            return cls.disabled(f"{type(error).__name__}: {error}")

    def callback(self, study: optuna.Study, trial: FrozenTrial) -> None:
        if self.disabled_reason is not None:
            return
        assert self.run is not None and self.official is not None
        try:
            self.run.log(trial_metrics(trial), step=trial.number, commit=False)
            self.official(study, trial)  # commits the same trial-number row
        except Exception as error:
            self.disabled_reason = f"{type(error).__name__}: {error}"

    @property
    def callbacks(self) -> tuple[Callable[[optuna.Study, FrozenTrial], None], ...]:
        return () if self.disabled_reason is not None else (self.callback,)

    def close(self) -> None:
        if self.run is None:
            return
        try:
            self.run.finish(quiet=True)
        except Exception as error:
            self.disabled_reason = f"{type(error).__name__}: {error}"
        finally:
            self.run = None
```

Inspect the installed 4.9.0 callback in the RED cycle and adapt only the seam
needed to guarantee a preinitialized single run and one committed row. Do not
import W&B or `optuna_integration` at module import; `--no-wandb` must work when
either package is unavailable. Do not call private Optuna storage APIs or create
per-trial W&B runs. `trial_metrics`
must validate finite scalar values and include state, rung, games, failures,
throughput, objective components, ETA, current best, and every opponent summary
present in trial user attributes.

- [ ] **Step 3: Test resume and CLI isolation**

Create a fake study with trials 0–7 already terminal and trial 8 new. Reopen a
session with the same deterministic run ID and assert only trial 8 is passed to
the callback by the new `study.optimize` invocation. Inject W&B initialization
failure in the CLI and assert arena/search still completes a trial and SQLite
records it.

Run:

```bash
.venv/bin/python -m pytest tests/search/test_optuna_wandb.py tests/search/test_hybrid_optuna_cli.py -q
```

Expected: all pass with no network calls.

- [ ] **Step 4: Static checks and commit Task 6**

```bash
.venv/bin/ruff check src/kaggriculture/search/optuna_wandb.py src/kaggriculture/search/scripts/hybrid_optuna.py tests/search/test_optuna_wandb.py tests/search/test_hybrid_optuna_cli.py
.venv/bin/ty check src/kaggriculture/search/optuna_wandb.py src/kaggriculture/search/scripts/hybrid_optuna.py
git add src/kaggriculture/search/optuna_wandb.py src/kaggriculture/search/scripts/hybrid_optuna.py tests/search/test_optuna_wandb.py tests/search/test_hybrid_optuna_cli.py
git commit -m "feat: track Optuna hybrid trials in one W&B run"
```

---

### Task 7: Certified Optuna finalists and promotion adapter

**Files:**

- Create: `src/kaggriculture/search/optuna_finalists.py`
- Create: `tests/search/test_optuna_finalists.py`
- Modify: `src/kaggriculture/search/scripts/hybrid_holdout.py:1-220`
- Modify: `tests/search/test_hybrid_holdout.py`

**Interfaces:**

- Consumes: closed SQLite digest, validated `StudyIdentity`, rung-3 evidence, `HybridConfig`, legacy `FinalistArtifact`, verified frontier/report, and Task 9 claim boundary.
- Produces: `OptunaFinalist`, `OptunaFinalistArtifact`, `write_optuna_finalists(study: optuna.Study, identity: StudyIdentity, paths: StudyPaths, count: int = 4) -> OptunaFinalistArtifact`, `OptunaFinalistArtifact.load(path: Path) -> OptunaFinalistArtifact`, and normalized `PromotionInput` returned by `load_promotion_input(path: Path, frontier: VerifiedFrontier, report: FrontierReport) -> PromotionInput` for either legacy or Optuna schema.

- [ ] **Step 1: Write RED finalist writer and tamper tests**

```python
def test_writer_selects_only_clean_rung_three_trials_in_canonical_order(tmp_path: Path) -> None:
    artifact = write_optuna_finalists(COMPLETED_STUDY, IDENTITY, PATHS, count=4)
    assert [row.trial_number for row in artifact.finalists] == [17, 9, 31, 4]
    assert all(row.rung == 3 and not row.failures for row in artifact.finalists)
    assert artifact.promotion_seeds_used is False


@pytest.mark.parametrize("tamper", (
    "sqlite_digest", "identity_digest", "config", "objective", "rank",
    "rung_evidence_digest", "source_snapshot", "trial_state", "seed_count",
))
def test_loader_rejects_every_certification_tamper(tmp_path: Path, tamper: str) -> None:
    path = write_fixture_artifact(tmp_path)
    mutate_certified_input(path, tamper)
    with pytest.raises(ValueError, match=tamper.replace("_", " ")):
        OptunaFinalistArtifact.load(path)
```

Run: `.venv/bin/python -m pytest tests/search/test_optuna_finalists.py -q`

Expected: FAIL because the finalist schema does not exist.

- [ ] **Step 2: Implement schema and authoritative recomputation**

```python
class OptunaFinalist(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)
    trial_number: int = Field(ge=0)
    rung: Literal[3]
    parameters: dict[str, int | float | str]
    config: HybridConfig
    config_sha256: str
    primary: float
    dense_margin: float
    objective: float
    evidence_path: str
    evidence_sha256: str
    failures: tuple[str, ...] = ()


class OptunaFinalistArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    schema: Literal["optuna-finalists-v1"]
    study_identity: StudyIdentity
    study_identity_sha256: str
    sqlite_path: str
    sqlite_sha256: str
    trial_counts: dict[str, int]
    promotion_seeds_used: Literal[False] = False
    finalists: tuple[OptunaFinalist, ...]
    integrity_sha256: str
```

Before writing: validate no RUNNING trials; require 512 terminal trials; run the
128 economic gate; validate all study/evidence correspondence; require at least
one clean rung-3 trial; recompute each config from parameters; recompute every
score and ordering; close/checkpoint/hash SQLite; then write canonical artifact
bytes atomically. `load` repeats all checks from bytes and the referenced closed
database/evidence, including path containment and snapshot hashes.

- [ ] **Step 3: Write RED legacy/Optuna promotion normalization tests**

```python
def test_holdout_accepts_legacy_and_optuna_through_same_normalized_input(
    legacy_path: Path, optuna_path: Path
) -> None:
    legacy = load_promotion_input(legacy_path, FRONTIER, REPORT)
    modern = load_promotion_input(optuna_path, FRONTIER, REPORT)
    assert legacy.league == modern.league
    assert legacy.weights == modern.weights
    assert all(isinstance(config, HybridConfig) for config in modern.configs)


def test_optuna_preflight_tamper_fails_before_claim_or_holdout(
    monkeypatch: pytest.MonkeyPatch, tampered_optuna_path: Path
) -> None:
    monkeypatch.setattr(hybrid_holdout, "_execute_single_use", forbidden("claim"))
    with pytest.raises(SystemExit, match="invalid holdout provenance"):
        hybrid_holdout.run(args_for(tampered_optuna_path))
```

Run: `.venv/bin/python -m pytest tests/search/test_hybrid_holdout.py -k optuna -q`

Expected: FAIL because holdout imports only legacy `FinalistArtifact`.

- [ ] **Step 4: Refactor holdout behind `PromotionInput` without changing claims**

```python
@dataclass(frozen=True)
class PromotionInput:
    configs: tuple[HybridConfig, ...]
    finalist_rows: tuple[dict[str, object], ...]
    league: Mapping[str, str]
    snapshot_root: Path
    weights: StrengthWeights
    terminal_digest: str
    run_identity: dict[str, object]
    protected_paths: tuple[Path, ...]
```

Move only input-specific binding into `load_promotion_input`. Keep
`_execute_single_use`, canonical claim namespace, determinism seeds, promotion
seeds, bootstrap gates, and result schema unchanged. Legacy input must produce
byte-identical run identity and claims to current tests. Optuna input uses the
study terminal digest and exact v2 facts in its claim identity.

- [ ] **Step 5: Run finalist/promotion regression and commit Task 7**

```bash
.venv/bin/python -m pytest tests/search/test_optuna_finalists.py tests/search/test_hybrid_holdout.py tests/search/test_promotion.py tests/search/test_evolution.py -q
.venv/bin/ruff check src/kaggriculture/search/optuna_finalists.py src/kaggriculture/search/scripts/hybrid_holdout.py tests/search/test_optuna_finalists.py tests/search/test_hybrid_holdout.py
.venv/bin/ty check src/kaggriculture/search/optuna_finalists.py src/kaggriculture/search/scripts/hybrid_holdout.py
git add src/kaggriculture/search/optuna_finalists.py src/kaggriculture/search/scripts/hybrid_holdout.py tests/search/test_optuna_finalists.py tests/search/test_hybrid_holdout.py
git commit -m "feat: certify Optuna hybrid finalists"
```

---

### Task 8: Full verification and the 32-trial real pilot

**Files:**

- Modify only if a gate exposes a defect: Task 1–7 files and their focused tests.
- Create runtime artifact, not tracked: `run/hybrid/optuna-v2/pilot-verdict.json`
- Create ignored report: `.superpowers/sdd/2026-08-27-hybrid-optuna-search-v2/pilot-report.md`

**Interfaces:**

- Consumes: the complete search-v2 CLI and all accepted Task 1–7 interfaces.
- Produces: fresh release gates, one durable 32-terminal-trial SQLite study, one resumable W&B run, a canonical pilot verdict, and an explicit controller checkpoint before any continuation to 512.

- [ ] **Step 1: Run all focused search and hybrid gates**

```bash
.venv/bin/python -m pytest \
  tests/search/test_optuna_space.py \
  tests/search/test_optuna_seeds.py \
  tests/search/test_arena.py \
  tests/search/test_arena_pool.py \
  tests/search/test_optuna_protocol.py \
  tests/search/test_optuna_state.py \
  tests/search/test_optuna_search.py \
  tests/search/test_optuna_wandb.py \
  tests/search/test_optuna_finalists.py \
  tests/search/test_hybrid_optuna_cli.py \
  tests/search/test_hybrid_holdout.py -q
.venv/bin/python -m pytest tests/search tests/hybrid tests/test_package.py tests/test_submission.py -q
```

Expected: all selected tests pass; existing intentional deselections/skips are
reported exactly and investigated if their count changes.

- [ ] **Step 2: Run static, dependency, and unchanged-boundary gates**

```bash
.venv/bin/ruff format --check \
  src/kaggriculture/search/optuna_*.py \
  src/kaggriculture/search/arena.py \
  src/kaggriculture/search/arena_pool.py \
  src/kaggriculture/search/scripts/hybrid_optuna.py \
  src/kaggriculture/search/scripts/hybrid_holdout.py \
  tests/search/test_optuna_*.py \
  tests/search/test_arena*.py \
  tests/search/test_hybrid_optuna_cli.py \
  tests/search/test_hybrid_holdout.py
.venv/bin/ruff check \
  src/kaggriculture/search/optuna_*.py \
  src/kaggriculture/search/arena.py \
  src/kaggriculture/search/arena_pool.py \
  src/kaggriculture/search/scripts/hybrid_optuna.py \
  src/kaggriculture/search/scripts/hybrid_holdout.py \
  tests/search/test_optuna_*.py \
  tests/search/test_arena*.py \
  tests/search/test_hybrid_optuna_cli.py \
  tests/search/test_hybrid_holdout.py
.venv/bin/ty check
git diff --check
git diff d99a838 -- main.py src/kaggriculture/hybrid/winner.py
```

Expected: format/Ruff/ty/diff pass; the final diff command is empty.

- [ ] **Step 3: Record pre-pilot immutable inputs and Toad isolation**

Record in the report:

```bash
sha256sum run/hybrid/search-state.json run/hybrid/frontier.json
git rev-parse HEAD
tmux list-panes -t toad -F '#{pane_dead} #{pane_pid}'
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
```

Expected: legacy state hash is exactly
`c74ae9ff99fc6bd1d39c93f2edda8ac3a9ee1894d1422c2e02e9c6d02aab5c5b`;
Toad is live; no search process is on a GPU; no promotion/finalist/claim artifact
exists under the new run root.

- [ ] **Step 4: Launch the real pilot in persistent tmux**

```bash
tmux new-session -d -s hybrid-optuna-v2 \
  ".venv/bin/python -m kaggriculture.search.scripts.hybrid_optuna \
    --frontier-report run/hybrid/frontier.json \
    --legacy-state run/hybrid/search-state.json \
    --root run/hybrid/optuna-v2 \
    --workers 32 \
    --stop-after 32"
```

Poll only durable SQLite/evidence terminal counts and process health. Report
each newly terminal trial or failure boundary without touching promotion.

- [ ] **Step 5: Validate the 32-trial pilot and stop for review**

After tmux exits cleanly, run a read-only validator command exposed by the CLI:

```bash
.venv/bin/python -m kaggriculture.search.scripts.hybrid_optuna \
  --frontier-report run/hybrid/frontier.json \
  --legacy-state run/hybrid/search-state.json \
  --root run/hybrid/optuna-v2 \
  --workers 32 \
  --stop-after 32 \
  --validate-only
```

Require: exactly 32 terminal trials; eight pinned warm starts; zero engine,
provenance, storage, or evidence failures; at least one non-default same-rung
improvement over default; one W&B run with 32 unique trial steps and full
opponent metrics; clean split/reopen validation; Toad healthy; no GPU search
process; old state hash unchanged; no finalists, promotion output, candidate,
or claim tree.

Write the exact verdict and counts to the ignored pilot report. Present the
evidence to the user and **stop**. Do not resume to 512 until the user approves
the pilot.

- [ ] **Step 6: Commit any final gate repairs, otherwise leave code unchanged**

If Task 8 exposed and fixed a code defect, rerun the affected RED/GREEN and all
Task 8 gates, then commit only the repair:

```bash
git add \
  pyproject.toml uv.lock \
  src/kaggriculture/search/arena.py \
  src/kaggriculture/search/arena_pool.py \
  src/kaggriculture/search/optuna_space.py \
  src/kaggriculture/search/optuna_seeds.py \
  src/kaggriculture/search/optuna_protocol.py \
  src/kaggriculture/search/optuna_state.py \
  src/kaggriculture/search/optuna_search.py \
  src/kaggriculture/search/optuna_wandb.py \
  src/kaggriculture/search/optuna_finalists.py \
  src/kaggriculture/search/scripts/hybrid_optuna.py \
  src/kaggriculture/search/scripts/hybrid_holdout.py \
  tests/search/test_arena.py \
  tests/search/test_arena_pool.py \
  tests/search/test_optuna_space.py \
  tests/search/test_optuna_seeds.py \
  tests/search/test_optuna_protocol.py \
  tests/search/test_optuna_state.py \
  tests/search/test_optuna_search.py \
  tests/search/test_optuna_wandb.py \
  tests/search/test_optuna_finalists.py \
  tests/search/test_hybrid_optuna_cli.py \
  tests/search/test_hybrid_holdout.py
git commit -m "fix: make the Optuna pilot production safe"
```

If no repair was necessary, make no empty commit. Runtime SQLite, evidence,
W&B metadata, and the ignored report are never staged.
