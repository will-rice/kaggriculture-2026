# Market-Residual RL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, train, validate, and package a recurrent market-only residual
that improves a frozen Kaito v48 controller without changing its logistics or
investment actions.

**Architecture:** A dependency-light runtime wraps a fresh verified Kaito v48
callable, builds the canonical feature bundle once, opens decisions only at
typed market events, and either preserves Kaito exactly or replaces only
`SELL`/`BUY_PRODUCT` orders. A Torch GRU is initialized from immutable
single-intervention counterfactual rows, then fine-tuned with paired event-level
V-trace against frozen Kaito controls on pinned rolling-frontier generations.
Promotion uses unseen temporal and single-use seed banks and packages a NumPy
export only after exact Torch/export parity.

**Tech Stack:** Python 3.11, PyTorch 2.13, NumPy, Pydantic 2.12, the pinned
Kaggriculture tensor simulator and Kaggle engine 1.32.7, pytest, Ruff, ty, and
Weights & Biases.

**Spec:**
`docs/superpowers/specs/2026-08-28-market-residual-rl-design.md`

## Global Constraints

- The engine identity is exactly `1.32.7`; source, engine, feature schema,
  event schema, action schema, model schema, seeds, and frontier hashes are
  part of every run identity.
- The frozen baseline is the verified `kaito_v48` artifact declared by
  `src/kaggriculture/search/frontier_manifest.json`; production never silently
  substitutes another Kaito or Boatlee version.
- The learned boundary may create, cancel, or replace only `SELL` and
  `BUY_PRODUCT`. `HIRE`, `BUY_LAND`, `BUY_SEED`, `BUY_ANIMAL`, farmer actions,
  and hand actions are copied from Kaito.
- Categorical and integer market inputs are one-hot encoded by one canonical
  dependency-light function consumed by collection, Torch training, and NumPy
  inference.
- A malformed/nonfinite output, invalid state, illegal quantity, overspend,
  shed-capacity violation, or order-count violation returns the untouched Kaito
  action and records a typed fallback reason.
- Training and evaluation support exactly one explicitly selected GPU. CPU
  reference workers are bounded and recorded; no command auto-selects multiple
  GPUs.
- The consumed route-promotion seeds `840000..840127` are rejected by every
  market-residual artifact boundary.
- Seed banks are disjoint and fixed:
  - counterfactual train: `860000..860511`;
  - counterfactual temporal selection: `861000..861127`;
  - online train: `870000..871023`;
  - small online gate: `872000..872063`;
  - temporal frontier selection: `880000..880127`;
  - export determinism: `895000..895007`;
  - single-use promotion: `890000..890127`.
- The small online gate requires zero failures, replacement activation between
  1% and 50%, and positive mean paired win-point delta over offline
  initialization on all 256 cells (64 seeds × two seats × two fixed
  opponents). Failure stops the run before scale.
- Temporal selection requires direct win points against Kaito v48 at least
  `0.55`, a paired-seed 95% bootstrap lower bound above `0.50`, a positive
  weighted-cluster delta lower bound, no cluster mean regression below
  `-0.03`, and zero failures on 128 seeds in both seats.
- Promotion is single-use and atomic. Packaging may proceed on PASS, but an
  actual Kaggle submission requires explicit confirmation of the exact archive
  SHA-256.
- Runtime code imports neither Torch, Lightning, Optuna, nor W&B. Training code
  stays under `kaggriculture.learn`; offline orchestration stays under scripts.
- Every task follows RED → GREEN, scoped static checks, and a narrow commit.

## File Structure

Runtime files are small and dependency-light:

- `src/kaggriculture/market_residual/schema.py`: stable feature/action/event
  schemas and seed-bank constants.
- `src/kaggriculture/market_residual/features.py`: the one canonical market
  feature vector and one-hot encoding.
- `src/kaggriculture/market_residual/events.py`: pure event-state transition.
- `src/kaggriculture/market_residual/actions.py`: parsing, replacement,
  validation, and fail-closed merge.
- `src/kaggriculture/market_residual/numpy_policy.py`: exported GRU inference.
- `src/kaggriculture/market_residual/policy.py`: frozen-baseline wrapper and
  per-turn agent state.

Training and evidence files are separated by responsibility:

- `src/kaggriculture/learn/market_residual/model.py`: Torch GRU and heads.
- `src/kaggriculture/learn/market_residual/alternatives.py`: bounded legal
  counterfactual action generator.
- `src/kaggriculture/learn/market_residual/counterfactual.py`: exact simulator
  snapshots, transcript restoration, and branch execution.
- `src/kaggriculture/learn/market_residual/artifacts.py`: strict Pydantic
  identities, rows, shards, locks, and atomic publication.
- `src/kaggriculture/learn/market_residual/offline.py`: sequence collation and
  pairwise/Q/auxiliary losses.
- `src/kaggriculture/learn/market_residual/online.py`: paired event
  trajectories, V-trace targets, and update.
- `src/kaggriculture/learn/market_residual/export.py`: Torch-to-NumPy export and
  exact parity validation.
- `src/kaggriculture/search/market_frontier.py`: immutable daily frontier
  generations and behavior clustering.
- `src/kaggriculture/search/market_selection.py`: paired temporal selection
  and bootstrap gates.
- `src/kaggriculture/scripts/market_counterfactuals.py`: bounded dataset CLI.
- `src/kaggriculture/scripts/market_pretrain.py`: offline learner CLI.
- `src/kaggriculture/scripts/market_train.py`: one-GPU online learner CLI.
- `src/kaggriculture/scripts/market_evaluate.py`: temporal selection CLI.
- `src/kaggriculture/scripts/freeze_market_residual.py`: verified runtime
  artifacts and alternate entrypoint.

Tests mirror those boundaries under `tests/market_residual/`; no test imports a
training module to validate the submitted runtime.

---

### Task 1: Canonical Market Schema and Seed Boundaries

**Files:**

- Create: `src/kaggriculture/market_residual/__init__.py`
- Create: `src/kaggriculture/market_residual/schema.py`
- Create: `src/kaggriculture/market_residual/features.py`
- Test: `tests/market_residual/test_schema.py`
- Test: `tests/market_residual/test_features.py`

**Interfaces:**

- Consumes: `features.EncodedObservation`, `action_codec.MARKET_SLOTS`,
  `action_codec.QUANTITIES`, and canonical scalar/shop names.
- Produces: `ALLOWED_SLOTS`, `MarketFeatureSchema`, `MarketFeatureVector`,
  `market_feature_vector(encoded, kaito_buckets, previous)`, and
  `validate_seed_bank(name, seeds)`.

- [ ] **Step 1: Write the failing schema and one-hot tests**

```python
def test_only_commodity_slots_are_learnable() -> None:
    assert {MARKET_SLOTS[index][0] for index in ALLOWED_SLOTS} == {
        "SELL",
        "BUY_PRODUCT",
    }


def test_integer_and_categorical_features_are_one_hot(encoded) -> None:
    row = market_feature_vector(encoded, (0,) * len(ALLOWED_SLOTS), None)
    assert sum(row.values[row.schema.day_slice]) == 1.0
    assert sum(row.values[row.schema.hour_slice]) == 1.0
    assert sum(row.values[row.schema.kaito_bucket_slices[0]]) == 1.0
    assert row.schema.sha256 == MarketFeatureSchema.current().sha256


def test_consumed_promotion_seeds_are_rejected() -> None:
    with pytest.raises(ValueError, match="consumed promotion seed"):
        validate_seed_bank("counterfactual_train", (840000,))
```

- [ ] **Step 2: Run the tests and record the missing-module RED**

Run:
`python -m pytest tests/market_residual/test_schema.py tests/market_residual/test_features.py -q`

Expected: collection fails because `kaggriculture.market_residual` does not
exist.

- [ ] **Step 3: Implement immutable schemas and one canonical encoder**

```python
ALLOWED_SLOTS = tuple(
    index
    for index, (verb, _item) in enumerate(MARKET_SLOTS)
    if verb in {"SELL", "BUY_PRODUCT"}
)


@dataclass(frozen=True)
class MarketFeatureVector:
    schema: MarketFeatureSchema
    values: tuple[float, ...]


def _one_hot(value: int, width: int, label: str) -> tuple[float, ...]:
    if not 0 <= value < width:
        raise ValueError(f"{label}={value} outside [0, {width})")
    return tuple(float(index == value) for index in range(width))
```

Build the vector in a fixed order from exact decoded day/hour, open-shop bits,
commodity quantities, live prices/inventories, own reserves/capacity/cash,
opponent public supply, Kaito buckets, and previous-event deltas. Compute the
schema hash from canonical JSON names and widths, never from Python object
repr.

- [ ] **Step 4: Run schema, existing feature, encoding, and hybrid tests**

Run:
`python -m pytest tests/market_residual/test_schema.py tests/market_residual/test_features.py tests/test_features.py tests/learn/test_encoding.py tests/hybrid -q`

Expected: PASS, with only existing marker deselections.

- [ ] **Step 5: Run static checks and commit**

```bash
ruff format --check src/kaggriculture/market_residual tests/market_residual
ruff check src/kaggriculture/market_residual tests/market_residual
ty check src tests/market_residual
git diff --check
git add src/kaggriculture/market_residual tests/market_residual
git commit -m "feat: define market residual features"
```

### Task 2: Market Events and Fail-Closed Action Merge

**Files:**

- Create: `src/kaggriculture/market_residual/events.py`
- Create: `src/kaggriculture/market_residual/actions.py`
- Test: `tests/market_residual/test_events.py`
- Test: `tests/market_residual/test_actions.py`

**Interfaces:**

- Consumes: `MarketFeatureVector`, `EncodedObservation`, Kaito action mappings,
  and canonical legality masks.
- Produces: `EventConfig`, `EventMemory`, `MarketEvent`, `ResidualMode`,
  `ResidualDecision`, `MergeResult`, `FallbackReason`, `detect_event`,
  `kaito_market_buckets`, and `merge_residual_action`.

- [ ] **Step 1: Write event transition and forbidden-mutation tests**

```python
def test_heartbeat_opens_once_and_deduplicates_identical_state(encoded) -> None:
    first = detect_event(encoded, kaito_action, EventMemory.initial(), config)
    heartbeat = detect_event(
        encoded, kaito_action, first.memory.advance(24), config
    )
    duplicate = detect_event(encoded, kaito_action, heartbeat.memory, config)
    assert heartbeat.event is not None
    assert "heartbeat" in heartbeat.event.triggers
    assert duplicate.event is None


@pytest.mark.parametrize("verb", ["HIRE", "BUY_LAND", "BUY_SEED", "BUY_ANIMAL"])
def test_merge_never_changes_frozen_orders(verb, encoded) -> None:
    baseline = action_with_frozen_order(verb)
    merged = merge_residual_action(encoded, baseline, replacement_decision())
    assert frozen_orders(merged.action) == frozen_orders(baseline)


def test_nonfinite_logits_fail_closed(encoded) -> None:
    result = merge_residual_action(encoded, baseline_action(), nonfinite_decision())
    assert result.action == baseline_action()
    assert result.fallback is FallbackReason.NONFINITE
```

- [ ] **Step 2: Run the focused tests and capture RED**

Run:
`python -m pytest tests/market_residual/test_events.py tests/market_residual/test_actions.py -q`

Expected: missing event/action interfaces.

- [ ] **Step 3: Implement the pure event state machine**

```python
@dataclass(frozen=True)
class EventConfig:
    price_relative: float = 0.05
    inventory_relative: float = 0.10
    opponent_supply_units: int = 2
    heartbeat_turns: int = 24
    liquidation_day: int = 26


@dataclass(frozen=True)
class EventTransition:
    event: MarketEvent | None
    memory: EventMemory
```

Detect Kaito commodity proposals, newly legal/affordable slots, relative price
and inventory crossings, shop changes, opponent-supply deltas, liquidation,
and heartbeat. Hash the event-visible state and suppress a duplicate hash until
the observation changes.

- [ ] **Step 4: Implement exact parsing and fail-closed merge**

```python
class ResidualMode(StrEnum):
    USE_KAITO = "use_kaito"
    REPLACE = "replace"


@dataclass(frozen=True)
class ResidualDecision:
    mode: ResidualMode
    buckets: tuple[int, ...]
    finite: bool = True
```

Parse Kaito commodity orders into canonical buckets without coercion. On
`REPLACE`, retain every frozen order in original relative order, insert legal
replacement commodity orders deterministically, enforce the ten-order cap,
then independently check masks, exact cash, exact held quantities, and shed
capacity. Any exception returns the original action and one typed reason.

- [ ] **Step 5: Fuzz safety and run codec/simulator market regressions**

Run:
`python -m pytest tests/market_residual/test_events.py tests/market_residual/test_actions.py tests/test_action_codec.py tests/hybrid/test_market.py tests/sim/test_rules_market.py tests/sim/test_market_coupling.py -q`

Expected: PASS, including property cases that feed random bucket vectors and
prove non-commodity equality.

- [ ] **Step 6: Run static checks and commit**

```bash
ruff format --check src/kaggriculture/market_residual tests/market_residual
ruff check src/kaggriculture/market_residual tests/market_residual
ty check src tests/market_residual
git diff --check
git add src/kaggriculture/market_residual tests/market_residual
git commit -m "feat: guard market residual actions"
```

### Task 3: Frozen Kaito Wrapper and Exact Runtime Parity

**Files:**

- Create: `src/kaggriculture/market_residual/baseline.py`
- Create: `src/kaggriculture/market_residual/policy.py`
- Test: `tests/market_residual/test_baseline.py`
- Test: `tests/market_residual/test_policy.py`
- Modify: `tests/test_package.py`

**Interfaces:**

- Consumes: a verified Kaito source path or packaged Kaito callable factory,
  canonical features/events/actions, and a residual inference protocol.
- Produces: `BaselineIdentity`, `load_verified_baseline`, `ResidualInference`,
  `MarketResidualAgent`, and `build_market_residual_agent`.

- [ ] **Step 1: Write source-hash, fresh-instance, and forced-Kaito parity tests**

```python
def test_baseline_hash_must_match(tmp_path: Path) -> None:
    source = copy_kaito_v48(tmp_path)
    source.write_text(source.read_text() + "\n# drift\n")
    with pytest.raises(BaselineIntegrityError, match="sha256"):
        load_verified_baseline(identity_for_original(source))


def test_fresh_instances_do_not_share_memory(kaito_identity) -> None:
    left = load_verified_baseline(kaito_identity)
    right = load_verified_baseline(kaito_identity)
    assert left is not right
    assert trace(left, corpus()) == trace(right, corpus())


def test_forced_use_kaito_is_byte_exact(kaito_identity, corpus) -> None:
    baseline = load_verified_baseline(kaito_identity)
    wrapped = build_market_residual_agent(kaito_identity, AlwaysUseKaito())
    assert canonical_trace(wrapped, corpus) == canonical_trace(baseline, corpus)
```

- [ ] **Step 2: Run the runtime tests and capture RED**

Run:
`python -m pytest tests/market_residual/test_baseline.py tests/market_residual/test_policy.py -q`

Expected: missing baseline and policy modules.

- [ ] **Step 3: Implement verified fresh-callable loading and state ownership**

```python
class ResidualInference(Protocol):
    def initial_state(self) -> tuple[float, ...]: ...
    def observe(
        self,
        features: MarketFeatureVector,
        state: tuple[float, ...],
        *,
        act: bool,
    ) -> tuple[ResidualDecision | None, tuple[float, ...]]: ...
```

Execute only the already verified source bytes, resolve the last callable, and
return a new namespace/callable for every episode. The wrapper calls Kaito
first, encodes once, advances event and recurrent state, and merges only when
`act=True`. Boundary exceptions are converted to a Kaito fallback; exceptions
inside the verified Kaito callable propagate as episode failures.

- [ ] **Step 4: Run both-seat reference parity and import audits**

Run:
`python -m pytest tests/market_residual/test_policy.py -m slow -q`

Expected: the forced-Kaito wrapper matches Kaito actions, final banks, and
DONE/DONE status on eight fixed seeds in both seats.

Run:
`python -m pytest tests/test_package.py tests/test_submission.py -q`

Expected: runtime imports exclude Torch, Lightning, Optuna, and W&B.

- [ ] **Step 5: Commit the parity boundary**

```bash
git add src/kaggriculture/market_residual tests/market_residual tests/test_package.py
git commit -m "feat: wrap the frozen market baseline"
```

### Task 4: Torch GRU and NumPy Export Parity

**Files:**

- Create: `src/kaggriculture/learn/market_residual/__init__.py`
- Create: `src/kaggriculture/learn/market_residual/model.py`
- Create: `src/kaggriculture/learn/market_residual/export.py`
- Create: `src/kaggriculture/market_residual/numpy_policy.py`
- Test: `tests/market_residual/test_model.py`
- Test: `tests/market_residual/test_export.py`

**Interfaces:**

- Consumes: `MarketFeatureSchema`, allowed-slot masks, event sequences.
- Produces: `ModelConfig`, `MarketResidualNet`, `PolicyHeads`,
  `export_numpy_policy`, `NumpyResidualPolicy`, and `ExportIdentity`.

- [ ] **Step 1: Write shape, masking, recurrence, parameter, and export tests**

```python
def test_model_is_recurrent_and_below_one_million(schema) -> None:
    model = MarketResidualNet(ModelConfig(input_size=schema.width))
    first, state = model(sequence("price_up"), None)
    second, _ = model(sequence("price_flat"), state)
    reset, _ = model(sequence("price_flat"), None)
    assert not torch.equal(second.mode_logits, reset.mode_logits)
    assert sum(parameter.numel() for parameter in model.parameters()) < 1_000_000


def test_export_matches_torch_for_full_sequence(tmp_path, model, sequence) -> None:
    artifact = export_numpy_policy(model.eval(), tmp_path / "residual.json")
    numpy_policy = NumpyResidualPolicy.load(artifact)
    assert_policy_parity(model, numpy_policy, sequence, atol=1e-6)
```

- [ ] **Step 2: Run the model/export tests and capture RED**

Run:
`python -m pytest tests/market_residual/test_model.py tests/market_residual/test_export.py -q`

Expected: missing training model and runtime inference modules.

- [ ] **Step 3: Implement the small GRU and factorized heads**

```python
@dataclass(frozen=True)
class ModelConfig:
    input_size: int
    hidden_size: int = 192
    projection_size: int = 192
    auxiliary_size: int = 3 * len(PRODUCT_NAMES)


class MarketResidualNet(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(config.input_size, config.projection_size), nn.SiLU()
        )
        self.gru = nn.GRU(config.projection_size, config.hidden_size)
        self.mode = nn.Linear(config.hidden_size, 2)
        self.quantities = nn.Linear(
            config.hidden_size, len(ALLOWED_SLOTS) * len(QUANTITIES)
        )
        self.value = nn.Linear(config.hidden_size, 1)
        self.auxiliary = nn.Linear(config.hidden_size, config.auxiliary_size)
```

Mask illegal quantity logits with finite dtype minima before sampling. Compute
mode, joint action log probability, entropy, Q/value, and auxiliary outputs in
FP32 even under autocast.

- [ ] **Step 4: Implement canonical JSON export and NumPy GRU inference**

Write arrays as shape/dtype/base64 records plus model/schema/source hashes.
Implement the exact PyTorch GRU equations, SiLU, linear heads, masking, and
argmax in NumPy. Reject duplicate keys, nonfinite arrays, wrong shapes,
unknown schema versions, and source or feature-schema drift.

- [ ] **Step 5: Run parity, BF16, and import gates**

Run:
`python -m pytest tests/market_residual/test_model.py tests/market_residual/test_export.py tests/learn/test_toad_precision.py -q`

Expected: FP32 export parity `atol=1e-6`; CPU BF16 training forward keeps
sensitive diagnostics FP32; runtime import audit stays training-stack-free.

- [ ] **Step 6: Commit the model boundary**

```bash
git add src/kaggriculture/learn/market_residual src/kaggriculture/market_residual tests/market_residual
git commit -m "feat: add the recurrent market residual"
```

### Task 5: Deterministic Bounded Counterfactual Alternatives

**Files:**

- Create: `src/kaggriculture/learn/market_residual/alternatives.py`
- Test: `tests/market_residual/test_alternatives.py`

**Interfaces:**

- Consumes: `MarketEvent`, Kaito buckets, legal masks, prices, inventories,
  cash, reserves, and capacity.
- Produces: `Alternative`, `AlternativeSet`, and
  `generate_alternatives(event, encoded, kaito_buckets, config)`.

- [ ] **Step 1: Write completeness, legality, ordering, and cap tests**

```python
def test_alternatives_include_required_families(event) -> None:
    alternatives = generate_alternatives(event, encoded(), kaito(), config())
    assert alternatives.rows[0].family == "kaito"
    assert {row.family for row in alternatives.rows} >= {
        "kaito", "cancel", "scale_down", "scale_up", "single", "ranked_multi"
    }
    assert len(alternatives.rows) <= 48


def test_alternative_identity_is_order_stable(event) -> None:
    assert generate_alternatives(event, encoded(), kaito(), config()).sha256 == (
        generate_alternatives(event, encoded(), kaito(), config()).sha256
    )
```

- [ ] **Step 2: Run the focused test and capture RED**

Run: `python -m pytest tests/market_residual/test_alternatives.py -q`

Expected: missing alternatives module.

- [ ] **Step 3: Implement the fixed 48-row maximum generator**

Generate, validate, deduplicate, and lexicographically order: exact Kaito;
all-zero commodity action; 50% and 150% Kaito quantities; every legal
single-slot nonzero bucket around Kaito and price-ranked quantities; and at
most eight multi-slot combinations ranked by immediate quoted edge. Preserve
Kaito as row zero and hash canonical JSON of all buckets/families.

- [ ] **Step 4: Differential-check every generated action**

Run:
`python -m pytest tests/market_residual/test_alternatives.py tests/sim/test_rules_market.py tests/sim/test_differential.py -q`

Expected: all alternatives are legal in both reference and tensor market
phases and preserve frozen action components.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/learn/market_residual/alternatives.py tests/market_residual/test_alternatives.py
git commit -m "feat: enumerate market counterfactuals"
```

### Task 6: Exact Snapshot Branching and Immutable Counterfactual Artifacts

**Files:**

- Create: `src/kaggriculture/learn/market_residual/counterfactual.py`
- Create: `src/kaggriculture/learn/market_residual/artifacts.py`
- Test: `tests/market_residual/test_counterfactual.py`
- Test: `tests/market_residual/test_artifacts.py`

**Interfaces:**

- Consumes: verified baseline/opponent factories, `SimState`, `pack`, `unpack`,
  `encode_turn`, alternatives, and seed-bank constants.
- Produces: `AgentTranscript`, `CounterfactualSnapshot`,
  `CounterfactualOutcome`, `CounterfactualRow`, `CounterfactualShard`,
  `branch_event`, `write_shard_atomic`, and `load_shard_strict`.

- [ ] **Step 1: Write transcript restoration and branch-truth tests**

```python
def test_replayed_callable_matches_snapshot_transcript(snapshot) -> None:
    restored = snapshot.learner_transcript.restore()
    assert restored.replayed_actions == snapshot.learner_transcript.actions


def test_single_intervention_matches_reference_engine(fixed_market_tape) -> None:
    tensor = branch_event(fixed_market_tape.snapshot, fixed_market_tape.alternative)
    reference = run_reference_intervention(fixed_market_tape)
    assert tensor.terminal_banks == reference.terminal_banks
    assert tensor.win_points == reference.win_points
```

- [ ] **Step 2: Write artifact corruption and atomicity tests**

```python
@pytest.mark.parametrize("damage", ["duplicate", "missing_kaito", "seed", "hash", "status"])
def test_damaged_shard_rejects_without_partial_rows(tmp_path, damage) -> None:
    path = write_valid_shard(tmp_path)
    damage_shard(path, damage)
    with pytest.raises(CounterfactualIntegrityError):
        load_shard_strict(path, expected_identity())
```

- [ ] **Step 3: Run focused tests and capture RED**

Run:
`python -m pytest tests/market_residual/test_counterfactual.py tests/market_residual/test_artifacts.py -q`

Expected: missing snapshot/artifact interfaces.

- [ ] **Step 4: Implement cloneable simulator snapshots and transcript restore**

```python
@dataclass(frozen=True)
class AgentTranscript:
    source: BaselineIdentity
    observations: tuple[Mapping[str, object], ...]
    actions: tuple[str, ...]


def clone_state(state: SimState) -> SimState:
    return SimState(**{field.name: getattr(state, field.name).clone() for field in fields(state)})
```

Restore a fresh callable by replaying every stored observation and asserting
canonical action equality. Snapshot tensor state, both transcripts, event
memory, seed, seat, step, engine, and source hashes. Batch alternatives from
cloned state, inject only the selected market action, then use restored frozen
controllers for all subsequent turns.

- [ ] **Step 5: Implement strict content-addressed shard publication**

Each row stores opponent, seed, seat, event index/step/hash, alternative ID and
buckets, Kaito outcome, alternative outcome, paired win-point delta, normalized
margin delta, price path, failure, and all identities. Write canonical JSONL to
a same-directory temporary file, flush/fsync, rename, fsync directory, and
publish a SHA-256 manifest last. Reject partial games or non-DONE outcomes.

- [ ] **Step 6: Run full simulator differential and crash-boundary gates**

Run:
`python -m pytest tests/market_residual/test_counterfactual.py tests/market_residual/test_artifacts.py tests/sim/test_differential.py tests/sim/test_tape.py -q`

Expected: PASS, including injected failures before temp write, after fsync, and
before manifest rename.

- [ ] **Step 7: Commit**

```bash
git add src/kaggriculture/learn/market_residual tests/market_residual
git commit -m "feat: record exact market counterfactuals"
```

### Task 7: Counterfactual Collection CLI and Throughput Gate

**Files:**

- Create: `src/kaggriculture/scripts/market_counterfactuals.py`
- Test: `tests/market_residual/test_counterfactual_cli.py`

**Interfaces:**

- Consumes: verified frontier generation, exact alternatives/branch runner,
  immutable shards, worker bound.
- Produces: explicit `create`, `resume`, and `validate-only` CLI modes plus
  `collection-report.json`.

- [ ] **Step 1: Write mode, lock, identity, resume, and dry-run tests**

```python
def test_cli_requires_exactly_one_mode() -> None:
    assert parse_fails([])
    assert parse_fails(["--create", "--resume"])


def test_validate_only_runs_no_games(tmp_path, monkeypatch) -> None:
    root = complete_fixture(tmp_path)
    monkeypatch.setattr(counterfactual, "branch_event", forbidden)
    run_cli(["--validate-only", "--resume", "--root", str(root)])
```

- [ ] **Step 2: Run the CLI test and capture RED**

Run: `python -m pytest tests/market_residual/test_counterfactual_cli.py -q`

Expected: CLI module absent.

- [ ] **Step 3: Implement owned, resumable collection**

Expose exact arguments for manifest, artifact root, frontier generation,
baseline name, seed-bank name, workers `1..32`, maximum events per game,
maximum alternatives per event, and root. Acquire one nonblocking root lock
before snapshot or W&B. Reconcile only fully published shards; quarantine safe
temporary files and fail closed on canonical corruption.

- [ ] **Step 4: Run an eight-seed throughput profile**

Run:

```bash
python -m kaggriculture.scripts.market_counterfactuals \
  --create --root run/market-residual/counterfactual-smoke \
  --manifest src/kaggriculture/search/frontier_manifest.json \
  --artifact-root /data/kaggriculture/search/public-frontier \
  --frontier-generation run/hybrid/frontier.json \
  --baseline kaito_v48 --seed-start 860000 --seed-count 8 \
  --workers 8 --max-events 8 --max-alternatives 16
```

Expected: zero failures, exact 16 seat-cells, at least one event per cell, and a
report with separate engine, feature, policy, snapshot, branch, and fsync
timings. Do not extrapolate a full run until this measured profile exists.

- [ ] **Step 5: Validate the smoke root and commit**

Run the same command with `--resume --validate-only`; expected: no games and
unchanged artifact hashes.

```bash
git add src/kaggriculture/scripts/market_counterfactuals.py tests/market_residual/test_counterfactual_cli.py
git commit -m "feat: collect market counterfactual shards"
```

### Task 8: Offline Sequence Training and Known-Exploit Gate

**Files:**

- Create: `src/kaggriculture/learn/market_residual/offline.py`
- Create: `src/kaggriculture/scripts/market_pretrain.py`
- Test: `tests/market_residual/test_offline.py`
- Test: `tests/market_residual/test_pretrain_cli.py`

**Interfaces:**

- Consumes: strict counterfactual shards and `MarketResidualNet`.
- Produces: `OfflineBatch`, `OfflineLoss`, `collate_event_sequences`,
  `offline_loss`, `PretrainConfig`, atomic checkpoints, and
  `offline-report.json`.

- [ ] **Step 1: Write exact ranking/Q/auxiliary loss tests**

```python
def test_decisive_flip_outranks_kaito() -> None:
    report = offline_loss(model_outputs(), decisive_flip_batch())
    report.total.backward()
    assert report.ranking > 0
    assert finite_nonzero_gradients(model())


def test_weak_evidence_targets_use_kaito() -> None:
    batch = equal_outcome_batch()
    assert batch.mode_target.item() == ResidualMode.USE_KAITO_INDEX
```

- [ ] **Step 2: Run offline tests and capture RED**

Run:
`python -m pytest tests/market_residual/test_offline.py tests/market_residual/test_pretrain_cli.py -q`

Expected: offline interfaces absent.

- [ ] **Step 3: Implement sequence collation and FP32 loss islands**

Use packed event sequences grouped by game, never shuffled across recurrent
order. Loss is the sum of: pairwise logistic ranking; Huber Q regression to
paired win-point delta and `0.1 * normalized_margin_delta`; mode behavior loss
that selects Kaito within a `0.01` win-point indifference band; next-event
price/inventory/opponent-supply auxiliary Huber losses; and 4× sample weight
for loss↔win flips. Report every term, calibration, action activation, and
finite checks.

- [ ] **Step 4: Implement explicit-create/resume pretraining with W&B**

Bind data manifests, temporal split, model/config/source hashes, and one CUDA
device. Store optimizer, scaler, RNG, epoch, best-selection identity, and exact
W&B run ID atomically. Resume rejects config or data drift.

- [ ] **Step 5: Pass synthetic and real known-market-exploit gates**

Run:
`python -m pytest tests/market_residual/test_offline.py -m slow -q`

Expected: the model reaches at least 95% held-out ranking accuracy on a
synthetic price-crossing task and chooses a known profitable real market
alternative over Kaito on at least 75% of held-out cells while retaining
`USE_KAITO` on equal-outcome cells.

- [ ] **Step 6: Run bounded pretraining**

Collect at most the first 64 counterfactual-train seeds and all 128 temporal
selection seeds, then run:

```bash
CUDA_VISIBLE_DEVICES=0 python -m kaggriculture.scripts.market_pretrain \
  --create --root run/market-residual/offline-v1 \
  --train-manifest run/market-residual/counterfactual-train/manifest.json \
  --selection-manifest run/market-residual/counterfactual-select/manifest.json \
  --device cuda --epochs 120 --wandb
```

Expected: known-exploit gates pass, unseen paired win-point delta is nonnegative,
zero nonfinite values, and exported activation lies in `[0.01, 0.50]`.

The budget was 20 epochs and that was too small, measured rather than
guessed: the 20-epoch run selected epoch 20 -- its own ceiling -- and a
120-epoch rerun on identical data selected epoch 87 and moved every gate,
the strict exploit rate from 0.6445 (failing) to 0.8956 (passing) and the
expected paired win-point delta from +0.178 to +0.280. Treat a
`best_epoch` equal to the budget as an unfinished run, not a result.

- [ ] **Step 7: Commit**

```bash
git add src/kaggriculture/learn/market_residual/offline.py src/kaggriculture/scripts/market_pretrain.py tests/market_residual
git commit -m "feat: pretrain the market residual"
```

### Task 9: Paired Event-Level V-Trace Fine-Tuning

**Files:**

- Create: `src/kaggriculture/learn/market_residual/online.py`
- Create: `src/kaggriculture/scripts/market_train.py`
- Test: `tests/market_residual/test_online.py`
- Test: `tests/market_residual/test_train_cli.py`

**Interfaces:**

- Consumes: offline checkpoint, pinned frontier opponents, event runtime, and
  identical candidate/control cells.
- Produces: `EventTrajectory`, `PairedEpisode`, `OnlineBatch`,
  `vtrace_market_loss`, `OnlineConfig`, atomic online checkpoints, and
  `online-report.json`.

- [ ] **Step 1: Write paired-reward and event-only credit tests**

```python
def test_reward_is_candidate_minus_kaito_on_same_cell() -> None:
    pair = paired_episode(candidate_points=1.0, kaito_points=0.0, margin=0.2)
    assert terminal_reward(pair) == pytest.approx(1.02)


def test_non_event_turns_create_no_policy_rows() -> None:
    episode = collect_fixture_episode(event_steps=(4, 20, 44))
    assert episode.trajectory.steps == (4, 20, 44)
```

- [ ] **Step 2: Write V-trace/off-policy and resume-equivalence tests**

```python
def test_vtrace_clips_raw_importance_overflow() -> None:
    report = vtrace_market_loss(batch_with_log_rho(1000.0), model())
    assert torch.isfinite(report.total)
    assert report.max_rho == pytest.approx(1.0)


def test_split_resume_matches_uninterrupted(tmp_path) -> None:
    assert run_two_updates(tmp_path, split=True) == run_two_updates(tmp_path, split=False)
```

- [ ] **Step 3: Run online tests and capture RED**

Run:
`python -m pytest tests/market_residual/test_online.py tests/market_residual/test_train_cli.py -q`

Expected: online interfaces absent.

- [ ] **Step 4: Implement paired collection and recurrent V-trace**

Each candidate cell and frozen-Kaito control share opponent, seed, and seat.
Collect only event rows, behavior log probabilities, masks, recurrent entry
state, safety adjustment, and terminal outcome. Reward is candidate win points
minus Kaito win points; add `0.1 * normalized_margin_delta` only when win points
tie. Use clipped V-trace rho/c, value Huber, factorized entropy, auxiliary loss,
global finite synchronization, and gradient clipping.

- [ ] **Step 5: Implement one-GPU owned training**

Require `--device cuda --devices 1`; reject CPU fallback and multi-GPU values
for the production command. Bind frontier generations and opponent sampling
weights. Publish actor snapshots only at completed update boundaries. Log
paired objective metrics, cluster metrics, event/activation/fallback rates,
market profit, Q calibration, recurrent norms, and collection/learner timing.

- [ ] **Step 6: Pass the fixed small online gate before scale**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python -m kaggriculture.scripts.market_train \
  --create --root run/market-residual/online-small \
  --offline-checkpoint run/market-residual/offline-v1/best.ckpt \
  --frontier-generation run/market-residual/frontier/generation-0001.json \
  --seed-start 872000 --seed-count 64 --opponents kaito_v48,economic \
  --device cuda --devices 1 --max-updates 32 --wandb
```

Expected: 256 paired cells, zero failures, replacement activation `[0.01,
0.50]`, and positive mean paired win-point delta over the immutable offline
checkpoint. If any condition fails, stop and diagnose; do not launch the full
online train bank.

- [ ] **Step 7: Commit**

```bash
git add src/kaggriculture/learn/market_residual/online.py src/kaggriculture/scripts/market_train.py tests/market_residual
git commit -m "feat: fine tune market decisions online"
```

### Task 10: Immutable Rolling Frontier Generations

**Files:**

- Create: `src/kaggriculture/search/market_frontier.py`
- Create: `src/kaggriculture/scripts/market_frontier.py`
- Test: `tests/market_residual/test_frontier.py`
- Test: `tests/market_residual/test_frontier_cli.py`

**Interfaces:**

- Consumes: verified public frontier, exact episode traces, current engine,
  development/selection seed identities.
- Produces: `BehaviorDescriptor`, `BehaviorCluster`, `MarketFrontierGeneration`,
  `build_generation`, `load_generation_strict`, and an atomic ingestion CLI.

- [ ] **Step 1: Write immutability, clustering, and latest-pointer tests**

```python
def test_generation_is_content_addressed_and_immutable(tmp_path) -> None:
    generation = build_generation(fixture_inputs(), tmp_path)
    with pytest.raises(FileExistsError):
        build_generation(changed_inputs(), generation.path)


def test_active_identity_never_reads_latest(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "readlink", forbidden)
    assert load_generation_strict(exact_generation_path(tmp_path)).number == 1
```

- [ ] **Step 2: Run frontier tests and capture RED**

Run:
`python -m pytest tests/market_residual/test_frontier.py tests/market_residual/test_frontier_cli.py -q`

Expected: market frontier modules absent.

- [ ] **Step 3: Implement observable behavior descriptors and deterministic clusters**

Descriptor fields are production route, commodity volume/mix, first/median/last
sell steps, realized prices, hire/land schedule, and market aggression. Scale
fields by fixed engine constants; select deterministic farthest-first medoids
with lexical source hash tie-breaks. Store membership and strength weights, but
exclude names and leaderboard scores from clustering inputs.

- [ ] **Step 4: Implement atomic daily generations**

Verify every archive/source hash and engine, record exact trace hashes, cluster
descriptors, strength weights, train/selection seed hashes, and generation
parent. Publish `generation-<number>-<sha>.json`; optionally update `latest`
only after the immutable generation exists. Training accepts only the explicit
generation file.

- [ ] **Step 5: Run compatibility and commit**

Run:
`python -m pytest tests/market_residual/test_frontier.py tests/search/test_frontier.py tests/search/test_build_league.py -q`

```bash
git add src/kaggriculture/search/market_frontier.py src/kaggriculture/scripts/market_frontier.py tests/market_residual
git commit -m "feat: snapshot the market frontier"
```

### Task 11: Temporal Selection and Atomic Promotion Evidence

**Files:**

- Create: `src/kaggriculture/search/market_selection.py`
- Create: `src/kaggriculture/scripts/market_evaluate.py`
- Test: `tests/market_residual/test_selection.py`
- Test: `tests/market_residual/test_evaluate_cli.py`

**Interfaces:**

- Consumes: frozen Kaito, candidate export, newest unseen frontier generation,
  cluster weights, and protected selection seeds.
- Produces: `MarketSelectionReport`, `ClusterResult`, `SelectionVerdict`, paired
  bootstrap intervals, and atomic `selection-report.json`.

- [ ] **Step 1: Write independent gate and protected-seed tests**

```python
def test_every_gate_is_independent() -> None:
    assert verdict(direct=0.54).reasons == ("direct win points 0.540 < 0.550",)
    assert verdict(lower=0.50).reasons == ("direct lower bound 0.500 <= 0.500",)
    assert verdict(cluster_delta=-0.031).reasons == (
        "cluster c2 regressed by 0.031 > 0.030",
    )


def test_selection_rejects_train_and_consumed_seeds() -> None:
    with pytest.raises(ValueError, match="selection seed bank"):
        SelectionIdentity(seeds=(870000, 840000))
```

- [ ] **Step 2: Run selection tests and capture RED**

Run:
`python -m pytest tests/market_residual/test_selection.py tests/market_residual/test_evaluate_cli.py -q`

Expected: selection interfaces absent.

- [ ] **Step 3: Implement paired-seed bootstrap and complete verdict**

Resample whole seeds so both seats remain paired. Report candidate and frozen
Kaito results for every opponent/seed/seat, direct Kaito win points, weighted
paired delta, every cluster delta, failures, determinism, activation, fallback,
and archive/export hashes. Apply all thresholds from Global Constraints without
short-circuiting.

- [ ] **Step 4: Run temporal selection on generation 2**

After the small online gate passes, build a generation whose public sources and
behavior traces were not used by offline selection. Train on online seeds, then
run:

```bash
python -m kaggriculture.scripts.market_evaluate \
  --create --root run/market-residual/selection-v1 \
  --candidate run/market-residual/online-v1/export.json \
  --frontier-generation run/market-residual/frontier/generation-0002.json \
  --seed-start 880000 --seed-count 128 --workers 16
```

Expected: exact 128 seeds × two seats for each required opponent, zero missing
cells, and one atomic PASS/FAIL report. A FAIL does not consume promotion seeds.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/search/market_selection.py src/kaggriculture/scripts/market_evaluate.py tests/market_residual
git commit -m "feat: gate market residual candidates"
```

### Task 12: Freeze, Package, and Runtime Acceptance

**Files:**

- Create: `src/kaggriculture/scripts/freeze_market_residual.py`
- Create: `market_residual_main.py`
- Modify: `src/kaggriculture/scripts/package.py`
- Test: `tests/market_residual/test_freeze.py`
- Test: `tests/market_residual/test_package.py`

**Interfaces:**

- Consumes: PASS selection report, exact Kaito v48 source, NumPy export,
  runtime source tree.
- Produces: content-addressed frozen baseline/export artifacts, alternate
  entrypoint, `submission-market-residual.tar.gz`, and package manifest.

- [ ] **Step 1: Write guarded-freeze and package-import tests**

```python
def test_freeze_requires_passing_exact_selection(tmp_path) -> None:
    with pytest.raises(ValueError, match="selection verdict did not pass"):
        freeze_candidate(failed_selection(), export(), tmp_path)


def test_packaged_agent_imports_no_training_stack(archive) -> None:
    modules = run_archive_import_probe(archive)
    assert not {"torch", "lightning", "optuna", "wandb"} & modules
```

- [ ] **Step 2: Run freeze/package tests and capture RED**

Run:
`python -m pytest tests/market_residual/test_freeze.py tests/market_residual/test_package.py -q`

Expected: freeze script and alternate entrypoint absent.

- [ ] **Step 3: Implement atomic freeze and alternate package inputs**

Verify the selection report, exact export, Kaito source, frontier generation,
and code hashes before writing. Copy Kaito source and export into
`src/kaggriculture/market_residual/generated/` through a same-directory atomic
staging root; publish a manifest last. Extend `package.build` only through its
existing alternate `entrypoint` and `required` seams; do not change the default
Boatlee package mapping.

- [ ] **Step 4: Run export determinism and full-episode acceptance**

Run eight seeds `895000..895007` in both seats twice. Require byte-identical
action traces, DONE/DONE, zero fallbacks caused by malformed inference, exact
Torch/NumPy decisions, and overage comfortably below 60 seconds. Run existing
package/submission tests and a tar extraction/import smoke.

- [ ] **Step 5: Build the exact archive and commit code only**

```bash
python -m kaggriculture.scripts.freeze_market_residual \
  --selection run/market-residual/selection-v1/selection-report.json \
  --export run/market-residual/online-v1/export.json
python -m kaggriculture.scripts.package \
  --entrypoint market_residual_main.py \
  --output submission-market-residual.tar.gz
sha256sum submission-market-residual.tar.gz
```

Commit runtime/freeze/package code and tests; keep generated weights, Kaito
source, reports, and archive ignored and content-addressed.

```bash
git add src/kaggriculture/market_residual src/kaggriculture/scripts/freeze_market_residual.py src/kaggriculture/scripts/package.py market_residual_main.py tests/market_residual
git commit -m "feat: package the market residual agent"
```

### Task 13: Full One-GPU Run, Single-Use Promotion, and Submission Boundary

**Files:**

- Create at runtime: `run/market-residual/online-v1/`
- Create at runtime: `run/market-residual/selection-v1/`
- Create at runtime: `run/market-residual/promotion-v1/`
- Update ignored report: `.superpowers/sdd/2026-08-29-market-residual-rl/report.md`

**Interfaces:**

- Consumes: all committed production contracts and fixed seed banks.
- Produces: one immutable trained candidate, temporal PASS, single-use
  promotion verdict, exact archive SHA, and an explicit user decision point.

- [ ] **Step 1: Run the complete relevant verification matrix**

```bash
python -m pytest tests/market_residual tests/test_features.py tests/test_action_codec.py tests/hybrid tests/search/test_frontier.py tests/search/test_promotion.py tests/sim/test_differential.py tests/sim/test_rules_market.py -q
ruff format --check src/kaggriculture/market_residual src/kaggriculture/learn/market_residual tests/market_residual
ruff check src/kaggriculture/market_residual src/kaggriculture/learn/market_residual tests/market_residual
ty check src tests/market_residual
git diff --check
```

Expected: all selected tests and static checks pass; hardware-only CUDA tests
run on the selected GPU rather than skip.

- [ ] **Step 2: Generate the reviewed full counterfactual train bank**

Use seeds `860000..860511`, both seats, a pinned generation-1 cluster-stratified
opponent panel, and measured worker count from Task 7. Validate all manifests
and record exact duration/throughput before pretraining.

- [ ] **Step 3: Pretrain and run full online training on one GPU**

Start from the bounded offline checkpoint. Use online train seeds
`870000..871023`, one explicit GPU, frozen opponents first, and the reviewed
update/budget configuration. Poll durable checkpoints and W&B; stop on any
nonfinite value, identity mismatch, execution failure, or sustained activation
outside `[0.01, 0.50]`.

- [ ] **Step 4: Run temporal selection before touching promotion seeds**

Execute Task 11 against immutable frontier generation 2. Continue only if the
atomic report passes every direct, confidence, weighted, cluster, failure, and
determinism gate.

- [ ] **Step 5: Consume the single-use promotion bank exactly once**

Use seeds `890000..890127` in both seats against Kaito v48, the current rolling
champion, stable anchors, and the newest protected clusters. Write per-cell
claims before each matchup and an atomic terminal verdict. Never retry a
claimed cell under the same candidate identity.

- [ ] **Step 6: Build and independently verify the archive**

Record archive SHA-256, byte size, entrypoint identity, generated-artifact
hashes, import set, deterministic trace hash, runtime, selection verdict hash,
promotion verdict hash, and current Kaggle submission quota.

- [ ] **Step 7: Stop for explicit submission confirmation**

Present the exact archive hash and all promotion evidence. Do not call the
Kaggle submission API until the user explicitly confirms submission of that
exact hash.

## Plan Self-Review

- **Spec coverage:** Sections 1–7 map to Tasks 1–3; model/export to Task 4;
  counterfactual generation and artifacts to Tasks 5–7; offline and online
  learning to Tasks 8–9; rolling frontier to Task 10; metrics/selection to Task
  11; packaging to Task 12; throughput, resume, promotion, and submission to
  Task 13.
- **Seed separation:** all seven banks are disjoint, the consumed
  `840000..840127` range is rejected, and promotion is unreachable before
  temporal PASS.
- **Type consistency:** `MarketFeatureVector`, `ResidualDecision`,
  `MarketResidualNet`, counterfactual artifacts, online trajectories, frontier
  generations, selection reports, and export identities each have one owning
  module and the same names at every consumer.
- **No placeholders:** every task names exact files, interfaces, RED command,
  GREEN behavior, and commit boundary; no deferred interface is left unnamed.
