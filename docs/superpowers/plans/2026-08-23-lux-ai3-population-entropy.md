# Lux AI3 Population and Adaptive Entropy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic frozen-opponent population, explicit teacher-distillation batches, head-aware teacher metadata, and per-head target-entropy controllers to the recurrent Lightning trainer.

**Architecture:** Extend the synchronous collection round with four explicit batch kinds and deterministic quotas. Rank zero is not required yet; this plan remains one device, but snapshot manifests, digests, controller state, and callback boundaries are designed so the scale plan can add DDP without changing their interfaces.

**Tech Stack:** Python 3.11+, Pydantic 2.12+, PyTorch 2.13+, Lightning 2.6.5+, pytest.

**Spec:** `docs/superpowers/specs/2026-08-23-lux-ai3-native-port-design.md`

## Global Constraints

- Complete the Lightning foundation and recurrent-model plans first.
- Supported online kinds are exactly `selfplay`, `scripted`, `frozen_opponent`, and `teacher_distill`; behavior cloning remains separate.
- Sampling is deterministic from global game ID plus population seed.
- A teacher constrains only heads declared present and confirmed by checkpoint keys.
- Quantity entropy is counted only for operations that consume quantity.
- Entropy controllers update once per completed collection round using valid-action means and global environment steps.
- Snapshot and manifest writes are atomic; corrupt or incompatible members fail instead of being silently replaced.

## File Structure

- Create `src/kaggriculture/learn/toad/population.py`: snapshot metadata, manifest, storage, sampling, and teacher metadata.
- Modify `src/kaggriculture/learn/toad/config.py`: pool, teacher, mixture, and entropy-controller fields/validators.
- Modify `src/kaggriculture/learn/toad/data.py`: four batch kinds, deterministic quotas, and opponent assignment.
- Modify `src/kaggriculture/learn/toad/lightning.py`: head-aware teacher losses, per-head entropy terms, and controller state.
- Modify `src/kaggriculture/learn/toad/callbacks.py`: snapshot creation and manifest publication.
- Create `tests/learn/test_toad_population.py` and `test_toad_entropy.py`.
- Modify data, Lightning, callback, and checkpoint tests.

---

### Task 1: Implement content-addressed snapshot manifests

**Files:**

- Create: `src/kaggriculture/learn/toad/population.py`
- Create: `tests/learn/test_toad_population.py`
- Modify: `src/kaggriculture/learn/toad/config.py`

**Interfaces:**

- Consumes: policy state dictionaries and structural config fingerprints.
- Produces: `SnapshotEntry`, `SnapshotManifest`, `SnapshotStore.add`, `SnapshotStore.load`, and `SnapshotPool.sample`.

- [ ] **Step 1: Write failing atomicity, digest, and sampling tests**

```python
def test_snapshot_store_writes_a_verified_manifest_entry(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path, capacity=3, structure="abc")
    entry = store.add(state_dict(), environment_steps=100, round_id=4, run_id="run")
    assert entry.path.is_file()
    assert sha256_file(entry.path) == entry.sha256
    assert SnapshotManifest.load(tmp_path / "manifest.json").entries == (entry,)
    assert not list(tmp_path.glob("*.tmp"))


def test_population_sampling_is_deterministic_by_game_id(tmp_path: Path) -> None:
    pool = populated_pool(tmp_path, count=3, seed=17)
    assert pool.sample(game_id=91) == pool.sample(game_id=91)


def test_corrupt_snapshot_is_never_substituted(tmp_path: Path) -> None:
    pool, selected = pool_with_one_member(tmp_path)
    selected.path.write_bytes(b"corrupt")
    with pytest.raises(SnapshotIntegrityError, match=selected.sha256):
        pool.load(selected)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/learn/test_toad_population.py -v`

Expected: FAIL because the population module does not exist.

- [ ] **Step 3: Implement immutable metadata and digest validation**

```python
class SnapshotEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    path: Path
    sha256: str
    structure: str
    environment_steps: NonNegativeInt
    round_id: NonNegativeInt
    run_id: str
    created_at: datetime
    evaluation: dict[str, float] = Field(default_factory=dict)


class SnapshotManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    entries: tuple[SnapshotEntry, ...] = ()
```

Save a snapshot to a sibling `.tmp`, fsync/close it, compute its SHA-256, atomically rename it to `snapshot-{steps:012d}-{digest[:12]}.pt`, then atomically replace `manifest.json`. Validate structure and digest before `torch.load(weights_only=True)`.

- [ ] **Step 4: Implement bounded deterministic sampling**

```python
class SnapshotPool:
    def sample(self, game_id: int) -> SnapshotEntry:
        if not self.manifest.entries:
            raise EmptySnapshotPoolError("frozen opponent requested before pool population")
        rng = random.Random((self.seed << 64) ^ game_id)
        return rng.choice(self.manifest.entries)
```

Capacity uses oldest-first replacement after the new manifest is durable. A deleted entry is first removed from the durable manifest, then its file is unlinked; interruption can leave an unreferenced file but never a referenced missing file.

Run: `uv run pytest tests/learn/test_toad_population.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/learn/toad/population.py src/kaggriculture/learn/toad/config.py tests/learn/test_toad_population.py
git commit -m "feat: add verified Toad snapshot pool"
```

### Task 2: Generate deterministic four-kind collection rounds

**Files:**

- Modify: `src/kaggriculture/learn/toad/data.py`
- Modify: `src/kaggriculture/learn/toad/config.py`
- Modify: `tests/learn/test_toad_data.py`
- Modify: `tests/learn/test_toad_population.py`

**Interfaces:**

- Consumes: `SnapshotPool`, teacher checkpoint metadata, global game IDs, and mixture probabilities.
- Produces: extended `BatchKind`, `OpponentAssignment`, and `allocate_round(config, start_game_id, round_id)`.

- [ ] **Step 1: Write ratio and identity tests**

```python
def test_round_allocator_realizes_configured_quotas() -> None:
    config = mixture_config(selfplay=0.25, scripted=0.25, frozen=0.25, teacher=0.25, environments=8)
    assignments = allocate_round(config, start_game_id=100, round_id=2, pool=pool(), teacher=teacher())
    assert Counter(item.kind for item in assignments) == {
        BatchKind.SELFPLAY: 2,
        BatchKind.SCRIPTED: 2,
        BatchKind.FROZEN_OPPONENT: 2,
        BatchKind.TEACHER_DISTILL: 2,
    }
    assert [item.game_id for item in assignments] == list(range(100, 108))


def test_round_allocator_rejects_frozen_quota_without_a_pool() -> None:
    with pytest.raises(EmptySnapshotPoolError):
        allocate_round(frozen_config(), 0, 0, SnapshotPool.empty(), None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/learn/test_toad_data.py -k "quota or frozen" -v`

Expected: FAIL because only two batch kinds exist.

- [ ] **Step 3: Extend kinds and assignments**

```python
class BatchKind(StrEnum):
    SELFPLAY = "selfplay"
    SCRIPTED = "scripted"
    FROZEN_OPPONENT = "frozen_opponent"
    TEACHER_DISTILL = "teacher_distill"


@dataclass(frozen=True)
class OpponentAssignment:
    game_id: int
    seed: int
    kind: BatchKind
    opponent_id: str
    checkpoint: Path | None
    checkpoint_sha256: str | None
```

Compute integer quotas with largest-remainder allocation: floor each `probability * environments`, then assign remaining slots by descending fractional remainder with `BatchKind` value as the stable tie breaker. Shuffle the completed assignment list with `Random(base_seed ^ round_id)`.

```python
raw = {kind: probability * environments for kind, probability in probabilities.items()}
counts = {kind: math.floor(value) for kind, value in raw.items()}
remaining = environments - sum(counts.values())
order = sorted(raw, key=lambda kind: (-(raw[kind] - counts[kind]), kind.value))
for kind in order[:remaining]:
    counts[kind] += 1
assignments = build_assignments(counts, start_game_id, pool, teacher)
random.Random(base_seed ^ round_id).shuffle(assignments)
```

- [ ] **Step 4: Load frozen policies once per selected digest per round**

`ReferenceRoundSource` groups assignments by digest, validates each snapshot once, builds a frozen eval-mode policy, and passes it to existing rollout functions. `LearnerBatch.kind`, opponent ID, and digest must reflect the actual selected policy.

```python
policies: dict[str, StatefulPolicy] = {}
for assignment in assignments:
    if assignment.checkpoint_sha256 and assignment.checkpoint_sha256 not in policies:
        policies[assignment.checkpoint_sha256] = pool.load_policy(assignment)
    opponent = policies.get(assignment.checkpoint_sha256)
    trajectories.extend(self.play(assignment, opponent))
```

Run: `uv run pytest tests/learn/test_toad_data.py tests/learn/test_toad_population.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/learn/toad/data.py src/kaggriculture/learn/toad/config.py tests/learn/test_toad_data.py tests/learn/test_toad_population.py
git commit -m "feat: collect explicit Toad population batches"
```

### Task 3: Make teacher compatibility and loss semantics explicit

**Files:**

- Modify: `src/kaggriculture/learn/toad/population.py`
- Modify: `src/kaggriculture/learn/toad/config.py`
- Modify: `src/kaggriculture/learn/toad/lightning.py`
- Modify: `tests/learn/test_toad_population.py`
- Modify: `tests/learn/test_toad_lightning.py`

**Interfaces:**

- Consumes: teacher checkpoint keys and teacher-declared head metadata.
- Produces: `TeacherSpec`, `LoadedTeacher`, `load_teacher`, and head-aware policy/baseline losses.

- [ ] **Step 1: Write failing compatibility tests**

```python
def test_teacher_cannot_claim_a_missing_quantity_head(tmp_path: Path) -> None:
    checkpoint = legacy_checkpoint_without_quantity(tmp_path)
    spec = TeacherSpec(checkpoint=checkpoint, operation=True, quantity=True, market=True, value=True)
    with pytest.raises(TeacherCompatibilityError, match="quantity_head"):
        load_teacher(spec, control_model_config())


def test_teacher_loss_ignores_undeclared_quantity_head() -> None:
    report = compute_loss(student(), teacher_batch(), config(), teacher_without_quantity())
    assert report.terms["teacher/quantity_kl"] == 0
    assert report.terms["teacher/operation_kl"] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/learn/test_toad_population.py tests/learn/test_toad_lightning.py -k "teacher" -v`

Expected: FAIL because `TeacherSpec` is undefined.

- [ ] **Step 3: Implement teacher metadata and key verification**

```python
class TeacherSpec(BaseModel):  # in toad/config.py; population.py imports it
    model_config = ConfigDict(frozen=True, extra="forbid")
    checkpoint: Path
    sha256: str | None = None
    operation: bool = True
    quantity: bool = False
    market: bool = True
    value: bool = False


@dataclass(frozen=True)
class LoadedTeacher:
    policy: StatefulPolicy
    spec: TeacherSpec
```

Replace foundation `PopulationConfig.teacher_checkpoint` with `teacher: TeacherSpec | None`; provide a config migration that turns the old path-only field into `TeacherSpec(checkpoint=path)` before Pydantic validation. Update the cross-field validator to require `population.teacher` for teacher losses or `teacher_distill` probability.

Load with `weights_only=True`, compare the declared digest when present, verify every declared head key exists, freeze all parameters, and call `eval()`.

```python
def load_teacher(spec: TeacherSpec, model_config: ModelConfig) -> LoadedTeacher:
    if spec.sha256 is not None and sha256_file(spec.checkpoint) != spec.sha256:
        raise TeacherCompatibilityError("teacher digest does not match TeacherSpec")
    state = torch.load(spec.checkpoint, map_location="cpu", weights_only=True)
    required = declared_teacher_keys(spec)
    missing = sorted(required - state.keys())
    if missing:
        raise TeacherCompatibilityError(f"teacher is missing declared keys: {missing}")
    policy = StatefulPolicy(model_config)
    policy.load_state_dict(state, strict=False)
    policy.requires_grad_(False).eval()
    return LoadedTeacher(policy=policy, spec=spec)
```

- [ ] **Step 4: Split teacher losses by valid head and batch kind**

For normal RL batches, apply configured teacher KL to declared heads. For `teacher_distill`, apply the same head-aware KL and optional value alignment; configuration decides whether RL losses remain active. The default keeps RL active and adds teacher terms. Log each raw and weighted term separately.

```python
teacher_terms = {
    "operation_kl": masked_kl(student.unit_logits, teacher.unit_logits, batch.unit_masks)
    if loaded.spec.operation else zero,
    "quantity_kl": masked_kl(student.quantity_logits, teacher.quantity_logits, batch.quantity_masks)
    if loaded.spec.quantity else zero,
    "market_kl": masked_kl(student.market_logits, teacher.market_logits, batch.market_masks)
    if loaded.spec.market else zero,
    "value": F.mse_loss(student.values, teacher.values) if loaded.spec.value else zero,
}
teacher_total = config.optimizer.teacher_kl_cost * sum(teacher_terms.values())
total = rl_total + teacher_total
```

Run: `uv run pytest tests/learn/test_toad_population.py tests/learn/test_toad_lightning.py tests/learn/test_toad_runner.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/learn/toad/population.py src/kaggriculture/learn/toad/config.py src/kaggriculture/learn/toad/lightning.py tests/learn/test_toad_population.py tests/learn/test_toad_lightning.py
git commit -m "feat: enforce head-aware teacher distillation"
```

### Task 4: Split entropy accounting by action head

**Files:**

- Modify: `src/kaggriculture/learn/toad/lightning.py`
- Create: `tests/learn/test_toad_entropy.py`
- Modify: `tests/learn/test_toad_runner.py`

**Interfaces:**

- Consumes: masked log-probabilities, transfer-slot condition, and existing entropy helper.
- Produces: `HeadEntropy`, `compute_head_entropy`, and per-head loss/metric terms.

- [ ] **Step 1: Write conditional-quantity and padding tests**

```python
def test_quantity_entropy_counts_only_transfer_decisions() -> None:
    entropy = compute_head_entropy(logits(), masks(), unit_actions=no_transfer_actions())
    assert entropy.quantity.valid == 0
    assert entropy.quantity.sum == 0


def test_padded_unit_slots_do_not_count_toward_operation_entropy() -> None:
    entropy = compute_head_entropy(logits(), masks_with_one_real_unit(), unit_actions=actions())
    assert entropy.operation.valid == turns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/learn/test_toad_entropy.py -v`

Expected: FAIL because `compute_head_entropy` is undefined.

- [ ] **Step 3: Implement sum/count pairs instead of premature means**

```python
@dataclass(frozen=True)
class EntropyStat:
    sum: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class HeadEntropy:
    operation: EntropyStat
    quantity: EntropyStat
    market: EntropyStat
```

Return positive entropy sums and integer valid counts. Convert to the negative entropy loss only after multiplying each mean by its configured controller multiplier. Retain the old fixed summed coefficient when adaptive entropy is disabled.

```python
operation = EntropyStat(
    sum=entropy_of(unit_log_probs, unit_masks).masked_fill(padded_units, 0).sum(),
    valid=(~padded_units).sum(),
)
quantity_valid = transfer_slots(unit_actions) & ~padded_units
quantity = EntropyStat(
    sum=entropy_of(quantity_log_probs, quantity_masks).masked_fill(~quantity_valid, 0).sum(),
    valid=quantity_valid.sum(),
)
market = EntropyStat(
    sum=entropy_of(market_log_probs, market_masks).sum(),
    valid=torch.tensor(math.prod(market_log_probs.shape[:-1]), device=market_log_probs.device),
)
```

- [ ] **Step 4: Prove disabled adaptive mode matches the control fixture**

Run: `uv run pytest tests/learn/test_toad_entropy.py tests/learn/test_toad_control_fixture.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/learn/toad/lightning.py tests/learn/test_toad_entropy.py tests/learn/test_toad_runner.py
git commit -m "refactor: measure Toad entropy by head"
```

### Task 5: Add checkpointed target-entropy controllers

**Files:**

- Modify: `src/kaggriculture/learn/toad/config.py`
- Modify: `src/kaggriculture/learn/toad/lightning.py`
- Modify: `src/kaggriculture/learn/toad/callbacks.py`
- Modify: `tests/learn/test_toad_entropy.py`
- Modify: `tests/learn/test_toad_checkpoint.py`

**Interfaces:**

- Consumes: round entropy sum/counts and global environment-step delta.
- Produces: `EntropyControllerConfig`, `EntropyControllerState`, and `update_entropy_controller`.

- [ ] **Step 1: Write direction, decay, bound, and resume tests**

```python
def test_controller_reduces_multiplier_above_target() -> None:
    state = EntropyControllerState(target=1.0, multiplier=0.1, last_steps=10)
    updated = update_entropy_controller(state, observed=1.2, steps=20, config=controller_config())
    assert updated.multiplier < state.multiplier


def test_controller_increases_multiplier_below_target() -> None:
    state = EntropyControllerState(target=1.0, multiplier=0.1, last_steps=10)
    updated = update_entropy_controller(state, observed=0.8, steps=20, config=controller_config())
    assert updated.multiplier > state.multiplier


def test_entropy_controller_round_trips_in_checkpoint(tmp_path: Path) -> None:
    resumed = save_and_resume(module_with_controller_state(), tmp_path)
    assert resumed.entropy_state == module.entropy_state
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/learn/test_toad_entropy.py tests/learn/test_toad_checkpoint.py -k "controller or entropy" -v`

Expected: FAIL because controller types are undefined.

- [ ] **Step 3: Implement the pure controller**

```python
class EntropyControllerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    initial_target: NonNegativeFloat
    target_change_per_step: float = 0.0
    target_floor: NonNegativeFloat = 0.0
    initial_multiplier: NonNegativeFloat
    multiplier_change_per_step: NonNegativeFloat
    minimum: NonNegativeFloat = 0.0
    maximum: PositiveFloat


class EntropyControllersConfig(BaseModel):
    operation: EntropyControllerConfig
    quantity: EntropyControllerConfig
    market: EntropyControllerConfig


@dataclass(frozen=True)
class EntropyControllerState:
    target: float
    multiplier: float
    last_steps: int
```

Add `adaptive_entropy: bool = False` and `entropy: EntropyControllersConfig` to `OptimizerConfig`. Its validator requires `minimum <= initial_multiplier <= maximum` and `multiplier_change_per_step * max_steps_per_round < 1`.

```python
def update_entropy_controller(state, observed, steps, config):
    delta = steps - state.last_steps
    target = max(config.target_floor, state.target + config.target_change_per_step * delta)
    change = config.multiplier_change_per_step * delta
    factor = 1.0 - change if observed > target else 1.0 + change
    multiplier = min(config.maximum, max(config.minimum, state.multiplier * factor))
    return EntropyControllerState(target=target, multiplier=multiplier, last_steps=steps)
```

Validate `0 <= multiplier_change_per_step * max_steps_per_round < 1` so the decreasing factor cannot become negative.

- [ ] **Step 4: Aggregate and update at end of round**

Accumulate detached FP32 sum/counts in `ToadLightningModule`. At `end_of_round`, form each valid mean, update its controller, log observed/target/multiplier, clear accumulators, and checkpoint all three controller states.

```python
for name, stat in report.entropy.items():
    self.round_entropy[name].sum += stat.sum.detach().float()
    self.round_entropy[name].valid += stat.valid.detach()
if batch.end_of_round:
    for name, accumulated in self.round_entropy.items():
        observed = accumulated.sum / accumulated.valid.clamp_min(1)
        self.entropy_state[name] = update_entropy_controller(
            self.entropy_state[name], float(observed), self.environment_steps, self.config.optimizer.entropy[name]
        )
    self.round_entropy = empty_entropy_accumulators(self.device)
```

Run: `uv run pytest tests/learn/test_toad_entropy.py tests/learn/test_toad_checkpoint.py tests/learn/test_toad_lightning.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/learn/toad/config.py src/kaggriculture/learn/toad/lightning.py src/kaggriculture/learn/toad/callbacks.py tests/learn/test_toad_entropy.py tests/learn/test_toad_checkpoint.py
git commit -m "feat: adapt Toad entropy to per-head targets"
```

### Task 6: Snapshot through callbacks and gate the population run

**Files:**

- Modify: `src/kaggriculture/learn/toad/callbacks.py`
- Modify: `src/kaggriculture/learn/toad/lightning.py`
- Modify: `tests/learn/test_toad_callbacks.py`
- Create: `tests/learn/test_toad_population_integration.py`

**Interfaces:**

- Consumes: `SnapshotStore`, completed round metadata, and recurrent Lightning state.
- Produces: `PopulationSnapshotCallback`, `RoundMetricAccumulator`, and end-to-end population checkpoint/resume.

- [ ] **Step 1: Write callback cadence and resume tests**

Assert no snapshot before the configured global environment-step threshold, exactly one atomic snapshot at the threshold, the new manifest in the Lightning checkpoint, and identical next opponent assignment after resume.

```python
def test_population_snapshot_is_boundary_only_and_resume_stable(tmp_path: Path) -> None:
    trainer, module, data, callback = population_callback_fixture(tmp_path, interval=100)
    callback.on_train_batch_end(trainer, module, None, non_end_batch(steps=100), 0)
    assert not list(tmp_path.glob("snapshot-*.pt"))
    callback.on_train_batch_end(trainer, module, None, end_batch(steps=100), 1)
    assert len(list(tmp_path.glob("snapshot-*.pt"))) == 1
    resumed = checkpoint_and_resume(trainer, module, tmp_path)
    assert resumed.next_assignment == module.next_assignment
```

- [ ] **Step 2: Implement boundary-only snapshot callback**

```python
class PopulationSnapshotCallback(L.Callback):
    def on_fit_start(self, trainer, module) -> None:
        if module.config.population.snapshot_at_start and not self.store.manifest.entries:
            self.store.add(
                module.policy.state_dict(),
                environment_steps=module.environment_steps,
                round_id=module.collection_round,
                run_id=module.run_id,
            )
            trainer.datamodule.publish_manifest(self.store.manifest)

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx) -> None:
        if not batch.end_of_round or module.environment_steps < self.next_snapshot_steps:
            return
        entry = self.store.add(
            module.policy.state_dict(),
            environment_steps=module.environment_steps,
            round_id=module.collection_round,
            run_id=module.run_id,
        )
        trainer.datamodule.publish_manifest(self.store.manifest)
        module.population_manifest = self.store.manifest
        self.next_snapshot_steps += self.interval
```

Add one round accumulator to `ToadLightningModule`; update it from detached batch/report values and flush only at `end_of_round`:

```python
def flush_round_metrics(self) -> None:
    metrics = self.round_metrics.compute()
    metrics.update(
        {
            "progress/environment_steps": float(self.environment_steps),
            "progress/optimizer_steps": float(self.global_step),
            "actor/version": float(self.actor_version),
            "actor/lag_optimizer_steps": float(self.global_step - self.actor_source_global_step),
            "state/hidden_norm": self.last_state.hidden.float().norm(),
            "state/cell_norm": self.last_state.cell.float().norm(),
        }
    )
    self.log_dict(metrics, on_step=True, on_epoch=False, sync_dist=False)
    self.round_metrics.reset()
```

`RoundMetricAccumulator.compute()` emits collection/learner seconds and steps per second; games, returns, and losses by batch kind/opponent; mask densities; belief error; per-head entropy/target/multiplier/valid count; and population selection counts. The DDP plan changes `sync_dist` and reductions, not these metric names.

- [ ] **Step 3: Run the four-kind integration test**

Run a small round with two games of each kind. Assert observed counts, valid opponent IDs/digests, finite losses, and correct entropy valid counts.

```python
def test_four_kind_round_trains_and_reports_actual_mix(tmp_path: Path) -> None:
    result = run_population_round(tmp_path, games_per_kind=2)
    assert result.kind_counts == {kind: 2 for kind in BatchKind}
    assert all(result.opponent_ids)
    assert all(torch.isfinite(loss) for loss in result.losses)
    assert result.entropy_valid["quantity"] <= result.entropy_valid["operation"]
    assert "throughput/collection_steps_per_second" in result.metrics
    assert "belief/loss" in result.metrics
    assert "progress/environment_steps" in result.metrics
```

Run: `uv run pytest tests/learn/test_toad_population_integration.py tests/learn/test_toad_callbacks.py -v`

Expected: PASS.

- [ ] **Step 4: Run full validation**

Run: `uv run pre-commit run -a`

Expected: all hooks PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/learn/toad/callbacks.py src/kaggriculture/learn/toad/lightning.py tests/learn/test_toad_callbacks.py tests/learn/test_toad_population_integration.py
git commit -m "feat: train Toad against a checkpointed population"
```

## Completion Gate

This plan is complete only when all four batch kinds are observed at configured deterministic ratios, every selected snapshot is digest-verified, teacher heads are constrained only when present, adaptive entropy moves in the correct direction per head, and checkpoint/resume reproduces the next opponent selection and controller update.
