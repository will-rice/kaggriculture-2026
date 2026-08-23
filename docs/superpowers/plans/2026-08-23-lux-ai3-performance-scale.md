# Lux AI3 Performance and Scale Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable Lightning-managed BF16, optional `torch.compile`, multi-device DDP with disjoint experience, and the existing tensor simulator without changing the accepted training semantics.

**Architecture:** Add one optimization at a time behind typed runtime flags. Keep numerically sensitive RL math in FP32, use a single compile/unwrap boundary, explicitly shard game IDs instead of relying on Lightning's sampler, and adapt both rollout backends to the same `LearnerBatch` contract.

**Tech Stack:** Python 3.11+, PyTorch 2.13+, Lightning 2.6.5+, Pydantic 2.12+, CUDA where available, pytest.

**Spec:** `docs/superpowers/specs/2026-08-23-lux-ai3-native-port-design.md`

## Global Constraints

- Complete the Lightning foundation, recurrent-model, and population/entropy plans first.
- Eager single-device FP32 remains the permanent control.
- BF16 is `precision="bf16-mixed"`; no manual GradScaler or learner autocast context is introduced.
- Masked log-softmax, importance ratios, V-trace, UPGO, TD(lambda), value targets, teacher KL, entropy controllers, and finite checks execute in FP32.
- A requested but unavailable precision, compile mode, DDP topology, or rollout backend fails explicitly.
- Every DDP rank collects the same learner-batch count and disjoint global game IDs.
- Global environment steps are the all-rank sum; optimizer `global_step` is never used as the sample budget.
- Native and reference collection must agree within the tensor simulator's declared action/reward domain before native becomes default.

## File Structure

- Modify `src/kaggriculture/learn/toad/config.py`: precision, compile, DDP, and rollout-backend validation.
- Modify `src/kaggriculture/learn/toad/lightning.py`: FP32 loss islands, distributed counters, and finite diagnostics.
- Create `src/kaggriculture/learn/toad/compile.py`: compile/unwrap/export boundary.
- Modify `src/kaggriculture/learn/toad/data.py`: rank-aware game allocation and native round source.
- Modify `src/kaggriculture/learn/toad/callbacks.py`: rank-zero atomic writes plus barriers/broadcasts.
- Modify `src/kaggriculture/sim/rollout.py`: stateful policy/output support and common batch adapter.
- Modify `src/kaggriculture/learn/scripts/toad.py`: complete Trainer topology and runtime preflight.
- Create `tests/learn/test_toad_precision.py`, `test_toad_compile.py`, `test_toad_ddp.py`, and `test_toad_native.py`.
- Modify CUDA, simulator rollout, CLI, checkpoint, and integration tests.

---

### Task 1: Add runtime preflight and BF16-safe loss islands

**Files:**

- Modify: `src/kaggriculture/learn/toad/config.py`
- Modify: `src/kaggriculture/learn/toad/lightning.py`
- Modify: `src/kaggriculture/learn/scripts/toad.py`
- Create: `tests/learn/test_toad_precision.py`

**Interfaces:**

- Consumes: resolved `RuntimeConfig` and `PolicyOutput`.
- Produces: `runtime_preflight(config)`, `fp32_policy_terms(output, batch)`, and structured `NonFiniteTrainingError`.

- [ ] **Step 1: Write failing config and dtype tests**

```python
def test_bf16_is_rejected_when_accelerator_has_no_support(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    with pytest.raises(RuntimePreflightError, match="bf16-mixed"):
        runtime_preflight(bf16_cuda_config())


def test_sensitive_policy_math_is_fp32_under_autocast() -> None:
    module = ToadLightningModule(bf16_config())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        report = module.compute_report(synthetic_batch())
    assert report.debug_dtypes["learner_log_probs"] == torch.float32
    assert report.debug_dtypes["importance_ratios"] == torch.float32
    assert report.debug_dtypes["value_targets"] == torch.float32
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/learn/test_toad_precision.py -v`

Expected: FAIL because preflight and dtype diagnostics do not exist.

- [ ] **Step 3: Permit BF16 in configuration and implement preflight**

```python
class RuntimeConfig(BaseModel):
    precision: Literal["32-true", "bf16-mixed"] = "32-true"
    accelerator: Literal["auto", "cpu", "gpu"] = "auto"
    devices: PositiveInt | tuple[NonNegativeInt, ...] | Literal["auto"] = 1


def runtime_preflight(config: ToadConfig) -> None:
    if config.runtime.precision == "bf16-mixed":
        if config.runtime.accelerator == "gpu" and not torch.cuda.is_bf16_supported():
            raise RuntimePreflightError("bf16-mixed requested but CUDA BF16 is unavailable")
        if config.runtime.accelerator not in {"auto", "cpu", "gpu"}:
            raise RuntimePreflightError(f"unsupported BF16 accelerator {config.runtime.accelerator}")
```

The test suite may skip an actual GPU BF16 update when CUDA is absent, but config/preflight unit tests do not skip.

- [ ] **Step 4: Promote sensitive math and add finite diagnostics**

Immediately after the model forward, call `.float()` on logits and values used by policy/value losses. Keep recurrent feature computation under Lightning autocast. Before returning total loss, check every named input/output/loss tensor with `torch.isfinite`; raise an exception containing batch kind, game IDs, opponent digests, actor version, precision, and hidden/cell norms.

```python
unit_logits = output.unit_logits.float()
quantity_logits = output.quantity_logits.float()
market_logits = output.market_logits.float()
values = output.values.float()
checked = {"unit_logits": unit_logits, "values": values, **report.terms}
nonfinite = [name for name, tensor in checked.items() if not torch.isfinite(tensor).all()]
if nonfinite:
    raise NonFiniteTrainingError.from_batch(nonfinite, batch, output.state, self.config.runtime.precision)
```

Run: `uv run pytest tests/learn/test_toad_precision.py tests/learn/test_toad_lightning.py -v`

Expected: PASS. On BF16-capable GPU also run:

Run: `uv run pytest tests/learn/test_toad_precision.py -m cuda -v`

Expected: PASS or SKIP only when CUDA/BF16 is unavailable.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/learn/toad/config.py src/kaggriculture/learn/toad/lightning.py src/kaggriculture/learn/scripts/toad.py tests/learn/test_toad_precision.py
git commit -m "feat: train Toad with Lightning BF16"
```

### Task 2: Add one compile and unwrap boundary

**Files:**

- Create: `src/kaggriculture/learn/toad/compile.py`
- Modify: `src/kaggriculture/learn/toad/config.py`
- Modify: `src/kaggriculture/learn/scripts/toad.py`
- Modify: `src/kaggriculture/learn/toad/callbacks.py`
- Create: `tests/learn/test_toad_compile.py`

**Interfaces:**

- Consumes: `ToadLightningModule` and compile runtime fields.
- Produces: `maybe_compile(module, config)`, `unwrap_compiled(module)`, and `policy_state_dict(module)`.

- [ ] **Step 1: Write failing wrapper and export tests**

```python
def test_compile_disabled_returns_the_same_module() -> None:
    module = ToadLightningModule(control_config())
    assert maybe_compile(module, control_config()) is module


def test_unwrap_returns_original_module(monkeypatch) -> None:
    module = ToadLightningModule(control_config())
    monkeypatch.setattr(torch, "compile", fake_compile_wrapper)
    compiled = maybe_compile(module, compile_config())
    assert unwrap_compiled(compiled) is module
    assert policy_state_dict(compiled).keys() == module.policy.state_dict().keys()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/learn/test_toad_compile.py -v`

Expected: FAIL because the compile module does not exist.

- [ ] **Step 3: Implement explicit compile configuration and helper**

```python
class CompileConfig(BaseModel):
    enabled: bool = False
    mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"
    fullgraph: bool = False
    dynamic: bool = False


def maybe_compile(module: ToadLightningModule, config: ToadConfig) -> nn.Module:
    if not config.runtime.compile.enabled:
        return module
    try:
        return torch.compile(
            module,
            mode=config.runtime.compile.mode,
            fullgraph=config.runtime.compile.fullgraph,
            dynamic=config.runtime.compile.dynamic,
        )
    except Exception as error:
        raise CompileRequestedError(str(error)) from error


def unwrap_compiled(module: nn.Module) -> ToadLightningModule:
    return cast(ToadLightningModule, getattr(module, "_orig_mod", module))
```

Every actor publication, population snapshot, checkpoint metadata hook, and submission export calls `unwrap_compiled` or `policy_state_dict`; no caller reaches `_orig_mod` directly.

- [ ] **Step 4: Add eager-versus-compiled smoke parity**

On supported systems, compile the recurrent module, run one fixed training batch, and compare loss/parameter update within declared FP32 tolerance. Then save, restore eager, and assert state-dict keys have no `_orig_mod.` prefix.

```python
@pytest.mark.compile
def test_compiled_update_and_eager_checkpoint_keys(tmp_path: Path) -> None:
    eager, compiled, batch = compile_parity_fixture()
    eager_loss = fit_one_batch(eager, batch)
    compiled_loss = fit_one_batch(compiled, batch)
    assert compiled_loss == pytest.approx(eager_loss, rel=1e-5, abs=1e-6)
    path = save_policy_state(compiled, tmp_path / "policy.pt")
    state = torch.load(path, weights_only=True)
    assert all(not key.startswith("_orig_mod.") for key in state)
```

Run: `uv run pytest tests/learn/test_toad_compile.py -v`

Expected: PASS; mark only the real compiler smoke test `@pytest.mark.compile` so normal unit coverage still exercises the wrapper with a fake compiler.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/learn/toad/compile.py src/kaggriculture/learn/toad/config.py src/kaggriculture/learn/scripts/toad.py src/kaggriculture/learn/toad/callbacks.py tests/learn/test_toad_compile.py
git commit -m "feat: compile Toad through one export boundary"
```

### Task 3: Shard experience and counters explicitly under DDP

**Files:**

- Modify: `src/kaggriculture/learn/toad/data.py`
- Modify: `src/kaggriculture/learn/toad/lightning.py`
- Modify: `src/kaggriculture/learn/toad/callbacks.py`
- Modify: `src/kaggriculture/learn/scripts/toad.py`
- Create: `tests/learn/test_toad_ddp.py`

**Interfaces:**

- Consumes: `global_rank`, `world_size`, next global game ID, and environments per rank.
- Produces: `rank_game_ids`, round-step all-reduction, rank-zero durable writes, and DDP Trainer configuration.

- [ ] **Step 1: Write pure sharding tests**

```python
def test_rank_game_ids_are_disjoint_and_contiguous() -> None:
    ids = [rank_game_ids(start=100, per_rank=3, rank=rank, world_size=2) for rank in range(2)]
    assert ids == [(100, 101, 102), (103, 104, 105)]
    assert set(ids[0]).isdisjoint(ids[1])


def test_changed_world_size_continues_after_checkpoint_boundary() -> None:
    resumed = rank_game_ids(start=106, per_rank=2, rank=0, world_size=3)
    assert resumed == (106, 107)
```

- [ ] **Step 2: Run pure tests to verify they fail**

Run: `uv run pytest tests/learn/test_toad_ddp.py -k "rank_game_ids" -v`

Expected: FAIL because rank-aware allocation does not exist.

- [ ] **Step 3: Implement rank-aware allocation and equal batch preflight**

Extend `RuntimeConfig.num_nodes` to `PositiveInt` and `strategy` to `Literal["auto", "ddp"]`. Validation requires `strategy="ddp"` whenever resolved world size exceeds one.

```python
def rank_game_ids(start: int, per_rank: int, rank: int, world_size: int) -> tuple[int, ...]:
    rank_start = start + rank * per_rank
    return tuple(range(rank_start, rank_start + per_rank))
```

Each rank receives the same per-kind quota and number of optimizer batches but samples frozen opponents independently from its unique game IDs. Before collection begins, all-gather expected policy/value batch counts and raise `DistributedRoundMismatch` if they differ.

- [ ] **Step 4: All-reduce steps and make writes rank-safe**

Accumulate `round_local_steps` without incrementing the global counter on each batch. At end-of-round:

```python
local = torch.tensor(module.round_local_steps, device=module.device, dtype=torch.long)
global_delta = trainer.strategy.reduce(local, reduce_op="sum")
module.environment_steps += int(global_delta.item())
module.next_game_id += config.population.environments_per_rank * trainer.world_size
module.round_local_steps = 0
```

All ranks enter checkpoint/snapshot callbacks. Only `trainer.is_global_zero` writes; after atomic rename, call `trainer.strategy.barrier()`, then broadcast the serialized manifest/counters from rank zero before the next collection.

Reduce round metric numerators and denominators before forming means:

```python
for metric in module.round_metrics.reducible_values():
    metric.sum = trainer.strategy.reduce(metric.sum, reduce_op="sum")
    metric.count = trainer.strategy.reduce(metric.count, reduce_op="sum")
module.flush_round_metrics(sync_dist=False)
```

Rank-local timing diagnostics are logged with a `rank/{global_rank}/` prefix; objective, loss, entropy, belief, population, and throughput totals use the reduced values above.

Before backward, synchronize the finite flag so one bad rank cannot leave its peers blocked in gradient reduction:

```python
local_bad = torch.tensor(bool(nonfinite), device=module.device, dtype=torch.int32)
any_bad = trainer.strategy.reduce(local_bad, reduce_op="max")
if int(any_bad.item()):
    details: list[dict[str, object] | None] = [None] * trainer.world_size
    torch.distributed.all_gather_object(details, local_nonfinite_details(batch, nonfinite))
    raise NonFiniteTrainingError.render_distributed(details)
```

- [ ] **Step 5: Add a two-process CPU DDP integration test and commit**

Launch `Trainer(accelerator="cpu", devices=2, strategy="ddp", max_steps=2, use_distributed_sampler=False)` in a subprocess fixture. Write each rank's game IDs and final parameter digest to separate temporary files. Assert disjoint IDs, identical parameter digests, equal batch counts, one checkpoint, one snapshot, and successful resume.

```python
def test_two_process_ddp_collects_unique_games_and_syncs_weights(tmp_path: Path) -> None:
    run_ddp_fixture(tmp_path, devices=2, rounds=1)
    rank0 = json.loads((tmp_path / "rank-0.json").read_text())
    rank1 = json.loads((tmp_path / "rank-1.json").read_text())
    assert set(rank0["game_ids"]).isdisjoint(rank1["game_ids"])
    assert rank0["parameter_sha256"] == rank1["parameter_sha256"]
    assert rank0["batch_count"] == rank1["batch_count"]
    assert len(list(tmp_path.glob("*.ckpt"))) == 1
    assert len(list(tmp_path.glob("snapshot-*.pt"))) == 1
    assert resume_ddp_fixture(tmp_path).next_game_id == rank0["next_game_id"]
```

Run: `uv run pytest tests/learn/test_toad_ddp.py -v`

Expected: PASS.

```bash
git add src/kaggriculture/learn/toad/data.py src/kaggriculture/learn/toad/lightning.py src/kaggriculture/learn/toad/callbacks.py src/kaggriculture/learn/scripts/toad.py tests/learn/test_toad_ddp.py
git commit -m "feat: shard Toad experience across DDP ranks"
```

### Task 4: Adapt the tensor simulator to the common stateful contract

**Files:**

- Modify: `src/kaggriculture/sim/rollout.py:83-122,427-620`
- Modify: `src/kaggriculture/learn/toad/data.py`
- Create: `tests/learn/test_toad_native.py`
- Modify: `tests/sim/test_rollout.py`
- Modify: `tests/sim/test_cuda.py`

**Interfaces:**

- Consumes: `StatefulPolicy`, `PolicyOutput`, `PolicyState`, `SimState`, and `LearnerBatch`.
- Produces: recurrent fields on tensor trajectories and `NativeRoundSource.collect_round`.

The adapted simulator signature is `collect_segment(state: SimState, policy: StatefulPolicy, *, policy_state: PolicyState | None = None, turns: int = 32, generator: torch.Generator | None = None, opponent: ScriptedOpponent | None = None) -> tuple[SimState, PolicyState | None, Trajectory]`.

- [ ] **Step 1: Write stateful tensor-rollout tests**

```python
def test_tensor_rollout_carries_and_resets_policy_state() -> None:
    state = terminal_crossing_state(batch=2)
    policy = recording_stateful_policy()
    _, _, trajectory = collect_segment(state, policy, turns=2)
    assert trajectory.initial_hidden.shape[:3] == (2, 2, policy.hidden_channels)
    assert policy.recorded_reset[1].all()


def test_native_source_emits_the_common_learner_batch() -> None:
    batches = list(NativeRoundSource(native_config()).collect_round(round_request()))
    assert all(isinstance(batch, LearnerBatch) for batch in batches)
    assert batches[-1].end_of_round
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/learn/test_toad_native.py tests/sim/test_rollout.py -k "stateful or common_learner" -v`

Expected: FAIL because tensor rollout expects the old four-tuple policy output.

- [ ] **Step 3: Extend tensor trajectories and policy calls**

Initialize one `PolicyState` for `(sim_batch * seats)` and call:

```python
output = policy(
    boards.flatten(0, 1),
    scalars.flatten(0, 1),
    positions.flatten(0, 1),
    state=policy_state,
    dones=dones.flatten(0, 1),
)
policy_state = output.state
```

Record the pre-action state needed at each segment boundary, belief targets from tensor `SimState`, and the same actor version/game/opponent metadata as reference collection. Preserve all existing on-device tensors and avoid `.item()` inside `collect_segment`.

- [ ] **Step 4: Implement `NativeRoundSource` and differential tests**

`NativeRoundSource` creates/reset tensor states from assigned seeds, loads the same current/frozen policies as `ReferenceRoundSource`, invokes `collect_segment`, and converts its trajectory directly into `LearnerBatch` without a CPU `Trajectory` round trip.

```python
class NativeRoundSource:
    def collect_round(self, request: RoundRequest) -> Iterator[LearnerBatch]:
        state = reset(self.sim_config, torch.tensor(request.seeds, device=self.device))
        state, policy_state, trajectory = collect_segment(
            state,
            self.actor_for(request.actor_version),
            policy_state=self.initial_policy_state(state),
            turns=self.config.optimizer.unroll_length,
            generator=self.generator,
            opponent=self.scripted_opponent(request),
        )
        yield from self.expander.expand_native(trajectory, request.meta)
```

For fixed legal action tapes and seeds, compare reference versus native observations, masks, decoded actions, rewards, dones, terminal banks, recurrent reset locations, belief targets, and metadata. Scripted-opponent differential coverage must include bulk pickup/place.

Run: `uv run pytest tests/learn/test_toad_native.py tests/sim/test_rollout.py tests/sim/test_differential.py -v`

Expected: PASS.

- [ ] **Step 5: Preserve CUDA graph capture and commit**

Update the existing graph-capture test to include static recurrent state buffers and copy successor state back into those buffers inside the captured region. There must be no host synchronization in the self-play capture path.

```python
graph = torch.cuda.CUDAGraph()
static_state = reset(Config(), seeds.cuda())
static_policy_state = policy.initial_state(static_state.batch_size * 2, like=static_state.money)
with torch.cuda.graph(graph):
    next_state, next_policy_state, trajectory = collect_segment(
        static_state, policy, policy_state=static_policy_state, turns=16
    )
    copy_state_(static_state, next_state)
    copy_policy_state_(static_policy_state, next_policy_state)
graph.replay()
assert static_state.step.eq(32).all()
```

Run: `uv run pytest tests/sim/test_cuda.py -v`

Expected: PASS on CUDA or existing hardware-based SKIP.

```bash
git add src/kaggriculture/sim/rollout.py src/kaggriculture/learn/toad/data.py tests/learn/test_toad_native.py tests/sim/test_rollout.py tests/sim/test_cuda.py
git commit -m "feat: collect stateful Toad batches in the tensor simulator"
```

### Task 5: Gate runtime combinations and retire old orchestration

**Files:**

- Modify: `src/kaggriculture/learn/scripts/toad.py`
- Modify: `src/kaggriculture/learn/scripts/curriculum.py`
- Modify: `src/kaggriculture/learn/toad/config.py`
- Modify: `tests/learn/test_toad_cli.py`
- Modify: `tests/learn/test_toad_model_integration.py`
- Modify: `docs/superpowers/specs/2026-08-23-lux-ai3-native-port-design.md` only if implementation-established constraints differ, with an explicit dated amendment.

**Interfaces:**

- Consumes: all completed port interfaces.
- Produces: the sole active Lightning Toad entry point and compatibility loader for old checkpoints.

- [ ] **Step 1: Add a runtime matrix test**

Parameterize accepted combinations: reference/eager/FP32 on CPU; reference/eager/BF16 on supported accelerator; reference/compiled/BF16; native/eager/BF16; and DDP reference/native. Parameterize rejected combinations with exact preflight errors, including unavailable devices and native scripted CUDA-graph capture.

```python
@pytest.mark.parametrize(
    "config",
    [
        runtime_config("cpu", 1, "32-true", compile=False, backend="reference"),
        runtime_config("gpu", 1, "bf16-mixed", compile=False, backend="reference"),
        runtime_config("gpu", 1, "bf16-mixed", compile=True, backend="reference"),
        runtime_config("gpu", 1, "bf16-mixed", compile=False, backend="native"),
        runtime_config("cpu", 2, "32-true", compile=False, backend="reference"),
    ],
)
def test_supported_runtime_matrix(config: ToadConfig) -> None:
    assert runtime_preflight(config) is None
```

- [ ] **Step 2: Remove executable legacy orchestration**

Delete the old `while steps < total_steps`, manual device selection, direct `wandb.log`, custom optimizer/checkpoint loop, and direct `_step` optimizer mutation after their callers and compatibility tests have moved. Retain only old-checkpoint conversion helpers under names that cannot be selected for new training.

```python
def test_active_toad_entrypoint_has_no_legacy_training_loop() -> None:
    source = inspect.getsource(toad.main)
    assert "while steps" not in source
    assert "wandb.log" not in source
    assert "Trainer" in source or "run(config)" in source
```

- [ ] **Step 3: Run fixed-budget behavior and resume gates**

Run the accepted control config for a small fixed environment-step budget, compare scripted-opponent results with the characterized baseline interval, stop at a boundary, resume, and verify next game IDs/opponents/controller state. Repeat on two devices where available.

```python
def test_fixed_budget_resume_preserves_behavior_and_stream(tmp_path: Path) -> None:
    uninterrupted = run_control_budget(tmp_path / "full", steps=fixture_budget())
    first = run_control_budget(tmp_path / "split", steps=fixture_budget() // 2)
    resumed = run_control_budget(tmp_path / "split", steps=fixture_budget(), resume=first.checkpoint)
    assert baseline_interval().contains(resumed.scripted_win_rate)
    assert resumed.next_game_id == uninterrupted.next_game_id
    assert resumed.next_opponents == uninterrupted.next_opponents
    assert resumed.entropy_state == uninterrupted.entropy_state
```

- [ ] **Step 4: Run repository-wide validation**

Run: `uv run pre-commit run -a`

Expected: all hooks PASS.

Run on supported CUDA host: `uv run pytest -m "cuda or compile" -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/learn/scripts/toad.py src/kaggriculture/learn/scripts/curriculum.py src/kaggriculture/learn/toad/config.py tests/learn docs/superpowers/specs/2026-08-23-lux-ai3-native-port-design.md
git commit -m "refactor: make Lightning the active Toad trainer"
```

## Completion Gate

The port is complete only when eager FP32 remains equivalent, BF16 is finite with bounded update drift, compiled checkpoints/export have stable keys, DDP ranks collect disjoint games and converge to identical weights, reference/native differential tests pass, CUDA graph capture remains valid, boundary resume is deterministic, and the existing curriculum/frontier evaluation gate accepts the resulting policy.
