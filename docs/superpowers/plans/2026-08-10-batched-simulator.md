# Batched Kaggriculture Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the interpreter-shaped simulator with an exact, eager-PyTorch batched engine that matches Kaggle 1.32.6 after every turn and exceeds 110,000 complete environment-steps/second at batch 1,024.

**Architecture:** A compact mutable `SimState` exposes named canonical views over dtype-grouped tensors and reuses a device workspace. Device-cached rule tables drive one semantic scan over unit slots and one over market-order slots; all per-kind and per-quantity work is vectorized. Kaggle remains the sole oracle through phase-level and complete per-turn differential tests.

**Tech Stack:** Python 3.11, PyTorch 2.13, `kaggle-environments==1.32.6`, pytest, CUDA, `torch.profiler`.

## Global Constraints

- The installed `kaggle-environments==1.32.6` source and configuration are the sole fidelity oracle.
- The production runtime is plain eager PyTorch under `torch.inference_mode()`; compilation and CUDA graphs are optional measurements only.
- Runtime Python loops may scan only the fixed twenty unit slots and ten market-order slots.
- No hot-path loop may scan environments, players, products, crops, animals, shops, action types, or quantities.
- `step`, `observe`, `legal`, and `decode` may not call `.cpu()`, `.numpy()`, or `.item()`, branch on tensor data in Python, or construct/transfer CPU tensors.
- Every optimization must pass a direct Kaggle comparison before performance work proceeds.
- The final fidelity gate is zero unexplained divergences over at least 10,000 complete episodes.
- The final performance gate is at least 110,000 complete eager environment-steps/second at batch 1,024.

---

## File map

- `src/kaggriculture/sim/schema.py`: field indices, canonical sentinels, and named view descriptors.
- `src/kaggriculture/sim/state.py`: mutable grouped `SimState`, allocation, reset-facing accessors, and canonical views.
- `src/kaggriculture/sim/workspace.py`: reusable scratch tensors keyed by batch and device.
- `src/kaggriculture/sim/tables.py`: reference-derived host tables and cached device tensors.
- `src/kaggriculture/sim/reference.py`: reference identity, source manifest, supported-domain audit, and single-phase adapters.
- `src/kaggriculture/sim/codec.py`: pack/unpack between Kaggle objects and canonical state views.
- `src/kaggriculture/sim/units.py`: table-driven PLANT guard and ordered unit scan.
- `src/kaggriculture/sim/market.py`: ordered market-slot scan and vectorized quantity resolver.
- `src/kaggriculture/sim/day.py`: town, decay, day-boundary, RNG, reset, and clock phases.
- `src/kaggriculture/sim/observe.py`: vectorized observations.
- `src/kaggriculture/sim/legality.py`: vectorized unit and market masks.
- `src/kaggriculture/sim/decode.py`: vectorized bucket expansion.
- `src/kaggriculture/sim/engine.py`: public reset/step orchestration under inference mode.
- `src/kaggriculture/sim/rollout.py`: state-owning segmented rollout integration.
- `tests/sim/reference_cases.py`: reusable Kaggle state and action builders.
- `tests/sim/test_reference_contract.py`: hashes, branch manifest, and supported-domain bounds.
- `tests/sim/test_state_storage.py`: grouped storage, views, mutation, and workspace reuse.
- `tests/sim/test_tables.py`: reference-derived table contents and device residency.
- `tests/sim/test_units_vectorized.py`: direct unit-phase differential and loop-shape tests.
- `tests/sim/test_market_vectorized.py`: exhaustive market equivalence and mutations.
- `tests/sim/test_day_vectorized.py`: direct day-phase and RNG differential tests.
- `tests/sim/test_io_vectorized.py`: observation, legality, decode, and hot-path AST tests.
- `tests/sim/test_engine_vectorized.py`: complete per-turn CPU/CUDA differential tests.
- `tests/sim/test_rollout_vectorized.py`: segment ownership, rewards, masks, and no-autograd tests.
- `tests/sim/test_campaign.py`: campaign schema, resumption, and acceptance gating.
- `tests/sim/test_benchmark.py`: benchmark schema and eager acceptance gating.
- `scripts/sim_fidelity_campaign.py`: resumable 10,000-episode reference campaign.
- `scripts/benchmark_simulator.py`: eager benchmark and stage profiler matrix.
- `docs/experiments/fidelity/`: committed reference and acceptance evidence.

---

### Task 1: Freeze the reference contract and supported state domain

**Files:**

- Create: `src/kaggriculture/sim/reference.py`
- Create: `tests/sim/reference_cases.py`
- Create: `tests/sim/test_reference_contract.py`
- Modify: `src/kaggriculture/sim/fidelity.py`
- Modify: `tests/sim/test_reference_tripwire.py`

**Interfaces:**

- Produces: `ReferenceIdentity`, `reference_identity()`, `reference_branches()`, `SupportedDomain`, `supported_domain()`.
- Produces: `run_reference_unit_phase(...)` and `run_reference_market_phase(...)` adapters used by Tasks 4 and 5.

- [ ] **Step 1: Write failing tests for the reference identity and domain proof**

```python
def test_supported_domain_justifies_every_fixed_axis() -> None:
    domain = supported_domain()
    assert domain.players == 2
    assert domain.market_slots == 10
    assert domain.max_order_quantity == 64
    assert domain.unit_slots >= domain.max_reachable_units
    assert domain.max_reachable_units_proof.checked_states > 0


def test_reference_manifest_includes_market_quote_and_commit_branches() -> None:
    branches = reference_branches()
    assert any(name.startswith("_process_market:") for name in branches)
    assert any(name.startswith("_commit_unit:") for name in branches)
```

- [ ] **Step 2: Run the tests and verify the missing contract fails**

Run: `.venv/bin/python -m pytest tests/sim/test_reference_contract.py -vv`

Expected: FAIL because `SupportedDomain`, `supported_domain`, and the phase adapters do not exist.

- [ ] **Step 3: Implement the contract and a reachable-unit exhaustive search**

```python
@dataclass(frozen=True)
class BoundProof:
    checked_states: int
    maximum: int
    witness: tuple[int, ...]


@dataclass(frozen=True)
class SupportedDomain:
    players: int
    market_slots: int
    max_order_quantity: int
    unit_slots: int
    max_reachable_units: int
    max_reachable_units_proof: BoundProof
```

Parse constants from the installed wheel. Compute the maximum reachable daily
hires using exact integer money, Fibonacci hire costs, ten order slots per turn,
twenty-four turns per day, and the project's legal market action grammar. Store
the maximizing witness in the proof; do not assert `MAX_UNITS == 20` without it.

- [ ] **Step 4: Verify the reference contract and existing tripwire**

Run: `.venv/bin/python -m pytest tests/sim/test_reference_contract.py tests/sim/test_reference_tripwire.py -vv`

Expected: PASS with the installed 1.32.6 hashes and every fixed axis justified.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/sim/reference.py src/kaggriculture/sim/fidelity.py tests/sim/reference_cases.py tests/sim/test_reference_contract.py tests/sim/test_reference_tripwire.py
git commit -m "test: freeze simulator reference contract"
```

---

### Task 2: Introduce grouped mutable state and reusable workspace

**Files:**

- Create: `src/kaggriculture/sim/schema.py`
- Create: `src/kaggriculture/sim/codec.py`
- Create: `src/kaggriculture/sim/workspace.py`
- Modify: `src/kaggriculture/sim/state.py`
- Create: `tests/sim/test_state_storage.py`
- Modify: `tests/sim/test_codec.py`

**Interfaces:**

- Consumes: `supported_domain()` from Task 1.
- Produces: `SimState.allocate(batch: int, device: torch.device | str) -> SimState`.
- Produces: `Workspace.allocate(state: SimState) -> Workspace` and `Workspace.zero_scratch() -> None`.
- Preserves named properties such as `state.kind`, `state.money`, `state.inv_seq`, and `state.step` for codec and differential assertions.

- [ ] **Step 1: Write failing storage and no-allocation tests**

```python
def test_named_views_share_grouped_storage() -> None:
    state = SimState.allocate(3, "cpu")
    state.kind[0, 0, 2, 4] = 3
    assert state.tile_i16[0, 0, 2, 4, TileI16.KIND] == 3
    assert state.kind.data_ptr() == state.tile_i16[..., TileI16.KIND].data_ptr()


def test_workspace_reuses_every_scratch_buffer() -> None:
    state = SimState.allocate(8, "cpu")
    workspace = Workspace.allocate(state)
    pointers = workspace.data_ptrs()
    workspace.zero_scratch()
    assert workspace.data_ptrs() == pointers
```

- [ ] **Step 2: Run the tests and verify they fail on the old dataclass**

Run: `.venv/bin/python -m pytest tests/sim/test_state_storage.py -vv`

Expected: FAIL because grouped fields, schema enums, and `Workspace` are absent.

- [ ] **Step 3: Implement dtype-grouped storage with named views**

```python
@dataclass
class SimState:
    tile_i16: torch.Tensor
    tile_i32: torch.Tensor
    tile_bool: torch.Tensor
    farm_i64: torch.Tensor
    farm_i16: torch.Tensor
    unit_i16: torch.Tensor
    unit_bool: torch.Tensor
    private_i16: torch.Tensor
    private_i32: torch.Tensor
    market_i64: torch.Tensor
    town_i16: torch.Tensor
    clock_i32: torch.Tensor
    status_bool: torch.Tensor
    reward_f64: torch.Tensor
    rng_u32: torch.Tensor

    @property
    def kind(self) -> torch.Tensor:
        return self.tile_i16[..., TileI16.KIND]
```

Define all canonical fields as views. Move `pack` and `unpack` to `codec.py`,
and re-export them from `state.py` during migration so existing imports remain
valid until Task 8 updates callers. Preserve `inv_seq`, negative schedule
sentinels, integral money, and exact dtypes. Make `SimState` mutable; remove
phase-level `dataclasses.replace` assumptions from storage tests only—runtime
phase migration happens in later tasks.

- [ ] **Step 4: Adapt pack/unpack and verify exact codec round trips**

Run: `.venv/bin/python -m pytest tests/sim/test_state_storage.py tests/sim/test_codec.py tests/sim/test_differential.py -vv`

Expected: PASS, including named localization for a corrupted grouped view.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/sim/schema.py src/kaggriculture/sim/state.py src/kaggriculture/sim/codec.py src/kaggriculture/sim/workspace.py tests/sim/test_state_storage.py tests/sim/test_codec.py
git commit -m "refactor: add grouped simulator state"
```

---

### Task 3: Build reference-derived device tables and allocation-free reset

**Files:**

- Create: `src/kaggriculture/sim/tables.py`
- Create: `tests/sim/test_tables.py`
- Modify: `src/kaggriculture/sim/engine.py`
- Modify: `tests/sim/test_reset.py`

**Interfaces:**

- Produces: `host_tables() -> HostTables`.
- Produces: `device_tables(device: torch.device | str) -> DeviceTables` cached by resolved device index.
- Produces: `reset_(state: SimState, seeds: torch.Tensor, tables: DeviceTables, workspace: Workspace) -> None`.

- [ ] **Step 1: Write failing table-source and device-cache tests**

```python
def test_unit_action_tables_are_derived_from_encoding_contract() -> None:
    tables = host_tables()
    for action, index in tables.unit_action_index.items():
        assert UNIT_OPS[index] == action
    assert tables.unit_item.shape == (len(UNIT_OPS),)
    assert tables.move_delta.shape == (len(UNIT_OPS), 2)


@pytest.mark.cuda
def test_device_tables_are_cached_on_the_requested_device() -> None:
    first = device_tables("cuda")
    second = device_tables("cuda")
    assert first is second
    assert all(t.device.type == "cuda" for t in first.tensors())
```

- [ ] **Step 2: Run and verify RED**

Run: `.venv/bin/python -m pytest tests/sim/test_tables.py tests/sim/test_reset.py -vv`

Expected: FAIL because the table interfaces and in-place reset do not exist.

- [ ] **Step 3: Implement immutable host/device tables and in-place reset**

Use tuples and frozen dataclasses on the host. Convert each tensor exactly once
per resolved device. Replace hot-path `torch.tensor([...], device=...)` calls
with table views. Decorate public reset orchestration with inference mode.

```python
@torch.inference_mode()
def reset_(state: SimState, seeds: torch.Tensor, tables: DeviceTables,
           workspace: Workspace) -> None:
    state.zero_canonical_()
    state.money.fill_(STARTING_MONEY)
    state.inventory.copy_(tables.market_initial.expand(state.batch_size, -1))
    state.prices.copy_(tables.market_base.expand(state.batch_size, -1))
```

- [ ] **Step 4: Verify reset, tables, codec, and CPU/CUDA equality**

Run: `.venv/bin/python -m pytest tests/sim/test_tables.py tests/sim/test_reset.py tests/sim/test_codec.py -vv`

Expected: PASS; reset outputs have no `grad_fn` and all table tensors are on the state device.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/sim/tables.py src/kaggriculture/sim/engine.py tests/sim/test_tables.py tests/sim/test_reset.py
git commit -m "feat: add simulator device tables"
```

---

### Task 4: Replace unit dispatch with one table-driven semantic scan

**Files:**

- Modify: `src/kaggriculture/sim/units.py`
- Create: `tests/sim/test_units_vectorized.py`
- Modify: `tests/sim/test_rules_units.py`
- Modify: `tests/sim/test_hot_path.py`

**Interfaces:**

- Consumes: `SimState`, `Workspace`, and `DeviceTables`.
- Produces: `apply_unit_phases_(state: SimState, actions: torch.Tensor, tables: DeviceTables, workspace: Workspace) -> None`.

- [ ] **Step 1: Write a failing direct-reference unit-phase test**

```python
@pytest.mark.parametrize("case", UNIT_REFERENCE_CASES, ids=lambda case: case.name)
def test_vectorized_unit_phase_matches_kaggle(case: UnitCase) -> None:
    reference = case.environment()
    state = pack([reference])
    expected = run_reference_unit_phase(reference, case.actions)
    apply_unit_phases_(state, case.tensor_actions(state.device), device_tables("cpu"), Workspace.allocate(state))
    assert_identical(expected, state, 0)
```

Add an AST assertion that `apply_unit_phases_` contains exactly one runtime
`for` loop and that its iterator is `range(unit_slots)`.

- [ ] **Step 2: Run and verify the old nested scans fail the structural gate**

Run: `.venv/bin/python -m pytest tests/sim/test_units_vectorized.py -vv`

Expected: FAIL on the AST loop count and missing in-place interface.

- [ ] **Step 3: Implement gathered operation metadata and indexed mutations**

```python
for unit in range(state.unit_slots):
    op = actions[..., unit].long()
    op_class = tables.unit_class[op]
    item = tables.unit_item[op]
    crop = tables.unit_crop[op]
    animal = tables.unit_animal[op]
    delta = tables.move_delta[op]
    # Each branch below is a `(B, 2)` mask; no per-kind loop.
```

Implement atomic PLANT demand before the loop with `scatter_add_`. Use gathered
indices for pickup, placement, planting, harvest, care, and movement. Preserve
the exact reference ordering and `inv_seq` DROP behavior.

- [ ] **Step 4: Verify unit rules, mutations, codec equality, and AST shape**

Run: `.venv/bin/python -m pytest tests/sim/test_units_vectorized.py tests/sim/test_rules_units.py tests/sim/test_mutations.py -vv`

Expected: PASS with every named unit mutation detected.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/sim/units.py tests/sim/test_units_vectorized.py tests/sim/test_rules_units.py tests/sim/test_hot_path.py tests/sim/test_mutations.py
git commit -m "feat: vectorize simulator unit dispatch"
```

---

### Task 5: Prove and implement vectorized market quantities

**Files:**

- Modify: `src/kaggriculture/sim/market.py`
- Create: `tests/sim/test_market_vectorized.py`
- Modify: `tests/sim/test_rules_market.py`
- Modify: `tests/sim/test_mutations.py`

**Interfaces:**

- Produces: `resolve_market_slot_(state, order_type, item, quantity, tables, workspace) -> None`.
- Produces: `apply_market_phase_(state: SimState, actions: MarketActions, tables: DeviceTables, workspace: Workspace) -> None`.
- Runtime shape: one ten-slot loop; quantity candidates have tensor shape `(B, 2, 64)`.

- [ ] **Step 1: Write failing exhaustive one-seat market tests**

```python
@pytest.mark.parametrize("quantity", range(65))
@pytest.mark.parametrize("money_delta", (-1, 0, 1))
def test_one_seat_market_prefix_matches_reference(quantity: int, money_delta: int) -> None:
    case = market_boundary_case(quantity=quantity, money_delta=money_delta)
    assert_market_case_matches_reference(case)
```

Cover every product, crop, animal, price breakpoint, shed count `0..100`, and
sell count `0..64` through generated parameter cases.

- [ ] **Step 2: Write failing paired-seat quote-order tests**

```python
@pytest.mark.parametrize("pair", SAME_PRODUCT_ORDER_PAIRS)
def test_both_seats_use_the_same_precommit_quote(pair: OrderPair) -> None:
    result = run_vectorized_market_case(pair)
    expected = run_reference_market_case(pair)
    assert result.money == expected.money
    assert result.inventory == expected.inventory
    assert result.shed == expected.shed
```

Include buy/buy, sell/sell, buy/sell, sell/buy, first-round failure, unequal
quantities, price-floor transitions, and survivor tails.

- [ ] **Step 3: Run and verify RED against the fixed 65-iteration resolver**

Run: `.venv/bin/python -m pytest tests/sim/test_market_vectorized.py -vv -x`

Expected: FAIL because the vectorized slot resolver does not exist; the AST test also rejects a quantity loop.

- [ ] **Step 4: Implement a tensor quantity axis and prefix-valid resolver**

```python
rounds = tables.quantity_axis.view(1, 1, 64)
requested = rounds < quantity[..., None]
quotes = paired_quotes(state.inventory, order_type, item, rounds, tables)
costs = quotes * requested
affordable = costs.cumsum(-1) <= state.money[..., None]
capacity = rounds < (SHED_CAPACITY - state.shed.sum(-1))[..., None]
valid_prefix = requested & affordable & capacity & prior_rounds_succeeded(...)
accepted = valid_prefix.sum(-1)
```

Implement explicit dispatch masks for the six pair classes in the spec. Both
seat quotes come from each round's pre-commit inventory. Compute survivor tails
after the first failed seat with gathered prefix endpoints. Apply final exact
money, shed, seed, and market deltas once per slot, then refresh prices.

- [ ] **Step 5: Add and prove market mutation sensitivity**

Run each mutation separately: post-seat-zero quoting, prefix off by one,
price-floor supply, continuing after failure, per-item capacity, parallel slot
resolution. Each must make a named test fail before restoration.

Run: `.venv/bin/python -m pytest tests/sim/test_market_vectorized.py tests/sim/test_rules_market.py tests/sim/test_mutations.py -vv`

Expected: PASS after all mutations are restored.

- [ ] **Step 6: Commit**

```bash
git add src/kaggriculture/sim/market.py tests/sim/test_market_vectorized.py tests/sim/test_rules_market.py tests/sim/test_mutations.py
git commit -m "feat: vectorize simulator market resolution"
```

---

### Task 6: Vectorize day phases and preserve exact RNG

**Files:**

- Modify: `src/kaggriculture/sim/day.py`
- Modify: `src/kaggriculture/sim/rng.py`
- Create: `tests/sim/test_day_vectorized.py`
- Modify: `tests/sim/test_rules_day.py`
- Modify: `tests/sim/test_rng.py`

**Interfaces:**

- Produces: `apply_day_phases_(state: SimState, tables: DeviceTables, workspace: Workspace) -> None`.
- Preserves: bit-identical `rng_words(seeds)` and exact per-day cursor semantics.

- [ ] **Step 1: Write failing no-kind-loop and reference-phase tests**

```python
@pytest.mark.parametrize("case", DAY_REFERENCE_CASES, ids=lambda case: case.name)
def test_day_phase_matches_kaggle(case: DayCase) -> None:
    reference = case.environment()
    state = pack([reference])
    expected = case.advance_reference()
    apply_day_phases_(state, device_tables("cpu"), Workspace.allocate(state))
    assert_identical(expected, state, 0)
```

Add AST checks rejecting loops over `CROP_NAMES`, `ANIMAL_NAMES`, `SHOP_NAMES`,
and `SHOPS` in `apply_day_phases_` and its callees.

- [ ] **Step 2: Run and verify RED**

Run: `.venv/bin/python -m pytest tests/sim/test_day_vectorized.py tests/sim/test_rng.py -vv`

Expected: FAIL on the new in-place interface and runtime shop-drain construction.

- [ ] **Step 3: Implement gathered crop/animal schedules and table-driven drains**

Replace kind loops with indexed crop/animal property tensors. Replace per-call
shop drain construction with `tables.shop_drain[drawn_shop]`. Reuse workspace
buffers for DROP sorting and weed RNG gathers. Keep MT19937 generation exact;
its reset-time recurrence is outside the per-step hot path.

- [ ] **Step 4: Verify day rules, RNG vectors, and mutations**

Run: `.venv/bin/python -m pytest tests/sim/test_day_vectorized.py tests/sim/test_rules_day.py tests/sim/test_rng.py tests/sim/test_mutations.py -vv`

Expected: PASS, including full day-boundary CPU/CUDA equality.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/sim/day.py src/kaggriculture/sim/rng.py tests/sim/test_day_vectorized.py tests/sim/test_rules_day.py tests/sim/test_rng.py tests/sim/test_mutations.py
git commit -m "feat: vectorize simulator day phases"
```

---

### Task 7: Vectorize observation, legality, and decoding

**Files:**

- Modify: `src/kaggriculture/sim/observe.py`
- Modify: `src/kaggriculture/sim/legality.py`
- Modify: `src/kaggriculture/sim/decode.py`
- Create: `tests/sim/test_io_vectorized.py`
- Modify: `tests/sim/test_observe.py`
- Modify: `tests/sim/test_legality.py`
- Modify: `tests/sim/test_decode.py`

**Interfaces:**

- Produces: `observe(state, seat, tables, workspace) -> tuple[Tensor, Tensor, Tensor]`.
- Produces: `legal(state, seat, tables, workspace) -> tuple[Tensor, Tensor]`.
- Produces: `decode_actions(unit_buckets, market_buckets, tables, workspace) -> tuple[Tensor, MarketActions]`.

- [ ] **Step 1: Write failing bulk-output and AST tests**

```python
def test_io_paths_have_no_runtime_kind_scans() -> None:
    forbidden = {"CROP_NAMES", "ANIMAL_NAMES", "SHED_NAMES", "MARKET_SLOTS"}
    for function in (observe, legal, decode_actions):
        assert not loop_iterated_names(function) & forbidden


@pytest.mark.cuda
def test_io_outputs_remain_on_device_without_autograd() -> None:
    state, tables, workspace = cuda_fixture(batch=32)
    with torch.inference_mode():
        outputs = (*observe(state, 0, tables, workspace), *legal(state, 0, tables, workspace))
    assert all(t.is_cuda and t.grad_fn is None for t in outputs)
```

- [ ] **Step 2: Run and verify RED on current per-kind loops**

Run: `.venv/bin/python -m pytest tests/sim/test_io_vectorized.py -vv`

Expected: FAIL on loops in observation and legality.

- [ ] **Step 3: Implement table gathers and bulk scatter construction**

Use operation metadata for masks, kind-to-feature maps for observations, and a
fixed quantity axis for market decoding. Preserve exact feature ordering and
float conversion behavior from `kaggriculture.learn.encoding`.

- [ ] **Step 4: Verify bit equality across codec builders and random states**

Run: `.venv/bin/python -m pytest tests/sim/test_io_vectorized.py tests/sim/test_observe.py tests/sim/test_legality.py tests/sim/test_decode.py tests/learn/test_mask.py -vv`

Expected: PASS with bit-equal outputs.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/sim/observe.py src/kaggriculture/sim/legality.py src/kaggriculture/sim/decode.py tests/sim/test_io_vectorized.py tests/sim/test_observe.py tests/sim/test_legality.py tests/sim/test_decode.py
git commit -m "feat: vectorize simulator observations and actions"
```

---

### Task 8: Integrate the in-place engine and direct Kaggle differential suite

**Files:**

- Modify: `src/kaggriculture/sim/engine.py`
- Create: `tests/sim/test_engine_vectorized.py`
- Modify: `tests/sim/test_episodes.py`
- Modify: `tests/sim/test_determinism.py`
- Modify: `tests/sim/test_cuda.py`
- Modify: `tests/sim/test_hot_path.py`

**Interfaces:**

- Produces: `Simulator.create(batch, device, config=Config()) -> Simulator`.
- Produces: `Simulator.reset(seeds: Tensor) -> SimState` and `Simulator.step(unit_actions, market_actions) -> SimState`.
- `Simulator` owns `state`, `tables`, and `workspace`; calls mutate the owned state.

- [ ] **Step 1: Write a failing complete-turn reference test**

```python
@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)],
)
def test_complete_turn_matches_kaggle_after_every_phase(device: str) -> None:
    harness = DifferentialHarness(seeds=[241, 251], device=device)
    for actions in adversarial_action_sequence(turns=48):
        harness.step(actions)
        harness.assert_identical()
```

- [ ] **Step 2: Replace the obsolete mandatory graph-capture test**

Remove `test_cuda_hot_path_is_graph_capturable`. Add:

```python
def test_step_runs_under_inference_mode_without_grad_history() -> None:
    simulator = Simulator.create(2, "cuda")
    simulator.reset(torch.tensor([257, 263], device="cuda"))
    state = simulator.step(*pass_actions(2, "cuda"))
    assert all(t.grad_fn is None for t in state.tensors())
```

- [ ] **Step 3: Run and verify RED on return-new-state orchestration**

Run: `.venv/bin/python -m pytest tests/sim/test_engine_vectorized.py tests/sim/test_cuda.py -vv`

Expected: FAIL because `Simulator` and in-place orchestration do not exist.

- [ ] **Step 4: Implement `Simulator` and ordered phase orchestration**

```python
@dataclass
class Simulator:
    state: SimState
    tables: DeviceTables
    workspace: Workspace

    @torch.inference_mode()
    def step(self, unit_actions: torch.Tensor,
             market_actions: MarketActions) -> SimState:
        apply_unit_phases_(self.state, unit_actions, self.tables, self.workspace)
        apply_market_phase_(self.state, market_actions, self.tables, self.workspace)
        apply_day_phases_(self.state, self.tables, self.workspace)
        return self.state
```

- [ ] **Step 5: Run complete CPU and CUDA differential tests**

Run: `.venv/bin/python -m pytest tests/sim/test_engine_vectorized.py tests/sim/test_episodes.py tests/sim/test_determinism.py tests/sim/test_cuda.py tests/sim/test_hot_path.py -vv`

Expected: PASS with exact CPU/CUDA and Kaggle equality after every turn.

- [ ] **Step 6: Commit**

```bash
git add src/kaggriculture/sim/engine.py tests/sim/test_engine_vectorized.py tests/sim/test_episodes.py tests/sim/test_determinism.py tests/sim/test_cuda.py tests/sim/test_hot_path.py
git commit -m "feat: integrate vectorized simulator engine"
```

---

### Task 9: Integrate segmented rollout without autograd or host transfer

**Files:**

- Modify: `src/kaggriculture/sim/rollout.py`
- Create: `tests/sim/test_rollout_vectorized.py`
- Modify: `tests/sim/test_rollout.py`

**Interfaces:**

- Produces: `collect_segment(simulator: Simulator, policy: PolicyFn, length: int, seat: int = 0) -> Trajectory`.
- Preserves: `Trajectory.illegal`, terminal reward, potential telescoping, and on-device tensors.

- [ ] **Step 1: Write failing state-ownership and no-grad tests**

```python
@pytest.mark.cuda
def test_collect_segment_stays_on_device_and_builds_no_environment_graph() -> None:
    simulator = Simulator.create(128, "cuda")
    trajectory = collect_segment(simulator, pass_policy, length=32)
    assert trajectory.illegal == 0
    assert all(t.is_cuda for t in trajectory.tensors())
    assert simulator.state.step.eq(32).all()
    assert trajectory.rewards.grad_fn is None
```

- [ ] **Step 2: Run and verify RED on the old functional-step API**

Run: `.venv/bin/python -m pytest tests/sim/test_rollout_vectorized.py -vv`

Expected: FAIL because `collect_segment` does not accept an owning `Simulator`.

- [ ] **Step 3: Implement segment collection and table-driven potential**

Move potential constants to `DeviceTables`; remove hot-path tensor construction
and `.tolist()`. Keep policy forward behavior unchanged, but detach sampled
actions before the environment step. Store rollout tensors on device.

- [ ] **Step 4: Verify rollout invariants and scripted bridge**

Run: `.venv/bin/python -m pytest tests/sim/test_rollout_vectorized.py tests/sim/test_rollout.py tests/learn/test_rollout.py -vv`

Expected: PASS with zero illegal actions and exact potential/terminal behavior.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/sim/rollout.py tests/sim/test_rollout_vectorized.py tests/sim/test_rollout.py
git commit -m "feat: integrate vectorized simulator rollout"
```

---

### Task 10: Run the fidelity campaign and commit acceptance evidence

**Files:**

- Create: `scripts/sim_fidelity_campaign.py`
- Create: `tests/sim/test_campaign.py`
- Modify: `docs/experiments/fidelity/README.md`
- Create: `docs/experiments/fidelity/2026-08-10-1.32.6-vectorized.json`
- Modify: `.github/workflows/test.yml`

**Interfaces:**

- CLI: `sim_fidelity_campaign.py --episodes N --resume PATH --output PATH --device DEVICE`.
- Output: schema-versioned JSON containing reference identity, commit, seeds, branch counts, comparisons, first divergence, and zero-divergence acceptance flag.

- [ ] **Step 1: Write failing evidence-schema and resume tests**

```python
def test_campaign_record_requires_zero_divergence_and_reference_hashes(tmp_path: Path) -> None:
    record = run_campaign(episodes=2, output=tmp_path / "record.json")
    assert record["reference"]["version"] == "1.32.6"
    assert record["comparisons"] > 0
    assert record["divergences"] == 0
    assert record["accepted"] is False  # fewer than 10,000 episodes
```

- [ ] **Step 2: Run and verify RED**

Run: `.venv/bin/python -m pytest tests/sim/test_campaign.py tests/sim/test_reference_tripwire.py -vv`

Expected: FAIL on the missing vectorized acceptance record integration.

- [ ] **Step 3: Implement the resumable campaign and CI smoke tripwire**

Use deterministic seed partitions for adversarial, random legal,
boundary-biased, and replay actions. Flush progress atomically after each block.
CI runs a 50-episode smoke campaign and checks reference hashes; only the manual
10,000-episode run sets `accepted: true`.

- [ ] **Step 4: Run the full campaign**

Run: `.venv/bin/python scripts/sim_fidelity_campaign.py --episodes 10000 --device cpu --output docs/experiments/fidelity/2026-08-10-1.32.6-vectorized.json`

Expected: `divergences: 0`, all required branch outcomes covered, `accepted: true`.

- [ ] **Step 5: Verify the evidence tripwire**

Run: `.venv/bin/python -m pytest tests/sim/test_reference_tripwire.py tests/sim/test_episodes.py -vv`

Expected: PASS against the committed record.

- [ ] **Step 6: Commit**

```bash
git add scripts/sim_fidelity_campaign.py docs/experiments/fidelity/README.md docs/experiments/fidelity/2026-08-10-1.32.6-vectorized.json .github/workflows/test.yml tests/sim/test_campaign.py tests/sim/test_reference_tripwire.py tests/sim/test_episodes.py
git commit -m "test: accept vectorized simulator fidelity"
```

---

### Task 11: Benchmark eager GPU performance and record profiler evidence

**Files:**

- Modify: `scripts/benchmark_simulator.py`
- Create: `tests/sim/test_benchmark.py`
- Create: `docs/experiments/fidelity/2026-08-10-vectorized-throughput.json`
- Modify: `docs/experiments/fidelity/README.md`

**Interfaces:**

- CLI: `benchmark_simulator.py --batches 64 128 256 1024 --iterations 200 --device cuda --output PATH`.
- Output: eager latency percentiles, environment-steps/s, trajectories/hour, peak memory, reset/table setup time, hardware, versions, and per-stage operator summaries.

- [ ] **Step 1: Write failing benchmark-schema and eager-gate tests**

```python
def test_benchmark_record_has_required_eager_rows(record: dict[str, object]) -> None:
    rows = {(row["batch"], row["mode"]): row for row in record["rows"]}
    for batch in (64, 128, 256, 1024):
        assert (batch, "eager") in rows
    assert rows[(1024, "eager")]["environment_steps_per_second"] >= 110_000
```

- [ ] **Step 2: Run and verify RED against the old graph-first benchmark**

Run: `.venv/bin/python -m pytest tests/sim/test_benchmark.py -vv`

Expected: FAIL because the benchmark does not emit the required schema, percentiles, memory, or stage summaries.

- [ ] **Step 3: Implement synchronized eager timing and stage profiling**

Warm tables and state outside timing. Use CUDA events or synchronized
`perf_counter` intervals, reset peak-memory stats per row, and record separate
decode, observe/legal, units, market, day, and complete-step profiles. Do not
include optional compile/graph rows unless eager already passes.

- [ ] **Step 4: Run the benchmark matrix on the less-loaded GPU**

Run: `.venv/bin/python scripts/benchmark_simulator.py --batches 64 128 256 1024 --iterations 200 --device cuda --output docs/experiments/fidelity/2026-08-10-vectorized-throughput.json`

Expected: the eager batch-1,024 row is at least 110,000 environment-steps/s.

- [ ] **Step 5: Run the complete verification suite**

Run: `.venv/bin/python -m pytest -m 'not slow'`

Expected: PASS with no mandatory CUDA-graph test.

Run: `.venv/bin/python -m pytest tests/sim --sim-episodes=100 --sim-turns=719 -vv`

Expected: PASS with zero direct-reference divergences.

- [ ] **Step 6: Commit**

```bash
git add scripts/benchmark_simulator.py tests/sim/test_benchmark.py docs/experiments/fidelity/2026-08-10-vectorized-throughput.json docs/experiments/fidelity/README.md
git commit -m "perf: accept vectorized simulator throughput"
```

---

### Task 12: Final requirement audit

**Files:**

- Modify only if evidence links or commands are inaccurate: `docs/experiments/fidelity/README.md`

**Interfaces:**

- Consumes all prior deliverables; produces no new runtime API.

- [ ] **Step 1: Audit every design requirement against evidence**

Create a temporary checklist mapping design sections 2–11 to a test, campaign
field, profiler row, or source invariant. Fail the audit if a requirement has no
evidence; add the missing test to the task that owns the requirement rather than
waiving it here.

- [ ] **Step 2: Verify repository status and the complete suite**

Run: `git diff --check`

Expected: no whitespace errors.

Run: `.venv/bin/python -m pytest -m 'not slow'`

Expected: PASS.

- [ ] **Step 3: Verify acceptance records**

Run: `.venv/bin/python -c "import json; f=json.load(open('docs/experiments/fidelity/2026-08-10-1.32.6-vectorized.json')); p=json.load(open('docs/experiments/fidelity/2026-08-10-vectorized-throughput.json')); assert f['accepted'] and f['divergences']==0; assert next(r for r in p['rows'] if r['batch']==1024 and r['mode']=='eager')['environment_steps_per_second'] >= 110000"`

Expected: exit 0.

- [ ] **Step 4: Commit any evidence-link correction**

```bash
git add docs/experiments/fidelity/README.md
git commit -m "docs: finalize simulator acceptance evidence"
```
