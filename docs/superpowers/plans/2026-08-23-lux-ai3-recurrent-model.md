# Lux AI3 Recurrent Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the ConvLSTM memory, privileged opponent-belief supervision, spatial transformer, interaction value head, and per-unit local residual patches from the approved port while preserving a disabled-feature control model.

**Architecture:** Extend the Lightning foundation with explicit `PolicyState` and `PolicyOutput` tensor contracts. Rollout and learner both replay the same recurrent state/reset semantics; hidden opponent state appears only as a supervised target, while the current/previous belief prediction is the sole inference feedback.

**Tech Stack:** Python 3.11+, PyTorch 2.13+, Lightning 2.6.5+, Pydantic 2.12+, pytest.

**Spec:** `docs/superpowers/specs/2026-08-23-lux-ai3-native-port-design.md`

## Global Constraints

- Complete `docs/superpowers/plans/2026-08-23-lux-ai3-lightning-foundation.md` first.
- Board size remains 10x10; action dimensions come from `learn/encoding.py`.
- The control configuration with recurrence, transformer, belief, local patch, and interaction value disabled must retain foundation numerical parity.
- Recurrent state is reset by `dones` before processing the next observation and detached at 16-turn segment boundaries.
- Opponent private state is a training target only; it must never enter the inference observation.
- The default adapted dimensions are 128 spatial channels and a padded 7x7 local unit patch.
- Masked/padded unit slots contribute to no action, entropy, or belief loss.

## File Structure

- Create `src/kaggriculture/learn/toad/model.py`: state/output containers and all optional model components.
- Create `THIRD_PARTY_NOTICES.md`: upstream MIT copyright, permission notice, repository URL, inspected commit, and adapted files.
- Modify `src/kaggriculture/learn/toad/config.py`: validated recurrent, belief, transformer, local-patch, and value settings.
- Modify `src/kaggriculture/learn/encoding.py`: privileged opponent-private target encoding only.
- Modify `src/kaggriculture/learn/rollout.py`: actor state, target capture, and trajectory fields.
- Modify `src/kaggriculture/learn/toad/data.py`: segment initial states and target masks.
- Modify `src/kaggriculture/learn/toad/lightning.py`: stateful forward and belief loss.
- Create `tests/learn/test_toad_model.py` and `test_toad_belief.py`.
- Modify `tests/learn/test_rollout.py`, `test_toad_data.py`, and `test_toad_lightning.py`.

---

### Task 1: Introduce stable model input/output and control compatibility

**Files:**

- Create: `src/kaggriculture/learn/toad/model.py`
- Create: `THIRD_PARTY_NOTICES.md`
- Modify: `src/kaggriculture/learn/toad/config.py`
- Create: `tests/learn/test_toad_model.py`

**Interfaces:**

- Consumes: current `learn.model.Policy` and `ModelConfig`.
- Produces: `PolicyState`, `PolicyOutput`, `StatefulPolicy.initial_state`, and `StatefulPolicy.forward`.

- [ ] **Step 1: Write failing control-adapter tests**

```python
def test_disabled_features_match_the_current_policy() -> None:
    torch.manual_seed(11)
    old = Policy(blocks=1, channels=16, value_bound=1.0)
    new = StatefulPolicy(ModelConfig.control(blocks=1, channels=16))
    new.load_control_state_dict(old.state_dict())
    board, scalars, positions = model_inputs(batch=2)

    old_output = old(board, scalars, positions)
    new_output = new(board, scalars, positions, state=None, dones=None)

    assert torch.equal(new_output.unit_logits, old_output[0])
    assert torch.equal(new_output.quantity_logits, old_output[1])
    assert torch.equal(new_output.market_logits, old_output[2])
    assert torch.equal(new_output.values, old_output[3])
    assert new_output.state is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/learn/test_toad_model.py::test_disabled_features_match_the_current_policy -v`

Expected: FAIL because `StatefulPolicy` does not exist.

- [ ] **Step 3: Implement typed tensor containers and control delegation**

```python
@dataclass(frozen=True)
class PolicyState:
    hidden: torch.Tensor
    cell: torch.Tensor
    prior_belief: torch.Tensor

    def detach(self) -> "PolicyState":
        return PolicyState(
            hidden=self.hidden.detach(),
            cell=self.cell.detach(),
            prior_belief=self.prior_belief.detach(),
        )


@dataclass(frozen=True)
class PolicyOutput:
    unit_logits: torch.Tensor
    quantity_logits: torch.Tensor
    market_logits: torch.Tensor
    values: torch.Tensor
    belief_logits: torch.Tensor | None
    state: PolicyState | None


class StatefulPolicy(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.control = Policy(config.blocks, config.channels, config.value_bound)

    def initial_state(self, batch: int, *, like: torch.Tensor) -> PolicyState | None:
        if not (self.config.recurrent or self.config.belief):
            return None
        return PolicyState(
            hidden=like.new_zeros(batch, self.config.recurrent_channels, 10, 10),
            cell=like.new_zeros(batch, self.config.recurrent_channels, 10, 10),
            prior_belief=like.new_zeros(batch, self.config.belief_size),
        )

    def forward(self, board, scalars, positions, state=None, dones=None) -> PolicyOutput:
        unit, quantity, market, values = self.control(board, scalars, positions)
        return PolicyOutput(unit, quantity, market, values, None, None)
```

Expose `ModelConfig.control(blocks=8, channels=128)` as the single way to disable every new component.

Extend `ModelConfig` with validated concrete dimensions:

```python
recurrent_channels: PositiveInt = 128
recurrent_kernel_size: Literal[3, 5] = 3
belief_size: PositiveInt = len(SHED_NAMES) + len(CROP_NAMES) + len(PRODUCT_NAMES)
belief_loss_weight: NonNegativeFloat = 0.0
belief_feedback: bool = False
transformer_blocks: NonNegativeInt = 0
transformer_heads: PositiveInt = 4
transformer_mlp_ratio: PositiveInt = 2
local_patch_size: PositiveInt = 7
local_patch_blocks: NonNegativeInt = 0
```

The model validator requires odd `local_patch_size` and `channels % transformer_heads == 0` whenever the transformer is enabled.

```python
@model_validator(mode="after")
def validate_optional_model_paths(self) -> Self:
    if (self.belief_feedback or self.belief_loss_weight > 0) and not self.belief:
        raise ValueError("belief feedback/loss requires the belief head")
    if self.local_patch and self.local_patch_size % 2 == 0:
        raise ValueError("local patch size must be odd")
    if self.transformer and self.channels % self.transformer_heads:
        raise ValueError("transformer channels must divide evenly across heads")
    return self
```

Add the upstream MIT text and provenance for `lux_ai/nns/models.py` and `lux_ai/nns/transformer.py` to `THIRD_PARTY_NOTICES.md`; add the inspected commit to the new model module's top-level docstring.

- [ ] **Step 4: Run control and checkpoint compatibility tests**

Run: `uv run pytest tests/learn/test_toad_model.py tests/learn/test_toad_control_fixture.py tests/learn/test_toad_checkpoint.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add THIRD_PARTY_NOTICES.md src/kaggriculture/learn/toad/model.py src/kaggriculture/learn/toad/config.py tests/learn/test_toad_model.py
git commit -m "feat: define stateful Toad policy contract"
```

### Task 2: Add ConvLSTM with exact terminal resets

**Files:**

- Modify: `src/kaggriculture/learn/toad/model.py`
- Modify: `src/kaggriculture/learn/toad/config.py`
- Modify: `tests/learn/test_toad_model.py`

**Interfaces:**

- Consumes: sequence tensors shaped `(time, batch, channels, 10, 10)`.
- Produces: `ConvLSTMCell`, `ConvLSTM.forward(x, state, dones)`, and recurrent `StatefulPolicy` output.

- [ ] **Step 1: Write terminal-reset and chunk-equivalence tests**

```python
def test_terminal_reset_erases_previous_episode_memory() -> None:
    layer = ConvLSTM(input_channels=8, hidden_channels=8, kernel_size=3)
    first = torch.randn(3, 2, 8, 10, 10)
    second = torch.randn(2, 2, 8, 10, 10)
    state = layer.initial_state(batch=2, height=10, width=10, like=first)
    _, carried = layer(first, state, torch.zeros(3, 2, dtype=torch.bool))
    reset = torch.tensor([[True, False], [False, False]])
    output, _ = layer(second, carried, reset)
    fresh, _ = layer(second[:, :1], None, reset[:, :1])
    assert torch.allclose(output[:, :1], fresh, atol=1e-6, rtol=1e-6)


def test_two_chunks_equal_one_sequence_without_a_terminal() -> None:
    full, _ = layer(torch.cat([a, b]), None, all_false_dones)
    left, state = layer(a, None, all_false_dones[: len(a)])
    right, _ = layer(b, state, all_false_dones[len(a) :])
    assert torch.allclose(torch.cat([left, right]), full, atol=1e-6, rtol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/learn/test_toad_model.py -k "terminal_reset or two_chunks" -v`

Expected: FAIL because `ConvLSTM` is undefined.

- [ ] **Step 3: Implement the cell and sequence loop**

```python
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, x: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        i, f, o, g = self.gates(torch.cat((x, hidden), dim=1)).chunk(4, dim=1)
        cell = torch.sigmoid(f) * cell + torch.sigmoid(i) * torch.tanh(g)
        hidden = torch.sigmoid(o) * torch.tanh(cell)
        return hidden, cell
```

For every time row, multiply hidden/cell/prior-belief by `(~dones[t]).view(batch, 1, 1, 1)` before applying the cell. A `None` state allocates zeros from `x.new_zeros`.

- [ ] **Step 4: Integrate after the residual trunk**

Reshape flattened foundation batches back to time-major before ConvLSTM, merge `trunk`, `hidden`, and `cell` using a 1x1 convolution, then flatten only for the existing heads. Do not loop recurrence over a flattened time/batch axis.

```python
time, batch = board.shape[:2]
features = self.encode(board.flatten(0, 1), scalars.flatten(0, 1))
features = features.view(time, batch, self.config.channels, 10, 10)
recurrent, next_state = self.recurrent(features, state, dones)
merged = self.merge(
    torch.cat((features, recurrent.hidden_sequence, recurrent.cell_sequence), dim=2).flatten(0, 1)
).view(time, batch, self.config.channels, 10, 10)
```

Run: `uv run pytest tests/learn/test_toad_model.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/learn/toad/model.py src/kaggriculture/learn/toad/config.py tests/learn/test_toad_model.py
git commit -m "feat: add terminal-aware ConvLSTM policy memory"
```

### Task 3: Carry recurrent state through rollout and segments

**Files:**

- Modify: `src/kaggriculture/learn/rollout.py:194-328,329-503,671-840`
- Modify: `src/kaggriculture/learn/toad/data.py`
- Modify: `tests/learn/test_rollout.py`
- Modify: `tests/learn/test_toad_data.py`

**Interfaces:**

- Consumes: `StatefulPolicy`, `PolicyState`, and full episode trajectories.
- Produces: `RecordedPolicyState`, recurrent fields on `Trajectory`, and `LearnerBatch.initial_state`.

- [ ] **Step 1: Write failing rollout-state tests**

```python
def test_segments_store_the_state_before_their_first_observation() -> None:
    trajectory = recurrent_trajectory(turns=32, hidden_channels=4)
    batches = segments(trajectory, unroll_length=16)
    assert torch.equal(batches[0]["initial_hidden"], trajectory.hidden[0])
    assert torch.equal(batches[1]["initial_hidden"], trajectory.hidden[16])


def test_segment_initial_state_is_detached() -> None:
    segment = segments(recurrent_trajectory(), unroll_length=16)[0]
    assert not segment["initial_hidden"].requires_grad
    assert not segment["initial_cell"].requires_grad
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/learn/test_rollout.py tests/learn/test_toad_data.py -k "state or recurrent" -v`

Expected: FAIL because trajectories have no recurrent fields.

- [ ] **Step 3: Record pre-action state in the reference actor**

```python
@dataclass(frozen=True)
class RecordedPolicyState:
    hidden: torch.Tensor
    cell: torch.Tensor
    prior_belief: torch.Tensor


def _decide(policy, encoded, state, done):
    output = policy(*encoded, state=state, dones=done)
    action = sample_existing_heads(output)
    return action, output.state.detach() if output.state is not None else None
```

`Stream` appends the state used for each action before replacing it with the returned next state. Store one extra trailing recurrent state beside the extra bootstrap observation.

- [ ] **Step 4: Add segment initial state and learner replay**

`segments` copies only the state at `start`, not all actor hidden states. `compute_loss` passes that initial state plus the segment's time-major `dones` through the learner and ignores the actor's later hidden states.

```python
segment["initial_hidden"] = trajectory.hidden[start].detach()
segment["initial_cell"] = trajectory.cell[start].detach()
segment["initial_belief"] = trajectory.prior_belief[start].detach()
initial = PolicyState(
    hidden=stacked("initial_hidden"),
    cell=stacked("initial_cell"),
    prior_belief=stacked("initial_belief"),
)
output = policy(board, scalars, positions, state=initial, dones=dones)
```

Run: `uv run pytest tests/learn/test_rollout.py tests/learn/test_toad_data.py tests/learn/test_toad_lightning.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/learn/rollout.py src/kaggriculture/learn/toad/data.py tests/learn/test_rollout.py tests/learn/test_toad_data.py tests/learn/test_toad_lightning.py
git commit -m "feat: replay recurrent state across Toad segments"
```

### Task 4: Add privileged opponent-belief targets without leakage

**Files:**

- Modify: `src/kaggriculture/learn/encoding.py`
- Modify: `src/kaggriculture/learn/rollout.py`
- Modify: `src/kaggriculture/learn/toad/model.py`
- Modify: `src/kaggriculture/learn/toad/lightning.py`
- Create: `tests/learn/test_toad_belief.py`

**Interfaces:**

- Consumes: the opponent seat's own private observation during collection.
- Produces: `encode_private_belief_target(observation)`, belief target tensors/masks, `belief_logits`, and `loss/belief`.

- [ ] **Step 1: Write target and anti-leakage tests**

```python
def test_belief_target_encodes_opponent_private_state() -> None:
    observation = private_observation(shed_wheat=7, seed_wheat=3, carried_wheat=4)
    target = encode_private_belief_target(observation)
    assert target.shed[SHED_NAMES.index("WHEAT")] == pytest.approx(7 / SHED_CAPACITY)
    assert target.seeds[CROP_NAMES.index("WHEAT")] == pytest.approx(3 / SEED_SCALE)
    assert target.carried[PRODUCT_NAMES.index("WHEAT")] == pytest.approx(4 / CARRIED_SCALE)


def test_actor_input_is_unchanged_when_only_privileged_target_changes() -> None:
    public, own_private = inference_observation()
    first = collect_encoded_input(public, own_private, opponent_target_a)
    second = collect_encoded_input(public, own_private, opponent_target_b)
    assert_tree_equal(first.policy_input, second.policy_input)
    assert not torch.equal(first.belief_target, second.belief_target)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/learn/test_toad_belief.py -v`

Expected: FAIL because the belief target encoder does not exist.

- [ ] **Step 3: Implement normalized target encoding and rollout capture**

```python
@dataclass(frozen=True)
class BeliefTarget:
    shed: torch.Tensor
    seeds: torch.Tensor
    carried: torch.Tensor


def encode_private_belief_target(observation: Mapping[str, object]) -> BeliefTarget:
    private = observation["private"]
    carried = Counter()
    for inventory in private["inventories"]:
        carried.update(inventory)
    return BeliefTarget(
        shed=torch.tensor([private["shed"][name] / SHED_CAPACITY for name in SHED_NAMES]),
        seeds=torch.tensor([private["seeds"][name] / SEED_SCALE for name in CROP_NAMES]),
        carried=torch.tensor([carried[name] / CARRIED_SCALE for name in PRODUCT_NAMES]),
    )
```

The reference collector obtains this from the opponent seat's agent observation. It stores the target separately from `board`, `scalars`, and `positions`, with a boolean validity mask.

- [ ] **Step 4: Add prediction, feedback, and supervised loss**

Project the prior prediction through the scalar branch, emit current belief logits from the merged recurrent/spatial representation, and store the current prediction as next state's `prior_belief`. Use masked Smooth L1 for normalized counts:

```python
belief_error = F.smooth_l1_loss(output.belief_logits, batch.belief_targets, reduction="none")
belief_loss = (belief_error * batch.belief_valid).sum() / batch.belief_valid.sum().clamp_min(1)
total = rl_loss.total + config.model.belief_loss_weight * belief_loss
```

Run: `uv run pytest tests/learn/test_toad_belief.py tests/learn/test_toad_lightning.py tests/learn/test_rollout.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/learn/encoding.py src/kaggriculture/learn/rollout.py src/kaggriculture/learn/toad/model.py src/kaggriculture/learn/toad/lightning.py tests/learn/test_toad_belief.py tests/learn/test_toad_lightning.py tests/learn/test_rollout.py
git commit -m "feat: supervise opponent private-state beliefs"
```

### Task 5: Add transformer and interaction-aware value

**Files:**

- Modify: `src/kaggriculture/learn/toad/model.py`
- Modify: `src/kaggriculture/learn/toad/config.py`
- Modify: `tests/learn/test_toad_model.py`

**Interfaces:**

- Consumes: merged 10x10 recurrent feature maps and projected global context.
- Produces: `SpatialTransformer`, `InteractionValueHead`, and enabled forward path.

- [ ] **Step 1: Write spatial-shape and interaction tests**

```python
def test_spatial_transformer_preserves_map_shape() -> None:
    layer = SpatialTransformer(channels=32, blocks=2, heads=4, mlp_ratio=2)
    x = torch.randn(3, 32, 10, 10)
    assert layer(x).shape == x.shape


def test_value_changes_when_only_remote_opponent_context_changes() -> None:
    model = StatefulPolicy(interaction_value_config())
    left = encoded_position(opponent_tile_value=0.0)
    right = encoded_position(opponent_tile_value=1.0)
    assert not torch.equal(model(*left).values, model(*right).values)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/learn/test_toad_model.py -k "transformer or remote_opponent" -v`

Expected: FAIL because the components are undefined.

- [ ] **Step 3: Implement spatial attention with explicit positional embeddings**

Flatten `(batch, channels, 10, 10)` to `(batch, 100, channels)`, add a learned `(1, 100, channels)` position tensor, apply pre-norm `MultiheadAttention` plus MLP residual blocks, and reshape back. Validate `channels % heads == 0` in Pydantic.

```python
tokens = x.flatten(2).transpose(1, 2) + self.position
for norm1, attention, norm2, mlp in self.blocks:
    normalized = norm1(tokens)
    attended, _ = attention(normalized, normalized, normalized, need_weights=False)
    tokens = tokens + attended
    tokens = tokens + mlp(norm2(tokens))
return tokens.transpose(1, 2).reshape(x.shape)
```

- [ ] **Step 4: Implement the interaction value readout**

Use one learned value token attending over 100 spatial tokens plus one projected global token, then map the value token to a scalar and apply the existing optional value bound. Do not mean-pool before attention.

```python
spatial = features.flatten(2).transpose(1, 2)
global_token = self.global_projection(scalars).unsqueeze(1)
value_token = self.value_token.expand(features.shape[0], -1, -1)
tokens = torch.cat((value_token, global_token, spatial), dim=1)
attended = self.value_transformer(tokens)
value = self.value_projection(attended[:, 0]).squeeze(-1)
return bound_value(value, self.value_bound)
```

Run: `uv run pytest tests/learn/test_toad_model.py tests/learn/test_toad_lightning.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/learn/toad/model.py src/kaggriculture/learn/toad/config.py tests/learn/test_toad_model.py tests/learn/test_toad_lightning.py
git commit -m "feat: add spatial attention and interaction value"
```

### Task 6: Add padded per-unit local residual patches

**Files:**

- Modify: `src/kaggriculture/learn/toad/model.py`
- Modify: `src/kaggriculture/learn/toad/config.py`
- Modify: `tests/learn/test_toad_model.py`

**Interfaces:**

- Consumes: spatial features and flattened unit positions.
- Produces: `extract_unit_patches(features, positions, size)` and `LocalUnitHead`.

- [ ] **Step 1: Write patch centering, edge, and padding tests**

```python
def test_unit_patch_is_centered_on_the_flat_position() -> None:
    features = numbered_board(channels=1)
    positions = torch.tensor([[5 * 10 + 4]])
    patch = extract_unit_patches(features, positions, size=7)
    assert patch[0, 0, 0, 3, 3] == features[0, 0, 5, 4]


def test_edge_patch_marks_out_of_bounds_cells() -> None:
    patch = extract_unit_patches(torch.ones(1, 2, 10, 10), torch.tensor([[0]]), size=7)
    out_of_bounds = patch[0, 0, -1]
    assert out_of_bounds[:3, :3].all()
    assert not out_of_bounds[3:, 3:].any()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/learn/test_toad_model.py -k "patch" -v`

Expected: FAIL because patch extraction is undefined.

- [ ] **Step 3: Implement integer-index patch extraction**

Pad spatial features by `size // 2`, append a padding-indicator plane, convert flat positions with `y = position // 10` and `x = position % 10`, and gather each 7x7 window without normalized `grid_sample` coordinates.

```python
radius = size // 2
spatial = F.pad(features, (radius, radius, radius, radius))
indicator = F.pad(
    features.new_zeros(features.shape[0], 1, 10, 10),
    (radius, radius, radius, radius),
    value=1,
)
padded = torch.cat((spatial, indicator), dim=1)
windows = F.unfold(padded, kernel_size=size)
indices = positions[:, None, :].expand(-1, windows.shape[1], -1)
selected = windows.gather(2, indices).transpose(1, 2)
return selected.reshape(features.shape[0], positions.shape[1], -1, size, size)
```

- [ ] **Step 4: Add residual local heads and control fallback**

The local operation and quantity heads share the extracted patch input but have separate final projections. The operation head pools the processed patch to `len(UNIT_OPS)`; the quantity head pools to `len(QUANTITIES)`. When disabled, retain the exact gathered-column control heads.

```python
patches = extract_unit_patches(features, positions, self.patch_size)
local = self.local_blocks(patches.flatten(0, 1))
pooled = local.mean(dim=(-2, -1)).view(features.shape[0], positions.shape[1], -1)
unit_logits = self.local_operation(pooled)
quantity_logits = self.local_quantity(pooled)
```

Run: `uv run pytest tests/learn/test_toad_model.py tests/learn/test_toad_control_fixture.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/learn/toad/model.py src/kaggriculture/learn/toad/config.py tests/learn/test_toad_model.py
git commit -m "feat: add local residual unit readouts"
```

### Task 7: Run the recurrent-model integration gate

**Files:**

- Modify: `tests/learn/test_toad_lightning.py`
- Modify: `tests/learn/test_toad_checkpoint.py`
- Create: `tests/learn/test_toad_model_integration.py`

**Interfaces:**

- Consumes: the complete recurrent model and Lightning foundation.
- Produces: an end-to-end recurrent checkpoint/resume and short-training proof.

- [ ] **Step 1: Add a two-segment stateful training test**

Build a 32-turn synthetic trajectory with a terminal between two episodes, train two optimizer batches, and assert finite losses, nonzero gradients in every enabled component, and zero state carry into the second episode.

```python
def test_two_segment_training_resets_and_updates_every_component() -> None:
    module, batches = recurrent_training_fixture(turns=32, terminal_at=16)
    reports = [module.compute_report(batch) for batch in batches]
    assert all(torch.isfinite(report.total) for report in reports)
    sum(report.total for report in reports).backward()
    assert all(parameter.grad is not None for parameter in enabled_parameters(module))
    assert torch.count_nonzero(module.debug_state_after_reset.hidden) == 0
```

- [ ] **Step 2: Add checkpoint round-trip assertions**

Save after the first collection boundary, resume, and assert that the next batch begins with the same initial hidden/cell/prior-belief tensors and produces the same update as uninterrupted training.

```python
def test_recurrent_boundary_resume_matches_uninterrupted(tmp_path: Path) -> None:
    uninterrupted, resumed, next_batch = recurrent_resume_fixture(tmp_path)
    assert_tree_equal(resumed.next_initial_state, uninterrupted.next_initial_state)
    uninterrupted.fit_one(next_batch)
    resumed.fit_one(next_batch)
    assert_state_dict_equal(resumed.policy, uninterrupted.policy)
```

- [ ] **Step 3: Run focused integration tests**

Run: `uv run pytest tests/learn/test_toad_model_integration.py tests/learn/test_toad_checkpoint.py -v`

Expected: PASS.

- [ ] **Step 4: Run control and full-suite gates**

Run: `uv run pytest tests/learn/test_toad_control_fixture.py -v`

Expected: PASS with every new feature disabled.

Run: `uv run pre-commit run -a`

Expected: all hooks PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/learn/test_toad_model_integration.py tests/learn/test_toad_lightning.py tests/learn/test_toad_checkpoint.py
git commit -m "test: gate the recurrent Toad model"
```

## Completion Gate

This plan is complete only when the disabled-feature control still matches, actor and learner recurrence agree across segment boundaries, terminal resets are exact, belief targets cannot influence inference inputs, every enabled component receives finite gradients, and recurrent checkpoint/resume reproduces the next update.
