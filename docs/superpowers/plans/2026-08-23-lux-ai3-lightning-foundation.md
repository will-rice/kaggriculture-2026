# Lux AI3 Lightning Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the active Toad runner's untyped orchestration with a Pydantic-configured, single-device Lightning control path that is numerically equivalent when all new Lux-derived features are disabled.

**Architecture:** Preserve the existing `Policy`, masks, rollout, `_segments` semantics, and V-trace/UPGO/TD(lambda) math. Move one optimizer-sized group of four 16-turn segments into one Lightning batch, let Lightning own automatic optimization, and retain collection-round markers for scheduler, actor synchronization, metrics, stopping, and checkpoints.

**Tech Stack:** Python 3.11+, Pydantic 2.12+, PyTorch 2.13+, Lightning 2.6.5+, pytest, W&B.

**Spec:** `docs/superpowers/specs/2026-08-23-lux-ai3-native-port-design.md`

## Global Constraints

- This plan implements only Stages 0-2 of the spec; ConvLSTM, belief, transformers, population snapshots, adaptive entropy, BF16, compile, DDP, and native rollout remain disabled.
- `src/kaggriculture/learn/toad_loss.py` remains the authoritative V-trace/UPGO/TD(lambda) implementation.
- The control configuration must retain four 16-turn segments per optimizer step, the current value-warmup semantics, extra value passes, actor lag, and one LR step per collection round.
- Pydantic is the only serializable experiment configuration; tensor batches and model outputs use dataclasses.
- `RuntimeConfig.precision` is fixed to `"32-true"` and `devices` to one for this plan.
- No Lux environment, observation, action, runner, Hydra, shared-memory queue, or C++ code is copied.
- Any substantially translated upstream source must carry the MIT notice and exact upstream commit provenance before merge.

## File Structure

- Create `src/kaggriculture/learn/toad/__init__.py`: public native Toad interfaces.
- Create `src/kaggriculture/learn/toad/config.py`: nested Pydantic config, loading, dotted overrides, and fingerprints.
- Create `src/kaggriculture/learn/toad/data.py`: typed learner batches, trajectory segmentation, collection-round expansion, and synchronous reference DataModule.
- Create `src/kaggriculture/learn/toad/lightning.py`: loss adapter, LightningModule, optimizer, scheduler, logging, and checkpoint extensions.
- Create `src/kaggriculture/learn/toad/callbacks.py`: actor publication, environment-step stop, and boundary checkpoint callbacks.
- Modify `.pre-commit-config.yaml` and `pyproject.toml`: keep only legacy vendored Toad files excluded; lint/type-check every new native module.
- Modify `src/kaggriculture/learn/scripts/toad.py`: retain compatibility helpers temporarily but reduce `main` to config resolution plus `Trainer.fit`.
- Modify `src/kaggriculture/learn/scripts/curriculum.py`: construct validated phase overrides instead of argparse flag lists.
- Create `tests/learn/test_toad_config.py`, `test_toad_data.py`, `test_toad_lightning.py`, and `test_toad_callbacks.py`.
- Modify existing Toad runner, checkpoint, and curriculum tests to exercise the public native interfaces.

---

### Task 1: Characterize the current optimizer boundary

**Files:**

- Create: `tests/learn/fixtures/toad_control_batch.pt`
- Create: `tests/learn/test_toad_control_fixture.py`
- Modify: `tests/learn/test_toad_runner.py:39-133`

**Interfaces:**

- Consumes: `toad._segments`, `toad._step`, `Policy`, and current `Trajectory`.
- Produces: `load_control_fixture() -> dict[str, object]`, a deterministic fixture used to test later tasks.

- [ ] **Step 1: Write the deterministic fixture test**

```python
def test_control_fixture_pins_one_optimizer_step() -> None:
    fixture = load_control_fixture()
    policy = Policy(blocks=1, channels=16, value_bound=toad.VALUE_BOUND)
    policy.load_state_dict(fixture["initial_model"])
    optimizer = toad._optimizer(policy, fixture["lr"])
    optimizer.load_state_dict(fixture["initial_optimizer"])

    terms = toad._step(
        policy,
        optimizer,
        fixture["segments"],
        "cpu",
        "shaped_money",
        entropy_cost=fixture["entropy_cost"],
        discounting=fixture["gamma"],
        lmb=fixture["lmb"],
    )

    assert terms == pytest.approx(fixture["terms"], rel=1e-6, abs=1e-7)
    for name, value in policy.state_dict().items():
        assert torch.equal(value, fixture["updated_model"][name])
```

- [ ] **Step 2: Run the test before generating the fixture**

Run: `uv run pytest tests/learn/test_toad_control_fixture.py -v`

Expected: FAIL because `toad_control_batch.pt` does not exist.

- [ ] **Step 3: Add a one-shot fixture generator and commit its output**

```python
def write_control_fixture(path: Path) -> None:
    torch.manual_seed(20260823)
    policy = Policy(blocks=1, channels=16, value_bound=toad.VALUE_BOUND)
    optimizer = toad._optimizer(policy, LEARNING_RATE)
    segments = [_segment() for _ in range(toad.BATCH_SEGMENTS)]
    initial_model = copy.deepcopy(policy.state_dict())
    initial_optimizer = copy.deepcopy(optimizer.state_dict())
    terms = toad._step(policy, optimizer, segments, "cpu", "shaped_money")
    torch.save(
        {
            "segments": segments,
            "initial_model": initial_model,
            "initial_optimizer": initial_optimizer,
            "updated_model": policy.state_dict(),
            "terms": terms,
            "lr": LEARNING_RATE,
            "entropy_cost": ENTROPY_COST,
            "gamma": DISCOUNTING,
            "lmb": LMB,
        },
        path,
    )
```

Run the generator once from the test module, then remove the invocation so the fixture cannot rewrite itself during tests.

- [ ] **Step 4: Pin segmentation and full-round clocks**

Add assertions that one current collection round produces `_batches_per_update(econ_fraction)` policy batches, that extra value passes do not increment collected steps, and that `_decay` advances once per round.

```python
def test_control_fixture_pins_round_clocks() -> None:
    fixture = load_control_fixture()
    assert fixture["policy_batches"] == toad._batches_per_update(0.5)
    assert fixture["collected_steps_with_value_passes"] == fixture["collected_steps"]
    assert fixture["scheduler_steps"] == fixture["collection_rounds"]
```

Run: `uv run pytest tests/learn/test_toad_control_fixture.py tests/learn/test_toad_checkpoint.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/learn/fixtures/toad_control_batch.pt tests/learn/test_toad_control_fixture.py tests/learn/test_toad_runner.py
git commit -m "test: characterize the Toad optimizer boundary"
```

### Task 2: Add the validated Pydantic experiment contract

**Files:**

- Create: `src/kaggriculture/learn/toad/__init__.py`
- Create: `src/kaggriculture/learn/toad/config.py`
- Create: `tests/learn/test_toad_config.py`
- Modify: `.pre-commit-config.yaml`
- Modify: `pyproject.toml:104-119`

**Interfaces:**

- Consumes: constants from `learn/model.py`, `learn/toad_loss.py`, and `learn/scripts/toad.py`.
- Produces: `ToadConfig`, its five nested config models, `load_config(path, overrides)`, `apply_overrides(config, overrides)`, and `structural_fingerprint(config)`.

- [ ] **Step 1: Write failing config tests**

```python
def test_control_config_reproduces_current_constants() -> None:
    config = ToadConfig.control()
    assert config.model.blocks == toad.BLOCKS
    assert config.model.channels == toad.CHANNELS
    assert config.optimizer.unroll_length == UNROLL_LENGTH
    assert config.optimizer.batch_segments == toad.BATCH_SEGMENTS
    assert config.runtime.precision == "32-true"
    assert config.runtime.devices == 1
    assert config.population.selfplay + config.population.scripted == 1.0


def test_batch_probabilities_must_sum_to_one() -> None:
    with pytest.raises(ValidationError, match="sum to one"):
        ToadConfig(population={"selfplay": 0.8, "scripted": 0.8})


def test_teacher_loss_requires_a_teacher() -> None:
    with pytest.raises(ValidationError, match="teacher checkpoint"):
        ToadConfig(optimizer={"teacher_kl_cost": 0.1})


def test_native_toad_modules_are_not_in_the_vendored_exclusion() -> None:
    precommit = Path(".pre-commit-config.yaml").read_text()
    exclude = next(line.strip() for line in precommit.splitlines() if line.startswith("exclude:"))
    assert exclude != "exclude: ^src/kaggriculture/learn/toad/"
    assert "config\\.py" not in exclude
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/learn/test_toad_config.py -v`

Expected: FAIL because `kaggriculture.learn.toad.config` does not exist.

- [ ] **Step 3: Implement the nested models and cross-field validation**

First narrow the vendored exclusions:

```yaml
exclude: ^src/kaggriculture/learn/toad/(conf/|core/|nns/|monobeast\.py$|reward_spaces_lux\.py$)
```

Change the `tool.ty.overrides` include list to the same legacy Python paths (`core/**`, `nns/**`, `monobeast.py`, and `reward_spaces_lux.py`). Do not include the new root-level native modules.

```python
Precision = Literal["32-true", "bf16-mixed"]


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    blocks: PositiveInt = 8
    channels: PositiveInt = 128
    kernel_size: Literal[3, 5] = 3
    activation: Literal["relu", "leaky_relu"] = "relu"
    value_bound: PositiveFloat | None = 1.0
    recurrent: bool = False
    transformer: bool = False
    local_patch: bool = False
    belief: bool = False
    interaction_value: bool = False


class PopulationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    selfplay: float = 0.5
    scripted: float = 0.5
    frozen_opponent: float = 0.0
    teacher_distill: float = 0.0
    scripted_opponent: str = "economic"
    teacher_checkpoint: Path | None = None
    actor_sync_every_rounds: PositiveInt = 4
    environments_per_rank: PositiveInt = 24
    collection_processes: PositiveInt = 24
    pool_capacity: PositiveInt = 8
    initial_snapshots: tuple[Path, ...] = ()
    snapshot_every_environment_steps: PositiveInt | None = None
    snapshot_at_start: bool = False
    pool_sampling: Literal["uniform"] = "uniform"
    pool_replacement: Literal["oldest"] = "oldest"
    population_seed: int = 0


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    lr: PositiveFloat = LEARNING_RATE
    adam_eps: PositiveFloat = ADAM_EPS
    gamma: float = DISCOUNTING
    lmb: float = LMB
    entropy_cost: float = ENTROPY_COST
    teacher_kl_cost: float = 0.0
    teacher_baseline_cost: NonNegativeFloat = 0.0
    vtrace_pg_cost: NonNegativeFloat = 1.0
    upgo_pg_cost: NonNegativeFloat = 1.0
    baseline_cost: NonNegativeFloat = 1.0
    clip_grad_norm: PositiveFloat = CLIP_GRADS
    final_lr_multiplier: float = Field(default=MIN_LR_MOD, ge=0.0, le=1.0)
    unroll_length: PositiveInt = UNROLL_LENGTH
    batch_segments: PositiveInt = 4
    value_warmup_batches: NonNegativeInt = VALUE_WARMUP_BATCHES
    value_passes: NonNegativeInt = 0


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    seed: int = 0
    accelerator: str = "auto"
    devices: Literal[1] = 1
    num_nodes: Literal[1] = 1
    strategy: Literal["auto"] = "auto"
    precision: Literal["32-true"] = "32-true"
    deterministic: bool = False
    benchmark: bool | None = None
    total_environment_steps: PositiveInt = TOTAL_STEPS
    log_every_n_steps: PositiveInt = 1
    checkpoint_every_environment_steps: PositiveInt = 1_000_000
    profiler: Literal["simple", "advanced"] | None = None
    output_dir: Path = Path("run/toad")
    resume: Path | None = None
    compile: Literal[False] = False
    rollout_backend: Literal["reference"] = "reference"


class EvaluationGate(BaseModel):
    metric: str
    minimum: float
    opponent: str
    seeds: PositiveInt


class CurriculumConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    phase: str = "phase1"
    reward_field: Literal["shaped_money", "shaped", "sparse", "own"] = "shaped_money"
    gate: EvaluationGate | None = None
    on_gate_failure: Literal["stop"] = "stop"


class ToadConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    model: ModelConfig = Field(default_factory=ModelConfig)
    population: PopulationConfig = Field(default_factory=PopulationConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)

    @classmethod
    def control(cls) -> "ToadConfig":
        return cls()

    @model_validator(mode="after")
    def validate_relationships(self) -> Self:
        probabilities = (
            self.population.selfplay,
            self.population.scripted,
            self.population.frozen_opponent,
            self.population.teacher_distill,
        )
        if any(value < 0 for value in probabilities) or not math.isclose(sum(probabilities), 1.0):
            raise ValueError("population probabilities must be nonnegative and sum to one")
        if (self.optimizer.teacher_kl_cost or self.population.teacher_distill) and self.population.teacher_checkpoint is None:
            raise ValueError("teacher checkpoint is required by teacher loss or batches")
        if self.population.teacher_checkpoint is not None and not self.population.teacher_checkpoint.is_file():
            raise ValueError(f"teacher checkpoint is not readable: {self.population.teacher_checkpoint}")
        if self.population.frozen_opponent and not (
            self.population.initial_snapshots
            or (self.population.snapshot_at_start and self.population.snapshot_every_environment_steps)
        ):
            raise ValueError("frozen opponent batches require initial snapshots or a snapshot schedule")
        return self
```

- [ ] **Step 4: Implement JSON loading and typed dotted overrides**

`load_config` reads JSON first. YAML is not introduced because it is not a project dependency. An override such as `optimizer.lr=5e-5` is parsed with `json.loads` and validated by reconstructing `ToadConfig`; unknown paths raise `ValueError`.

```python
def load_config(path: Path | None, overrides: Sequence[str] = ()) -> ToadConfig:
    payload = {} if path is None else json.loads(path.read_text())
    return apply_overrides(ToadConfig.model_validate(payload), overrides)


def apply_overrides(config: ToadConfig, overrides: Sequence[str]) -> ToadConfig:
    payload = config.model_dump(mode="python")
    for override in overrides:
        path, raw = override.split("=", 1)
        cursor = payload
        parts = path.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                raise ValueError(f"unknown override path {path!r}")
            cursor = cursor[part]
        if parts[-1] not in cursor:
            raise ValueError(f"unknown override path {path!r}")
        cursor[parts[-1]] = json.loads(raw)
    return ToadConfig.model_validate(payload)


STRUCTURAL_FIELDS = (
    "model",
    "optimizer.unroll_length",
    "optimizer.batch_segments",
)


def structural_fingerprint(config: ToadConfig) -> str:
    payload = select_paths(config.model_dump(mode="json"), STRUCTURAL_FIELDS)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
```

Run: `uv run pytest tests/learn/test_toad_config.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add .pre-commit-config.yaml pyproject.toml src/kaggriculture/learn/toad/__init__.py src/kaggriculture/learn/toad/config.py tests/learn/test_toad_config.py
git commit -m "feat: add typed Toad experiment config"
```

### Task 3: Make collection rounds yield optimizer-sized typed batches

**Files:**

- Create: `src/kaggriculture/learn/toad/data.py`
- Create: `tests/learn/test_toad_data.py`
- Modify: `src/kaggriculture/learn/scripts/toad.py:1329-1564`

**Interfaces:**

- Consumes: current `Trajectory`, `_segments`, `_collect`, and `ToadConfig`.
- Produces: `BatchKind`, `LearnerBatch`, `RoundBatchExpander`, `ReferenceRoundSource`, and `ToadDataModule.publish_actor(state_dict, version)`.

- [ ] **Step 1: Write failing batch-expansion tests**

```python
def test_round_expansion_preserves_policy_and_value_pass_counts() -> None:
    trajectories = [_trajectory(turns=32) for _ in range(2)]
    expander = RoundBatchExpander(batch_segments=4, value_passes=2, seed=7)
    batches = list(expander.expand(trajectories, RoundMeta.control(round_id=3)))

    policy = [batch for batch in batches if not batch.baseline_only]
    replay = [batch for batch in batches if batch.baseline_only]
    assert len(policy) == 1
    assert len(replay) == 2
    assert batches[-1].end_of_round
    assert sum(batch.collected_steps for batch in batches) == 64


def test_collection_error_names_the_failed_game() -> None:
    source = failing_reference_source(game_id=41, seed=99)
    with pytest.raises(CollectionError, match="game_id=41 seed=99"):
        next(iter(source))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/learn/test_toad_data.py -v`

Expected: FAIL because the data module does not exist.

- [ ] **Step 3: Implement immutable metadata and batch types**

```python
class BatchKind(StrEnum):
    SELFPLAY = "selfplay"
    SCRIPTED = "scripted"


@dataclass(frozen=True)
class RoundMeta:
    round_id: int
    actor_version: int
    game_ids: tuple[int, ...]
    seeds: tuple[int, ...]
    opponent_ids: tuple[str, ...]
    kind: BatchKind


@dataclass(frozen=True)
class LearnerBatch:
    segments: tuple[dict[str, torch.Tensor], ...]
    kind: BatchKind
    baseline_only: bool
    first_of_round: bool
    end_of_round: bool
    collected_steps: int
    round_id: int
    actor_version: int
    game_ids: tuple[int, ...]
    opponent_ids: tuple[str, ...]
```

Add the explicit control constructor used by tests:

```python
@classmethod
def control(cls, round_id: int = 0) -> "RoundMeta":
    return cls(
        round_id=round_id,
        actor_version=0,
        game_ids=(0,),
        seeds=(0,),
        opponent_ids=("self",),
        kind=BatchKind.SELFPLAY,
    )
```

`RoundBatchExpander` calls the extracted public `segments(trajectory, unroll_length)` function, groups exactly `batch_segments`, emits each fresh group once, then emits deterministic shuffled value-only passes. Only the first policy batch carries the round's collected-step count; all replay batches carry zero.

- [ ] **Step 4: Implement the synchronous iterable DataModule**

```python
class ToadDataModule(L.LightningDataModule):
    def __init__(self, config: ToadConfig, source: ReferenceRoundSource) -> None:
        super().__init__()
        self.config = config
        self.source = source

    def publish_actor(self, state_dict: Mapping[str, torch.Tensor], version: int) -> None:
        self.source.publish_actor(
            {name: tensor.detach().cpu() for name, tensor in state_dict.items()},
            version,
        )

    def train_dataloader(self) -> DataLoader[LearnerBatch]:
        return DataLoader(
            RoundIterableDataset(self.source, self.config),
            batch_size=None,
            num_workers=0,
        )
```

Wrap collection without substituting data:

```python
try:
    trajectories = self.collect_assignment(assignment)
except Exception as error:
    raise CollectionError(
        f"collection failed for game_id={assignment.game_id} seed={assignment.seed} "
        f"opponent={assignment.opponent_id}"
    ) from error
```

The Trainer will later be constructed with `use_distributed_sampler=False` even on one device so this ownership does not change silently in the DDP plan.

Run: `uv run pytest tests/learn/test_toad_data.py tests/learn/test_toad_runner.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/learn/toad/data.py src/kaggriculture/learn/scripts/toad.py tests/learn/test_toad_data.py tests/learn/test_toad_runner.py
git commit -m "refactor: expose optimizer-sized Toad batches"
```

### Task 4: Add the numerically equivalent LightningModule

**Files:**

- Create: `src/kaggriculture/learn/toad/lightning.py`
- Create: `tests/learn/test_toad_lightning.py`
- Modify: `src/kaggriculture/learn/scripts/toad.py:1566-1778`
- Modify: `tests/learn/test_toad_control_fixture.py`

**Interfaces:**

- Consumes: `LearnerBatch`, `ToadConfig`, current `_acted`, `_kl`, and `toad_loss.losses`.
- Produces: `LossReport`, `compute_loss(policy, batch, config, teacher)`, and `ToadLightningModule`.

- [ ] **Step 1: Write a failing parity test around one Lightning batch**

```python
def test_lightning_training_step_matches_control_fixture(tmp_path: Path) -> None:
    fixture = load_control_fixture()
    module = ToadLightningModule(ToadConfig.control())
    module.policy.load_state_dict(fixture["initial_model"])
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        max_steps=1,
        logger=False,
        enable_checkpointing=False,
        gradient_clip_val=CLIP_GRADS,
    )
    trainer.fit(module, train_dataloaders=one_batch_loader(fixture))

    for name, value in module.policy.state_dict().items():
        assert torch.equal(value, fixture["updated_model"][name])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/learn/test_toad_lightning.py::test_lightning_training_step_matches_control_fixture -v`

Expected: FAIL because `ToadLightningModule` does not exist.

- [ ] **Step 3: Extract pure loss computation and implement automatic optimization**

```python
@dataclass(frozen=True)
class LossReport:
    total: torch.Tensor
    terms: Mapping[str, torch.Tensor]


class ToadLightningModule(L.LightningModule):
    def __init__(self, config: ToadConfig) -> None:
        super().__init__()
        self.config = config
        self.policy = Policy(
            blocks=config.model.blocks,
            channels=config.model.channels,
            value_bound=config.model.value_bound,
        )
        self.environment_steps = 0
        self.collection_round = 0
        self.actor_version = 0
        self.actor_source_global_step = 0
        self.warmup_remaining = config.optimizer.value_warmup_batches
        self._round_ended = False
        self._round_started_warming = False
        self.save_hyperparameters(config.model_dump(mode="json"))

    def training_step(self, batch: LearnerBatch, batch_idx: int) -> torch.Tensor:
        if batch.first_of_round:
            self._round_started_warming = self.warmup_remaining > 0
        baseline_only = batch.baseline_only or self.warmup_remaining > 0
        report = compute_loss(self.policy, batch, self.config, baseline_only=baseline_only)
        self._round_ended = batch.end_of_round
        if not batch.baseline_only and self.warmup_remaining:
            self.warmup_remaining -= 1
        self.environment_steps += batch.collected_steps
        if batch.end_of_round:
            self.collection_round += 1
        self.log_dict({f"loss/{name}": value.detach() for name, value in report.terms.items()})
        return report.total
```

`compute_loss` is the tensor-returning body of current `_step`; remove its `zero_grad`, backward, clipping, optimizer step, and `.item()` calls. Preserve current tensor order and FP32 behavior.

- [ ] **Step 4: Preserve optimizer, clipping, and round scheduler semantics**

```python
def configure_optimizers(self) -> dict[str, object]:
    optimizer = torch.optim.Adam(
        self.policy.parameters(),
        lr=self.config.optimizer.lr,
        eps=self.config.optimizer.adam_eps,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        round_decay(self.config),
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
    }


def lr_scheduler_step(self, scheduler: LRScheduler, metric: object | None) -> None:
    if self._round_ended:
        scheduler.step()
```

Configure Trainer with `gradient_clip_val=config.optimizer.clip_grad_norm` and `gradient_clip_algorithm="norm"`.

Run: `uv run pytest tests/learn/test_toad_lightning.py tests/learn/test_toad_control_fixture.py tests/learn/test_toad_runner.py -v`

Expected: PASS, including exact control parameter equality.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/learn/toad/lightning.py src/kaggriculture/learn/scripts/toad.py tests/learn/test_toad_lightning.py tests/learn/test_toad_control_fixture.py
git commit -m "feat: add equivalent Lightning Toad learner"
```

### Task 5: Add boundary callbacks and full-state resume

**Files:**

- Create: `src/kaggriculture/learn/toad/callbacks.py`
- Create: `tests/learn/test_toad_callbacks.py`
- Modify: `src/kaggriculture/learn/toad/data.py`
- Modify: `src/kaggriculture/learn/toad/lightning.py`
- Modify: `tests/learn/test_toad_checkpoint.py`

**Interfaces:**

- Consumes: module counters, `ToadDataModule.publish_actor`, and Lightning checkpoint APIs.
- Produces: `ActorSyncCallback`, `EnvironmentStepStop`, `BoundaryCheckpoint`, plus `on_save_checkpoint` and `on_load_checkpoint` state.

- [ ] **Step 1: Write failing callback tests**

```python
def test_actor_is_not_published_during_warmup() -> None:
    callback = ActorSyncCallback(every_rounds=2)
    trainer, module, data = callback_fixture(warmup_remaining=1, round_id=2)
    callback.on_train_batch_end(trainer, module, None, end_batch(), 0)
    assert data.published == []


def test_environment_stop_uses_collected_steps_not_optimizer_steps() -> None:
    callback = EnvironmentStepStop(total_environment_steps=128)
    trainer, module, _ = callback_fixture(environment_steps=128)
    callback.on_train_batch_end(trainer, module, None, end_batch(), 9)
    assert trainer.should_stop
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/learn/test_toad_callbacks.py -v`

Expected: FAIL because the callbacks do not exist.

- [ ] **Step 3: Implement actor publication and environment-step stop**

```python
class ActorSyncCallback(L.Callback):
    def on_fit_start(self, trainer: L.Trainer, module: ToadLightningModule) -> None:
        trainer.datamodule.publish_actor(module.policy.state_dict(), module.actor_version)

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx) -> None:
        if not batch.end_of_round or module._round_started_warming:
            return
        if module.collection_round % module.config.population.actor_sync_every_rounds:
            return
        module.actor_version += 1
        module.actor_source_global_step = int(module.global_step)
        trainer.datamodule.publish_actor(module.policy.state_dict(), module.actor_version)


class EnvironmentStepStop(L.Callback):
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx) -> None:
        if batch.end_of_round and module.environment_steps >= self.total_environment_steps:
            trainer.should_stop = True
```

- [ ] **Step 4: Extend Lightning checkpoints and test deterministic resume**

```python
def on_save_checkpoint(self, checkpoint: dict[str, object]) -> None:
    checkpoint["toad"] = {
        "config": self.config.model_dump(mode="json"),
        "fingerprint": structural_fingerprint(self.config),
        "environment_steps": self.environment_steps,
        "collection_round": self.collection_round,
        "actor_version": self.actor_version,
        "actor_source_global_step": self.actor_source_global_step,
        "warmup_remaining": self.warmup_remaining,
    }


def on_load_checkpoint(self, checkpoint: dict[str, object]) -> None:
    state = checkpoint["toad"]
    assert_resume_compatible(self.config, ToadConfig.model_validate(state["config"]))
    self.environment_steps = int(state["environment_steps"])
    self.collection_round = int(state["collection_round"])
    self.actor_version = int(state["actor_version"])
    self.actor_source_global_step = int(state["actor_source_global_step"])
    self.warmup_remaining = int(state["warmup_remaining"])


def assert_resume_compatible(effective: ToadConfig, stored: ToadConfig) -> None:
    differences = {
        path: (read_path(stored, path), read_path(effective, path))
        for path in STRUCTURAL_FIELDS
        if read_path(stored, path) != read_path(effective, path)
    }
    if differences:
        rendered = ", ".join(
            f"{path}: stored={before!r}, effective={after!r}"
            for path, (before, after) in differences.items()
        )
        raise ResumeConfigError(f"structural config mismatch: {rendered}")
```

`BoundaryCheckpoint` calls `trainer.save_checkpoint(temp_path)`, `os.replace(temp_path, final_path)`, and only does so for an end-of-round batch whose global environment-step threshold is due.

The DataModule contributes the collector stream to the same Lightning checkpoint:

```python
def state_dict(self) -> dict[str, object]:
    return {
        "next_game_id": self.source.next_game_id,
        "collector_rng": self.source.rng.getstate(),
        "published_actor_version": self.source.actor_version,
    }


def load_state_dict(self, state_dict: dict[str, object]) -> None:
    self.source.next_game_id = int(state_dict["next_game_id"])
    self.source.rng.setstate(state_dict["collector_rng"])
    self.source.actor_version = int(state_dict["published_actor_version"])
```

Run: `uv run pytest tests/learn/test_toad_callbacks.py tests/learn/test_toad_checkpoint.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kaggriculture/learn/toad/callbacks.py src/kaggriculture/learn/toad/data.py src/kaggriculture/learn/toad/lightning.py tests/learn/test_toad_callbacks.py tests/learn/test_toad_checkpoint.py
git commit -m "feat: checkpoint Lightning Toad at round boundaries"
```

### Task 6: Switch the CLI and curriculum to the control Trainer

**Files:**

- Modify: `src/kaggriculture/learn/scripts/toad.py:347-649`
- Modify: `src/kaggriculture/learn/scripts/curriculum.py:39-283`
- Modify: `tests/learn/test_toad_runner.py`
- Modify: `tests/learn/test_curriculum.py`
- Create: `tests/learn/test_toad_cli.py`

**Interfaces:**

- Consumes: all public foundation interfaces.
- Produces: `build_trainer(config)`, `run(config)`, CLI `--config`/`--set`, and `phase_config(phase) -> ToadConfig`.

- [ ] **Step 1: Write failing CLI and curriculum tests**

```python
def test_cli_resolves_config_and_overrides(tmp_path: Path) -> None:
    path = tmp_path / "toad.json"
    path.write_text(json.dumps(ToadConfig.control().model_dump(mode="json")))
    config = parse_config(["--config", str(path), "--set", "optimizer.lr=5e-5"])
    assert config.optimizer.lr == 5e-5


def test_every_curriculum_phase_is_a_valid_toad_config() -> None:
    for phase in PHASES:
        config = phase_config(phase)
        assert config.curriculum.phase == phase.name
        assert config.model.blocks == phase.blocks
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/learn/test_toad_cli.py tests/learn/test_curriculum.py -v`

Expected: FAIL because `parse_config` and `phase_config` do not exist.

- [ ] **Step 3: Make `toad.py` a thin Trainer entry point**

```python
def build_trainer(config: ToadConfig) -> L.Trainer:
    return L.Trainer(
        accelerator=config.runtime.accelerator,
        devices=config.runtime.devices,
        num_nodes=config.runtime.num_nodes,
        strategy=config.runtime.strategy,
        precision=config.runtime.precision,
        deterministic=config.runtime.deterministic,
        benchmark=config.runtime.benchmark,
        profiler=config.runtime.profiler,
        log_every_n_steps=config.runtime.log_every_n_steps,
        gradient_clip_val=config.optimizer.clip_grad_norm,
        gradient_clip_algorithm="norm",
        max_steps=-1,
        max_epochs=-1,
        use_distributed_sampler=False,
        callbacks=[
            ActorSyncCallback(config.population.actor_sync_every_rounds),
            EnvironmentStepStop(config.runtime.total_environment_steps),
            BoundaryCheckpoint(config.runtime.output_dir),
        ],
        logger=build_wandb_logger(config),
    )


def build_wandb_logger(config: ToadConfig) -> WandbLogger:
    return WandbLogger(
        project="kaggriculture-2026",
        name=config.curriculum.phase,
        save_dir=str(config.runtime.output_dir),
        config=config.model_dump(mode="json"),
    )


def build_reference_data_module(config: ToadConfig) -> ToadDataModule:
    source = ReferenceRoundSource(config)
    return ToadDataModule(config, source)


def run(config: ToadConfig) -> None:
    seed_everything(config.runtime.seed, workers=True)
    module = ToadLightningModule(config)
    data = build_reference_data_module(config)
    build_trainer(config).fit(module, datamodule=data, ckpt_path=config.runtime.resume)
```

The argparse surface is only `--config PATH` and repeatable `--set PATH=JSON_VALUE`. Keep a documented compatibility translator for old flags for one release, but log the fully resolved Pydantic config and route both paths through `run`.

- [ ] **Step 4: Replace curriculum flag assembly with typed overrides**

```python
def phase_config(phase: Phase) -> ToadConfig:
    teacher = _checkpoint(phase.teacher_from) if phase.teacher_from else None
    return ToadConfig(
        model=ModelConfig(blocks=phase.blocks, channels=toad.CHANNELS),
        optimizer=OptimizerConfig(
            lr=phase.lr,
            lmb=phase.lmb,
            entropy_cost=phase.entropy_cost,
            teacher_kl_cost=phase.teacher_kl_cost,
        ),
        population=PopulationConfig(teacher_checkpoint=teacher),
        runtime=RuntimeConfig(total_environment_steps=phase.steps),
        curriculum=CurriculumConfig(phase=phase.name, reward_field=phase.reward),
    )
```

The test must assert every resulting nested value is its declared Pydantic model type.

Run: `uv run pytest tests/learn/test_toad_cli.py tests/learn/test_curriculum.py tests/learn/test_toad_runner.py -v`

Expected: PASS.

- [ ] **Step 5: Run the full control gate and commit**

Run: `uv run pre-commit run -a`

Expected: all hooks PASS.

```bash
git add src/kaggriculture/learn/scripts/toad.py src/kaggriculture/learn/scripts/curriculum.py tests/learn
git commit -m "refactor: run Toad through Lightning"
```

## Completion Gate

This plan is complete only when:

- the checked-in control fixture matches logits, losses, parameters, Adam moments, LR, actor version, and next-round metadata;
- `Trainer(fast_dev_run=2)` completes on CPU with automatic optimization;
- full-state checkpoint/resume begins the same next collection round;
- old curriculum phases resolve to the same blocks, rewards, teachers, LR, entropy, lambda, and budgets; and
- the repository's full pre-commit suite passes.
