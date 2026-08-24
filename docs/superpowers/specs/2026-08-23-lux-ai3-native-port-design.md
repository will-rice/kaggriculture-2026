# Port the Lux AI Season 3 training stack, not its game

**Design, 2026-08-23.** Target: Kaggriculture on
`kaggle-environments` 1.32.7. Source inspected:
[`tonykozlovsky/lux-ai3-pub`](https://github.com/tonykozlovsky/lux-ai3-pub)
at commit `8001d70d939c78725d198a70586e8ba77efa2a24`, specifically the final
`final_versions/08_03_tune_against_mask_cont` tree.

## 1. Decision

Port the missing ideas from the published Lux AI Season 3 winner into the
active Kaggriculture Toad path as native components. Do not import its Lux
environment, observation builder, action decoder, submission runner, Hydra
configuration, shared-memory queue topology, or C++ extension.

The resulting trainer will:

- keep Kaggriculture's existing encodings, masks, four action/value heads,
  simulator, reward definitions, and V-trace/UPGO/TD(lambda) implementation;
- add the useful missing model and population mechanisms from the actual final
  upstream code;
- make the whole experiment configuration a validated Pydantic model;
- organize optimization and collection as a PyTorch Lightning trainer, with
  DDP and automatic mixed precision controlled by configuration; and
- land in stages whose first acceptance condition is equivalence with today's
  active Toad runner when every new component is disabled.

This is a native staged port, not a line-for-line transplant. A direct fork
would preserve a large quantity of Lux-specific machinery and assumptions; a
small feature cherry-pick would leave the same monolithic runner and make
multi-device state ownership ambiguous. The native staged design preserves
the algorithms while giving Kaggriculture one coherent model, rollout, and
checkpoint contract.

## 2. What the upstream code actually contains

The useful source is not just the repository README. The inspected final tree
implements the following:

- `lux_ai/nns/models.py`: residual spatial trunks, ConvLSTM state with
  terminal resets, per-unit patch extraction, prediction heads, and alternate
  baseline heads;
- `lux_ai/nns/transformer.py`: spatial transformer blocks and positional
  embeddings;
- `lux_ai/torchbeast/core/learn.py` and the loss modules: autocast learner
  forward, V-trace, UPGO, TD losses, teacher policy/baseline losses, prediction
  losses, and distinct handling for self-play, frozen-actor, frozen-teacher,
  and behavior-cloning batches;
- `lux_ai/torchbeast/core/stats.py`: independent entropy measurement and a
  multiplicative controller that follows decaying entropy targets;
- `lux_ai/torchbeast/core/act.py`, `batch_and_learn.py`, and
  `model_inference.py`: actor/learner separation, frozen opponents, frozen
  teachers, batch-type ratios, device-specific inference, bfloat16, and
  `torch.compile`;
- `conf/two_gpu.yaml`, `x8.yaml`, and `x2_teacher.yaml`: the real switches and
  mixtures used to compose those parts; and
- `runner.py`, `agent.py`, `lux_gym`, observation/action spaces, and the C++
  library: Lux-specific execution that this port deliberately excludes.

Some upstream switches are marked deprecated, heavy, untested, or not working
in its own final configs. In particular, behavior-cloning actors are configured
with zero probability and comments saying they are not working. This design
does not promote every available switch into a requirement merely because it
exists in the repository.

## 3. What Kaggriculture already has

The active implementation is `src/kaggriculture/learn/scripts/toad.py`, not the
older vendored `src/kaggriculture/learn/toad/monobeast.py`. The distinction is
important: code present in the vendored directory does not count as an active
capability.

| Capability                                         | Active status                               | Source of truth                                                                 |
| -------------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------- |
| SE residual CNN                                    | Present                                     | `learn/model.py`; 3x3/ReLU rather than the final upstream 5x5/LeakyReLU variant |
| 8/16/24-block curriculum                           | Present/configurable                        | `learn/scripts/curriculum.py` and `learn/scripts/toad.py`                       |
| Per-unit operation head                            | Present                                     | gathered from the unit's board position in `learn/model.py`                     |
| Per-unit transfer-quantity head                    | Present, Kaggriculture-specific             | `learn/model.py`, encoding, masks, rollout, and simulator                       |
| Market head                                        | Present, Kaggriculture-specific             | pooled board/market readout in `learn/model.py`                                 |
| Value head and bounded-value option                | Present                                     | `learn/model.py`                                                                |
| Legal-action masking                               | Present                                     | `learn/mask.py`, rollout, and learner rescoring                                 |
| V-trace + UPGO + TD(lambda)                        | Present                                     | `learn/toad_loss.py` and vendored return primitives                             |
| Teacher KL                                         | Present                                     | active Toad learner                                                             |
| Value warmup and extra critic passes               | Present                                     | active Toad learner                                                             |
| Actor lag and scheduled synchronization            | Present                                     | explicit frozen actor copy in active Toad learner                               |
| Adam epsilon, gradient clipping, LR decay          | Present                                     | active Toad learner                                                             |
| Checkpoint/resume and detailed metrics             | Present                                     | active Toad learner                                                             |
| Snapshot opponent pool                             | Present elsewhere                           | older `learn/scripts/selfplay.py`, not active Toad                              |
| Batched tensor simulator/CUDA graph                | Present elsewhere                           | `sim/rollout.py`, not active Toad                                               |
| Behavior cloning                                   | Present elsewhere                           | corpus/dataset/training path, intentionally not part of this population         |
| Pydantic                                           | Dependency and small harness config present | no unified typed training config                                                |
| Lightning                                          | Dependency and `seed_everything` present    | no `LightningModule`, `Trainer`, or `DataModule`                                |
| ConvLSTM/recurrent state                           | Missing from active model                   | only vendored upstream code exists                                              |
| Spatial transformer                                | Missing from active model                   | only vendored upstream code exists                                              |
| Per-unit residual patch processing                 | Missing from active model                   | current head gathers one trunk column                                           |
| Interaction-aware value head                       | Missing                                     | current value head mean-pools the trunk                                         |
| Hidden-opponent belief head                        | Missing                                     | opponent private state is not observable at inference                           |
| Frozen opponents/teacher-generated batches in Toad | Missing                                     | current Toad has mirror/scripted batches and teacher KL only                    |
| Per-head target entropy                            | Missing                                     | current learner uses one fixed summed entropy coefficient                       |
| BF16 and compile in active Toad                    | Missing                                     | vendored code contains both but active runner does not call them                |
| Multi-device learner                               | Missing                                     | active runner selects one device manually                                       |

The port therefore reuses more than it replaces. The existing action semantics,
loss math, simulator fidelity, curriculum, and diagnostics stay authoritative.

## 4. Competition translation contract

The upstream architecture is adapted by role, not by Lux vocabulary.

| Lux final-stack concept        | Kaggriculture adaptation                                                                           |
| ------------------------------ | -------------------------------------------------------------------------------------------------- |
| 24x24 spatial state            | existing 10x10 encoded board                                                                       |
| global/non-spatial features    | existing market, season, farm, and private scalar vector                                           |
| 16 unit move/sap heads         | up to `MAX_UNITS` operation and transfer-quantity heads                                            |
| 15x15 unit patch               | configurable 7x7 local patch on the smaller board                                                  |
| global sap interactions        | market and cross-farm interaction context                                                          |
| enemy-next-position prediction | belief over hidden opponent shed, seed stock, and aggregate carried inventory                      |
| transformer baseline           | interaction-aware value readout over spatial tokens plus global context                            |
| frozen actor                   | frozen Kaggriculture policy snapshot used as an opponent                                           |
| frozen teacher batch           | trajectories generated against/by a frozen teacher, with explicitly configured distillation losses |
| external BC batch              | excluded from this port                                                                            |

The belief targets are privileged training labels available from the simulator's
complete state. They are never added to the inference observation. At inference,
the policy receives only the previous belief prediction and public/current-seat
features, so the head cannot leak hidden opponent state.

The current Kaggriculture action masks remain the final authority. The port
does not infer Lux masks or reuse any Lux action indices. Every new head reads
the existing masks and calls the existing joint-log-probability semantics.

## 5. Typed configuration

`ToadConfig` is a Pydantic `BaseModel` and is the only experiment contract. It
contains five nested models:

### `ModelConfig`

- residual block count, channel width, kernel size, and activation;
- ConvLSTM enabled flag, hidden width, kernel size, and layer count;
- transformer enabled flag, block count, head count, head dimension, MLP
  multiplier, dropout, and positional encoding;
- local-unit patch enabled flag, odd patch size, residual block count, and
  patch width;
- interaction-value-head enabled flag and parameters;
- opponent-belief-head enabled flag, channel/MLP sizes, target fields, loss
  weight, and previous-prediction feedback flag; and
- value bound and every existing action-head dimension derived from the
  encoder rather than duplicated as literals.

### `PopulationConfig`

- probabilities for `selfplay`, `scripted`, `frozen_opponent`, and
  `teacher_distill` batches;
- scripted opponent identifier;
- teacher checkpoint and which teacher heads are valid;
- initial frozen snapshot paths, pool capacity, snapshot interval, sampling
  policy, and replacement policy;
- actor synchronization interval; and
- per-rank environments, collection processes, and opponent RNG seed offset.

### `OptimizerConfig`

- optimizer class and Adam parameters, including the existing epsilon;
- initial LR, final multiplier, gamma, lambda, clipping norm, unroll length,
  segments per batch, value warmup, and extra value passes;
- weights for V-trace PG, UPGO PG, baseline, teacher KL, teacher baseline,
  belief losses, and the independent entropy heads; and
- target-entropy initial values, target decay, multiplier initial values,
  multiplier adjustment rate, floor, and ceiling for each action head.

### `RuntimeConfig`

- seed, output directory, resume checkpoint, and debug controls;
- Lightning accelerator, devices, node count, DDP strategy, precision, and
  deterministic/benchmark flags;
- `torch.compile` enabled flag, mode, full-graph setting, and dynamic-shape
  setting;
- total global environment steps, logging cadence, checkpoint cadence, and
  profiler selection; and
- native/reference rollout backend plus rollout-device settings.

`precision` is a closed literal such as `"32-true"` or `"bf16-mixed"`, not an
arbitrary string. `devices` accepts Lightning's `auto`, a positive count, or an
explicit device list.

### `CurriculumConfig`

- the ordered phase definitions currently encoded by `curriculum.py`;
- the checkpoint/warm-start relationship between phases;
- reward field, teacher, block count, total-step budget, and optimizer/loss
  overrides per phase; and
- evaluation gate and failure policy at each phase boundary.

Pydantic validation rejects the configuration before W&B, worker processes,
or CUDA contexts start when:

- batch probabilities are negative or do not sum to one;
- teacher batches or teacher losses are enabled without a readable compatible
  teacher checkpoint;
- frozen-opponent probability is nonzero with neither an initial pool nor a
  snapshot schedule that can populate one before first use;
- belief feedback/loss is enabled without the belief head;
- transformer dimensions do not divide into valid heads;
- a local patch is even-sized or larger than its declared supported padding;
- recurrent training lacks initial recurrent states and terminal masks in the
  segment schema;
- a value-only phase enables policy-only losses; or
- structurally incompatible curriculum phases claim an in-place continuation
  without an explicit widening/migration rule.

Hardware availability is checked in a runtime preflight after Lightning has
resolved the accelerator. Requesting BF16, DDP, compile, or a native rollout
device that is unavailable fails explicitly. It never silently changes the
experiment.

The CLI accepts a config file plus explicit dotted overrides, validates them
once, and passes the resolved immutable model into the trainer. The complete
`model_dump(mode="json")` is stored in the Lightning checkpoint and W&B run.
There is no second collection of argparse defaults that can drift from it.

## 6. Model and recurrent-state contract

The policy API becomes conceptually:

```python
output = policy(board, scalars, positions, state, dones)
```

`PolicyOutput` contains unit-operation logits, unit-quantity logits, market
logits, state value, opponent-belief predictions, and the next `PolicyState`.
`PolicyState` contains ConvLSTM hidden state, ConvLSTM cell state, and the prior
opponent-belief tensor. These tensor containers are typed dataclasses or named
tuples, not Pydantic models; Pydantic owns serializable configuration, while
PyTorch owns tensors.

The enabled forward path is:

1. Encode the existing board through the spatial stem and the existing scalar
   vector through its projection.
2. Concatenate/project the prior belief where configured, then combine spatial
   and global features.
3. Run the existing SE residual trunk, retaining the configured 128-channel
   reference width and 10x10 extent.
4. Run ConvLSTM over the true time dimension, applying `dones` before the next
   state so no memory crosses an episode boundary.
5. Merge the residual trunk output, hidden state, and cell state with a 1x1
   projection.
6. Run the optional spatial transformer.
7. Produce the belief prediction and feed its current representation to the
   policy/value readouts without exposing its training target.
8. Extract a padded 7x7 patch around each active unit and process it through
   the optional local residual head before operation and quantity logits.
9. Produce market logits from global/spatial context.
10. Produce value from the interaction-aware value head, allowing both farms,
    the market context, and long-range board relationships to interact before
    reducing to a scalar.

The local-patch branch retains an out-of-bounds indicator so an edge patch is
not confused with genuine zero-valued tiles. Padded unit slots remain masked
and do not contribute to action, entropy, or belief losses.

Every trajectory segment stores the state immediately before its first
observation. Training replays that state across the segment and uses the
trailing observation for the value bootstrap exactly as the current learner
does. States are detached at segment boundaries; there is no backpropagation
between unrolls. `dones` reset hidden state, cell state, and prior belief at the
same temporal boundary in both actor and learner code.

A control configuration with ConvLSTM, transformer, belief, local patch, and
interaction value disabled instantiates the current `Policy` topology and
output semantics. That control is a permanent compatibility mode, not a
temporary migration aid.

## 7. Rollout and batch contract

The reference and native rollout backends emit the same `LearnerBatch` schema.
In addition to today's observations, actions, masks, behavior log-probabilities,
rewards, and dones, a batch carries:

- initial `PolicyState` for every segment;
- hidden-opponent targets and target-validity masks;
- `batch_kind`;
- opponent identity and snapshot digest;
- actor version;
- globally unique game IDs and seeds;
- collection round ID;
- local environment-step count; and
- first/end-of-round markers.

The four supported kinds are:

1. `selfplay`: latest synchronized actor against itself;
2. `scripted`: latest actor against the configured economic/scripted opponent;
3. `frozen_opponent`: latest actor against a sampled snapshot pool member; and
4. `teacher_distill`: configured teacher-generated or teacher-opponent games
   that activate the declared teacher policy/baseline losses.

There is no behavior-cloning population kind. Existing imitation learning
remains a separate pretraining/validation workflow and can produce a warm-start
checkpoint, but it does not silently enter the online batch mixture.

One collection round is expanded into the same optimizer units used today:
four 16-turn segments per learner batch. Policy batches see each new segment
once. Configured value-only passes reshuffle and replay the same round as
separate `baseline_only` batches. The expansion records which batch closes the
round so actor synchronization, LR scheduling, metrics, checkpoints, and the
next collection all observe the same boundary.

The initial implementation is intentionally synchronous:

- each DDP process owns one `ToadDataModule` and one rank-local collector;
- the Lightning DataLoader uses `num_workers=0` and
  `use_distributed_sampler=False`;
- the collector may retain the existing `ProcessPoolExecutor` underneath it;
- no DataLoader prefetch may begin the next collection before the actor-sync
  callback completes; and
- each rank collects the same fixed number of learner batches but different
  game IDs.

Global game IDs are allocated from the checkpointed environment-step/episode
counter. A rank receives a non-overlapping contiguous subrange for each round;
the seed is a deterministic function of the game ID and base seed. Changing
world size after a boundary changes future grouping but neither repeats nor
skips a global ID.

The later native-rollout stage substitutes `sim/rollout.py` behind this
contract. It does not change `ToadLightningModule`. Asynchronous queues or
dedicated actor GPUs are a later throughput extension only after the
synchronous DDP path is correct.

## 8. Lightning ownership

### `ToadLightningModule`

The module owns:

- the learner policy and optional frozen teacher used during loss evaluation;
- forward and loss computation;
- optimizer and LR scheduler construction;
- value-warmup and target-entropy controller state;
- environment-step, collection-round, actor-version, and batch counters;
- distributed metric logging; and
- checkpoint extension hooks.

Each group of four 16-turn segments is one Lightning `training_step`. This
lets Lightning retain automatic optimization even though the old outer
`_update` function performed many optimizer steps: the batch boundary is moved
to the optimizer boundary. A `baseline_only` or `teacher_distill` batch changes
the loss composition, not who calls backward.

Lightning therefore owns accelerator placement, DDP gradient synchronization,
autocast, backward, optimizer steps, and gradient clipping. The Trainer uses
the configured `gradient_clip_val`/norm algorithm. There are no `.cuda()` or
hard-coded learner device calls inside the module.

The scheduler must preserve today's collection-round clock. An
`end_of_round` flag is set during `training_step`; an overridden scheduler hook
advances only after the optimizer step that closes a round. It does not advance
once per segment batch. Lightning still serializes the scheduler state.

### `ToadDataModule`

The data module owns the collector, batch expansion, population selection,
and publication of actor weights. At fit start it receives the restored
learner state. At an eligible end-of-round callback it receives a detached CPU
actor snapshot directly in the same rank process. The next synchronous
collection uses that version.

During value warmup the actor remains pinned to the warm-start policy, matching
the existing stability behavior. After warmup, publication follows
`actor_sync_every_rounds`.

### Distributed semantics

Standard DDP is the initial multi-device strategy. Every rank:

- has an identical learner replica after optimizer synchronization;
- collects unique games with equal learner-batch counts;
- applies the same ordered batch-kind quota per round;
- reaches every end-of-round hook together; and
- all-reduces actual collected steps before updating the global counter.

Population snapshots and durable checkpoint files are rank-zero writes. All
ranks enter the callback, synchronize after an atomic write, and receive the
new manifest/version before continuing. No rank independently decides whether
a checkpoint or pool mutation is due.

Training terminates on `RuntimeConfig.total_environment_steps`, not Lightning's
optimizer `global_step`: value-only replay adds optimizer steps without adding
experience. Termination is checked only at a collection boundary and is
therefore identical on every rank.

### Mixed precision and compilation

`precision="bf16-mixed"` delegates autocast and optimizer integration to
Lightning. The network forward may use BF16, but masked log-softmax, joint
log-probabilities, importance ratios, V-trace, UPGO, TD(lambda), value targets,
teacher KL reductions, entropy statistics/controllers, and finite checks are
promoted to FP32. Masks remain boolean.

`torch.compile` is a distinct, default-off runtime stage. It is enabled only
after eager BF16 parity. The compiled object is created through one helper so
checkpoint export, actor publication, teacher loading, and submission building
all unwrap the same original module. Compile configuration is part of the
checkpoint and a requested compilation failure aborts the run.

FSDP, DeepSpeed, and multi-optimizer manual optimization are not part of this
port. The model is small enough for DDP, and target entropy uses the upstream
multiplicative controller rather than a learned alpha optimizer.

## 9. Losses and controllers

The base loss remains the active Kaggriculture implementation:

`V-trace PG + UPGO PG + baseline + teacher terms + entropy terms`.

The port extends it as follows:

- compute entropy separately for unit operation, conditional unit quantity,
  and market heads;
- retain the existing condition that quantity contributes only for operations
  that consume a quantity;
- attach one target schedule and multiplier controller to each meaningful
  head instead of multiplying one summed entropy by one constant;
- add supervised belief losses only where complete-state targets are valid;
- optionally add teacher baseline alignment when both student and teacher
  value semantics match; and
- preserve the existing `baseline_only` path during reward/critic warmup and
  extra value passes.

Each entropy target decays as a function of global environment steps. At a
round boundary the controller compares the distributed mean valid-action
entropy with the target. If entropy is above target it reduces the multiplier;
if below, it raises it, using configured multiplicative speed and bounds. The
target and multiplier states are checkpointed.

Teacher KL is head-aware. A legacy teacher that lacks the quantity head cannot
constrain a freshly initialized quantity distribution. Structural metadata in
the teacher manifest declares which heads are valid, and Pydantic validation
plus load-time key checks must agree.

## 10. Population and checkpoint state

`SnapshotPool` is extracted from the useful concept already present in the old
self-play runner, then used by active Toad. A manifest entry contains:

- immutable snapshot path;
- SHA-256 digest;
- model-structure fingerprint;
- environment steps and collection round;
- source run/checkpoint identifier;
- evaluation metadata when available; and
- creation timestamp.

Sampling is deterministic from the global game ID and population seed. The
initial policy may be excluded or weighted explicitly; no implicit "latest"
entry is mixed into the frozen pool because current self-play already covers
it. Capacity/replacement policy is configured and logged.

Lightning's checkpoint remains the authoritative resume artifact and contains
model, optimizer, scheduler, precision-plugin-compatible state, and global
step. `on_save_checkpoint`/`on_load_checkpoint` additionally store:

- resolved Pydantic configuration and its structural fingerprint;
- global environment steps, next game ID, collection round, and actor version;
- warmup balance and value-pass counters;
- target-entropy targets and multipliers;
- curriculum phase;
- population manifest and active teacher metadata; and
- collector RNG state needed to begin the next round.

Checkpoints are taken only at completed collection boundaries and are written
atomically. Resume republishes learner weights to the rank-local collector
before the first new game. A same-world-size resume must reproduce the next
game IDs, batch kinds, opponents, and parameter updates. A changed-world-size
resume is allowed only at a boundary and guarantees a non-overlapping future
game stream, not bitwise-identical grouping.

Structural configuration fields cannot be overridden on resume. Operational
fields such as devices, node count, logging destination, profiler, and output
directory may change. Any allowed overrides are recorded beside both the
stored and effective configuration.

## 11. Metrics

Direct `wandb.log` calls move behind Lightning's `WandbLogger` and
`self.log_dict`. Existing metric names remain stable where their meaning does.
Round-level metrics are reduced across ranks; process-local diagnostic timing
may be tagged by rank rather than reduced.

New required series include:

- collection throughput, learner throughput, queue/collection wait, and actor
  lag by version;
- batches, games, and returns by `batch_kind` and opponent identity;
- per-head entropy, target, multiplier, valid decision count, and mask density;
- belief loss and calibrated error per target family;
- ConvLSTM hidden/cell norms and terminal reset counts;
- policy/value/teacher loss components in FP32;
- global environment steps versus optimizer steps;
- population selection frequencies and snapshot evaluation results; and
- precision, compile status, world size, and rollout backend as immutable run
  metadata.

Metric aggregation happens at end-of-round so it does not insert a distributed
synchronization into every recurrent segment unless a numerical-failure check
requires one.

## 12. Code organization

The active implementation is split into focused native modules:

- `src/kaggriculture/learn/toad/config.py`: Pydantic models, loading,
  overrides, fingerprints, and resume compatibility;
- `src/kaggriculture/learn/toad/model.py`: recurrent/attention policy and state
  containers;
- `src/kaggriculture/learn/toad/data.py`: learner batches, round expansion,
  rank sharding, and data module;
- `src/kaggriculture/learn/toad/lightning.py`: LightningModule and Trainer
  construction;
- `src/kaggriculture/learn/toad/population.py`: snapshot manifests and
  deterministic selection;
- `src/kaggriculture/learn/toad/callbacks.py`: actor publication,
  environment-step stopping, snapshots, and boundary checkpoints;
- `src/kaggriculture/learn/toad_loss.py`: retained authoritative return/loss
  primitives, refactored only as tests require; and
- `src/kaggriculture/learn/scripts/toad.py`: thin CLI that resolves config,
  runs preflight, builds Trainer/module/data module, and calls `fit`.

`curriculum.py` selects phase configs and explicit overrides rather than
constructing argparse lists. The existing vendored `learn/toad/core` math may
remain while it is still imported, but no new active orchestration is added to
vendored `monobeast.py`.

## 13. Staged migration

### Stage 0: characterization

Freeze deterministic fixtures from the current policy, `_segments`, `_step`,
actor sync, schedule, and checkpoint restore. Record current one-round metrics
and parameter deltas in FP32. No production behavior changes.

### Stage 1: typed config

Introduce `ToadConfig`, translate the current curriculum, and make the old
runner consume the resolved config. Prove defaults and overrides serialize
round-trip and reproduce current constants.

### Stage 2: Lightning control path

Add LightningModule/DataModule/callbacks with all new model and population
features disabled, reference rollouts, `32-true`, one device, and no compile.
Require numerical and checkpoint/resume parity before making the Lightning
entry point default.

### Stage 3: recurrent belief model

Add recurrent-state batch plumbing, ConvLSTM, opponent-belief targets/head,
and previous-belief feedback. Gate independently so recurrence without belief
and belief without feedback can be diagnosed.

### Stage 4: spatial readouts

Add transformer, interaction value, and per-unit local residual patch one at a
time. Each feature gets an ablation/config switch and its own shape, masking,
gradient, and short-budget evaluation result.

### Stage 5: population

Extract the snapshot pool, then add `frozen_opponent` and `teacher_distill`
batches. Keep scripted and mirrored populations as controls. Verify configured
ratios from observed batches rather than configuration alone.

### Stage 6: adaptive entropy

Split entropy statistics and enable target controllers with fixed-target
tests before target decay. Compare against the fixed-entropy control.

### Stage 7: BF16 and compile

Enable Lightning BF16 on one supported GPU, establish FP32-sensitive loss
islands, and pass finite/update tests. Enable compile only afterward and
measure compile amortization plus steady-state throughput.

### Stage 8: DDP and native rollout

Run two-device DDP with reference collection, proving seed uniqueness and
synchronized weights. Then substitute the already-tested tensor simulator
behind the batch contract. CUDA graphs or dedicated actor devices are
performance work after exact native/reference differential tests pass.

### Stage 9: retirement

Make the Lightning path the sole active Toad runner only after it passes the
same evaluation gate and at least one checkpoint has been resumed through a
phase boundary. Keep a conversion/read-only loader for old checkpoints; do not
maintain two mutable trainers indefinitely.

## 14. Verification and acceptance gates

### Unit and property tests

- Pydantic accepts every checked-in phase and rejects every invalid
  cross-field combination named above.
- The control model matches current output shapes and action-mask semantics.
- Recurrent state resets exactly on terminal rows, detaches between segments,
  and round-trips through batch/checkpoint serialization.
- Padded and edge local patches are correct, including the out-of-bounds plane.
- Belief labels contain simulator truth only in targets; zeroed/hidden
  inference inputs cannot recover them directly.
- Each entropy controller moves in the correct direction, respects bounds,
  decays by global environment steps, and resumes exactly.
- Population sampling is deterministic, respects ratios, rejects incompatible
  snapshots, and never duplicates rank game IDs.

### Numerical parity

With new components disabled and fixed initial weights, the Lightning control
path must match the current FP32 runner for:

- logits, values, masked log-probabilities, bootstrap value, and loss terms;
- gradients before clipping and clipped gradient norm;
- parameters and Adam moments after one batch and one full collection round;
- LR after a round;
- actor version/synchronization behavior, including no sync during warmup; and
- checkpoint resume into the next round.

Exact equality is required for discrete outputs and counters. Floating-point
tolerances are declared in the test beside each tensor; they may not be widened
merely to make a regression pass.

### Lightning integration

- CPU `fast_dev_run` in `32-true` completes forward, backward, logging, and a
  boundary checkpoint.
- One-GPU eager FP32 matches the control fixture.
- One-GPU BF16 produces finite losses/gradients and a bounded update difference
  against FP32 over a fixed short run.
- A two-device subprocess test proves disjoint game IDs, equal batch counts,
  synchronized learner weights, one durable snapshot, and successful resume.
- Compile covers recurrent forward, all masked heads, backward, actor export,
  and checkpoint restore.

### Simulator and training behavior

- Reference and native rollout backends produce identical state transitions,
  legality, rewards, dones, recurrent resets, and batch metadata for fixed
  seeds/actions within the native simulator's declared domain.
- A fixed-budget control run matches the current policy's scripted-opponent
  result within a predeclared statistical interval.
- Every added architecture/population stage reports its delta against the
  immediately prior accepted stage on held-out seeds.
- The existing curriculum and frontier holdout gates remain the final shipping
  criteria. Training throughput alone cannot accept a weaker policy.

A stage does not advance merely because it runs. It must retain legal actions,
finite optimization, deterministic boundary resume, and the expected
fixed-budget evaluation result.

## 15. Failure handling

- Configuration errors fail before logging, processes, files, or CUDA
  allocation.
- A NaN/Inf in inputs, logits, losses, gradients, values, importance ratios,
  or recurrent state records batch kind, game IDs, opponent digest, actor
  version, precision, and state norms, then stops all ranks.
- Collector exceptions retain the original traceback and failing seed/game ID
  and propagate to Trainer. There is no silent replacement with another game.
- Every rank verifies the expected batch count and end-of-round marker before
  entering a distributed round. A mismatch aborts with per-rank counts rather
  than waiting for a DDP hang.
- Snapshot and checkpoint writes use a temporary sibling followed by atomic
  rename. The digest and structural fingerprint are validated before a
  snapshot enters the manifest.
- A missing/corrupt population member is a hard error for the selected game;
  the sampler does not quietly choose a different opponent.
- Structural checkpoint mismatches list the exact differing fields. Only the
  documented operational overrides are allowed.
- Requested BF16, compile, DDP, or native rollout failures are explicit. There
  is no automatic fallback that would make the recorded experiment differ
  from its config.

## 16. Non-goals

- No Lux environment, observations, rewards, action space, runner, submission
  code, C++ library, or Kaggle-specific inference augmentation.
- No wholesale import of Hydra or upstream flag/config plumbing.
- No behavior-cloning batch population. Existing BC may provide a declared
  warm start but remains a separate workflow.
- No recreation of the upstream shared-memory queue system in the first port.
- No FSDP, DeepSpeed, multi-node optimization research, or learned entropy
  optimizer.
- No reward redesign, simulator rule change, action-space change, or market
  objective change as part of this port.
- No promise that an upstream feature improves Kaggriculture merely because it
  helped Lux. Every feature remains independently switchable and gated.

## 17. Provenance and licensing

The source repository is MIT licensed, copyright 2025 Tony Kozlovsky. Any code
copied or substantially translated from it must retain its copyright and MIT
permission notice in the repository's third-party notices and in source-file
headers where appropriate. Each adapted component should name the exact
upstream file and inspected commit in its docstring or adjacent provenance
note.

Original Kaggriculture glue, Lightning integration, Pydantic schema, market and
quantity heads, belief target translation, and tests remain under this
repository's license. The implementation plan must include the notice change
in the first commit that adds derived source, not as cleanup after the port.

## 18. Implementation amendment — 2026-08-24

The completed runtime preflight establishes the following selectable boundary:

- reference rollout supports eager FP32 CPU, capability-checked BF16, compile,
  and explicit DDP; native rollout supports eager execution and explicit DDP;
- native rollout combined with compile is rejected until that joint path has a
  proof, and CUDA-graph round collection remains unselectable. Scripted native
  graph capture is rejected specifically because the opponent consumes host
  simulator rows;
- the verified native scripted opponent is `economic`; other scripted
  opponents fail before Trainer construction. Native CUDA requests and all
  explicit accelerator/BF16 requests also fail when the required hardware
  capability is unavailable; and
- an explicit GPU device tuple is ordered, range checked, and duplicate-free.
  Any resolved world size above one requires `strategy="ddp"`.

Each logger config now includes a frozen resolved runtime record containing
precision, the complete compile configuration, world size, and rollout
backend. Resume treats precision, compile configuration, and rollout backend
as immutable experiment identity. Only the proven operational placement and
diagnostic fields (`accelerator`, `devices`, `num_nodes`, `strategy`, logging
frequency, profiler, output directory, and resume path), plus consumed
warm-start provenance, may differ from the stored config.

The final performance integration review fixes four additional runtime
contracts:

- CUDA availability, device count, and BF16 support are resolved together in
  one disposable subprocess. The parent process performs no CUDA runtime
  inspection before Lightning creates its workers. CPU learner DDP combined
  with native CUDA rollout is rejected because the port has no explicit
  rank-local rollout-device mapping.
- Stochastic rollout is keyed by game seed. A vector group owns one generator
  per environment, shared by that environment's two self-play seats, so
  regrouping games changes neither actions nor simulator trajectories while
  model forward and simulator stepping remain vectorized.
- A row-addressable grouped collector exception carries its row to the owning
  game and seed. Failures without a trustworthy row are reported against the
  complete affected group; distributed envelopes preserve the same identity
  on every rank.
- Finite checks keep per-tensor flags and recurrent-state norms on device.
  Healthy forward and gradient phases expose only one aggregate scalar (and
  one all-rank reduction under DDP); tensor names and state norms are
  materialized only after that aggregate reports a failure.

The fixed-budget acceptance is the production Lightning path: train through
two collection boundaries, checkpoint after the first, resume through the
second, require exact resumed/uninterrupted policy and clock parity, then run
eight fixed held-out economic games. The explicit gate is
`pytest tests/learn/test_toad_ddp.py -m economic_acceptance -v`; it is not
excluded by the default `not slow` selection.
