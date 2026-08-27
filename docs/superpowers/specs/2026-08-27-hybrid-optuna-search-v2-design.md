# Hybrid Optuna search v2: semantic tuning with progressive league evidence

**Design, 2026-08-27.** This design replaces the hybrid agent's ineffective
one-hot evolutionary search with a resumable Optuna study over the actual
typed policy parameters. It supersedes only the optimizer and search protocol
in sections 6 and 7 of the
[hybrid rule-agent design](./2026-08-25-hybrid-rule-agent-design.md). The
hybrid runtime, canonical features, public-frontier evidence, promotion gate,
packaging rules, and submission boundary remain unchanged.

## 1. Why replace the current search

The first production search completed eight valid generations and is retained
as evidence, but it did not produce a viable policy. Every developed elite
scored zero win points against every one of the 11 league members. Its best
normalized margin improved from about `-0.91948` to `-0.83970`, but that
improvement did not cross the discrete result boundary that matters.

The problem is the optimizer representation and evaluation schedule, not a
crash or invalid artifact:

- the runtime policy has roughly 59 semantic choices, while `GenomeCodec`
  expands them into 3,617 coordinates;
- 3,606 of those coordinates are categorical logits, including 33 unrelated
  101-way groups;
- Gaussian mutation perturbs every logit, so a child commonly changes most
  discrete choices at once instead of making one intelligible policy change;
- the flat zero-win objective leaves strength weights inert, after which the
  old ranker falls back to a global margin diagnostic;
- every candidate pays for all 11 opponents before the search has established
  that it can score against an attainable baseline;
- each candidate/opponent evaluation creates new process pools, making the
  450,560-game plan much more expensive than its useful evidence warrants.

The immutable generation-8 state remains at its original path for audit, with
SHA-256 `c74ae9ff99fc6bd1d39c93f2edda8ac3a9ee1894d1422c2e02e9c6d02aab5c5b`.
Search v2 neither migrates nor mutates it. Search v2 gets a new study identity,
storage directory, artifact schema, and W&B run.

## 2. Goals and non-goals

### Goals

1. Tune the exact `HybridConfig` fields directly with types and domains that
   match Pydantic validation.
2. Learn from dense paired-margin evidence while preserving win points as the
   lexicographically primary objective.
3. Spend full-league games only on candidates that survive cheaper rungs.
4. Resume completed trials from durable local storage without replaying them.
5. Keep one deterministic Optuna coordinator while parallelizing only
   independent episodes.
6. Stream useful, non-duplicated search telemetry to one resumable W&B run.
7. Produce a certified finalist artifact that the untouched promotion boundary
   can validate without trusting Optuna or W&B implicitly.
8. Run a bounded real pilot before committing to the complete trial budget.

### Non-goals

- No reuse of the one-hot genome as Optuna's parameter space.
- No mutation of the old evolution database or its frontier snapshot.
- No multi-objective Pareto study; promotion still needs one canonical order.
- No concurrent Optuna optimizers or distributed database writers.
- No GPU use and no interruption of the single-GPU Toad run.
- No use of promotion seeds during tuning, pilot validation, or finalist
  selection.
- No automatic edit to `main.py`, packaging, Kaggle submission, or weakening
  of the Task 9 promotion gate.

## 3. Architecture

The v2 flow has five isolated components:

```text
verified frontier + immutable source snapshots
                    |
                    v
          typed semantic parameter space
                    |
                    v
      one Optuna coordinator / SQLite study
                    |
                    v
       persistent CPU episode arena (32 workers)
                    |
          +---------+---------+
          |                   |
          v                   v
  atomic trial evidence   one W&B run
          |
          v
  certified Optuna finalist artifact
          |
          v
  existing untouched promotion boundary
```

The coordinator is the only process allowed to ask for parameters, report
intermediate values, prune trials, write study state, or publish finalist
metadata. The arena workers only execute fixed `(candidate, opponent, seed,
seat)` games and return results in canonical input order. SQLite and the
fsynced evidence files are authoritative. W&B observes those results but is
never an input to sampling, resumption, certification, or promotion.

Search v2 is added beside the old `evolution.py` path. The old search loader,
state schema, CLI, and evidence remain readable so the failed experiment stays
auditable. Shared frontier verification, source snapshotting, `HybridConfig`,
fitness primitives, and promotion seed definitions are reused rather than
forked.

## 4. Semantic parameter space

Optuna suggests one value for each actual authoring field. There is no flat
genome and no categorical logit vector.

- ordered counts, days, targets, reserves, and liquidation day use
  `suggest_int`, including the exact step of 100 for cash reserves;
- crop names use `suggest_categorical` over canonical `CROP_NAMES`;
- job and market weights use bounded `suggest_float(0.0, 10.0)`;
- constant domains, including the first phase's fixed start day, are copied
  into the payload rather than presented as fake choices;
- tuple ordering and all cross-field constraints are validated by constructing
  `HybridConfig` before any game starts.

Parameter names are stable dotted paths such as
`opening.phase_1.target_hands`, `opening.phase_2.crop.STRAWBERRY`,
`jobs.recovery`, and `market.live_price`. A canonical parameter-space manifest
records every name, type, domain, step, and mapping to a `HybridConfig` path.
Its SHA-256 is part of the study identity. Adding, removing, renaming, or
changing the domain of a parameter therefore requires a new study.

The runtime and serialized winning policy still use the current validated
Pydantic model and its frozen standard-library `RuntimeConfig`. Canonical
one-hot encoding remains available at the RL/config boundary where it is part
of the model representation; it is not an optimizer coordinate system.

## 5. Sampler and starting trials

The study uses one `TPESampler` with:

```text
seed = 20260827
multivariate = true
group = true
n_startup_trials = 16
```

Optuna runs with `n_jobs=1`. This keeps the trial order stable, gives one owner
to SQLite and W&B, and avoids claiming deterministic behavior from concurrent
TPE suggestions. Parallelism is confined to episode execution inside one
objective call. The locked Optuna version and every sampler constructor value
are recorded in the study identity; upgrading Optuna requires a new study.

Before TPE suggestions begin, the coordinator enqueues eight exact, unique,
validated configurations in canonical order:

1. `HybridConfig.default()`;
2. the four generation-8 elites from the immutable v1 state;
3. three checked-in hand-authored strategies: an economic-balanced seed, a
   production-aggressive seed, and a conservative-liquidation seed.

The three deliberate seeds are full literal `HybridConfig` payloads, not
runtime mutations or partially specified overrides. Tests pin their canonical
JSON and hashes. Loading v1 elites validates the v1 terminal state and records
its file digest, but does not make the v1 study part of v2 resume state.
Duplicate decoded configurations are a preflight error rather than silently
changing the promised eight-trial warm start.

## 6. Progressive evaluation and pruning

Every game is seat-swapped and all candidates at a rung use the same fixed
seeds. Search v2 uses nested development-only banks:

```text
rung 1: 860000..860003
rung 2: 860000..860015
rung 3: 860000..860031
```

These seeds are disjoint from the existing frontier, v1 screening, v1
development, determinism, and promotion banks. The promotion seeds remain
exactly the protected Task 9 set. A later rung reuses prior game cells and runs
only the missing opponent/seed/seat combinations.

The opponent panels are derived deterministically from the verified strongest-
first frontier report:

- **Rung 1:** `economic_policy` plus the two weakest distinct public opponents,
  4 seeds, for 24 games per trial.
- **Rung 2:** the rung-1 panel plus Boatlee v14, the measured public frontier,
  and the median-ranked distinct opponent, 16 seeds, for 192 cumulative games
  and 168 new games per surviving trial. If named roles overlap, fill the panel
  to six from weakest to strongest without duplicates.
- **Rung 3:** all 11 verified opponents, 32 seeds, for 704 cumulative games and
  512 new games per surviving trial.

The exact opponent names, source hashes, seed tuples, and panel derivation are
stored in the study identity before trial zero.

`SuccessiveHalvingPruner(min_resource=1, reduction_factor=4,
min_early_stopping_rate=0)` receives intermediate results at resource steps 1,
4, and 16. This is asynchronous successive halving, so approximately one
quarter rather than an exact fixed fraction advances at each boundary. The
coordinator always finishes and durably records a complete rung before asking
the pruner. A pruned trial remains valid evidence and is never silently
reported as a failed trial.

For 512 trials, a one-quarter survival rate costs roughly 50,200 distinct games
instead of the old 450,560-game plan. Actual advancement and total games are
logged, not assumed.

## 7. Objective and canonical ranking

Each rung records, per opponent, wins, draws, losses, win points, game count,
paired normalized margin, runtime, and failures. Any illegal action, exception,
timeout, malformed result, or non-DONE episode makes the trial `FAIL`; it cannot
receive a finite optimization value.

For clean trials, retain the existing all-opponent primary score over the
current rung panel:

```text
primary = 0.70 * strength_weighted_mean(win_points)
        + 0.30 * min(win_points)
```

The dense tie-break is the strength-weighted mean paired normalized margin,
mapped from `[-1, 1]` to `[0, 1]`. Optuna receives:

```text
objective = primary + 1e-6 * dense_margin_0_to_1
```

At every rung's finite game count, `1e-6` is smaller than the minimum possible
nonzero change in `primary`. Consequently margin can order candidates with the
same discrete result evidence, but it cannot outrank a real win-point
improvement. Tests enumerate each rung's minimum score increment and prove the
lexicographic property rather than relying on an informal epsilon assumption.

The final canonical order is: eligibility, rung reached, primary score, dense
margin, lower runtime, canonical config digest. Finalists must have completed
rung 3. Rung-1 or rung-2 promise is not promotion evidence.

## 8. Arena execution

The current `arena.outcomes` repeatedly constructs process pools per opponent
and seat. Search v2 adds a context-managed persistent arena with one
`ProcessPoolExecutor(max_workers=32)` by default. The coordinator flattens a
whole rung into canonical `(opponent rank, seed, candidate seat)` tasks, maps
them through the same pool, and reconstructs results by explicit provenance.

The default of 32 workers uses half of the 64-logical-core host and leaves
headroom for Toad and the operating system. The accepted operational range is
1–32. Worker count is operational metadata, not semantic study identity:
changing it on resume is permitted because task inputs and canonical output
ordering do not change. There is no internal `os.nice` call. The CLI reports
the resolved worker count before creating the pool.

Every returned row includes opponent, seed, seat, terminal statuses, rewards,
runtime, and any failure detail. Missing, duplicate, or reordered provenance is
an error. The persistent implementation must match the existing reference
arena byte-for-byte on fixed candidates and seeds before it can be used in a
real pilot.

## 9. Durability, identity, and resume

The default run root is `run/hybrid/optuna-v2/` and contains:

- `study.sqlite3`, the Optuna RDB storage;
- `identity.json`, the canonical immutable semantic identity;
- `evidence/trial-<number>-rung-<number>.json`, atomic rung evidence;
- `diagnostic.json`, written only by a stop gate;
- `finalists.json`, written only after the full target completes;
- a content-addressed immutable league snapshot owned by the existing
  certification boundary.

The study name is `hybrid-optuna-v2`. The identity includes engine version,
frontier manifest/report hashes, source and snapshot hashes, parameter-space
hash, sampler and pruner settings, sampler seed, every search seed tuple,
opponent panels, strength weights, objective schema, enqueued config hashes,
and the immutable maximum target of 512 trials. The identity digest is stored
both in `identity.json` and as an Optuna study user attribute; resume requires
exact equality. W&B
project/entity/run name, logging enablement, worker count, and display settings
are operational fields and may change without changing game evidence.

Rung evidence is canonical JSON written atomically and fsynced before the
coordinator reports the intermediate value or completes the trial. Completed,
pruned, and failed Optuna trials are immutable. On startup, any stale `RUNNING`
trial left by interruption is marked `FAIL` with an interruption reason;
partial evidence is retained for diagnosis but never reused as a completed
rung. The trial budget counts every terminal Optuna trial so repeated failures
cannot create an unbounded run.

A resumed study preserves all completed trial parameters, values, states, and
evidence exactly. With a fixed sampler seed and sequential optimization, normal
resumes are expected to continue consistently and are covered by a split-run
test. The system does **not** claim that future TPE suggestions after an
arbitrary crash are byte-for-byte identical to an uninterrupted in-memory
process; Optuna does not guarantee that stronger property. Certification binds
the trials that actually ran, not a counterfactual suggestion stream.

## 10. W&B telemetry

Search v2 uses one persistent Optuna W&B integration callback with
`as_multirun=False`, a deterministic run ID derived from the study identity,
and resume enabled. `wandb.init` occurs at most once per process and
`wandb.finish` only when the coordinator exits. It never creates one W&B run
per trial or per generation.

The W&B run records:

- semantic parameters and Optuna trial number/state;
- rung reached, resource step, primary score, dense margin, and objective;
- per-opponent wins/draws/losses, win points, paired margin, and failures;
- games completed, rung advancement counts, trial throughput, wall time, and
  ETA;
- current best trial/config digest and its economic, Boatlee, frontier, worst,
  and full-league results;
- study identity, engine, manifest, source hashes, seed-bank hashes, commit,
  and SQLite path as immutable run config.

Optuna trial number is the W&B step axis. Only newly terminal trials are logged
on resume, preventing the duplicate-step behavior of the old polling sidecar.
The official callback is wrapped by a failure-isolation boundary: a W&B import,
authentication, network, initialization, or logging failure emits a local
warning and disables external logging for that process, but cannot fail,
prune, reorder, or retry an Optuna trial. SQLite and evidence files remain the
source of truth.

## 11. Pilot, full-run, and diagnostic gates

The permanent study is created with an immutable maximum target of 512 trials.
The first invocation uses an operational `--stop-after 32` boundary. It is a
real pilot, not a throwaway benchmark. Full search removes that early stop and
resumes the same study to its target of 512 terminal trials only when all of
these hold:

1. the persistent arena matches the existing reference arena;
2. the study stops and resumes without changing completed trial evidence;
3. no engine failures, illegal actions, missing provenance, or database
   inconsistencies occur;
4. at least one non-default trial beats the enqueued default's dense objective
   at the same rung;
5. W&B shows one resumable run with unique trial steps and the required
   per-opponent telemetry;
6. Toad remains healthy and no search process appears on the GPU.

After 128 terminal trials, the coordinator checks whether any clean candidate
has scored nonzero win points against `economic_policy`. If none has, it writes
`diagnostic.json`, closes the pool and W&B run, and stops without producing
finalists. This prevents another multi-day run from optimizing only a losing
margin surface. Continuing after that gate requires a new reviewed search
design or an explicit diagnostic override that is recorded in a new study
identity.

The complete search stops at 512 terminal trials. It validates all rung-3
evidence, checkpoints and closes SQLite so no WAL bytes are outstanding, hashes
the closed database, writes finalists atomically, and exits. It does not invoke
promotion.

## 12. Finalist and promotion trust boundary

Search v2 writes a new explicit `optuna-finalists` schema rather than forging a
legacy evolutionary `SearchState`. The artifact contains:

- schema and study identity digests;
- SQLite file digest and canonical terminal-trial summary;
- engine, manifest, report, source, and snapshot hashes;
- parameter-space, sampler, pruner, objective, seed, panel, and weight facts;
- each finalist's trial number, exact semantic parameters, canonical validated
  `HybridConfig`, config digest, full rung-3 matchup evidence, and ranking
  components;
- total trial-state counts and the absence of promotion-seed use.

The existing Task 9 promotion loader gains a second, narrowly validated input
schema. It accepts either the legacy certified evolution artifact or the new
certified Optuna artifact, never an ambiguous hybrid. For Optuna input it
recomputes identity hashes, config validation, trial evidence aggregates,
objective/rank order, source snapshot bindings, seed coverage, and SQLite
terminal-trial correspondence before the first holdout game. Any discrepancy
fails before claiming promotion seeds.

Promotion continues to use the existing 128 untouched seat-swapped seeds,
paired bootstrap gates, single-use claim protocol, deterministic action traces,
package verification, and submission separation. Search or W&B results cannot
waive any promotion requirement.

## 13. Error handling

- Parameter construction or Pydantic validation errors fail the trial before
  arena work and record the exact field error.
- A game exception or invalid terminal status fails the complete trial; partial
  wins cannot compensate for execution failure.
- Worker-pool failure closes the arena and stops the coordinator after marking
  the active trial failed; it is not silently recreated inside an objective.
- SQLite or evidence-write failure stops the search immediately because
  durability is authoritative.
- Identity mismatch, source mutation, report drift, seed overlap, stale RUNNING
  trial inconsistency, or evidence/SQLite disagreement stops before new games.
- W&B failures are the sole failure class allowed to degrade to local-only
  operation.
- SIGINT/SIGTERM closes W&B and the arena at a bounded boundary. The current
  trial may become a reconciled interrupted failure; completed trials remain
  resumable.

## 14. Testing and acceptance

### Parameter and objective tests

- Round-trip every suggested field through `HybridConfig` and `RuntimeConfig`.
- Prove exact domains, steps, constant handling, stable names, and space hash.
- Reject extra, missing, non-finite, invalid, or constraint-breaking values
  before play.
- Prove one primary win-point increment outranks the largest possible margin
  difference at every rung.
- Pin the eight unique enqueued configurations and v1 elite source digest.

### Arena and pruning tests

- Differential-test persistent and legacy arena results, ordering, margins,
  failures, and provenance.
- Prove one executor serves multiple candidates and is closed exactly once.
- Pin deterministic panel derivation and all seed disjointness.
- Exercise complete, pruned, failed, and interrupted trials through all rungs.
- Confirm the 128-trial zero-economic-score stop gate cannot write finalists.

### Storage and W&B tests

- Resume from SQLite without re-running completed/pruned/failed trials.
- Reject every semantic identity drift before worker creation.
- Permit only documented operational overrides.
- Reconcile stale RUNNING trials atomically and preserve partial diagnostics.
- Inject W&B import/init/log/finish failures and prove the study result is
  unchanged.
- Prove a resume uses one W&B run and logs each terminal trial step once.

### Finalist and integration tests

- Recompute every finalist field from SQLite plus atomic evidence.
- Reject config, rank, source, seed, objective, trial-state, or database
  tampering before promotion claims.
- Run a fresh 32-trial reference-engine pilot, stop, resume, and validate W&B
  telemetry and Toad/GPU isolation.
- Only after the pilot gate passes, resume the same certified study to 512.
- Run the existing untouched promotion suite only on qualifying rung-3
  finalists.

## 15. Delivery order

1. Add the pinned Optuna and Optuna-W&B integration dependencies.
2. Implement and test the semantic parameter-space manifest and deliberate
   seed configurations.
3. Add the persistent provenance-preserving arena.
4. Implement objective, panels, successive-halving reports, SQLite identity,
   atomic evidence, and resume reconciliation.
5. Add the one-run failure-isolated W&B callbacks.
6. Add the certified Optuna finalist schema and Task 9 validation adapter.
7. Run the 32-trial pilot and review its evidence.
8. If every pilot gate passes, resume the same study to 512 and monitor it.
9. Invoke untouched promotion only for qualifying finalists; package and
   submit only after that independent gate passes.
