# Fully async campaign loop

Date: 2026-09-05. Amends the schedule in `2026-09-04-codex-campaign-design.md`
§3; every operator (mutation, validation, fast and deep evaluation, archive,
pool, gate, prompt) is unchanged.

## 1. Why

Algorithm 1 as implemented is a barrier: one iteration fires one codex call
per island and waits for the slowest session before migration, the epoch or
the next iteration can happen. Sessions take 8–25 minutes with a wide spread,
so four islands cap the loop at roughly 12 calls an hour whatever
`CODEX_CONCURRENCY` says, and the box idles between the first session to
finish and the last. The week's codex quota is 59% unspent with a day to go.

Goal: `CODEX_CONCURRENCY` sessions in flight at every moment the daily cap
allows, evaluations running as soon as their child exists, and epochs that
never stop dispatch. Nothing in the loop waits on anything it does not need.

## 2. Model

One `asyncio` event loop, one thread that owns all shared state. Every
blocking operation runs somewhere else and is awaited:

| Work                                      | Where it runs                           | How it is awaited                             |
| ----------------------------------------- | --------------------------------------- | --------------------------------------------- |
| codex session                             | child process, own process group        | `asyncio.create_subprocess_exec` + `wait_for` |
| `validate.validate`                       | its own spawned child (already)         | `asyncio.to_thread`                           |
| `evaluator.fast`, `evaluator.deep`        | `ProcessPoolExecutor` workers (already) | `asyncio.to_thread`                           |
| sandbox build, archive, pool, gate, wandb | the event loop                          | plain calls                                   |

The archive, the pool, the state and the wandb run are touched only from
coroutines on the loop thread, so they need no locks. The harness and the
evaluators stay synchronous process-pool code: the loop schedules, processes
do the CPU work. Making the harness itself asyncio-native would not add
throughput, because a game is one CPU-bound process either way.

## 3. Resources

Two budgets, both asyncio primitives on the loop:

- **Sessions.** `asyncio.Semaphore(CODEX_CONCURRENCY)`. Held for the life of
  a codex call only, not for the evaluation that follows, so a finished
  session's slot is refilled while its child is still being scored.
- **Cores.** A counter of `CORE_BUDGET` permits with an `asyncio.Condition`
  (`async with cores.take(n)`). Every evaluation takes the permits it will
  fan out over — fast and deep alike take `workers` permits — and releases
  them when its process pool has joined. An epoch's deep evaluations are
  ordinary takers, so an epoch never pauses dispatch; it just competes for
  cores like everything else. The box is never oversubscribed by the loop's
  own work.

Codex sessions run the harness on their own, outside the budget, at up to
`SANDBOX_WORKER_CAP` workers each; that cap drops from 8 to 4 so eight
sessions cannot take 64 cores between them.

## 4. Tasks

**Dispatcher** (one coroutine, runs for the life of the loop):

1. Acquire a session permit.
2. If the day's cap is spent, release it, sleep `QUOTA_SLEEP_SECONDS`, retry
   (the day rolls over on the calendar as today).
3. Pick the island with the fewest jobs in flight (ties: lowest index), plan
   there (`_plan`: UCB parent, sometimes a crossover partner), count the call
   against the day, and `create_task(job(...))`.

**Job** (one per call):

1. Build the sandbox; `await mutator(box, program_id)` (holds the session
   permit; releases it in `finally`).
2. No child → record the failure on the island, done.
3. `await to_thread(validate)`; a failed verdict → record, done.
4. Store the child; `async with cores.take(workers)`: `await to_thread(fast)`.
   `OpponentCrash` propagates (§6). Other `RuntimeError` → record, done.
5. Insert the program on the loop; log the call's wandb record; advance
   `state.calls`; write `state.json`; run the cadence (below).

**Cadence** (on the loop, in the job's completion step, keyed on
`state.calls`, which counts completed calls ever):

- `calls % MIGRATION_INTERVAL == 0` → `archive.migrate()` (instant).
- `calls % EPOCH_INTERVAL == 0` → `create_task(epoch())` unless an epoch is
  already running, in which case log and skip; the next crossing fires it.
- `calls % RESET_INTERVAL == 0` and the archive has a top → reset the worst
  island (instant).

**Epoch** (a task like any other):

1. Snapshot `top = archive.top(DEEP_TOP_K)` and the champion path.
2. `gather` the deep evaluations — champion re-measure plus each candidate —
   each as its own coroutine that takes `workers` permits and awaits
   `to_thread(deep)`. They overlap with each other and with jobs.
3. On the loop: Spearman, the promotion rule, the file writes and pool save
   of `gate.promote`, the wandb epoch record and artifact — all milliseconds.
   The promotion's git commit is the one call that is not: `promote` takes a
   `commit` callable and the epoch awaits it through `to_thread`, so the loop
   never blocks on a subprocess. A promotion changes the pool while fast evaluations against
   the previous pool may be in flight; their fitness is inserted as measured.
   The archive's mean fitness already spans pool changes across epochs, so
   this is the same inconsistency at a finer grain, accepted.

**Intervals are in calls.** `MIGRATION_INTERVAL = 40`, `EPOCH_INTERVAL = 100`,
`RESET_INTERVAL = 160` (the old 10, 25, 40 iterations × 4 islands; the
wall-clock cadence is unchanged). `State.iteration` becomes `State.calls`.

## 5. Shutdown and resume

- `SIGTERM`/`SIGINT` cancel every task. A cancelled codex call kills its
  process group, so a restart never orphans sessions (today it does, and a
  `pkill` of the loop leaves eight sessions running).
- `state.json` is written after every completed call; `champion.json` by the
  gate as today. A restart resumes `calls` and the champion and loses only
  the sessions in flight. A `state.json` from the old loop has `iteration`
  and no `calls`: it loads as `calls = 0`, which only shifts the cadence.
- `run(calls, ...)` runs until `state.calls` reaches the target, then drains
  in-flight tasks (they complete and count). `campaign loop --calls N`.

## 6. Failure handling

Unchanged in kind. A candidate's failure at any step is its lineage's failure
and is recorded with the reason. `OpponentCrash` (a pool opponent failing in
its own seat) is not a candidate's failure: it propagates out of the job or
the epoch, the dispatcher stops, in-flight sessions are killed, and `run`
raises. A session that ends without a verdict — the provider refused
(`gpt-6-astra` answers "at capacity" some of the time) or codex died — is
retried once on `CODEX_FALLBACK_MODEL` (`gpt-5.6-sol`) in the same sandbox,
and the call records which model produced its child (`calls/fallback`). A
failure on both models is logged with the provider's message and the island
is dispatched again; it is never the lineage's failure and writes no
failure line. `DAILY_CALL_BUDGET` is the guard that keeps a spent quota from
becoming hours of such calls, and is 800 for this week's ramp.

## 7. Telemetry

One wandb record per completed call, step metric `calls`: `calls/ok`,
`calls/timed_out`, `calls/input_tokens`, `calls/output_tokens`,
`calls/seconds`, `calls/today`, `calls/in_flight`, `fast/fitness` (when
evaluated), `archive/top`, `archive/programs`. The epoch record is unchanged
except its axis is `calls`. Run name and config as today.

## 8. Expected throughput

Eight sessions in flight, sessions averaging 15 minutes: about 32 calls an
hour, roughly 750 a day, against 12 an hour today. Fast evaluations (48
games at 7 workers, about a minute) and epochs no longer sit on the critical
path of any session.

## 9. Tests

1. End to end with the fake mutator: two islands, concurrency 2,
   `EPOCH_INTERVAL` 2, `calls=2` → promotes; the pool, floor and champion
   file are as the current end-to-end test asserts; wandb saw two call
   records and one epoch record with `deep/promoted == 1`.
2. Concurrency: fake mutator sleeping 1 s, concurrency 4, `calls=4`, wall
   clock under 2.5 s.
3. Cores: with `CORE_BUDGET` patched to `workers`, two fast evaluations never
   overlap (a stub records entry/exit); with `2 * workers` they do.
4. The epoch does not pause dispatch: a stubbed epoch parked on an event
   while a new job starts and completes.
5. A spent cap parks dispatch without advancing `calls` or the cadence; the
   day rolling over resumes it.
6. Cancellation kills the codex process group (the existing `bash -c "sleep
30 & sleep 30"` pattern); no member of the pgid survives.
7. `OpponentCrash` from a job stops the loop, and in-flight sessions are gone.
8. Resume, champion re-scoring and the core-budget-for-deep tests carry over
   in terms of `calls`.
9. Mutator tests await; timeout-keeps-child and kill-the-group still pass.

## 10. Out of scope

An asyncio-native harness; per-island session quotas; changing the operators
or the promotion rule; the tape evaluator.
