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
allows, evaluations running as soon as their child exists, and a champion
that can be cut the moment a candidate proves itself, not at the next
hundred-call boundary. Nothing in the loop waits on anything it does not need.

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
  them when its process pool has joined. Deep evaluations are ordinary
  takers, so the gate never pauses dispatch; it competes for cores like
  everything else. The box is never oversubscribed by the loop's
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
- `calls % RESET_INTERVAL == 0` and the archive has a top → reset the worst
  island (instant).

There is no epoch. The epoch was the synchronous design's way of batching
expensive sealed evaluations; in an event loop the deep evaluation is
triggered by the event that matters and promotion happens the moment a
result clears the rule.

**The gate** (event-driven):

1. **Trigger.** When a child is inserted and stays in the archive's top
   `DEEP_TOP_K` by mean fast fitness, and it has never been deep-scored, a
   deep task is created for it. An `asyncio.Semaphore(DEEP_CONCURRENCY)`
   (2) bounds how many run at once, on top of the core permits each takes.
   A program is scored on the sealed block once; the result is stored on
   the program (`Program.deep: DeepResult | None`, an archive log event so
   it survives a restart).
2. **Promote on arrival.** When a deep result lands (on the loop), the
   promotion rule runs against the champion's current score. If it passes:
   `gate.promote`'s writes and pool save, the record and artifact, the git
   commit awaited in a thread — exactly the epoch's step 3, for one
   candidate. Promotions are serialised by the loop thread: the state is
   updated before the commit is awaited, so a second result arriving during
   the commit compares against the new champion.
3. **The champion's score follows the pool.** The pool changes only at a
   promotion. Right after one, a task re-scores the new champion on the
   post-promotion pool (its promotion result was measured on the previous
   pool) and stores that as `state.champion`. Until it lands, comparisons
   use the promotion-time result; the window is one deep evaluation long.
4. **Correlation.** `rho(fast, deep)` is Spearman over every (mean fast
   fitness, deep score) pair scored so far, logged with each deep record.

A promotion changes the pool while fast evaluations against the previous
pool may be in flight; their fitness is inserted as measured. The archive's
mean fitness already spans pool changes, so this is the same inconsistency
at a finer grain, accepted.

**Intervals are in calls.** `MIGRATION_INTERVAL = 40`, `RESET_INTERVAL =
160` (the old 10 and 40 iterations × 4 islands). `EPOCH_INTERVAL` is gone;
`DEEP_TOP_K = 3` keeps its meaning as the trigger's rank; `DEEP_CONCURRENCY
= 2` is new. `State.iteration` becomes `State.calls`.

## 5. Sessions, shutdown and resume

**A codex session is independent of the loop process.** The loop launches
it in its own process group and from then on merely tracks it. Every
session has a record under `run/campaign/sessions/<program_id>.json`:
island, parents, kind, model, start time, sandbox path, pid and process
group, written before the session starts and deleted when its call
completes.

- **Shutdown** (`SIGTERM`/`SIGINT`, or `OpponentCrash`) stops dispatching,
  abandons the in-flight evaluations and exits. It kills no session. The
  records and the sandboxes stay.
- **Start** scans the records. A session whose process is still alive is
  re-attached: a task waits for the process to exit (polling, since it is
  not the new loop's child), with the cap measured from the recorded start,
  then runs the normal chain — validate, fast, insert — and completes the
  call. A session whose process has already exited is harvested the same
  way from what it left in the sandbox. Re-attached sessions hold session
  permits, so the dispatcher only tops up to `CODEX_CONCURRENCY`.
- The timeout kill stays: a session past its cap is stuck, not
  independent. A restart therefore loses nothing but the seconds it takes,
  and loop code can be deployed at will.
- `state.json` is written after every completed call; `champion.json` by
  the gate. A `state.json` from the old loop has `iteration` and no
  `calls`: it loads as `calls = 0`, which only shifts the cadence.
- `run(calls, ...)` plans exactly `calls` new sessions, then drains
  in-flight tasks (they complete and count). `campaign loop --calls N`.

**One wandb run per code version.** The run id is `<model>-<short
revision>`, the same string as the name, with `resume="allow"`: a restart
on unchanged code resumes its run; a restart on changed code opens a new
one on the same `calls` axis, so versions are separate lines that line up.
The loop reads the revision from a clean tree and refuses to start from a
dirty one (uncommitted changes under `src/`), because a run named by a hash
must be that hash.

## 6. Failure handling

Unchanged in kind. A candidate's failure at any step is its lineage's failure
and is recorded with the reason. `OpponentCrash` (a pool opponent failing in
its own seat) is not a candidate's failure: it propagates out of the job or
a deep task, the dispatcher stops, in-flight sessions are left running
for the next start to harvest, and `run` raises. A session that ends without a verdict — the provider refused
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
evaluated), `archive/top`, `archive/programs`. One record per deep
evaluation, axis `calls`: `deep/score`, `deep/low`, `deep/high`,
`deep/field`, `deep/rate/<opponent>`, `deep/held_out/<name>`,
`deep/champion` (the score it was compared against), `deep/rho_fast_deep`,
`deep/promoted` (0/1). Run name and config as today.

## 8. Expected throughput

Eight sessions in flight, sessions averaging 15 minutes: about 32 calls an
hour, roughly 750 a day, against 12 an hour today. Fast evaluations (48
games at 7 workers, about a minute) and deep evaluations no longer sit on
the critical path of any session. A candidate can be cut within one deep
evaluation of its child landing.

## 9. Tests

1. End to end with the fake mutator: two islands, concurrency 2,
   `calls=2` → the stronger child is deep-scored on insert and promoted
   without any cadence boundary; the pool, floor and champion file are as
   the current end-to-end test asserts; wandb saw two call records and one
   deep record with `deep/promoted == 1`.
2. Concurrency: fake mutator sleeping 1 s, concurrency 4, `calls=4`, wall
   clock under 2.5 s.
3. Cores: with `CORE_BUDGET` patched to `workers`, two fast evaluations never
   overlap (a stub records entry/exit); with `2 * workers` they do.
4. A deep evaluation does not pause dispatch: with the core budget at one
   share, a call completes and logs while a deep task is queued behind it
   (the ordering construction already in `test_loop.py`).
5. A spent cap parks dispatch without advancing `calls` or the cadence; the
   day rolling over resumes it.
6. A session survives the loop: a loop killed with SIGTERM while a session
   (`bash -c` writing `child.py` after a sleep) is running leaves it
   running, and a fresh loop started against the same run directory
   re-attaches, harvests its child and inserts it, completing the call. A
   session that already exited before the restart is harvested the same
   way. The timeout still kills a session past its cap, process group and
   all.
7. `OpponentCrash` from a job stops the loop; its sessions are left running
   and their records remain.
8. A loop launched from a tree with uncommitted changes under `src/`
   refuses to start; one launched from a clean tree names its run
   `<model>-<revision>` and a second launch resumes that run.
9. Resume carries over in terms of `calls`. A program is deep-scored once:
   re-inserting it into the top K never creates a second task, and a
   restart reads its stored result from the archive log. After a
   promotion the champion is re-scored on the post-promotion pool and the
   next comparison uses that score.
10. Mutator tests await; timeout-keeps-child and timeout-kills-the-group still
    pass.

## 10. Out of scope

An asyncio-native harness; per-island session quotas; changing the operators
or the promotion rule; the tape evaluator.
