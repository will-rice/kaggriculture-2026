# The campaign script

Date: 2026-09-05. Supersedes `2026-09-05-async-loop-design.md` (the schedule) and
amends `2026-09-04-codex-campaign-design.md` §3 (the schedule only). Every
operator — mutation, validation, fast and deep evaluation, archive, pool,
gate rule, prompt — is unchanged.

## 1. The script

`loop.py` is four steps, and reads as four steps:

1. **Initialize wandb from `{model}-{hash}`.** The run id and name are both
   `f"{config.CODEX_MODEL}-{revision}"`, `resume="allow"`. The revision is
   the short HEAD hash read through GitPython; if `src/` has uncommitted
   changes the script refuses to start, naming the files. A restart on the
   same code resumes its run; a code change is a new run on the same
   `calls` axis.
2. **Start the event loop.** `asyncio.run(campaign(...))`.
3. **Async spin off N codex sessions.** A concurrent codex session _is_
   an island: `ISLANDS = CODEX_CONCURRENCY = N`, and island `i` is one
   long-lived coroutine that runs sessions back to back. There is no
   dispatcher and no semaphore. Each pass: check the daily cap (park until
   the day rolls over when spent) → plan on this island (UCB parent,
   sometimes a crossover partner) → build the sandbox → run codex (asyncio
   subprocess, own process group, cap, keep a written child, one retry on
   the fallback model) → validate → fast-evaluate → insert on this island →
   log the call → advance `state.calls` and write `state.json` → migrate /
   reset on the call counter → hand the child to the gate → next pass.
   On start, an island with no programs (the archive was built with fewer
   islands) is seeded from the current top program, or the floor when the
   archive is empty.
4. **Gate the improvements.** A child that enters the archive's top
   `DEEP_TOP_K` and has never been deep-scored gets a deep task, bounded by
   `asyncio.Semaphore(DEEP_CONCURRENCY)`. Its sealed result is stored on the
   program (an archive event, so it survives a restart) and logged. If it
   clears the promotion rule against the champion's current score, it is
   promoted at once: floor and champion file written, pool updated and
   saved, record written, artifact logged, git commit in a thread. The new
   champion is then re-scored on the post-promotion pool as a task and
   that becomes the comparison score.

Blocking work never runs on the loop thread: codex is an asyncio
subprocess; validate, fast, deep, the transcript parse, the sandbox removal
and the git commit are `asyncio.to_thread`. Archive, pool, state and the
wandb run are touched only from coroutines.

## 2. What is not there

- No epoch, no `EPOCH_INTERVAL`.
- No core permit counter. Every evaluation uses `workers = CORE_BUDGET //
N`. N fast evaluations plus `DEEP_CONCURRENCY` deep ones at once
  oversubscribe the budget by a bounded fraction; accepted.
- No dispatcher, no session semaphore, no per-island accounting: N
  island coroutines, each running one session at a time.
- No session records; a restart cancels sessions and kills their process
  groups, as today.
- `run(calls)` plans exactly `calls` sessions then drains; `--calls`.

## 3. Failure handling

A candidate's failure at any step is its lineage's failure, recorded with
the reason. A session that ends without a verdict (provider refusal, codex
death) is retried once on `CODEX_FALLBACK_MODEL`; a failure on both is
logged and the island is dispatched again, never charged to the lineage.
`OpponentCrash` stops the script: sessions cancelled and killed, `run`
raises. `DAILY_CALL_BUDGET` (800 for the ramp) parks dispatch until the day
rolls over.

## 4. Telemetry

Per call, axis `calls`: `calls/ok`, `calls/timed_out`, `calls/fallback`,
`calls/input_tokens`, `calls/output_tokens`, `calls/seconds`, `calls/today`,
`fast/fitness` (when evaluated), `archive/top`, `archive/programs`. Per deep
evaluation, axis `calls`: `deep/score`, `deep/low`, `deep/high`,
`deep/field`, `deep/rate/<opponent>`, `deep/held_out/<name>`,
`deep/champion`, `deep/rho_fast_deep` (Spearman over every scored pair),
`deep/promoted`.

## 5. Tests — real code, not mocks

Only the codex process is substituted (`FakeMutator`, or `CodexMutator`
with a real shell command). Real validate, fast, deep, harness, archive,
pool, gate, files under `tmp_path`, `wandb.init(mode="disabled")`.

1. End to end: two islands (so N=2), `calls=2`; the stronger child is
   deep-scored on insert and promoted with no cadence boundary; floor,
   champion file, pool as today's test; two call records, one deep record
   with `deep/promoted == 1`.
2. Four islands' sessions overlap (stamps from real `date` in the
   stand-in).
3. Resuming an archive built with fewer islands seeds each new island from
   the current top program and runs a session on it.
4. A spent cap parks dispatch without advancing `calls`; the day rolling
   over resumes it.
5. Cancellation kills the codex process group; `OpponentCrash` stops the
   run and its sessions are gone.
6. A program is deep-scored once; a restart reads the stored result.
7. After a promotion the champion is re-scored and the next comparison
   uses that score.
8. A provider failure is retried on the fallback and never records a
   lineage failure.
9. Dirty `src/` refuses to start; a clean tree names the run
   `<model>-<revision>` and a second launch resumes it.
10. Resume from `state.json`/`champion.json`.

Each test is shown red by a named mutation of the new code.

## 6. Size

`loop.py` under 350 lines including docstrings. If a sentence above costs
more than that, the four steps win and the sentence is dropped, named in
the commit message.
