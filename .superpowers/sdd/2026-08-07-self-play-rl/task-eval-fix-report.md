# The evaluator: rendering the curve, watching every arm, and a reference line

Three defects, one instrument. The evaluator became the project's primary
measurement the moment its first records showed the self-play numbers were a
private equilibrium; it was, at that moment, discarding every record it wrote,
watching one arm out of four, and had nothing to be read against.

Commit: `44df756` -- _fix: the eval curve was never rendering, and three arms
were unwatched_. One file, `src/kaggriculture/learn/scripts/evaluate.py`.

## Task 1 -- the curve was empty, and for a worse reason than the brief said

The brief said wandb silently drops records whose step decreases, which is true
and was happening: the daemon walks a prefix newest-first, so after logging
update 200 it logged 175 and 150, and wandb rejected both with

```
wandb: WARNING Tried to log to step 175 that is less than the current step 200.
```

But the run rendered **zero** points, not one. The measurement that found this:

```
run a5bundp1 (toad-eval-c3)  rows=0   summary={}
```

Four records on disk, no summary at all -- so the surviving step-200 record was
not rendering either. The second defect is that `wandb.log(data, step=n)` does
not commit; the partial history at step _n_ is flushed only when a later step
arrives. Every subsequent step was rejected as decreasing, so the newest record
sat uncommitted forever. The two bugs compose into total data loss: a
newest-first walker using `step=` publishes nothing, ever.

Both die to the same fix -- declare the axis and stop passing `step=`:

```python
run.define_metric("eval/ckpt_step")
run.define_metric("eval/*", step_metric="eval/ckpt_step")
...
payload["eval/ckpt_step"] = record["update"]
run.log(payload)          # no step=, so every call commits
```

The checkpoint's update number now travels as a logged _value_ on a declared
x-axis rather than as wandb's monotonic counter, which makes arrival order
irrelevant by construction rather than by the daemon being careful about it.

`recorded()` replays every record already in the arm's jsonl into the run at
startup, reading across every log the arm has ever written (the daemon opens a
new file each launch) and keeping one record per checkpoint. A fresh run
therefore opens with the whole history instead of growing a point at a time.

### Verification (wandb API, row counts -- not eyeballed)

| run                                         | before | after         |
| ------------------------------------------- | ------ | ------------- |
| `a5bundp1` toad-eval-c3 (old, killed)       | 0 rows | 0 rows (dead) |
| `9bsqbkb8` eval-toad-phase1c3-clone-teacher | --     | **5 rows**    |

The new run's x values, in the order they were logged:

```
ckpt_step = [125, 150, 175, 200, 100]
bank_mean = [7.125, 10.625, 10.3125, 8.4375, 3.75]
```

The last row is the point that matters: update **100** logged _after_ update
200 and retained. That is the exact write the old code dropped.

## Task 2 -- one daemon, all arms

The daemon took one prefix per process. Arms c5a, c5b and c6 had no evaluator
at all, and each arm that did have one paid for its own nice'd python and its
own corpus load for the same sixteen episodes.

It now takes a list of prefixes, opens one wandb run per arm from the single
process (`reinit="create_new"`), and each poll picks the newest unmeasured
checkpoint **across** all arms:

```python
checkpoint, prefix = max(pending, key=lambda pair: pair[0].stat().st_mtime)
```

By write time, not by update number -- update numbers restart per arm and are
not comparable across them. Every record carries an `arm` field.

A list of explicit prefixes rather than a `toad-phase1c*` glob: the glob would
also sweep up the dead c2/c4 arms, and the arms are a decision, not a pattern.

### Launch

Old single-prefix daemon (pids 2451704/2451707) killed; `ps` confirmed no
evaluator running before relaunch. Committed, then `setsid`:

- pid file `/data/kaggriculture/toad/eval_arms.pid` -> **2462377** (uv),
  **2462386** (python)
- log `/data/kaggriculture/toad/eval_arms.log`
- watching `toad-phase1c3-clone-teacher`, `toad-phase1c5a-kl-mid`,
  `toad-phase1c5b-money-x10-signed`, `toad-phase1c6-opponent-mix`

Confirmed by the first record written, not by the process starting:

```json
{
  "arm": "toad-phase1c3-clone-teacher",
  "update": 100,
  "checkpoint": "toad-phase1c3-clone-teacher_000100.pt",
  "eval_bank_mean": 3.75,
  "eval_bank_max": 36.0,
  "opponent_bank_mean": 159386.875,
  "win_rate": 0.0,
  "eval_percentile": 0.2525,
  "games": 16,
  "seconds": 390.1
}
```

Written to `/data/kaggriculture/toad/eval_toad-phase1c3-clone-teacher.jsonl` at
15:42:08. The daemon then immediately picked up
`toad-phase1c3-clone-teacher_000225.pt` -- a checkpoint the c3 run had written
while that eval was in flight -- which is the newest-first behaviour working:
fresh beats backfill.

c5a, c5b and c6 have **no checkpoints yet** (the runs started 15:23-15:27 and
have not reached update 25), so their runs stand at 0 rows by correctness, not
by failure. They will fill as those arms checkpoint.

## Task 3 -- the reference line

`--once` measures a single checkpoint that has no training curve of its own.

```
/data/kaggriculture/selfplay/resumed-3cc6068-1786139297/policy.pt
```

**Architecture inferred: 8 blocks, 256 channels**, from a bare `OrderedDict`
(no `learner` key) -- read off `stem.weight` being `[256, 48, 3, 3]` and the
highest `blocks.N.` index. `_load` already inferred both rather than assuming
`BLOCKS`/`CHANNELS`, so no per-arm flag was needed; it was confirmed against
all three lineages, and the 128-channel `phase1_*` checkpoints load through the
same path.

Result, run `eval-reference-old-ppo` (`kuoinfmi`), 2 rows:

```
ours 21,662  vs economic_policy 156,794   win rate 0.00   percentile 3.8
bank_max 26,443   16 games   384.5s
```

**21,662 against the gate-measured ~22,450** -- the same neighbourhood, and the
gap is what sixteen episodes versus the gate's 128 should produce. The load
path is sound; had it landed at ~10 it would have been an architecture
mismatch, and that is exactly the failure the inferred constants make
structurally impossible.

The single measurement is published at `eval/ckpt_step` 0 and 2000
(`REFERENCE_SPAN`) so it draws flat across the arms' axis rather than as a lone
point at the origin. Both rows are the same measurement, and the code says so.

## What the numbers say

This is the finding, not a side note. On the same axes, against the same
opponent, in the same pipeline:

|                                   | bank vs economic_policy |
| --------------------------------- | ----------------------- |
| old shaped-PPO `policy.pt`        | **21,662**              |
| c3 clone-teacher, updates 100-225 | **4 - 11**              |
| economic_policy itself            | ~157,000                |

The c3 lineage is not a weaker version of the old PPO agent. It is three orders
of magnitude below it and flat across 100 updates, and it never wins a game.
The old shaped-PPO checkpoint loses to `economic_policy` too, and always did --
21,662 against 156,794 is not a rung it cleared. The reference line's value is
that it is a floor the clone lineage is nowhere near, not that it is good.

## Concerns

1. **The corpus reference median is 114,404 against an expected 125,773.** The
   evaluator logs the discrepancy at every startup and proceeds. Pre-existing,
   untouched, and the percentile column is only as trustworthy as it is -- but
   at banks of 4 and of 21,662 the percentile is not the number anyone is
   reading, so it did not block this work.
2. **c5a/c5b/c6 are unverified end to end.** Their runs opened and the daemon
   globs for them correctly, but no checkpoint has existed to load, so nothing
   has proved that path with a real file. First confirmation arrives when those
   arms reach update 25.
3. **Sixteen episodes is a wide interval.** Fine for the shape of a curve and
   fine for a three-orders-of-magnitude gap; not fine for adjudicating between
   arms that land close together. The gate's 128 exists for that.
4. **Backfill re-logs the full history on every relaunch.** Correct for a fresh
   run, and cheap, but it means run-to-run comparisons should use the newest
   run per arm rather than assuming one run per arm forever.
5. **Stale pid file** `/data/kaggriculture/toad/eval_c3.pid` names dead
   processes from the 13:54 launch. Left in place -- not written by this task.
   `eval_arms.pid` is the live one.
