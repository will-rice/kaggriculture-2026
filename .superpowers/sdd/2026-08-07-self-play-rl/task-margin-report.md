# Arm M — the margin reward, with the opponent mix

Worktree `agent-a787fb13772848793`, branch `worktree-agent-a787fb13772848793`.
Commits `b6b0b89` (the arm) and `32272c2` (a metric the arm's own first records
showed was missing). Run `toad-phase1m-margin`, pid 2700820,
<https://wandb.ai/will-rice/kaggriculture-2026/runs/ne7rz7sx>.

## Prerequisite: the engine

The worktree was on kaggle-environments **1.32.3**. Bumped its pyproject to
`>=1.32.6` and synced; **1.32.6 installed and `townCenterSellInterval == 24`
verified from the engine's own schema and from the live configuration**.

`src/kaggriculture/constants.py` was mirrored from the main tree, so the sell
intervals are now read from `kaggriculture.json` rather than typed, and
`LEGACY_TOWN_CENTER_SELL_INTERVAL = 12` /
`LEGACY_TOWN_CENTER_DEMAND_SCHEDULE` exist for the pre-change archives.

One consequence surfaced immediately and is worth recording, because it was
_not_ a code change of ours: the `economic_policy` characterization tests failed
in the worktree and passed in the main tree. `economic_policy.py` is
byte-identical between the two — the main tree had already re-recorded the
goldens for 1.32.6 (149,176 / 147,181 / 108,290 against 1.32.3's 152,803 /
152,145 / 159,087). That fix and its fixture were mirrored in rather than
re-derived. The third seed moving 159,087 → 108,290 is a 32% fall in what the
reference banks; it is the size of the engine change, not of anything we did.

## The reward

```
R_t = ( MARGIN_WEIGHT   * Δ(our bank − their bank)
      + ABSOLUTE_WEIGHT * Δ(producing animals owned) ) / 500
R_719 += GAME_RESULT_WEIGHT * rank / 500
```

with **`MARGIN_WEIGHT = 0.001`**, **`ABSOLUTE_WEIGHT = 0.05`**,
`GAME_RESULT_WEIGHT = 10.0` (Toad's, unmodified) and the `/500` normaliser kept.
`clip_grads` stays 10.0 and `VALUE_BOUND` stays 1.0 — not tightened; the
ledger's ±1/MAX_DAYS entry is a trap, not a to-do.

**Margin term.** The literal win condition, decomposed onto the turns that
produced it. Both farms open on the same `startingMoney`, so the opening margin
is zero, no constant leaks in, and the 719 values telescope to the terminal
margin exactly. `MARGIN_WEIGHT` is the money weight the project already used,
because margin and bank are both coins: a 160,000 margin reads 0.32, inside the
0.13–0.31 band the shaped reward spanned and well inside the value bound.

**Absolute term, and why it is not coins.** Its only job is to stop the reward
being identically zero-sum, which is a self-play fixed point measured at 4 coins
of 200,000. The controller's mid-task finding decided its _form_: a seat's bank
correlates **+0.98** with its opponent's, so a bank-proportional floor is 98% a
reading of the shared market draw. Sizing made that concrete — at the
competitive operating point (median bank 80,660, median winning margin 2,561):

| term                                 | per episode                     |
| ------------------------------------ | ------------------------------- |
| margin `0.001 × 2,561`               | **+2.56**                       |
| terminal rank                        | **+10.0**                       |
| own bank at 0.0002 (my first choice) | +15.53 — six times the margin   |
| own bank at 0.00002                  | +1.55 — still 61% of it         |
| **capital count at 0.05 × 15**       | **+0.75 — 6% of the objective** |

A count does not scale with the draw, so its share stays ~6% whether the market
paid 80,000 or 160,000. FLG's precedent (`+0.3 per factory`) is a count too.
My original 0.0002-on-own-bank was wrong and would have rebuilt the failure mode
this arm exists to escape.

**Why it cannot pump, structurally.** The margin term already charges an
animal's full purchase price, which is 6x to 9x what the capital term credits:

|       | cost | margin charge | capital credit | net        |
| ----- | ---- | ------------- | -------------- | ---------- |
| GOOSE | 300  | −0.300        | +0.050         | **−0.250** |
| COW   | 400  | −0.400        | +0.050         | **−0.350** |
| SHEEP | 500  | −0.500        | +0.050         | **−0.450** |

The weight would have to exceed 0.30 before a buying spree paid for itself.
This is also what discharges the **Grześ** condition without a zeroing special
case: two of the three components are coins or a terminal rank and carry no
state to the horizon, and the third is only ever profitable if the capital
_earned_. Held produce is worth exactly nothing — `fuel` never enters.

**A latent pump found while proving that.** `Counts.capital` counted animal
_structures_. **`BUILD_COOP` and `BUILD_PASTURE` cost nothing in this engine** —
they need only an empty tile — so a structure count is roughly a hundred free
points a season, 0.01 after the normaliser against an objective worth 0.025.
The count is now of animals (`isinstance(tile, dict) and "animal" in tile`,
which is how the engine spells a placed animal). The same scan also silently
dropped the GOOSE, whose structure is a COOP, so buying a goose cost 300 coins
and _lost_ a point of capital when it was placed. Both are fixed and both are
guarded.

Toad's five shaped components are dropped from this arm's reward. They are still
computed on every trajectory, so `shaped_mean` remains the counterfactual.

## Opponent mix and the holdout

`--econ-fraction 0.5`, `economic_policy` as the scripted trainer, our seat's
trajectories only. `boatlee_v14_policy` and `kaito_v23_policy` are **held out**;
they do not exist in this worktree at all, which makes training against them
structurally impossible rather than merely unintended.

Kept from the arms proven to survive a resume: `--value-warmup` and
`--teacher-kl-cost 0.005`. Warmup matters more here than it did for C6/C7,
because those resumed a critic already trained on the reward they continued
with, and this one changes the reward under it.

**Resume tested against the real checkpoint**, not a synthetic one: restoring
`toad-phase1c3-clone-teacher_000200.pt` gives update 200, step 6,902,400 and
`lr = 6.54577e-05 = 1e-4 × decay(200)` exactly, where a reset would read 1e-4.

## Metrics

Per update, per population, never merged: `win_rate_vs_econ` (the decider),
`win_rate_mirror`, `margin_mean_*`, `bank_*`, `final_capital_*`,
`mean_sale_price_*`, `realisation_*`, `sales_*`, `units_sold_*`, `bought_*`,
plus `reward_mean` for the series actually being trained on.

Sale metrics run **every** update. Measured against a 17.5-second episode:
building the 719 snapshots costs 0.6 ms and scanning them 0.3 ms — **0.01%** of
collection, not the ~10% the brief budgeted for.

## Guards, and proof they discriminate

`tests/learn/test_toad_margin.py` (20) plus three wiring guards in
`test_toad_rollout`. Thirteen mutations applied one at a time, each observed red
on the named guard and restored green; full suite 449 passed, 1 skipped.

| break                                                    | caught by                           |
| -------------------------------------------------------- | ----------------------------------- |
| B1 margin weight → 0                                     | 9 tests                             |
| B2 absolute weight → 0 (zero-sum)                        | dead-heat, mirror-not-negations, +5 |
| B3 normaliser → 1.0                                      | 12 tests                            |
| B4 opponent bank read as our own                         | seat-reading guard                  |
| B5 opponent dropped from the delta                       | 6 tests                             |
| B6 shed stock paid at the horizon                        | held-produce (Grześ) guard          |
| B7 city term leaks in                                    | ignores-non-monetary guard          |
| B8 terminal rank on every turn                           | terminal-turn guard                 |
| B9 absolute weight 0.05 → 0.5 (past the purchase charge) | per-animal pump guards              |
| B10 absolute term back to own bank                       | 9 tests                             |
| B11 capital counts free structures                       | empty-structure + placement guards  |
| B12 placed animals stop counting                         | placement guards                    |
| W1 terminal state dropped from the series                | shape guard                         |
| W2 `final_capital` read from the opening state           | telescope guard                     |
| W3 sale metrics read the opponent's seat                 | own-seat guard                      |
| W4 margin field wired to the own-bank series             | telescope guard                     |

Two honest notes on the guards.

_The sale-metrics guard was worthless at first._ It ran on the untrained policy,
which has `SELL` legal on 0 of 719 turns, so every number was zero and stayed
zero under any wiring mistake — W3 passed. It now runs on the behaviour clone
(45 clears, 126 units, realisation 0.951), and cross-seat attribution lands at
21 coins a unit and 0.45 realisation, which the band separates. This is exactly
the `X == X` failure the brief warned about, found by insisting each break go red.

_One mutation I could not make discriminate, and it turned out not to be a bug._
`strict=True` on `zip(series[:-1], series[1:])` is **unreachable** — the two
halves are balanced by construction, so the flag guards nothing, here or in the
existing `shaped`/`_differences`. Real off-by-one protection is the length check
and the shape assertion.

_And one assumption of mine that measurement refuted._ I justified `_snapshot`
as dodging the engine's in-place mutation. It does not: `state[seat].observation`
is a fresh object with a fresh `farms` list every step, and retaining live
observations gives byte-identical metrics over a real episode. The alias warning
in `rollout`'s docstring is about `env.steps`, a different structure. `_snapshot`
is kept and re-justified on **memory** — 24 environments × 719 turns × both
seats × a 10×10 grid of tile dicts, in each of twelve workers.

## Launch

`setsid`, pid file `/data/kaggriculture/toad/phase1m_run.pid`, log
`phase1m_run.log`, jsonl `toad-phase1m-margin_1786225437.jsonl`, checkpoints
every 25 updates via the tested atomic write.

Three launches were needed and the reasons are worth recording, because two of
them were my instrument being wrong rather than the arm being wrong.

**Launch 1, killed deliberately at two updates** (archived under
`phase1m_aborted_no_capital_metric/`). Its own first records showed
`final_capital` was not logged — and "margin improves while capital and bank
both collapse" is a pre-registered read, so the number it needs had to exist
from update one rather than be reconstructed. That is what reading the first
record is for. Fixed in `32272c2`.

**Launch 2, which I twice misdiagnosed** (archived under `phase1m_attempts/`).
I wrote the _bash wrapper's_ pid into the pid file, because
`pgrep -f "toad_phase1 --margin"` matches the wrapper's command line too and
`head -1` took it. When a ten-minute blocking poll of mine hit the harness
timeout, the wrapper died and the pid file's process vanished — so I read
"alive: 0", saw a log ending cleanly at update 212 with no traceback, and
concluded the harness had killed the run. **It had not.** `setsid` had done its
job and the training process was still running with twelve workers; only my
handle on it was gone. I relaunched, and for about a minute two runs were live
against the same checkpoint prefix. Clearing that, I over-broadened a `pkill`
pattern and took down both.

Net cost: ~15 minutes and the loss of a run that had reached update 213 without
having written its first checkpoint (due at 225). Two lessons, both already in
the ledger in other forms: **a pid file is only as good as the pid you put in
it** — capture it by `PPID == 1 && args ~ venv/bin/python`, never by a `pgrep`
that can match your own shell — and a process is confirmed dead by enumerating
what is running, not by one `ps` on a pid you assumed.

What launch 2 did show, in 13 records: warmup completed at u208-211 (`warming`
False, `total` 0.000 → 0.35 as the policy gradient switched on), and by u213
`bank_vs_econ` had gone 4.7 → 133.8 with the margin improving −147,790 →
−128,762. No cliff at the resume. Suggestive only, at 13 updates.

**Launch 3 is the live run**, pid 2705902, resumed from the same c3 u200
checkpoint.

### First two records (live run, pid 2705902)

Reproduced bit for bit across all three launches — same seeds, same episodes —
which is itself a check that nothing in the arm depends on wall clock or on
which GPU it landed on.

|                                 | u201       | u202       |
| ------------------------------- | ---------- | ---------- |
| **`win_rate_vs_econ`**          | **0.000**  | **0.000**  |
| `win_rate_mirror`               | 0.375      | 0.417      |
| `margin_mean_vs_econ`           | −137,731   | −147,790   |
| `margin_mean_mirror`            | 0.000      | 0.000      |
| `bank_vs_econ`                  | 4.67       | 2.25       |
| `bank_mirror`                   | 48.21      | 44.33      |
| `final_capital_vs_econ`         | 1.50       | 1.50       |
| `mean_sale_price_vs_econ`       | 43.26      | 36.93      |
| `realisation_vs_econ`           | 0.903      | 0.892      |
| `units_sold_vs_econ`            | 24.3       | 24.7       |
| `reward_mean`                   | −0.0983    | −0.1051    |
| `illegal`                       | 0          | 0          |
| `warming` / `total == baseline` | True / yes | True / yes |

`margin_mean_mirror` is **exactly** 0.000 — both seats recorded, margins exact
negations. That is the advantage cancellation the ledger diagnosed, visible in
the metric, and the whole reason the absolute term and the scripted half exist.

The absolute term is currently 0.05 × 1.5 / 500 = 0.00015 against a reward of
−0.098 — **0.15%**. It is small because the policy owns almost no capital; at
the reference's 15 it would be the designed ~6%.

## First 26 updates (u201-u226), and the first checkpoint

Checkpoint `toad-phase1m-margin_000225.pt` written at 22:30, no stale `.tmp`,
and it **reloads**: update 225, step 7,549,500, `lr = 6.11399e-05` exactly
`1e-4 × decay(225)`, Adam moments present. `illegal` is 0 on every update.

**No cliff at the resume.** The Jump-Start tripwire did not fire. `total` sat at
0.001-0.006 through warmup, stepped to 0.35 at u211-212 as `warming` went False
and the policy gradient switched on, peaked ~1.5 at u217-218 and settled
0.3-1.1. That is the gradient engaging, not a collapse — and it is the first
time this project has changed a reward under a resumed critic without one.

|                           | first 5   | last 5    |
| ------------------------- | --------- | --------- |
| **`win_rate_vs_econ`**    | **0.000** | **0.000** |
| `margin_mean_vs_econ`     | −138,830  | −139,782  |
| `bank_vs_econ`            | 4.8       | **114.0** |
| `units_sold_vs_econ`      | 26.8      | **71.0**  |
| `mean_sale_price_vs_econ` | 40.3      | 47.6      |
| `realisation_vs_econ`     | 0.931     | 0.972     |
| `final_capital_vs_econ`   | 1.38      | 1.62      |
| `bank_mirror`             | 39.5      | 78.9      |

So: production is climbing hard (bank 24x, volume 2.6x, price +18%) and **the
margin is flat**. That is not any of the pre-registered branches exactly — it is
neither "margin improving while win rate stays 0" nor "bank collapsing while
margin improves", and in particular it is _not_ the signature that says the
absolute term is too weak. It is 26 updates of roughly 580.

**The measurement that explains it, and my sharpened concern.** Against
`economic_policy` the margin is almost entirely made of the opponent's bank,
which our seat barely influences:

|            | min     | max     | sd    |
| ---------- | ------- | ------- | ----- |
| our bank   | 1.3     | 246.4   | 72    |
| their bank | 126,693 | 149,774 | 6,658 |

Their per-update variation is **92x** ours. In reward units, our bank's _entire_
observed range is worth 0.00049 while their swing between updates is worth
0.046 — **94x larger**. So in the scripted half the margin term is currently
about 99% variance we do not control, and only the terminal rank (a constant
−0.02, because we lose every one) and the absolute term are clean.

The mirror half is the mirror image of that problem: there the margin is purely
relative play and its variance _is_ action-correlated, but with one policy in
both seats it is what cancels in expectation. So at today's skill level neither
half yields a clean margin gradient, which is a sharper statement of the
mechanism than "the differential needs an opponent" and was not visible before
the arm ran.

This is reported, not tuned. It should shrink on its own as our bank approaches
the opponent's, and the value baseline absorbs the predictable part of the
opponent's draw. The thing to watch is whether `bank_vs_econ` keeps climbing far
enough for our contribution to the margin to stop being rounding error.

## Pre-registered reads (unchanged)

- `win_rate_vs_econ` off 0 → the objective is right; main line.
- Margin improving with `win_rate_vs_econ` at 0 → closing without winning.
  Report, do not tune.
- Bank **and capital** collapsing while margin improves → absolute term too
  weak. Report immediately.
- Any cliff within a sync cycle of the resume → Jump-Start signature.

## Top concern

The value head is resuming onto a reward whose sign has flipped. Every return
`c3` ever saw was positive (Toad's shaped components are near-monotone up); the
margin reward's true return against `economic_policy` is about **−0.28**. The
critic must relearn from "everything is mildly positive" to "this seat is
losing badly", and it has ~7.6 updates of warmup to do it in before the policy
gradient switches on and the actor starts syncing. If there is a cliff, this is
where it comes from — and the tell will be `baseline` still large when `warming`
goes False. Second, `teacher` rose 0.41 → 1.19 across the first two warmup
updates: the baseline loss backpropagates through the shared trunk (faithful to
monobeast, not a bug), so the policy is drifting slightly while nominally
frozen.
