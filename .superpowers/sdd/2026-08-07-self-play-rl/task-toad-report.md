# Reproducing Toad Brigade (Lux AI S1, 1st) on Kaggriculture

Status: **recipe verified and learner vendored; boundary layer not built; no training run, therefore no
go/no-go bank.** Read the "What is not done" section before planning on any of this.

Primary artifact: `github.com/IsaiahPressman/Kaggle_Lux_AI_2021` at commit `973a6c6`, cloned fresh and
read directly. Every number below carries a file and line citation into that clone. Nothing here is taken
from the prior analysis.

## 0. A provenance check worth recording

A copy of exactly the seven files this task names (five phase YAMLs, `reward_spaces.py`, `monobeast.py`)
was already sitting in the session scratchpad, not under git. Given that this project has already been
burned by a secondary source that transposed two numbers, I cloned the real repository and diffed all
seven. **All seven are byte-identical to upstream.** The pre-placed copy was authentic. I used the git
clone as canonical anyway.

## 1. The verified recipe

Five phases, not four. Config files are `conf/conv_phase{1..4}*.yaml` and `conf/conv_phase5+_final_model.yaml`.

|                          | Phase 1               | Phase 2            | Phase 3            | Phase 4            | Phase 5+           |
| ------------------------ | --------------------- | ------------------ | ------------------ | ------------------ | ------------------ |
| `reward_space`           | `StatefulMultiReward` | `GameResultReward` | `GameResultReward` | `GameResultReward` | `GameResultReward` |
| `total_steps`            | 2e7                   | **1e7**            | 2e7                | 2e7                | 2e7                |
| `discounting` (γ)        | 0.999                 | 0.999              | 0.999              | 0.999              | 0.999              |
| `lmb` (λ)                | 0.8                   | 0.8                | 0.8                | 0.8                | **0.9**            |
| `optimizer_kwargs.lr`    | 1e-4                  | 1e-4               | 5e-5               | 5e-5               | 5e-5               |
| `optimizer_kwargs.eps`   | 3e-4                  | 3e-4               | 3e-4               | 3e-4               | 3e-4               |
| `entropy_cost`           | 1e-3                  | 1e-3               | 2e-4               | 2e-4               | 2e-4               |
| `teacher_kl_cost`        | **0.0**               | 0.005              | 0.01               | 0.001              | 0.005              |
| `n_value_warmup_batches` | —                     | **4000**           | 0                  | 0                  | 0                  |
| `n_blocks`               | **8**                 | **8**              | **16**             | **16**             | **24**             |
| `baseline_cost`          | 1.0                   | 1.0                | 1.0                | 1.0                | 1.0                |
| `use_teacher`            | (absent)              | True               | True               | True               | True               |

Constant across all phases: `num_actors: 2`, `n_actor_envs: 16`, `unroll_length: 16`, `batch_size: 4`,
`hidden_dim: 128`, `embedding_dim: 32`, `kernel_size: 5`, `reduction: sum`, `min_lr_mod: 0.01`,
`rescale_value_input: True`, `model_arch: conv_model`, `optimizer_class: Adam`.

### Values I was asked to verify — verdicts

| Reported by prior analysis                                                | Verdict                                                   | Evidence                                                                                                                                                                                                     |
| ------------------------------------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| shaped phase 2e7 steps, then sparse ±1                                    | **verified**                                              | `conv_phase1_shaped_reward.yaml:32` `total_steps: 2e7`, `:23` `StatefulMultiReward`; `conv_phase2_game_result.yaml:23` `GameResultReward`, which returns ±1 by rank (`reward_spaces.py:112`)                 |
| teacher-KL 0.005 → 0.01 → 0.001 → 0.005 "across phases"                   | **verified as a sequence, corrected in its phase labels** | Those are phases **2→3→4→5**. Phase 1 is `teacher_kl_cost: 0.` (`conv_phase1_shaped_reward.yaml:66`). Read as "phases 1-4" it puts a teacher on the from-scratch phase, which has no teacher to distil from. |
| γ = 0.999                                                                 | **verified**, all five phases                             | `discounting: 0.999` in each config                                                                                                                                                                          |
| λ 0.8 for phases 1-4, then 0.9                                            | **verified exactly**                                      | `lmb: 0.8` phases 1-4; `conv_phase5+_final_model.yaml:63` `lmb: 0.9`                                                                                                                                         |
| Adam 1e-4 → 5e-5, eps 3e-4                                                | **verified**                                              | `lr: 1e-4` phases 1-2, `5e-5` phases 3-5; `eps: 0.0003` everywhere                                                                                                                                           |
| entropy 1e-3 → 2e-4                                                       | **verified**                                              | `entropy_cost: 0.001` phases 1-2, `0.0002` phases 3-5                                                                                                                                                        |
| 4,000 value-warmup batches at the reward switch                           | **verified, and it is phase 2 only**                      | `conv_phase2_game_result.yaml:78`. Phases 4 and 5 explicitly set `n_value_warmup_batches: 0`                                                                                                                 |
| reward weights game_result 10, city 1, unit 0.5, research 0.1, fuel 0.005 | **verified exactly**                                      | `reward_spaces.py:166-177`                                                                                                                                                                                   |

### Things the prior analysis did not report, which matter

1. **A `/500.` normaliser on the entire shaped reward.** `reward_spaces.py:227` returns
   `reward / 500. / max(positive_weight, negative_weight)`. Every weight above is scaled by this before
   it reaches the learner. Porting the weights without the divisor gives a shaped reward 500× too large.
2. **`n_blocks` is a curriculum: 8 → 8 → 16 → 16 → 24.** The network grows between phases and is
   re-loaded `weights_only: True`. This is not a fixed architecture trained for 9e7 steps.
3. **Phase 1 has no teacher at all** (`teacher_kl_cost: 0.`), consistent with training from scratch.
4. **All shaped components are per-turn deltas, not levels** (`reward_spaces.py:196-207`), against a state
   the space carries itself; `_reset()` seeds city/unit counts at **1**, research/fuel at 0.
5. **Fuel is clamped non-negative**: `np.maximum(new_total_fuel - self.total_fuel, 0)` with the comment
   "Don't penalize losing fuel at night" (`reward_spaces.py:201`).
6. **`step` and `fuel` are different components that both happen to be 0.005.** `step` defaults to `0.`
   (`reward_spaces.py:176`) and phase 1 overrides it to `0.005` via `reward_space_kwargs`
   (`conv_phase1_shaped_reward.yaml:29`). `fuel` is `0.005` by default. These are easy to conflate and
   they are not the same knob.
7. **`full_workers: 0.`** — a cargo-full penalty is present but disabled, with the live value commented
   out at `-0.01` (`reward_spaces.py:172-174`). Not part of the winning recipe.
8. **`game_result` really does ride inside the shaped reward from the start**, weight 10, firing only on
   `done` (`reward_spaces.py:209-219`). Confirmed as the task described.

### The loss, which is the part that cannot be approximated

`monobeast.py:420-425`:

```
total_loss = (vtrace_pg_loss + upgo_pg_loss + baseline_loss +
              teacher_kl_loss + teacher_baseline_loss + entropy_loss)
```

- `vtrace_pg_loss` — V-trace advantages (`monobeast.py:358`, `core/vtrace.py`)
- `upgo_pg_loss` — UPGO advantages, **multiplied by clipped V-trace importance weights**
  `min(exp(log_rhos), 1)` (`monobeast.py:386-394`). UPGO's target is
  `r_t + γ·max(V(s_{t+1}), (1-λ)V(s_{t+1}) + λ·G_{t+1})` (`core/upgo.py:23-25`) — it bootstraps from the
  better of the value and the λ-return, so it only pushes toward better-than-expected trajectories.
- `baseline_loss` — smooth L1 against **TD(λ)** targets, not against V-trace (`monobeast.py:395-399`,
  `core/td_lambda.py`)
- `entropy_loss` is **added**, so `combine_policy_entropy` must return negative entropy
- `reduction: sum`, not mean

There are therefore **three distinct return computations** (V-trace, TD(λ), UPGO) over the same batch.
`core/{vtrace,upgo,td_lambda}.py` are pure tensor code with no Lux dependency and were vendored with **zero
edits**. Substituting PPO drops UPGO entirely and changes the baseline target; that is the reproduction
failure the task warned about.

## 2. Proposed reward mapping (designed, not yet implemented)

Theirs is a gather → deposit economy scored on city tiles. Ours is plant → water → harvest → pick up →
carry → drop → sell, scored on coins banked. Keeping their five slots and adding none:

| Toad component | Weight          | Kaggriculture analogue                        | Reasoning                                                                                                                                                                                                                                                   |
| -------------- | --------------- | --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `game_result`  | 10.             | Δ rank by final `money`, ±1 on `done`         | Same role: the true objective, 10× everything else, only on termination. Our terminal reward already exists — the engine sets `state[i].reward = farms[i]["money"]`.                                                                                        |
| `city`         | 1.              | Δ `unlocked_tile_count()` + Δ `plant_count()` | Their city tile is the durable, compounding asset that also scores. Ours is worked land: unlocked quadrants and standing plants are what generate future income.                                                                                            |
| `unit`         | 0.5             | Δ `unit_count` = `1 + len(farm["hands"])`     | Exact analogue — hired hands are our workers, half the weight of the productive asset.                                                                                                                                                                      |
| `research`     | 0.1             | Δ `len(town["unlocked_shops"])`               | Their research points are an irreversible one-way tech unlock gating capability. Shops unlock the same way and are the only monotone progress counter we have.                                                                                              |
| `fuel`         | 0.005           | Δ total `private["shed"]` units, clamped ≥ 0  | Their fuel is stored, spendable resource in hand. Our shed stock is the same: harvested goods not yet sold. Clamped ≥ 0 for the same reason they clamp — selling stock should not read as a loss, since the sale is already rewarded through `game_result`. |
| `step`         | 0.005 (phase 1) | 1.0 per turn                                  | Copied unchanged.                                                                                                                                                                                                                                           |
| `full_workers` | 0.              | disabled                                      | Copied disabled, as in the recipe.                                                                                                                                                                                                                          |

Normalisation: their `/500.` is tuned against a 360-turn game with `reward_min/max = ±1/MAX_DAYS`. Ours is
719 decisions. **Deliberate open question, not silently resolved:** whether to keep `/500.` verbatim
(faithful) or scale to `/1000.` for our roughly-doubled horizon. Faithfulness says keep `/500.`; I would
keep it and record the divergence rather than pre-emptively tune.

Deliberately **not** added: any market/pricing component. Toad had no such component and inventing one
would stop this being a reproduction. See deviation D1 — this is the risk, and it should be measured, not
patched.

## 3. Deviations, ranked by risk

**D1 — Nothing in Toad's recipe teaches market trading, and pricing decides our game. (Highest risk.)**
Lux has no shared price-forming market. Our action space devotes 21 slots × 17 quantity buckets to market
orders, and the sole terminal objective is coins, which is a _price_ × _quantity_ outcome. Their five
reward components reward accumulating productive assets and stock; **none of them reward selling well.**
The shaped phase can therefore drive a policy that farms hard and trades badly. This is the single most
likely reason a faithful reproduction underperforms here, and it is a property of the recipe/game fit, not
of the transcription. It should be measured by the corpus bank percentile, not pre-empted.

**D2 — 719 decisions vs their 360.** Doubles the horizon at γ=0.999 (effective ≈1000 steps, so their
discount barely reaches their own game end and comfortably fails to span ours), and interacts with the
`/500.` normaliser and with `unroll_length: 16`.

**D3 — Action space is per-unit slots, not per-tile planes.** Their `DictActor` (`nns/models.py:15-110`)
is a 1×1 conv over the board producing logits at every tile, and asserts every action space is a 4-dim
`MultiDiscrete` of shape `(planes, 2, h, w)`. Ours is 20 unit slots × 44 ops plus 21 market slots × 17
buckets — the market head has no spatial extent at all. **Their actor head cannot be vendored unmodified**;
this is the largest genuine adaptation and the most likely place for a silent transcription bug.

**D4 — Fixed 10×10 board removes their variable-board machinery.** Strictly a simplification, low risk,
but their `in_blocks`/input-mask paths assume padding-to-32 and will carry dead code.

**D5 — Their self-play uses a frozen teacher checkpoint from phase 1 onward.** Phases 2-5 all set
`use_teacher: True` against a specific `.pt`. Reproducing phases 2+ requires our own phase-1 output as the
teacher; there is no shortcut and no external checkpoint to borrow.

**D6 — `n_blocks` curriculum crosses our existing model.** Our `Policy` is 10.26M params at 8 blocks ×
256 channels; theirs is 8-24 blocks × `hidden_dim: 128`. Matching their recipe means changing width as
well as depth.

## 3b. Second pass: the loss and the reward are built; the run is not

Added after the coordinator confirmed both corrections and asked for phase 1 to be run.

**Built and tested (`25a53ce`):**

- `src/kaggriculture/learn/toad_loss.py` — their loss composition, transcribed from
  `monobeast.py:355-425` with their line numbers quoted at each step, calling the vendored
  `vtrace`/`upgo`/`td_lambda` unmodified. Three separate return computations, as theirs has: V-trace for
  the policy gradient, TD(λ) for the baseline, UPGO for the second policy gradient with clipped
  importance weights.
- `src/kaggriculture/learn/toad_reward.py` — the section-2 mapping, with the `/500.` normaliser.
- 14 tests, all passing.

**Two findings from writing them:**

1. **`flags.baseline_cost` is dead code in their learner.** It appears in every config at `1.` but is
   never read by the loss — only `teacher_baseline_cost` is (`monobeast.py:405`). The baseline enters
   unweighted. Harmless at `1.`, but anyone "restoring" it as a tunable would be inventing a knob.
2. **The second divisor is a no-op for phase 1.** `reward_spaces_lux.py:227` divides by
   `500. * max(positive_weight, negative_weight)`; both weights default to `1.`
   (`reward_spaces_lux.py:150-151`) and phase 1's `reward_space_kwargs` sets only `step`, so `max(1,1)=1`
   and `/500.` alone is correct here. It would stop being a no-op in any phase that sets those weights.

**A guard that did not discriminate, and now does.** The first version of the reward tests stated every
expectation as `STEP_WEIGHT / NORMALISER` — i.e. in terms of the constants under test. Setting
`NORMALISER = 500.0 → 1.0` left all five green. They now pin the published numbers as bare literals;
the same break turns two tests red. Recorded because the vacuous version looked like a thorough test file
and would have shipped.

**Not run.** There is still no training loop, therefore still no bank. What remains:

- Per-turn shaped reward must be threaded through `rollout.py`, which currently records only bank
  differential and own bank. `Stream` needs the five counts per turn; `toad_reward.StatefulMultiReward`
  then consumes them. This is the last correctness-critical seam and is not written.
- A phase-1 script: Adam(1e-4, eps 3e-4), `min_lr_mod` 0.01 decay, `unroll_length` 16, `reduction: sum`,
  to 2e7 steps, checkpointing bank at intervals.
- The actor/learner lag that gives V-trace something to correct. Monobeast gets it from async mp queues;
  a synchronous stand-in (actor weights synced every *k* updates) preserves the loss exactly and must be
  named as a deviation — **D7**, below.
- The unit head can keep their structure per the coordinator's ruling: a per-tile 1×1 conv gathered at
  unit positions, which our `model.py` already does. The market head stays non-spatial and is **ours,
  not theirs** — **D8**.

**D7 — synchronous actor/learner instead of monobeast's multiprocessing queues.** Same loss, same
off-policy correction; the staleness distribution differs. Low risk to the method, but if actor and
learner are never out of sync the importance ratios are identically 1, V-trace silently degenerates to
TD, and the UPGO term loses its clipping — a reproduction failure that looks like a working run.

**D8 — the market head has no Lux analogue and is our addition.** Kept structurally separate from the
vendored actor rather than restructuring theirs to accommodate it, so the seam stays visible.

## 3c. Third pass: phase 1 runs

### The most important correction in this whole report: `clip_grads: 10.0`

**The five phase YAMLs are not a complete recipe.** `monobeast.py:502-505` clips the global gradient norm
before every optimizer step:

```
total_loss.backward()
if flags.clip_grads is not None:
    torch.nn.utils.clip_grad_norm_(learner_model.parameters(), flags.clip_grads)
optimizer.step()
```

`clip_grads` appears in **none** of `conf/conv_phase{1..5}*.yaml`. It appears in the run config saved
beside each trained checkpoint under `internal_testing/`, and **all eight of those read `clip_grads: 10.0`**
(e.g. `internal_testing/hall_of_fame/11-09_21-32-04_59822400/lux_ai/rl_agent/config.yaml:53`).

I missed it, launched without it, and the run diverged to a **total loss of 6.2e20 within two updates**.
That is not a subtle degradation — with `reduction: sum` over a joint log-probability across ~41 decisions
a turn, it is immediate. Anyone reproducing this recipe from the phase configs alone will hit the same
wall, and the briefing analysis did not mention it.

### Four omissions, one cause — read this before reproducing anything

| # | Value | Where it actually lives | Cost of omitting it |
|---|---|---|---|
| 1 | `/500.` normaliser on the whole shaped reward | `reward_spaces.py:227` (code, not config) | shaped signal 500× too large |
| 2 | phase 1 `teacher_kl_cost: 0.` | `conv_phase1_shaped_reward.yaml:66` | a teacher on the from-scratch phase, which has none |
| 3 | `clip_grads: 10.0` | run configs beside their checkpoints — **not in any phase YAML** | 6.2e20 loss in two updates |
| 4 | value head bounded to the reward range | `nns/models.py:177-181` (code, not config) | 1.5e17 baseline term; run diverges |

**The five phase YAMLs are not a complete recipe.** Two of these four live in their Python rather than in
any config, and a third lives only in the run configs saved beside each trained checkpoint under
`internal_testing/`. The briefing analysis captured the phase YAMLs and nothing outside them, which is
exactly the set of values it got right — and the four it missed are each individually fatal to a run.
Anyone repeating this must read `reward_spaces.py`, `nns/models.py` and `monobeast.py`, not just `conf/`.

### The run's actual result: the baseline diverges, and the value head is why

Phase 1 ran. Two checkpoints, then stopped deliberately:

| update | steps | wall | bank mean | bank max | vtrace_pg | upgo_pg | entropy | **baseline** |
|---|---|---|---|---|---|---|---|---|
| 1 | 34,512 | ~30 s | **0.0** | 0.0 | 0.10 | 0.15 | -0.017 | **1.5e17** |
| 2 | 69,024 | ~60 s | **0.0** | 0.0 | 0.00 | 0.00 | 0.00 | **7.4e17** |

**Bank is 0 against the ~21,000 plateau and the 125,773 corpus median.** A randomly-initialised policy banks
nothing, which is expected at 69k steps out of 2e7 — but the run is not merely early, it is *diverging*, and
the number would not have improved by waiting.

The three policy terms are healthy and correctly scaled (0.10, 0.15, -0.017). **The baseline term alone
carries the entire 1e17.** By update 2 the policy terms have gone to exactly 0.0 — the value head has
swamped everything.

**Cause, identified but not yet fixed.** Toad's value head is not a plain linear layer. `BaselineLayer`
(`nns/models.py:140-181`) ends:

```
x = self.activation(x)                                    # Sigmoid, or Softmax if zero-sum
return x * (self.reward_max - self.reward_min) + self.reward_min
```

Their value output is **structurally bounded to the reward space's range** — for `StatefulMultiReward`,
`±1/MAX_DAYS` expanded by `MAX_DAYS` (because `only_once=False`), i.e. `[-1, +1]`. Every phase config sets
`rescale_value_input: True`.

Our `model.py` value head is `Linear(channels, 1)` with no activation and no bound — inherited from the PPO
model, where it regressed coin-scale returns in the thousands. Fed Toad's shaped reward, which is ~1e-5 per
turn after the `/500.` normaliser, an unbounded head initialised for coin scale produces a smooth-L1 target
mismatch that no gradient clip can rescue: clipping bounds the *gradient*, not the loss, and the run
diverged with clipping correctly in place.

This is the same failure mode this project already hit once — a value target left on the coin scale — arriving
from the opposite direction. **D9: their bounded `BaselineLayer` is a load-bearing part of the architecture,
not a detail, and it was not ported.** The fix is small and localised (bound the value head to `[-1, +1]` and
regress the shaped reward against it), but it is not written, so no bank number here is a measurement of
their recipe. Everything above is provisional.

### The fixed run: healthy, training, bank still ~0 at 3.5% of phase 1

With the value head bounded, the same configuration trains instead of diverging. The baseline term drops
from 1.5e17 to ~1e-4 — five orders of magnitude past "fixed", and the clearest possible confirmation that
D9 was the whole cause.

| steps | hours | bank mean | bank max | shaped | vtrace_pg | upgo_pg | baseline | entropy |
|---|---|---|---|---|---|---|---|---|
| 34,512 | 0.009 | 0.0 | 0.0 | 0.01049 | +0.0458 | +0.1838 | 0.00024 | −0.1426 |
| 207,072 | 0.054 | 0.0 | 0.0 | 0.01053 | −0.0540 | +0.0782 | 0.00008 | −0.1554 |
| 379,632 | 0.105 | 0.0 | 0.0 | 0.00990 | −0.0205 | +0.0831 | 0.00010 | −0.1449 |
| 552,192 | 0.157 | **0.1** | **4.0** | 0.01070 | −0.0461 | +0.1092 | 0.00014 | −0.1511 |
| 690,240 | 0.196 | 0.0 | 0.0 | 0.01160 | — | — | — | — |

**Bank against the ~21,000 plateau and the 125,773 corpus median: effectively 0.** The first non-zero bank
appears at 552k steps (mean 0.1, max 4.0) — a policy that has learned to bank a handful of coins in one
episode out of 48.

**This is 3.5% of phase 1.** At the measured 3.97M steps/hour, 2e7 is 5.0 hours away, so these numbers say
nothing yet about whether the recipe clears the plateau. What they do establish is that the machinery is
sound: the loss is stable at ~−0.05 to −0.11 with no divergence, the baseline is well-conditioned, V-trace
and UPGO are both contributing non-trivially, and the shaped reward is **rising** — first five updates
0.01061, last five 0.01160, about +9%. A flat shaped reward would have been the early warning that nothing
was being learned; it is not flat.

**The go/no-go checkpoint is not yet reached and must not be inferred from this table.** The run needs to go
to 2e7.

### Not a deviation: the value-head bound is exactly theirs

Recorded because it is a genuine trap that has already caught one careful reader, and getting it "more
faithful" would break the run.

Reading `StatefulMultiReward.get_reward_spec()` alone (`reward_spaces_lux.py:141-146`) suggests the value
head should be bounded to `±1/MAX_DAYS` = **±0.002778**, i.e. ~360× tighter than our `VALUE_BOUND = 1.0`.
That reading stops one branch too early. `BaselineLayer.__init__` (`nns/models.py:157-160`) then does:

```
if not reward_space.only_once:
    # Expand reward space to n_steps for rewards that occur more than once
    reward_space_expanded = GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]
    self.reward_min *= reward_space_expanded
    self.reward_max *= reward_space_expanded
```

`StatefulMultiReward` sets `only_once=False` (`reward_spaces_lux.py:145`), and `MAX_DAYS = 360`, so:

    ±(1/360) × 360 = ±1.000000

**Their effective bound is exactly `[-1, +1]`.** `VALUE_BOUND = 1.0` is faithful, not a loosening, and it
belongs in no deviation list. The `±1/MAX_DAYS` figure is a pre-expansion intermediate that never reaches
the sigmoid.

The comment in their code says why the expansion exists, and the reasoning transfers to us unchanged: the
head estimates a *multi-step return*, so bounding it by the *per-step* reward range would be a category
error. A per-step bound of ±0.0028 could not even represent our observed episode returns of ~0.01-0.05, and
tightening toward it — the obvious "increase faithfulness" move — would clip every value target to the rail
and destroy the run while looking like a correction.

**Any future ablation on this constant should start from `[-1, +1]` as the faithful baseline**, and treat a
tighter bound as the deviation requiring justification, not the reverse.

### Two further infidelities the first launch exposed

Both now fixed, both worth recording because each would have produced a plausible-looking but wrong run:

1. **The learner batch is four 16-step unrolls, not the collection round.** I had originally stacked every
   segment of a round into one forward and taken a single optimizer step on all of it. That is a different
   algorithm — a different effective learning rate and a different gradient — and their `batch_size: 4`
   (`conv_phase1_shaped_reward.yaml:36`) says so plainly. It also OOMed at 34k rows, which is how it was
   caught. One optimizer step now sees 16 × 4 = 64 transitions, so the round produces ~528 steps, i.e.
   one update per 64 environment steps — the same ratio monobeast's actors feed its learner at.
2. **Torch must be pinned to one thread per worker.** Twelve workers each defaulting to a 64-wide thread
   pool drove load average past 230 and the first round never finished. `selfplay.py:509` already does
   this; I had not. Not a correctness bug, but it is the difference between a run and a hang.

The GPU is also chosen by free memory rather than hardcoded — the cards are shared with another run, and
the first attempt died on one that was already 18.6 GiB full.

### Retraction: throughput is *not* the constraint

An earlier draft of this report claimed 2e7 steps was unreachable. **That claim was wrong and is
withdrawn.** It was derived from a measurement taken *before* `torch.set_num_threads(1)` was added, when
twelve workers were each claiming all 64 cores and the machine was thrashing at load average 230.

Measured from the actual run's own log, after the fix:

| | steps | hours |
|---|---|---|
| update 1 | 34,512 | 0.0086 |
| update 2 | 69,024 | 0.0173 |

**3.97M steps/hour, putting 2e7 at 5.0 hours** — a single overnight run, and consistent with the
independent 9.7M steps/hour estimate to within the difference in worker count. There is no throughput
cliff in evidence. Phase 1 should simply be run to completion, and no work should be spent on throughput.

### The original (superseded) throughput note

**Throughput is the binding constraint.** The engine is single-threaded Python and the rollout is
env-bound, not network-bound, so the GPUs do not help collection. Round latency is ~719 sequential turns
regardless of worker count; parallelism buys episodes per round, not shorter rounds. 2e7 steps is not
reachable in the time available here, so the numbers below are a **partial run** and are labelled
provisional throughout.

## 4. What is not done, and what it would take

Done and committed:

- Recipe verified against the primary artifact, with the eight corrections/additions above.
- `core/{vtrace,upgo,td_lambda,buffer_utils,prof}.py`, `nns/*`, `monobeast.py`, `reward_spaces.py` and all
  eight config YAMLs vendored **byte-identical** to upstream at `src/kaggriculture/learn/toad/`.
- `.pre-commit-config.yaml` excludes that tree. This was not precautionary: on the first commit attempt
  the hooks reformatted `nns/unet.py`, `nns/conv_blocks.py`, `conf/resume_config.yaml` and
  `conf/conv_phase5+_final_model.yaml`, and ruff was mid-way through rewriting the rest. Files were
  restored from the clone and re-diffed as pristine.

Not done — **no training has been run, so there is no bank at the go/no-go checkpoint and nothing to
compare against the ~21,000 plateau or the 125,773 corpus median.**

The remaining work is the boundary layer, and it is not small:

1. An obs space adapter feeding our `(1,48,10,10)` board + `(1,64)` scalars + `(1,20)` positions into their
   `DictInputLayer`.
2. An action space adapter — the real work, per D3. Their `DictActor` asserts spatial 4-dim MultiDiscrete;
   our unit head is per-slot and our market head is non-spatial.
3. A `StatefulMultiReward` subclass over our observation implementing the section 2 mapping.
4. A vectorised env wrapper exposing our `rollout.py` loop with their actor/learner queue protocol.
5. Corpus validation: bank percentile via `corpus.read_manifest`, and state coverage.

## 5. Environment note for whoever picks this up

The worktree was created from `main`, which is the **4-commit repository template** — it contained no
`learn/`, no `encoding.py`, no `corpus.py`, and none of the project's actual work. The live branch is
`setup-kaggriculture-agent` (`dagger-arms` is one commit ahead). I fast-forwarded onto
`setup-kaggriculture-agent`; it was lossless because the worktree branch was a strict ancestor with zero
unique commits. Any other agent handed a worktree here should check this before trusting an empty search
result.

Gitignored files a run needs are **not** in a fresh worktree. `.python-version` and
`src/kaggriculture/routes/prototypes.json.gz` must be copied in or four `tests/test_submission.py` tests
fail. Note the prototype store lives at `src/kaggriculture/routes/`, not `routes/`. There is no `policy.pt`
in the shared checkout at all.
