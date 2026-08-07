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
