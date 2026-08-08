# Which published solution to reproduce faithfully — ranking, winner spec, and grounding review

Research run 2026-08-07, revised same day after the priority change: the
primary deliverable is now **which single published solution is closest to our
situation and reproducible faithfully**, with an implementation-grade spec for
the winner. The grounding review of the two prior decisions (DAgger, shaping)
is kept as Part B — its findings feed the deviations list.

Every numeric claim carries its primary artifact — repository files fetched
today, writeups cached at `/data/kaggriculture/research/`, paper PDFs — or an
explicit UNVERIFIED label.

---

# Part A — The reproduction decision

## A.1 Candidates, ranked

Criteria: (1) game-structure similarity, (2) one-workstation compute — hard
disqualifier, (3) leans on a strong-player replay corpus — scores higher
because we hold 8 archives / ~1.04M labelled decisions, underused, (4)
reproducibility from a primary public artifact — hard gate.

| #   | Solution                       | Game fit                                                                                                                 | Compute                                                                                                                               | Data fit                                       | Reproducible?                                                                                                                                                              | Verdict                                                                                                                                                                                 |
| --- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Toad Brigade** — Lux S1, 1st | **Best**: gather→carry→deposit economy, day/night cycles, variable unit count, per-cell unit orders, 360 turns, full obs | "my personal PC — an 8-core/16-thread dual-GPU system" (writeup, verbatim)                                                            | None (pure self-play RL)                       | **Full training code + the five phase config YAMLs**, github.com/IsaiahPressman/Kaggle_Lux_AI_2021                                                                         | **WINNER**                                                                                                                                                                              |
| 2   | **Frog Parade** — Lux S3, gold | Mid: combat/positioning, fog of war, fixed 16 units, 6 actions                                                           | Ryzen 9950X, RTX 3090 + 2070 Super; 300M steps / 8 days / 430 steps/s (writeup)                                                       | None                                           | Full code, github.com/IsaiahPressman/kaggle-lux-2024                                                                                                                       | Runner-up. PPO/10M-params/hardware match ours almost exactly, but the recipe is sparse ±1 from scratch — in our game that is a _measured_ dead end (fresh init: SELL legal 0/719 turns) |
| 3   | **adg4b** — Lux S3, 3rd        | Low-mid: fog-of-war state reconstruction is its core difficulty (absent here); per-tile 6-action UNet                    | "a couple of hours of training" (writeup)                                                                                             | **Best**: pure IL from two top teams' replays  | Full code + released weights, github.com/w9PcJLyb/lux3-bot                                                                                                                 | The corpus-leaning candidate. Ranked 3rd because our own measurement (BC at 90.5% banks 0) contradicts pure IL _in this game_ — though not this exact recipe; see A.5                   |
| 4   | **FLG** — Lux S2, 4th          | Mid-high: economy + the closest philosophy (IL→RL, shaped 65M, trailing teacher)                                         | UNVERIFIED — writeup names no hardware                                                                                                | Mid: ~2,000 replays for IL architecture search | **FAILS the hard gate: no public repository**; detailed writeup only                                                                                                       | Best prose, not reproducible faithfully                                                                                                                                                 |
| 5   | **Flat Neurons** — Lux S3, 1st | Mid                                                                                                                      | **Disqualified**: "We had substantial computing resources… ~1.5B environment steps [in] ~3–4 days", 20B total; ConvLSTM + Transformer | Considered BC from replays, dropped it         | Repo exists (github.com/tonykozlovsky/lux-ai3-pub)                                                                                                                         | Out on compute, plainly                                                                                                                                                                 |
| 6   | **Kore 2022**                  | Low: fleet flight-plan strings, autoregressive actions                                                                   | The one public IL artifact used 2×A100-80GB                                                                                           | High in principle (top-5-replay IL)            | khanhvu207/kore2022 claims **no placement**, pipeline marked unfinished ("Redo the modeling approach and training pipeline"); actual 1st-place writeup artifact UNVERIFIED | Fails gates                                                                                                                                                                             |
| 7   | **Halite IV**                  | Mid (ships/shipyards, 21×21)                                                                                             | —                                                                                                                                     | —                                              | Searched; no public single-box RL winner artifact found (top entries rule-based; the RL work found is a student thesis DQN)                                                | Fails gate 4                                                                                                                                                                            |

**No candidate scores on both game structure and replay-corpus use.** The
published field splits cleanly: economy-game winners on one box trained
self-play RL with shaped-then-sparse curricula (Toad, FLG, Frog Parade);
replay-corpus winners are IL solutions in games where imitation is
error-tolerant (adg4b; Kore's unverified artifact). Nothing published does
"big IL on replays, then single-box RL to surpass" with a repo — that is FLG's
shape, and FLG has no repo. So the honest answer to "is anything close enough
to reproduce faithfully?" is: **yes for the RL recipe (Toad Brigade,
everything public down to the YAMLs), but no published reproducible solution
uses a replay corpus the way ours could be used.** The corpus stays in the
reproduction as the evaluation set and, optionally, the rank-3 arm (A.5).

## A.2 Why Toad Brigade wins

- **Game structure.** Lux S1 is the only candidate whose core loop is ours:
  acquire (mine wood / buy seed), tend under a day-night clock (fuel cities /
  water crops — and their nightly city-fuel consumption rhymes with our
  nightly `_end_of_day` hand-clear), carry to a depot, convert to the scoring
  resource, survive to a fixed horizon (360 vs our 719). Variable unit count
  with per-unit orders and a joint action per turn — their "sum the
  log-probabilities of all selected actions" is literally our Task-4 joint
  ratio. Full observability both (their fog is trivial; ours is the
  opponent's private shed only).
- **Compute.** Stated in the writeup: one personal PC, dual GPU, training
  overnight. Their configs run 2 actors × 16 envs; our measured 13,449
  trajectories/hour collection (≈2,700 steps/s) exceeds what their configs
  imply they had.
- **Reproducibility.** The strongest artifact in this field: the repo carries
  the model (`lux_ai/nns/`), the IMPALA learner (`lux_ai/torchbeast/
monobeast.py`), the reward spaces (`lux_ai/lux_gym/reward_spaces.py`), and —
  decisive — `conf/conv_phase1_shaped_reward.yaml` … `conv_phase5+_final_
model.yaml`: the actual five-phase schedule with every coefficient.
- **It is the published recipe for our measured barrier.** Its phase 1 exists
  precisely because sparse win/loss from scratch does not train ("to speed up
  training and aid the agent in developing rudimentary behaviors despite the
  sparse win/loss reward signal"). Frog Parade's recipe, by contrast, _opens_
  with the experiment we already ran and watched fail.

## A.3 The winner's spec (all numbers from the repo/writeup; file named per item)

### Network (writeup §"Neural network architecture"; configs)

- Input: each discrete observation channel through its own **learnable 32-dim
  embedding** (`embedding_dim: 32`); continuous channels per-feature
  normalised; each group concat → 1×1 conv to 128 → LeakyReLU
  (`n_merge_layers: 1`); groups concat → final 1×1 conv → **128×H×W**.
- Trunk: fully-convolutional ResNet, **5×5 kernels** (`kernel_size: 5`),
  **hidden_dim 128**, **squeeze-excitation** on every block, **no
  normalisation layers** (`normalize: False`). Blocks per phase: **8 → 16 →
  24** (`n_blocks`). ~20M parameters at 24 blocks (writeup).
- Heads: per-cell actor logits per actor type (workers 19 / carts 17 / city
  tiles 4, all 1×1 convs), logits of illegal actions set to −inf; critic
  scalar in [−1, 1] with rescaled input (`rescale_value_input: True`).
- Joint policy: log-probs of every selected action summed per turn — one
  joint ratio, one reward.

### Algorithm and loss (monobeast.py, lines ~300–410)

IMPALA (torchbeast), single learner, 2 actor processes × 16 envs each
(`num_actors: 2`, `n_actor_envs: 16`), unroll 16, batch 4. Total loss =
**V-trace policy gradient + UPGO policy gradient** (UPGO importance weights
clipped at 1; both PG terms unweighted) **+ 1.0 × TD(λ) baseline loss**
(`baseline_cost: 1.`) **+ teacher_kl_cost × KL(teacher ‖ student)** −
`entropy_cost` × summed per-head entropy; `reduction: sum`. On a reward
switch, **`n_value_warmup_batches: 4000`** updates run `baseline_only` — the
value head adapts to the new reward scale while the policy holds still
(monobeast.py:673).

### Optimiser and RL constants (all phase configs)

- **Adam, lr 1e-4** (phases 1–2) → **5e-5** (3+), **eps 3e-4** (the config
  cites arXiv:2105.05246 for the large eps), lr annealed to `min_lr_mod:
0.01` of initial.
- **discounting γ = 0.999** (all phases — identical to our GAMMA).
- **λ = 0.8** for TD(λ) and UPGO (phases 1–4), **0.9** in phase 5+.
- **entropy_cost 1e-3** (phases 1–2) → **2e-4** (3+).

### Reward (reward_spaces.py:138–178; phase configs)

- **Phase 1 — `StatefulMultiReward`, selfish (zero_sum=False), per-step
  changes** in: game*result **10.**, city **1.**, unit **0.5**, research
  **0.1**, fuel **0.005**, plus per-step constant **0.005** (phase-1 config
  override), full_workers 0. Scaled into **±1/360 per step**
  (`reward_min/max = ±1/MAX_DAYS`). Note the win term rides \_inside* the
  shaped phase at 10× the biggest sub-goal — the terminal objective is never
  absent, merely accompanied.
- **Phase 2 onward — `GameResultReward`**: ±1 terminal, zero-sum, nothing
  else.

### Schedule (the five YAMLs, verbatim numbers)

| Phase | Net      | Steps   | Reward         | teacher_kl_cost | lr   | entropy | Notes                                                                                                                                       |
| ----- | -------- | ------- | -------------- | --------------- | ---- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | 8-block  | 2e7     | shaped (above) | 0               | 1e-4 | 1e-3    | from random init                                                                                                                            |
| 2     | 8-block  | 1e7     | sparse ±1      | 0.005           | 1e-4 | 1e-3    | warm-start phase-1 weights; **4,000 value-warmup batches**                                                                                  |
| 3     | 16-block | 2e7     | sparse ±1      | 0.01            | 5e-5 | 2e-4    | **fresh 16-block student, 8-block as frozen teacher**                                                                                       |
| 4     | 16-block | 2e7     | sparse ±1      | 0.001           | 5e-5 | 2e-4    | continue, teacher pressure ÷10                                                                                                              |
| 5+    | 24-block | 2e7/run | sparse ±1      | 0.005           | 5e-5 | 2e-4    | λ→0.9; 16-block as teacher; repeated runs ("training overnight most nights"); **total steps beyond the ~9e7 visible in configs UNVERIFIED** |

### Opponents and imitation

- **Pure mirrored self-play, latest weights both seats. No opponent pool.**
  Stability against strategic cycles comes from the frozen-teacher KL
  (writeup: "helped to stabilize behavior and prevent strategic cycles —
  both of which are problems that plague a pure self-play setup").
- **No behaviour cloning, no replays, anywhere.** Teachers are always the
  pipeline's own earlier, smaller checkpoints.

### Budget check on our box

Our measured end-to-end rate is 4,800 trajectories/hour ≈ 3.45M steps/hour.
Phase 1 (2e7) ≈ 6 hours; the whole published floor (~9e7) ≈ 26 hours; ten
times that fits in 54 days with room for the gate and re-runs. Throughput is
genuinely not the constraint — consistent with Task 5's measurement.

## A.4 Forced deviations — named, with risk

1. **The market head — the riskiest by far.** Lux has no market; nothing in
   Toad's recipe prices, times, or sizes trades against an opponent moving the
   same curve. Structurally their formalism absorbs it (market orders become a
   fourth actor type inside the same joint log-prob — their machinery is
   built for heterogeneous heads), but _no part of the reproduced curriculum
   teaches trading_, and the vendored header for `economic_policy` says
   pricing "is the layer STRATEGY.md identifies as deciding this game." A
   faithful reproduction can plausibly reach six-figure banking and still
   lose the kaito rung on market play. This is the unhedged risk.
2. **Shaped-phase components.** Theirs are Lux-specific (city/unit/research/
   fuel deltas). Ours must be farm-specific. Faithful-in-form: per-step
   deltas of holdings, selfish, small per-step scale, terminal term riding
   along at ~10× — which is almost exactly what `progress.py` already builds
   (and Part B's Grzes finding applies to it: zero the terminal potential).
   Content deviation, unavoidable, low risk _if_ the Part-B terminal fix is
   in.
3. **IMPALA + UPGO vs our PPO.** Faithful = port their monobeast (it is all
   in the repo). If the implementers keep our tested PPO loop instead, that
   is a real deviation: FLG measured "I tested PPO and Vtrace … and achieved
   similar results" (writeup, small maps), so the V-trace/PPO half is
   low-risk to swap — but **UPGO has no PPO analogue and would be silently
   dropped**, and no primary source prices that omission. Name it in the run
   log either way.
4. **719 turns vs 360.** Same γ=0.999; discount half-life ≈ 693 steps ≈ our
   horizon. Their per-step reward scale (±1/360) should become ±1/719. Low
   risk.
5. **Input encoder.** Theirs: 32-dim embeddings per discrete feature. Ours:
   fixed planes. Keeping ours is a deviation; their own writeup half-licenses
   it ("32-dimensional embeddings were excessive for some of the features…
   I would reduce the dimensionality"). Low risk.
6. **No variable board sizes, no 0-padding masks, no overlap-sampling.** Our
   fixed 10×10 and per-unit (not per-cell) readout make three of their
   mechanisms unnecessary. Simplifications, not risks.
7. **The replay corpus goes unused in training.** Toad used zero replays; a
   faithful reproduction uses the archives only for the gate and the corpus
   median. If "train on ALL of the data" is a requirement and not a
   preference, the faithful answer is that **the winning approach does not
   do that**, and the closest reproducible approach that does is rank-3
   adg4b — see A.5.

## A.5 The corpus question, stated plainly

The instruction "train from scratch on ALL of the data" and the winner's
recipe are in tension: Toad's pipeline consumes no demonstrations. Two honest
options:

- **Reproduce Toad faithfully and accept the corpus as evaluation-only.**
  This is the recommendation.
- **Run adg4b's recipe as a cheap parallel arm on the corpus** — it is the
  one reproducible published approach whose fuel is exactly our 1.04M
  decisions, and its recipe differs from our failed BC in ways that target
  our clone's exact pathology: imitate **one** donor team's replays (not a
  mixed corpus — a mixture of policies is not a policy), keep **only won
  matches** from lost games, drop **95% of all-idle steps** (our clone's
  379-hires-per-episode is a class-imbalance signature), weighted
  cross-entropy. Hours of training by their own account. If it also banks ~0,
  that closes the pure-IL question in this game with a faithful data point
  instead of our unfaithful one; if it banks six figures, we have a strong
  phase-3-style teacher for pennies. Either result is worth the day.

What would make the reproduction _not_ work, and how we would know early:
phase 1 must take the bank off ~21k. Toad's shaped phase differs from our
plateaued shaped runs in two testable ways — the win term rides inside the
shaped reward at 10× weight (ours phased it in later), and the shaping is
multi-component per-step deltas rather than bank-only. If, after the
equivalent of phase 1 (≈2e7 steps, ≈6 hours), bank against the frozen pool is
not clearly above the 21k plateau, stop and re-diagnose rather than running
the remaining phases on hope.

---

# Part B — Grounding review of the two prior decisions

Kept from the original dispatch; its findings are load-bearing for deviations
2 and 3 above and for the DAgger/`learn.dagger` work already in flight.

## B.1 Verdicts

**DAgger against `economic_policy`: right call, three implementation hazards,
one ceiling.** DAgger is the published remedy for the measured BC failure
(90.5% held-out accuracy, banks 0): it trains on the learner's own induced
distribution, which is exactly what open-loop accuracy cannot see. β
schedule, from the paper itself: "We will typically use β₁ = 1 … The simple,
parameter-free version … β_i = I(i = 1) … often performs best in practice";
their Super Tux Kart runs used `I(i=1)`, and in handwriting the decayed
`p^{i−1}` variants did no better (p=0.5 comparable, p=0.9 explicitly slower).
**Use β₁=1 then β=0; do not build a decay schedule.** The ceiling: the
guarantee is `J(π̂) ≤ J(π*) + uTε` — expert-relative. DAgger's target is
~155k parity, and it inherits the expert's losses to kaito.

**Sub-goal shaping: right call, precedents verified, one real invariance bug
in the in-flight implementation.** FLG's ~65M-step shaped phase and Toad's
20M-step phase are verbatim in their own artifacts. Ng–Harada–Russell's
policy invariance is not purity here — it is the anti-farming property this
project was burned twice for lacking (cycles pay zero). But the episodic
corollary is violated in the current code: see B.3.

## B.2 DAgger details

**Does εH² buy anything at H=719? No — and it doesn't matter.** At ε≈0.1,
T²ε ≈ 49,000 against a maximum episode cost of 719: vacuous, as is uTε when
one mid-chain mistake forfeits a crop cycle (u is large; the paper concedes
"in the worst case, u could be O(T)"). The non-vacuous part is Theorem 3.3:
the surrogate loss **on the learner's own induced distribution** converges to
the best achievable. So the metric is _on-policy expert agreement_ on the
learner's own rollouts — never held-out replay accuracy, which is the number
already proven to predict nothing here.

**Hazards found in this repository:**

1. **The expert is not a function of the observation.** Module globals
   `_SIGNATURE_LAST_STEP`, `_SIGNATURE_ACTIVE` (economic_policy.py:284–285,
   latched at steps 24/192/264 from opponent counts) and
   `_SCHEDULE_WHEAT_REQUESTED` (line 286, a day-11 wheat manoeuvre) persist
   across calls. Queried out of step order — shuffled labelling, interleaved
   envs sharing the module — labels become order-dependent noise on the
   market head. **Label at collection time, per episode, in step order, one
   expert instance (or explicit 3-global reset) per environment.**
2. **Silent PASS pollution.** `agent()` wraps `_decide` in
   `except Exception:` → all-PASS (economic_policy.py:2098). DAgger visits
   states the expert's author never saw; any that raise become confident
   do-nothing labels on exactly the learner's off-distribution states. **Call
   `_decide` directly when labelling; count exceptions; exclude those states.
   A growing count is itself the u=O(T) alarm.**
3. **Realizability.** Every expert action must encode into the 44-op × 20 +
   21×17 vocabulary; Task 1b already found one silently-dropped class.
   Assert-or-count at labelling time; inexpressible actions are an ε floor no
   iteration count removes.

**The exit past the expert: QDagger, not a timer.** Reincarnating RL's
distillation weight is `λ_t = max(1 − G^π/G^{π_T}, 0)` — tied to the measured
student/teacher return ratio, vanishing at parity; their students "quickly
outperform [the] teacher policy within 5M frames." This reconciles the two
measured facts that look contradictory (the teacher penalty is load-bearing
AND the teacher caps the policy): the pull fades exactly as fast as the
student earns it. `KL_STEPS = 100` — a wall-clock timer — is wrong in both
directions at once. FLG's trailing teacher ("updated the teacher regularly,
keeping it around 30M steps behind") and Toad's phase-teacher cascade are the
same idea from the competition side.

## B.3 Shaping details

**The terminal-potential bug.** Grześ (AAMAS 2017), Eq. 3: the shaped return
differs from the true return by `γᴺΦ(s_N) − Φ(s₀)`; the second term is a
constant, but the first "depends on actions … and, as a result, this term can
modify the policy"; conclusion, verbatim: "the potential function **has to be
set to zero** for a state, s_N, at which a particular learning trajectory
stops." Our every trajectory stops at turn 719 in an action-dependent state,
and `rollout.py:753–754` stores `potential(terminal)` un-zeroed. At γ=0.999:
holding one unit of produce from turn t to the end returns `0.999^(719−t) ·
base` of shaped value — **0.905·base from turn 619** — so the shaped optimum
late-game is to refuse any price below ~90% of base, in a market that both
farms are pushing below base by selling. "Shed fills, bank flat" is built
into the reward as written. **Fix: zero the terminal potential row.** Then
unsold stock is confiscated at the horizon — exactly the true objective —
and the telescoped shaping is the constant −Φ(s₀).

**Multi-agent status.** Devlin & Kudenko (AAMAS 2011): potential-based
shaping does not modify the Nash equilibria of a stochastic game; it does
alter exploration and thus which equilibrium is reached — which is the
purpose, not a defect.

**A softer distortion to measure, not necessarily fix.** With stored/carried
at full base value, a SELL below base is _locally_ negative in combined
reward even though the return favours it. If median combined reward on SELL
transitions is negative in early shaped runs, re-value stored/carried at
~0.8·base (any Φ is a legal potential; same move as `GROWING = 0.4`).

**Why shaping reaches the barrier at all:** the fresh policy already plants
32 times per episode (measured) — random exploration reaches the chain's
early links, it just isn't paid there. Under the potential, the first WATER
in a bonus window pays 0.4·price immediately, so the gradient arrives at a
link exploration already visits, and each reinforced link exposes the next.
Fallback if it stalls mid-chain: expert roll-ins — play `economic_policy`
for the first h turns, hand over, anneal h→0 (Salimans & Chen: demo-state
starts cut sparse-reward exploration from exponential to quadratic in chain
length; JSRL: same curriculum, guide-policy roll-ins, "exponential in
horizon to polynomial").

## B.4 Cross-check of the prior research file

- **Wrong:** "FLG's +0.3 per factory [is] absolute, paying regardless of the
  opponent" — the writeup says "a flat + or − 0.3 for each factory it had
  **more / less than its opponent**." Relative. Also "all one machine" for
  FLG: the writeup names no hardware; UNVERIFIED.
- **Misapplied (recommendation #4, "drop or fast-decay the teacher KL"):**
  JSRL's remedy is guide roll-ins, not KL removal — its premise _agrees with_
  the fresh-init measurement. Posterior BC is about pretraining with action
  coverage, and says nothing about removing regularisation. Kickstarting's
  42.2% is the multi-teacher DMLab-30 figure with **PBT-adapted** λ
  schedules, not a fixed decay — the honest lesson is _adaptive_ teacher
  weight (QDagger's ratio), not removal on a timer. Your framing ("those
  results assume a fresh policy can reach rewarding states unaided") is
  right about the conclusion, slightly off on mechanism: the papers don't
  assume it — the file's _use_ of them did.
- **Refuted by measurement, correctly:** the KL-is-a-pin diagnosis. The
  refinement the data supports: the KL substituted for the missing dense
  chain reward by holding the policy where reward is obtainable.
  **Falsifiable prediction: with potential shaping live (terminal fixed),
  rerun the KL-removal diagnostic; it should no longer collapse.** If it
  still does, the substitution story is wrong too.
- Citation numbering in the prior file's Finding 5 collides with Finding 3
  ([3] is both FLG and JSRL) — the same defect class as the transposed-number
  incident.

## B.5 What to measure

1. **Phase-1 gate (reproduction go/no-go):** after ≈2e7 shaped steps, bank vs
   frozen pool must clear the 21k plateau decisively; else stop and
   re-diagnose.
2. **DAgger:** on-policy expert-agreement per iteration; bank toward ~155k;
   gate vs `economic_policy` off 0.000; expert `_decide` exception count ≈0
   and flat; inexpressible-action count. Relabel one fixed episode under two
   worker interleavings — identical labels or the statefulness fix failed.
3. **Terminal-potential fix:** `stored` must fall over the last 48 turns
   while bank rises; final-day SELL count up versus pre-fix.
4. **SELL local signal:** median combined reward on SELL transitions; if
   negative, re-value stored/carried at 0.8·base.
5. **Teacher weight:** log `λ_t = max(1 − bank/teacher bank, 0)` next to the
   KL; policy keeps improving as λ→0, no collapse.
6. **B.4's KL-substitution prediction:** one small-budget arm.

## Sources

Primary artifacts fetched/read today: Toad Brigade repo
(github.com/IsaiahPressman/Kaggle*Lux_AI_2021 — README/writeup, `conf/conv*
phase{1..5+}\*.yaml`, `lux_ai/lux_gym/reward_spaces.py:138–178`,
`lux_ai/torchbeast/monobeast.py`); Frog Parade writeup
(github.com/IsaiahPressman/kaggle-lux-2024/write-up.md); adg4b writeup
(cached) + github.com/w9PcJLyb/lux3-bot; Flat Neurons writeup (cached) +
github.com/tonykozlovsky/lux-ai3-pub; FLG writeup (cached; no repo found);
khanhvu207/kore2022 README. Papers: DAgger arXiv:1011.0686 (PDF: §3, §5,
Thms 2.2/3.2–3.4); Grześ AAMAS 2017 (ifaamas.org p565.pdf, Eq. 3); Devlin &
Kudenko AAMAS 2011 (ifaamas.org D1_G45.pdf); Reincarnating RL/QDagger
arXiv:2206.01626; Kickstarting arXiv:1803.03835 (PDF text); JSRL
arXiv:2204.02372; Salimans & Chen arXiv:1812.03381; Posterior BC
arXiv:2512.16911. UNVERIFIED this run: Toad's total steps beyond the ~9e7 in
committed configs; FLG hardware; Kore 2022's actual 1st-place artifact; DORA
compute; Frog Parade's 110k steps/s simulator figure.
