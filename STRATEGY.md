# Kaggriculture: strategy

Working plan derived from [RESEARCH.md](RESEARCH.md). That document holds the evidence; this one holds
the decisions, the order of work, and the gates that stop us wasting weeks.

Written 2026-08-04. Submission deadline **2026-09-30**, then ~2 weeks of continued play and a
Bradley-Terry tournament for the final standing. **Eight weeks of build time.**

---

## The bet

Train a self-play RL policy with a board-shaped action head, and let it learn the market rather than
hand-coding it. Keep the heuristic alive the whole time as a league opponent, a submission floor, and a
hedge.

Why this and not a tuned heuristic: the spatial layer of this game is easy (10×10, no combat, units may
overlap), and the layer that decides games is economic — a shared price-forming market that no previous
Kaggle simulation had, so there is no borrowed strategy to copy, only borrowed machinery. Crop value is
not a property of a crop, it is a property of a crop given what the opponent is selling, and that
signal arrives every turn in `market["inventory"]`. A recipe tuned to today's monoculture decays; a
policy conditioned on live market state adapts by construction, which matters over eight weeks of meta
drift.

Why this is not obviously right: Lux S2 went to a forward-simulation heuristic that beat the best RL
entry. We keep the heuristic track funded for exactly that reason.

---

## The opponent is a tape

The most-voted public kernel (`romantamrazov/kaggriculture-hamburger`, 82 votes) is not a strategy. Its
embedded agent's docstring reads _"Fixed public Tran H Hoang policy from episode 89674601, seat 0"_, and
the notebook asserts `len(anchor_trace) == 720`. It replays **one recorded game's action sequence,
open-loop**, with thin overlays for terminal liquidation, weed repair, and mirror detection.

This explains the monoculture exactly: 298 of 400 mined agent-games had byte-identical builds because
they are the same recording. It is decoded to `/data/kaggriculture/baselines/meta_tape.py` and runs as a
local opponent.

**It banks 174k–194k against our current agent's 28.7k.** That is the real bar, not the 123k mined
median — the median is depressed because tape-versus-tape games crash each other's prices.

Because it is open-loop, its entire market footprint is known in advance:

|       | units over the season                                                                                     |
| ----- | --------------------------------------------------------------------------------------------------------- |
| sells | wheat 1103, milk 413, **fertilizer 408**, strawberry 390, wool 251, melon 182, carrot 13, egg 8, tomato 4 |
| buys  | **wheat 967**                                                                                             |
| other | 306 hires (~10/day), 2 land purchases, 8 cows, 6 sheep                                                    |

And its dump schedule is known per day — strawberry is negligible before day 17 then floods days 18–29
(38, 27, 42, 23, 30, 24, 49, 33, 20, 14, 49, 22); melon front-loads days 10–12 (39, 27, 12); wheat spikes
on day 12 (204) and day 29 (148).

Five edges follow directly, all measured rather than inferred:

1. **Melon is dead.** Against the tape it finishes at **$1** in every game — its 182 units plus our ~170
   floor it. Our entire current strategy is built on a crop the meta destroys. This alone invalidates
   `melon loop v1`.
2. **Fertilizer is mispriced downward.** The tape sells 408 units of it — the engine permits fertilizer
   sales even though the documentation says otherwise — and it ends at $57 against a $100 base. It is
   _buyable_, so cheap fertilizer is a yield multiplier nobody is using as one.
3. **The field is a guaranteed wheat buyer**, 967 units of forced demand per tape, to feed 14 animals.
4. **Strawberry, milk and wool hold high against a single tape** (264 / 303 / 248 at the end) because
   only one player is supplying them. The tape captures all of that. We supply none of it.
5. **Carrot, tomato and eggs remain unsupplied** — 13, 4 and 8 units respectively across a whole season.

There is also an asymmetric weapon here. The ladder scores wins, not margins, and the tape cannot react.
Selling into its known dump windows depresses the prices its own revenue depends on, and it will keep
executing regardless. Costing it 40k while costing ourselves 10k is a winning trade.

**Caveat on using it for training.** An open-loop opponent is a poor sole adversary — beating it can be
achieved by exploiting its blindness rather than by playing well. It belongs in the league as one
opponent among several, never as the only one, and the RL reward must stay the win rather than the margin
against it.

---

## Where we stand

> **Superseded 2026-08-04 — we no longer submit our own heuristic.**
>
> `pilkwang/kaggriculture-structured-economic-policy`, a public notebook with 74 votes, turned out to be a
> genuinely reactive agent rather than another copy of the tape: 44 functions, stdlib only, pricing from
> the live market curve and from the opponent's visible supply. Measured over 20 seeded games it beat our
> tuned heuristic **20-0 at 148k to 51k**, and it costs 0.9 ms a turn at the median against a 1000 ms
> budget, so the strength is not bought with compute. It is vendored at
> `src/kaggriculture/economic_policy.py` with attribution, linted and typed to this project's standards,
> and guarded by a 2,157-turn decision-level characterization test plus 117 unit tests that check its
> internal price model against the engine's — they agree exactly across every product and inventory level.
>
> This answers the open question at the end of this document about which public kernel the field is
> copying: it is neither of the two guessed at, and the kernel has been renamed since that was written.
>
> Our heuristic is frozen at `baselines/heuristic_v2.py` and remains a league opponent. The narrative below
> describes how it got there and is kept because the reasoning still holds — it is simply no longer the
> thing we ship. The heuristic track's remaining items now apply to the vendored policy instead.

`melon loop v1` (submission 55222583) is on the ladder at **478**, below its 600 seed, with a 1W–4L
recent record. It wins every game against the built-in agents and banks ~48k doing it, then banks 7.7k
to 47k against real opponents.

The diagnosis is in the research: it ranks crops at _base_ price, which picks melon, and melon is the one
crop with zero shop demand. Locally, playing bots that never touch the market, melon inventory drains
and the price holds at $288. On the ladder it gluts to +100 and the price collapses to $94–154. The agent
is not badly implemented; it is optimising a number that stops being true the moment a second melon
farmer appears.

Median bank at the top of the ladder is **123,334**. We are a factor of three off.

What v1 does get right and should be preserved: capacity-capped planting (never sow more tiles than the
crew can water — opponents routinely finish with 11–61 weeds), harvest-on-yield-cap, and the end-of-season
flush.

---

## The public kernels are exhausted

Surveyed the competition's top kernels on 2026-08-04, by votes, classifying each as a replay of a
recorded episode or a reactive agent that reads the observation:

| kernel                    | votes | what it is                              |
| ------------------------- | ----- | --------------------------------------- |
| romantamrazov (hamburger) | 105   | the recorded tape                       |
| **pilkwang**              | 74    | **the only genuinely reactive agent**   |
| prvsiyan (frontier lab)   | 55    | analysis and visualisation, no agent    |
| degnonguidi               | 43    | replay variant, carries `TRACE_ACTIONS` |
| lucifer19 (night harvest) | 37    | replay variant, carries `TRACE_ACTIONS` |
| kaitofukami (closed loop) | 32    | analysis; never reads the observation   |

So the field's public code is one recording, several re-wrappings of that recording, some analysis, and
one real agent — which we adopted. **There is nothing left to borrow.** Every improvement from here has
to be ours.

That reframes the gap to the top of the board. The leader sits at 3070 and the recorded tape at ~1720,
so the teams above us are running private agents substantially stronger than anything published. Copying
has taken us as far as it goes; the remaining distance is work.

---

## What actually decides games

Three levers, in order of measured impact.

**Market timing and allocation.** Strawberry runs 120 → 233 → 87 because the whole field harvests in
unison around days 16–27. Wheat climbs monotonically 25 → 56 because 14 animals per agent eat one a day
and nobody plants enough. Tomato reaches 93 and eggs 67 with essentially no supplier. Melon is a trap
that looks like a gold mine.

**Labour.** The n-th hire of a day costs `fib(n)` and resets daily, while a worked tile returns tens of
coins a day. Unit-turns, not money, are the binding constraint — but only up to the point where the crew
outruns what it can walk to, past which extra tiles become weeds.

**Endgame liquidation.** Unsold stock scores zero, the end-of-day free drop lands after the final scored
turn, and dumping into a thin market crushes the price. Both the timing and the rate of the final sell-off
are worth real money.

---

## Plan

Six phases. Each has an exit gate; **do not start the next phase until the gate passes.** Phases 1 and 2
run in parallel with the heuristic track.

### Phase 0 — Retire the deployment risk (1 day)

The largest single risk is that we train something we cannot run. A Lux S2 competitor abandoned a trained
model because no fast inference runtime could be installed in the Kaggle sandbox, and error logs came
back 404.

Submit a probe agent that plays the current heuristic but additionally logs, on turn 0: the Python
version, whether `torch` and `numpy` import, their versions, available core count, and timed forward
passes. Read it back with `kaggle competitions logs <episode> <index>`.

The networks timed are the two we would actually ship, not a token small one — a benchmark of a net we
have no intention of training answers a question nobody asked:

- **~20M parameters**, the 24-block residual trunk Toad Brigade won Lux S1 with on a personal PC [1].
- **~300M parameters**, the scale Lux S3's second place trained on an RTX 3090 and an RTX 2070 Super [8],
  which is the configuration this workstation strictly beats and therefore the one worth sizing for.

Training scale and inference budget are separate constraints and this measures only the second. On two
threads of the local workstation the 300M trunk runs a batch-1 forward in **311 ms** and the 20M trunk in
**29 ms**, so neither is obviously excluded by the 1-second turn — but the sandbox CPU is the number that
counts, which is the whole point of submitting.

The probe measures the small trunk first and extrapolates the large one's cost from it, building the
large one only if the prediction fits the remaining budget. A single 300M forward pass on a slow sandbox
CPU could outlast the entire 60-second overage pool, and an agent that overruns forfeits; a prediction
that says it does not fit is the Phase 0 answer without betting the episode on it.

**Gate:** we know the inference budget in milliseconds for a concrete network, and whether torch is
usable. Costs one of five daily slots.

#### Result — gate passed, and it caps the model (submission 55242090, episode 89948290)

|                       | sandbox                                             |
| --------------------- | --------------------------------------------------- |
| python                | 3.11.13, glibc 2.35                                 |
| cpus                  | **2**                                               |
| memory                | **6.8 GB**                                          |
| numpy                 | 2.4.6                                               |
| torch                 | **2.6.0+cu124, available**; import costs **10.7 s** |
| ~20M-parameter trunk  | **105 ms** per batch-1 forward                      |
| ~300M-parameter trunk | **1131 ms** per batch-1 forward                     |

**Torch is usable, so the RL track is not dead on arrival** — the Lux S2 failure mode does not apply
here. Everything else the probe found is a constraint:

- **The budget is arithmetic, not parameters.** Both measurements agree on roughly **20–26 GMAC/s** at
  two threads, so a 300 ms turn buys about **6 GMACs**. What that converts to in parameters depends
  entirely on where they sit: a convolution weight does one multiply-add _per cell_, a hundred of them on
  a 10×10 board, while a dense weight at a bottleneck does exactly one. Measured at two threads locally,
  300M in convolutions costs 307 ms and 152M in dense layers costs 26 ms — about **60× cheaper per
  parameter**.
- **So 300M in convolutions is excluded, and 300M is not.** Six GMACs is ~60M convolution parameters, or
  effectively unlimited dense ones. At 1131 ms the conv version of the Lux S3 scale would draw on the
  60-second overage pool every turn and forfeit within a minute of game time; a dense-heavy 300M model
  extrapolates to ~190 ms and fits. The constraint rules out a _shape_, not a size — which is
  convenient, because the market branch that decides this game is ~26 numbers and belongs in exactly the
  cheap kind of layer. Toad Brigade's ~20M conv trunk fits with ten times headroom.
- **The 300M question is moot: their model was 10M.** Reading the primary source [16] rather than the
  roundup [8] settles it. Frog Parade's submitted model is an **8-block 3×3 residual CNN with
  squeeze-excitation at d_model 256, about 10M parameters**; the 300M is their _training steps_
  (600M per-player observations). The two numbers were transposed in [8] and copied into RESEARCH.md.
  Nothing about their solution is out of reach — it is _smaller_ than Toad Brigade's 20M, and the same
  author wrote both, having gone down in size rather than up. A model that shape costs roughly 40–50 ms
  in our sandbox, since our board has 100 cells to their 576.
- **Two cores, not sixty-four.** Every inference number above is a two-thread number, so local timings
  taken on the workstation will flatter the sandbox by an order of magnitude. Measure Phase 2's
  candidates under `OMP_NUM_THREADS=2` or the architecture selection is measuring the wrong machine.
- **The 10.7-second torch import lands on turn 0** and comes out of the overage pool, leaving ~49 s of
  cushion for the rest of the episode. A model load has to go there too. The probe episode spent 31 s on
  turn 0 and 0.8 ms on average across the other 719 turns, and completed — so the pool absorbs a
  one-off startup comfortably, but nothing recurring.

The probe has answered these and is removed; it is in git history if a later question needs it.

### Phase 1 — Evaluation and league infrastructure (3–4 days)

Nothing downstream is measurable without this, and the existing `Harness` is most of the way there.

- Freeze named opponents in `baselines/`: `heuristic-v1`, plus a reconstructed **meta build** (wheat 11,
  strawberry 40, melon 11, cow 8, sheep 6, 3 quadrants) which is what 75% of the ladder plays.
- Head-to-head evaluation with seeds paired across matchups, reporting win rate with a confidence
  interval — not bank, and never leaderboard position, which the Kore winner observed converges unstably.
- A replay-analysis module: given any episode, report per-player crop mix, bank trajectory, realised
  prices and inventory deltas. We already have this as scratch code; it needs to become a real module
  because every later phase reads it.

**Gate:** `heuristic-v1` vs the strongest league opponent over 100 seeded games returns a win rate with a CI narrower than
±0.1, in under 10 minutes wall-clock on 64 cores.

#### Result — gate passed 2026-08-04

Runs are in `will-rice/kaggriculture-2026`, named `<agent>-vs-<opponents>-<commit>`:
`heuristic_v1-vs-meta_tape-82a4130` (`s8mmlfea`) and `main-vs-league-82a4130` (`urrclpeq`).

`meta-build` was ruled out of the league during infrastructure work as byte-for-byte the shipped agent, so
the matchup would have been self-play; the recorded tape took its place as the strongest opponent actually
in the league. `heuristic-v1` vs the recorded tape over 100 seeded games (run `s8mmlfea`): win rate
0.000 [0.000, 0.037], half-width 0.0185 — half of the stated interval, since the lower bound is 0 — in 15s
on 64 cores. Both criteria clear, though not by as much as the interval suggests: the width depends on the
observed rate, and at 100 games the worst case (a rate of 0.5) gives a half-width of 0.0962. **One hundred
games is the minimum that satisfies this gate, not a comfortable margin** — size future sweeps from the
worst case, not from this run.

`heuristic-v1` lost all 100 games. The tape plays the recorded 75th-percentile ladder build open-loop, and
nothing we had written beat it.

Reference league standing for the current submission, 300 seeded games per opponent, seed 1000 (run
`urrclpeq`). The submission is now the vendored economic policy, not our own heuristic — see the adoption
note below:

| opponent     | win rate | 95% interval   | our bank | theirs  |
| ------------ | -------- | -------------- | -------- | ------- |
| meta-tape    | 0.000    | [0.000, 0.013] | 118,672  | 140,585 |
| heuristic-v2 | 1.000    | [0.987, 1.000] | 148,931  | 50,972  |
| heuristic-v1 | 1.000    | [0.987, 1.000] | 151,012  | 16,056  |
| starter      | 1.000    | [0.987, 1.000] | 161,120  | 3,496   |

It beats all three frozen baselines cleanly and still loses every game against the recorded tape — the
honest current state. What adoption bought is visible in the banks rather than the win column: our own
tuned heuristic banked 54,569 against the tape while the tape reached 171,353; the served agent banks
118,672 and holds the tape to 140,585. The gap narrowed from 3.1x to 1.2x without a single win.

**Two earlier runs carry a wrong configuration.** Runs `fsrlzxgh` and `l73v8htf`, cited here before
2026-08-04, recorded correct win rates against a config describing the wrong agent: the runner logged the
package's default `Strategy` regardless of which agent `--agent` selected, so the gate's config claimed the
tuned crop mix and a herd of fourteen while `heuristic-v1` actually played melon monoculture with no
livestock. The measurements were sound and reproduce exactly; only the metadata lied. Fixed at commit
`82a4130` by deriving the strategy from the agent module, and the two runs above replace them. The wrong
runs are named rather than deleted, because an audit trail that quietly drops its own errors is not one.

**From here, a change ships only if it beats the current submission on win
rate against this league, with non-overlapping intervals.** Mean bank against a
single opponent is retired as an acceptance metric; it was what let a herd
tuned to one open-loop recording look like progress.

### Phase 2 — Imitation dataset and architecture selection (1 week)

Pick the architecture with cheap supervised signal _before_ spending the expensive RL budget — FLG's
procedure, and the most reusable process lesson in the research.

> **Premise weakened, 2026-08-04.** The compute-saving half of this argument rested on microRTS
> "70 GPU-days, later reduced to 23 by behaviour cloning". The primary source contradicts it: the
> 23-GPU-day model is behaviour-cloning only and materially weaker (44% against Mayari, where the winner
> scored 90%+), and cloning followed by PPO fine-tuning cost **72** GPU-days — more than the 70 of plain
> RL [9]. Imitation bought quality-per-compute nowhere in that paper. It is still worth doing here to
> escape a cold start and to select an architecture cheaply, but **not** on the grounds that it saves
> training budget, and the phase should be time-boxed accordingly.

- Build the dataset from the daily replay dumps, **filtered to top-decile banks**. The pool is
  three-quarters one copied kernel, so unfiltered cloning teaches the monoculture including its
  badly-timed strawberry allocation. Treat imitation as an initialisation to escape, never a target.
- Encode observations: per-farm tile planes (crop one-hot, weed, locked, empty, structure, animal,
  `watered_today`, `consecutive_unwatered`, `yield_units`, `fertilized_until_day`, age), unit-count
  planes, and **separate learned embeddings for `hour` (0–23) and `day` (0–29)** — the analogous phase
  features are credited with producing distinct opening/midgame/endgame play in Lux S1.
- **Start from Frog Parade's shape rather than inventing one** [16]. Their Lux S3 second place is open
  source, was trained on hardware we beat, and its structure maps onto this game almost line for line:

  | their component                                                                                        | ours                                                                 |
  | ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------- |
  | 8-block 3×3 residual CNN, squeeze-excitation, `d_model` 256, ~10M parameters                           | same, on 10×10                                                       |
  | temporal + nontemporal spatial features → 2-layer CNN projection                                       | tile planes, plus a frame stack                                      |
  | global features (80 of them) → 2-layer MLP, broadcast and **added** to the spatial tensor              | the market's ~26 numbers, which is the branch we had already planned |
  | value head: 2-layer 1×1 CNN → mean-pool → scalar                                                       | same                                                                 |
  | actor head: index the alive units, append normalised energy, 2-layer MLP → per-unit logits             | index unit-bearing cells, append carried inventory, → 22 op logits   |
  | second actor head: 2-layer CNN → a 24×24 probability map shared across units, per-unit illegal masking | the market-order head, and any op that names a target tile           |

  Note the sizing this settles: **~10M parameters, not 300M** — the 300M is their step count, and the
  numbers were transposed in [8]. At 10M convolutional parameters on a 100-cell board this costs ~40–50 ms
  of a 1-second turn, and the 6 GMAC budget above leaves room to grow it several times over if that ever
  pays.

  **Every verified winner is small.** The 2026-08-04 verification pass established this across four
  competitions: microRTS RAISocketAI **5.0M** [9], with Huang et al.'s best policies under **1M**; Lux S3
  tenth place **1.8M**, 3.2M with a separate critic [17]; Lux S3 second place **10M** [16]; Lux S1 first
  place **~20M** [1]. No primary source anywhere supports a winning model above 20M — both figures that
  suggested otherwise, 300M and 23B, were step counts. **Start at ~10M and treat 20M as a ceiling to
  justify rather than a floor to exceed.** Frog Parade's 10M model shipped at ~38 MB against a 100 MB
  submission cap [16], so file size is not the binding constraint here; the turn budget is.

- Candidate trunks: the above, against a downscaling U-shaped variant. Select on action-prediction
  accuracy **per millisecond of CPU inference measured at two threads**, which is what the sandbox has.
- Steal three cheap things from [16] while we are here: **test-time augmentation** (they average the
  policy over both diagonal reflections and a 180° rotation before sampling — our board is square and
  symmetric, so this applies directly), **illegal-action masking** in the loss and not just at sampling,
  and a **teacher-KL term** alongside entropy, which Phase 3 already plans.
- What bought that team their scale was a Rust rewrite of the environment, worth ~3.2× data collection
  and a final 430 steps/second [8][16]. We are deferring that (see Phase 3), which caps how much of their
  recipe is reachable in eight weeks — though note their 300M steps at 430/s is eight days, and our
  Python interpreter already does ~38k steps/s across 64 cores.

**Gate:** a chosen architecture with a measured forward pass comfortably inside the Phase 0 budget, and a
behaviour-cloned agent that beats `random` and ideally `heuristic-v1`.

#### Result — behaviour cloning, 2026-08-05 (run `bc-10M-seed0-0cf4416`)

Dataset: 868 seats selected from 434 episodes across 6 daily archives at a rating floor of 2500 (the
weaker player's rating, so both players in a kept episode were strong) with a per-archive cap of 200.
Split by episode: 780 train seats → 140,400 rows and 88 holdout seats → 15,840 rows, at stride 4.
10,134,294 parameters, 8 epochs, batch 512, AdamW at 3e-4.

**Holdout accuracy 0.857** over 149,481 acting units, against a majority-class baseline of 0.175
(`NORTH`). The model is not a collapsed predictor: it emits 19 distinct ops and its per-op recall tracks
the corpus — `WATER` 0.97, `PASS` 0.95, `PICKUP` 0.92, `PLANT:MELON` 1.00, down to `PLACE` 0.29. Train
and holdout loss finish at 0.247 and 0.249, so it is not overfitting either. As a model of what a strong
farmer's hands do next, cloning worked.

| opponent     | win rate | 95% interval   | mean bank | opponent bank |
| ------------ | -------- | -------------- | --------- | ------------- |
| meta-tape    | 0.000    | [0.000, 0.037] | 3,000     | 189,245       |
| heuristic-v2 | 0.000    | [0.000, 0.037] | 3,000     | 68,304        |
| heuristic-v1 | 0.000    | [0.000, 0.037] | 3,000     | 41,376        |
| starter      | 0.000    | [0.000, 0.037] | 3,000     | 3,501         |

League 0.000 over 4 opponents, 400 games, 0 errors (run `play-vs-league-0cf4416`).

**The gate is missed, and no amount of training clears it.** The agent banks exactly its starting 3,000
in all 400 episodes because `UNIT_OPS` has no market verbs, and the engine increases a farm's money in
exactly one place — `_commit_unit` crediting a `SELL` order. A policy restricted to unit ops therefore
cannot gain a single coin, so its bank is pinned at 3,000 and its win rate against any opponent that nets
anything at all is zero by construction, whatever its accuracy. `starter` finishes on 3,501: we lost by
501 coins to an opponent we could not out-earn if the weights were perfect.

This is the gap between predicting a recording and playing the game, and it is sharper than the phase
text anticipated. Every economically decisive choice in Kaggriculture — buying seed, hiring, buying land
and animals, and selling — is a market order, and the clone learned the 85.7% of the decision stream that
moves hands around a farm it can never stock. The lesson for Phase 3 is that the market head is not an
enhancement to schedule after the unit head works; it is the half of the action space that carries the
score, and the encoder already feeds it (Task 3's market branch reaches the trunk). Deliberately not
fixed here: adding a market head to a behaviour-cloning script would be Phase 3 work done in Phase 2's
file, and the corpus's market orders deserve their own encoding decision rather than an afterthought.

Kept as an initialisation, which is what the phase text says imitation is for: the trunk has learned to
read a board, and Phase 3 starts from that rather than from noise.

#### Result — market head, 2026-08-06 (run `bc-market-10M-seed0`, gate `play-vs-league-008c3b6`)

The market head was added to close the gap above: the clone now predicts, per turn, a bucketed quantity
for each of 21 market slots (sell per product, buy-seed per crop, buy-product, buy-animal, hire, buy
land) alongside its per-unit ops, and `learn/play.py` decodes both heads into one action. Same trunk,
same dataset, one extra loss term at `MARKET_WEIGHT = 1.0`.

**Both heads clone well.** Units 0.857 over 149,481 acting units against a 0.175 majority-class
baseline, unchanged. Market 0.989 over 332,640 slots — but the all-zero baseline is 0.943, because the
teacher trades on only 5.7% of slots, so the honest figure is **0.818 restricted to the 19,006 slots
where the teacher actually traded**. Verified through the play path, loading the shipped checkpoint the
way the agent does, so the numbers describe the artefact and not just the training loop.

**The agent trades, and goes bankrupt doing it.**

| opponent          | win rate | 95% interval   | mean bank | opponent bank |
| ----------------- | -------- | -------------- | --------- | ------------- |
| meta-tape         | 0.000    | [0.000, 0.037] | 0         | 188,104       |
| heuristic-v2      | 0.000    | [0.000, 0.037] | 0         | 68,884        |
| heuristic-v1      | 0.000    | [0.000, 0.037] | 0         | 41,372        |
| starter           | 0.000    | [0.000, 0.037] | 0         | 3,501         |
| `economic_policy` | 0.000    | [0.000, 0.037] | 0         | 162,136       |

League 0.000 over the four frozen opponents, 400 games, 0 errors (`play-vs-league-008c3b6`), plus 100
games against the served agent (`play-vs-economic_policy-008c3b6`).

**Phase 2's number to beat was 3,000. This banks 0.** The bank moved off its opening balance, which is
what the market head was for, and it moved the wrong way: traced over one seeded episode against
`starter`, the farm spends 3,000 down to zero by about turn 25 and never recovers, emitting 1,508 HIRE
orders, 325 `BUY_PRODUCT WHEAT`, 47 `BUY_SEED MELON` and 34 `BUY_ANIMAL COW` across the season against
311 `SELL FERTILIZER` and 247 `SELL WHEAT` that never fund them. Every hand it hires then plays PICKUP
on an empty tile, turn after turn, because nothing was ever planted for them to work.

**This is covariate shift, not a broken play path**, and the distinction is the finding. The same
checkpoint, through the same encoders and the same decode, reproduces its holdout accuracies exactly on
teacher states. It is only on states of its own making — a farm with nothing planted, a board no
top-decile player ever stood on — that it degenerates. Two mechanisms compound: the model's own errors
walk it off the data distribution within twenty turns, and the 21 market slots are argmaxed
independently, so a turn's orders are a combination the teacher never played even when each slot is
individually likely. Accuracy on a recording bounds nothing about a season played out.

**The submission entrypoint is unchanged.** `main.py` still serves the vendored economic policy, which
banks 162k where this banks 0. The gate to repoint it was beating that policy; it lost 100–0.

**The sandbox budget is not the obstacle.** Measured out of the built 36.0 MiB archive at two threads: a
0.911 s torch import inside turn 0, turn 0 at 1.028 s, and turns 1+ at 13.8 ms mean, 15.9 ms p99,
123.5 ms max — one turn of 719 over the 1 s `actTimeout`, consuming 0.028 s of the 60 s overage pool.
Scaling by the Phase 0 sandbox probe (10.7 s import, ~20–26 GMAC/s against this workstation's ~90) puts
the sandbox at roughly 11 s of the pool on turn 0 and ~50 ms a turn thereafter — comfortably inside.
**A torch policy is submittable; this one is just not worth submitting.** Re-measure with
`uv run python -m kaggriculture.learn.scripts.budget`.

What Phase 3 inherits: a trunk that reads a board, a market head that reads a market, and the measured
fact that neither survives its own trajectory. That is an argument for the RL phase's frozen-teacher KL
term rather than against the initialisation.

### Phase 3 — RL, small model (2 weeks)

- Board-shaped action head: `10×10×22`, read only at cells holding units, **illegal actions masked to
  −inf**. Separate heads for market orders (sell quantity per product in buckets, buy-seed per crop,
  buy-animal, hire count, buy-land).
- Units sharing a tile: sample without replacement until a no-op.
- **Rare-event upsampling** via a dedicated actor replaying openings — hires, land and animal purchases
  occur on a handful of steps per episode and will otherwise never learn.
- Reward: bank-difference shaping for the first ~20M steps, then **sparse zero-sum win/loss**. Every top
  team across three competitions converged on win-only as the final signal; the roundup's conclusion is
  that fundamentals over fine-grained shaping separated winners from mid-tier.
- **Frozen teacher with a KL penalty (~5e-3), kept tens of millions of steps behind** — this is what
  prevents the strategic cycling that pure self-play produces.
- PPO or IMPALA; do not spend time choosing, the evidence says they measure the same.

Throughput is not the bottleneck: 1.68 ms/step, ~38k steps/s across 64 cores, 100M steps ≈ 0.7 h of pure
simulation. **Defer any fast-simulator rewrite** — a rewrite that silently diverges from the official
interpreter trains the agent on the wrong game.

**Gate:** beats `heuristic-v1` over 100 seeded games at >60% win rate.

### Phase 4 — Scale and league (2 weeks)

- Train a larger model with the Phase 3 model as KL teacher.
- League: past checkpoints, the recorded tape, `heuristic-v1`, `heuristic-v2`, sampled with preference for opponents that beat
  us — prioritised fictitious self-play, which exists precisely because self-play alone cycles among
  non-transitive strategies.
- Fine-tune against the strongest ladder opponents we can reconstruct from replays.

**Gate:** beats the Phase 3 model at >60% and the recorded tape at >65%. The tape replaces `meta-build`
here for the reason given under Phase 1: a baseline built on our own policy is our own code with
different constants, and the one we built turned out byte-for-byte identical to the shipped agent.

### Phase 5 — Ship and iterate (ongoing from week 4)

Submit early and often; only the latest two submissions are scored, and five slots a day cost nothing.
Re-mine the replay corpus **weekly** — the top rating went 1152 → 2627 in four days, so the meta is a
moving target and every strategic conclusion has a short half-life.

Reserve the final week for stability, not gains. The Kore leader was overtaken on the last day by an
agent that exploited a specific weakness in his opening; a diverse league and deliberate adversarial
testing are the defence.

---

## The heuristic track, run in parallel

RL takes weeks and the ladder is live now. The heuristic is also the honest hedge against the Lux S2
outcome where a good heuristic beats a good policy.

Immediate work, reordered after measuring against the tape. The benchmark for every step is bank against
`meta_tape.py`, which v1 played at 28.5k versus its 182k over 16 seeds.

1. ~~**Animals. Nothing else comes close.**~~ **Done — 28.5k → 40.1k** over the same 16 seeds, with the
   tape's own bank pulled down from 182k to 164k. Still 0 wins: the herd closes a third of a gap that is
   a factor of four wide, so items 2–7 all still stand. Three things had to be true at once and each was
   worth measuring separately: budget every market order against the balance the previous orders left
   (independent budgeting spent $3000 three ways on day zero and starved the farm to $56 for twelve
   days); buy an animal only when its feed is already in the shed and build its pasture only when it is
   waiting for one; and charge the herd's daily feed, care and harvest against the same crew the crops
   draw on. A target herd of 8 cows and 6 sheep — the tape's mix — measured _worse_ than 8 cows and 2
   sheep (32.8k vs 40.8k): past about ten animals this crew cannot walk the feed round, and the animals
   it misses escape. Splitting the feed run across several carriers measured worse than one loaded
   carrier at every split tried, because the pastures sit together beside the shed.

   The original reasoning below is kept because items 2–7 are ordered by it.
   Valuing the tape's known sales at the market prices recorded
   during a real game against us decomposes its gross revenue as milk ~112k (413 units at ~$272),
   strawberry ~99k, wool ~61k, wheat ~46k (1103 units but only ~$42 each), melon ~43k, fertilizer ~32k.
   **Animal products are over half of gross revenue**, and we produce none of them. (Start-of-step
   pricing overstates the absolute figures, since a large order walks its own price down; the ranking is
   what matters.) Cows and sheep need a pasture tile, daily wheat, and `CARE`; the wheat can simply be
   bought, which is what the tape's 967 wheat purchases are for.

2. ~~**Abandon melon** — but only after animals.~~ **Done — melon 40.8k → strawberry 49.1k**, confirmed
   on holdout seeds (40.1k → 47.3k). Sweeping all five crops with the herd in place puts strawberry
   first, wheat second (40.4k), then tomato 37.6k and carrot 33.2k.

   The ordering completely inverted. Pre-livestock, melon was the best crop at 28.7k and strawberry the
   **worst at 1.5k**; a farm at $56 could not carry a 100-coin seed to a day-10 first yield, so the crop
   that needs working capital measured as the crop that does not work. Fixing the budget changed which
   crop is best — which is the same lesson as the melon trap, one level up: we were ranking crops under a
   constraint we then removed.

   Two mechanics explain the winner, both worth remembering. Strawberry is _ongoing_, and the engine
   accrues an ongoing crop's yield on schedule whether or not it was watered that day — watering only
   prevents death and enables the fertilizer bonus. So a strawberry tile costs roughly half the
   unit-turns of a melon tile, which is exactly the constraint the herd competes for. And melon's price
   collapses to $31 against the tape's 182 units while strawberry holds near $269 because only one player
   supplies it.

   Note it still measures _worse_ against the built-in agents (65k versus melon's 87k), because bots
   never touch the market and melon's price holds at $288 there. That is the trap in `melon loop v1`
   verbatim, and the reason the tape is the benchmark and the built-ins are not.

3. **Price from live inventory, never from the base table.** So the next crop that gets flooded doesn't
   cost us another submission cycle.
4. **Sell around the tape's known windows.** Strawberry before day 18, melon outside days 10–12. This is
   free money against 75% of the field and needs no market model at all — only a calendar.
5. **Buy fertilizer, don't sell it.** It ends at $57 with a $100 base because the field dumps it, and
   fertilizer doubles the per-day yield bonus on a watered crop.
6. **Marginal-value allocation** across crops and animals, accounting for committed volume and
   season-long town plus shop absorption.
7. **Fix the cash-starvation window.** v1 sits at exactly zero money from day 4 to day 14 and cannot hire
   through it.

Every one of these is also a better league opponent for the RL track.

---

## How we decide anything is better

Local head-to-head win rate against a fixed opponent set — `meta_tape`, `heuristic-v1`, `starter` — seeds
paired, with a confidence interval.
Never bank (it is not what the ladder scores), never leaderboard position (too noisy, and the Kore winner
documented the instability), never a single game.

A change ships when it beats the current submission at >55% over ≥100 seeded games against the league.

---

## Timeline

| weeks           | RL track                                          | heuristic track                    |
| --------------- | ------------------------------------------------- | ---------------------------------- |
| 1 (Aug 4–10)    | Phase 0 probe, Phase 1 infrastructure             | live-price allocation, sell timing |
| 2 (Aug 11–17)   | Phase 2 dataset + architecture                    | animals, cash-flow fix             |
| 3–4 (Aug 18–31) | Phase 3 small-model RL                            | maintain as league opponent        |
| 5–6 (Sep 1–14)  | Phase 4 scale + league                            | —                                  |
| 7 (Sep 15–21)   | fine-tune vs ladder replays                       | —                                  |
| 8 (Sep 22–30)   | stability, adversarial testing, final submissions | fallback ready                     |

---

## Kill criteria

Stated in advance so they are not rationalised away later.

- **Phase 0 fails** (no usable runtime in the sandbox and numpy export cannot hit the budget) → abandon
  the RL track entirely, put everything into the heuristic plus forward search.
- **Phase 3 gate missed by Sept 1** → freeze RL, ship the heuristic, and spend the remaining month on
  forward-simulation planning over market decisions, which is the Lux S2 winner's approach and fits our
  1s budget for the low-dimensional economic choices.
- **RL plateaus below the heuristic for two consecutive weeks** → the heuristic is the submission and RL
  becomes a research side-track.

---

## Open questions

1. Does torch run in the agent sandbox, and how fast? (Phase 0 answers this.)
2. Is the 198,630-bank diversified build reproducible, or was it one lucky market?
3. What does the ladder's top decile bank _today_, versus the 123,334 median measured on 2026-08-02?
4. Which public kernel is the monoculture source — `romantamrazov/kaggriculture-hamburger` (82 votes) or
   `pilkwang/kaggriculture-observable-economic-control` (71)? Reading it tells us exactly what 75% of the
   field does.
