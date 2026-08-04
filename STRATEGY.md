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

### Phase 1 — Evaluation and league infrastructure (3–4 days)

Nothing downstream is measurable without this, and the existing `Harness` is most of the way there.

- Freeze named opponents in `baselines/`: `heuristic-v1`, plus a reconstructed **meta build** (wheat 11,
  strawberry 40, melon 11, cow 8, sheep 6, 3 quadrants) which is what 75% of the ladder plays.
- Head-to-head evaluation with seeds paired across matchups, reporting win rate with a confidence
  interval — not bank, and never leaderboard position, which the Kore winner observed converges unstably.
- A replay-analysis module: given any episode, report per-player crop mix, bank trajectory, realised
  prices and inventory deltas. We already have this as scratch code; it needs to become a real module
  because every later phase reads it.

**Gate:** `heuristic-v1` vs `meta-build` over 100 seeded games returns a win rate with a CI narrower than
±0.1, in under 10 minutes wall-clock on 64 cores.

### Phase 2 — Imitation dataset and architecture selection (1 week)

Pick the architecture with cheap supervised signal _before_ spending the expensive RL budget — FLG's
procedure, and the most reusable process lesson in the research.

- Build the dataset from the daily replay dumps, **filtered to top-decile banks**. The pool is
  three-quarters one copied kernel, so unfiltered cloning teaches the monoculture including its
  badly-timed strawberry allocation. Treat imitation as an initialisation to escape, never a target.
- Encode observations: per-farm tile planes (crop one-hot, weed, locked, empty, structure, animal,
  `watered_today`, `consecutive_unwatered`, `yield_units`, `fertilized_until_day`, age), unit-count
  planes, and **separate learned embeddings for `hour` (0–23) and `day` (0–29)** — the analogous phase
  features are credited with producing distinct opening/midgame/endgame play in Lux S1.
- The market gets its **own MLP branch** concatenated at the trunk bottleneck, not broadcast planes. It is
  ~26 numbers that decide the game.
- Candidate trunks: a residual ResNet with squeeze-excitation (no normalisation layers), and a
  downscaling U-shaped variant. Select on action-prediction accuracy **per millisecond of measured CPU
  inference**. Size against the Lux S3 second-place configuration — dual ~300M-parameter PPO models,
  ten million steps over eight days on an RTX 3090 and an RTX 2070 Super [8] — because that is the
  largest solution known to have been trained on hardware we have, and Phase 0 measures whether it fits
  the turn. Note what actually bought them that scale: a Rust rewrite of the environment lifting
  collection from one to ten million steps a day. We are deferring that (see Phase 3), which caps how
  much of their recipe is reachable in eight weeks.

**Gate:** a chosen architecture with a measured forward pass comfortably inside the Phase 0 budget, and a
behaviour-cloned agent that beats `random` and ideally `heuristic-v1`.

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
- League: past checkpoints, `meta-build`, `heuristic-v1`, sampled with preference for opponents that beat
  us — prioritised fictitious self-play, which exists precisely because self-play alone cycles among
  non-transitive strategies.
- Fine-tune against the strongest ladder opponents we can reconstruct from replays.

**Gate:** beats the Phase 3 model at >60% and `meta-build` at >65%.

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

2. **Abandon melon** — but only after animals. Measured against the tape, melon is _still_ our best
   single crop at 28.7k, versus wheat 9.4k, tomato 10.5k, carrot 4.1k and strawberry 1.5k. Crop choice is
   not the gap; the absence of livestock is. Changing crop first would have been a wasted cycle, which is
   exactly why this list is now ordered by measurement rather than by the earlier reasoning.
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
