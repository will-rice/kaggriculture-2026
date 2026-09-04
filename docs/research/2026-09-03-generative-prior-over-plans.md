# A generative prior over agent plans — final report

2026-09-03. The line is **closed on evidence**. Nothing was committed; the repo
working tree was never written to. Code and logs survive under the session
scratchpad at `plangen/`.

---

## Controller's note on framing, added when persisting this

The "355,281 → 217" collapse below is measured against a **whole-turn symbol
alphabet**, where one turn is one token. That was the right baseline for a
sequence model over tapes, which is what this investigation built.

It is **not** an improvement over our RL policy's action representation, and
early summaries of mine implied that it was. `action_codec` has been factored
all along — `UNIT_OPS` 44, `QUANTITIES` 21, `MARKET_SLOTS` 19 — with matching
per-unit heads in `model.py`, and it already restricts item-reading to the
three verbs that take one. What genuinely differs is **resolution** in the
quantity dimension (our grid jumps 12 → 16 → 24 → 40, and 23.4% of the
teacher's own orders land in that gap) and **ordering** in the market
dimension. Read §1 as a finding about tape corpora, not about the policy.

---

## 1. The factored action space — resolved, and the artifact worth keeping

Measured on the **identical 3,937,244 turn-slots** that produced the 355,281
baseline, which reproduces exactly, confirming both measure the same thing.

| stream                | vocab   | hapax   | % of vocab | % of slots | slots      |
| --------------------- | ------- | ------- | ---------- | ---------- | ---------- |
| whole-turn (baseline) | 355,281 | 278,484 | 78.38%     | 7.073%     | 3,937,244  |
| `unit_op`             | 41      | 0       | 0.00%      | 0.00000%   | 42,386,711 |
| `unit_quantity`       | 20      | 0       | 0.00%      | 0.00000%   | 42,386,711 |
| `market_slot`         | 23      | 0       | 0.00%      | 0.00000%   | 8,559,072  |
| `market_quantity`     | 133     | 19      | 14.29%     | 0.00022%   | 8,559,072  |
| **combined**          | **217** | 19      |            |            |            |

**Losslessness verified against the engine, not against its own inverse.**
Byte-exact round-trip is 99.154%. Pushing _both_ the original and the
round-tripped action through the repo's `sim.rollout.encode_turn`, on 4,744
differing turns sampled 1-in-7, **4,744 of 4,744 encode to identical simulator
codes**. Every divergence is an item the engine never reads, or a
`SELL X 999` sentinel clipped above `SHED_CAPACITY`. **Lossless as executed.**

Design: 44 unit ops (set equality with `action_codec.UNIT_OPS` asserted); 22
market slots as an **ordered sequence of up to 10 orders**, queue position
preserved, no affordability mask; quantity as a plain integer; `UNKNOWN` and
`END` per stream.

**Alias fix, done structurally.** `FEED WHEAT` (32,872 slots) and
`FERTILIZE FERTILIZER` (8,469) are 0.097% of unit slots. Verified in the
reference engine: `op = action[0]`, and `action[1]` is read only inside the
PLANT / PICKUP / PLACE branches. The tokenizer now reads an item _only for the
three verbs that take one_, so the entire alias class is unrepresentable rather
than two strings deleted. Shards keep old indices; `dataset.unit_remap()`
builds the remap **by name**, and `simdecode` raises a named error rather than
a device-side assert if an un-remapped label arrives.

## 2. Corpus

35 archives, **24,874 episodes, 49,748 clean seat tapes, 0 faults**, 6.1 GB, in
**13 minutes**. 850 teams; held out **85 teams / 4,155 tapes**, by team and
never by tape. Median bank 92,912; win rate 0.4904.

| key                 | distinct | largest | top-5     | singletons  |
| ------------------- | -------- | ------- | --------- | ----------- |
| step-1 market queue | 241      | 20.4%   | **51.1%** | 15          |
| opening-24          | 575      | 18.0%   | **40.5%** | 138 (24.0%) |

- largest cluster: 8,937 tapes, 142 teams, median bank **85,898**, win 0.485
- strong cluster (5th): 2,601 tapes, 94 teams, median bank **131,152**, win 0.512

**The biggest cluster is the weak one** — an unweighted prior imitates the weak
majority.

## 3. Model and held-out likelihood

15.0M params, one transformer position per turn (384/8/6), factored parallel
heads. 12,000 steps, 23 minutes. wandb `will-rice/kaggriculture-2026/runs/rieaq9vk`.

Full 4,155-tape holdout across 85 unseen teams: `unit_op` **0.7842** nats
(uniform ln 48 = 3.871), `unit_quantity` 0.00670, `market_slot` 0.11159,
`market_quantity` 0.06128. Train and holdout within 0.004 nats — no
overfitting, which is itself a monoculture symptom.

> **0.784 is the flattering cut.** It averages over `END` padding for units
> that do not exist. On slots where a unit actually acts, cross entropy is
> **0.953**. Quote the active-slot number.

## 4. Pipeline validation

Everything else rests on this, so it is a first-class result.

`check.py`: worst scalar-state disagreement **0.00024** (float16 storage only),
unit frame **exact**, final banks **[106532, 131667] identical to published**.
Conditioning is **per unit** — each of 16 slots carries its own
(x, y, carried load).

**105 recorded tapes replayed against their own opponents on their own seeds:
median |simulated − published| = 0 coins.**

## 5. Central strategic finding

| condition                                                                  | win rate    | cost                    | games    |
| -------------------------------------------------------------------------- | ----------- | ----------------------- | -------- |
| own opponent, own seed                                                     | 0.562       | —                       | 105      |
| own opponent, **exam seeds**                                               | 0.508       | seed −0.054             | 1,680    |
| **top-1% opponent** (bank 160,177 vs 88,422), exam seeds, **still frozen** | 0.434       | strength −0.074         | 4,480    |
| **live lineages**, exam seeds                                              | 0.004–0.017 | **live opponent −0.42** | 768 each |

**It is neither the seed nor opponent strength. What kills a frozen tape is an
opponent that can re-plan.** The control is strong by construction: the frozen
top-1% opponents were _produced by_ the same lineages the gate plays live, so
strength is matched and only "thinking versus not" remains. Per-tape spread
against the frozen top-1%: [0.12, 0.33, 0.44, 0.61, 0.66], best 0.83.

Field gate, 768 games each: `recorded_best` 0.0169 [0.0099, 0.0287];
`recorded_median` 0.0039; **`searched_route_policy.py`, our own shipped route,
as a positive control: 0.0013**; generated route **0.0000** [0.0000, 0.0050],
all six lineages 0.0.

**Retraction.** From one tape this investigation claimed "a tape is a plan plus
a seed, and the seed is most of it." **Wrong.** Across 105 tapes × 16 exam
seeds median retention is **0.961**, quartiles [0.646, 0.783, 0.961, 1.153,
1.53]. The tape generalised from retained 0.22 — below the 10th percentile —
and was chosen _because_ it was extreme. What survives:
**Spearman(recorded bank, other-seed bank) = 0.232**, and the **top decile
transfers to 76,210 against a corpus-wide 80,916 — worse than average.**
Do not rank harvested seats by episode bank.

## 6. Generation

64 plans, closed-loop inside the exact simulator, legality-masked, against
top-5% opponents. The real seat that faced those opponents won 0.219.

| temp | bank    | win   | distinct openings | reproduces top opening |
| ---- | ------- | ----- | ----------------- | ---------------------- |
| 0.7  | **269** | 0.016 | 64/64             | 0.000                  |
| 1.0  | **98**  | 0.000 | 64/64             | 0.000                  |

Zero samples reproduce the monoculture — **and that means nothing.** Novelty
measured on garbage is not creativity.

**Verb totals per season.** Sampled: EAST 1078, WEST 1132, NORTH 1013, SOUTH
828, WATER 28, HARVEST 2, CARE 0, COLLECT_FERT 0, FEED 0, PLANT 20, PLACE 0,
SELL 23, HIRE 180. Recorded best: WATER 1043, HARVEST 372, CARE 323,
COLLECT_FERT 365, FEED 337, PLANT 214, PLACE 119, SELL 329, HIRE 302.

**4,051 of ~4,546 unit slots are movement.** It hires 180 hands and walks them
around for 719 turns.

**The absorbing-state mechanism, which is a fact about the game rather than
about this model:** movement is legal almost everywhere and the productive
verbs are legal only in specific states, so once a policy drifts somewhere
nothing productive is legal, the only actions _left_ are movement and PASS,
which drifts it further. It has no representation of a destination because the
16 slots are drawn independently. **A self-play policy with per-unit
conditionally-independent heads inherits this risk.** The mask is not starving
it — the actions are legal and varied, just useless.

## 7. No second life as a proposal distribution

|                       | top-1      | top-2  | top-3  | top-5      |
| --------------------- | ---------- | ------ | ------ | ---------- |
| unit slots            | 0.7553     | 0.8307 | 0.8781 | 0.9275     |
| **active unit slots** | **0.4974** | 0.6522 | 0.7496 | **0.8511** |
| order slots           | 0.9434     | 0.9644 | 0.9728 | 0.9827     |

Read the middle row; the other two flatter it. On slots where a unit acts,
top-1 is a coin flip, and top-5 _out of 44 ops_ misses one slot in seven.
Keeping 5 candidates across 16 slots retains ~`0.851**16` ≈ **7% of the true
turn** while blowing branching to `5**16`. That is a worse search, not a
narrower one.

**Legal-random control**, same rollout with the model switched off: prior mean
bank **21.4** (median 0, max 992, win 0.0) against legal-random **0.0** (median
0, max 0, win 0.0). The prior learned _something_ — it does not instantly
bankrupt itself — but that is a difference between two kinds of failure against
a corpus median of 92,912.

## 8. Outcome conditioning is inert

Each tape scored under its own outcome and under its **opponent's actual
outcome**: deltas −0.00017, +0.00001, −0.00002, +0.00005 nats, about 0.02% of
the loss. **The model does not read the outcome at all**, so the
decision-transformer-style steering was doing nothing, and reweighting the
corpus by outcome would push on the same dead input.

**Why, and it generalises:** a per-tape constant broadcast to all 719 turns is
perfectly confounded with everything else constant about that tape, and the
state features already carry a running proxy for how the season is going (own
money, opponent money). **Any future return-conditioned setup here must use
return-to-go, which varies within the tape.**

## 9. Headline

**Per-token held-out likelihood on this corpus is nearly uninformative about
whether a sampled policy can play.** Perplexity 2.19 over 48 symbols across 85
unseen teams, train ≈ holdout — and a closed-loop bank of 269 against a corpus
median of 92,912, and **0.0000 over 768 gated games**. Against the bar (0.7969
shipped, 1.0000 best) nothing here is progress.

## 10. Operational notes, all learned the hard way

- **`CUDA_DEVICE_ORDER` defaults to `FASTEST_FIRST`**, so torch's `cuda:0` was
  the _reserved_ card. Pin by UUID.
- **The simulator `step` is launch-bound:** 310 ms at batch 4, 144 ms at 64,
  148 ms at 256. Always batch every (plan × opponent × seed) pair into one call.
- **`pgrep -f` matches your own monitor shells** — two jobs launched twice and
  OOMed each other. Use the `ps aux | grep "[f]nord"` form.
- **Averages over mixtures mislead.** Both the 0.784 nats and the 0.7553 top-1
  are inflated by a trivially-predictable padding mode. Ask what an average
  averages over before quoting it.

**Two self-caught errors kept deliberately in the record:** an outcome-flip
test that reflected the bank scalar around a constant, leaving typical tapes
unchanged so it could not have detected anything; and the n=1 seed
generalisation retracted in §5.
