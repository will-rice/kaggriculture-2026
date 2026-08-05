# Spec: a market head, so the agent can bank money

**Goal.** Extend the policy's action space to cover market orders, so a cloned
agent can sell what it grows. Phase 2 proved everything else works and that this
one omission makes winning impossible.

## The evidence this exists to answer

Phase 2's behaviour clone reached **0.8566 holdout accuracy** against a 0.175
majority-class baseline, emitting 19 distinct ops — a real clone, not collapse.
It then **lost 0–100 to every league opponent across 400 games**, banking exactly
its starting 3,000 in every episode.

The cause is structural, not a training failure. The engine has exactly one site
that increases money — `farm["money"] += price` inside `_commit_unit`, crediting
a completed `SELL` — and none of the 22 `UNIT_OPS` verbs is a market verb. No
assignment of weights to that action space can score a single coin. `starter`
finished on 3,501 and won by doing nothing much.

## The action space, from the engine

`_parse_order` accepts six verbs in two shapes:

| verb          | shape             | notes                                                     |
| ------------- | ----------------- | --------------------------------------------------------- |
| `HIRE`        | `["HIRE"]`        | atomic; one hand per order, priced `fib(n)`, resets daily |
| `BUY_LAND`    | `["BUY_LAND"]`    | atomic; unlocks the next quadrant in `LAND_ORDER`         |
| `SELL`        | `[verb, item, n]` | draws from the **shed**, never a unit's carried inventory |
| `BUY_SEED`    | `[verb, item, n]` |                                                           |
| `BUY_PRODUCT` | `[verb, item, n]` |                                                           |
| `BUY_ANIMAL`  | `[verb, item, n]` |                                                           |

`n` must parse as an int and be `> 0`, or the order is dropped. The queue is
truncated to `maxMarketOrdersPerTurn`, which defaults to **10**. Orders execute
in per-unit lockstep across both players, so each successive unit sold is quoted
at a price the previous sale already moved — selling ten in one order is not ten
sales at the opening price.

Items: 9 products (`CARROT`, `EGG`, `FERTILIZER`, `MELON`, `MILK`, `STRAWBERRY`,
`TOMATO`, `WHEAT`, `WOOL`), 5 crops for seed, 3 animals.

## What the corpus actually does

Measured across **all six** archives, 10 episodes each, 86,400 seat-turns. An
earlier draft of this spec sampled two archives and drew two conclusions that the
full corpus contradicts — the exact mistake this repo has now made twice.

- Orders per turn: 0 on 42,333 turns, 1 on 22,981, 2 on 8,861, then a decreasing
  tail — **except at exactly 10, which occurs 903 times, more often than 9
  (202)**. That spike is the `maxMarketOrdersPerTurn` cap clipping agents who
  wanted more. The head must be able to emit a full ten orders; the two-archive
  draft claimed the cap was never approached.
- Verb mix shifts hard across the month. On 07-30 `BUY_SEED` (1,118) outnumbered
  `SELL` (837); by 08-04 `SELL` (8,084) dominates and `BUY_SEED` is 1,080. Any
  statistic here must be reported per archive.
- Quantities: 62 distinct values to a max of 85. **34.5% are exactly 1**, 51.8%
  are ≤2, 72.6% are ≤4, and **93.5% are ≤12**.
- Item sets per verb are narrower than the engine allows: `BUY_SEED` uses all 5
  crops, `SELL` all 9 products (`WHEAT` 12,148 leading), `BUY_ANIMAL` all 3, but
  `BUY_PRODUCT` is `WHEAT` 27,635 against `FERTILIZER` twice — effectively a
  single-item verb, because wheat is the feed for every animal.

**Aggregation is very nearly lossless.** 22.3% of order-bearing turns repeat a
`(verb, item)` pair, but that is overwhelmingly `HIRE` — atomic at one hand per
order, so repetition is the only way to hire several, and the design already
models it as a count. Non-`HIRE` repeats are ~1.8%, almost all a doubled `SELL`.
Merging those two sells into one changes their interleaving against the
opponent's queue under lockstep quoting; that is the fidelity this design trades
away, and it is small enough to accept.

## Design

**Recommended: per-item quantity heads, not order slots.**

An order _list_ is a sequence, and modelling it as one invites an autoregressive
decoder for a distribution that is 80% zero-or-one orders. The information a
policy actually carries is _how much of each item to move_, so emit that
directly:

- `SELL`, `BUY_SEED`, `BUY_PRODUCT`, `BUY_ANIMAL` × their item set → one
  quantity distribution each.
- `HIRE` → one count distribution (how many hires this turn).
- `BUY_LAND` → one binary.

Quantities are **bucketed, not regressed**. The measured distribution puts 93.5%
of orders at 12 or below, so the buckets are exact there and coarse above:
`0, 1, 2, …, 12, 13-16, 17-24, 25-40, 41+` — 17 classes, of which the first
thirteen are exact values rather than ranges. Regression on a long-tailed
count trains toward the mean and emits 2.7 of a melon; classification over
buckets matches how the corpus behaves and keeps the loss comparable to the unit
head's cross-entropy.

Serialising back to an order list is then deterministic: emit the non-zero
entries, capped at `maxMarketOrdersPerTurn`. **Order within the list matters**
because of lockstep quoting, so the serialisation must pick one and justify it —
sells before buys is the obvious default, since selling funds buying within the
same turn.

The rejected alternative — `MAX_ORDERS` fixed slots each predicting
`(verb, item, quantity)` — is worth a paragraph in the plan explaining why not:
it has to learn a slot-ordering convention the game does not define, and permuted
labels for identical semantics would teach it noise.

**This is a second head on the existing trunk**, not a bigger network. The market
scalars already feed the trunk through the MLP branch built in Phase 2, and a
test already proves they reach the output. The head reads the pooled trunk — this
is genuinely global, unlike unit ops, which is why the pooled readout that was
wrong for units is right here.

## What changes downstream

Shapes change, so **the dataset must be rebuilt** — that is now a clean
operation, since `write_dataset` clears the directory before writing.

- `encoding.py`: `encode_market(action) -> labels`, `decode_market(logits) ->
list[order]`, and the bucket table. The unit encoders are untouched.
- `dataset.py`: rows gain the market labels.
- `model.py`: a second head; `forward` returns both.
- `train.py`: a second loss term. **State the weighting and justify it** — the
  unit head sees ~4 acting units per turn against the market head's mostly-zero
  targets, so an unweighted sum lets units dominate. Measure both accuracies
  separately; a combined number hides which head is learning.
- `play.py`: merge both decodes into one action dict. `decode_units` already
  returns `{"farmer": …, "hands": …, "market": []}` with the empty list waiting.

## The gate

Unlike Phase 2's gate, this one is reachable — but set it honestly:

1. **Beats `starter`.** The floor. `starter` banks 3,501; anything that sells at
   all should clear it.
2. **Beats `heuristic-v2`** (banked 54.6k against the tape).
3. **Approaches `economic_policy`** (banks 118k against the tape's 140k).

Rung 3 is the real target and probably out of reach for pure cloning — the
corpus is dominated by one public kernel, so cloning it well means approaching
it, not beating it. Say which rung was reached and stop; do not tune until the
number looks better.

Report holdout accuracy for **each head separately**, and report bank, not just
win rate — an agent that sells badly still beats one that cannot sell, and the
bank says how far it got.

## Risks

- **Cloning caps at the teacher.** The ceiling here is the corpus, and beating
  `economic_policy` needs RL, which is Phase 4. This phase is about clearing the
  floor and producing a policy worth fine-tuning.
- **Lockstep pricing is not modelled.** The head emits quantities against the
  prices it _observes_; the price it _gets_ moves as the order fills, and the
  opponent is selling into the same book on the same turn. A clone inherits
  whatever the teacher learned about this implicitly. Do not try to model it
  here.
- **Bucket boundaries are a hidden hyperparameter.** They come from a two-archive
  sample, and the ladder's mix changes daily — the corpus statistic that started
  Phase 2's worst mismeasurement varied from 0% to 8.1% across archives.
  Measure the quantity distribution across **all** archives before fixing the
  table, and record the per-archive spread.

## Out of scope

RL fine-tuning (Phase 4). Shipping weights in the submission archive — but note
`package.py` currently has no mechanism to ship weights at all, since `EXCLUDED`
drops `learn/` wholesale, and **this phase is the first that produces a model
worth submitting**, so it must build one.
