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

Sampled across the two most recent archives:

- Verb mix: `SELL` 6,158, `HIRE` 4,849, `BUY_PRODUCT` 4,836, `BUY_SEED` 840,
  `BUY_ANIMAL` 224, `BUY_LAND` 32.
- Orders per turn: 0 on 3,842 turns, 1 on 3,790, 2 on 1,581, tailing to 7. Never
  close to the cap of 10.
- Quantities: 39 distinct values, heavily skewed small — 1 is the mode (2,598),
  then 2 (1,397), 3 (1,028) — with a long thin tail to 56.

So the head does not need to express ten orders, and it does not need a
fine-grained quantity range. It needs to be good at "sell a few of one thing".

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

Quantities are **bucketed, not regressed**: `0, 1, 2, 3, 4, 5, 6-8, 9-12, 13-20,
21+`, chosen from the measured distribution above. Regression on a long-tailed
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
