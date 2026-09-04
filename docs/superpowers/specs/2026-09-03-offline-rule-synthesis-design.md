# Offline rule synthesis — design

2026-09-03. Track B of two. Track A (a factorized transformer trained by
gradient descent on `P(a_t | s_1:t, a_1:t-1)`) is deferred until this is tried,
because this needs no training and no submission-time inference budget.

## The idea

An LLM writes **decision rules** offline. The exact simulator verifies each
candidate at 640 games. Results feed back and the LLM proposes the next batch.

The distinction that makes it work: **you cannot precompute the answer, but you
can precompute the function.** Which plan to switch to depends on the shop
draw, the seed, and the opponent's play — none known until runtime. The _rule_
mapping those observations to a choice is fixed, and that is what gets written
in advance.

This is what the strongest agent in the competition already is.
`yhay81/three-day-shop-router` branches exactly once, at step 216, on a
condition evaluated live: the first three unlocked shops are
`ICE_CREAM, FARMERS, FARMERS` **and** the rival holds at least two goose tiles.
The decision is made during the game; the rule was written before it.

## Why this shape

**No training.** Sidesteps the wall a competitor documented publicly on
2026-09-01: PPO with the full action space, JAX at ~10k steps/s, plateaued at
80k terminal cash — below the scripted top band — with "logits very saturated"
and exploration collapsing. Those are gradient-descent failure modes. An LLM
proposing structurally different rules does not collapse toward a mode.

**No latency budget.** `actTimeout` is 1.0s per step over 719 steps. Emitting a
full 26-slot action as JSON is 150-200 tokens, which fails that budget on CPU
by roughly 10x. Evaluating a precomputed rule costs microseconds, so the
submission ships a small rule evaluator and the GPU question becomes irrelevant
to the submitted artifact.

**A perfect verifier.** The batched simulator reproduces the reference engine's
final banks to the coin on 96 of 96 seasons, and 105 recorded tapes replay to a
median difference of 0 coins. Every proposal is checked before it is believed,
which converts an LLM's characteristic failure — confident nonsense — into a
caught error.

**The search space is measurably wide open.** Two numbers in a day-0 wheat
trade moved our own agent from FIELD 0.6641 to 0.9115 and 640-0-0 against 61.8%
of ladder occupancy. Nobody in the top band had tried it. Six of seven
dissected agents emit a byte-identical step-1 market queue.

## The constraint this does NOT escape

A rule set is still a fixed strategy, and we measured what happens to those.
Same tapes, one factor at a time: unseen seeds cost 0.054 of win rate, stronger
opponents cost 0.074, and **a live opponent that re-plans costs 0.42** —
0.434 against frozen top-1% opponents versus 0.004-0.017 against the same
lineages live, with strength matched by construction.

So the rules must **condition on the opponent**, not merely on our own farm.
This is why it matters that the engine gives us the opponent's complete board
at call time: `farms[1]` carries their full 10x10 tiles, farmer, hands,
hires_today, money and unlocked_quadrants. Their `private` (seeds, shed) is
hidden and their actions are never exposed — only state, from which actions are
partially inferable across turns.

## Architecture

Four components, each independently testable.

**`RuleSpec` (Pydantic).** A rule is a condition over observable state plus an
effect. Conditions read: step, shop draw, our cash and holdings, market prices
and inventory, and the opponent's board. Effects: choose a plan branch, insert
or modify market orders, adjust a threshold. `model_config =
ConfigDict(frozen=True, extra="forbid")`, matching the convention already in
`learn/toad/config.py`. The schema is what constrains the LLM's structured
output, so an unparseable proposal is rejected at the boundary rather than
becoming a bad agent.

**`RuleEvaluator`.** Applies a `RuleSpec` to an observation and returns the
action modification. Pure, fast, no allocation in the hot path. This is what
ships in the submission.

**`Synthesizer`.** Runs `codex exec` non-interactively with the schema, the
game's mechanics, the current best rules, and the last batch's gate results.
Returns candidate `RuleSpec` objects.

**`Gate`.** The existing `field_gate.score_field` at 640 games on exam seeds
700000-700063, against all six lineages. Unchanged semantics, because every
number in this project's record was produced by it.

## Data flow

    codex exec + schema + prior results
        -> candidate RuleSpec objects (structured output, validated at parse)
        -> RuleEvaluator wraps the base agent
        -> batched simulator, non-exam seeds, cheap screen
        -> survivors only: field_gate at 640 games on exam seeds
        -> results appended to the record, fed back to the next round

## Measurement discipline

Non-negotiable, each item earned by a failure on this project.

- Seeds 700000-700063 are **exam seeds**: measurement only, never fit on.
- **Nothing under 640 games is a result.** A paired McNemar test on 640
  identical deterministic games at p=2.1e-08 once predicted almost nothing
  here; the exam block retained 6-8% of it. Within-block significance answers
  "real on these boards", not "transfers".
- **Never optimise mean bank margin.** A searched tape that banked +9,593 more
  lost 0.24 field points, because margin and a relative-bank win condition are
  opposed in this game.
- A crashed agent banks **exactly 3000** with ERROR statuses and
  `arena._run_banks` raises on it. A bank of 0.0 with clean ACTIVE statuses is
  a different failure — the ctypes collision. Never run two compiled agents in
  one episode.
- **Never rank candidates by recorded bank.** Spearman between recorded bank
  and bank on other seeds is 0.232, and the top decile transfers _worse_ than
  average.
- The bar: our shipped `squeeze_v58` gates 0.9115, `counter_43_38` 1.0000.
  A rule set below 0.9115 is not progress.

## What is shared with Track A

Deliberately, so Track A stays cheap if this fails: the same `ContractSpec`
defining what is in `s_t` and `a_t`, the same Pydantic action schema, the same
rollout, the same 640-game gate, the same checkpoint stamping. Only the
renderer differs — text for B, tensors for A.

## Open questions, to be settled by measurement not argument

1. Whether the submission container provides a GPU. Irrelevant to this design,
   but it decides whether a per-turn LLM variant is ever possible. Settle with
   a probe agent; a submission slot is cheap.
2. How rich a condition language the evaluator needs before rules can express
   something the top band has not already tried.
3. Whether `codex exec` produces structurally diverse proposals or converges on
   variations of what it is shown. Measured by clustering proposals, the way we
   clustered agent plans.
