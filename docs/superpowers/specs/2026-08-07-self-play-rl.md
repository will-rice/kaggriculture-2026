# Spec: self-play RL — the only thing here that can exceed its teacher

**Goal.** Train the two-headed policy by self-play PPO until it beats the
vendored kaito agent, which is currently the strongest thing we have.

## Why this and not more of what we have tried

Every method this project has used is bounded above by its data:

| approach                           | result                        | ceiling                     |
| ---------------------------------- | ----------------------------- | --------------------------- |
| hand-written heuristic             | 646.8                         | our own judgement           |
| vendored `economic_policy`         | 1013.8                        | that kernel                 |
| blind single-episode tape          | 1457.1                        | that episode                |
| behaviour cloning                  | banked 0                      | the corpus, if it worked    |
| route memory, 190 routes retrieved | 1452.3                        | the best route in the store |
| single fixed route, no retrieval   | beats retrieval by 4.7%       | that one route              |
| vendored kaito                     | pending, kernel scored 2863.9 | that author                 |

Cloning copies a teacher, replay _is_ a teacher, retrieval picks among
teachers. None can be better than the best demonstration it was given. RL is
the only method available that improves past what it was shown, and it is what
the winning solutions in comparable competitions used.

## What we already have

- **A full-state encoder.** 48 board planes, 64 scalars, audited field by field
  against the engine, including the private shed, seeds and carried inventory
  that were missing until yesterday. `tests/learn/test_encoding.py` pins each
  against the engine's own constructors.
- **A two-headed network.** 10.2M parameters — an 8-block, 256-channel residual
  trunk, a per-unit head that gathers at each unit's tile, and a pooled market
  head. 0.86 ms/turn on two threads.
- **A behaviour-cloned checkpoint.** Holdout accuracy 0.7864 units, 0.9805
  market. It plays badly, but it is a far better initialisation than noise.
- **A league.** `Harness` already runs episodes across a `ProcessPoolExecutor`,
  with Wilson intervals and four strength tiers of opponent.
- **Measured throughput.** 1.29 s per 719-turn episode with trivial agents;
  128M game steps per hour across 64 cores. **300M steps — the Lux S3
  runner-up's entire budget — is about 2.3 hours of pure simulation here.**
  Network inference will cost several times that, so a full run is a day or
  two, not a month. This is affordable, which is the fact that makes the whole
  plan viable.

## Design

### Algorithm

PPO with clipping, GAE-λ, and entropy. γ close to 1 — the episode is 719 turns
and the payoff is terminal, so discounting hard would blind it to the second
half of the season.

### Action masking is the first-order concern

Most of the 22 unit ops are illegal on any given turn: you cannot `HARVEST` an
empty tile, `PLANT` without a seed, or `PICKUP` away from the shed. An unmasked
policy spends its budget rediscovering legality instead of strategy.

The mask must be **derived from the engine's own checks**, not written from the
rules documentation. This project has been bitten three times by exactly that
gap — a verb the docs omit but the engine implements, an item gate narrower
than the catalogue, a `step` key present for one seat only. Read
`_apply_unit_action` and `_process_market`, mirror their conditions, and assert
the mask against the engine by executing candidate actions and checking which
the engine actually accepted.

### Reward

Terminal bank is one scalar after 719 decisions and is close to unlearnable.
Use the **per-turn change in bank differential** — our bank minus theirs, turn
over turn — which is dense, sums to the terminal margin, and matches the win
condition rather than a proxy for it.

Do not add shaping terms for intermediate goals like planting or hiring. Every
shaped reward here would encode our own beliefs about good play, and the whole
point is to exceed them. The one exception worth considering is an illegal-
action penalty, and only if masking proves insufficient.

### Opponent pool

Training against one opponent produces an agent that beats one opponent. The
pool holds the vendored kaito agent, the fixed best route, `economic_policy`,
and periodic checkpoints of the policy itself. Sample opponents rather than
fixing them, and hold kaito out of training as the evaluation gate so the
headline number is never one the agent trained against.

### Initialisation and drift

Start from the BC checkpoint and penalise KL divergence from it, decaying the
penalty over training. Early PPO updates on a competent policy are destructive,
and this is what the comparable published solutions used to survive them.

## The gate

1. **Beats the BC checkpoint it started from.** If it cannot, the loop is
   broken, not the idea.
2. **Beats `economic_policy`** — 1013.8 on the ladder.
3. **Beats the fixed route** — our current best local agent that is ours.
4. **Beats vendored kaito** — the real target, and the only rung that means we
   have exceeded what was available to copy.

Report bank and win rate with Wilson intervals at every rung, and report the
illegal-action rate, which is the fastest signal that masking is wrong.

## Risks, stated rather than solved

- **Self-play collapse.** Two copies of one policy can agree on a bad
  equilibrium that loses to anything outside it. The opponent pool and the
  held-out gate are the guard; measure against fixed opponents throughout, not
  only against self.
- **Reward is a differential, so a draw-ish policy is stable.** An agent that
  neutralises the opponent scores as well as one that outproduces them. Watch
  absolute bank alongside the differential.
- **Masking bugs are silent.** A mask that is too permissive wastes budget; one
  that is too strict removes a winning move from the action space and nothing
  ever raises. Assert it against the engine.
- **The BC checkpoint is weak.** It banked 0 in play. Initialising from it may
  be worse than initialising fresh, and that is an experiment, not an
  assumption — run both for a short budget and compare.
- **Throughput may not hold.** 128M steps/hour is with trivial agents. Measure
  it again with the network in the loop before planning a run length around it.

## Out of scope

Distillation to a smaller network for inference. The current model already runs
in 0.86 ms against a 1 s budget, so there is nothing to buy.
