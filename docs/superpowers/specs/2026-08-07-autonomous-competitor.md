# Spec: an autonomous system that can win Kaggriculture

**Goal.** A system that improves the agent, measures honestly, and submits
without a human, such that the agent standing on 30 September is the strongest
we can build — and is strong against the _whole field_, not against one
opponent.

## What winning actually requires

Three facts about this competition's scoring shape the entire design, and two of
them make it unlike a normal leaderboard chase.

**The final leaderboard is a Bradley-Terry tournament over episodes played in
the ~2 weeks after the 30 September deadline.** Kaggle chose that deliberately
to damp hot streaks. So the thing being optimised is not our peak public score
on any given day — it is the strength of the single agent frozen at the
deadline, measured against whatever field exists afterwards. A agent that spikes
to 2900 in August and is stale by October loses to one that is 2400 and still
current.

**Only the latest two submissions are scored.** A bad automated submission does
not merely waste one of five daily slots; it _displaces_ a good agent from the
scored pair. This is why the submission guard must be strict rather than
optimistic.

**Public scores decay as the field strengthens.** Measured on this project:
`pilkwang` peaked at 1289.6 on 1 August and fell to ~1051 by 6 August across
versions its author presumably intended as improvements. Our own route memory
went 1452.3 → 1346.1 in a day without changing. **An undated score comparison is
meaningless**, and any autonomous system reasoning about "are we improving" must
compare against a contemporaneous baseline, never against a remembered number.

54 days remain, and 270 submission slots. Slots are not the constraint. Ideas
that survive measurement are.

## Where we honestly are

| agent                | public LB  | provenance                                        |
| -------------------- | ---------- | ------------------------------------------------- |
| vendored kaito v21.1 | **1965.4** | someone else's code, decoded from a public kernel |
| route memory         | 1346.1     | ours                                              |
| economic policy      | ~1014      | vendored, superseded                              |
| top of ladder        | ~2900      |                                                   |

Our best agent is not ours. That is a fine position to _start_ from and a bad
one to finish in, because a vendored memory decays as the meta moves away from
the episodes it memorised, and we cannot improve code we do not understand well
enough to change.

What this project has established about the approaches available, each measured
rather than assumed:

- **Behaviour cloning caps at its teacher, and badly.** 90.5% held-out action
  accuracy, banks 0 in play. Compounding error: it hires on nearly every turn
  where the teacher hires on 14.6%.
- **Replay caps at the episode replayed.** A blind single-episode tape scored
  1457; 190 retrieved routes scored 1452. The retrieval machinery bought
  nothing.
- **A fixed route beats retrieval.** Measured 186,614 vs 184,500 against
  `starter` and 134,862 vs 128,821 against `economic_policy`.
- **RL is the only method here that can exceed its teacher.** It is also the
  only one not yet working.

## Architecture: three layers, different clock speeds

The system fails if these are conflated. Each has a different failure mode and a
different appropriate degree of autonomy.

### Layer 1 — the mechanical pass (minutes, no judgement, fully autonomous)

Already built: `scripts/autonomous_pass.sh` and
`kaggriculture.scripts.autonomous`. Refreshes the replay corpus, kernel and
discussion caches; reads ladder standing and spent slots; gates a candidate and
submits when it wins.

**Runs from cron under `flock`.** Every step timeout-bounded so a hung Kaggle
call cannot wedge the slot; a failing step never skips later steps.

**The submission guard, which is the only irreversible thing this layer does:**

1. Working tree clean, so a score traces to a commit.
2. `gate.py` reports the candidate beating **the agent `main.py` currently
   serves**, read from the entrypoint rather than assumed — which agent we ship
   has changed four times.
3. At least `RESERVE_SLOTS` remain, so a human can always correct the day.
4. A tie is not an improvement. Absent evidence, the incumbent stays.

### Layer 2 — the judgement loop (hours, an agent, bounded autonomy)

A cron entry invoking Claude with a standing prompt. Each pass: read the live
training numbers, interpret them **against banked coins rather than curve
health**, check them against the research findings and the ledger, and act —
message a running implementer, dispatch the next task, stop a run that cannot
learn, or run the cheapest falsifying diagnostic.

The specific failure this layer exists to catch has already happened twice:
a run whose entropy, KL and value curves all look healthy while the objective
never moves. Smooth curves are not evidence of learning the objective.

**What it may do without asking:** run experiments, stop runs, dispatch
implementers, commit code, and submit through Layer 1's gate.

**What it must escalate:** a design change that invalidates the dataset or the
checkpoint; any change to the submission guard itself; and a result that
contradicts the current strategy rather than refining it.

### Layer 3 — strategy (days, human-in-the-loop by default)

Which approach to pursue at all. This session changed strategy four times —
heuristic → vendored policy → imitation → route memory → vendored memory → RL —
and each pivot came from a measurement that invalidated a premise, not from a
schedule. That is the right cadence, and it is not something to automate now.

## The improvement engine

RL is the only path above the teacher, and it is currently learning the wrong
thing: margin improved from −107,573 to −8,549 over 28 iterations while our own
bank stayed at exactly 0 — **the agent learned to suppress the opponent in the
shared market rather than earn**. The differential reward pays equally for both,
which the spec anticipated in words and permitted in code.

The research (`docs/research/2026-08-07-self-play-rl-findings.md`) gives the
corrections, all from primary sources:

1. **Per-seat value baselines.** With one policy in both seats and a difference
   reward, `R₁ = −R₀`, so a shared baseline makes the advantage cancel exactly.
   Keeping both seats — added as a free doubling of data — is what made it
   exact.
2. **Shaped own-progress reward first, differential second.** All three
   single-box winners did this: FLG shaped for ~65M steps, Toad Brigade for
   20M, Frog Parade used sparse "as soon as training was running stably".
3. **An absolute term must survive into the final reward**, or the agent is
   paid the same for denial as for production. FLG kept `+0.3 per factory`
   alongside their differentials for exactly this reason.
4. **Drop the KL to a clone that banks zero.** Jump-Start RL documents a fresh
   critic unlearning a pretrained policy; Posterior BC shows a clone can fail
   the coverage condition finetuning needs.
5. **γ scaled to a 719-step horizon**, and policy and value gradients clipped
   separately — jointly clipping an unbalanced pair is how the policy component
   gets zeroed, which already happened here once with a value target on the coin
   scale.

**Throughput is not the constraint and further work on it is waste.** Frog
Parade took gold on one workstation at ~434 effective steps/second; we are at
~694.

## Robustness is the win condition, not peak score

Because the finale is a Bradley-Terry tournament against the whole field, the
system must optimise for _breadth_, not for beating one opponent.

- **The gate is a ladder, not a threshold**: BC checkpoint → `economic_policy` →
  the fixed route → vendored kaito. Report the rung reached and stop.
- **Vendored kaito stays out of the training opponent pool.** A number the agent
  trained against is not a gate.
- **The opponent pool holds frozen past checkpoints plus scripted agents that
  actually bank.** OpenAI Five sampled 80% latest-self / 20% past checkpoints;
  the scripted agents are what break the symmetry that zeroes our margin.
- **Never submit an agent that wins on average and loses catastrophically
  sometimes.** Route memory lost seed 99 head-to-head while winning on mean;
  the ladder scores individual episodes.

## What tells us it is working

Measured, not felt:

- **Bank per episode leaves zero and rises.** This is the objective. Everything
  else is diagnostic.
- **Win rate against the _whole_ gate ladder improves**, not against one rung.
- **Public LB improves against a contemporaneous baseline** — our two scored
  submissions move relative to each other, not relative to a remembered number.
- **Illegal-action rate stays at 0.** It is the fastest signal that masking and
  sampling have diverged.

## Risks, and which are already real

- **The agent optimises denial rather than production.** _Already happening_ —
  margin closed 12× with bank at zero.
- **Self-play collapse to a mutually bad equilibrium.** Guarded by scripted
  opponents in the pool and the held-out gate.
- **Automated submission displaces a good agent.** Guarded by the three-condition
  gate and the slot reserve; this is the one irreversible autonomous action and
  deserves the strictest guard in the system.
- **The vendored agent decays.** Its 403 KB of memorised routes were harvested
  from a meta that is moving. Our own route store rebuilds nightly; kaito's
  cannot.
- **We finish shipping someone else's code.** The honest failure mode. Avoided
  only by RL actually working, which is why Layer 2 exists.

## Out of scope

Distillation to a smaller network — the model runs in 0.86 ms against a 1 s
budget. Building a faster simulator — throughput is not the constraint.
Automating Layer 3 — strategy pivots have come from invalidated premises, and
four of five in this project were correct.
