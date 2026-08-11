TOAD VENDORED (partial). 21 files byte-identical to upstream 973a6c6. Commits
db3dea5 (learner + pre-commit exclude), 8f47ae2 (verified-recipe report), on
branch worktree-agent-a787fb13772848793. 381 tests pass. No training run yet.
MY DISPATCH ERROR: isolation:worktree branched from `main`, which is the
4-commit repo TEMPLATE -- no learn/, no encoding.py. Real work lives on
setup-kaggriculture-agent. The agent fast-forwarded (strict ancestor, zero
unique commits, lossless). Branch from the working branch next time.
TWO CORRECTIONS TO FABLE'S RECIPE, BOTH VERIFIED BY ME IN THE VENDORED SOURCE:

1.  `reward_spaces_lux.py:227`: `reward / 500. / max(positive_weight,
negative_weight)`. A /500 normaliser on the WHOLE shaped reward, plus a
    second max-weight divisor. Omitting it makes the signal 500x too large.
    This is the same class of bug that already cost this project a run (value
    target on the coin scale -> max_grad_norm annihilated the policy gradient).
2.  FIVE phases, not four. `conv_phase1_shaped_reward.yaml:66` has
    `teacher_kl_cost: 0.` against phase 2's `0.005`; the cascade
    0.005 -> 0.01 -> 0.001 -> 0.005 runs across phases 2-5.
    **_ TOAD'S PHASE 1 IS GENUINELY TEACHER-FREE SELF-PLAY FROM SCRATCH. _**
    Every teacher-KL argument on this project concerned a mechanism Toad only
    introduces in phase 2, AFTER a shaped phase has produced a competent policy.
    The pure baseline is phase 1 alone and needs no teacher machinery.
3.  Also unreported: an n_blocks curriculum 8 -> 8 -> 16 -> 16 -> 24.
    DECISIONS FOR THE BASELINE PASS:

- Unit head stays faithful. Our units stand on tiles as Lux's do and model.py
  already gathers at each unit's tile, so their per-tile 1x1 conv + gather is
  the natural fit for 20 slots x 44 ops, not a workaround.
- The MARKET HEAD is the one named deviation -- non-spatial, no Lux analogue.
  Smallest thing that works, structurally separate from the vendored actor.
- DO NOT invent a selling reward this pass. No Toad component rewards selling
  well, yet price x quantity decides our game. That gap is the most important
  open question, and letting phase 1 plateau because of it is a CLEAN,
  interpretable result and the strongest argument for the market-reward work
  that follows. Inventing a term now destroys that evidence.

PHASE 1 RAN AND DIVERGED -- ROOT CAUSE FOUND. Commits e6129b9 (reward threaded
through rollout + run script), 81a5e11 (gradient clipping), 81458be/0becaa3
(report). Bank 0.0 at updates 1 and 2 (34,512 / 69,024 steps), but this is NOT
a measurement of Toad's recipe -- the run was diverging, and it was stopped
rather than burn hours on a number that would stay 0.
MY HYPOTHESIS WAS WRONG. I guessed exploding V-trace importance ratios from a
masked/unmasked log-prob mismatch. The policy terms were HEALTHY (0.10, 0.15,
-0.017); the BASELINE term alone carried 1.5e17.
ROOT CAUSE: Toad's `BaselineLayer` (models.py:180) is BOUNDED -- a sigmoid
rescaled to the reward range, i.e. [-1,+1]. Ours is a bare `Linear` inherited
from the coin-scale PPO model. Fed a 1e-5 shaped reward it diverges.
This is the same failure family as the earlier value-target-on-the-coin-scale
bug: an unbounded value head against a tiny reward.
FOUR RECIPE VALUES THE ANALYSIS OMITTED, ALL LOAD-BEARING:

1. `/500.` reward normaliser (+ a max(positive,negative) divisor)
2. phase-1 `teacher_kl_cost: 0.` -- five phases, not four
3. `clip_grads: 10.0` -- in NONE of the five phase YAMLs, in ALL EIGHT run
   configs saved beside their checkpoints. Without it: 6.2e20 in two updates.
4. bounded `BaselineLayer`
   ONE CAUSE: the analysis captured the phase YAMLs and nothing outside them.
   **_ THE PHASE YAMLs ALONE ARE NOT A COMPLETE RECIPE. The run configs stored
   beside their checkpoints carry values the YAMLs omit. _** Anyone repeating this
   needs that; it is the most valuable output of the reproduction so far.
   CLAIM I DID NOT ACCEPT: "throughput makes 2e7 unreachable". The agent's own
   numbers contradict it -- 34,512 steps in ~30 s is ~1,150 steps/s, putting 2e7
   at ~5 h; an independent measurement here gave ~9.7M steps/hour, ~2 h. Sent back
   for re-derivation. Do NOT start throughput work on this; it has been claimed
   and refuted twice already.
   AGENT'S TOP CONCERN, WORTH KEEPING: it shipped two bugs itself and caught both
   only because they failed LOUDLY. The dangerous class is the quiet one -- a
   wrong reward field or a silently-inert V-trace would produce a plausible curve
   and no alarm. Two discrimination checks dispatched: (1) perturb rewards in one
   batch and confirm advantages change, catching an inert V-trace; (2) assert the
   reward against LITERAL expected numbers from a known state, not against
   expressions built from the constants under test -- the vacuous-test trap it
   already fell into once.

**_ BANK LEFT ZERO -- FIRST POSITIVE SIGNAL ON THE RL PATH (2026-08-07 21:54) _**
Bounded value head (commit feed1ac) fixed the divergence. phase1d, verified by
me in the live log:
update 14 483,168 bank 0.0 (max 0.0) loss -0.052
update 15 517,680 bank 0.0 (max 1.0) loss -0.053
update 16 552,192 bank 0.1 (max 4.0) loss -0.088
Max bank 0 -> 1 -> 4 across three consecutive updates. A RANDOMLY INITIALISED
policy completed the full 7-step, 5-day chain (plant, water across days,
harvest, pickup, carry, drop, sell) with NO teacher, NO demonstrations, and no
shaping we invented. The exploration barrier every previous approach stalled on
is being crossed by Toad's recipe unmodified.
CALIBRATE: 4 coins against a corpus median of 125,773, at 2.8% of phase 1, mean
0.1 still at noise level. This is a DIRECTION, not an achievement. The
mechanism works; the agent does not yet.
Loss stable -0.05..-0.09, shaped steady ~0.0107, no trace of the 1.5e17 blowup.
~4.8 h remaining on the 2e7-step phase 1.
THROUGHPUT CLAIM WITHDRAWN by the agent after re-derivation: 3.97M steps/hr from
the run's own log, 2e7 = 5.0 h. Its earlier figure predated THREADS=1. That is
three times throughput has been claimed as the constraint here and three times
it has not been. Stop entertaining it.
DISCRIMINATION CHECKS PROVEN RED THEN GREEN: (1) feeding torch.zeros_like(rewards)
into V-trace turned the advantage check red -- V-trace is not inert; (2)
pointing counts() at farms[1 - seat] turned TWO tests red, including the
literal-valued one -- the reward reads our own seat and fields. 30 toad tests
pass. These guard the QUIET failure class the agent flagged as its top concern.

ATTEMPT 1 KILLED EXTERNALLY at 00:04:34, update 253, 8,731,536 steps (43.7% of
2e7). NOT a crash: no OOM (kernel logs clean, 212 GB free), no traceback (log
captures stderr and simply stops mid-rollout). Killer unknown. Bank 0.0
throughout except the single max-4.0 blip at 552k. jsonl metrics survived
(phase1_1786139116.jsonl, 253 updates); WEIGHTS LOST -- toad_phase1.py never
called torch.save. Toad's monobeast checkpoints continuously; our runner is
the one non-vendored piece and dropped exactly that property.
8.7M flat zero LEANS D1 (nothing pays for selling) but the pre-registered
verdict is at 2e7 and stands -- the relaunch costs the same either way, so
there is no reason to re-decide under partial evidence.
RELAUNCH DISPATCHED with: atomic checkpoints every 25 updates + tested resume
(LR schedule continuity asserted); wandb owned by the runner itself after
commit-then-run (user discipline, now a saved memory:
commit-then-run-never-skip-tracking); setsid + logs under /data (session
scratchpads die with sessions). Phase-1b stays shelved until the full-budget
verdict.
