# Run the recipe: the five-phase Toad curriculum, faithfully

**Design, 2026-08-22.** Engine `kaggle-environments` 1.32.7.

## 1. The finding this rests on

In August the project surveyed seven published RL solutions against four
criteria and chose Toad Brigade (Lux AI S1, 1st) to reproduce: an economy game
of gather-carry-deposit, trained on one personal workstation, with full public
training code down to the five phase config YAMLs. The survey is at
`docs/research/2026-08-07-fable-rl-grounding.md` and it is good work. The
choice still looks right.

**The recipe was then never run.** Every arm from A to T executed phase 1 of a
five-phase curriculum, and executed it with settings from phase 2:

|                   | the recipe's phase 1         | every arm we ran              |
| ----------------- | ---------------------------- | ----------------------------- |
| `teacher_kl_cost` | **0**                        | 0.005                         |
| teacher           | **none**                     | a behaviour-cloned checkpoint |
| trunk width       | 128                          | 256                           |
| phases 2-5        | sparse reward, self-teaching | never implemented             |

The recipe is explicit that the teacher is never an imitation model:
_"No behaviour cloning, no replays, anywhere. Teachers are always the
pipeline's own earlier, smaller checkpoints."_ Ours anchored a from-random
policy to a clone that banks 385, at a KL cost the recipe applies only after
phase 1 has produced something worth anchoring to.

This dissolves several results that were read as findings about RL:

- _"Removing the teacher took the bank 17,675 -> 9 in five updates."_ That
  removed a teacher mid-run from a policy trained with one. It is not evidence
  about phase 1 running teacher-free by design, which is what the recipe
  specifies.
- _The BC clone's quality, DAgger, and corpus expressibility._ All moot: the
  recipe uses no demonstrations at any phase.
- _A learning-rate sweep._ The recipe already fixes lr at 1e-4 for phases 1-2
  and 5e-5 for 3+. Our arms used the right lr; the teacher was wrong.

**Cost.** The research file's own budget check: phase 1 is about six hours on
this box, the whole published floor (~9e7 steps) about 26 hours. Weeks were
spent on a misconfigured first phase of a 26-hour recipe.

## 2. The schedule, verbatim from the configs

| Phase | Net      | Steps   | Reward      | teacher_kl | lr   | entropy | lambda | Teacher                       |
| ----- | -------- | ------- | ----------- | ---------- | ---- | ------- | ------ | ----------------------------- |
| 1     | 8-block  | 2e7     | shaped      | **0**      | 1e-4 | 1e-3    | 0.8    | **none**, random init         |
| 2     | 8-block  | 1e7     | sparse +/-1 | 0.005      | 1e-4 | 1e-3    | 0.8    | phase 1's own weights, frozen |
| 3     | 16-block | 2e7     | sparse +/-1 | 0.01       | 5e-5 | 2e-4    | 0.8    | frozen 8-block                |
| 4     | 16-block | 2e7     | sparse +/-1 | 0.001      | 5e-5 | 2e-4    | 0.8    | continue                      |
| 5+    | 24-block | 2e7/run | sparse +/-1 | 0.005      | 5e-5 | 2e-4    | 0.9    | frozen 16-block               |

Fixed across all phases: Adam with **eps 3e-4**, lr annealed to 0.01 of
initial, **gamma 0.999**, loss = V-trace PG + UPGO PG + 1.0 x TD(lambda)
baseline + `teacher_kl_cost` x KL(teacher||student) - `entropy_cost` x summed
entropy, reduction `sum`. On a reward switch, **4,000 `baseline_only` warmup
batches** let the value head adapt while the policy holds still.

Phase 1's reward already matches: five weighted per-turn deltas plus the
terminal result at 10x, selfish, scaled per step. `toad_reward.py` carries
`GAME_RESULT_WEIGHT = 10.0` and the rest. Nothing to change there.

Opponents: **pure mirrored self-play, latest weights both seats, no opponent
pool.** Stability comes from the frozen-teacher KL, not from an opponent mix.

## 3. What has to be built

The current script implements phase 1 only, with several knobs welded shut.

1. **`--teacher` takes a checkpoint path**, not a boolean bound to the BC
   clone. Phases 2-5 each anchor to a specific earlier checkpoint of our own.
   The flag's help text must stop describing the clone.
2. **A sparse +/-1 terminal reward field** for phases 2+: +1 for the win, -1
   for the loss, zero-sum, nothing else. Check whether the existing `margin`
   field already is this before adding a second one; if it is, say so and use
   it.
3. **`--blocks`**, for the 8 -> 16 -> 24 progression.
4. **`--lambda`** if TD(lambda)'s value is not already settable, for the 0.8 ->
   0.9 change at phase 5.
5. **`--value-warmup-batches`** with the recipe's 4,000, replacing the current
   boolean.
6. **A phase runner** that executes the schedule end to end, so a phase
   boundary is a config change rather than a hand-typed command. Every phase's
   full config goes into its run's logged config, which the metric work of
   2026-08-21 already makes legible.

Each is a small change to `learn/scripts/toad_phase1.py`, which should be
renamed to reflect that it runs a curriculum rather than one phase.

## 4. Gates between phases

A phase is not merely "2e7 steps elapsed". Before advancing:

- **Phase 1 -> 2:** the shaped return must be rising, and mean terminal bank
  must exceed the best any prior arm reached under the misconfigured phase 1
  (**36,189**, the gate figure recorded on 2026-08-07). Below that, the
  faithful phase 1 is not beating the unfaithful one and something else is
  wrong -- stop and diagnose rather than proceeding.
- **Phase 2 -> 3:** win rate against `economic_policy` must be **> 0.000**.
  Eleven arms never cleared this. It is the first rung that has ever mattered.
- **Phase 3 -> 4 -> 5:** each phase must not regress the previous phase's win
  rate against `economic_policy`, measured on held-out seeds.
- **Ship:** the frontier-hardened holdout gate, unchanged.

## 5. Deviations, named as the research file demands

Carried forward from `A.4`, plus new ones:

1. **The market head.** Lux has no market; no part of the reproduced
   curriculum teaches trading. The research file calls this "the riskiest by
   far" and it remains unhedged: a faithful reproduction can reach six-figure
   banking and still lose on market play.
2. **PPO in place of IMPALA's exact loop.** V-trace/PPO is low-risk per FLG's
   own measurement, but **UPGO has no PPO analogue** and must not be silently
   dropped -- if it is absent, say so in the run log.
3. **719 turns against their 360**, same gamma; per-step reward scale becomes
   +/-1/719.
4. **Trunk width 128**, per the config, not the 256 our arms used.
5. **The quantity lane.** Our action space now carries per-unit transfer
   quantities and market quantities to 165, which Lux has no analogue for.
   This is ours and is not in their recipe.

## 6. Non-goals

- No behaviour cloning, no DAgger, no corpus training. The recipe uses none,
  and the corpus stays what the earlier decision made it: **validation** --
  bank percentile against the corpus distribution, and state coverage as the
  early warning for self-play drifting into a private equilibrium.
- No hyperparameter search. Every constant is specified by the configs; a
  sweep is what you do when you do not have the recipe.
- No changes to the simulator, the search stack, or the gate.
