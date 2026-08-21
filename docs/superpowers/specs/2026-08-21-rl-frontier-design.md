# RL to the frontier: widen the action space, clone the meta, then fine-tune

**Design, 2026-08-21.** Engine `kaggle-environments` 1.32.7. The bar is
**frontier-competitive**: holds its own against fresh top-30 tapes on a held-out
arena — the bar the searched route failed (4-to-1 losses above rating 1400) and
the only one the ladder pays.

## 1. The evidence this design is bound by

Ten RL arms failed at `win_rate_vs_econ` 0.000. What was measured on the way:

- **From-scratch is dead at our budget** (verbatim Toad phase 1 banked ~0 at
  2e7 steps). Clone-init is load-bearing. Teacher KL 0.005 holds; 0.001 cliffs
  within a sync cycle.
- **Bank-proportional rewards are ~98% market draw**; the margin reward is
  inert at low skill; shaped rewards get pumped by free actions.
- **The value target is fixed and proven**: it reaches its supervised ceiling
  (0.348 vs 0.344 pooled EV) when given enough gradient per sample; the in-loop
  constraint was one pass per sample. `value_passes` exists; arm S was trending
  positive (live EV +0.22/+0.11, `baseline_passes` 25% under `baseline`) when
  the 1.32.7 bump killed it at u221. **Arm T resumes that config on 1.32.7 now**
  (run `ed688nrd`); its curves set Phase 3's `value_passes`.
- **`TRANSFER_QUANTITY = 1` is a structural ceiling.** The RL action space
  cannot move more than one item per `PICKUP`/`PLACE`; 78% of a top route's
  transfers are bulk (283/362 in episode 93454366, median quantity 6). This
  taxes logistics ~6x, makes top-play demonstrations partly inexpressible for
  BC, and keeps recorded tapes outside the batched simulator's domain.
- **Assets the failed arms never had**: ~3,500 top-30 episodes on the current
  engine (~700/day since 08-16); an honest objective axis (arena + held-out
  gate, seat-swapped, real opponents); a fidelity-proven batched simulator whose
  declared domain is exactly the RL action space.

## 2. Phase 1 — the quantity lane (~1 week)

**Change.** A second per-unit head on `Policy` emits logits over the existing
`QUANTITIES` bins (the market head's idiom), sampled jointly with the op,
decoded only for `PICKUP`/`PLACE`, masked to the 1-bin otherwise. Cascade,
known and finite: `model.py`, `learn/encoding.py` (decode; `TRANSFER_QUANTITY`
dies), `learn/mask.py`, `ppo.joint_log_prob`, Toad loss/entropy plumbing,
`Trajectory` (+1 field), and the simulator — `unit_actions` gains a quantity
lane, `sim/units.py` transfers per-unit amounts, and `sim/rollout.py`'s
bulk-transfer guard inverts from _raise_ to _encode faithfully_ (its tests
invert with it).

**Proof.** Replay recorded 1.32.7 top episodes — both seats — through the
widened simulator and require **exact bank equality**, several episodes. The
assertion that exposed the limitation becomes its differential proof, and it
exercises branches the RL-action-space tests never reach.

**Kill gate.** Exact-bank replay unachieved inside the phase budget → stop and
reassess; fallback is reference-engine rollouts for the BC path (correct, ~25x
slower, ~4 s/episode).

**Also this week (small, standalone): harden the holdout gate.** Fresh
top-band tapes join `holdout`'s exam league beside the served agent, so PASS
means "survives the current meta". This is the recorded lesson of 2026-08-20,
and it serves the ladder side regardless of RL.

## 3. Phase 2 — behaviour-clone the meta (~4 days)

Clone the **winning seats** of top-rated 1.32.7 episodes into the widened-head
policy, reusing the existing imitation machinery and `learn/corpus` selection.
The old clone's documented failures are addressed by construction: it trained
on old-engine data (void), and its targets were partly inexpressible (the
quantity lane fixes that).

**Gates, in order:**

1. Action-level: top-1 agreement on held-out episodes, **including transfer
   quantities**, reported per head.
2. **Beats `economic_policy` on held-out seeds** — the rung no RL arm reached.
3. Recovered-bank fraction against the same episodes replayed as tapes.

**Kill gate.** A clone that cannot beat `economic_policy` is too weak a base
for RL to rescue — the community's documented failure mode ("the base
imitation is just too weak") — and the program stops _before_ GPU-weeks, not
after.

## 4. Phase 3 — RL fine-tune (~2–3 weeks)

Only what the evidence supports: clone-init; teacher KL 0.005; value warmup;
corrected critic with `value_passes` (N from arm T's curves); Toad's shaped
reward; opponent mix of **frontier tapes replayed in-simulator** (~100
games/s — what Phase 1 uniquely unlocks) plus mirror play.

**Objective axis:** arena win rate against _held-out_ fresh tapes never used as
training opponents, read beside within-turn EV. **Ship rule:** passes the
frontier-hardened holdout gate → serve and submit. **Kill gates:** within-turn
EV and the win-vs-tape trajectory, budgets fixed before launch in the plan.

## 5. Non-goals

- No reward-function research: shaped reward as-is; the margin/differential
  line stays closed (measured inert at full budget).
- No from-scratch arms, no architecture search, no MCTS.
- No changes to the route-search stack beyond the gate hardening; ladder
  maintenance (resubmits) continues in parallel untouched.

## 6. Risks, stated

- **The cascade is the risk in Phase 1.** It touches the fidelity-proven core;
  the exact-bank replay proof and the existing 90+ sim/differential tests are
  the containment. Budgeted at a week for that reason.
- **BC may cap below the frontier** even with expressible targets — tapes
  cannot teach reacting. Phase 3's tape-opponent mix is the counter; if the
  clone plateaus under econ, the kill gate fires and the residual value is the
  quantity lane itself, which the meta side also wants.
- **The deadline is 2026-09-30.** Timeline fits with ~1 week of margin; the
  ladder pair (boatlee v14 + searched copy) holds the fort meanwhile.
