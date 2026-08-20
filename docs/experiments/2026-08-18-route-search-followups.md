# Route search: what landed, and what must happen before it is run

Written 2026-08-18, when `worktree-route-arena-and-search` merged to `main` at
`f89ad27`. The plan and design are in `docs/superpowers/`; this file records what
the execution learned that the commit messages do not carry.

## The one thing that blocks running it

**There is no held-out gate, and without one a search run produces a selection
maximum rather than a measurement.** `hillclimb` now rotates its seed block per
iteration, but every accepted route is still chosen by the same arena it is
scored in, and the reference engine is deterministic given seed and actions. The
design spec's §5 asks for **128 seat-swapped reference-engine games against the
agent we currently serve, with the win rate's 95% interval excluding 0.5**.
`search/scripts/holdout.py` implements it.

This is the same distinction that cost this project ten RL arms: a number that
moves is not a number that means something.

## Findings that outlive the plan

**`economic_policy` emits bulk `PICKUP`/`PLACE`.** Measured: `["PICKUP","WHEAT",2]`
by turn 14 on seed 223, larger later. The batched simulator's action encoding has
no quantity lane and `sim/units.py` transfers a hardcoded 1, so
`collect_segment(opponent=economic_policy.agent)` was silently diverging from the
reference engine past the opening turns. It now raises. Nothing in production was
affected — RL training rolls out on the reference engine — but if that bridge is
ever used for training, this is why it stops.

**The simulator is faithful to its declared domain, not to the whole engine.**
Its design spec bounds it to states reachable "through the project's legal action
interface", which is the RL policy's action space. Recorded routes from other
competitors use the engine's wider grammar. That is not a defect in the
simulator; pointing it at out-of-domain input was the defect, and `tests/sim`
could never have caught it because those tests drive the same interface.

**The action/observation off-by-one.** `environment.steps[t]["action"]` is the
action chosen from the observation whose `step` field reads `t-1`, so a replay
agent must index `route[step + 1]`. Nothing structurally prevents reintroducing
this; the arena's fidelity test is the only guard.

## Deferred, with rulings

- `mutate` excludes turn 0 (its action is never submitted) but `BUY_PRODUCT` is
  now in the catalogues; whole-order **insertion** was implemented, then removed
  in favour of stratified retarget. It survives in history at `abef0a4` and is
  the first thing to try if the search stalls without finding buy-side gains.
- `build_league` raises mid-harvest on an episode with invalid rewards, leaving
  earlier routes on disk and aborting later archives. Correct for an offline
  script — a loud non-zero exit beats the silent seat-0 harvest it replaced — and
  idempotent on rerun. A caller ignoring the exit code could mistake a partial
  directory for a complete league.
- `route.load` validates length; element types in `rewards` are checked for
  count but not numeric type. Real archives always carry numbers.
- A crashed hill-climb resumes from the seed route, not the last incumbent.
  Accepted routes are written per acceptance, so nothing accepted is lost.

## The pattern that produced most of the value

Seven checks in this plan asserted something true regardless of the bug they
named, and each was caught by breaking the code and watching the test stay
green — never by reading it. A mirror route scores 0.5 whether or not seats are
swapped. `tensor[i] == 1` holds at any dtype. "Two files, 720 turns each" holds
whether you harvest the winner or the loser. In every case the assertion was
fine and the **fixture** was too simple to express the failure.

The habit worth keeping is mechanical: before trusting a green test, mutate the
behaviour it names and confirm it goes red.

## First real run, 2026-08-19/20 — and the first gate PASS

League: three fresh 08-18 tapes, one 08-16 tape, boatlee's public V16-RC5 tape,
and the served `boatlee_v14_policy` as the one reacting opponent. Seed: the
strongest 08-18 harvest (episode 94106275, seat 1). 200 candidates over ~14
hours, ~22% behaviourally inert (textual edits the engine masks), paired SEs of
0.002-0.018 — pairing cancelled the shared-market noise as designed.

**One acceptance** (candidate 149): `SELL WOOL 3 -> 5` at turn 646, +1.95% ±
0.50% paired. Everything else: large edits hurt (down to -49%), small ones did
not clear 2·SE. The predicted fitness valley was confirmed in the log: a
`SELL -> CARROT` retarget scored -6% because a route that grows no carrots
aborts the order and forfeits the original revenue — the hinge strategy needs
plant+sell as one move, which single-edit mutation cannot express. If a future
run is wanted past this ceiling, a two-edit macro-mutation is the widening to
try, not more candidates.

**Gate: PASS.** `holdout` on the final incumbent, one pre-registered run (no
gate-shopping): rate 0.7344, Wilson 95% [0.6518, 0.8032], 128 held-out games
vs the served agent. Embedded as `searched_route_policy.py` and proven
behaviourally identical to the gated route — same final banks (97711, 95180) on
a gate seed via the real engine, with the off-by-one mutation flipping the
result to (40244, 131100), so the fidelity test genuinely bites.

The searched route is, honestly stated, a top player's play plus one wool order:
the pipeline's contribution is not the edit but the measurement — a held-out,
seat-swapped 73% edge over what we field, which no agent of ours has ever had.
