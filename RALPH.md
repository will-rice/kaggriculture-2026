# RALPH.md — the loop's memory. Read first, update last, every iteration.

**Mission:** win the Kaggriculture competition. Deadline **2026-09-30**; only
the **latest 2** submissions are scored; finale is Bradley-Terry over the
post-deadline window. Win condition per episode: most coins banked.

## Iteration protocol

1. Read this file, then `docs/experiments/2026-08-07-rl-ledger.md` (tail) for
   anything a parallel session added.
2. `git log --oneline -5`, check running processes and GPU tenancy before
   launching anything.
3. Do the **top unblocked item** below. One item per iteration unless trivial.
4. Update this file: move finished items to the log with the measured result,
   re-rank what remains, record any new fact with its source.
5. Commit (never `git add -A`; the hook runs the full suite ~9 min; prettier
   bounces md once — re-stage and re-commit).

If the session dies, a fresh one resumes with
`/ralph-loop:ralph-loop "win this competition"` — this file carries all the
state, so nothing outside it needs to survive.

## Standing rules (each has cost us before)

- Measure before acting; read a metric's definition in `docs/metrics.md`
  before trusting its name.
- One writer per checkout. Training runs pin to a commit, never a dirty tree.
- The reference engine is the authority; sim changes need the exact-bank
  replay proof (`tests/sim/test_tape.py`).
- Submissions: max 5/day, each displaces the older of the latest-2. The
  untouched promotion gate governs promotion claims; user authorized ONE
  submission for a candidate that passes it (ledger Task 10, 08-27). Routine
  resubmits of already-fielded agents remain allowed.
- Verify subagent claims independently; test on data nobody selected.

## Priority stack

1. **Field our strongest agent.** `main.py` serves `boatlee_v14_policy`
   (ladder ~1107, win 0.36 recent). `searched_route_policy` measures better
   (~1227, 0.66 recent-60). The hybrid work references a stronger frozen
   controller, **Kaito v48**, whose artifact is NOT in this repo — it was
   produced in another session (`/root/optuna_task_8`, Optuna study
   artifacts). Locate or reconstruct v48, arena it against
   `searched_route_policy` on held-out seeds, serve the winner, and refresh
   the scored pair (two submissions to displace boatlee under latest-2).
2. **Execute the market-residual RL plan** —
   `docs/superpowers/plans/2026-08-29-market-residual-rl.md` (approved
   design: frozen logistics controller + learned market residual, recurrent
   state, rolling frontier opponents). This is the only path to an agent that
   _reacts_, which is what the whole field's tapes cannot do. No SDD ledger
   exists yet; start one.
3. **Fresh harvest + widened search.** The served route was harvested 08-18
   and searched with quantity ceiling 64; the ceiling is now 165 and
   archives run through 08-30. A re-run reaches candidates the old search
   structurally could not. Gate through `holdout` (frontier-hardened).
4. **Daily maintenance:** fetch archives, keep the scored pair fresh, watch
   ratings (monitor task exists).

## Log (newest first)

- 08-31: Loop set up. State at `7daee94`: nothing running; hybrid candidate
  tied Kaito v48 at exactly 0.500 on the untouched gate (not promoted;
  submission authorization not triggered); market-residual RL designed and
  planned but unstarted; Phase 1 RL closed — four correct reward fixes, no
  behavioural change; barrier is chain discovery, not specification.

## Facts that keep being re-derived (with sources)

- Ladder is tapes: 162 public kernels, zero learned (kernel survey).
- Winners: +3.77 mean sale price at −92 units — timing, not volume
  (corpus mining; `docs/research/2026-08-08-what-wins-elo.md`).
- Chain economics: nothing sellable at turn 1; short loops net-zero;
  profitable chain ≈7 links, ~120-turn delay (ledger 08-23).
- Engine quotes BUY_PRODUCT at post-buy inventory: round trips net zero,
  not negative (networth-report, 08-23).
- Toad phase-1 budget is 2e7 steps; our arms ran 5× that with no effect.
