# Quantity Lane Implementation Plan (RL Frontier, Phase 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the action space a per-unit transfer quantity — through the simulator, the policy, and every layer between — proven by replaying recorded top episodes to exact bank equality; and harden the holdout gate with frontier opponents.

**Architecture:** Bottom-up. The simulator learns to _execute_ per-unit quantities first (Task 1), then to _encode_ them from the action grammar (Task 2), which together make the exact-bank replay proof possible (Task 3). Only then does the policy grow a quantity head (Task 4) and the learning stack thread it through (Task 5). The gate hardening (Task 6) is independent and can run any time.

**Tech Stack:** Python 3.11, PyTorch, `kaggle-environments` 1.32.7, pytest, `uv`.

## Global Constraints

- Engine is `kaggle-environments>=1.32.7`; do not upgrade it in this plan.
- `tests/sim` must stay green throughout, including `test_reference_tripwire.py`.
  Where a sim test's _premise_ changes (the bulk-transfer guard inverts), the
  task says so explicitly; nothing else in that suite may be weakened.
- The reference engine is the authority. When simulator and reference disagree,
  the simulator is wrong; never re-record the reference to fit.
- Python 3.11; Google-style docstrings; absolute imports; no
  `from __future__ import annotations`; prefer `nn.functional`; `.tile` over
  `.unsqueeze(1).expand`; `.numpy(force=True)`; `.flatten` for collapsing dims.
- Required argparse args take no `--` prefix; optional ones do.
- Never `git add -A`; stage explicit paths. Commit with `git commit -F <file>`
  (message explains _why, not what_; no Claude reference of any kind). The
  pre-commit hook runs the full suite (~3.5 min, longer than the default Bash
  timeout — pass an explicit longer timeout, run in the foreground, never edit
  files while it runs).
- Never end a turn waiting for a background signal; if a command returned, act
  on its output. Check the process table before deciding you are blocked.
- Key constants: `MAX_UNITS = 20`; `QUANTITIES = (0, 1, ..., 12, 16, 24, 40,
64)` (17 bins, dense through 12 — every observed transfer quantity in top
  play is exactly expressible); `TRANSFER_QUANTITY = 1` at
  `learn/encoding.py:660` is the constant this plan deletes.

---

### Task 1: The simulator executes per-unit transfer quantities

The engine reads `n = int(action[2])` for `PICKUP`/`PLACE` and clamps a
transfer to what the source holds and the destination admits
(`take = min(n, room)`; a shed-bound PLACE adds up to capacity; an oversized
request is legal-but-clamped, never an error). The batched simulator hardcodes

1. This task makes the simulator execute what the engine executes.

**Files:**

- Modify: `src/kaggriculture/sim/engine.py:130-145` (`step` signature)
- Modify: `src/kaggriculture/sim/units.py` (`apply_unit_phases` signature at
  :216; the PICKUP block around :262-270; the PLACE block around :272-292)
- Modify: every `step(`/`apply_unit_phases(` caller: `src/kaggriculture/sim/rollout.py`
  (`collect_segment`'s step call), `scripts/benchmark_simulator.py`,
  `scripts/sim_codec_campaign.py`, `scripts/sim_rollout_acceptance.py`, and all
  of `tests/sim/` (grep `step(` / `apply_unit_phases(`; most tests pass an
  all-ones tensor built by the new helper below)
- Test: `tests/sim/test_rules_units.py`

**Interfaces:**

- Consumes: nothing new.
- Produces: `step(state: SimState, unit_actions: torch.Tensor,
market_actions: MarketActions, unit_quantities: torch.Tensor) -> SimState`
  where `unit_quantities` is `(batch, 2, MAX_UNITS)` int16, read **only** where
  the op is `PICKUP:*` or `PLACE:*`, in engine semantics (clamped, never
  raising). Same shape validation as `unit_actions`, same error style. Also a
  helper `unit_quantity_ones(batch: int, device) -> torch.Tensor` in
  `sim/engine.py` returning the all-ones tensor, so the many callers that mean
  "single-item transfers, exactly the old behaviour" say so in one word.
- `FEED`/`FERTILIZE`/care amounts stay 1 — those are rules, not transfers; the
  engine has no quantity for them.

- [ ] **Step 1: Write the failing differential test**

Add to `tests/sim/test_rules_units.py`, following that file's existing
reference-vs-simulator pattern (build the same state both ways, apply the same
action, `assert_identical`):

```python
def test_bulk_pickup_and_place_match_the_reference() -> None:
    """A quantity-6 transfer moves six items, clamped exactly as the engine clamps.

    The engine reads ``n = int(action[2])`` and takes ``min(n, available,
    room)``; until this task the simulator silently moved one. Three cases pin
    the semantics: a plain bulk PICKUP, a bulk PICKUP larger than the farmer
    holds room for (clamped, no error), and a bulk PLACE into a shed near
    capacity (partial, no error).
    """
    # Use this file's existing fixture helpers to construct a farm whose shed
    # holds 10 WHEAT and whose farmer stands shed-adjacent. Drive the reference
    # with {"farmer": ["PICKUP", "WHEAT", 6], ...} and the simulator with the
    # PICKUP:WHEAT op index and unit_quantities[..., 0] = 6; assert_identical.
    # Repeat with quantity 99 against 10 available (expect 10 moved), and a
    # PLACE of 6 into a shed with room for 4 (expect 4 placed).
```

Write the three cases fully, using the file's real fixtures (read the file
first; it already builds paired reference/simulator states for exactly this
kind of assertion — copy its idiom, not this sketch).

- [ ] **Step 2: Run to verify it fails** — the simulator moves 1 where the
      reference moves 6: `uv run pytest tests/sim/test_rules_units.py -q` → FAIL.

- [ ] **Step 3: Implement.** Thread `unit_quantities` through
      `step` → `apply_unit_phases` → the per-unit serialized loop. In the PICKUP
      block, the moved amount becomes
      `amount = quantities[:, unit].to(shed.dtype).clamp_min(0)` combined with the
      existing availability/room minimums (the block already computes both; replace
      the implicit 1 with the elementwise minimum). In the PLACE block the same,
      including `_inv_take(state, unit, place_slot, amount, ...)` — widen
      `_inv_take`/`_inv_put`'s `amount` parameter from `int` to `int | torch.Tensor`
      (they already operate on per-batch tensors; a scalar broadcast is the current
      special case). Animal PLACE stays amount-1 (an animal is one animal; the
      engine transfers livestock singly — verify against `_apply_unit_action`
      before assuming, and record what you find in the docstring).

- [ ] **Step 4: Update every caller** with `unit_quantity_ones(...)` and run
      the whole suite: `uv run pytest tests/sim -q` → all pass, including the
      three new cases and every pre-existing differential test (which now assert
      that all-ones quantities reproduce the old behaviour exactly).

- [ ] **Step 5: Commit** (staged paths, `-F` message: why the lane exists —
      78% of top-play transfers are bulk; the simulator's declared domain grows to
      match the engine's).

---

### Task 2: The encoder carries quantities instead of raising on them

**Files:**

- Modify: `src/kaggriculture/sim/rollout.py` (`TurnActions` :160-175;
  `_unit_index` :148-157 and its bulk-transfer guard; `encode_turn`;
  `scripted_actions`; `collect_segment`'s step call gains the quantities tensor)
- Test: `tests/sim/test_rollout.py`

**Interfaces:**

- Consumes: Task 1's `step(..., unit_quantities)`.
- Produces: `TurnActions` gains `quantities: tuple[int, ...]` (length
  `MAX_UNITS`, 1 wherever the op is not a transfer); `_unit_index` returns
  `tuple[int, int]` (op index, quantity) and **no longer raises** on bulk —
  the guard's job is done by faithful encoding; `scripted_actions` returns
  `(units, quantities, markets)`.

- [ ] **Step 1: Invert the guard's tests.** `tests/sim/test_rollout.py`
      currently pins that a bulk `PICKUP` raises (three tests from the guard work,
      plus `test_scripted_actions_raises_on_a_bulk_pickup_from_the_opponent`).
      Rewrite them to pin the opposite, stronger property: `encode_turn` on
      `{"farmer": ["PICKUP", "WHEAT", 6], ...}` yields op `PICKUP:WHEAT` **and**
      quantity 6; a quantity-1 and an arity-2 form yield quantity 1; and the
      `scripted_actions` bridge test extends its segment length past turn 14 on
      seed 223 — the exact case that used to raise — asserting the encoded
      quantity matches what `economic_policy.agent` emitted. Update
      `collect_segment`'s docstring: the bulk-transfer limitation paragraph is now
      history; say the domain grew and when.
- [ ] **Step 2: Run to verify the new tests fail** (encoder still raises).
- [ ] **Step 3: Implement** as specified in Interfaces. Quantity clamp: none
      here — the engine clamps at execution; encode faithfully (int() coercion
      only, mirroring `_parse_order`'s tolerance; non-int quantity encodes the op
      as PASS exactly as malformed orders become code 7 — document the choice).
- [ ] **Step 4:** `uv run pytest tests/sim -q` → green.
- [ ] **Step 5: Commit.**

---

### Task 3: The proof — recorded top episodes replay to exact banks

**Files:**

- Create: `src/kaggriculture/sim/tape.py` — in-simulator tape replay
- Test: `tests/sim/test_tape.py`

**Interfaces:**

- Consumes: `search.route.from_episode`/`load` (720-turn routes);
  `sim.engine.reset/step/unit_quantity_ones`; Task 2's `encode_turn` with
  quantities; `MarketActions`.
- Produces: `replay(seat_zero: Route, seat_one: Route, seeds: Sequence[int],
device) -> torch.Tensor` returning `(len(seeds), 2)` int64 final banks —
  the batched analogue of `search.arena.play`, whole batch advanced together,
  no host synchronisation after the one-time encode. Remember the alignment
  the reference-engine arena established at cost: the action recorded at
  `steps[t]` was chosen from the observation whose `step` reads `t-1`,
  so turn `k` of the step loop executes `route[k + 1]`; episodes seed from
  `episode["info"]["seed"]`.

- [ ] **Step 1: Write the failing proof test**

```python
"""Recorded top episodes must replay through the simulator to their exact banks.

This is the differential proof of the whole quantity lane. A recorded top
episode exercises bulk transfers on 78% of its PICKUP/PLACE actions -- branches
the RL-action-space differential tests structurally never reach -- and the
engine is deterministic given seed and both seats' actions, so equality is
exact or the lane is wrong. Do not weaken to a tolerance; a mismatch is a bug
with a turn number, found by comparing money per turn.
"""
# For the three top-rated 1.32.7 episodes of the newest archive on disk
# (select via kaggriculture.learn.corpus.read_manifest, skipif the corpus
# directory is absent): decode, from_episode both seats, replay([s0], [s1],
# [info.seed]) on CPU, assert banks equal the episode's recorded rewards.
# One extra case: replay all three in ONE batch and assert the same three
# results -- the batch path is the product, not the loop path.
```

- [ ] **Step 2: FAIL** (module absent). **Step 3: Implement** `tape.py` (~60
      lines: encode each route once into op/quantity/market tensors via
      `encode_turn`, broadcast across the batch, 719 `step` calls, return
      `state.money`). **Step 4:** the proof passes — this is the plan's
      acceptance milestone; if it will not pass within budget, STOP per the spec's
      kill gate and report the first divergent turn. **Step 5: Commit.**

---

### Task 3b: The market lane carries what top play actually sells

Added 2026-08-21 after Task 3's proof passed on three episodes and the
controller replayed episodes the implementer had not selected. Those raise:
`_order_quantity` (`sim/rollout.py:257`) rejects any market quantity above 64,
and **85% of 1.32.7 top episodes (128 of 150 scanned) contain at least one** --
real fills of 65-165, clustered at 77-80, plus a rare 1,000,000 "sell
everything" sentinel the engine clamps to stock. The cap is a true execution
limit: `QUANTITY_AXIS = max(QUANTITIES) = 64` (`sim/market.py:39`) is the width
of the tensor axis that resolves a market slot, and requests are clamped onto
it by construction because a host-synchronising assertion cannot run inside a
captured CUDA graph.

This matters beyond replay. Phase 2 clones this corpus and Phase 3 replays it
as in-simulator opponents, so a lane that cannot express a 77-unit sale makes
the dominant selling idiom of top play inexpressible -- the same defect as
`TRANSFER_QUANTITY = 1`, in the other lane.

**Files:**

- Modify: `src/kaggriculture/learn/encoding.py` (`QUANTITIES`)
- Modify: `src/kaggriculture/sim/market.py:39` (`QUANTITY_AXIS`) and the slot
  resolution that sweeps it
- Modify: `src/kaggriculture/sim/rollout.py:257` (`_order_quantity`'s cap and
  its raised message)
- Modify: `tests/sim/test_tape.py` (broaden the corpus selection, below)
- Test: `tests/sim/test_rules_market.py`, `tests/learn/test_mask.py`

**Interfaces:**

- Produces: `QUANTITIES` extended above 64 to cover observed play with headroom,
  and `QUANTITY_AXIS` following it. Choose the new bins from the measured
  distribution, not from taste: 65-80 is where the mass sits, with a tail to
  ~165. State the chosen ceiling and its justification in a comment beside the
  constant. Bins stay the sampling vocabulary AND the execution axis width --
  they are coupled today and this task keeps them coupled; decoupling is a
  larger change nobody has asked for.

- [ ] **Step 1: Measure before choosing.** Scan the newest archive for the
      distribution of market order quantities and of realised fills (a request is
      clamped by stock and shed room, so the fill is what the axis must span).
      Record the numbers in your report; they justify the ceiling you pick.
- [ ] **Step 2: Benchmark the cost first, so the choice is informed.** Run
      `scripts/benchmark_simulator.py` at the current axis width and record
      throughput. The market phase sweeps the axis, so widening it costs roughly
      linearly there.
- [ ] **Step 3: Write the failing test.** In `tests/sim/test_rules_market.py`,
      a differential case: a `SELL WHEAT 77` against stock that can fill it must
      match the reference exactly. It fails today by raising.
- [ ] **Step 4: Widen** `QUANTITIES`, confirm `QUANTITY_AXIS` follows, and
      relax `_order_quantity`'s cap to the new ceiling -- keeping the raise for
      anything beyond it, because a silent clamp would diverge from the reference
      and that is exactly what the guard exists to prevent.
- [ ] **Step 5: Re-benchmark** and report throughput before and after. If the
      simulator slows by more than half, say so plainly in the report rather than
      absorbing it silently -- Phase 3 trains against this.
- [ ] **Step 6: Broaden the proof.** Change `tests/sim/test_tape.py` to select
      its episodes without needing them to dodge the cap, and raise the count from
      3 to at least 8. If any episode still cannot replay, the test must fail
      loudly naming it -- never skip it silently, and never select only episodes
      that happen to pass.
- [ ] **Step 7: Full suite, then commit.**

---

### Task 4: The policy grows a quantity head

**Files:**

- Modify: `src/kaggriculture/learn/model.py` (`Policy.__init__` :88-101,
  `forward` :103-160)
- Modify (unpack-site updates only): `src/kaggriculture/sim/rollout.py:277`,
  `src/kaggriculture/learn/rollout.py`, `src/kaggriculture/learn/play.py`,
  `src/kaggriculture/learn/scripts/toad_phase1.py` (`_step` :1012-1050,
  teacher forward :1040-1049), `src/kaggriculture/learn/scripts/critic_ev.py`,
  `src/kaggriculture/learn/scripts/evaluate.py`, plus any other
  `= policy(`/`= learner(` unpack (grep `unit_logits, market_logits, values`).
- Test: `tests/learn/test_model.py`

**Interfaces:**

- Produces: `forward` returns
  `(unit_logits, unit_quantity_logits, market_logits, values)` where
  `unit_quantity_logits` is `(batch, MAX_UNITS, len(QUANTITIES))`, read from
  the same per-unit gathered column as the op head (a transfer's size depends
  on what the unit stands next to and holds — the op head's own argument) via
  its own `Linear(channels, len(QUANTITIES))`.
- Produces: `load_policy_weights(policy: Policy, weights: Mapping) -> list[str]`
  in `model.py`: loads with `strict=False`, returns the missing keys, and
  **raises** unless every missing key belongs to the quantity head — old
  checkpoints (the BC clone, every arm's) predate the head and must keep
  loading, but a checkpoint missing anything else is corrupt and must not
  half-load silently. All existing `load_state_dict(...)` calls on `Policy`
  (`toad_phase1._warm_start`/`_restore`/`_teacher`, `critic_ev.load`,
  `evaluate`, `play`) go through it.

- [ ] **Step 1: Failing tests** — forward-shape test for the 4-tuple and its
      quantity shape/dtype; `load_policy_weights` accepts a pre-lane checkpoint
      state dict (build one in-test by deleting the quantity keys from a fresh
      `state_dict()`) and raises on one missing a value-head key. **Step 2: FAIL.**
- [ ] **Step 3: Implement**, then update every unpack site (mechanical:
      `unit_logits, unit_quantity_logits, market_logits, values = ...`; sites that
      ignore quantities discard the new element explicitly with `_`); the teacher
      KL in `toad_phase1._step` computes over op+market heads only for now (Task 5
      decides quantity-KL when both sides have the head — leave a one-line comment
      saying so). **Step 4:** full `tests/learn` + `tests/sim` green.
      **Step 5: Commit.**

---

### Task 5: The learning stack threads the quantity through

The wide-but-shallow task: sampling, recording, masking, log-probs, entropy,
decoding. Every touch is listed; nothing else moves.

**Files:**

- Modify: `src/kaggriculture/learn/encoding.py` — `_op` :861-874 gains a
  `quantity: int` parameter and emits it for `PICKUP`/`PLACE` (arity-3);
  `decode_actions` (the caller that turns sampled indices into engine actions)
  passes the sampled bin's value; **delete `TRANSFER_QUANTITY` (:660)** and
  rewrite the lossy-on-purpose paragraph (:790-800) to say the loss is gone
  and why it existed.
- Modify: `src/kaggriculture/learn/mask.py` — new
  `unit_quantity_mask(...) -> (..., MAX_UNITS, len(QUANTITIES))` bool: for a
  live unit, bins with values 1..12 are legal (the engine clamps oversize, so
  legality here is "sane", not "exact"; document that the engine is the
  authority and the mask exists to keep sampling in the useful range); for a
  padded slot, only the value-1 bin (a well-defined distribution the log-prob
  can ignore, the op-mask idiom at mask.py:40-46).
- Modify: `src/kaggriculture/learn/ppo.py` — `joint_log_prob` (:657) gains
  `unit_quantities` logits+actions and adds the per-unit quantity log-prob
  **only where the sampled op is a transfer** (a quantity that was never
  executed must not enter the ratio; compute the transfer mask from the sampled
  op indices against the `PICKUP:`/`PLACE:` range of `UNIT_OPS`); `entropy_of`
  is reused as-is on the quantity head.
- Modify: `src/kaggriculture/learn/rollout.py` — sample the quantity head
  jointly, record `unit_quantities` `(turns, MAX_UNITS)` int64 and
  `unit_quantity_masks` on `Trajectory` (:187 docstring included), decode
  through the new `_op`.
- Modify: `src/kaggriculture/learn/scripts/toad_phase1.py` — `ACTED_FIELDS`
  (:121-132) gains both new fields; `_step` scores the quantity head into the
  learner log-probs and entropy (same masking rule as ppo); teacher KL: if the
  teacher state dict has the quantity head, include it, else op+market only
  (implement the check, not a comment).
- Modify: `src/kaggriculture/learn/play.py` + `evaluate.py` — decode path only.
- Modify: `src/kaggriculture/sim/rollout.py` — `collect_segment`'s sampling
  body: sample the quantity head beside the op head (same `_sample` idiom,
  masked by a sim-side quantity mask that mirrors `learn/mask.py`'s rule),
  thread the sampled `(batch, 2, MAX_UNITS)` quantities into `step`, and record
  them on the sim `Trajectory`. This is substantive, not an unpack tweak: it is
  what lets self-play RL _use_ the lane at simulator speed, and it must stay
  graph-capturable — no host synchronisation in the added sampling.
- Test: `tests/learn/test_encoding.py`, `tests/learn/test_mask.py`,
  `tests/learn/test_ppo.py`, `tests/learn/test_rollout.py`,
  `tests/learn/test_toad_runner.py`.

**Interfaces:** consumes Task 4's 4-tuple and `QUANTITIES`; produces the full
loop: a policy rollout on the reference engine can now emit
`["PICKUP", "WHEAT", 6]`.

- [ ] **Step 1: Failing tests, one per property, fixtures able to violate them**
      (this repo's seven insensitive-test incidents all came from fixtures too
      simple to express the bug): decode emits the sampled quantity for a transfer
      and arity-1/2 forms for non-transfers; the quantity log-prob term is ZERO for
      a turn whose sampled ops contain no transfer and nonzero otherwise (fixture
      must contain both kinds of unit in ONE turn); the mask's padded-slot row;
      round-trip — a recorded `Trajectory` from a 3-turn reference-engine rollout
      carries quantities that reproduce through decode; and sim-side, a short
      `collect_segment` self-play run steps without error and records quantities
      whose transfer-op rows are in 1..12 (the graph-capture property itself is
      covered by the existing capture tests once the body compiles).
- [ ] **Step 2: FAIL. Step 3: implement the list above. Step 4:** full suite
      green (`uv run pytest -q`, ~700 tests). Run the established mutation checks:
      drop the transfer-only condition in `joint_log_prob` → the zero-term test
      fails; revert. **Step 5: Commit.**

---

### Task 6: The gate examines the frontier

Independent of Tasks 1–5; the 2026-08-20 lesson (a 73% pass converged 130
rating points _below_ what it replaced, because the exam had no frontier).

**Files:**

- Modify: `src/kaggriculture/search/scripts/holdout.py`
- Modify: `src/kaggriculture/search/scripts/build_league.py` (only if a flag is
  needed to emit the frontier dir; prefer reuse)
- Test: `tests/search/test_holdout.py`

**Interfaces:**

- Produces: `run(candidate, against=SERVED, frontier: Path = FRONTIER_DIR,
workers=None) -> Verdict` where `FRONTIER_DIR =
Path("/data/kaggriculture/search/frontier-league")` holds routes harvested
  from the newest archive (refreshed by the operator with `build_league`; the
  gate **raises** if the dir is empty or its newest file is older than 7 days —
  a stale frontier is the 2026-08-20 failure with extra steps, so it fails
  loudly rather than examining against history). `Verdict` gains
  `frontier_rate`, `frontier_low`, and the PASS rule becomes, pre-registered:
  **lower > 0.5 vs served AND pooled Wilson lower bound ≥ 0.45 over the
  frontier pool** (≥ 3 tapes × 128 games; at 384 games that requires a pooled
  rate ≈ 0.50 — "statistically not worse than the frontier". The searched
  route's 0.36–0.38 fails it; a true peer passes). Both thresholds are module
  constants with the derivation in comments, fixed before any candidate is
  seen.

- [ ] Steps: failing tests first — the two-condition verdict (a candidate that
      beats served but scores 0.37 pooled-frontier FAILS; both-pass PASSES;
      monkeypatched outcomes, recorder-style, per this suite's existing idiom);
      the staleness raise (tmp dir with an old mtime); `GATE_SEEDS` discipline
      unchanged (existing structural tests keep passing). Then implement, full
      suite, mutation check (invert the frontier condition → the 0.37 test fails),
      commit.

---

## After the plan

Phase 1 ends when Task 3's proof and the full suite are green and Task 6's
gate is live. Then Phase 2 (BC on the fresh corpus) gets its own plan — its
targets are expressible only because of this one. Arm T's curves keep
accumulating in the background throughout; nothing here touches its run.
