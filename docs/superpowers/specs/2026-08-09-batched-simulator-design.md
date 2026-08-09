# Batched simulator — design

**Status: in progress.** Sections 1–3 are approved. The equivalence suite,
integration, task breakdown and risks are still being drafted; this file is
committed early because an hour of design decisions living only in a
conversation is how this project nearly lost its experiment ledger.

## Why

RL throughput is ~550–1,100 environment-steps/second, bound by the reference
Python engine and our observation encoders on CPU. The game is stateful and
strictly sequential per environment — turn _t+1_ cannot begin until _t_
resolves — so no amount of GPU helps an interpreter walk one board at a time.

Frog Parade (Lux S3 gold, single workstation) needed a custom simulator at
**110,000 steps/second** to train at the scale that won. At our rate, a 1e8-step
run is ~50 hours and Frog-Parade-scale training is a week of continuous compute.

The target is not "a faster engine" but a **batched** one: represent _N_
environments as tensors and step all of them at once on the same device as the
policy, so 24 sequential environments becomes 1,024 parallel ones. Target
10⁴–10⁵ steps/second.

**Secondary value, and the reason this is worth building even if the
competition goes badly:** ten RL arms have died leaving open questions nobody
could afford to settle, because each re-run costs ten hours. At 10⁴–10⁵
steps/second those experiments cost minutes.

## The central risk

Fidelity. This project trained five arms on a deleted economy after the ladder
moved to `kaggle-environments` 1.32.6 on 2026-08-07 and we stayed pinned at
1.32.3. That was caught only because a version number existed to compare.
**A reimplementation has no version number.** A batched engine that differs
subtly from the reference is worse than no batched engine, because every result
afterwards silently rests on it.

Hence: the primary deliverable is **a committed pytest suite proving the
batched engine's outputs are identical to the reference engine's**, and the
implementation is the thing that has to satisfy it.

## Decisions

| decision       | choice                                             | rationale                                                                         |
| -------------- | -------------------------------------------------- | --------------------------------------------------------------------------------- |
| Location       | `src/kaggriculture/sim/`, standalone               | imports nothing from `learn/`; extractable as its own package                     |
| Interface      | tensors in, tensors out                            | no dict round-trip in the training loop, which is the cost being removed          |
| Config support | every knob in the schema; **one config per batch** | full parity without ragged tensor shapes; a different config is a different batch |
| Execution      | pure torch, batched, on GPU                        | one language, one debugger, and it composes with the policy for free              |

## Section 1 — Module boundary and public interface

`sim/` knows the game and nothing else. It imports torch and reads constants
from the reference engine's schema. It imports nothing from `learn/` — no
reward, no policy, no training concepts.

```python
Config      # one config per batch, from kaggriculture.json defaults + overrides
State       # batched tensors, frozen dataclass, batch dimension first
reset(config, seeds) -> State
step(state, unit_actions, market_actions) -> State
legal(state) -> (unit_mask, market_mask)
observe(state, seat) -> ObsTensors
```

Actions arrive as **integer index tensors**: `unit_actions` is
`(batch, MAX_UNITS)` of op indices, `market_actions` is
`(batch, slots, quantity)`. Same vocabulary `encoding.py` already uses, but as
the native interface rather than something decoded into strings.

`observe` returns the game's **facts** per seat — board contents, money,
inventory, shed, market book, town state — as tensors. Not our 48-plane feature
encoding: which features a model wants is a modelling choice and stays in
`learn/` as a tensor→tensor transform.

`legal` lives here because "which actions the engine accepts" is a property of
the rules, and the same knowledge `step` needs to apply them. `mask.py`'s logic
is reimplemented in tensor form here; `mask.py` itself is untouched and becomes
a test oracle — a good role for it, since it is already asserted directly
against the engine at 0 over-permissive and 0 over-strict across 106,216 unit
pairs and 36,288 market cells.

`step` returns a new `State` rather than mutating, so a batch can be
snapshotted, forked or replayed for differential testing without defensive
copying. The allocation is cheap on GPU and worth it for testability.

**Permanently on the reference engine:** gate evaluation, submission
validation, and anything producing a number acted on externally. The batched
engine is for training throughput; the reference stays the authority for
anything we would stake a submission on.

## Section 2 — State layout

A tile is `None` (owned, empty), the string `"LOCKED"`, or a dict whose `kind`
is `PLANT`, `WEED`, `COOP` or `PASTURE`, carrying kind-specific attributes.

That becomes a **struct-of-arrays**, one plane per attribute, each
`(batch, 2, H, W)` for the two farms:

- `tile_kind` — `int8` enum: `EMPTY / LOCKED / PLANT / WEED / COOP / PASTURE`
- `tile_species` — `int8`, crop index for plants, animal index for structures,
  `-1` otherwise; one plane, because a tile is never both
- `tile_planted_day`, `tile_yield_units`, `tile_consecutive_unwatered`,
  `tile_max_lifespan_step`, `tile_fertilized_until_day`,
  `tile_consecutive_unfed`, `tile_pending_care_bonus` — `int16`/`int32`
- `tile_flags` — `uint8` bitfield: `watered_today`, `fed_today`, `cared_today`,
  `fertilizer_available`

Attributes that apply to only one kind hold a sentinel elsewhere. That wastes a
little memory and buys branchless updates: write every plane every turn and let
the kind mask decide which writes count.

The rest, batch-first:

- `money` `(batch, 2)`, `hires_today` `(batch, 2)`
- `unit_pos` `(batch, 2, MAX_UNITS, 2)`, `unit_count` `(batch, 2)`,
  `unit_carry` `(batch, 2, MAX_UNITS, n_items)`
- `shed` `(batch, 2, n_items)`
- `market_inventory` `(batch, n_items)`
- `town_shops` `(batch, MAX_SHOP_INSTANCES)` of shop indices, `-1` unfilled
- `step` `(batch,)`, `rng_state` for shop unlocks and spawns

Two choices worth naming as decisions rather than details:

- **`unit_carry` is a dense count tensor**, not the reference's per-unit dict.
  Twelve integers per unit is trivial and removes all ragged handling.
- **Prices are derived, not stored.** The reference recomputes them from
  inventory via `market_price`; storing them would create a second source of
  truth that can drift.

## Section 3 — Step decomposition

The turn is five phases, in order:

1. **PLANT atomicity.** Before any action applies, the interpreter counts PLANT
   requests per crop and drops **all** requests for a crop whose total exceeds
   seeds held. A per-environment reduction, then a mask.
2. **Unit actions**, farmer then hands, in index order.
3. **Market processing.**
4. **Town consumption**, then price refresh.
5. **End of day** on day boundaries, including the hand-wipe that clears every
   inventory.

Phases 1, 2, 4 and 5 are ordinary batched tensor work. Phase 3 is the hard one.

The reference processes up to `maxMarketOrdersPerTurn` (10) order slots. Within
each slot both players' orders resolve in a **per-unit lockstep**: quote both
players' price for their current unit, commit both, repeat until exhausted. It
is sequential because `market_price` is a function of inventory, so each unit
sold moves the price for the next.

Naively that is 10 slots × up to ~100 units ≈ 1,000 sequential iterations per
turn, over 719 turns. As tiny GPU kernels, launch overhead would consume the
entire win.

**But the inner loop has closed form.** Selling _k_ units from inventory _I_
yields `Σ price(I+j)` for _j_ in 0..*k*−1 — a prefix sum over consecutive
inventory levels, not an iteration. The two-player lockstep interleaves them, so
one player takes levels _I, I+2, I+4…_ and the other _I+1, I+3…_: strided prefix
sums, still one operation. The genuinely data-dependent part is a buy failing on
insufficient funds, which becomes "largest _k_ whose cumulative cost fits" —
a `searchsorted` over the cumulative-cost vector.

That collapses phase 3 to **one bounded loop of 10 iterations** — the order
slots really are sequential, since a sale in slot 1 funds a purchase in slot 2 —
with each slot resolved in closed form across the whole batch.

**Risk, stated plainly:** the closed form is read from the price function and
the lockstep, but not yet proved on every path — fertilizer, `BUY_ANIMAL`, land
purchases and hires each have their own branch. The plan's **first task is a
proof of equivalence for market resolution alone**, before any other phase is
written. If the closed form does not hold, the fallback is the bounded 10×100
loop: slower, still correct, still batched across environments. Better to
discover that in task one than in week two.
