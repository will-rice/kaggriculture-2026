# Batched Kaggriculture simulator — design

**Status:** approved design, 2026-08-10

**Runtime:** eager PyTorch, CPU and CUDA

**Ground truth:** `kaggle-environments==1.32.6`

## 1. Goal

Build a faithful, batched implementation of the Kaggriculture engine whose
ordinary eager-PyTorch runtime is fast on a GPU. Speed must come from algorithms
that operate on whole batches, not from translating the Python interpreter into
thousands of tiny tensor operations and asking a compiler or CUDA graph to hide
the result.

The simulator is accepted only when both of these are true:

1. it matches the installed Kaggle engine exactly after every turn; and
2. the complete eager GPU hot path reaches at least 110,000 environment-steps
   per second at batch size 1,024 on the project hardware.

The installed Kaggle source and configuration are the only fidelity oracle. If
this document, the existing simulator, a test fixture, or competition prose
disagrees with that source, the source wins.

## 2. Design principles

### 2.1 Fidelity before performance

Every optimized phase is compared directly with Kaggle before its speed is
measured. An unexplained divergence blocks the build. Statistical similarity,
episode-end equality, or agreement with the previous tensor simulator is not a
substitute for per-turn reference equality.

### 2.2 Vectorize avoidable work

The batch and player axes are always tensorized. Runtime Python loops may not
scan environments, players, products, crops, animals, shops, action types, or
requested quantities.

Two fixed scans remain because they are part of the game semantics:

- twenty unit slots, because later units observe earlier unit mutations;
- ten market-order slots, because an earlier order can fund or otherwise alter
  a later order.

Everything inside those scans operates over the complete batch and both seats
with lookup tensors, masks, gathers, scatters, prefix operations, and reductions.

### 2.3 Eager PyTorch is the product

The required implementation is ordinary eager PyTorch. `torch.compile` and
CUDA graphs may be benchmarked later as optional accelerators, but they cannot
be required for correctness or used to rescue a failing eager throughput result.

### 2.4 The engine is not differentiable

Environment transitions run under `torch.inference_mode()`. The engine may use
in-place masked and indexed updates, preallocated scratch buffers, and reusable
outputs. It must not construct an autograd graph. Policy gradients stop at
sampled actions; observations and rewards are rollout data.

### 2.5 Fixed supported domain, stated honestly

The engine supports the competition configuration and the states reachable
from its reset through the project's legal action interface. It raises on a
different configuration or an input outside that domain.

Any fixed bound, including the unit-slot bound, must be justified against that
reachable domain. A convenient bound is not called exact until the proof or an
exhaustive reference search establishes it.

## 3. Reference contract

The reference version is pinned by package version plus hashes of:

- `kaggriculture.py`;
- `kaggriculture.json`.

Rules tables and configuration defaults are imported or parsed from the
installed reference. They are not copied into simulator source. Index maps are
derived from those tables and checked at import.

The reference audit records at least:

- phase order and every action branch;
- board, farm, private, market, town, clock, and terminal mutations;
- action parsing and malformed-action behavior in the supported input domain;
- randomness consumption and reseeding;
- insertion-order behavior for carried inventory;
- market quote and commit ordering;
- observation asymmetries between seats.

CI fails if either reference hash or the parsed branch manifest changes without
a new audited acceptance record.

## 4. Runtime state

### 4.1 Storage layout

The runtime uses a small number of dense tensors grouped by dtype and access
pattern rather than one allocation per named field:

- tile categorical fields;
- tile short-integer counters;
- tile long-integer schedules;
- tile boolean flags;
- farm and unit fields;
- shed, seeds, and carried inventory;
- market and town fields;
- clock, status, reward, seed, and RNG state.

The leading dimension is always batch. Per-player groups carry a player
dimension of two. Board coordinates are stored consistently as `[y, x]`; unit
positions retain explicit `x` and `y` views at the API boundary.

Grouping reduces allocations and permits bulk updates. It does not use lossy
bit packing or mix values whose dtypes or invariants differ.

### 4.2 Named canonical views

The fidelity codec exposes named views for every reference field. Tests compare
these views, not opaque packed storage. A failure reports the field name and
index even when several fields share a physical tensor.

Tile attributes are canonical outside their applicable kind: zero except for
documented negative sentinels. Canonicalization occurs on both the reference
and tensor sides before comparison and may not conceal a live reference value.

### 4.3 Mutation model

`step` mutates a state object in place. The rollout collector owns state lifetime
and may swap between two preallocated states only where an operation cannot be
expressed safely in place. Phase-level full-state clones are forbidden.

Scratch tensors are allocated by a workspace object keyed by batch size and
device, then reused. Timed execution contains no allocator-dependent setup.

### 4.4 Required special state

Carried inventory retains both counts and insertion order. `inv_seq` records the
order in which an absent item became present, and `inv_tick` advances on each
such transition. DROP and end-of-day transfer sort present entries by this
sequence so a full shed matches Python dictionary insertion order exactly.

Money, inventory, prices, prefix sums, and cumulative costs use exact integer
tensors with explicit overflow proofs. Rewards convert to floating point only
at the rollout boundary.

## 5. Device rule tables

`sim/tables.py` owns immutable rules-derived tables and caches tensor forms by
device. Device materialization occurs before stepping or benchmarking.

Tables include:

- unit operation class, item, crop, animal, direction, and movement delta;
- crop and animal structures, products, costs, schedules, and yield limits;
- market order class and item mapping;
- hire and land costs;
- shed, product, crop, animal, and shop index maps;
- shop drain matrices;
- observation feature maps and normalization constants;
- market price parameters and bounded quantity axes.

Static Python loops are allowed while constructing these tables. The hot path
may only gather from them. Each table has a construction test against the
installed Kaggle rules and a device-residency test.

## 6. Step algorithm

The public operation is:

```python
step(state, unit_actions, market_actions, workspace) -> None
```

It executes the Kaggle phases in reference order and mutates `state`.

### 6.1 Atomic PLANT guard

Before any unit acts, scatter requested PLANT counts by crop for each farm. If
demand for a crop exceeds available seeds, replace every request for that crop
with PASS. This preserves the reference's all-or-nothing behavior.

### 6.2 Ordered unit resolution

Loop over the fixed unit slots once. For the current slot, gather operation
metadata into `(B, P)` tensors and apply masked branches across the whole batch.

There are no nested runtime scans over directions, items, crops, or animals.
Movement uses gathered deltas; pickup and placement use gathered item indices;
planting and harvesting use gathered rule properties. Tile and inventory changes
use indexed updates.

All reference ordering remains explicit, including:

- shed-access operations before the locked-tile guard;
- farmer before hands;
- insertion-ordered DROP under shed capacity;
- crop and animal maturity, fertilizer, feeding, care, and destruction rules;
- actions from dead or absent unit slots becoming no-ops.

### 6.3 Ordered market slots

Loop over the ten market-order slots once. Each slot is resolved across the
whole batch without a Python quantity loop.

Create a tensor quantity axis `q = 0..63`. For each active order pair, calculate
the reference quote and commit outcome at every candidate unit-round in parallel,
then use prefix-valid masks, cumulative costs, and indexed reductions to find
the exact successful prefix and final mutations.

The pair resolver has explicit cases:

1. one active order;
2. two orders against different products or non-market items;
3. same-product BUY_PRODUCT/BUY_PRODUCT;
4. same-product SELL/SELL;
5. same-product BUY_PRODUCT/SELL;
6. same-product SELL/BUY_PRODUCT.

Both seats receive quotes from the same pre-commit inventory in each round.
Seat-zero commit may mutate inventory before seat one commits, but it does not
change seat one's already-computed quote. If one order fails, it dies at that
round and the survivor continues from the resulting state. The vectorized
resolver must reproduce this transition and its one-seat tail, not merely match
the final total quantity.

Fixed-price seed and animal purchases use affordability and capacity prefixes.
HIRE and BUY_LAND are one masked atomic operation per slot. Prices refresh after
each order slot exactly as in Kaggle.

Sales at price one do not add market inventory. Product and animal purchases
obey total shed capacity. Malformed decoded operations are dead orders in the
same cases as the supported reference parser.

### 6.4 Town, decay, day boundary, and clock

Town consumption, price refresh, crop decay, plant refresh, animal refresh,
weed spawning, inventory drop, crew reset, shop unlock, and clock advancement
remain distinct ordered subphases. Each is a bulk tensor operation with no
per-kind runtime loop.

Randomness uses the exact reference MT19937 word stream and cursor semantics.
Any precomputation must be bit-identical to Python's `random.Random`; statistical
equivalence is insufficient.

## 7. Observation, legality, and decode

`observe`, `legal`, and `decode` operate directly on device tensors.

- Observation planes use gathered kind metadata and bulk scatter operations.
- Scalar observations use device normalization tables.
- Unit and market legality use operation metadata tables and broadcast masks.
- Bucket decoding produces fixed-shape unit actions and expanded market orders
  without Python loops over action types or batch entries.

The hot path may not call `.cpu()`, `.numpy()`, or `.item()`, branch on tensor
data in Python, build a CPU tensor, or transfer an index implicitly.

## 8. Fidelity verification

### 8.1 Direct per-turn differential driver

For identical seeds and decoded actions, run Kaggle and the tensor engine in
lockstep. After every decision compare:

- every canonical state field;
- both seats' board, scalar, and position observations;
- both seats' unit and market legality masks;
- the decoded action structures sent to Kaggle;
- clock, done status, and terminal rewards.

Stop at the first difference and report seed, turn, phase, field, tensor index,
expected value, and actual value.

### 8.2 Phase replacement rule

Rewrite one phase at a time. During replacement compare:

1. Kaggle against the candidate phase path;
2. Kaggle against the retained implementation;
3. candidate against retained implementation.

Only the first comparison certifies correctness. The other two localize faults
and detect pre-existing simulator errors.

### 8.3 Market proof harness

Before the vectorized resolver replaces the sequential reference-shaped path,
compare it exhaustively with `_process_market` over:

- every supported order and item type;
- quantities zero through 64;
- each order-slot position;
- each price breakpoint and the price floor;
- money at every affordability boundary;
- shed count at every relevant capacity boundary;
- empty and partially stocked sell inventories;
- both seats alone and together;
- same-item and different-item order pairs;
- buy/buy, sell/sell, buy/sell, and sell/buy ordering;
- one order failing before the other;
- an earlier sale funding a later purchase.

Dimensions may be factorized into focused exhaustive tests, but every listed
interaction and boundary must be covered. Random campaigns supplement this
proof rather than replacing it.

### 8.4 Per-rule and mutation tests

Each reference branch has a focused test with a named expected mutation. The
suite must demonstrably fail when that rule is deliberately broken and return
green when restored.

Required market mutations include:

- quoting seat one after seat zero's commit instead of from pre-commit state;
- an off-by-one successful prefix;
- counting price-floor sales as supply;
- continuing after the first failed unit;
- applying capacity per item instead of to the whole shed;
- resolving order slots in parallel.

### 8.5 Campaign and evidence record

The final campaign runs at least 10,000 complete episodes across adversarial,
random legal, boundary-biased, and corpus-replayed actions. It records zero
unexplained divergences, branch counts, seeds, package version, reference hashes,
device, dtype configuration, test command, and source commit.

No throughput claim is accepted before this record is green.

## 9. Performance verification

### 9.1 Headline benchmark

Measure plain eager PyTorch on one GPU under `torch.inference_mode()`. A timed
iteration includes:

- action decoding;
- both-seat observations;
- both-seat legality masks;
- the complete environment step.

Policy inference is excluded. Table construction, reset, and warmup are reported
separately and occur outside the timed region.

Measure batch sizes 64, 128, 256, and 1,024. Record median and tail latency,
environment-steps per second, trajectories per hour, peak allocated and reserved
memory, GPU model, and PyTorch version.

The acceptance target is at least **110,000 environment-steps per second at
batch 1,024**. Smaller batches have no predeclared threshold but are mandatory
because multiple concurrent experiment groups are an intended workload.

### 9.2 Stage profiling

Record an operator summary or profiler trace after state storage, unit resolution,
market resolution, observation/legality, and full-step integration. Source code
with few loops is not sufficient evidence: traces must show that per-kind and
per-quantity work collapsed into bulk operations.

If fidelity passes but eager throughput misses the target, profile and redesign
the dominant algorithm. Do not relax fidelity or make compilation mandatory.

Optional `torch.compile` and CUDA graph results may be reported only as separate
rows after eager acceptance.

## 10. Failure policy

- Any unexplained Kaggle divergence blocks acceptance.
- A fast approximation is rejected.
- A state bound without a reachable-domain proof blocks the relevant phase.
- Integer overflow or an out-of-domain index fails before rollout; clamping is
  forbidden.
- If a vectorized market case cannot be proven exact, retain the exact resolver
  for that case, record its frequency and cost, and continue investigating. The
  overall eager throughput target remains unchanged.
- A reference hash change invalidates the acceptance record and requires a new
  audit and campaign.

## 11. Deliverables

1. Reference-derived device rule tables.
2. A compact, named-view runtime state with reusable workspace buffers.
3. In-place, table-driven unit resolution with only the semantic unit scan.
4. Exact vectorized market resolution with only the semantic order-slot scan.
5. Vectorized observation, legality, and decoding.
6. Direct Kaggle differential, per-rule, mutation, and market-proof tests.
7. A zero-divergence 10,000-episode evidence record.
8. The eager GPU benchmark matrix and stage profiler evidence.

CUDA compatibility alone is not a deliverable. The simulator is complete only
when it is both reference-faithful and fast in eager PyTorch.
