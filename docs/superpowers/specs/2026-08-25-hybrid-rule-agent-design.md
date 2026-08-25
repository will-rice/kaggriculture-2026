# Hybrid rule agent: search a guarded strategy against the public frontier

**Design, 2026-08-25.** Target engine: `kaggle-environments` 1.32.7. This
design adds a submission-grade hybrid rule agent without interrupting the
single-GPU Toad run. It complements rather than replaces the RL curriculum and
supersedes the assumption in the earlier route-search design that vendored
Boatlee v14 is the public frontier.

## 1. Why build this

The current phase-one RL run is mechanically healthy but has not discovered the
long economic chain required to bank money. Its recent policy still spends the
opening 3,000 coins, ends near zero bank, and loses every diagnostic game to the
economic policy. More unguided exploration is therefore not the only path worth
funding.

The public field demonstrates a different viable strategy class: strong
openings or routes with small reactive corrections. A pure action tape is
efficient but brittle; a completely reactive policy is robust but has not
matched the strongest public openings. The new agent combines a compact opening
program with state-aware recovery, job assignment, market decisions, and final
liquidation.

Boatlee v14 is only the repository's served baseline. It is not treated as the
strongest public agent. Kaggle's scored history includes
[Kaito v27](https://www.kaggle.com/code/kaitofukami/25-27-strict-future-v27-midgame-meta-reset)
at 3090.1, the v45 version in
[Kaito's later notebook](https://www.kaggle.com/code/kaitofukami/40-40-early-floor-39-46-top-10-v48-fast-routes)
at 3009.0, and public successors such as Kaito v43/v48 and Boatlee v20/v21.
Ratings are path- and time-dependent, so the design establishes the frontier by
local cross-play instead of trusting a notebook title or one leaderboard
number.

## 2. Goals and non-goals

### Goals

1. Produce a deterministic, standalone submission candidate that beats the
   currently served Boatlee v14 and the strongest locally measured public agent
   on untouched, seat-swapped games.
2. Optimize against a multi-generation league rather than one opponent.
3. Produce the exact observation features, legality masks, and decoded actions
   for both RL and the hybrid agent from one submission-safe canonical core.
4. Combine a strong searched opening with reactive execution, recovery, market
   handling, and liquidation.
5. Search offline on CPU while the active GPU training run continues
   undisturbed.
6. Leave a reproducible record of every public opponent, configuration,
   matchup, and promotion decision.

### Non-goals

- No inference-time rollout search, MCTS, Torch dependency, or GPU requirement.
- No second observation parser for the rule agent.
- No optimization against bank alone or against only the latest public agent.
- No automatic Kaggle submission or replacement of `main.py`; promotion remains
  a separately reviewed action.
- No attempt to merge the rule controller and neural policy into an ensemble at
  inference time.

## 3. Establish the public frontier first

Frontier discovery is an explicit prerequisite, not an informal collection of
filenames.

### 3.1 Acquire exact versions

Archive exact scored versions of promising public notebooks, including at
least:

- Kaito v27 and the v45/v48 lineage;
- Boatlee v14, v16, v20, and v21;
- strong recent public-state, multi-route, and adaptive agents;
- the repository's economic policy and current searched-route incumbent.

For every artifact, record its Kaggle URL, notebook version, source hash,
claimed score, engine assumptions, and extraction method. Deduplicate identical
source hashes without erasing their provenance. A notebook's latest version is
not assumed to be its strongest version.

### 3.2 Measure rather than assume

Run an initial seat-swapped round robin on the current reference engine with 64
common seeds, producing 128 games per pair. Report win points,
wins/draws/losses, paired margin, failures, and runtime per exact version. A
public artifact that cannot execute under the current engine remains documented
but cannot become the measured frontier.

The `public_frontier` is the strongest agent under the fixed local evaluation
protocol, not the agent with the largest historical leaderboard number. Keep a
small diverse frontier league rather than only its winner because the live
field contains older meta generations and can be non-transitive.

## 4. One canonical feature and action boundary

The hybrid agent and RL must not independently interpret raw observations. The
current `learn.encoding` functions allocate Torch tensors, while the submission
must not import Torch, so simply calling those functions from the hybrid agent
is not a valid design.

Extract their feature calculations into a submission-safe canonical producer
outside `kaggriculture.learn`. It parses the raw observation once and returns an
immutable `EncodedObservation` whose storage uses only submission-safe Python
types. It contains:

- board planes from `encode_board`;
- normalized scalar features from `encode_scalars`;
- unit positions from `encode_positions`;
- unit-operation, quantity, and market legality masks;
- stable names or typed accessors for every feature used by rule logic.

Both the reference RL collector and hybrid controller call this producer. A
thin Torch adapter converts the already-computed values and masks to the exact
tensors expected by the neural policy; it performs no feature calculation of
its own. The existing `encode_board`, `encode_scalars`, `encode_positions`, and
mask functions become compatibility wrappers around the same producer. The
hybrid controller applies thresholds to those canonical normalized values; it
does not reach around the bundle to parse farm, market, opponent, or private
fields again. Named accessors prevent the controller from depending on magic
tensor offsets.

The tensor-native CUDA simulator cannot call a Python observation parser from
inside its device loop. It remains a vectorized backend implementation, but its
channel schema, constants, and normalization definitions come from the same
canonical feature specification, and the existing fixed-tape differential gate
must compare every native feature and mask with `EncodedObservation`. It is the
only permitted duplicate execution backend.

The action boundary follows the same structure. A submission-safe canonical
decoder owns engine action names, arity, and mask interpretation. The hybrid
controller uses it directly; the existing tensor decoders become adapters that
select categorical indices and delegate to the same core. The controller does
not contain a second spelling of engine actions or arity rules.

Refactoring the RL call sites is allowed only with corpus evidence that the
Torch adapter is bit-identical to today's separate board, scalar, position, and
mask outputs. Tensor shapes, ordering, normalization, and checkpoint
compatibility must not change. The pure-Python producer and decoder must also
pass the real packaged-agent runtime gate, proving that the shared boundary did
not smuggle Torch or another excluded dependency into the submission.

## 5. Hybrid controller

The runtime path is:

```text
raw observation
    -> canonical encoded observation
    -> opening targets for the current phase
    -> reactive job and market scores
    -> legality masking and deterministic selection
    -> canonical action decoders
```

### 5.1 Opening program

The opening is a compact schedule of desired state, not a 719-action tape. It
specifies phase milestones such as:

- land unlock timing;
- hiring curve;
- crop, animal, and structure targets;
- seed, inventory, and cash reserves;
- production and liquidation windows.

Milestones are checked against current encoded state. A missed target remains a
goal only while it is affordable, legal, and useful; the controller never emits
an action solely because a fixed turn number says it should.

### 5.2 Guarded execution and recovery

The guarded controller converts current targets into per-unit jobs. It assigns
available workers to planting, care, feeding, harvesting, transport, placement,
building, and land work according to target deficit, urgency, distance, and
legality. It recomputes every turn, so weeds, delayed production, displaced
units, missing inventory, or an unavailable worker do not shift the remainder
of a tape out of alignment.

The controller has explicit reserve constraints. It may not spend protected
cash or inventory merely to satisfy an opening milestone. When several actions
are valid, it ranks them deterministically and selects the highest legal one.

### 5.3 Market controller

Market decisions are always reactive to the canonical live-price and
opponent-supply features. The controller manages:

- affordability and protected cash;
- seed and animal purchases needed by the opening targets;
- inventory reserves for jobs and structures;
- product-specific sale timing and dump pressure;
- opponent supply and expected town demand;
- forced end-season liquidation.

A route may suggest _when_ selling should become urgent, but it never supplies a
blind market order. Prices, inventory, affordability, and order legality are
resolved from current state.

## 6. Typed configuration and optimizer representation

The winning runtime policy is described and validated by a Pydantic
configuration with strict bounds and no extra fields. It separates opening
milestones, job priorities, reserve rules, market thresholds, liquidation
rules, and guarded fallback behavior. Configuration validation rejects
non-finite values, impossible ranges, duplicate milestones, and incompatible
choices before an evaluation starts. Packaging converts the validated model to
a frozen standard-library representation so the submitted turn loop does not
depend on Pydantic; a round-trip equality test prevents conversion drift.

The offline optimizer uses the same categorical representation pattern as the
RL action heads:

- every categorical parameter is one logit group;
- every bounded integer parameter is one logit group over all legal values;
- decoding selects the group's `argmax` and stores the selected index;
- genuinely continuous parameters alone are normalized to `[0, 1]` and mapped
  into their declared range;
- cross-field constraints are either rejected or repaired by one deterministic,
  tested rule before evaluation.

The submitted configuration contains decoded Pydantic values, never optimizer
logits. One-hot groups avoid inventing numeric distances between categories or
evaluating fractional integer policies. The larger search vector is acceptable
because the runtime parameter surface is intentionally compact.

CPU-parallel evolutionary search is the default optimizer. It fits the mixed
discrete/continuous space, supports common-random-number evaluation, and does
not compete with the active GPU job. The optimizer interface is replaceable;
the decoded configuration and evaluation protocol are the stable contracts.

## 7. Search protocol and fitness

### 7.1 Data split

Fix three disjoint seed sets before search:

1. 8 screening seeds used for every candidate, or 16 games per matchup after
   seat swapping;
2. 32 development seeds used only for survivors, or 64 games per matchup;
3. 128 untouched promotion seeds used once for finalists, or 256 games per
   matchup.

Every matchup is seat-swapped. Candidates within a generation receive the same
seeds and opponent versions, reducing evaluation variance. Search results may
not add failed holdout seeds back to development.

### 7.2 All-opponent fitness

Every candidate is evaluated against every member of the development league.
Before search, rank the league by its frontier round robin and assign fixed
strength-tier weights of 4, 3, 2, and 1 from strongest to weakest quartile;
Boatlee v14 receives at least weight 3 because it remains the served-baseline
gate. The primary fitness is

```text
0.70 * weighted_mean(opponent_win_points)
    + 0.30 * min(opponent_win_points)
```

where a win/draw/loss is worth 1/0.5/0. Paired normalized margin breaks exact
fitness ties. Any illegal action, exception, or forfeit makes the candidate
ineligible regardless of score. Runtime breaks any remaining tie.

Bank is a diagnostic, not an optimization target. A candidate cannot compensate
for catastrophic failure against one opponent by farming easy wins against many
weak copies. League membership and weights are versioned with each search run.

### 7.3 Progressive evaluation

Evaluate all candidates cheaply on the screening set, promote only credible
survivors to the development set, and re-evaluate a small finalist set at higher
sample count. This spends games on uncertainty near the frontier instead of
giving every poor mutation a full tournament.

## 8. Failure handling and runtime guarantees

The opening program expresses targets rather than mandatory actions, so ordinary
execution drift is handled by replanning. Every proposed category is checked
against the canonical legality masks. An infeasible choice falls through to the
next-ranked legal category for that slot.

Feature-schema or invariant violations raise immediately with field-level
diagnostics in development. The packaged policy catches only the outer
invariant boundary and returns a deterministic legal PASS/no-market turn rather
than forfeiting the match. It does not delegate to `economic_policy`, because
that would introduce a second raw-observation parser and violate the shared
feature contract.

The submitted controller performs no rollout search, file IO, network IO, or
random exploration. Its work is linear in board cells, units, and products and
must remain inside the established per-turn runtime envelope with substantial
headroom.

## 9. Testing

### 9.1 Encoding and decoding contracts

- Compare the canonical producer plus Torch adapter with the current independent
  encoder functions across the replay corpus; require bit-identical tensors and
  masks.
- Differential-test the tensor-native CUDA feature backend against the
  canonical producer on fixed legal tapes.
- Verify named accessors select the intended channels and normalized values.
- Property-test that generated valid observations always produce actions
  accepted by the canonical masks and decoders.
- Preserve existing RL checkpoint inference and rollout outputs after the
  encoder call-site refactor.

### 9.2 Controller behavior

Test opening targets, worker assignment, reserves, market timing, recovery, and
liquidation independently. Maintain scenario regressions for delayed planting,
weeds, displaced units, insufficient cash, saturated shed, hostile opponent
supply, unavailable targets, and end-season liquidation. Identical observation,
seat, and configuration must produce identical actions.

### 9.3 Frontier and search integrity

- Verify source hashes and exact notebook versions on league load.
- Prove seat swaps and common seeds are applied to every candidate/opponent
  pair.
- Reject overlap among screening, development, and holdout seeds.
- Report every matchup independently rather than only one aggregate.
- Re-evaluate the final candidate on the reference engine even when search used
  the faster simulator.

### 9.4 Packaging and runtime

Build the actual submission archive and verify that it includes the canonical
feature/action core, decoded configuration, and runtime controller but excludes
search machinery, public opponent sources, Torch, and development artifacts.
Run packaged-agent episode smokes, measure per-turn latency distributions, and
require zero illegal actions, exceptions, and forfeits.

## 10. Promotion gate

The promotion holdout is the 128-seed, seat-swapped set fixed in section 7.1.
Use a paired bootstrap over seeds and report two-sided 95% confidence intervals.
A candidate may replace the served agent only when all of the following hold:

1. the lower confidence bound of its direct paired win-point result against
   Boatlee v14 is greater than 0.5;
2. the lower confidence bound of its paired win-point result against
   `public_frontier` is greater than 0.5;
3. the lower confidence bound of its weighted full-league delta over the
   incumbent is greater than zero;
4. no individual matchup's point-estimate win rate is more than 0.05 below the
   incumbent's result on the same paired games;
5. zero illegal actions, forfeits, or nondeterministic outputs;
6. encoder compatibility, controller, reference-engine, packaging, and runtime
   gates all pass.

Report source hashes, engine version, seeds, seats, confidence intervals,
win/draw/loss counts, paired margins, failures, and runtime. Historical Kaggle
ratings remain context only and cannot waive a local gate.

Changing `main.py`, packaging a candidate, or making a Kaggle submission is a
separate explicit decision after this evidence is reviewed.

## 11. Order of implementation

1. Archive and locally rank the exact public frontier versions.
2. Introduce the bit-identical canonical encoded-observation bundle and migrate
   RL call sites.
3. Define the strict Pydantic hybrid configuration and normalized optimizer
   genome.
4. Implement guarded opening, job, market, recovery, and liquidation
   components against the shared feature boundary.
5. Build CPU-parallel progressive league evaluation and evolutionary search.
6. Search, freeze finalists, and run the untouched promotion gate.
7. Only after approval, embed the winning decoded configuration in the
   standalone submission path.

Each step has an independently testable boundary. Frontier discovery can begin
without changing the submission; feature unification can land without a hybrid
controller; and the controller can be validated with hand-authored
configurations before the optimizer exists.
