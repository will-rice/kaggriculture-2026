# Market-Residual RL Design

**Date:** 2026-08-28
**Status:** Proposed
**Scope:** A market-only learned residual over a frozen Kaito v48 logistics controller

## 1. Decision and motivation

The next RL system will not learn Kaggriculture end to end. It will freeze the
verified Kaito v48 policy as the production and logistics controller and learn
only when to deviate in the shared commodity market.

This boundary follows the completed Phase 1 evidence. The production Toad run
used a randomly initialized 2.6M-parameter feed-forward policy, 16-step
unrolls, and atomic control of every unit and market decision. It disabled the
recurrent, belief, transformer, frozen-opponent, snapshot-pool, and teacher
features. After 20,008,332 decisions it still had a 0.000 win rate and a mean
bank below one coin against the economic policy. Legality, optimization, and
value learning were functioning; discovery of the long production chain was
not.

The route-portfolio promotion result establishes the complementary fact. A
Kaito-derived controller is already strong against the measured public field,
but tied Kaito v48 at exactly 0.500 on the untouched direct gate. A useful
learned component must therefore preserve the production plan while creating
profitable, opponent-dependent deviations.

## 2. Goals

1. Preserve Kaito v48's unit movement, production, and investment behavior
   exactly when the residual selects `USE_KAITO`.
2. Learn new `SELL` and `BUY_PRODUCT` orders from live market and opponent
   context.
3. Reduce the effective decision horizon from 719 atomic turns to a small
   sequence of market events.
4. Use recurrent state so price, inventory, demand, and opponent-supply trends
   are observable.
5. Train against a daily refreshed, identity-bound rolling frontier rather
   than one static opponent.
6. Measure incremental paired performance against frozen Kaito on identical
   opponent, seed, and seat cells.
7. Remain deterministic, legal, packageable, and comfortably inside Kaggle's
   runtime budget.

## 3. Non-goals

- The residual does not control farmer movement, hand actions, structures,
  crops, animals, or production jobs.
- It does not create or modify `HIRE`, `BUY_LAND`, `BUY_SEED`, or
  `BUY_ANIMAL` orders.
- It does not replace the existing full Toad implementation. Existing code may
  supply shared features, recurrent modules, loss primitives, and simulator
  infrastructure, but the market residual has a separate typed boundary.
- It does not tune on the consumed promotion seeds `840000..840127`.
- It does not use leaderboard rating as a training label.
- It does not submit automatically. Every Kaggle submission still requires an
  explicit final confirmation for the exact archive hash.

## 4. Runtime architecture

Each turn executes the following deterministic pipeline:

1. Build the canonical shared feature bundle once.
2. Run a fresh frozen Kaito v48 controller to produce the complete baseline
   action.
3. Feed the observation, feature bundle, Kaito commodity proposal, and
   residual recurrent state to the market-event detector.
4. If no event is open, return Kaito's action byte-for-byte and update only the
   passive recurrent observation state.
5. If an event is open, run the market residual policy.
6. Decode either `USE_KAITO` or a replacement set of commodity orders.
7. Apply the canonical market legality mask and independent cash, inventory,
   shed-capacity, and ten-order validations.
8. Merge the resulting `SELL` and `BUY_PRODUCT` orders with every untouched
   Kaito unit and investment action.
9. Record typed decision provenance for evaluation and training.

The merge boundary is fail closed. A malformed policy output, nonfinite logit,
invalid recurrent state, illegal order, or capacity inconsistency returns the
original Kaito action and records the reason. It never drops Kaito's
non-commodity orders.

## 5. Market events

The policy is evaluated every turn to update memory, but a sampled action is
created only when at least one of these conditions holds:

- Kaito proposes a `SELL` or `BUY_PRODUCT` order.
- A commodity becomes newly sellable or newly affordable.
- A live price, market inventory, or town-demand feature crosses a configured
  relative-change threshold since the last event.
- The set of open shops changes.
- Observable opponent public supply changes materially.
- A scheduled liquidation horizon is reached.
- A maximum number of turns has elapsed since the last event, providing a
  bounded heartbeat for gradual trends.

Event thresholds are part of the run identity. They are fixed before an
experiment and cannot drift on resume. Event deduplication prevents repeated
identical states from creating artificial action frequency.

## 6. State and features

The residual consumes the same canonical encoded observation used by the rule
and RL paths. Categorical and integer fields use the shared one-hot schema; no
parallel feature implementation is permitted.

The residual input contains:

- day, hour, turns to liquidation, and turns since the previous event;
- live commodity prices and market inventories;
- price and inventory deltas over the recurrent history;
- town demand, open shops, and shop changes;
- our commodity inventory, cash, protected reserves, shed capacity, and
  expected near-term production;
- observable opponent public supply and its recent changes;
- Kaito's proposed commodity slots and quantities;
- the previous residual action and whether it was adjusted by safety checks;
- the previous recurrent hidden state.

Hidden opponent information is never inferred from labels unavailable at
runtime. Auxiliary targets may estimate future observable supply and sell
timing, but their inputs remain public.

## 7. Action space

At each market event, the policy chooses one of two modes:

- `USE_KAITO`: preserve Kaito's commodity orders exactly.
- `REPLACE`: emit quantity buckets for the allowed `SELL` and `BUY_PRODUCT`
  slots.

All `HIRE`, `BUY_LAND`, `BUY_SEED`, and `BUY_ANIMAL` slots are copied from
Kaito and absent from the learned action space. The decoder uses the existing
canonical quantity vocabulary and legality masks. A replacement may contain
multiple commodity orders up to the environment's order limit.

Offline candidate generation uses a bounded alternative set rather than the
full Cartesian product. It includes Kaito, cancel/delay, scaled Kaito
quantities, legal single-product alternatives, and a small number of
price-ranked multi-order combinations. The generator is deterministic and
identity-bound.

## 8. Model

The first model is intentionally small:

- one feature projection;
- a single-layer GRU over market events;
- a mode head for `USE_KAITO` versus `REPLACE`;
- factorized quantity heads for allowed commodity slots;
- a Q/value head for expected paired incremental outcome;
- auxiliary heads for next-event price/inventory change and opponent public
  supply timing.

The initial size target is below one million parameters. Training uses PyTorch.
The submitted runtime exports only the residual parameters and a minimal
inference implementation. The preferred runtime is NumPy or an equivalent
small dependency-free implementation to avoid Torch startup cost; exact
export parity is mandatory. Torch runtime remains an allowed fallback only if
the extracted archive's measured overage retains a wide safety margin.

## 9. Counterfactual dataset

Counterfactual rows are generated from identity-bound game cells against the
rolling frontier.

For each selected market event:

1. Snapshot the exact simulator, policy, opponent, RNG, and recurrent state.
2. Apply each bounded legal alternative.
3. Continue the episode with frozen Kaito handling all later actions for the
   initial single-intervention dataset.
4. Store terminal banks, win points, normalized margin, realized commodity
   prices, execution failures, and exact provenance.
5. Define the label as the alternative result minus Kaito's result on the same
   opponent, seed, and seat.

Rows are immutable and content addressed. Duplicate cells, missing
alternatives, engine drift, source drift, or incomplete terminal states reject
the shard. The consumed route-promotion seed bank is prohibited at the schema
boundary.

Development, selection, and final-promotion banks are disjoint. Temporal
validation trains on older frontier snapshots and selects on the newest unseen
snapshot. At least one opponent behavior cluster is held out entirely.

## 10. Offline training

Offline initialization performs sequence-aware supervised learning:

- pairwise ranking between each alternative and `USE_KAITO`;
- Q regression to paired win-point delta, with normalized margin as a
  secondary tie-break target;
- behavior loss that initially favors `USE_KAITO` when evidence is weak;
- auxiliary next-event prediction losses;
- extra weight for decisive loss-to-win and win-to-loss flips.

The model is selected by unseen paired win-point delta, calibration, action
activation, and protected-regression counts—not training loss.

## 11. Online residual RL

Online fine-tuning uses PPO/V-trace only at event boundaries. Every candidate
game is paired with a frozen-Kaito control game on the identical cell. The
terminal reward is candidate win points minus Kaito win points; normalized
margin breaks equal-win-point ties for value learning but cannot override the
win objective.

The shorter event sequence and offline initialization replace random atomic
exploration. No raw-bank reward or self-play mirror score is treated as
capability. Frozen opponents dominate the first online phase. Candidate
self-play may be introduced only after the policy beats Kaito on an unseen
fixed-opponent panel.

The rolling opponent mixture includes current public sources, recent top-band
trajectory replays, stable historical anchors, and deliberately adversarial
market variants. Sampling is stratified by behavior cluster and emphasizes
opponents on which the current residual underperforms.

## 12. Rolling frontier

A daily read-only ingestion job creates an immutable frontier generation. Each
generation binds:

- engine version;
- public source and archive hashes;
- top-episode hashes and provenance;
- behavioral-cluster assignments;
- strength weights;
- the exact seed banks used for development and selection.

An active run never follows a mutable `latest` pointer. It pins one or more
frontier generations in its identity. A new generation starts a new experiment
or is admitted at a documented curriculum boundary.

Behavior clustering uses only observable episode behavior: production route,
commodity mix, sell timing, volume, realized prices, investment schedule, and
market aggression. Names and leaderboard ratings are metadata, not cluster
features.

## 13. Verification ladder

The project advances only through these gates:

1. **Runtime parity:** with residual mode forced to `USE_KAITO`, every action,
   bank, and status exactly matches Kaito across both seats and a fixed corpus.
2. **Safety:** fuzzed outputs cannot alter non-commodity actions or create an
   illegal order.
3. **Counterfactual truth:** single-intervention branches match reference-engine
   results on differential fixtures.
4. **Learning sanity:** the offline model learns a synthetic and a real known
   market exploit on unseen cells.
5. **Small online gate:** within a predeclared small budget, objective paired
   win points improve over offline initialization. If not, stop before scale.
6. **Temporal selection:** candidate beats frozen Kaito and the rolling champion
   on the newest unseen frontier generation, with no protected cluster
   regression beyond the declared tolerance.
7. **Package gate:** exported inference matches PyTorch logits/actions, imports
   no training stack, completes full episodes, and fits the sandbox budget.
8. **Single-use promotion:** one untouched bank produces a deterministic,
   failure-free atomic verdict.
9. **Submission:** show the exact archive hash, promotion evidence, and current
   Kaggle quota, then require explicit user confirmation.

The initial temporal-selection target is paired win points of at least 0.55
against Kaito v48 and a positive lower confidence bound over the weighted
top-behavior clusters. Exact thresholds and sample counts are fixed in the
implementation plan before data generation.

## 14. Metrics

Primary metrics:

- paired win-point delta versus frozen Kaito;
- direct win points versus the current rolling champion;
- weighted and worst-cluster win points;
- protected-regression count and magnitude;
- deterministic execution failures.

Diagnostic metrics:

- event count and trigger distribution;
- `USE_KAITO` versus `REPLACE` rate;
- action divergence and safety-adjustment rate;
- sell/buy volume, realized price, and market profit by commodity;
- Q calibration and advantage by action family;
- recurrent-state norms and reset counts;
- collection, counterfactual, learner, and export throughput.

Raw bank, proxy reward, entropy, and critic loss remain diagnostics and cannot
authorize promotion.

## 15. Throughput and resource constraints

All production training and evaluation must support one explicitly selected
GPU. CPU workers may be used for reference-engine games, but every command
records its worker limit and must coexist with other approved workloads.

The previous production run achieved only about 75 environment decisions per
second and took 75.8 hours for 20M decisions. The new pipeline therefore has a
mandatory throughput profile before training. Collection, simulator, feature,
policy, and counterfactual-branch timings are measured independently. No scale
estimate may reuse the earlier unverified 38k-steps/second assumption.

## 16. Artifact and resume safety

Run identity binds configuration, code, feature schema, model schema, event
rules, action generator, engine, frontier generations, seeds, and source
hashes. Counterfactual data, checkpoints, evaluation reports, and exported
runtime weights use atomic publication and strict validation.

Resume rejects semantic drift. Failed or interrupted stages leave enough
durable provenance to distinguish safe continuation from forbidden replay of a
protected bank. Only one coordinator may own a run root.

## 17. Delivery sequence

1. Specify exact interfaces and seed banks in an implementation plan.
2. Implement the runtime merge and safety boundary with parity tests.
3. Implement event detection and recurrent market model.
4. Implement exact snapshot branching and counterfactual artifacts.
5. Generate a bounded development dataset.
6. Pass offline learning sanity and export parity.
7. Implement paired online residual training and rolling-frontier ingestion.
8. Pass the small online gate before authorizing scale.
9. Train one reviewed candidate on a pinned rolling frontier.
10. Run temporal selection, package, single-use promotion, and the explicit
    submission boundary.

This sequence deliberately earns the right to spend compute. A failed parity,
counterfactual, or small-learning gate stops the project at the cheapest
evidence boundary rather than producing another multi-day flat run.
