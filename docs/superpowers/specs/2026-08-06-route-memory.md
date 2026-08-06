# Spec: route memory — replay the best farms we have, adapt the market

**Goal.** Bank more than 118,000 by replaying production routes harvested from
the top decile of our own replay corpus, matched to the live board by a
state signature, with the market layer recomputed rather than replayed.

## The evidence

Measured on 2026-08-06, 80 seat-routes sampled from one archive:

|                      | bank        |
| -------------------- | ----------- |
| minimum              | 76,929      |
| 25th percentile      | 104,309     |
| **median**           | **125,773** |
| 75th percentile      | 137,670     |
| top decile threshold | 149,120     |
| maximum              | 158,603     |

Our submitted agent — the vendored `economic_policy` — banks about **118,000**.
It is below the median of the corpus already on disk, and we hold roughly 9,400
seat-routes across six archives, refreshed nightly.

Separately, from the public kernel cache: the strongest published agent scores
**2863.9** on the public leaderboard against our **1289.6**, and 96.7% of it is
memorised route tables with 14 KB of retrieval logic. Route memory is not a
theory about this game; it is what is winning it.

A discussion thread on 2026-08-06 reached the same conclusion independently:

> "The large initial market depth makes the early game almost separable:
> expansion, hiring, planting, and animal production are mostly stable across
> opponents. However, later-game shared market depletion, order resolution, and
> terminal liquidation make the payoff opponent-dependent... the strongest
> long-run design is a static production backbone with selective,
> hysteresis-based adaptation."

That is exactly this design: a replayed production backbone, an adaptive market.

## Why this and not more behaviour cloning

Phase 3 cloned unit actions and market orders from the same corpus and banked 0.
The clone re-decides every turn from a state it has drifted into, and compounding
error destroys it — measured, it hired on essentially every turn where the
teacher hires on 14.6% and only in the first four hours of a day.

A route does not re-decide. It carries the teacher's _sequence_, which is
precisely the thing a per-turn policy cannot represent and precisely where the
clone failed. The two are complementary, not competing: this spec's fallback is
a learned or heuristic policy, and a working route memory gives the RL phase a
far better initialisation than a bankrupt clone.

## Design

### 1. Prototypes

A prototype is one seat's full action sequence from one episode, kept only if its
final bank clears the top-decile threshold. Harvest from the same rating-filtered
selection `corpus.select` already produces, so prototype quality inherits the
ladder filter as well as the bank filter.

Store the actions, not the observations. A route is ~719 turns of
`{"farmer": [...], "hands": [...], "market": [...]}`; the boards are
reconstructable and would dominate the artifact.

**Deduplicate.** The corpus is dominated by a handful of public kernels, so
hundreds of routes will be near-identical. Cluster by signature trajectory and
keep one exemplar per cluster with its bank — a thousand copies of one route is
a thousand times the storage and none of the coverage.

### 2. Signature

At each turn, reduce the public state to a fixed-width vector: tile counts by
crop and by animal, weeds, money, hands, hires today, unlocked quadrants, day
and hour, and the shed totals. **Identity-free** — nothing about which opponent
is playing, so a prototype transfers across matchups.

`encode_scalars` already computes most of this and is tested against the engine.
Reuse it rather than writing a second state summary that can disagree with the
first.

### 3. Retrieval

Match the live signature to the nearest prototype and take its action for the
current turn. Two things must be decided by measurement, not assumption, and the
plan must measure them:

- **Re-match cadence.** Matching every turn risks thrashing between prototypes
  whose sequences are individually coherent and jointly incoherent. Matching
  once and committing is brittle when the board diverges. Hysteresis — stay on
  the current prototype unless another is better by a margin — is the obvious
  middle and is what the published agent does.
- **Distance weighting.** Day and hour should dominate early (routes are
  phase-locked), board composition later.

### 4. Realignment

A replayed action assumes a unit is where the prototype's unit was. When the
boards diverge, the action is a no-op at best. Before issuing, remap the
prototype's unit actions onto our actual units — nearest unit to the prototype's
unit position, and drop actions for units we do not have. The published agent
carries a function for exactly this; so must this one.

### 5. The market is recomputed, never replayed

Prices depend on both players' cumulative sales, so a replayed `SELL` quantity is
priced for a market that no longer exists. Replay the production plan; compute
market orders live.

We already own this layer: `economic_policy`'s pricing reads the live curve and
the opponent's visible supply, it is linted, typed, and unit-tested against the
engine's `market_price` across every product and inventory level. Call into it
rather than reimplementing.

### 6. Fallback

When no prototype is within a distance threshold, fall through to
`economic_policy` whole. That makes the floor of this design the current
submission's 118,000 rather than zero, and it makes the first experiment
cheap: if route memory never beats the fallback, the fallback still plays.

## The gate

1. **Beats `economic_policy`** — the current submission, ~118k. This is the bar
   that matters; below it there is no reason to ship.
2. Beats the meta tape (~140k).
3. Approaches the corpus top decile (149k+), which is the ceiling of pure replay.

Report bank per episode, not only win rate, and report how often the fallback
fired — a route memory that falls back 90% of the time is `economic_policy` with
extra steps, and the win rate alone will not say so.

## Risks

- **The meta moves and prototypes decay.** Mitigated by the nightly corpus
  fetch: prototypes are rebuilt from fresh archives, unlike a competitor's
  frozen table. Measure decay explicitly — evaluate prototypes harvested from
  the oldest archive against the newest.
- **Replay is open-loop and exploitable.** An opponent that recognises a replayed
  route can counter it. This is why the market layer stays adaptive and why the
  gate includes the reactive `economic_policy` rather than only recordings.
- **Storage and per-turn cost.** Retrieval is a nearest-neighbour search over a
  few hundred prototypes at 1 second per turn — trivial, but the artifact must
  ship inside the submission archive, which currently has no mechanism for
  shipping data files. Phase 3 hit the same gap.
- **Prototype selection by final bank rewards luck.** A route that banked 158k
  against a weak opponent is not better than one that banked 150k against a
  strong one. Prefer routes whose opponent also banked well, and say what the
  criterion was.

## Out of scope

Vendoring the published 2864 agent. It is 403 KB of someone else's memorised
tables, deliberately obfuscated, and it decays with the meta while teaching us
nothing we cannot derive from our own corpus.
