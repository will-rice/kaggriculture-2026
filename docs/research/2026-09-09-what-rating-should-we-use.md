# What rating should the campaign steer on?

**Date:** 2026-09-09
**Question:** Our gate ranks and promotes on a Bradley-Terry fit. Should we
switch to something else?
**Answer:** No — but not because the fit is healthy. It is wrong by about 175
Elo on a typical pairing and biased in opposite directions on the two halves of
the pool. The cause is that 86% of its evidence is our own lineage measuring
itself, and no estimator fixes that.

---

## Executive summary

The rating overstates our strength against outsiders and understates it inside
the lineage. Measured off the live field (67 agents, 490 pairings, every rate
over 32 games):

| Measurement                      | Result                      | Reading                    |
| -------------------------------- | --------------------------- | -------------------------- |
| Over-claim, champion vs public   | **+4.8 points of win rate** | Fit flatters us            |
| Over-claim, champion vs champion | **−4.6 points**             | Fit understates us         |
| Residual after the deployed fit  | **1.19 logits²**            | Large                      |
| — of which sampling noise        | 0.18 logits²                | 15%                        |
| — of which genuine               | **1.01 logits² ≈ 175 Elo**  | 85%, seven times the bar   |
| Pool composition                 | **55 ours / 12 public**     | 86% of evidence is our own |

A 9.5-point swing between the two halves, and the half it flatters us on is the
half that predicts the leaderboard. The fit says champion_55 beats `router_v1`
at 0.997; it wins 0.812. That is the same discrepancy that came up as "we are
winning locally but it is not translating" — not a measurement error, but the
rating being optimistic by construction about the opponents it has least
evidence on.

This is Balduzzi's clone-inflation failure. It is a property of _who is in the
pool_, so **keep Bradley-Terry and fix the population.** The residual is real
structure — some agents are simply better against some others than a single
number can express — but it does not form cycles, which is what rules out every
method the literature would otherwise point us at.

---

## Why not switch the estimator

Every candidate method exists to handle cyclic populations. Ours is not one:
56 of 1,549 measured triples beat each other round, 3.6%. Nobody expected a
farming economy to be rock-paper-scissors, and it is not — the point of
recording the number is that it disqualifies the whole branch below, which is
otherwise the obvious place to look for a better rating.

**Nash averaging** ([Balduzzi et al. 2018](https://arxiv.org/pdf/1806.02643))
is the standard answer to a manipulable population. Its property P1 is
invariance to redundant copies, and the paper states Elo's two failures
bluntly: Elo "is meaningless — it has no predictive power — in cyclic games
like rock-paper-scissors", and "an agent's Elo rating can be inflated by
instantiating many copies of an agent it beats (or conversely)". Only the
second applies to us.

**mElo2k**, from the same paper, adds a low-rank cyclic term to the transitive
fit. It is the right shape for a field whose residual is cyclic. Ours is not
— and a
[2025 study](https://educationaldatamining.org/EDM2025/proceedings/2025.EDM.long-papers.99/index.html)
finds mElo "struggles to converge to true values the further true ability
deviates from zero", which is precisely where our champions sit relative to
the field.

**Deviation ratings**
([Feb 2025](https://arxiv.org/pdf/2502.11645)) are the newest clone-invariant
method and the closest match to our actual complaint. But they are defined as
the gap between an agent's win rate and its expected rate _against a Nash
equilibrium mixed strategy_, and they assume that equilibrium is available.

That last point is the one that settles it, and it is not a computational
objection. **The competition does not score a Nash equilibrium. It scores a
Bradley-Terry tournament.** Steering on a solution concept other than the one
we are graded by would optimise the wrong objective — the same class of error
as the gate ranking on `fitness` while promoting on rating, which this
codebase already made once and fixed (see `archive.Database.top`).

So: the model matches the finale and should stay.

_Caveat carried forward:_ that the finale is Bradley-Terry rests on a
competitor's account, flagged as such in
`2026-09-02-literature-competitive-sim-agents.md` [21], not on an organiser
statement. If that is wrong, this recommendation changes.

---

## What is actually broken

Public opponents _are_ harvested — twelve sit in `roster.TRAINING`, four of
them added on 2026-09-06 and one on 09-07. The defect is not that outsiders
never arrive. It is that the gate almost never plays the ones that do:

| Public opponent                            | Pairings on record |
| ------------------------------------------ | ------------------ |
| `thomastschinkel_router` — **gate anchor** | 33                 |
| `router_v1` — **gate anchor**              | 31                 |
| `pilkwang_economic`, `shopforge`, `v56`    | 10–12              |
| `router2929`, `v54`                        | 9                  |
| `boatlee_v29` (harvested 09-06)            | **2**              |
| `avioon_apex_v7`, `indarkarhana` (09-06)   | **1**              |
| `lynnsakurai_threeday` (09-06)             | **1**              |
| `tetsutani_shape0905` (09-06)              | **0**              |

Overall that is 405 champion-vs-champion pairings against 52 champion-vs-public
and 14 public-vs-public — 86% of the evidence is the lineage measuring itself.

`pool.sample` is the mechanism, and it is a feedback loop rather than an
oversight. Sixteen slots are filled three ways: six fixed `GATE_ANCHORS`, six
**contenders drawn as the highest rated**, the leader and floor, and whatever
random remainder is left — three slots or so, drawn from a pool of 67 in which
12 are public.

Each part is individually well-argued. The anchors exist because rating a
champion through a chain of overlapping pool eras is measurably wrong; the
docstring cites champion_37 rated 0.994 against champion_1 and beaten 0.729 in
the games. The contenders exist because topping the field means beating the
best of it.

Together they close a loop. Champions' ratings are inflated by the lineage
effect above, so champions _are_ the highest rated, so champions take all six
contender slots, so the next gate is another six intra-lineage pairings, which
inflates the ratings further. Publics reach a candidate only through the two
public anchors and the random remainder.

And the two public anchors are one agent. `champion_1` and
`thomastschinkel_router` return identical rates _and_ identical margins to
every decimal across all four statistics. So the fit's tie to the outside world
runs through a single program, harvested weeks ago from a field the corpus
shows turns over completely in about twelve days — while five opponents
harvested on 09-06 and 09-07 sit in the roster with five pairings between them.

The consequence is the +4.8 / −4.6 split. Long chains of near-identical
champions stretch the internal rating scale — each generation beats its
ancestors at 0.94-ish, and those margins compound into rating gaps that get
extrapolated onto outsiders where they do not hold. The fit says champion_55
beats `router_v1` at 0.997. It wins 0.812.

That number is worth sitting with, because it is the same discrepancy that
came up as "we are winning locally but it is not translating to the
leaderboard". It was not a measurement error. It is the rating being
optimistic by construction about exactly the opponents it has least evidence
on.

---

## Recommendation

**1. Keep the Bradley-Terry fit.** It matches the finale, it is order-independent
across eight concurrent workers, and it has no K-factor to tune. Nothing in the
measurements argues for replacing it.

**2. Report ratings in Elo points.** `Elo = 400/ln(10) × log-odds ≈ 173.7 ×`.
`PROMOTION_MARGIN = 0.15` is ~26 Elo. The residual measured above is ~175 Elo,
which is _seven times the promotion bar_ — a fact that is invisible while the
numbers are printed as bare log-odds.

**3. Fix the population. This is the finding.** In rough order of value per
unit of work:

- **Anchor on the freshest publics, not a hardcoded list.** Harvesting already
  works; the sampler is the bottleneck. `GATE_ANCHORS` guarantees games to two
  stale opponents and leaves five recent ones to chance, which is why four of
  them have one pairing or none. Anchoring on the most recently harvested
  publics would fix the evidence shortage without harvesting anything new.
- **Treat `champion_1` and `thomastschinkel_router` as one agent.** Two of six
  anchors are the same program, so the anchor set is really five.
- **Steer on the cross-population number.** `Result.field` — the win rate over
  vendored opponents alone — is already computed and its docstring already
  says it is "the one number that means the same thing on the first session and
  the thousandth". It is informed by 52 of 471 pairings. Fixing the first bullet
  is what makes it trustworthy.

**4. Do not add a diversity mechanism yet.** Balduzzi's own corollary: "If there
is a dominant agent then diversity is zero", and under a dominant agent PSRO
"degenerates to training against the best agent in the population". Best-
responding is the principled move while a monoculture holds. The decisive-games
guard added on 2026-09-08 already stops the degenerate case where the best
response is a copy.

---

## What would change this conclusion

- **Cyclic triples rising above ~15%.** Unlikely on this game, but it is the
  cheap check that would reopen the estimator question. Re-run `cycles.py`.
- **The finale turning out not to be Bradley-Terry.** Then the objective moves
  and deviation ratings become the right target.
- **The over-claim against publics not shrinking after harvesting.** That would
  mean the bias is not population composition and the estimator is implicated
  after all.

---

## Method

All four measurements are off `run/campaign/field.json`, the campaign's own
record of every pairing it has played, using the deployed `rating.standings`
fit rather than a re-derivation.

The noise floor is the delta-method variance of a measured logit,
`1 / (n · p · (1 − p))`, averaged over pairings. Saturated pairings — every game
to one side, 19 of 490 — are excluded from the residual analysis, because their
logit is infinite and including them would let the clipping constant decide the
answer. Errors are signed toward the side the fit favours so that over- and
under-confidence cannot cancel.

Scripts: `transitivity.py`, `cycles.py`, `inflation.py`, run 2026-09-09 against
a field of 67 agents and 490 pairings.

## Sources

- [Re-evaluating Evaluation (Balduzzi et al., 2018)](https://arxiv.org/pdf/1806.02643)
- [Deviation Ratings: A General, Clone-Invariant Rating Method (2025)](https://arxiv.org/pdf/2502.11645)
- [Evaluating multidimensional extensions of the Elo rating systems (EDM 2025)](https://educationaldatamining.org/EDM2025/proceedings/2025.EDM.long-papers.99/index.html)
- [Multiagent Evaluation under Incomplete Information (Rowland et al.)](http://papers.neurips.cc/paper/9395-multiagent-evaluation-under-incomplete-information.pdf)
- [Multi-Elo Rating System overview](https://www.emergentmind.com/topics/multi-elo-rating-system-mers)
- Prior in-repo research: `docs/research/2026-09-02-literature-competitive-sim-agents.md`, Finding 7
