# Re-baselining onto kaggle-environments 1.32.6

**Measured 2026-08-08** on `kaggle-environments` 1.32.6, the build the ladder
moved to on 2026-08-07. This project was pinned at 1.32.3 until today, so every
head-to-head, bank and fixture recorded here before this date describes an
economy that no longer exists. Section 5 lists what is void.

Reproduce the reference table with

```
uv run run --agent src/kaggriculture/<A>.py \
  --opponents src/kaggriculture/<B>.py ... \
  --games 256 --seed 2000000000 --workers 60
```

once per agent, and the checkpoint row with
`uv run python -m kaggriculture.learn.scripts.gate <weights>`.

**The question.** Does the ranking of the agents change on the new engine? It
does not. Section 2 is the answer and section 3 is the reason it is not the
non-event it looks like. Section 4 is what the upgrade broke in shipped code.

---

## 1. The engine delta, from the two wheels

Diffed directly:
`~/.cache/uv/archive-v0/9u4eSg9L8WbrHCwy/kaggle_environments/envs/kaggriculture/`
(1.32.3, sdist `2883724c…`, uploaded 2026-08-03) against the installed
1.32.6 (`0221b66e…`, uploaded 2026-08-07). Five behavioural changes and one
documentation-only change. Nothing else in `kaggriculture.py` moved.

1. **Town-centre demand collapsed about eightfold, late season.**
   `TOWN_CENTER_DEMAND_SCHEDULE = [(20, 4), (10, 2), (0, 1)]` is deleted and
   `_town_consume` now removes a flat `1`; `townCenterSellInterval` went 12 → 24
   in `kaggriculture.json`. On days 20–29 the centre used to take
   `2 ticks/day × 4 = 8` units per product per day and now takes `1 × 1 = 1`.
   On days 0–9 it is 2/day → 1/day. **This is the whole story of section 2.**

2. **Shops are drawn with replacement.** `_advance_day` was
   `rng.choice(sorted(remaining))` guarded by "shops not yet unlocked" and is
   now `rng.choice(sorted(SHOPS))` guarded by
   `len(unlocked_shops) < MAX_SHOP_INSTANCES`, with `MAX_SHOP_INSTANCES = 8`.
   `len(SHOPS) == 8`, so the _count_ of open instances is unchanged — 8 by day
   24 either way. What changed is composition: it was a permutation of all
   eight shops and is now an i.i.d. multiset, so per-product shop demand is
   lumpy and random per episode where it used to be identical in every episode.
   Expected total demand is unchanged; its variance is not.

3. **`shedCapacity` is enforced on BUY.** `_commit_unit` takes a
   `shed_capacity` argument and refuses `BUY_PRODUCT` and `BUY_ANIMAL` when
   `sum(shed.values()) >= shed_capacity`. It previously refused neither.

4. **Shed operations resolve before the `LOCKED` guard.** `DROP`, `PICKUP` and
   now `PLACE` moved above `if tile == "LOCKED": return`, so the shed is
   reachable from the three of four shed-access tiles that start locked. The
   guard still covers every op that mutates the tile itself.

5. **`PLACE` moved with them**, which also makes animal placement legal from a
   locked tile — except that a locked tile is the string `"LOCKED"` and never a
   structure dict, so that branch cannot match and `PLACE` falls through to the
   shed-drop path. Behaviourally this is item 4 for `PLACE`.

**One claim in circulation is wrong and is corrected here.** The town centre
did **not** stop buying fertilizer.
`TOWN_CENTER_PRODUCTS = [p for p in PRODUCTS if p != "FERTILIZER"]` is
byte-identical in both wheels; only the JSON _description_ of
`townCenterSellInterval` was reworded to mention it. Anything reconstructing
archived demand may keep using the live `TOWN_CENTER_PRODUCTS`, and
`learn/bands.py` does.

**And the archives are not all on one build.** Section 5 has the count: the
2026-08-07 archive alone holds 404 legacy episodes and 271 flat-rate ones, so
"the corpus predates the change" is false and any reconstruction that assumes
it mis-models 40% of the newest day.

The remaining diff is a comment on `MARKET_I0` clarifying that `T` is a 24-day
window rather than the 30-day season. No constant changed.

---

## 2. The reference table on 1.32.6

Five scripted agents, full round robin, **512 seeded games per pair**: 256 with
each agent in seat 0 and 256 with it in seat 1, on seeds
`2_000_000_000 … 2_000_000_255`. That is the seed block `gate.py` holds out of
every training run, so nothing here has been fitted to these seasons. Playing
both seat orders costs nothing and removes the question; it also shows the game
is near-perfectly seat-symmetric — seven of the ten pairs return exactly
complementary win counts and the other three differ by at most two games in 256.

Each agent's row pools its 2,048 games against the other four.

| agent                                       | win rate vs the field | 95% Wilson     | mean bank |
| ------------------------------------------- | --------------------: | -------------- | --------: |
| `boatlee_v14_policy`                        |             **0.960** | [0.951, 0.968] |    87,991 |
| `kaito_v23_policy`                          |                 0.697 | [0.677, 0.717] |    85,700 |
| `kaito_v22_policy`                          |                 0.516 | [0.494, 0.537] |    83,213 |
| `kaito_policy` (v21.1, served by `main.py`) |                 0.209 | [0.192, 0.228] |    79,033 |
| `economic_policy`                           |                 0.117 | [0.104, 0.132] |    73,428 |

Pairwise, so the transitivity can be checked rather than assumed. Cell is the
row agent's win rate over 512 games against the column agent.

|             | econ  | v21.1 | v22   | v23   | boatlee |
| ----------- | ----- | ----- | ----- | ----- | ------- |
| **econ**    | —     | 0.164 | 0.139 | 0.086 | 0.080   |
| **v21.1**   | 0.836 | —     | 0.002 | 0.000 | 0.000   |
| **v22**     | 0.861 | 0.998 | —     | 0.125 | 0.078   |
| **v23**     | 0.914 | 1.000 | 0.875 | —     | 0.000   |
| **boatlee** | 0.920 | 1.000 | 0.922 | 1.000 | —       |

The order is strict and total: every cell above the diagonal is below 0.5 and
every cell below it is above. No cycles.

Three shutouts, each at 512-0 with a Wilson upper bound of 0.007:

- `boatlee_v14` beats `kaito_v23` 512-0.
- `boatlee_v14` beats `kaito_policy` (v21.1) 512-0.
- `kaito_v23` beats `kaito_policy` (v21.1) 512-0.

**The ranking is unchanged from 1.32.3.** On the old engine boatlee beat v23
256-0 and v21.1 256-0, and v23 beat v21.1 128-0 with v22 beating v21.1 124-4.
Every one of those results reproduces, and the two that were near-shutouts are
now shutouts (v22 over v21.1 is 511-1 across 512 games).

### The shaped-PPO checkpoint

`/data/kaggriculture/selfplay/resumed-3cc6068-1786139297/policy.pt`, the 8-block
/ 256-channel bare state dict, loads strictly into the current `Policy` — no
architecture inference was needed. Gated through
`learn/scripts/gate.py` at 128 games per rung on the same seed block, sampled
under its training masks:

| rung              | checkpoint win rate | 95% Wilson     | checkpoint bank | rung bank |
| ----------------- | ------------------: | -------------- | --------------: | --------: |
| `bc_clone`        |               1.000 | [0.971, 1.000] |          27,437 |       180 |
| `economic_policy` |               0.000 | [0.000, 0.029] |          14,339 |   130,447 |
| `best_route`      |               0.000 | [0.000, 0.029] |          16,574 |   135,468 |
| `kaito` (v21.1)   |               0.000 | [0.000, 0.029] |          13,091 |   141,604 |
| `kaito_v22`       |               0.000 | [0.000, 0.029] |          13,029 |   145,312 |
| `kaito_v23`       |               0.000 | [0.000, 0.029] |          11,930 |   140,948 |
| `boatlee_v14`     |               0.000 | [0.000, 0.029] |          11,991 |   141,312 |

Mean bank across the five reference opponents: **12,876**. It clears rung 1
only, and rung 1 is two agents that both bank nothing (27,437 against 180, on a
3,000 opening). Zero illegal actions in 92,032 sampled decisions per rung, so
the masking is sound and the result is the policy rather than the harness. The
checkpoint is not a competitor on this engine and was not one on the last.

**A second reading of the same table.** The opponents bank 130k–145k here while
the _same_ agents bank 73k–88k in the round robin above. The difference is
entirely the other seat: against a farm that banks 12,876 the book is
uncontested. Roughly **45% of a strong agent's bank is denied to it by a second
strong agent** on 1.32.6 — which is the price-impact effect this project has
been chasing, now measured on the current engine.

### What did move: the level, not the order

Every bank in the table above is roughly half what this project used to record.
The cleanest apples-to-apples measurement is the characterization fixture, which
plays `economic_policy` against `pass` — an opponent that never touches the
book, so the whole difference is the town:

| seed | 1.32.3 (2026-08-04) | 1.32.6 (2026-08-08) |
| ---- | ------------------: | ------------------: |
| 11   |             152,803 |             149,176 |
| 22   |             152,145 |             147,181 |
| 33   |             159,087 |             108,290 |
| mean |             154,678 |    134,882 (−12.8%) |

Against `pass` the loss is 12.8%. In the round robin, where two farms flood a
book the town now drains eight times slower, mean banks land at 73k–88k against
the 145,604–162,251 this project last recorded for a league sweep. **The
absolute banks are the numbers the engine change destroyed; the relative order
is the number it left alone.**

---

## 3. Why the ranking held, and where it is fragile

The prior expectation was that the memorised tapes would degrade: a fixed
719-turn action table cannot adapt to a shop draw that is now random with
replacement. They degraded in coins and not in rank, and there are three
reasons, only one of which is comfortable.

**`kaito_v23` is not the agent we measured before.** Its `agent` takes the
engine configuration and branches on `townCenterSellInterval >= 24`, selecting a
route fitted for the announced PR #1394 rebalance. On 1.32.3 that branch was
dead code and it always played the legacy table. On 1.32.6 it selects the
rebalance table, which had **never once executed in this repository**. So the
v23 row is a first measurement, not a re-measurement, and the author's own
notebook numbers (42 of 45 at +3,249) describe the legacy branch we no longer
run.

**`boatlee_v14` was built for this engine.** Its route was fitted on
2026-08-08, after the ladder moved. Its `agent` takes only the observation, so
its internal demand model reads `_get(None, "townCenterSellInterval", 24)` and
falls through to the hardcoded default of **24** — which is wrong on 1.32.3 and
correct on 1.32.6. It was the strongest agent on the old engine while carrying
the new engine's constants, and it is the strongest on the new one. That is
consistency, not adaptation.

**`kaito_policy` (v21.1) and `kaito_v22_policy` cannot branch at all.** Neither
reads the configuration. They lost 512-0 and 472-40 to boatlee respectively, and
v21.1 now loses to v22 511-1 where it lost 124-4 before. They are the
generations the demand collapse hurt most, and they are at the bottom.

The fragility: the shop draw is now random per episode, so an agent whose route
is a _single_ table is exposed to composition variance no seed-averaged win rate
reveals. Nothing in this measurement separates "robust to the draw" from "lucky
across 512 draws" — the variance is inside the 512-game mean. That is the next
measurement, not this one.

---

## 4. Two bugs the upgrade put into shipped code

`learn/mask.py` ships inside the submission and decides what the learned policy
is allowed to do. Two of the engine deltas in section 1 made it wrong, and
neither raises anything — a wrong mask is invisible from outside.

**Over-strict on the shed (delta 4).** `_legal_ops` returned at
`tile == "LOCKED"` before considering `DROP`, `PICKUP` and `PLACE`, matching the
old dispatch order. The engine moved them above the guard, so from a locked
shed-access tile it now accepts ops the mask forbade. Measured on real archived
turns: **14 over-strict `PICKUP` disagreements**. A forbidden op the engine
accepts is deleted from the action space for good — the policy never learns the
move exists.

**Over-permissive on BUY (delta 3).** `_fillable` bounded `BUY_PRODUCT` by money
and `BUY_ANIMAL` by money alone. The engine now refuses both when the shed is
full. Measured: **1,038 over-permissive (slot, bucket) cells**, all of them
large buy quantities the engine would not fill.

Both are fixed and both directions are re-verified against the engine on real
observations: **0 over-permissive and 0 over-strict across 50,380 (unit, op)
pairs on 144 turns, and 36,288 (slot, bucket) cells on 108 turns**, with 300
`PICKUP`, 140 `PLACE` and 99 `DROP` acceptances exercised so the zero is not
vacuous.

A third test, `test_play`'s non-vacuity floor, asserted `ops > 3000` for a full
episode and read 2,344 on 1.32.6 — the clone hires a smaller crew now. That is
the economy moving, not the masking breaking; the assertion is now expressed as
a multiple of the never-hired floor (`ops > 2 * turns`) rather than a recorded
count, so it stops needing a re-record on every engine bump.

---

## 5. Numbers this project has recorded that are now void

Every one of these was measured on 1.32.3 and must not be cited, compared
against, or used in an acceptance decision.

**Void — engine-dependent measurements.**

- `main.py`'s header table: economic policy 145,604 bank / 0.750 league, route
  memory 162,251 / 0.922, and "22 of 32 against the recorded tape". Also
  `STRATEGY.md` line 489 and its surrounding 400-episode paired comparison
  (+16,979 coins on 347 of 400), the 32-seed tracked gate table
  (`ipsm421y`, `qrmywexm`), and the "0.1% of turns" fallback rate. The recorded
  meta-tape opponent is itself a 1.32.3 replay played open-loop, so _every_
  number involving it is void twice over.
- `RESEARCH.md` reference [14] and every primary measurement it covers: the
  400 episodes mined from `kaggriculture-episodes-2026-08-02` were played on
  v1.32.3.
- `gate.py`'s recorded rung results: v23 over v21.1 128-0 and v22 over v21.1
  124-4 on the pinned engine. Superseded by section 2 above.
- `kaito_v23_policy.py`'s author-quoted numbers, which describe the legacy
  branch. Superseded, not merely stale.
- The economic-policy characterization fixtures. Re-recorded; both engines'
  values are kept side by side in `tests/test_economic_policy.py`.

**Void — the replay corpus, for anything forward-looking. And it is not a
single-build corpus.** Measured 2026-08-08 by reading each episode's own
`configuration.townCenterSellInterval` out of the head of its JSON:

| archive                 | episodes | interval 12 | interval 24 |
| ----------------------- | -------: | ----------: | ----------: |
| 2026-07-30 … 2026-08-06 |    6,381 |       6,381 |           0 |
| 2026-08-07              |      675 |         404 |         271 |

**The change lands inside one archive, not between two.** 40% of the newest
archive is already the new engine. Any statistic pooled across a day mixes two
economies, and `learn/bands.py` modelled a single global town until this pass —
which silently mis-modelled those 271 episodes. `drain` now takes the interval
per episode and **refuses a default**, `coverage` threads it through, and
`tests/learn/test_bands.py` checks the transcription against a real episode of
_each_ build. The single-episode version of that test passed throughout the
period it was wrong.

The archives remain valid as history. What is void is using them to predict the
game we now play:

- **Corpus median seat banks 114,404 / 112,866** (`docs/research/README.md`).
  Still true of the archives; no longer a target. Two farms on 1.32.6 bank
  73k–88k against each other.
- Everything in [2026-08-08-what-wins-elo.md](2026-08-08-what-wins-elo.md) that
  is a _bank_ figure, including the bucketed rating table and the claim that
  `economic_policy`'s 161,730 is the 98.8th percentile.
- The 190 harvested routes behind `routes.play` and `baselines/best_route.py`,
  and the behaviour-cloned checkpoint, were all fitted on 1.32.3 seats.

**Not void.** The finding that ladder rating no longer tracks banked coins
(−0.043 on 2026-08-07 over 1,350 seats) is a statement about the _ladder_, not
about the engine's economy, and the engine change makes it more likely rather
than less: absolute banks fell for everyone while the order held. The
self-play negative results (advantage cancellation, fresh initialisation banking
~0, the teacher KL being load-bearing) are statements about the learning setup
and do not depend on the demand curve.

---

## 6. Recommendation — not acted on

Submission is a human decision and the autonomous guard stays off. Nothing in
this pass changed what `main.py` serves.

`main.py` serves `kaito_policy`, which is v21.1: **the weakest of the four
vendored agents, beaten 512-0 by boatlee_v14 and 512-0 by v23, and beaten
511-1 by v22.** It was already last on 1.32.3 and the engine change did not
rescue it. On the measurement above the served agent should be
`boatlee_v14_policy`, which wins 0.960 [0.951, 0.968] against the field and has
never lost a game to the agent currently shipped.

Two things to weigh before doing that, neither of which this pass resolves:

- All four vendored agents are decoded copies of published kernels. Serving one
  is a decision about the competition's rules and this project's purpose, not a
  measurement, and it is the same decision that was already live when v21.1 was
  chosen.
- Section 3's fragility applies to boatlee most of all: single table, no
  configuration branch, fitted to one engine on one day. v23 is the only
  vendored agent that adapts, and it is the one to prefer if the rules move
  again before the deadline.
