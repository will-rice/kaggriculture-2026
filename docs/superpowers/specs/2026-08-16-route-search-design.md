# Route search: play the meta the field plays, but search it

**Design, 2026-08-16.** Engine `kaggle-environments` 1.32.7. Supersedes the
tensor-native strategy search sketched earlier the same day, which was abandoned
before implementation.

## 1. Why this and not something else

Our best agent sits at a true rating of ~1,500 against a field top of 3,260.
Two byte-identical copies submitted this morning converged to 1,505 and 1,457
within fifteen hours, and the eight-day-old copy of the same agent reads 1,536 —
so 1,500 is the agent's level, not a bad draw, and path dependence is not hiding
a better result. Over 418 scored episodes it wins 29.2%.

Three findings decide what to build.

**The field does not learn.** 162 public kernels contain no torch, no weights,
and no model of any kind. Across 85 competition threads, zero mention neural
networks or tree search; the two that mention PPO are people reporting failure
at 5k–20k banked against rule-based agents' 150k.

**The field submits search results, searched by hand.** boatlee's V16-RC5
decodes to `_ACTIONS`: a literal 720-turn action tape with market orders on 282
turns. Rayk Kretzschmar's kernel decodes to a distilled trajectory of someone
else's episode. Kaito Fukami's v27 — the one genuine agent — is a fixed route
plus runtime reordering of the SELL slots the route already contains, and its
author describes his method as "build challengers → test multiple teams and both
seats → reject most candidates → freeze the winner". That is a search loop run
manually at perhaps tens of candidates a week.

This also explains the strangest number in the episode mining: across 45 top
episodes on 1.32.7, crew size and hire count were **identical in both seats of
every episode** (paired difference 0.00 in 0/45). The field is not converged by
optimisation. Much of it is replaying the same recordings.

**The field misprices engine 1.32.7.** The 08-15 balance change put carrot,
tomato and egg on a new `hinge` curve below `I0`, and carrot's `below_target`
went 0.20 → 1.00. Kaito v27 carries a hardcoded pre-1.32.7 `_MARKET_PARAMS` and
never reads `marketParams` from the observation, so its price impact scoring
understates scarce tomato by 3.6× at −450 and 7.8× at −642. The two tape agents
cannot respond at all. Our own `economic_policy` had the identical defect until
it was fixed this morning (`c45e649`).

What that leaves on the table, measured over 37 top episodes on 1.32.7:

| product | town drains/season | median min inventory | turns below `I0` |
| ------- | -----------------: | -------------------: | ---------------: |
| WHEAT   |              1,361 |                 −605 |          719/719 |
| TOMATO  |                264 |                 −264 |          719/719 |
| CARROT  |                268 |                 −256 |          719/719 |
| EGG     |                246 |                 −246 |          719/719 |
| MELON   |                 28 |                  −11 |              256 |

Carrot, tomato and egg are below `I0` for every turn of every game, and the
entire field sells 15 units into that for 721 coins. Selling the median game's
tomato drain is worth roughly 20,000–29,000 depending on whether it is metered
or dumped; carrot and egg add ~12k and ~14k. The field's median bank is 87,676
and the median winner beats the median loser by 4,733.

Production is not the obstacle: tomato is the only crop with `ongoing: True` and
`interval: 1`, yielding 4 units a day from day 8, so three tiles cover the whole
264-unit drain.

## 2. Constraints that shape the architecture

Read from a real episode's configuration and from our own packaging notes:

- `actTimeout` **1 second per turn**; `runTimeout` 1,200 s for the episode; a
  **60 s overage pool for the whole game**, which the top agent never touched.
- **The submission cannot carry torch.** Importing it alone costs 10.7 s of the
  60 s pool, and `scripts/package.py` excludes the `learn` package specifically
  to make that impossible.
- A full-season rollout in the reference engine costs ~4 s, so inference-time
  search could afford roughly a quarter of one rollout per turn.
- Offline, the batched simulator is graph-capturable at ~100 games/second when
  no host synchronisation occurs in the loop (17.6 s per 1,024-game season-batch,
  42.0 s per 4,096). A scripted Python opponent synchronises per row per turn and
  forfeits this entirely.

So: **no search at inference; unlimited search offline; the deliverable is a
route.** Which is what the competition already rewards.

## 3. Architecture

### A. Sell metering

**This section was rewritten after its original premise was measured false.** It
first proposed reordering "the SELL slots the route already contains" by a
corrected price, and claimed that captured the hinge. Measured on three seeds,
the agent we serve sells **zero** tomato, zero carrot and zero egg, and buys
neither their seed nor a goose. There are no hinge SELL slots to reorder, so a
corrected price model captures nothing by itself. The hinge money requires
production and therefore belongs to the route search in C, not here.

What the same measurement did find is a defect worth more than the mispricing,
and it needs no new crop. Against the town's per-season demand:

| product    | town drains | we sell | ratio |
| ---------- | ----------: | ------: | ----: |
| WHEAT      |       1,368 |     305 |  0.22 |
| STRAWBERRY |         367 |     320 |  0.87 |
| MILK       |         268 |     214 |  0.80 |
| WOOL       |         107 |     148 |  1.38 |
| MELON      |          28 |     144 |  5.14 |
| FERTILIZER |           0 |     212 |     ∞ |

We sell melon at **5.1x** the town's appetite and dump fertilizer into a market
with **no drain at all**, while serving 22% of wheat demand. Melon's above-`I0`
curve is `sq` at target 3.60 — the harshest in the table — so oversupplying it
collapses precisely our highest-base product.

**A IS CANCELLED. It was built as an experiment and it lost 0-24.**

The obvious reading of that table is that we should stop selling into markets we
have crashed. Metering each SELL to keep its marginal price above half of base,
with everything else in the agent untouched, banks **41,333 against the
unmodified agent's 96,528** over 24 seeded games — zero wins.

The reason is that a grown unit has no alternative use. Selling a melon at a
collapsed 78 still beats holding it at 0, because the town drains only 28 melon
a season so the price never recovers; and early sales fund the hires and seeds
that compound for the rest of the season, so withholding them starves the whole
farm. **The field's apparent over-dumping is correct play.**

The misallocation is real but it is entirely upstream: the error is _planting_
72 melon for a town that wants 28, not _selling_ what has already grown. That is
a production decision and it belongs to C.

Nothing ships before B and C. The two things this section leaves behind are a
correct price model — already landed in `c45e649` — and the knowledge that
sell-side tinkering is a dead end, which is worth more than another variation of
it would have been.

### B. Tape arena

Offline evaluation substrate. A route is pre-encoded into action tensors **once**
and replayed across a batch of seeds with no host synchronisation, which is what
makes it graph-capturable and fast. Legality masking is the engine's own: an
action the board does not permit is masked, and a market order larger than our
inventory partially fills, exactly as `_commit_unit` already does.

Opponents are a league of decoded agents — boatlee V16-RC5's tape, Kaito v27,
our boatlee v14, `economic_policy`, and the current incumbent — played
seat-swapped. Fitness is **win rate**, the competition's own objective, not bank:
a seat's bank correlates +0.98 with its opponent's and is uncorrelated with
rating.

The league is deliberately not just the strongest opponent. Kaito's own warning:
"Optimizing only against the latest Top-30 can lose to older meta generations
still active on the leaderboard."

**Interface.** `evaluate(route, opponents, seeds) -> per-opponent win rates`.

### C. Route search

Hill-climbing over routes. Seed from the strongest routes we hold, mutate,
evaluate in the arena, accept when the candidate beats the incumbent by more
than sampling noise, freeze and repeat.

**A mutation is one of four edits, confined to a single day (24 turns)** so that
a candidate differs from its parent in a way the fitness difference can be
attributed to:

- retime a market order — move a SELL or BUY to another turn within the day;
- resize one — change its quantity;
- retarget one — change the product or crop it names;
- replace a unit's op on one turn with another legal op (including `PASS`).

Compound mutations are not used: with a noisy binary fitness, a candidate that
differs in several places tells us nothing about which edit paid.

**Tape fragility is handled by the fitness function rather than by code.** A tape
recorded on one seed can misalign on another, which is why `routes/play.py`
needed realignment logic. Because every candidate is scored across many seeds, the
search selects for routes that survive reseeding: robustness is selected, not
implemented.

## 4. Order of work

**B then C.** A was to have shipped first and was cancelled on measurement, so
there is no early submission to be had: the first shippable artefact is a
searched route, which requires the arena.

The edge C targets is still time-limited — the field will re-record its tapes
against 1.32.7 and the mispricing will close — so the arena is built for
throughput, not for generality. Anything in B that is not needed to score a
route against a league of opponents is out of scope.

## 5. Acceptance gates

1. **Disjoint seeds.** Search seeds and evaluation seeds never overlap, and
   sample sizes are fixed before the result is seen.
2. **Seat-swapped against the whole league**, not against one opponent.
3. **Against a copy of itself.** If the edge disappears in self-play, what we
   have is a raid on today's field rather than a strategy. That is still worth
   submitting, because the ladder is played against today's field — but it decays,
   and nothing further should be built on it.
4. **Final gate on the reference engine**, not the simulator. The simulator is
   fidelity-proven and that proof is maintained, but what we submit is measured
   on the thing that scores it.

A change ships when it beats the agent we currently serve, seat-swapped over
**128 held-out reference-engine games**, with the win rate's 95% interval
excluding 0.5. At 128 games a win rate has a standard error of ~4.4%, so that
resolves a real edge of about ten points and refuses to ship a coin flip; 48
games — what our current harness runs by default — cannot. Both numbers are
fixed here rather than chosen after seeing a result.

Ladder feedback then costs ~15 hours per verdict: a fresh submission plays ~5.2
episodes/hour and its rating converges within ~80 episodes. That is the real
iteration clock, not implementation time.

## 6. Non-goals

- **Inference-time search.** Killed by 1 s/turn with no torch.
- **A tensor-native strategy skeleton.** The route search subsumes it and starts
  from a strong point rather than from nothing.
- **Further RL work.** Not because it cannot work — Halite 2020 was won by PPO
  and our Toad reproduction descends from a Lux AI winner — but because ten arms
  here have produced a win rate of 0.000 and the throughput objection those
  threads raise is not the binding constraint on _this_ plan.

## 7. Risks

**The edge is time-limited and self-limiting.** Our own supply fills the hole we
sell into, which the metering in A addresses; and the field will fix its price
tables, which nothing addresses. Gate 3 measures how much of the edge is
adaptation-proof.

**A searched route may overfit the league.** Mitigated by seat-swapping, a
league spanning meta generations, and disjoint seeds — not eliminated.

**The engine may change again.** The host has said 1.32.7 is the last balance
change excepting game-breaking bugs. Two changes have already voided
measurements; the reference tripwire in `tests/sim` is what catches a third.
