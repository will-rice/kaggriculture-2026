# What wins the ladder, measured

**Measured 2026-08-08** against the nine daily replay archives on disk,
`kaggriculture-episodes-2026-07-30.zip` through `-2026-08-07.zip`: 7,056
episodes, 14,112 seats. Every number below is from those archives or from the
public leaderboard on the date named beside it. Reproduce with
`uv run python -m kaggriculture.learn.scripts.bands`; the machine-readable
output is `docs/research/2026-08-08-what-wins-elo.json` and the code is
`src/kaggriculture/learn/bands.py`.

**The question.** The project has optimised banked coins since it started. This
asks the thing nobody had asked: does banking coins move the ladder?

> **The corpus is five economies, not one.** Every episode records the
> `kaggle-environments` build it was played on, and the nine archives span
> 1.32.2 to 1.32.6. One of those releases changed the market: 1.32.6 replaced
> the town centre's escalating demand curve with a flat rate and doubled its
> interval, cutting late-season demand roughly eightfold. At an unchanged rating
> the median seat banks **120,800 on 1.32.5 and 80,660 on 1.32.6** — and 268 of
> the 675 episodes in the 2026-08-07 archive are already on the new build. Every
> bank figure below is therefore cut by `(archive, build)`. An earlier draft of
> this document pooled them and reported a monotone negative relationship
> between bank and rating; that relationship is an artefact of the pooling and
> section 1 says so explicitly. See
> [2026-08-08-engine-1326-rebaseline.md](2026-08-08-engine-1326-rebaseline.md)
> for what the upgrade voids elsewhere.

---

## 1. The headline: the bank does not track the ladder any more

Ladder rating and final bank, correlated per archive. Per archive and not
pooled, because the rating scale inflates roughly 250 points a day and a pooled
correlation is mostly a correlation with the calendar — pooled over all nine it
reads +0.606, which is that artefact and not a finding.

Cut further by engine build, which is the version that should be quoted (cells
under 200 seats omitted):

| archive    | build  | seats |    Pearson |   Spearman | median rating | median bank |
| ---------- | ------ | ----: | ---------: | ---------: | ------------: | ----------: |
| 2026-07-30 | 1.32.2 | 1,728 | **+0.686** |     +0.704 |           688 |      28,860 |
| 2026-07-31 | 1.32.2 | 1,856 |     +0.355 |     +0.403 |         1,181 |      95,418 |
| 2026-08-01 | 1.32.2 | 1,658 |     +0.083 |     +0.088 |         1,353 |     115,880 |
| 2026-08-02 | 1.32.2 | 1,586 |     −0.035 |     −0.030 |         2,326 |     122,699 |
| 2026-08-03 | 1.32.2 |   562 |     +0.132 |     +0.132 |         2,710 |     128,979 |
| 2026-08-03 | 1.32.3 | 1,012 |     +0.175 |     +0.091 |         2,752 |     130,705 |
| 2026-08-04 | 1.32.3 |   998 |     −0.029 |     −0.035 |         2,765 |     129,559 |
| 2026-08-04 | 1.32.4 |   512 |     −0.064 |     −0.092 |         2,780 |     127,480 |
| 2026-08-05 | 1.32.4 | 1,484 |     +0.025 |     +0.016 |         2,844 |     125,855 |
| 2026-08-06 | 1.32.4 |   514 |     +0.020 |     +0.019 |         2,967 |     113,858 |
| 2026-08-06 | 1.32.5 |   852 |     +0.029 |     +0.031 |         2,984 |     117,364 |
| 2026-08-07 | 1.32.5 |   814 |     −0.016 |     −0.006 |         3,035 |     120,800 |
| 2026-08-07 | 1.32.6 |   536 | **−0.059** | **−0.031** |         3,028 |  **80,660** |

**Every cell since 2026-08-01 lies between −0.064 and +0.175.** On the economy
we now play — 2026-08-07 on 1.32.6, 536 seats — it is **−0.059** (Spearman
−0.031). This is a **null, not a negative**: banked coins carry essentially no
information about a seat's ladder rating.

The unstratified per-archive view, for comparison, and to show what the build
cut does and does not change:

| archive    | seats |    Pearson |   Spearman | median rating | median bank |
| ---------- | ----: | ---------: | ---------: | ------------: | ----------: |
| 2026-07-30 | 1,728 | **+0.686** |     +0.704 |           688 |      28,860 |
| 2026-07-31 | 1,856 |     +0.355 |     +0.403 |         1,181 |      95,418 |
| 2026-08-01 | 1,658 |     +0.083 |     +0.088 |         1,353 |     115,880 |
| 2026-08-02 | 1,586 |     −0.035 |     −0.030 |         2,326 |     122,699 |
| 2026-08-03 | 1,574 |     +0.151 |     +0.109 |         2,733 |     130,006 |
| 2026-08-04 | 1,510 |     −0.051 |     −0.062 |         2,769 |     129,152 |
| 2026-08-05 | 1,484 |     +0.025 |     +0.016 |         2,844 |     125,855 |
| 2026-08-06 | 1,366 |     +0.042 |     +0.043 |         2,976 |     115,658 |
| 2026-08-07 | 1,350 | **−0.043** | **−0.005** |         3,032 |     109,032 |

The null is the same either way, which is the good news — a null is hard to
manufacture. What the build cut overturns is the _story_ an earlier draft told
about why. **Two claims are withdrawn:**

- ~~"As the field's rating rose, its median bank fell: the field is getting
  stronger and banking less."~~ The 20,120-coin decline from 2026-08-04 to
  2026-08-07 is the engine, not the field. Held to one build the median bank
  _rose_ over that window, 117,364 to 120,800, both 1.32.5.
- ~~"The best-banking seats in the corpus are the weakest seats in it."~~ Pooled
  over the four recent archives, mean rating falls monotonically from 3,030 in
  the under-60,000 bucket to 2,828 in the over-160,000 bucket. Inside a single
  build it is flat:

| final bank      | 1.32.4 (n=2,510) | 1.32.5 (n=1,666) | 1.32.6 (n=536) |
| --------------- | ---------------: | ---------------: | -------------: |
| under 60,000    |                — |                — |          3,045 |
| 60,000–100,000  |            2,875 |            3,008 |          3,039 |
| 100,000–120,000 |            2,874 |            3,009 |          3,035 |
| 120,000–140,000 |            2,857 |            3,012 |          3,040 |
| 140,000–160,000 |            2,843 |            3,014 |              — |

Mean attributed rating, by bank bucket, by build. The 1.32.5 column moves 6
points across the entire range of banks; the 1.32.6 column moves 10. The pooled
decline was the low buckets filling with 1.32.6 episodes, which are all
late-2026-08-07 and therefore high-rated. **Banking a lot does not make a seat
weak. It says nothing about the seat at all.**

Our own submission is that null seen from outside the corpus, and it is the one
figure here that needs no stratifying: `economic_policy` banks 161,730 — the
**98.8th percentile** of the 5,710 recent seats — and scored **~1,014** on the
public leaderboard on 2026-08-04, against a top of 3,207.5.

### Why the bank says nothing: it is 98% a property of the episode

The two seats share one price-forming book, and the book turns out to set almost
the entire level. Correlating the two seats' final banks across the episodes of
one build:

| build  | episodes | corr(seat bank, opponent bank) | median seat bank |
| ------ | -------: | -----------------------------: | ---------------: |
| 1.32.3 |      499 |                         +0.792 |          129,559 |
| 1.32.4 |    1,255 |                         +0.944 |          123,424 |
| 1.32.5 |      833 |                     **+0.977** |          119,022 |
| 1.32.6 |      268 |                     **+0.976** |           80,660 |

**A seat's bank is +0.98 predicted by its opponent's.** Whatever makes an
episode a 120,000-coin episode or a 40,000-coin one — the build, the seed, which
shops the town opened, how hard the pair collectively leant on the book — both
seats get it in common. What is left for the seat itself is the margin on top,
and section 2 shows that is 2.2% of the bank.

That is the mechanism behind every null in this document, and it is the single
most actionable sentence in it: **a reward proportional to the bank is, to 98%,
rewarding the draw.**

The pairing is visible at the tails too. Within 1.32.6, 18% of seats bank under
60,000 — but **90% of those seats faced an opponent who also banked under
60,000**, against the 18% independence would predict. Episode 90775503, the
second-highest-rated episode of 2026-08-07 at avg 3,130.2, has two seats rated
3,136.9 and 3,123.5 banking 32,488 and 32,732. It is a 1.32.6 episode, so its
tiny total is mostly the new demand curve; an earlier draft cited it as proof
that strong pairs destroy value, and it is not that.

Within a single archive the episode total does still fall slightly across rating
bands — 0.977 of the day's median for a top-decile episode against 1.002 for a
low one, over 2,563 episodes — but two of those four archives straddle a build
change, so read the 2.5% as an upper bound on any "strong pairs extract less"
effect rather than as a measurement of one.

---

## 2. What the ladder does track: the head-to-head, decided by 2%

A rating is not a bank. It is an accumulation of head-to-head results, and the
head-to-heads are close. Over 426 decided episodes in the profiled sample the
**median winning margin is 2,561 coins on banks near 117,000 — 2.2%** (mean
margin 3,971, inflated by the rare blowout).

Within an episode, the higher-rated seat out-banks the lower-rated one in
**68.4%** of the corpus's 6,824 non-tied episodes, and the edge grows with the
rating gap exactly as it should:

| rating gap | episodes | higher-rated seat banks more |
| ---------- | -------: | ---------------------------: |
| 0–25       |    1,823 |                        0.540 |
| 25–50      |    1,669 |                        0.641 |
| 50–100     |    2,157 |                        0.765 |
| 100–200    |    1,066 |                        0.824 |
| 200–1,000  |      109 |                        0.798 |

So the ladder is measuring something real, and that something is _relative_
bank inside a shared market. It is not measuring how many coins you can
accumulate. (This table doubles as the check on the seat-rating attribution;
see "How the seats were rated" in section 7.)

---

## 3. Rating bands separate on nothing

480 episodes were decoded — 30 per within-day rating band per archive over the
four most recent archives — and both seats of each profiled: 918 seats, 205 in
the top band, 235 upper, 252 middle, 226 low. Within-day percentile bands
(≥p90, p65–p90, p35–p65, p10–p35), because of the daily inflation.

Nothing separates them. Ranked by standardised mean difference between the top
decile and the middle band:

| feature                       |     top |  middle | Cohen's d | Welch t |
| ----------------------------- | ------: | ------: | --------: | ------: |
| late revenue share            |   0.567 |   0.585 |     −0.26 |    −2.8 |
| unlocked tiles (season mean)  |   62.08 |   61.17 |     +0.18 |    +1.9 |
| pasture tiles (season mean)   |   12.01 |   11.79 |     +0.18 |    +1.9 |
| final bank                    | 115,172 | 119,336 |     −0.17 |    −1.8 |
| share of units sold by day 15 |   0.232 |   0.238 |     −0.16 |    −1.7 |
| crowding                      |   0.966 |   0.975 |     −0.16 |    −1.7 |
| realisation                   |   0.920 |   0.910 |     +0.13 |    +1.4 |
| denial share                  |   0.231 |   0.221 |     +0.12 |    +1.2 |
| animals held (season mean)    |   11.44 |   11.33 |     +0.10 |    +1.1 |
| sales per episode             |   149.8 |   149.3 |     +0.01 |    +0.1 |
| units per episode             |   1,167 |   1,164 |     +0.01 |    +0.1 |
| mean sale price               |   88.58 |   88.53 |     +0.00 |    +0.0 |

The largest effect in the whole table is d = 0.26. Trade volume, price, animals,
pasture and hands are all flat to three significant figures.

One thing does separate the bands, and it is the ladder itself: head-to-head
win rate runs **0.600 / 0.540 / 0.504 / 0.354** from top band to low. The bands
are real — they differ in what they achieve against the seat sitting opposite
them — while differing in no measured behaviour. Two reasons, and both matter
for how to read the rest of this document:

- The contrast is genuinely small. The profiled top band averages a rating of
  3,005 against the middle band's 2,906 — 99 points, 3.4%. The field is
  compressed and the ladder matches within it.
- An episode is a large random object. The seed decides the board, the weeds
  and which shops the town opens, and every band averages over that noise.

The second reason has a fix, and it is where the signal is.

---

## 4. What separates a winner from a loser in the same episode

Compare the two seats of _one_ episode. Same seed, same board, same shops, same
market, same 719 turns. 426 decided episodes, paired:

| feature                       |  winner |   loser | difference | paired t | tied | winner ahead |
| ----------------------------- | ------: | ------: | ---------: | -------: | ---: | -----------: |
| final bank (by construction)  | 118,605 | 114,634 |     +3,971 |    +10.6 | 0.00 |         1.00 |
| **coins denied to opponent**  |  29,993 |  28,078 | **+1,914** | **+7.7** | 0.02 |     **0.67** |
| denial share                  |   0.232 |   0.222 |     +0.010 |     +4.8 | 0.02 |         0.66 |
| **mean sale price**           |   89.64 |   86.81 |  **+2.83** | **+4.3** | 0.00 |     **0.61** |
| book price at time of sale    |   97.47 |   95.30 |      +2.17 |     +3.6 | 0.02 |         0.63 |
| share of units sold by day 15 |   0.233 |   0.241 |     −0.008 |     −3.6 | 0.04 |         0.43 |
| pasture tiles (season mean)   |   11.89 |   11.69 |     +0.200 |     +3.2 | 0.90 |         0.84 |
| animals held (season mean)    |   11.40 |   11.21 |     +0.185 |     +3.2 | 0.75 |         0.72 |
| **realisation**               |   0.917 |   0.909 | **+0.008** | **+3.0** | 0.00 |     **0.54** |
| sales per episode             |   147.4 |   151.6 |      −4.25 |     −2.9 | 0.09 |         0.56 |
| unlocked tiles (season mean)  |   61.62 |   60.83 |     +0.786 |     +2.7 | 0.91 |         0.62 |
| units per clear               |    8.08 |    7.85 |     +0.235 |     +1.6 | 0.02 |         0.44 |
| hands hired per day           |    9.43 |    9.46 |     −0.028 |     −1.6 | 0.49 |         0.44 |
| units bought                  |   499.5 |   473.9 |      +25.6 |     +1.0 | 0.13 |         0.49 |
| gross sale revenue            | 101,997 | 100,536 |     +1,462 |     +1.0 | 0.00 |         0.65 |
| **units sold**                | 1,162.2 | 1,159.9 |   **+2.3** | **+0.1** | 0.05 |         0.50 |

"Winner ahead" counts only the episodes where the two seats differ; capital
features tie often (unlocked tiles in 91% of episodes, pasture in 90%, animals
in 75%), so read their difference column rather than their share. `crowding`
and `coverage` are properties of the episode and tie in 100% of pairs, which is
the point made in section 7.

Read the gross-revenue row (+1,462, t = +1.0) against the bank row (+3,971,
t = +10.6) with section 7's coverage caveat in hand: the bank is read straight
off the observation and revenue is inferred from the ~65% of trade the shed
inference can see, so revenue is the noisier of the two by construction. What
the comparison does establish is that the winner is not simply moving more
goods.

**The winner sells the same number of units and gets more for them.** Units
sold is the flattest row in the table (t = +0.1). What the winner does instead:

1. Sells them in **fewer, larger clears** (147.4 vs 151.6 clears, t = −2.9, at
   equal volume).
2. Realises **+2.83 coins per unit**, +3.3%. Decomposed: +2.17 of that is
   selling what the book prices higher, at the moment it prices it higher, and
   the remaining ~0.7 is **realisation** — price obtained over the book price
   standing at the instant of the clear, which is mix-independent and still
   favours the winner (0.917 vs 0.909, t = +3.0).
3. Sells **later**, not earlier: 23.3% of its units are gone by day 15 against
   the loser's 24.1% (t = −3.6).

Production capital is a real but second-order edge: +0.79 unlocked tiles, +0.20
pasture and +0.19 animals averaged over the season, all significant and all
tiny — and all tied in 75–91% of episodes, so they decide the minority of games
where the two farms are built differently at all. Hands, plants and purchases do
not discriminate at all.

---

## 5. Opponent denial: yes, and it is the largest paired effect

**The measure.** The two farms sell into one book, and the engine's
`market_price` is a pure function of that book's inventory, so a seat's effect
on its opponent's prices is computable rather than inferable. For every unit the
opponent cleared, reprice it against the inventory the book _would_ have held if
this seat had never traded, and difference the two. The seat's contribution is
its own completed sales less its own market purchases, accumulated over the
season.

**Why the subtraction is exact.** The market's only other participant is the
town, whose shops and centre take a fixed quantity out of the book on a fixed
schedule that never reads the inventory level (`_town_consume`). That makes the
book exactly additive in the two farms' trading. Transcribed into
`bands.drain`, it reproduces the engine on every one of the 1,000+ per-good
turns in a real episode where neither farm traded —
`test_the_drain_transcription_matches_the_engine_on_quiet_turns` asserts it.

**The result.** Denial is the strongest paired discriminator in the study apart
from the bank itself: the winner takes **1,914 more coins out of its opponent's
prices than its opponent takes out of its own** (t = +7.73), and is ahead on it
in 67.1% of the 426 decided episodes. The median winning margin is 2,561 coins,
so **the denial edge is half the median margin** (median ratio **0.509**).

Every seat denies a lot: the mean seat removes **29,141 coins, 22.7%** of what
its opponent would otherwise have earned. This is not a rare tactic, it is the
ambient condition of the market — and it is why section 1's two seats move
together at +0.98. Each is taking about a fifth out of the other's prices, so
whatever one of them does to the book arrives on the other's balance sheet.

**How confident, and in what.** Confident in the mechanism and the sign;
much less confident in the interpretation.

- The arithmetic is exact given the inputs, and the paired design cancels every
  episode-level confound including the inference's own blind spot (below).
- It is a **partial-equilibrium attribution**, not a counterfactual game. It
  removes the sales from the book without asking what else that seat would have
  done, so it prices impact, not the game without the player.
- **It does not show intent, and probably is not a separate skill.** Denial and
  realisation are the same action seen from two sides: selling a high-value good
  into a book that will pay for it both earns more and costs the opponent more.
  `crowding` — cosine similarity of the two seats' sold-good mixes — is 0.97 in
  every band, so nobody is steering away from the opponent's goods, and the top
  decile does not crowd more than the middle (d = −0.16).
- **The size of a seat's denial edge barely predicts the size of its win**
  (r = **0.051** across the same 426 episodes). Which seat denies more is
  reliable; how much more is not a dose-response, so "deny harder" is not a
  supported instruction.

So: **there is denial, it decides about half the typical margin, and it looks
like a by-product of good execution rather than a distinct strategy.** That
distinction may not matter for what we optimise — a reward that pays for
realised price captures both.

---

## 6. Timing: the divergence is in the last third

Mean bank per day, top band against middle band, over the profiled sample:

| day | 5   | 10    | 15     | 20     | 25     | 29      |
| --- | --- | ----- | ------ | ------ | ------ | ------- |
| top | 997 | 7,596 | 20,292 | 48,894 | 88,037 | 115,172 |
| mid | 986 | 7,796 | 21,426 | 49,240 | 90,838 | 119,336 |

The bands are identical until day 9 (top ahead by 10 coins on day 9), and the
gap then widens monotonically to −4,163 by day 29 — the top band ending _below_
the middle, which is section 1's null restated over time and not a claim that
the top band banks less. The sample straddles two builds and the band means are
one standard error apart. Nothing happens early, and the build-out is the same
everywhere:

| day                    |    5 |   10 |   15 |   29 |
| ---------------------- | ---: | ---: | ---: | ---: |
| unlocked tiles, top    | 26.7 | 65.5 | 77.0 | 77.0 |
| unlocked tiles, middle | 25.8 | 63.7 | 75.9 | 75.9 |
| animals, top           |  5.6 | 11.1 | 14.3 | 14.0 |
| animals, middle        |  5.6 | 11.3 | 14.1 | 13.8 |
| hands per day, top     |  3.7 |  9.8 | 11.5 |  9.0 |
| hands per day, middle  |  3.4 |  9.6 | 11.6 |  9.0 |

Land buying stops by day 15 in both bands — at a mean 77.0 unlocked tiles of a
possible 100 for the top band and 75.9 for the middle, so neither band routinely
buys the fourth quadrant — pasture and animals top out at the same point, and
hiring is indistinguishable throughout. The paired comparison agrees
on where the action is: the winner's edge is in _late_ selling (23.3% of units
gone by day 15 against the loser's 24.1%), and 57–59% of all revenue arrives
after day 20 in every band.

The practical reading: **days 0–9 are a solved opening that every band plays the
same, and everything that separates seats happens in the trading of days
10–29.**

---

## 7. What could be an artefact

- **Rating is a property of the player, not the episode.** A 3,100-rated player
  having a bad game still counts as a top-decile seat. This adds noise to the
  band comparison and is one reason section 3 finds nothing; it does not touch
  section 4, which never uses rating.
- **`min_score` is the weaker of two players.** Section 1's per-seat ratings
  come from an attribution (below), not from the manifest directly.
- **The archives are dated and the meta moves.** Section 1's per-day table is
  the evidence that this matters more than anything else here: the same
  measurement returns +0.686 and −0.043 eight days apart. Every deep result is
  from 2026-08-04 to 2026-08-07 only. It may not hold next week.
- **And the engine moves underneath the meta.** The deep sample was drawn before
  the build cut existed, so it is 1.32.3 to 1.32.6 pooled, with roughly 40% of
  its 2026-08-07 episodes on the new economy. Sections 4 to 6 are paired within
  an episode, and a build is a property of the episode, so it cannot create a
  winner/loser difference — but it does mean those numbers average two market
  regimes. The one figure it corrupts outright is `coverage` for the 1.32.6
  episodes, whose town drain `bands.drain` deliberately models as 1.32.3;
  `denied` is unaffected, because `impact` reads the recorded inventory and
  never the drain. Redrawing the deep sample within one build is the obvious
  next measurement and was not done here.
- **The sale inference misses about a third of real trade.** `sale_metrics`
  reads a completed sale off the bank rising while shed stock falls, so a turn
  that sells and spends more than it sold for is invisible. Measured against the
  book identity, the inference accounts for a mean **0.65** of the episode's
  true net book movement (`coverage`, roughly equal in every band: 0.649 top,
  0.662 middle). Counts of sales, units and denial are therefore all
  _understated_, and the same inference is running against the training arms in
  `learn/scripts/evaluate.py`. Because coverage is a property of the episode, it
  applies equally to both seats — it ties in 100% of the paired comparisons —
  and **cannot** manufacture the section 4 differences.
- **The price walk within a turn is approximate.** The engine interleaves the
  two seats unit by unit and the observation does not record the order, so
  prices are walked from the inventory standing at the start of the turn.
  Modelled proceeds come to **1.037–1.044×** actual, band by band, so the
  approximation costs about 4% and costs it equally everywhere.
- **The paired comparison conditions on the outcome, which selects for luck as
  well as for skill.** The seed is shared but the two farms are not: weeds spawn
  per farm off the same stream, so one seat can be dealt a worse board. Any
  feature merely correlated with a lucky draw will read as "the winner does
  it". This is the likeliest explanation for the capital rows — +0.79 tiles,
  +0.20 pasture, +0.19 animals, all tied in most episodes — and the least
  likely explanation for realisation, which is a ratio against the seat's own
  book price at the moment it sells.
- **The band sample is 30 episodes per band per archive**, chosen with a fixed
  seed. The top band is a decile and so is smaller than the others; section 3's
  null result is at n = 205 top seats against 252 middle, which resolves
  d ≈ 0.26 at 80% power. Effects smaller than that would not have been seen —
  but effects that small are not what the project needs.
- **Shops are not a seat feature.** The town is shared and its unlock schedule
  is seeded, so both seats always face the same shop set. "Shops unlocked" was
  measured and cannot discriminate seats by construction.

### How the seats were rated

The manifest carries `avg_score` and `min_score`, so an episode's two ratings
are an unordered pair; the episode's JSON carries its two team names in seat
order. `bands.attribute` ties them together by the one fact that constrains
them — a player carries one rating across every episode it plays that day — and
converges in four passes over all 7,056 episodes.

It is checkable without being trusted. Under a random assignment the
higher-rated seat would out-bank the lower-rated one in half of all episodes at
every rating gap. It does so in 68.4% of them, rising monotonically from 0.540
at a gap under 25 points to 0.824 at a gap of 100–200 (section 2's table).
Noise cannot produce an ordering signal that strengthens with the size of the
thing being ordered.

---

## 8. What to do about it

1. **Stop optimising banked coins.** Bank and rating have been uncorrelated in
   every archive-and-build cell since 2026-08-01, and −0.059 over the 536 seats
   of the economy we now play. The current 161,730-coin `economic_policy` is a
   98.8th-percentile banker and a 1,014-rated agent, and those two facts are not
   in tension — they are the finding.
2. **Optimise realised price, per unit.** The winner and loser of an episode
   sell the same number of units; the winner gets 3.3% more for each. That is
   the entire margin. `sales.sale_metrics` already computes realisation and
   `learn/scripts/evaluate.py` already logs it — it should be the objective, not
   a diagnostic.
3. **A bank-proportional reward spends 98% of its gradient on nothing.**
   Episodes are decided by 2.2% of the bank, and both seats collect the other
   98% regardless. This is _not_ an argument for going back to a difference
   reward — [the self-play findings](2026-08-07-self-play-rl-findings.md) showed
   that cancels the advantage — it is an argument for a reward whose scale is
   the thing that varies, which is the per-unit price in point 2.
4. **Do not build a denial mechanic.** Denial is real and worth half the median
   margin, but it is what selling well already does. A separate term for it
   would be paying twice for one behaviour.
5. **Leave the opening alone.** Days 0–9 are identical across all four bands.
   Whatever the clone learned there is not what is costing us.

The one-sentence version: **this game is not won by banking more coins, it is
won by getting a few percent more for the same goods in a market both players
are crushing.**
