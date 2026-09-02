# What the top players build, and what we build

**Measured 2026-09-02** against the eight daily replay archives
`kaggriculture-episodes-2026-08-24.zip` through `-2026-08-31.zip`: 5,435
episodes, 10,870 seats, **every one of them decoded** (no sampling), plus 1,536
seasons our own agents played locally. The question is the diff that
[2026-08-08-what-wins-elo.md](2026-08-08-what-wins-elo.md) did not ask: not what
wins, but **what the top-rated seats build and when, against what we build**.

> **One engine, and it is the one we run.** Every episode records its
> `module_version`; all 5,435 in this window report **1.32.7**, and all 5,435
> report `townCenterSellInterval` 24. The installed `kaggle-environments` is
> also 1.32.7, so our seasons were generated on the same economy the archives
> were played on. No figure here is pooled across builds because there is only
> one build in the window. The prior study's build cut still applies to its own
> corpus and to anything quoted from before 2026-08-24.

**The headline is not the one the question expected.** The published field is
not a set of players with distinct farms. It is a sequence of _published-kernel
waves_: a strong kernel appears, most of the field forks it, everyone plays the
identical 30-day capital script, and a few weeks later another kernel replaces
it. **The build we serve is the current wave**, it is on 58% of the seats in
the newest archive, and it beats every other build in the corpus 0.700 of the
time. The top-rated _players_ build something else, and lose to it head to
head.

---

## Method

**Build fingerprint.** For each seat, the four 30-day series that a capital
plan is: animals of each species held at the close of each day, pasture tiles,
and unlocked quadrants. Two seats share a fingerprint when all 120 numbers
agree. A scripted agent reproduces its fingerprint exactly; an agent that
responds to the board does not.

**Per-seat profile.** Read from `steps[turn][seat]`: per-day herd by species,
crop tiles by type, pasture/coop/plant counts, unlocked quadrants, peak hands,
bank, shed level; and every `farmer`, `hands` and `market` verb the seat issued,
counted per day and per third of the season. The same extractor runs on an
archived episode and on `env.toJSON()`, so the corpus and our seasons are
measured by one piece of code.

**Ratings.** The manifest's unordered `(avg_score, min_score)` pair is resolved
onto seats by the fixed point in `kaggriculture.learn.bands.attribute`
(re-implemented, converges in 6 passes over these 8 archives). Bands are
**within-day** percentiles: top ≥p90, upper p65–p90, middle p35–p65, low
p10–p35.

**Our seasons.** `main.py` (which serves `kaito_v56_policy.kaggle_agent_v56`)
and `/data/kaggriculture/agents/tuned_v58_h7_ratio225.py`, played on 1.32.7
across three pairings (each vs `meta_tape`, and against each other), both seat
orders, seeds 1–128: **512 seats of v56 and 512 of tuned v58**. Plus 768 mirror
episodes (v56 vs v56, v58 vs v58, seeds 1–384) used for one controlled test in
section 6. Our banks come out at 91,215 (v56) and 91,494 (v58t) against a
corpus median of 87,012 (85,921 in the newest archive), so the local economy is the corpus economy.

---

## 1. The field is kernel waves, and we are riding the current one

Our two agents produce **four** distinct fingerprints across 1,024 seats — one
of them covers 78–80% of seasons. Those same four fingerprints appear on
**1,662 of the corpus's 10,870 seats**, played by other people. Their arrival
is abrupt:

| archive    | seats | seats on our build | share |
| ---------- | ----: | -----------------: | ----: |
| 2026-08-24 | 1,384 |                  0 | 0.000 |
| 2026-08-25 | 1,376 |                  0 | 0.000 |
| 2026-08-26 | 1,374 |                  1 | 0.001 |
| 2026-08-27 | 1,392 |                 11 | 0.008 |
| 2026-08-28 | 1,352 |                 11 | 0.008 |
| 2026-08-29 | 1,334 |                293 | 0.220 |
| 2026-08-30 | 1,328 |                570 | 0.429 |
| 2026-08-31 | 1,330 |                776 | 0.583 |

The wave it displaced is just as visible. The single most common fingerprint in
the corpus (2,704 seats; herd `0,0,0,0,1,1,2,6,8,9,11…` cows and a flat 4
sheep) holds 686 seats on 2026-08-24 and 39 on 2026-08-29, and none after.
1,654 distinct fingerprints exist in the corpus, and the eight largest account
for 6,364 of them — **59% of the field plays one of eight scripts**.

**Consequence for the question as posed.** "What do the top players build" has
no stable answer over a week, because the answer is "whatever kernel was
published most recently". Any build-shaped instruction derived from an archive
older than a few days is a description of a wave that has already broken.

## 2. Our build wins the head-to-head it is in

Episodes where one seat plays our exact build script and the other does not,
decided on bank:

| opponent                             | episodes | our build's win rate |    95% |
| ------------------------------------ | -------: | -------------------: | -----: |
| all other builds                     |      860 |            **0.700** | ±0.031 |
| opponent in the top decile           |       64 |                0.672 | ±0.122 |
| opponent among the six varying teams |      105 |                0.676 | ±0.096 |
| everyone else                        |      755 |                0.703 | ±0.036 |

For contrast, the six teams that vary their build every game (section 3) beat
the rest of the field only **0.568** (n = 869, ±0.033) — a smaller edge than
our build has over them.

The old top is sliding while this happens. Mean attributed rating by archive:

| team          | 08-24 | 08-26 | 08-28 | 08-30 | 08-31 |
| ------------- | ----: | ----: | ----: | ----: | ----: |
| Crop Dusta    | 3,035 | 3,077 | 2,903 | 2,979 | 2,720 |
| Ryo Hasegawa  | 2,929 | 2,960 | 2,856 | 2,780 | 2,691 |
| Subramanya N  | 2,998 | 2,992 | 2,980 | 2,855 | 2,832 |
| Milan Leonard |     — | 2,883 | 2,914 | 3,032 | 3,026 |

Our build's corpus copies rate 2,760 on average against 2,799 for everyone
else, which reads as a deficit and almost certainly is not one: a fork that
entered the ladder on 08-29 has had three days to climb, and it is winning 70%
of its games while it climbs. That is the same shape as our own submissions
reaching 2,340 within hours of a 2026-09-01 upload.

## 3. What the top-rated players actually build: a different farm every game

This is the one large, clean difference between the top decile and us, and it
is not about cows.

**Distinct build fingerprints per seat** — 1.0 means every season is a
different farm, 0.0 means one script forever:

| group               | seats | distinct builds / seat | on the coarse tuple |
| ------------------- | ----: | ---------------------: | ------------------: |
| top band (≥p90)     | 1,091 |              **0.732** |               0.527 |
| upper band          | 2,715 |                  0.203 |               0.119 |
| middle band         | 3,258 |                  0.114 |               0.067 |
| low band            | 2,715 |                  0.095 |               0.067 |
| **ours, v56**       |   512 |              **0.006** |               0.006 |
| **ours, tuned v58** |   512 |              **0.008** |               0.008 |

("Coarse tuple" is final cows, sheep, geese, pasture, quadrants and the day the
third quadrant was bought — a much blunter key, and the ordering survives it.)

Per team, over the 75 teams with ≥50 seats, **corr(distinct builds per seat,
mean rating) = +0.667** (+0.583 on the coarse tuple). At the extremes: Crop
Dusta 424 distinct builds in 433 seats (rating 2,976), Subramanya N 221 in 230
(2,941), Ryo Hasegawa 279 in 437 (2,900) — against Hiroyasu Okuno 0.035
(2,725) and our 0.006.

**This is the strongest descriptive difference in the study and the weakest
causal one.** The same six teams lose to our fixed script 0.676 of the time
(n = 105). So build variety tracks rating across teams without beating a script
in the market. Read it as: the players who got to the top of this ladder wrote
agents that decide their farm per game, and those agents are currently being
out-scored by a good fixed script. Whether variety is what earned them the
rating, or merely what their kind of agent looks like, this corpus cannot say.

## 4. Where our build sits on every axis measured

Band means against ours. Every cell is ≥512 seats.

| feature                            |    top | middle |    low |  ours v56 | ours v58t |
| ---------------------------------- | -----: | -----: | -----: | --------: | --------: |
| seats                              |  1,091 |  3,258 |  2,715 |       512 |       512 |
| final bank                         | 95,990 | 87,337 | 86,039 |    91,215 |    91,494 |
| cows at day 29                     |   7.01 |   8.50 |   8.59 |      8.33 |      8.49 |
| sheep at day 29                    |   5.20 |   5.35 |   5.19 |      6.17 |      5.72 |
| **geese at day 29**                |   0.61 |   0.05 |   0.06 |  **0.00** |  **0.00** |
| coops at day 29                    |   0.86 |   0.46 |   0.48 |      1.77 |      1.73 |
| herd at day 29                     |  12.82 |  13.90 |  13.84 |     14.50 |     14.20 |
| pasture tiles at day 29            |  14.56 |  14.44 |  14.39 |     14.67 |     14.29 |
| planted tiles (season mean)        |  42.45 |  42.86 |  42.69 |     43.91 |     44.07 |
| **day the 3rd quadrant is bought** |   9.65 |  10.53 |  10.60 | **11.00** | **11.00** |
| day the 2nd quadrant is bought     |   5.58 |   6.01 |   6.02 |      6.00 |      6.00 |
| final quadrants                    |   3.01 |   3.01 |   3.00 |      3.00 |      3.00 |
| peak hands                         |  12.09 |  13.22 |  13.28 |     12.00 |     12.00 |
| hires per season                   |    290 |    281 |    282 |       287 |       287 |
| sell orders                        |    278 |    409 |    419 |       294 |       293 |
| units sold                         |  1,877 |  2,065 |  2,116 |     1,909 |     1,764 |
| units bought from the book         |    688 |    432 |    396 |       637 |       492 |
| share of units sold by day 15      |  0.263 |  0.207 |  0.204 |     0.248 |     0.251 |
| shed level (season mean)           |  17.58 |  19.46 |  18.97 |     22.38 |     22.51 |

On most of these we are on the top band's side of the middle: fewer and larger
sell orders, more buying out of the book, more early selling, fewer peak hands,
more planted area. The places we are not:

**Land, which nobody contests and everybody rations.** Every band buys exactly
2.0 extra quadrants — `BUY_LAND` appears 2.006 times in the top band and 2.005
in the middle — and **nobody buys the fourth**. What differs is when the third
arrives. Cumulative share having bought it:

| by day | top (n=1,091) | upper (2,715) | middle (3,258) | ours (1,024) |
| ------ | ------------: | ------------: | -------------: | -----------: |
| 8      |     **0.263** |         0.019 |          0.007 |        0.000 |
| 9      |         0.387 |         0.071 |          0.018 |        0.000 |
| 10     |         0.698 |         0.448 |          0.443 |        0.000 |
| 11     |         1.000 |         0.999 |          0.999 |    **1.000** |

Our build takes it on day 11 in **1,024 of 1,024** local seasons and in
**1,662 of 1,662** corpus copies. The top decile has it by day 10 in 70% of
seats and by day 8 in 26%. We are last in the field on this axis, by
construction, and it is one constant in a vendored kernel.

**Geese, and the coops we build for them and never fill.** Across **2,686
seats** of our build — 1,024 local and 1,662 in the corpus — the number that
ever held a goose is **zero**. Meanwhile 5.2% of the other 9,208 corpus seats
hold one, and the rate rises monotonically with rating: 15.9% of top-band seats
end the season with a goose against 1.9% of the middle band. Among the
individual leaders it is a standing feature, not an accident: Crop Dusta 1.19
geese per season, Subramanya N 1.45, tetsuya 1.43.

And we build the housing anyway. Our seats issue `BUILD_COOP` 2.75 times a
season (1.13 from hands, 1.62 from the farmer) and finish with 1.75 coop tiles
standing; 96% of our seasons end with at least one **empty** coop. The engine
charges nothing for `BUILD_COOP` beyond the unit-turn and the tile, so this is
cheap waste rather than expensive waste — but it is waste with a ready
interpretation: the kernel has the goose branch and never takes it. On the
engine's own table the goose is the fastest-paying animal in the game — 300
coins against the cow's 400 and the sheep's 500, first yield on day 4 against
day 8 and day 6, and a yield **every** day against every second and every
third.

## 5. Why the within-episode paired test cannot arbitrate any of this

The obvious way to decide whether "more sheep" or "earlier third quadrant"
causes wins is the prior study's design: compare the two seats of one episode,
which shares the seed, the board, the town and the market. Run over all 5,330
decided episodes it produces a confident-looking table. Split the window in
half and it falls apart:

| feature                 | early (08-24…28, n=3,352) |       | late (08-29…31, n=1,978) |       |
| ----------------------- | ------------------------: | ----: | -----------------------: | ----: |
|                         |                     w − l |     t |                    w − l |     t |
| share sold by day 15    |                   +0.0179 | +13.2 |                  +0.0008 |  +0.7 |
| planted tiles (mean)    |                   +0.7247 | +12.5 |                  −0.0720 |  −1.4 |
| units bought from book  |                    +129.7 |  +5.4 |                    +13.3 |  +1.0 |
| geese at day 29         |                   +0.0710 |  +4.0 |                  −0.0076 |  −0.3 |
| day 3rd quadrant bought |                   −0.1447 |  −4.7 |                  +0.0379 |  +1.9 |
| sheep at day 29         |                   +0.1334 |  +4.0 |                  +0.8261 | +15.5 |
| pasture at day 29       |                   −0.0364 |  −1.0 |                  +0.4727 |  +9.4 |
| herd at day 29          |                   −0.0973 |  −1.9 |                  +0.6547 | +10.1 |

The late column is not measuring a lever. Sheep, pasture and herd size are
exactly the markers that separate our build (5.74 sheep, 14.12 pasture, 14.07
herd, in the corpus copies) from the recent non-lineage field (5.45, 13.77,
12.82). **When most seats play one of a handful of scripts, "what the winner
had more of" is "which script the winner ran".** The paired design cancels the
seed and the market, as advertised; it cannot cancel the confound between a
build feature and the whole kernel that carries it.

Only five features hold their sign across both halves — cows (−), sheep (+),
seeds bought (+), peak hands (−), coops (+) — and in the late half those five
are precisely the lineage's signature. Of the differences in section 4, the
only one where we sit on the _wrong_ side of a sign that holds in both halves
is **cows**: winners hold 0.30 (early) and 0.16 (late) fewer than losers, the
top band holds 7.01, and we hold 8.33.

## 6. A herd-composition test that came back negative

Our kernel does not always build the same herd: 78–80% of seasons end 9 cows
and 5 sheep, the rest 6/10, 6/11 or 7/6. In the corpus the sheep-heavier
variants out-bank the cow-heavier one 69–31 in 100 decided episodes where two
copies of the build meet on different variants. That looked like a free win.

It is not a farm-level choice. In **768 mirror episodes** (v56 vs v56 and v58
vs v58, seeds 1–384) the two seats took the **same** variant in **768 of 768**.
The variant is a property of the episode's economy, which both farms see, not a
branch either farm picks — so comparing variants compares episodes, which is
the trap section 1 of the prior study documented for banked coins. The corpus
69–31 is also confounded on the other side: the two variants are played by
disjoint sets of teams (9/5 by rian, OceanMix, gogogo; 7/6 by yukino, Umataro
Tenma, zhyphirus, Yusuke Hayashi), so it is a comparison of forks, not of
herds. **Withdrawn.**

## 7. What to do about it

1. **Move the third-quadrant purchase earlier and measure it.** It is a
   constant in a vendored kernel, we are the latest seat in the field on it
   (day 11, 1,024/1,024 seasons), and the top decile has it by day 10 in 70% of
   seats. The paired evidence for the direction is real in the early half and
   absent in the late half, so this is a hypothesis with a cheap experiment
   attached, not a known win. Test it as a gate against v58, not by looking at
   the bank.
2. **Either stock the coops or stop building them.** Zero geese in 2,686 seats
   of our build while 15.9% of top-band seats hold one, with the coop already
   built in 96% of our seasons, is the clearest structural hole the corpus
   shows. The goose is the cheapest animal, yields first and yields daily.
3. **Do not chase "the top build". There isn't one, and ours is currently
   ahead.** The top decile's defining property is that it builds differently
   every game (0.73 distinct builds per seat against our 0.006), and it still
   loses to our script 0.676 of the time over 105 episodes. The wave structure
   in section 1 says the useful watch is not "what do the leaders build" but
   "has a new kernel wave started", which the fingerprint table answers in
   seconds per archive.
4. **Stop using the within-episode winner/loser table on this corpus.** It was
   the right instrument when the field was heterogeneous. With 58% of the
   newest archive on eight scripts, it reports which script won and dresses it
   as a lever; section 5 shows five of its strongest columns reversing or
   vanishing across four days.

## 8. What could be an artefact

- **The corpus is a filtered slice of the ladder.** 665 episodes a day over 192
  teams, with attributed ratings running p10 = 2,684 to a maximum of 3,026. Our
  own 2,340-rated submissions are not in it. Everything here therefore
  describes the top of the ladder, and "our build's corpus copies rate 2,760"
  is a statement about the forks that were already rated highly enough to be
  published, not about a random fork.
- **Head-to-head is bank, not ladder points.** The archives give final banks
  and the ladder gives ratings; the two agree in direction (the higher-rated
  seat out-banks the lower in 68% of the prior study's corpus) but a 0.700 bank
  win rate is not a 0.700 ladder score.
- **Ratings are attributed, not read.** Same fixed point as the prior study,
  same check available; the band cut inherits whatever error it has.
- **"Varies its build" is not proven to mean "adapts".** A fingerprint that
  never repeats is equally consistent with a stochastic agent, or one whose
  timing shifts under the act timeout. The corpus cannot separate deliberate
  adaptation from noise, and the section 3 correlation is over 75 teams,
  dominated by four of them.
- **Our seasons were played against `meta_tape`, v56 and v58, not against the
  ladder.** Capital is nearly opponent-independent — v56's build is identical
  across pairings, and both seat orders give identical variant splits — but
  market-side numbers are not: `units bought from the book` moves from 659 to
  614 between our two pairings, so the comparison of that row against the top
  band is soft.
- **The fingerprint is exact-match.** An agent that plays our script but places
  one animal a day late is counted as a different build, so the 1,662 figure is
  a floor on how much of the field runs this kernel, and the top band's 0.732
  distinct-builds-per-seat is an upper bound on how much it really varies.
- **Eight days is one wave and a half.** Section 1's whole mechanism implies
  these numbers expire. The measurement to repeat is the fingerprint share
  table, on the next archive.
