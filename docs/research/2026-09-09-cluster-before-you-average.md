# Mining the corpus better: cluster before you average

**Date:** 2026-09-09
**Question:** We hold 16,955 games of agents that are beating us. Are we
extracting what is in them?
**Answer:** No. Every artifact we build from the corpus is a mean over the top
twelve, and those twelve are not doing the same thing. The mean of a day-3 land
rush and a day-6 opening is an instruction neither group follows and no agent
can execute. The fix is the one the replay-mining literature has used since
2009: cluster the population into strategies first, then describe each.

---

## The measurement that forced this

A pacer was built to test whether the build order is prescriptive: champion_67
with the corpus's capacity purchases bolted on, one variable, 128 games. It drew
0.5000, and **114 of the 128 games were identical**. The land purchase never
fired.

It could not fire. The build order says the top twelve hold **1.16 quadrants on
day 3** with a mean bank near 200, and land costs 1,000. Reading the
distribution behind that mean:

| day | mean | what is actually there                       |
| --- | ---- | -------------------------------------------- |
| 3   | 1.16 | **83.8% hold 1, 16.2% hold 2**               |
| 6   | 2.00 | 99.7% hold 2 — a real behaviour              |
| 8   | 2.34 | 65.7% hold 2, 34.3% hold 3                   |
| 10  | 2.53 | 47.4% / 52.6%, banks of 15,139 against 7,642 |

A mean of 1.16 over a quantity that only takes whole values is not a behaviour.
It is a mixture, and at day 10 the two components differ in bank by a factor of
two — not two stages of one strategy, two strategies.

## The split is a strategy, not a situation

The obvious repair is to separate the groups, which only means anything if the
groups are teams rather than games. They are teams, decisively:

    within a team, the day of the 2nd quadrant varies by 0.11 days
    between teams, the median day ranges from 3 to 6

And the split sorts almost perfectly by strength:

| team              | rating      | 2nd quadrant |
| ----------------- | ----------- | ------------ |
| HowardLeeTW       | **+2.33**   | **day 3**    |
| 沒有道歉 沒有道歉 | **+2.21**   | **day 3**    |
| 我都先道歉        | **+2.04**   | **day 3**    |
| Otter Vibe        | +1.16       | day 5        |
| ymg_aq            | +0.39       | day 5        |
| SpaTaro, keiz, +5 | +0.26…+0.60 | day 6        |

The three best agents on the ladder, and only they, take land on day 3. Fourth
place is 0.9 log-odds behind — about 155 Elo. Averaging across this group
destroys exactly the variable that separates first from fourth, and that average
is what every round of the campaign has been shown.

What pays for the rush is visible once the group is separated: banks of **625.6
on day 1 and 995.7 on day 2** against 119.1 and 162.5 for the rest, with 7.0
hands against 4.2. They staff up, hold cash to just under the land price, and
spend it on day 3.

## What the literature says we should have been doing

This has a name. [Simpson's paradox](https://doi.org/10.1145/3580305.3599859) is
the reversal or destruction of a trend under aggregation, and the KDD treatment
states the connection we ran into directly: the underlying cause is clustering —
individual points must be grouped before an aggregate over them means anything.
It is [a known and studied risk in data mining](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4829534)
rather than an exotic failure.

The RTS replay-mining literature has never done it our way. Strategies there are
characterised by **build orders as sequences** and **army compositions**, and the
standard pipeline
[clusters replays into strategy classes](https://annals-csis.org/Volume_25/drp/pdf/48.pdf)
before predicting or describing anything;
[army compositions are reduced to mixtures of Gaussians](https://arxiv.org/pdf/1211.4552)
so that reasoning happens at the level of components. Mixture models, not means.

Note what this is not. `2026-09-02-literature-competitive-sim-agents.md`
Finding 5 reviewed build-order _planning_ — Churchill and Buro's BOSS and its
descendants — and correctly closed it, because those planners abstract away
income and labour, which is our binding constraint. Replay _mining_ is a
different literature and was never reviewed here.

## Three things we are not extracting

**1. Strategy clusters.** Everything is a mean over the top twelve. The opening
signature — the day the 2nd and 3rd quadrants land — is discrete, almost
perfectly consistent within a team (0.11 days), and sorts by rating. It is a
ready-made cluster label and it costs one query.

**2. Sequences.** The dataset holds **220M moves and 28M orders**, and every
artifact we build reduces a game to 29 per-day state quantities. The literature
mines the order of actions; we mine only their accumulated effect. A build order
is a sequence and we have never represented it as one.

**3. The best cluster as a target.** The top three are a coherent strategy with
a 155-Elo edge over fourth place, and we have 114 games of them. That is a
target an agent could actually be built against, where the mean was not.

## What to change

- **Segment `dataset.build_order` by opening cluster** and emit one table per
  cluster, with the strongest cluster named. The mean stays useful as a
  description of the field; it stops being handed to a round as an instruction.
- **Re-run the claim measurement within clusters.** 148 of 870 forms settle over
  the pooled corpus. Claims that are open because two strategies disagree should
  settle inside each.
- **Re-test the pacer against the top-3 opening** rather than the mean: second
  quadrant on day 3, third on day 8, hands to 7 by day 2, bank held to 1,000.
  That is a behaviour, so a pacer can execute it, which the mean version could
  not.

## What would change this conclusion

- **Clusters that do not separate on rating.** The opening split sorts the top
  twelve almost perfectly today; if that is an artifact of twelve teams and one
  ten-day window, a wider window should break it.
- **A cluster-specific build order that still cannot be followed.** The mean
  failed on affordability. If the top-3 table is also unaffordable when followed
  exactly, the problem is not aggregation but that state summaries cannot carry
  a strategy at all — and the sequence data becomes the only remaining route.

## Method

Distributions and opening signatures are read from
`/data/kaggriculture/corpus.sqlite` over the rated window, top twelve by the
deployed Bradley-Terry fit. The pacer is champion_67 with the corpus's quadrant
and pen targets appended to its market orders, played 128 games against
champion_67 itself on fresh seeds, both seats.

Scripts: `blend.py`, `openings.py`, `rush.py`, `build_pacer.py`, `pace_off.py`,
run 2026-09-09.

## Sources

- [Learning to Discover Various Simpson's Paradoxes (KDD 2023)](https://doi.org/10.1145/3580305.3599859)
- [Simpson's Paradox and Data Mining: a New Risk, a New Potential](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4829534)
- [StarCraft strategy classification of a large human versus human dataset](https://annals-csis.org/Volume_25/drp/pdf/48.pdf)
- [A Dataset for StarCraft AI & an Example of Armies Clustering](https://arxiv.org/pdf/1211.4552)
- [Sequential Pattern Mining in StarCraft: Brood War (AAAI)](https://cdn.aaai.org/ojs/12736/12736-52-16253-1-2-20201228.pdf)
- [Replay-based strategy prediction and build order adaptation for StarCraft AI bots](https://scispace.com/pdf/replay-based-strategy-prediction-and-build-order-adaptation-h7a82er0p0.pdf)
