# Can a model judge a move in this game?

**Date:** 2026-09-09
**Question:** Give a model the full game state and a structured move format, and
ask it what the right move is. Could it be an oracle — a reference for what good
play looks like, feeding diagnosis into the optimizer's rounds?
**Answer:** No, not as a source of feedback. It reasons about the farm it can
see, and this game is decided by the farm you are buying. Measured: it lands on
a quantity that decides games **3% of the time against a 31% chance baseline**.

---

## What this is not

Not distillation. This project already cloned a teacher into a policy — 92.8%
per-action agreement, **0 of 48 games won**, and one DAgger round leaving the
win rate at 0.000 — and diagnosed it as non-realizability rather than covariate
shift (see `2026-09-02-literature-competitive-sim-agents.md`, Finding 3).

An oracle is queried, not imitated. Nothing is fit to it, so that failure does
not apply. It also costs far less than it first appears: a season is 719 calls
to play, but a handful to _ask about_.

## The result

| Measure                     | Result          | Chance |
| --------------------------- | --------------- | ------ |
| **Lands on a settled form** | **1 / 36 = 3%** | 31%    |
| Agrees, given settled       | 1 / 1           | 50%    |

Both criteria were registered before the run. Expected ~11 hits of 36; one
arrived (p ≈ 7 × 10⁻⁵). It is not merely failing to find the levers — it avoids
them significantly better than chance.

Across 49 verdicts in three configurations, **one** landed on a claim the corpus
can adjudicate.

## What it reaches for

Quantities chosen over 36 verdicts, with the full season trajectory available:

    bank 7   planted 5   seed_units 5   pens 4   sold_units 3
    animal_units 3   seeds 3   hands 2   shed 1   bought_units 1
    fed 1   watered 1

`land_orders` and `quadrants` — the corpus's two strongest signals, at 99%
agreement on day 3 — appear **zero times**.

The pattern explains itself. `bank`, `planted`, `seeds` are states of the farm:
things visible in the frame. `land_orders` and `quadrants` are purchases —
irreversible commitments that pay off ten days later and look like nothing at
the moment they are made. The model reasons about the farm it can see, and the
game is decided by the farm being bought.

That is not a logging gap. The trajectory handed to it carried both columns for
both sides on every day of the season, and it still never looked.

## What it does well, which is why this needed measuring

The first seven verdicts were extremely convincing:

- **State-specific.** "The five hires cost only 1+1+2+3+5 = 12, while the five
  wheat seeds cost 50, leaving 90 coins from the farm's 152."
- **Discriminating.** Two of seven called the played move correct. A filler
  generator flags everything.
- **Well-formed.** Zero unusable claims across 49 verdicts; it uses the
  quantity vocabulary exactly.
- **Consistent.** Four of five flagged moves converged on one diagnosis.

That last one is what makes it dangerous rather than useless. The shared
diagnosis was that `champion_65` under-hires. Checked against the corpus, the
champion tracks the ladder's top twelve within 0.7 hands at every mark, and
`hire_orders` is unsettled at every day it raised, over 1,231 to 1,795 games
each.
Hiring is not a lever, so there was nothing there to win.

Wrong at random is survivable: the gate turns away candidates that act on bad
advice. Wrong the same way every time, fluently and with arithmetic, is not.

## Two corrections made along the way

The first two probes asked about days 8 to 26. Settled forms concentrate in the
opening — 31% at days 0-8 against 4% at days 16-22 — so most of that null result
was the question being aimed badly rather than the answer being wrong. The
decisive run asks only about days 0-8 and pre-registers its criteria, because
three iterations of adjusting a probe until it agrees would prove nothing.

Choosing turning points by bank divergence is also wrong, and wrong in an
interesting way: bank gaps open late because banks are large late, while the
decisions that cause them are early. Selecting on outcome divergence points at
symptoms.

## What to keep

Not the oracle. **The harness that scored it.**

`evidence.questions()` already crosses every quantity with every day, so the
claim store holds 870 pre-measured statements about strong play, 148 of them
settled over thousands of paired games. Any proposed source of feedback can be
made to state its claims in that vocabulary and scored against them, for about
fifty cheap calls, before it is allowed near a round.

That is worth more than this negative result. Without it, the first seven
verdicts — specific, numerate, discriminating, unanimous — would have been
convincing enough to wire in.

## Method

`gpt-5.6-luna` at medium reasoning, one call per moment, each in its own
directory holding `rules.md`, `state.json` (a seat's whole observation plus the
move played) and, in the later runs, `trajectory.json` (29 quantities for both
sides on every day). Verdicts are written as JSON files, scored against
`strategy.Strategies` with `CONFIRM = 0.65`, `REFUTE = 0.35`, `SUPPORT = 200`.

Games are champion_65 against `router2929`, its worst matchup at 0.688, played
over fresh seeds until narrow losses turned up — narrow because a game lost by
611 coins in 60,000 turns on something findable, where a rout turns on
everything at once.

Scripts: `oracle.py`, `calibrate.py`, `trajectory.py`, `calibrate2.py`,
`decisive.py`, run 2026-09-09.

## What would change this conclusion

- **A model that raises `land_orders` or `quadrants` unprompted.** The whole
  finding is that it reasons about farm state rather than commitments. Naming
  those in the prompt would not test it — it would supply the answer.
- **Feedback scored above 31% by the same harness.** The bar is now measured
  rather than argued, and it applies to any source, not only this one.
