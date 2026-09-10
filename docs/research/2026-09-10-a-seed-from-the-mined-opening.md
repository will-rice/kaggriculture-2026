# Writing a seed from the mined opening

**Date:** 2026-09-10
**Question:** The corpus now yields the ladder's strongest opening as executable
orders. Can that be written into a seed program, so the campaign starts
somewhere other than the one public agent every champion descends from?
**Answer:** Not by hand, and the failure is worth recording. The agent I wrote
executes the opening faithfully — seven hires, land on day three — and **loses
to an agent that passes every turn, 0.000 over 32 games**, with final banks of
541 against 3,000.

---

## What it does and does not do

It follows the mined opening. By day 3 it holds two quadrants, which is the
signature the ladder's three best agents share and champion_69 does not. It
hires to seven. Those parts work.

What it never does is convert any of it. Planted tiles sit at 0 or 1 all
season, nothing is harvested, and the starting $3,000 leaks away on hires and
seed until 541 is left. The PASS agent spends nothing and keeps 3,000.

## The finding underneath

**Copying what winners spend, without the loop that converts it, is worse than
doing nothing.** Not "less good" — worse than an agent that sits still and
banks its opening balance.

This is the sharpest form of the warning that has run through all of today's
corpus work. The top three agents spend heavily and early: seven hires on day
zero, $2,500 of livestock, land on day three. Every one of those is a real
order, mined from real games, and reproducing the spending alone converts a
$3,000 balance into $541. The spending is not the strategy. It is what the
strategy can afford.

## The mechanics that defeated it

Recorded because each cost a debugging cycle and none is obvious from the
rules:

- **Animals are not pens.** `BUY_ANIMAL` puts the animal in the shed. It earns
  nothing until a unit builds a pasture, carries it there and places it. Bought
  without that, five cows and a sheep is $2,500 spent on livestock that never
  leaves storage — the first version scored a flat zero.
- **Hands are `[row, column]`, not objects.** Guessed as a dict with a
  "position" key, and every hand was treated as standing on the same square,
  so seven of them repeated one move all season.
- **A seed weeds the night it is planted** unless watered that day, so planting
  and watering cannot be separate priorities for separate units.
- **`agent` must be the last callable in the file.** The runner takes
  `[v for v in env.values() if callable(v)][-1]`, by insertion order. A helper
  defined below `agent` is served instead — the runner picked `_market` and
  every game died on a TypeError.

That last one bit twice more in the same session, in a wrapper that ended with
`_BASE_AGENT = agent` after the wrapper's own definition. Rebinding `agent`
does not move it in insertion order, so `_BASE_AGENT` was the last callable and
the wrapper never ran. **Two earlier pacer experiments were void because of
it**, and both returned 0.5000 with near-identical play — which is what "no
code ran" looks like, and was read as "no effect".

## What to do instead

The mined opening is already where it belongs: the round prompt carries it as
orders, so the campaign's own search can build it into an agent that also
farms. The search has produced a 156KB agent over 69 generations and is better
at this than a hand-written seed.

What a seed could still be worth is a _different_ starting point rather than a
strong one — `config.SCRATCH_AGENT`, the blank slate one session in eight
begins from, is currently an agent that passes every turn. This one is not a
candidate for that either: it loses to that blank.

## Method

`seed_mined.py`, played against `router_v1` and against the current
`config.SCRATCH_AGENT` over fresh seeds, both seats, on the engine port.
