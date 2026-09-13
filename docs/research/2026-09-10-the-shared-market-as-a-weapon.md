# The shared market as a weapon

The two farms never touch. Neither seat can reach the other's land, take its
animals or spend its money, so it is easy to conclude the game is two solitaires
scored against each other. That is wrong, and the reason is one line in the
engine:

```cpp
struct State {
    Farm farms[2];
    Market market;      // one, not two
```

`SELL` raises that shared inventory by a unit, `BUY_PRODUCT` lowers it, and
every price is a pure function of it. Whatever either seat does to a price, it
does to both. This is the whole attack surface, and it is a real one.

## How thin the markets are

Units to take each product from its opening inventory to the floor price of 1,
and what the seller collects on the way down. The curve is `market_price` from
`sim.hpp`, replicated and checked against the engine's own reported prices on
all eight products before any of this was believed.

| product    | base | units to floor it | seller collects | per unit |
| ---------- | ---- | ----------------- | --------------- | -------- |
| WOOL       | 200  | **59**            | 7,928           | 134      |
| STRAWBERRY | 120  | 62                | 3,809           | 61       |
| MILK       | 160  | **76**            | 6,181           | 81       |
| MELON      | 250  | 158               | 26,485          | 168      |
| TOMATO     | 60   | 529               | 11,128          | 21       |
| CARROT     | 35   | 842               | 10,680          | 13       |
| WHEAT      | 25   | 4,000             | 75,464          | 19       |
| EGG        | 50   | 4,000             | 149,763         | 37       |

Fifty-nine wool. That is not many sheep.

## How slowly they recover

Inventory only comes down through `town_consume`: each unlocked shop pulls its
multiplier from every product in its recipe once every four steps -- six times
a day -- and the town centre pulls one of everything, once a day. Nothing else
removes supply.

So recovery is set by which shops happen to be open, and shops unlock one every
three days, drawn uniformly from all eight. Before the yarn store opens, wool
climbs back at **one unit a day**, against a season of thirty. Melon appears in
no shop's recipe at all, so a melon trough only ever refills at that same one a
day, all season, whatever unlocks.

## The experiment: hold, then dump

`experiments/denial/main.py` is the seed with one change. Rather than selling
each unit the turn it appears, it withholds the thin goods and releases them in
one block, timed at the opponent's own production -- their animals are public,
and an animal is a commitment, so `placed_day` plus the yield tables say which
day their wool starts arriving a week before it does.

It lost.

| denial vs the seed it came from, 32 games, both seats |       |
| ----------------------------------------------------- | ----- |
| score                                                 | 0.250 |
| mean relative bank                                    | -764  |

It was not untested: the block was held on 1,382 turns for melon, 1,250 for
strawberry, 644 for milk. The idea ran and was wrong.

## Why: the crash already happens, for free

Market inventory over an ordinary game, at noon each day:

| product    | day 1 | day 10 | day 20 | day 29 | price at day 29 |
| ---------- | ----- | ------ | ------ | ------ | --------------- |
| WOOL       | 9,999 | 9,994  | 10,025 | 10,057 | 200 -> ~11      |
| MELON      | 9,999 | 9,994  | 10,013 | 10,152 | 250 -> ~19      |
| MILK       | 9,999 | 9,994  | 9,989  | 10,054 | 160 -> ~89      |
| STRAWBERRY | 9,999 | 9,979  | 9,908  | 10,046 | 120 -> ~65      |

Both seats flood the thin markets by the end of the season without trying. The
denial is not unavailable; it is already spent, by both sides, at no cost to
either. Holding stock to crash a price later means arriving into a crash that
was coming anyway, having given up the good prices on the way there.

## What the same measurement found instead

The other half of the board never gluts. The town eats those products faster
than two farms can supply them, and the price curve _rises_ below the 10,000
baseline, so they pay above sticker all season and no one can crash them.

| product | day 29 inventory | actually sold at | base |      |
| ------- | ---------------- | ---------------- | ---- | ---- |
| WHEAT   | 9,000            | 52               | 25   | 2.1x |
| TOMATO  | 9,745            | 162              | 60   | 2.7x |
| CARROT  | 9,883            | 90               | 35   | 2.6x |
| EGG     | 9,760            | 42               | 50   |      |

The useful division is not cheap against valuable. It is **consumed against
hoarded**. The high-base luxuries look like the prize and end the season near
the floor; the unglamorous consumed goods pay above base forever.

## The bug that followed from it

`_crop_choices` prices a harvest at the inventory it projects for the day the
harvest lands, and subtracts what the town will have eaten by then -- including
demand from shops that have not opened yet. The weights for that sit in
`expected`.

Each unlock draws uniformly from all eight shops, so the demand one adds in
expectation is exact: six times that shop's multiplier, over eight.

| crop       | seed assumed | truth    | consuming shops |
| ---------- | ------------ | -------- | --------------- |
| CARROT     | 2.25         | 2.25     | 2               |
| TOMATO     | 1.50         | 1.50     | 2               |
| STRAWBERRY | 3.00         | 3.00     | 4               |
| WHEAT      | 3.00         | **3.75** | 5               |
| MELON      | 6.00         | **0.00** | **0**           |

Three are exact. Melon is out by the entire quantity: the table budgets six
units a day of future demand against a product that appears in no shop's
recipe and never will. Every melon harvest was priced at an inventory with
phantom consumption removed from it, and came out looking far better than it is.

The flat `score *= 0.30` melon penalty for the first ten days, sitting further
down the same function, is the patch over that hole -- the symptom corrected at
the far end of the calculation that caused it.

## The fix

`experiments/portfolio/main.py` sets `expected` to the true values and drops the
melon multiplier, which with the demand right was correcting twice.

| portfolio vs the seed, 32 games, both seats |                    |
| ------------------------------------------- | ------------------ |
| score                                       | **0.781**          |
| mean relative bank                          | **+2,800 +/- 583** |

Against the fourteen outside opponents it is indistinguishable from the seed
(-19,070 against -18,896, two opponents beaten against one). That is expected
rather than disappointing: those games are unpaired and there are eight per
opponent, and resolving a difference of a few thousand coins unpaired takes
upwards of a hundred games. The head-to-head is paired -- same episodes, both
seats -- which is why sixteen seeds settle it there.

## For whoever picks this up

- The denial idea is closed. The markets crash themselves; there is nothing
  left to deny. What is _not_ closed is the defensive half -- an agent that
  sells a thin good early because it can see the glut coming, rather than
  holding it into one.
- Melon is a trap and the engine says so structurally: no shop will ever
  consume it. Any valuation that treats it as clearable is wrong.
- `expected` was three-fifths right, which is how it survived. Weights derived
  for some entries and guessed for others look identical in the source.
- Both experiments are variants of `src/kaggriculture/seed/main.py` differing
  in one place each, kept that way on purpose so a head-to-head against the
  seed means something. Neither is on the campaign's path: the copy checker
  reads `/data/kaggriculture/opponents` and `/data/kaggriculture/agents`, and
  the seed is named by `config.SEED`, so nothing under `experiments/` is
  reachable by the optimizer.

## Chasing champion_65

The bar for a submission is not the seed, it is the agent a submission would
displace. Kaggle scores the latest two, so anything sent up bumps champion_64
or champion_65 out of a scored slot.

The portfolio agent, which beat its own parent by +2,800, loses to champion_65
by **-22,918** over 24 games, winning 2. Beating a parent measures whether a
change helped. It says nothing about whether an agent is worth a slot, and
reading one as the other is how the tape lineage saturated against itself.

### The gap is farming, not fighting

|             | alone   | facing each other |
| ----------- | ------- | ----------------- |
| portfolio   | 143,026 | 75,299            |
| champion_65 | 166,017 | 92,128            |

Contact costs both sides about 45% of their bank -- the shared market again --
but it costs _us_ 6,163 less than it costs them. The whole deficit is farming
in isolation: 22,992 with the board to ourselves.

### And it is not for want of inputs

Averaged over six episodes, each alone:

|                        | day 5 | day 15 | day 25   | day 29  |
| ---------------------- | ----- | ------ | -------- | ------- |
| hands, portfolio       | 7.0   | 13.0   | 13.0     | 12.0    |
| hands, champion_65     | 5.0   | 11.0   | 11.0     | 8.5     |
| plants, portfolio      | 21.0  | 61.8   | **37.3** | 16.3    |
| plants, champion_65    | 19.0  | 57.7   | **56.0** | 15.0    |
| quadrants, portfolio   | 1.0   | 4.0    | 4.0      | 4.0     |
| quadrants, champion_65 | 1.0   | 3.0    | 3.0      | 3.0     |
| money, portfolio       | 157   | 22,388 | 98,396   | 131,978 |
| money, champion_65     | 95    | 33,894 | 128,524  | 162,362 |

More labour and more land, fewer plants and less money. The farm empties out
after day 15 while theirs stays full.

### Twenty-two tiles, bare, with the seed for them in the shed

From day 15 to the end, 22 to 25 unlocked tiles sit empty, and `_crop_choices`
would take a crop on every one of them -- 45 of them by day 25. Meanwhile the
seed counts stand frozen at `{WHEAT: 12, STRAWBERRY: 10, MELON: 2}` for ten
days: nothing planted, nothing bought. Planting falls from 2.19% of worker
actions in the first half to 0.91% in the second, while 11% of actions are
workers passing.

Three things were tested against that, and only one of them was the cause.

**Planting priority.** Tasks are ranked by value and planting sits at 65,
under watering at 110, placing an animal at 120 and fertilizing at 135. Raising
it to 115 made the agent _worse_: -2,887 against the seed. Watering earns its
place, and stealing workers from it costs more than the planting gains.

**The seed cap.** `n = min(12, n) - seeds.get(c, 0)` holds at most twelve seeds
of a crop however many plots want it -- 45 plots asking for wheat against 12
wheat seeds -- and seeds cost no shed space, so the cap buys nothing. Removing
it changed nothing measurable: +5,603 against +5,968, inside the error bar.
Seeds were not the constraint either.

**The fourth quadrant.** This was. It costs 4,000 on a farm that already cannot
work the land it has, and champion_65 never buys it. Pricing it out of reach:

|                 | vs the seed, 32 games           | vs champion_65, 24 games |
| --------------- | ------------------------------- | ------------------------ |
| portfolio       | +2,800 +/- 583, score 0.781     | -22,918, score 0.083     |
| three quadrants | **+5,968 +/- 722, score 0.938** | **-15,038, score 0.208** |

A third of the gap to champion_65, closed by not buying something.

### Where this leaves it

Still losing to champion_65, and the remaining 15,000 is the same idle land
seen from the other side: the farm cannot work three quadrants either. Worker
routing is the constraint -- not priority, not seeds, not land. Workers are
assigned plant tasks and walk toward them, and the seed counts never fall,
so they are being re-targeted before they arrive.

That is the next thing to measure: how often a worker's target changes before
it reaches it. `_TARGETS` carries the previous assignment across turns, and if
that persistence is weaker than the re-ranking, the farm spends its labour
walking. Movement is 49% of all worker actions after day 15.
