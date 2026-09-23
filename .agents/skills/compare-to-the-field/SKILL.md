---
name: compare-to-the-field
description: Use when you want to know what the strongest agents on the competition do differently from this program — the same database holds 26,000 recorded competition games alongside your own, and this is how to put the two side by side, day by day.
---

# You are not the only agent in the database

`games.days` holds one row per side per day for every game. `source` says which
kind: `campaign` is a game this lineage played, `ladder` is one recorded off the
competition. The ladder rows are the agents that are actually rated above this
program, playing each other, through 2026-09-22.

Your own games say whether an edit beat the opponents this lineage is scored
against. The ladder rows say what a stronger agent was doing on day 14 while
this program was doing something else. Those are different questions and only
one of them can tell you about a part of the game this lineage has never
played well.

## Which agents are worth comparing against

Find them, do not assume them -- who is strong changes weekly.

    curl -s http://127.0.0.1:8123 --data-binary "
        select team, round(avg(bank)) as bank, count() as seats
        from games.days
        where source = 'ladder' and day = 29
        group by team having seats >= 40
        order by bank desc limit 10 format TabSeparated"

## Where the two diverge

A season is thirty days and a game is usually decided in one stretch of it. The
cheapest question in this whole database is which stretch.

    curl -s http://127.0.0.1:8123 --data-binary "
        select day,
               round(avgIf(bank, source = 'ladder')) as ladder,
               round(avgIf(bank, source = 'campaign')) as ours
        from games.days
        group by day order by day format TabSeparated"

Read it as a shape, not as a score. Bank is two-sided -- the market and the
town are shared, so what a farm ends with depends on who it played -- and the
two columns are different opponents. What transfers is _when_ the curves part.
A lead that survives to day 10 and is gone by day 20 is a statement about the
second half, and it points at the days worth reading next.

## What a stronger agent was holding while you were not

Every quantity the campaign measures is a column: `planted`, `ripe`,
`yield_held`, `pens`, `hands`, `seeds`, `shed`, `shops`, `watered`,
`fertilised`, `fed`, `cared`, `plant_age`, `weeds`, `bare`, `quadrants`, and
the running counts `sold_units`, `bought_units`, `seed_units`, `animal_units`,
`hire_orders`, `land_orders`.

Pick the stretch the curves part in, then ask what differs across it:

    curl -s http://127.0.0.1:8123 --data-binary "
        select source, day, round(avg(planted)) as planted,
               round(avg(hands)) as hands, round(avg(seed_units)) as seeds_bought,
               round(avg(sold_units)) as sold
        from games.days
        where day between 10 and 22
        group by source, day order by day, source format TabSeparated"

Swap the columns for whatever the stretch suggests. A column where the two
sources separate early and stay separated is worth more than one that differs
on a single day.

## Then make it a rule

A difference in the table is a hypothesis, not an edit. `plan.json` is ordinary
JSON: the useful change is a rule applied to every step that matches it, not
one step retyped. Measure it the way the message says, and keep it only if the
experiment that would have disproved it did not.

## What this cannot tell you

The ladder rows are recordings, not opponents. You cannot play them, and an
agent that issued a different command sequence in every one of its recorded
games will not be reproduced by replaying one of them. Use these rows to find
out _what_ to change; use `measure.py` to find out whether the change is good.
