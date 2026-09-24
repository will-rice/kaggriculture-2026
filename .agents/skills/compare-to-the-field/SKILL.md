---
name: compare-to-the-field
description: Use when you want to know what the strongest agents on the competition do differently from this program — the same database holds 26,000 recorded competition games alongside your own, and this is how to put the two side by side, day by day.
---

# You are not the only agent in the database

`games.days` holds one row per side per day for every game. `source` says which
kind: `campaign` is a game against the gate's pool, `ladder` is one recorded off
the competition between other teams, and `live` is a game this lineage played on
the competition and lost. The ladder rows are agents rated above this program
playing each other; the live rows are the opponents that actually beat us.

Your own games say whether an edit beat the opponents this lineage is scored
against. The ladder rows say what a stronger agent was doing on day 14 while
this program was doing something else. Those are different questions and only
one of them can tell you about a part of the game this lineage has never
played well.

## The games you actually lost

`source = 'live'` is the narrowest and most valuable slice: games this lineage
played on the competition and lost, both seats, every day. They are loaded
closest-first, so the ones here are games a small change would have turned --
the closest of them went by 34 coins.

The loop reloads them hourly from the submission that is standing, so they are
the current program's losses rather than a predecessor's.

The two seats are `ours` and `opponent`. No opponent is named, and almost none
is met twice, so there is nothing to recognise and nothing to be gained by
trying.

    curl -s http://127.0.0.1:8123 --data-binary "
        select day,
               round(avgIf(bank, team = 'ours')) as ours,
               round(avgIf(bank, team = 'opponent')) as theirs,
               round(avgIf(bank, team = 'ours') - avgIf(bank, team = 'opponent')) as gap
        from games.days where source = 'live'
        group by day order by day format TabSeparated"

Read the gap column down the season and find where it turns. That is the
stretch the game was decided in, and it is the one worth asking the rest of
your questions about -- every column below works the same way with
`source = 'live'` and `team` in place of `source`.

These are the only rows here that are both _real opponents_ and _our losses_.
The pool games are neither: they are opponents chosen weeks ago that this
program beats most of the time.

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
