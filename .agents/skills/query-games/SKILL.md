---
name: query-games
description: Use when improving child.py and you want to know what actually happened in the games it was scored on — seasons.db holds every day of every game, and this is its schema and the queries worth running.
---

# The games are a database. Ask it.

`seasons.db` sits beside `child.py`. It holds every game the campaign scored
this program on: one row per side per day, at full width. It is SQLite, so ask
it rather than read it.

    sqlite3 seasons.db ".schema"
    sqlite3 seasons.db "select * from swings order by moved limit 5"

## What names a game

Two integers. `matchup` is one opponent; `season` is one game against them.

Inside a matchup the opponent is fixed, so what changes between seasons is the
world -- the map, the prices, the seat. That is the variation a program has to
hold up across, and two seasons of one matchup are comparable because of it.
Between matchups the opponent changes too, so a difference there tells you
nothing about either.

Seasons are numbered narrowest first, so `season = 1` is the game a small
change would have turned and the last is the one furthest out of reach.

The opponents are not named and it does not matter which they were. They are
drawn from a field that turns over: the agent across the table in a scored game
will be one this program has never seen. A change that wins because it
recognised who it was playing wins nothing that counts.

## Tables

`days` is the body: one row per side per day, as that day closed at hour 23.
`seat` is the engine seat, `team` is who held it. Every quantity the campaign
measures is a column -- `bank`, `planted`, `ripe`, `yield_held`, `pens`,
`weeds`, `bare`, `quadrants`, `hands`, `seeds`, `shed`, `shops`, `watered`,
`dry_worst`, `fertilised`, `fed`, `hungry_worst`, `cared`, `plant_age`, and the
running counts `sell_orders`, `sold_units`, `buy_orders`, `bought_units`,
`seed_orders`, `seed_units`, `animal_orders`, `animal_units`, `hire_orders`,
`land_orders`.

`holdings` is the per-crop breakdown behind four of those totals: one row per
`(day, seat, kind, item, count)` with `kind` in plants, animals, seeds, shed.
`prices` is the shared market per day. `episodes` is how each game finished.
`candidate` says which seat was this program's, which is the one thing a
recorded game from the public ladder cannot tell you.

`orders`, `moves` and `teams` exist and are empty. The schema is the public
corpus's own so that these games and recorded ones are the same kind of row;
those three are what a recorded game carries and a played one does not.

## The two views, which are the questions worth asking

`gaps` is every one of those quantities as **yours minus theirs**, one row per
day per game. Negative is behind.

`swings` is `gaps` plus `moved`: the day-on-day change in the bank gap. It
excludes each game's first day on purpose -- there is nothing to subtract
there, and a NULL sorts before every number, so an unguarded query would answer
"day 0" for every game.

## Queries that pay

Where this program loses, across every game at once:

    select matchup, season, day, round(bank) as gap, round(moved) as moved
    from swings order by moved limit 10;

If one day keeps coming back, that is the day to fix. Then ask what was
different about it:

    select day, round(avg(planted),2) as planted, round(avg(quadrants),2) as quads,
           round(avg(fertilised),2) as fert, round(avg(bank)) as bank
    from gaps where day between 8 and 14 group by day order by day;

Which matchup is worst, and by how much:

    select matchup, count(*) as seasons,
           sum(bank > 0) as won, round(avg(bank)) as mean
    from (select matchup, season, bank from gaps where day = (select max(day) from gaps))
    group by matchup order by mean;

One season end to end, the columns you care about:

    select day, round(bank), planted, ripe, quadrants, fertilised, weeds
    from gaps where matchup = 1 and season = 1 order by day;

What was in the shed when the bank stalled:

    select h.day, h.item, h.count from holdings h
    join candidate c on c.episode = h.episode and c.seat = h.seat
    where h.kind = 'shed' and c.matchup = 1 and c.season = 1 and h.day between 8 and 12;

A quantity where this program is behind on most days -- a habit rather than one
bad game:

    select 'planted' as measure, sum(planted < 0) as days_behind, count(*) as days
    from gaps
    union all select 'quadrants', sum(quadrants < 0), count(*) from gaps
    union all select 'fertilised', sum(fertilised < 0), count(*) from gaps;

## Then measure the change

The database says what happened; it does not say whether an edit helps.
`measure.py` beside it plays `child.py` against `parent.py` on fixed seeds --
`python measure.py` for all of them, `python measure.py --replay 103` to write
one season out day by day and name the day its gap moved most. Replaying the
same seed after an edit holds the map, the prices and the opponent still, so
what moved is the edit.

Read the day, change one thing, replay that season, then check the total.
