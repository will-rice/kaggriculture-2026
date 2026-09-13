---
name: query-games
description: Use when improving child.py and you want to know what actually happened in the games it was scored on — one database holds every day of every game this program played and every game recorded off the competition, and this is its schema and the queries worth running.
---

# The games are a database. Ask it.

One database holds every game. Your own, from the evaluation the message
indexes, and every game the competition has recorded. It is ClickHouse, so ask
it over HTTP — no client to install, no driver, SQL in the body:

    curl -s http://127.0.0.1:8123 --data-binary "select count() from games.days"
    curl -s http://127.0.0.1:8123 --data-binary "
        select name from system.tables where database = 'games'"

Add `format TabSeparated` for something readable, or `format JSONCompact` for
something to parse. Wrap a long query in single quotes and keep the SQL in
double, or the shell will eat it.

## Which rows are yours

`source` says. `campaign` is a game this lineage played; `ladder` is one
recorded off the competition.

Your own evaluation's games are keyed by `matchup` and `season`, joined
through `candidate`. A matchup is one opponent and every season played against
them; within it the opponent is fixed, so what changes from season to season
is the world -- the map, the prices, the seat -- which is the variation a
program has to hold up across. Between matchups the opponent changes too.

Seasons are numbered narrowest first, so `season = 1` is the game a small
change would have turned.

The opponents in your own games are written as `opponent` and it does not
matter which they were. You are writing a program that has to beat an agent it
has never seen; a change that wins because it recognised a particular one wins
nothing that counts.

## Tables

`days` is the body: one row per side per day, as that day closed at hour 23.
`seat` is the engine seat and `team` is who held it. Every quantity the
campaign measures is a column -- `bank`, `planted`, `ripe`, `yield_held`,
`pens`, `weeds`, `bare`, `quadrants`, `hands`, `seeds`, `shed`, `shops`,
`watered`, `dry_worst`, `fertilised`, `fed`, `hungry_worst`, `cared`,
`plant_age`, and the running counts `sell_orders`, `sold_units`, `buy_orders`,
`bought_units`, `seed_orders`, `seed_units`, `animal_orders`, `animal_units`,
`hire_orders`, `land_orders`.

`holdings` is the per-crop breakdown behind four of those totals: one row per
`(episode, seat, day, kind, item, count)`, `kind` in plants, animals, seeds,
shed. `prices` is the shared market per day. `episodes` is how each game
finished. `candidate` says which seat was this lineage's, for its own games.

`orders` and `moves` are filled for recorded games only: every market order
and every farmer or hand command, at the hour it was sent. A played game has
its states and not its keystrokes.

## Queries that pay

Where your program loses, across every game of its last evaluation:

    with mine as (
        select c.matchup, c.season, d.day,
               maxIf(d.bank, d.seat = c.seat) - maxIf(d.bank, d.seat != c.seat) as gap
        from games.days d
        join games.candidate c on c.episode = d.episode
        where d.source = 'campaign'
        group by c.matchup, c.season, d.day
    )
    select matchup, season, day, round(gap) as gap,
           round(gap - lagInFrame(gap) over (partition by matchup, season order by day)) as moved
    from mine order by moved limit 10

If one day keeps coming back, that is the day to fix. Then ask what was
different about it -- your side against the other, on the days around it:

    select d.day,
           round(avgIf(d.planted, d.seat = c.seat), 1) as mine,
           round(avgIf(d.planted, d.seat != c.seat), 1) as theirs
    from games.days d join games.candidate c on c.episode = d.episode
    where d.source = 'campaign' and d.day between 8 and 14
    group by d.day order by d.day

## The recorded games, if you want them

They are there and nothing requires you to look. What they answer is "what
does the field do", not "what should I do" -- three attempts to hand a program
the field's opening as a plan made it markedly worse, because bolting another
strategy's orders onto this one breaks the economy underneath. Read them as a
question, not a recipe.

What the field holds on a given day:

    select day, round(avg(planted), 1) as planted, round(avg(quadrants), 2) as quads,
           round(avg(fertilised), 1) as fert, round(avg(bank)) as bank
    from games.days where source = 'ladder' and day in (3, 5, 10, 20, 29)
    group by day order by day

The same, for the stronger half of the field only -- about half of any
ladder's winners are the weaker agent having a good day, so "what winners do"
averaged flat is mostly noise:

    select d.day, round(avg(d.planted), 1), round(avg(d.quadrants), 2)
    from games.days d
    join games.episodes e on e.episode = d.episode
    join games.teams t on t.team = d.team
    where d.source = 'ladder' and t.place <= 20
    group by d.day order by d.day

## Then measure the change

The database says what happened; it does not say whether an edit helps.
`measure.py` beside `child.py` plays it against `parent.py` on fixed seeds --
`python measure.py` for all of them, `python measure.py --replay 103` to write
one season out day by day and name the day its gap moved most. Replaying the
same seed after an edit holds the map, the prices and the opponent still, so
what moved is the edit.

Read the day, change one thing, replay that season, then check the total.
