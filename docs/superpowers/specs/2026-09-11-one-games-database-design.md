# One games database

Date: 2026-09-11. Every game in one place, no SQLite, no users, and the
proposer may look at any of it.

The campaign played about 2.2 million recorded days an hour and kept none of
them. The competition's replays — 17,617 episodes, 1.06 million days, 228
million moves — sat in a second store the loop never wrote to. A proposer was
handed a third: a small file of its own last games, deleted with its workspace.

So the campaign held a million day rows of what the field does, none of what it
does itself, and no question could span the two. There is one database now, and
both writers write to it.

## 1. What the proposer gets

It queries the database. Its own games and every recorded game are rows in the
same tables, and `source` tells them apart — `campaign` for a game this lineage
played, `ladder` for one recorded off the competition.

Nothing requires it to look at either. The message carries an index of its last
evaluation and a pointer; a round that wants to change one thing and measure it
never opens the database at all. The `query-games` skill carries the schema and
the queries worth running, and loads only when a round wants it.

This is the fourth time ladder-derived material has been in front of a round,
and the previous three were measured. `winning_pace.md` took medians over every
winner and eleven quantities came back between 45% and 60%, because about half
a ladder's winners are the weaker agent having a good day. `build_order.md`,
clustered to one opening, held the median candidate at 0.275 with roughly one
promotion an hour. The same build order expressed as the orders those agents
send took the median to 0.026 over 156 gates and promoted nothing in ten hours.

The diagnosis each time was grafting: the model bolts another strategy's
opening onto this one and breaks the economy underneath. What is different here
is the direction — those pushed a conclusion at the model, and this answers the
question the round already has. That is an argument, not evidence, which is why
section 7 is a tripwire rather than a hope.

## 2. Why ClickHouse

Two properties at once, both load bearing, both measured on the real corpus.

**Parallel writes.** Eight sessions record. SQLite admits one writer, and both
ways around that fail: a busy timeout does not remove `database is locked` so
much as schedule it, and a lock makes the loop take turns for a store's
convenience. Eight sessions each recording a full evaluation took 14.7s of wall
clock against 75.3s serial, and every row landed.

**Speed.** The questions are analytical. Against the same 1,057,020 day rows:

| query                             | ClickHouse | SQLite      |
| --------------------------------- | ---------- | ----------- |
| one day, one measure              | 0.005s     | 0.11s       |
| every day, four measures          | 0.009s     | 1.03s       |
| ladder against ours, one day      | 0.004s     | 0.15s       |
| a window over one program's games | 0.012s     | 0.22s       |
| every day, all 29 measures        | 0.031s     | never asked |
| count every move, 228M rows       | 0.044s     | never asked |

Compression was not a reason and is the largest number: the corpus is 14.9 GB
of SQLite and 402 MiB here, and a recorded day costs 6.3 bytes. A billion day
rows — the campaign's rate for the competition's remaining weeks — is about
6 GB rather than the 310 a row store would take.

Postgres gives the first property and not the second. DuckDB gives the second
and not the first: it is documented as one-machine, one-user analytics.

## 3. No users

One user, `default`, as the server ships. Nothing is created, granted or torn
down, and the proposer authenticates as the campaign does.

A restricted per-round user was designed and thrown away. It bought one thing —
hiding opponent names from a round, so it could not learn to recognise a pool
opponent by its play — and cost a view with definer rights, a user created and
dropped around every call, and a `system.*` lockdown. The campaign's own games
already write the other side as `opponent`, so the only names in reach are on
recorded ladder games, and that is a risk taken deliberately rather than a hole
nobody saw. It is what section 7 watches.

The server is bound to 127.0.0.1.

## 4. No SQLite

The extraction parses 25 archives across 8 processes and each one inserts
straight into ClickHouse. The shards, the merge and the load step are gone —
they existed because SQLite takes a single writer and the parse is what we were
parallelising.

Rows are held for a whole archive and sent a table at a time. That is the
batching ClickHouse asks for: `async_insert` exists to coalesce many small
writes and only adds latency to a writer that already arrives with a batch.

The build fills a staging database and swaps it in a partition at a time.
`REPLACE PARTITION` is atomic, so a rebuild that raises leaves the ladder
exactly as it was — the guarantee that mattered on 2026-09-09, when a build
that deleted first hit one unreadable market order fifty-five minutes in and
left no corpus at all. The campaign's own games are in the other partition and
are never touched either way.

## 5. Shape

Database `games`, started by `docker-compose.yml`, data under
`/data/kaggriculture/clickhouse`.

    episodes    one per game: seeds, banks, who won, and `source`
    days        one row per side per day, every measure the campaign defines
    holdings    the per-crop breakdown behind four of those totals
    prices      the shared market, per day
    orders      every market order a recorded game submitted
    moves       every farmer and hand command of a recorded game
    candidate   which seat was ours, per campaign episode
    teams       a fitted rating per recorded team

Every table is `PARTITION BY source`. `days` is
`ORDER BY (source, day, episode, seat)`, which is the order the questions
arrive in. `ReplacingMergeTree` keyed on the ordering columns, so a champion
re-scored is one program rather than two.

Column types are read off the corpus rather than guessed from the names. Across
1,057,020 recorded days nothing is ever negative, `plant_age` is the only
measure ever fractional (936,492 of them), and `sold_units` reaches 15,002,111
where every other count stays inside sixteen bits. So `Float32` for `bank` and
`plant_age`, `UInt32` for the ten cumulative counters, `UInt16` for the rest.

`dataset.measures` returns every quantity as a float, so a tile count arrives
as `4.0` for a `UInt16` column and fails the whole insert. The casts therefore
live in `Batch`, beside the schema and shared by both writers — the loop met
that wall first and the extraction met it the moment it was written.

## 6. What goes away

`browse` and the per-round SQLite extract. `seasons.db`. `dataset.SCHEMA`,
`INDEXES`, `_merge`, `_shard` and `_carry`. The `build-order` script and the
`openings`/`opening_orders`/`build_order` machinery behind it. The `report`
script, which was a page built around that table.

## 7. The tripwire

The three previous attempts failed the same way and it was visible in the same
two numbers: the median candidate's fitness and the promotion rate. Both are
logged per call.

If the median candidate falls toward zero, or promotions stop for several hours
while calls continue to land, that is the 2026-09-10 collapse again and the
recorded games come back out of reach. This is written down because the
argument in section 1 is an argument, and the campaign's record on it is nought
for three.

## 8. Testing

Against a live server, each test in a database of its own that is dropped
afterwards — a test that writes into the campaign's store and tidies up
afterwards tidies the tables somebody remembered, and ten probe episodes sat in
the real one before anything noticed.

- a failed rebuild leaves the ladder exactly as it was
- the rebuild drops only what it owns; the campaign's games survive
- eight sessions record at once and every row of every session lands
- a program recorded twice is one program
- every measure has a type, and the cast matches the column it is written into
- a name carrying a tab cannot shift the columns of a TabSeparated row
- every SQL block the skill prints runs against a database holding both sources
- the skill names no column the schema does not have
