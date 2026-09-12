# Selection by games won

Date: 2026-09-12. Replaces sections 5.1 and 5.2 of
`2026-09-05-campaign-script-design.md`.

## Why

79 programs, zero champions. Measured on the day:

- `gate.promotion` needs `place == 1` of a Bradley-Terry tournament in both of
  its regimes. The best program placed 25 of 28, 628 Elo short. Four of the five
  above it are champions of the tape lineage, and three of those are not in the
  pool and cannot be played.
- With no promotion, nothing is retained and the search re-converges. Five of 14
  programs in that day's run came from a parent at 0.206 and landed at
  0.180-0.203.
- The session's starting program came from a weighted draw ranked on the rating,
  which tied the best program (0.331) with one at 0.206, so the draw took the
  worse one half the time.
- Refusals were logged only when they were not refusals.
- All 79 programs hold 5,777-6,236 on day 10 against opponents averaging 16,219.

That is a broken scaffold, not a verdict on what a model can write: handed the
games database, codex diagnosed the day-10 problem unprompted.

## The pool

Every public agent, plus the most recent champion. One champion, replaced on
promotion rather than accumulated, so the pool drifts toward the field and never
toward our own lineage.

## The score

Win rate over that pool. Every candidate plays all of it, on the session's
shared seed block, in both seats: 1,088 games today, 9.1 minutes at five
workers.

Playing everyone is what removes Bradley-Terry. A rating is for a sparse graph,
and the graph was sparse only because opponents were drawn per candidate, leaving
two candidates' rates incomparable. A complete balanced design makes the win rate
the whole of what the rating estimated.

The opponent set is never capped. A cap is the same subset chosen once instead of
per candidate, and it decides which opponents a program is never measured
against. The pool's size is therefore a linear cost -- 60 opponents is 16
minutes, 100 is 27 -- absorbed on `GATE_SEEDS`, `SESSIONS` and the worker split,
or by accepting longer evaluations, which stay short against a codex call.

Seeds and opponents trade at a fixed budget: seeds buy resolution on the bar that
promotes, opponents buy breadth on the rate that selects. Only the first is ours
to set.

Bradley-Terry stays in `dataset.py`, over the ladder corpus, where 48,337 games
were paired by Kaggle and the graph really is sparse.

## Promotion

One condition: the Wilson lower bound of `result.rates[champion]` above 0.5, over
at least `DECISIVE_GAMES` decisive games. At 32 decisive games that is 22 wins,
0.688.

The champion is in the pool -- the seed is champion zero, so there is always
one -- and this is therefore directly measured from the first round: 16 seeds in
both seats, both programs in the same games, so the map is shared and the seat
swap cancels position.

`DECISIVE_GAMES = 8` blocks 4 to 7 decisive games -- a candidate drawing 78% to
88% of its games with the champion. Below 4 the interval refuses on its own
(3/3 is 0.438).

**Known risk.** Promotion is a ratchet, not a proof of progress, so a chain of
head-to-head wins can walk around a cycle. Each champion's pool win rate is
recorded, so drift reads as bars passing while that rate falls.

## Where a session starts

The champion. Always, and nothing else.

At a cold start the seed is champion zero: scored, written as the champion, and
in the pool like any other. So there is no pre-champion regime -- every session
starts from a champion and every candidate plays one.

That deletes the weighted draw over the database's best, which was only ever
reached before the first promotion and never again once a champion existed. Its
case was that a search starting from one program "explores with one hand", and
the other hand was five instructions drawn per session; there is one instruction
now, and eight concurrent sessions from the same champion are already eight
independent attempts at it.

The whole search therefore sits on one program at a time, which was already true
whenever a champion existed. The only escape is a round choosing to rewrite,
which is what the instruction asks for.

## What a round is given

One game: its episode key and the query that reads it out of the games database,
day by day, both sides. Not the game itself -- one season is 8,888 characters
across 68 columns, and the database already holds it.

A different game each round, so a session's `ROUNDS_PER_SESSION = 12` rounds see
twelve maps and no change gets twelve consecutive attempts at entrenching on one.

Plus the chain: for each round before it in this session, what it changed and the
win rate that came back. Twelve rounds on twelve maps are twelve independent
attempts without it. The chain is the session's own record, not a query over a
graph, and it needs:

- `Program.change`, a short description taken from the docstring at the top of
  the file the round wrote.
- The round prompt asking for that docstring.

## No lineage

Nothing records or consults what came from what. A round edits whatever the round
before it produced, but that is the session's local state and nothing downstream
can ask about it.

This removes the campaign's only mechanism for starting outside the seed's basin,
which is where all 79 programs sat. The replacement is the instruction: "Write a
program that beats the opponent" permits a rewrite.

## What `measure.py` measures

The champion, on `GATE_SEEDS` seeds in both seats: 32 games, about twenty seconds
at five workers. Before there is a champion, the best program in the database.

It reports the head-to-head rate with its Wilson interval, which is exactly the
promotion bar -- so a round's local tool and the gate compute the same statistic.
`--seeds` trades time for tightness. Playing the whole pool is nine minutes and
is the campaign's job.

## One database

`archive.py` is deleted. Programs live in the games database beside the games,
as a `programs` table declared in `games.TABLES` like the other eight
(`episodes`, `days`, `holdings`, `prices`, `orders`, `moves`, `candidate`,
`teams`).

One table, not two: a round that produced nothing is a row with a reason and no
rate, so what was `Database.failures` is `where reason != ''`.

| column     | is                                                    |
| ---------- | ----------------------------------------------------- |
| `id`       | the program id, which also prefixes its episode keys  |
| `change`   | what the round said it changed, from its docstring    |
| `rate`     | its win rate over the pool; zero when it never played |
| `reason`   | why it was rejected, or empty                         |
| `model`    | the slug that wrote it                                |
| `promoted` | whether this became the champion                      |
| `created`  | when                                                  |

The source itself stays a file on disk under the run's `programs/`, because the
pool plays files.

This is what makes the log worth keeping. A JSONL the loop replays at startup is
a structure only the loop can read; the same rows in the games database are
queryable by the `query-games` skill, so a round can ask what has been tried and
what it scored with the tool it already uses for its games -- and so can we,
joined against the games themselves.

**The champion is a query**, not a file of its own: the most recently promoted
row. `champion.json` and `archive.jsonl` go. `state.json`'s counters are counts
over the same table.

**The cost.** ClickHouse moves from recording the campaign to being in its
control flow: with it down, the loop cannot find its champion, where before that
was a local JSON file. Accepted -- it is the same process's own docker container
and the loop cannot score anything without it either, since every evaluation
records its games.

## Module structure

The loop's transitive imports, measured 2026-09-12: 22 modules, 8,773 lines.

| group           | modules                                                                                                                          | lines |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------- | ----- |
| core mechanism  | loop, harness, config, games, mutate, gate, validate, evaluator, prompt, archive, pool, copycheck, measure, arena, pools, roster | 6,427 |
| harvest         | kernel_watch, harvest, field_gate                                                                                                | 980   |
| corpus analysis | dataset, rating, tapes                                                                                                           | 1,166 |
| dead            | evidence                                                                                                                         | 149   |

Four changes fall out of that:

**Break the `dataset` coupling.** `harness` uses `dataset.measures`; `games` uses
`dataset.measures` and `dataset.COLUMNS`. Those two names pull `dataset` ->
`tapes` -> `rating` -- 1,166 lines of corpus analysis -- into the loop's import
graph. Move the measure definitions to where the campaign uses them, and the
corpus becomes corpus-only.

**`rating` then leaves the campaign by itself.** `gate` uses `rating.Field` and
`rating.standings`, both deleted here. With the `dataset` coupling broken, the
only importer left is `dataset`, which is where it belongs.

**Delete `evidence.py`.** 149 lines, imported by nothing. (`strategy.py` stays:
the `strategies` script uses it.)

**Split `loop.py`.** 1,208 lines holding the CLI, the wandb run, resumable
state, the campaign object, sessions, rounds, `keep`, `consider`, promotion
logging and the harvester coroutine. This design already rewrites much of it, so
the split happens while it is open rather than after.

Out of scope, recorded because it is the next boundary worth questioning: harvest
is 980 lines -- a Kaggle client, notebook parsing, base64 and zlib extraction,
compiled-kernel builds, a gate of its own -- reached from the loop through one
hourly coroutine, and nothing to do with producing programs.

## Deleted

| deleted                                                              | why                                                |
| -------------------------------------------------------------------- | -------------------------------------------------- |
| `rating` from the campaign path                                      | the design is balanced; the win rate is sufficient |
| `field.json`, `rating.Field`                                         | stored pairings existed only to connect the fit    |
| `gate.refresh`, the startup anchor pairings                          | nothing needs a connected graph                    |
| `GATE_ANCHORS` as a rating origin, `_anchored`                       | no additive constant left to pin                   |
| `GATE_OPPONENTS`, `GATE_CONTENDERS`, `must_play`, `Pool.sample`      | everyone is played                                 |
| `Program.rating`, `Program.place`                                    | nothing selects or promotes on them                |
| `Program.field`, `vendored_field`, the `roster.TRAINING` rate        | ignores 22 of 34 opponents; nothing reads it       |
| `POOL_CHAMPIONS`                                                     | there is one champion                              |
| the `place == 1` gate and its paired-margin test                     | replaced by the bar above                          |
| `Program.started_from`, `Database.children`, `Database.descendants`  | ancestry does not bear on winning                  |
| `SCRATCH_CHANCE`, `SCRATCH_ID`, `SCRATCH_AGENT`, the stagnation note | lineage machinery                                  |
| `parent.py` in the round directory                                   | `measure.py` measures the champion                 |
| `evidence.py`                                                        | imported by nothing                                |
| `archive.py`, `archive.jsonl`, `champion.json`, `state.json`         | the games database holds programs too              |
| `Result.margins`, `Result.hardest`                                   | fed a deleted tie-break and a deleted choice       |
| `PARENT_POOL`, `PARENT_DECAY`, the weighted draw, `Database.top(k)`  | a session starts from the champion                 |
| the no-champion branches in `start`, `floor` and `consider`          | the seed is champion zero                          |

`rating.py` itself stays, for `dataset.py`.

## Seed robustness

Four mechanisms, none of them obviously the one doing the work:

- The score is 1,088 games, not one map. A fixed plan's bank swings 19.5% season
  to season.
- The bar is paired: candidate and champion on identical seeds in both seats.
  Telling apart five thousand coins takes ~114 games unpaired and ~4 paired.
- The chain reports the global rate, so a round that improved only the map it was
  shown sees a number that did not move.
- Rounds get different maps.

Not here: a confirmation pass on fresh seeds before promoting. A two-stage gate
was deleted once, because its cheap stage was 8 seeds and 78 of 471 topped it
with none surviving. This is a bet that 16 paired seeds is deep enough. If
promotions arrive and fail to hold up, confirmation is the first thing to add,
and it is cheap because it runs only on a promotion.

## Already implemented

Green and uncommitted:

- `evaluator.score` plays the whole pool.
- `Database.top` ranks on the win rate rather than the rating. Superseded: the
  draw it fed is deleted, so `top` goes with it. The rank-0 tie it fixed stops
  mattering once a session starts from the champion.
- `Kept` carries the evaluation, standings and gate verdict out of `keep`, so a
  round makes one Bradley-Terry fit where it made three.
- `consider` logs its reason whether it promotes or refuses.

## Tests

Each verified by mutation, not assumed:

- A decisive head-to-head win over the champion promotes.
- A candidate drawing nearly every game against the champion does not, however
  its rate reads.
- At a cold start the seed is the champion, is in the pool, and is played
  head-to-head by the first candidate.
- A session starts from the champion, whatever else the database holds.
- The round message names one game and carries the chain.
- Twelve rounds of a session get twelve different games.
- `measure.py`'s printed rate matches what the gate computes on the same games.
- Nothing in the campaign reads `started_from`.
- No module outside `dataset.py` imports `rating`.
- Importing `loop` does not import `dataset`, `tapes` or `rating`.
- A program and a rejected round are both rows in `games.programs`, and the
  champion is the most recent promoted one.
- A round's query for what has already been tried returns the chain.
