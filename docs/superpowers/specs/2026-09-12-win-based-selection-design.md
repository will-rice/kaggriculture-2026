# Selection by games won

Date: 2026-09-12. Replaces sections 5.1, 5.2 and the prompt half of section 4 of
`2026-09-05-campaign-script-design.md`.

## The mechanism

```
champion = seed                        # a file on disk, and a row

8 sessions in parallel, forever:
    current = champion                 # the program being edited
    for 12 rounds:

        # one season `current` played, a different one each round
        episode   = next(episodes_of(current))

        message   = RULES_OF_KAGGRICULTURE
                  + episode.key + the query that reads it out of the database
                  + the chain: what each earlier round of this session changed,
                    and the win rate that came back
                  + "Write a program that beats the opponent."

        candidate = codex(message)                 # writes one file, stops
        if not valid(candidate):
            row(reason); continue                  # the next round is told

        seasons   = play(candidate, publics + champion, 16 seeds, both seats)
        rate      = wins(seasons) / len(seasons)
        record(seasons); row(candidate.id, change, rate)

        current   = candidate                      # the next round edits this

        if wilson_low(rate vs champion) > 0.5 and decisive vs champion >= 8:
            champion = candidate
            break
```

`episode` is one recorded season, `current` is the file being edited, `candidate`
is what the round wrote. "The opponent" in the instruction is whoever is across
the table in a scored game, never the opponent in `episode`: the round is not
being pointed at that agent, and nothing in the message names one.

Everything below is that in detail. Nothing else is in scope.

## Why

79 programs, zero champions. Measured 2026-09-12:

- `gate.promotion` needs `place == 1` of a Bradley-Terry tournament in both of
  its regimes. The best program placed 25 of 28, 628 Elo short. Four of the five
  above it are champions of the tape lineage; three of those are not in the pool
  and cannot be played.
- With no promotion nothing is retained, so the search re-converges. Five of 14
  programs in that day's run started from a program at 0.206 and landed at
  0.180-0.203.
- The starting program came from a weighted draw ranked on the rating, which tied
  the best program (0.331) with one at 0.206, so the draw took the worse one half
  the time.
- Refusals were logged only when they were not refusals.
- All 79 programs hold 5,777-6,236 on day 10 against opponents averaging 16,219.

A broken scaffold, not a verdict on what a model can write: handed the games
database, codex diagnosed the day-10 problem unprompted.

## The pool

Every public agent, plus the most recent champion. One champion, replaced on
promotion rather than accumulated, so the pool drifts toward the field and never
toward our own lineage.

## The score

Win rate over that pool. Every candidate plays all of it, on the session's shared
seed block, in both seats: 1,088 games today, 9.1 minutes at five workers.

Playing everyone is what removes Bradley-Terry. A rating is for a sparse graph,
and the graph was sparse only because opponents were drawn per candidate, leaving
two candidates' rates incomparable. A complete balanced design makes the win rate
the whole of what the rating estimated.

The opponent set is never capped and never sampled. A cap is the same subset
chosen once instead of per candidate, deciding which opponents a program is never
measured against. So the pool's size is a linear cost -- 60 opponents is 16
minutes, 100 is 27 -- absorbed on `GATE_SEEDS`, `SESSIONS` and the worker split,
or by accepting longer evaluations, which stay short against a codex call. Seeds
buy resolution on the bar that promotes; opponents buy breadth on the rate that
selects, and only the first is ours to set.

Bradley-Terry stays in `dataset.py`, over the ladder corpus, where 48,337 games
were paired by Kaggle and the graph really is sparse.

## Promotion

One condition: the Wilson lower bound of the candidate's win rate against the
champion, above 0.5, over at least `DECISIVE_GAMES` decisive games. At 32
decisive games that is 22 wins, 0.688.

The champion is in the pool, so this is directly measured from the first round:
16 seeds in both seats, both programs in the same games, so the map is shared and
the seat swap cancels position.

`DECISIVE_GAMES = 8` blocks 4 to 7 decisive games, a candidate drawing 78% to 88%
of its games with the champion. Below 4 the interval refuses on its own -- 3/3 is
0.438.

## Where a session starts

The champion. Always, and nothing else.

At a cold start the seed is champion zero: scored, recorded as promoted, and in
the pool like any other opponent. There is no pre-champion regime, so no
cold-start bar and no no-champion branches.

Eight concurrent sessions from one champion are eight independent attempts at it;
depth is the twelve rounds. The whole search therefore sits on one program at a
time, and the only escape is a round choosing to rewrite.

## What a round is given

One game played by the program this round is editing: its episode key and the
query that reads it out of the games database, day by day, both sides. Not the
game itself -- one season is 8,888 characters across 68 columns, and the database
already holds it.

The program it is editing, not the champion. Round 1 edits the champion, so those
are the champion's games; round 7 edits round 6's output, and a champion game
there would be a game played by a program six edits away from the one in front of
it. Every round's program is scored over the whole pool, so its own games are
always recorded.

A different game each round, so a session's 12 rounds see twelve maps and no
change gets twelve consecutive attempts at entrenching on one.

Plus the chain: for each round before it in this session, what it changed and the
win rate that came back. Twelve rounds on twelve maps are twelve independent
attempts without it. `change` comes from the docstring at the top of the file the
round wrote, which the prompt asks for.

Nothing else. No rank, no place, no opponent names, no per-opponent table, no
instructions on how to work.

## One database

The games database holds the programs too. `archive.py` and its JSONL go, and
`programs` becomes a table in `games.TABLES` beside the other eight.

One table for programs and rejections both: a round that produced nothing is a
row with a `reason` and no `rate`.

| column     | is                                                    |
| ---------- | ----------------------------------------------------- |
| `id`       | the program id, which also prefixes its episode keys  |
| `change`   | what the round said it changed                        |
| `rate`     | its win rate over the pool; zero when it never played |
| `reason`   | why it was rejected, or empty                         |
| `model`    | the slug that wrote it                                |
| `promoted` | whether this became the champion                      |
| `created`  | when                                                  |

Sources stay as files under the run's `programs/`, because the pool plays files.

The champion is a query -- the most recently promoted row -- so `champion.json`
and `state.json` go, and their counters are counts over the same table.

This is what makes the record worth keeping. A JSONL the loop replays at startup
is readable only by the loop; the same rows in the games database are reachable
through the `query-games` skill, so a round can ask what has been tried and what
it scored with the tool it already uses on its games, joined against the games
themselves.

## What `measure.py` measures

The champion, on `GATE_SEEDS` seeds in both seats: 32 games, about twenty seconds
at five workers.

It reports the head-to-head rate with its Wilson interval, which is exactly the
promotion bar, so a round's local tool and the gate compute the same statistic.
`--seeds` trades time for tightness. Playing the whole pool is nine minutes and is
the campaign's job.

## No lineage

Nothing records or consults what came from what. A round edits whatever the round
before it produced, but that is the session's local state and nothing downstream
can ask about it.

## Module structure

The loop's transitive imports today: 22 modules, 8,773 lines.

| group           | lines | modules                                                                                                                          |
| --------------- | ----- | -------------------------------------------------------------------------------------------------------------------------------- |
| core mechanism  | 6,427 | loop, harness, config, games, mutate, gate, validate, evaluator, prompt, archive, pool, copycheck, measure, arena, pools, roster |
| harvest         | 980   | kernel_watch, harvest, field_gate                                                                                                |
| corpus analysis | 1,166 | dataset, rating, tapes                                                                                                           |
| dead            | 149   | evidence                                                                                                                         |

- **Break the `dataset` coupling.** `harness` uses `dataset.measures`; `games`
  uses that and `dataset.COLUMNS`. Those two names pull dataset -> tapes ->
  rating into the loop's graph. Move the measure definitions to the campaign.
- **`rating` then leaves by itself.** `gate` uses `rating.Field` and
  `rating.standings`, both deleted; `dataset` becomes its only importer.
- **Delete `evidence.py`** -- imported by nothing. `strategy.py` stays; the
  `strategies` script uses it.
- **Split `loop.py`.** 1,208 lines holding the CLI, the wandb run, resumable
  state, the campaign object, sessions, rounds, `keep`, `consider`, promotion
  logging and the harvester. This design rewrites much of it, so the split
  happens while it is open.
- `pool` loses `sample` and becomes a name-to-path map. `gate` keeps the bar and
  the champion's files. `evaluator` keeps per-opponent rates and the head-to-head
  interval; `margins` fed a deleted tie-break, `hardest` chose a game the round
  now gets by rotation, `field` is deleted.

Out of scope, recorded as the next boundary worth questioning: harvest is 980
lines -- a Kaggle client, notebook parsing, base64 and zlib extraction,
compiled-kernel builds, a gate of its own -- reached through one hourly coroutine,
with nothing to do with producing programs.

## Deleted

| deleted                                                              | why                                               |
| -------------------------------------------------------------------- | ------------------------------------------------- |
| `rating` from the campaign path                                      | balanced design; the win rate is sufficient       |
| `field.json`, `rating.Field`                                         | stored pairings existed only to connect the fit   |
| `gate.refresh`, the startup anchor pairings                          | nothing needs a connected graph                   |
| `GATE_ANCHORS` as a rating origin, `_anchored`                       | no additive constant left to pin                  |
| `GATE_OPPONENTS`, `GATE_CONTENDERS`, `must_play`, `Pool.sample`      | everyone is played                                |
| `Program.rating`, `place`, `field`, `vendored_field`                 | nothing selects or promotes on them               |
| `POOL_CHAMPIONS`                                                     | there is one champion                             |
| the `place == 1` gate and its paired-margin test                     | replaced by the bar above                         |
| `Program.started_from`, `Database.children`, `Database.descendants`  | ancestry does not bear on winning                 |
| `PARENT_POOL`, `PARENT_DECAY`, the weighted draw, `Database.top`     | a session starts from the champion                |
| `SCRATCH_CHANCE`, `SCRATCH_ID`, `SCRATCH_AGENT`, the stagnation note | lineage machinery                                 |
| `archive.py`, `archive.jsonl`, `champion.json`, `state.json`         | the games database holds programs                 |
| `Result.margins`, `Result.hardest`                                   | fed a deleted tie-break and a deleted choice      |
| `parent.py` in the round directory                                   | `measure.py` measures the champion                |
| `evidence.py`                                                        | imported by nothing                               |
| the five-step work loop, the aggregate margin, the matchup table     | told a round how to work, or summarised 768 games |

## Seed robustness

Four mechanisms:

- The score is 1,088 games, not one map. A fixed plan's bank swings 19.5% season
  to season.
- The bar is paired: candidate and champion on identical seeds in both seats.
  Telling apart five thousand coins takes ~114 games unpaired and ~4 paired.
- The chain reports the global rate, so a round that improved only the map it was
  shown sees a number that did not move.
- Rounds get different maps.

## Risks taken knowingly

- **No confirmation pass** on fresh seeds before promoting. A two-stage gate was
  deleted once because its cheap stage was 8 seeds and 78 of 471 topped it with
  none surviving. This bets that 16 paired seeds is deep enough. If promotions
  arrive and fail to hold up, confirmation is the first thing to add, and it is
  cheap because it runs only on a promotion.
- **Sideways drift.** Promotion is a ratchet, not proof of progress, so a chain of
  head-to-head wins can walk around a cycle. Each champion's pool win rate is
  recorded, so drift reads as bars passing while that rate falls.
- **One program at a time.** Nothing starts a session outside the champion's
  basin, and that basin is where all 79 programs sat. The escape is the
  instruction permitting a rewrite, which is weaker than a reserved slot.
- **ClickHouse in the control flow.** With it down the loop cannot find its
  champion, where before that was a local file. Accepted: nothing can be scored
  without it either.

## Tests

Each verified by mutation, not assumed:

- A decisive head-to-head win over the champion promotes.
- A candidate drawing nearly every game against the champion does not, however
  its rate reads.
- At a cold start the seed is the champion, is in the pool, and is played
  head-to-head by the first candidate.
- A session starts from the champion, whatever else the database holds.
- Every candidate plays every pool opponent; none is sampled out.
- The round message names one game and carries the chain.
- Twelve rounds of a session get twelve different games.
- The game a round is shown was played by the program that round is editing, not
  by the champion, once they differ.
- `measure.py`'s printed rate matches what the gate computes on the same games.
- A program and a rejected round are both rows in `games.programs`, and the
  champion is the most recent promoted one.
- Nothing in the campaign reads a program's ancestry.
- No module outside `dataset.py` imports `rating`, and importing `loop` does not
  import `dataset`, `tapes` or `rating`.

## Already implemented

Green and uncommitted:

- `evaluator.score` plays the whole pool.
- `Kept` carries the evaluation and the gate's verdict out of `keep`, so a round
  makes one Bradley-Terry fit where it made three.
- `consider` logs its reason whether it promotes or refuses.
- `Database.top` ranks on the win rate. Superseded: the draw it fed is deleted.
