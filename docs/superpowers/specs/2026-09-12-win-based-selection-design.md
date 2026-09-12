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

### When the champion moves under a session

Eight sessions run at once and the pool is copied per evaluation, so a promotion
by one lands while the others are mid-flight and their head-to-head is against a
champion that has been replaced.

The bar is re-played, not refused: if the champion changed since the evaluation
started, play the 32 games against the current one -- twenty seconds -- and read
the bar off those.

So the promotion is always measured against the champion as it stands, and there
is no snapshot semantics to specify beyond the obvious: the pool copy decides the
score, the champion at check time decides the promotion. What this replaces threw
the candidate away and logged `gate/stale`, discarding a nine-minute evaluation
over a twenty-second measurement.

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

## Where a program runs

Playing a program executes it, and a candidate or a harvested kernel may write
files -- relative ones, so the working directory decides where. `validate` bans
`open`, `eval`, `exec` and `__import__`, but `pathlib` is an allowed import and
`Path.write_text` walks through; harvested opponents are not validated at all.
The sandbox is the only thing between a stranger's kernel and the repository, and
one has already overwritten our own `main.py` with a 158KB replay agent.

`Sandbox.submit` wraps every task, so the sandbox is the pool's property and not
something a task function has to remember. The wrapper is the whole of it:

```python
def _sandboxed(call, *args, **kwargs):
    """Runs in the child, once. `Sandbox.submit` is the only caller."""
    with tempfile.TemporaryDirectory(prefix="campaign-worker-") as scratch:
        os.chdir(scratch)
        return call(*args, **kwargs)
```

Three properties, each of which was a failure:

- **In the child, never the parent.** A spawned child inherits the cwd of
  whoever spawned it, so a parent that chdirs relocates every game it starts
  next, and deleting the tree afterwards leaves them standing nowhere. That
  killed the campaign on 2026-09-11 and again on 2026-09-12.
- **No `Path.cwd()`.** Reading the cwd to restore it later is what raised
  `FileNotFoundError: [Errno 2]` with no filename -- the crash that reads like
  nothing at all -- when the directory was already gone.
- **No restore.** `max_tasks_per_child=1`, so the process takes one task and
  exits; there is nothing after it to put a directory back for.

`TemporaryDirectory` rather than `mkdtemp` and an `atexit` hook: both remove the
tree, only one says when, and a worker killed mid-game -- routine, since the
pacer cancels codex calls and the loop tears its pools down with them -- never
reaches an `atexit`. 164 scratch directories from that shape were in /tmp when it
was last changed.

`os.chdir` appears in that function and nowhere else in the package, enforced by
a test that reads the source.

## No lineage

Nothing records or consults what came from what. A round edits whatever the round
before it produced, but that is the session's local state and nothing downstream
can ask about it.

## The rebuild

`src/kaggriculture/campaign/` is replaced in place. Nothing is carried over as
code; what is carried over is the list of behaviours below, each of which is a
failure that already happened.

The case for rebuilding rather than refactoring is the ratio: the mechanism above
is 25 lines, the package is 9,135 across 25 modules, and the tests are 8,422 --
of which 4,161 describe ratings, opponent sampling, lineage, stagnation, the
scratch lineage, the draw and a two-condition gate, all deleted here. Those tests
do not shrink, they vanish with what they were describing. `test_loop.py` alone is
2,148 lines, a quarter of the suite, for one file holding nine responsibilities.

One job per module:

| module       | lines | does                                                                                                               |
| ------------ | ----- | ------------------------------------------------------------------------------------------------------------------ |
| `pool.py`    | ~60   | the opponent list: name to path, load, save                                                                        |
| `sandbox.py` | ~50   | the worker pool, one task per child, tempdir and chdir in the child                                                |
| `play.py`    | ~250  | one game between two files on a seed, both seats; extract the day table                                            |
| `score.py`   | ~120  | play a candidate against the pool, count wins, the Wilson bar                                                      |
| `codex.py`   | ~100  | spawn in a process group, pipe the message, kill the group, verify the file changed, recover the reason, fall back |
| `check.py`   | ~200  | syntax, entrypoint, imports, copy, and a timed play                                                                |
| `store.py`   | ~200  | the ClickHouse tables: games, days, programs                                                                       |
| `message.py` | ~120  | compose the round message                                                                                          |
| `loop.py`    | ~250  | the controller: sessions, rounds, promotion                                                                        |
| `config.py`  | ~120  | the constants                                                                                                      |

About 1,550 lines against 6,427 of core today.

### What must survive

The tests target these rather than the modules. Every one is a bug that has
already cost this campaign a run or a measurement:

- Kill the process group, not the process. `codex` spawns children, and a killed
  call otherwise leaves them burning quota.
- Verify `child.py` actually changed. A call can exit 0 having written nothing,
  and the loop then scores the previous program as if it were new.
- Recover the failure reason from the transcript. Without it a failed round is
  recorded with no cause.
- Retry the fallback model once on a capacity refusal.
- `chdir` only in the child, only inside a `TemporaryDirectory`, and never read
  `Path.cwd()`.
- Reject a program whose entrypoint is shadowed by a later callable, since
  Kaggle's loader takes the last one; that imports outside the whitelist; or that
  resembles an opponent it did not start from.
- Catch `BaseException` when loading a candidate, not `Exception`. `raise
SystemExit` needs no import and would otherwise escape.
- A program never plays itself. This is also what the resemblance check is for: a
  candidate that copies an agent in the pool scores about 0.5 against it by
  playing itself, and would promote as an improvement. The check is calibrated,
  not a guess -- a lifted 110-line block scores 0.231 and stays above the floor
  under reformatting, where an independent 70-line agent scores 0.0066.
- Play every opponent, never a sample.
- Log the gate's reason on refusal as well as on success.
- Every test mutation-verified: break the code, watch it go red.

### The one thing that cannot be verified by reading

`play.py` drives the engine, and a wrong game silently poisons every measurement
above it. So the old `harness.py` stays on disk until the new `play.py`
reproduces its banks and day tables over a set of seeds, and only then is it
deleted.

That is a differential check against an oracle that shares its assumptions, so it
cannot catch a misreading both versions make. It catches transcription, and that
is what a rewrite risks.

### What the optimizer needs from outside

The Rust engine and its bindings, `report.py` for the Wilson interval,
`constants.py`, and the `served/` skeleton. That is all.

`dataset.py`, `rating.py`, `tapes.py` and `strategy.py` -- about 1,465 lines of
ladder-corpus analysis -- are **not** part of the optimizer and nothing in this
design consumes them. Their output used to reach a round as the build order and
the settled claims; both were removed on 2026-09-10 after the first measurably
hurt. What remains is offline analysis for us, and whether it is worth keeping is
its own decision, taken separately and not by being carried along.

The only hard requirement is the direction of the dependency: the loop must not
import them, which a test asserts.

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
- A candidate whose champion was replaced mid-evaluation is measured against the
  new champion rather than discarded.
- The game a round is shown was played by the program that round is editing, not
  by the champion, once they differ.
- `measure.py`'s printed rate matches what the gate computes on the same games.
- A program and a rejected round are both rows in `games.programs`, and the
  champion is the most recent promoted one.
- Nothing in the campaign reads a program's ancestry.
- No module outside `dataset.py` imports `rating`, and importing `loop` does not
  import `dataset`, `tapes` or `rating`.

## Work already done, and its fate

Four changes were made to the old package on 2026-09-12 and are green. The
rebuild discards all four as code; three of them are design decisions that carry
over, and the fourth was already superseded.

- `evaluator.score` plays the whole pool -- carried over, as `score.py`.
- `consider` logs its reason on refusal -- carried over, in the behaviour list.
- `Kept` collapses three Bradley-Terry fits per round into one -- moot, there are
  no fits.
- `Database.top` ranks on the win rate -- moot, the draw it fed is deleted.

The prompt rewritten earlier that day (one game per round, the instruction, no
work loop) carries over into `message.py` and is already committed as `4824ed7`.
